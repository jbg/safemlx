//! Rotary position embedding initialization and variants.

use std::collections::HashMap;

use safemlx::macros::ModuleParameters;
use safemlx::{
    builder::Builder,
    error::Exception,
    module::Module,
    nn,
    ops::{
        arange, concatenate_axis, cos,
        indexing::{NewAxis, TryIndexOp},
        sin, which,
    },
    Array, Stream,
};
use serde::Deserialize;

#[derive(Debug, Clone, PartialEq)]
/// Borrowed scalar value from a RoPE scaling config.
pub enum FloatOrStr<'a> {
    /// Numeric floating-point value.
    Float(f32),
    /// Borrowed string value.
    Str(&'a str),
    /// Boolean option used by scaling schemes such as YaRN `truncate`.
    Bool(bool),
}

// TODO: check if additional serde attributes are needed
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
/// Deserialized RoPE scaling-config value.
pub enum FloatOrString {
    /// Numeric floating-point value.
    Float(f32),
    /// String value, used by config fields such as `rope_type`.
    String(String),
    /// Boolean scaling option.
    Bool(bool),
}

impl FloatOrString {
    /// Borrows the value without allocating.
    pub fn borrowed(&self) -> FloatOrStr<'_> {
        match self {
            FloatOrString::Float(f) => FloatOrStr::Float(*f),
            FloatOrString::String(s) => FloatOrStr::Str(s),
            FloatOrString::Bool(value) => FloatOrStr::Bool(*value),
        }
    }
}

/// Get a numeric float value from a scaling config by key.
///
/// Note: str variants in the config are not always floats — values like "default" or "linear"
/// are also valid for non-numeric fields. This function should only be called for keys that
/// are expected to hold numeric values.
fn get_numeric_from_config(
    config: &HashMap<String, FloatOrString>,
    key: &str,
) -> Result<f32, Exception> {
    match config
        .get(key)
        .map(FloatOrString::borrowed)
        .ok_or_else(|| {
            Exception::custom(format!(r#"key "{key}" is not found in scaling config"#))
        })? {
        FloatOrStr::Float(f) => Ok(f),
        FloatOrStr::Str(s) => s
            .parse::<f32>()
            .map_err(|_| Exception::custom(format!(r#"key "{key}" is not a valid number"#))),
        FloatOrStr::Bool(_) => Err(Exception::custom(format!(r#"key "{key}" is not numeric"#))),
    }
}

/// Llama3-style RoPE with frequency scaling.
///
/// Applies piecewise frequency scaling based on wavelength cutoffs derived from
/// `low_freq_factor`, `high_freq_factor`, `factor`, and `original_max_position_embeddings`.
// TODO: support derive ModuleParameters for structs with non-param Array fields
#[derive(Debug, Clone, ModuleParameters)]
pub struct Llama3Rope {
    /// Number of rotary dimensions.
    pub dimensions: i32,
    /// Whether to use traditional pair ordering.
    pub traditional: bool,
    /// Runtime scale factor passed to MLX RoPE.
    pub scale: f32,
    /// Pre-computed scaled frequencies. Not a module parameter.
    pub freqs: Array,
}

impl Llama3Rope {
    /// Builds a Llama 3 RoPE module with frequency scaling.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dims: i32,
        traditional: bool,
        original_max_position_embeddings: i32,
        base: f32,
        factor: f32,
        low_freq_factor: f32,
        high_freq_factor: f32,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let half_dims = dims / 2;

        // Compute freqs using MLX ops, matching Python:
        //   freqs = base ** (mx.arange(0, dims, 2) / dims)
        // which equals base^(2i/dims) for i in 0..half_dims
        let indices = arange::<_, f32>(None, half_dims, None, stream)?;
        let exponents = indices.multiply(Array::from_f32(2.0 / dims as f32), stream)?;
        let freqs = Array::from_f32(base).power(&exponents, stream)?;

        let old_context_len = original_max_position_embeddings as f32;
        let low_freq_wavelen = old_context_len / low_freq_factor;
        let high_freq_wavelen = old_context_len / high_freq_factor;

        // wavelens = 2 * pi * freqs
        // Apply piecewise scaling matching Python exactly:
        //   freqs = where(wavelens > low_freq_wavelen, freqs * factor, freqs)
        //   is_medium = (wavelens > high_freq_wavelen) & (wavelens < low_freq_wavelen)
        //   smooth_factors = (old_context_len / wavelens - low_freq_factor) / (high - low)
        //   smooth_freqs = freqs / ((1 - smooth_factors) / factor + smooth_factors)
        //   freqs = where(is_medium, smooth_freqs, freqs)
        let two_pi = Array::from_f32(2.0 * std::f32::consts::PI);
        let wavelens = freqs.multiply(&two_pi, stream)?;

        // First pass: scale low frequencies (long wavelengths) by factor
        let is_low = wavelens.gt(Array::from_f32(low_freq_wavelen), stream)?;
        let freqs = which(
            &is_low,
            &freqs.multiply(Array::from_f32(factor), stream)?,
            &freqs,
            stream,
        )?;

        // Second pass: smooth interpolation for medium frequencies
        let is_medium = wavelens
            .gt(Array::from_f32(high_freq_wavelen), stream)?
            .logical_and(
                &wavelens.lt(Array::from_f32(low_freq_wavelen), stream)?,
                stream,
            )?;

        let smooth_factors = wavelens
            .reciprocal(stream)?
            .multiply(Array::from_f32(old_context_len), stream)?
            .subtract(Array::from_f32(low_freq_factor), stream)?
            .divide(Array::from_f32(high_freq_factor - low_freq_factor), stream)?;

        // smooth_freqs = freqs / ((1 - smooth_factors) / factor + smooth_factors)
        let one_minus_smooth = Array::from_f32(1.0).subtract(&smooth_factors, stream)?;
        let denom = one_minus_smooth
            .divide(Array::from_f32(factor), stream)?
            .add(&smooth_factors, stream)?;
        let smooth_freqs = freqs.divide(&denom, stream)?;

        let freqs = which(&is_medium, &smooth_freqs, &freqs, stream)?;

        Ok(Self {
            dimensions: dims,
            traditional,
            scale: 1.0,
            freqs,
        })
    }
}

impl<'a, Input> Module<Input> for Llama3Rope
where
    Input: Into<nn::RopeInput<'a>>,
{
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, input: Input, stream: &Stream) -> Result<Self::Output, Self::Error> {
        let nn::RopeInput { x, offset } = input.into();
        let shape = x.shape();
        let x = x.reshape(&[-1, x.dim(-2), x.dim(-1)], stream)?;
        if !self.traditional {
            let seq_len = x.dim(-2);
            let half_dims = self.dimensions / 2;
            let positions = Array::arange::<_, f32>(Some(offset), offset + seq_len, None, stream)?
                .try_index_device((.., NewAxis), stream)?;
            let freqs = self.freqs.try_index_device(NewAxis, stream)?;
            let angles = positions
                .divide(&freqs, stream)?
                .multiply(Array::from_f32(self.scale), stream)?;
            let cos = cos(&angles, stream)?.try_index_device((NewAxis, .., ..), stream)?;
            let sin = sin(&angles, stream)?.try_index_device((NewAxis, .., ..), stream)?;
            let x1 = x.try_index_device((.., .., ..half_dims), stream)?;
            let x2 = x.try_index_device((.., .., half_dims..), stream)?;
            let out1 = x1
                .multiply(&cos, stream)?
                .subtract(x2.multiply(&sin, stream)?, stream)?;
            let out2 = x2
                .multiply(cos, stream)?
                .add(x1.multiply(sin, stream)?, stream)?;
            return concatenate_axis(&[out1, out2], -1, stream)?.reshape(shape, stream);
        }
        let x = safemlx::fast::rope(
            x,
            self.dimensions,
            self.traditional,
            None::<f32>,
            self.scale,
            offset,
            &self.freqs,
            stream,
        )?;
        x.reshape(shape, stream)
    }

    fn training_mode(&mut self, _mode: bool) {}
}

/// Proportional RoPE used by Gemma 4 full-attention layers.
///
/// Proportional RoPE keeps the full head dimension so non-traditional
/// half-rotation pairs match Hugging Face's layout. Frequency slots outside the
/// configured rotary proportion use the largest finite `f32` denominator, which
/// MLX reciprocates to an effectively zero inverse frequency and therefore
/// leaves unchanged for real context lengths.
#[derive(Debug, Clone, ModuleParameters)]
pub struct ProportionalRope {
    /// Head dimension passed to MLX RoPE.
    pub dimensions: i32,
    /// Whether to use traditional pair ordering.
    pub traditional: bool,
    /// Runtime scale factor passed to MLX RoPE.
    pub scale: f32,
    /// Frequency vector for the full half-head dimension.
    pub freqs: Array,
}

/// YaRN rotary embeddings with frequency interpolation and attention scaling.
#[derive(Debug, Clone, ModuleParameters)]
pub struct YarnRope {
    /// Number of rotary dimensions.
    pub dimensions: i32,
    /// Whether adjacent pairs, rather than split halves, are rotated.
    pub traditional: bool,
    /// Multiplicative attention scale applied to the rotated inputs.
    pub concentration: f32,
    /// Pre-computed YaRN frequency denominators.
    pub freqs: Array,
}

fn yarn_frequency_values(
    dims: i32,
    base: f32,
    factor: f32,
    original_context: f32,
    beta_fast: f32,
    beta_slow: f32,
    truncate: bool,
) -> Vec<f32> {
    let correction = |rotations: f32| {
        dims as f32 * (original_context / (rotations * 2.0 * std::f32::consts::PI)).ln()
            / (2.0 * base.ln())
    };
    let low = if truncate {
        correction(beta_fast).floor()
    } else {
        correction(beta_fast)
    }
    .max(0.0);
    let high = if truncate {
        correction(beta_slow).ceil()
    } else {
        correction(beta_slow)
    }
    .min((dims - 1) as f32);
    let width = if low == high { 0.001 } else { high - low };
    (0..dims / 2)
        .map(|index| {
            let base_frequency = base.powf(2.0 * index as f32 / dims as f32);
            let ramp = ((index as f32 - low) / width).clamp(0.0, 1.0);
            let extrapolation_mask = 1.0 - ramp;
            factor * base_frequency / (factor * extrapolation_mask + (1.0 - extrapolation_mask))
        })
        .collect()
}

impl YarnRope {
    /// Constructs a YaRN RoPE module.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dims: i32,
        traditional: bool,
        base: f32,
        factor: f32,
        original_context: f32,
        beta_fast: f32,
        beta_slow: f32,
        mscale: f32,
        mscale_all_dim: f32,
        truncate: bool,
    ) -> Self {
        let scale = |coefficient: f32| {
            if factor <= 1.0 {
                1.0
            } else {
                0.1 * coefficient * factor.ln() + 1.0
            }
        };
        let values = yarn_frequency_values(
            dims,
            base,
            factor,
            original_context,
            beta_fast,
            beta_slow,
            truncate,
        );
        Self {
            dimensions: dims,
            traditional,
            concentration: scale(mscale) / scale(mscale_all_dim),
            freqs: Array::from_slice(&values, &[dims / 2]),
        }
    }
}

impl<'a, Input> Module<Input> for YarnRope
where
    Input: Into<nn::RopeInput<'a>>,
{
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, input: Input, stream: &Stream) -> Result<Self::Output, Self::Error> {
        let nn::RopeInput { x, offset } = input.into();
        let shape = x.shape();
        let x = x
            .multiply(Array::from_f32(self.concentration), stream)?
            .reshape(&[-1, x.dim(-2), x.dim(-1)], stream)?;
        let x = safemlx::fast::rope(
            x,
            self.dimensions,
            self.traditional,
            None::<f32>,
            1.0,
            offset,
            &self.freqs,
            stream,
        )?;
        x.reshape(shape, stream)
    }

    fn training_mode(&mut self, _mode: bool) {}
}

fn proportional_rotary_dims(dims: i32, proportion: f32) -> (i32, i32) {
    let half_dims = dims / 2;
    let rotary_dims = ((proportion * dims as f32).round() as i32).clamp(2, dims);
    let rope_angles = (rotary_dims / 2).clamp(1, half_dims);
    (rope_angles * 2, rope_angles)
}

fn proportional_frequency_values(dims: i32, base: f32, factor: f32, proportion: f32) -> Vec<f32> {
    let (_, rope_angles) = proportional_rotary_dims(dims, proportion);
    let half_dims = dims / 2;
    let mut freqs = Vec::with_capacity(half_dims as usize);
    for index in 0..half_dims {
        if index < rope_angles {
            freqs.push(base.powf(2.0 * index as f32 / dims as f32) / factor);
        } else {
            freqs.push(f32::MAX);
        }
    }
    freqs
}

impl ProportionalRope {
    /// Builds proportional RoPE for partial-rotary models.
    pub fn new(
        dims: i32,
        traditional: bool,
        base: f32,
        factor: f32,
        proportion: f32,
        _stream: &Stream,
    ) -> Result<Self, Exception> {
        let freqs = proportional_frequency_values(dims, base, factor, proportion);
        let freqs = Array::from_slice(&freqs, &[dims / 2]);
        Ok(Self {
            dimensions: dims,
            traditional,
            scale: 1.0,
            freqs,
        })
    }
}

impl<'a, Input> Module<Input> for ProportionalRope
where
    Input: Into<nn::RopeInput<'a>>,
{
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, input: Input, stream: &Stream) -> Result<Self::Output, Self::Error> {
        let nn::RopeInput { x, offset } = input.into();
        let shape = x.shape();
        let x = x.reshape(&[-1, x.dim(-2), x.dim(-1)], stream)?;
        let x = safemlx::fast::rope(
            x,
            self.dimensions,
            self.traditional,
            None::<f32>,
            self.scale,
            offset,
            &self.freqs,
            stream,
        )?;
        x.reshape(shape, stream)
    }

    fn training_mode(&mut self, _mode: bool) {}
}

/// Enum wrapping different RoPE variants so that `initialize_rope` can return
/// either a standard RoPE or a Llama3 RoPE.
#[derive(Debug, Clone)]
pub enum RopeVariant {
    /// Standard MLX RoPE.
    Default(nn::Rope),
    /// Llama 3 scaled RoPE.
    Llama3(Llama3Rope),
    /// Proportional RoPE used by Gemma 4.
    Proportional(ProportionalRope),
    /// YaRN scaled RoPE.
    Yarn(YarnRope),
}

// TODO: support derive ModuleParameters for enum
impl safemlx::module::ModuleParameters for RopeVariant {
    fn num_parameters(&self) -> usize {
        match self {
            RopeVariant::Default(rope) => rope.num_parameters(),
            RopeVariant::Llama3(rope) => rope.num_parameters(),
            RopeVariant::Proportional(rope) => rope.num_parameters(),
            RopeVariant::Yarn(rope) => rope.num_parameters(),
        }
    }

    fn freeze_parameters(&mut self, _recursive: bool) {
        match self {
            RopeVariant::Default(rope) => rope.freeze_parameters(_recursive),
            RopeVariant::Llama3(rope) => rope.freeze_parameters(_recursive),
            RopeVariant::Proportional(rope) => rope.freeze_parameters(_recursive),
            RopeVariant::Yarn(rope) => rope.freeze_parameters(_recursive),
        }
    }

    fn unfreeze_parameters(&mut self, _recursive: bool) {
        match self {
            RopeVariant::Default(rope) => rope.unfreeze_parameters(_recursive),
            RopeVariant::Llama3(rope) => rope.unfreeze_parameters(_recursive),
            RopeVariant::Proportional(rope) => rope.unfreeze_parameters(_recursive),
            RopeVariant::Yarn(rope) => rope.unfreeze_parameters(_recursive),
        }
    }

    fn parameters(&self) -> safemlx::module::ModuleParamRef<'_> {
        match self {
            RopeVariant::Default(rope) => rope.parameters(),
            RopeVariant::Llama3(rope) => rope.parameters(),
            RopeVariant::Proportional(rope) => rope.parameters(),
            RopeVariant::Yarn(rope) => rope.parameters(),
        }
    }

    fn parameters_mut(&mut self) -> safemlx::module::ModuleParamMut<'_> {
        match self {
            RopeVariant::Default(rope) => rope.parameters_mut(),
            RopeVariant::Llama3(rope) => rope.parameters_mut(),
            RopeVariant::Proportional(rope) => rope.parameters_mut(),
            RopeVariant::Yarn(rope) => rope.parameters_mut(),
        }
    }

    fn trainable_parameters(&self) -> safemlx::module::ModuleParamRef<'_> {
        match self {
            RopeVariant::Default(rope) => rope.trainable_parameters(),
            RopeVariant::Llama3(rope) => rope.trainable_parameters(),
            RopeVariant::Proportional(rope) => rope.trainable_parameters(),
            RopeVariant::Yarn(rope) => rope.trainable_parameters(),
        }
    }

    fn all_frozen(&self) -> Option<bool> {
        match self {
            RopeVariant::Default(rope) => rope.all_frozen(),
            RopeVariant::Llama3(rope) => rope.all_frozen(),
            RopeVariant::Proportional(rope) => rope.all_frozen(),
            RopeVariant::Yarn(rope) => rope.all_frozen(),
        }
    }

    fn any_frozen(&self) -> Option<bool> {
        match self {
            RopeVariant::Default(rope) => rope.any_frozen(),
            RopeVariant::Llama3(rope) => rope.any_frozen(),
            RopeVariant::Proportional(rope) => rope.any_frozen(),
            RopeVariant::Yarn(rope) => rope.any_frozen(),
        }
    }
}

impl<'a, Input> Module<Input> for RopeVariant
where
    Input: Into<nn::RopeInput<'a>>,
{
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, input: Input, stream: &Stream) -> Result<Self::Output, Self::Error> {
        match self {
            RopeVariant::Default(rope) => rope.forward(input, stream),
            RopeVariant::Llama3(rope) => rope.forward(input, stream),
            RopeVariant::Proportional(rope) => rope.forward(input, stream),
            RopeVariant::Yarn(rope) => rope.forward(input, stream),
        }
    }

    fn training_mode(&mut self, mode: bool) {
        match self {
            RopeVariant::Default(rope) => {
                <nn::Rope as Module<nn::RopeInput>>::training_mode(rope, mode)
            }
            RopeVariant::Llama3(rope) => {
                <Llama3Rope as Module<nn::RopeInput>>::training_mode(rope, mode)
            }
            RopeVariant::Proportional(rope) => {
                <ProportionalRope as Module<nn::RopeInput>>::training_mode(rope, mode)
            }
            RopeVariant::Yarn(rope) => {
                <YarnRope as Module<nn::RopeInput>>::training_mode(rope, mode)
            }
        }
    }
}

/// Creates the RoPE implementation requested by a model config.
pub fn initialize_rope(
    dims: i32,
    base: f32, // rope_theta
    traditional: bool,
    scaling_config: &Option<HashMap<String, FloatOrString>>,
    _max_position_embeddings: i32,
    stream: &Stream,
) -> Result<RopeVariant, Exception> {
    let rope_type = scaling_config
        .as_ref()
        .and_then(|config| {
            config
                .get("type")
                .or_else(|| config.get("rope_type"))
                .map(FloatOrString::borrowed)
        })
        .unwrap_or(FloatOrStr::Str("default"));

    if rope_type == FloatOrStr::Str("default") || rope_type == FloatOrStr::Str("linear") {
        let scale = if rope_type == FloatOrStr::Str("linear") {
            let den = get_numeric_from_config(scaling_config.as_ref().unwrap(), "factor")?;
            1.0 / den
        } else {
            1.0
        };

        let rope = nn::RopeBuilder::new(dims)
            .traditional(traditional)
            .base(base)
            .scale(scale)
            .build()
            .expect("Infallible");
        return Ok(RopeVariant::Default(rope));
    } else if rope_type == FloatOrStr::Str("llama3") {
        let config = scaling_config
            .as_ref()
            .ok_or_else(|| Exception::custom("scaling_config is required for llama3 RoPE"))?;

        let factor = get_numeric_from_config(config, "factor")?;
        let low_freq_factor = get_numeric_from_config(config, "low_freq_factor")?;
        let high_freq_factor = get_numeric_from_config(config, "high_freq_factor")?;
        let original_max_position_embeddings =
            get_numeric_from_config(config, "original_max_position_embeddings")? as i32;

        let rope = Llama3Rope::new(
            dims,
            traditional,
            original_max_position_embeddings,
            base,
            factor,
            low_freq_factor,
            high_freq_factor,
            stream,
        )?;
        return Ok(RopeVariant::Llama3(rope));
    } else if rope_type == FloatOrStr::Str("proportional") {
        let config = scaling_config
            .as_ref()
            .ok_or_else(|| Exception::custom("scaling_config is required for proportional RoPE"))?;
        let factor = config
            .get("factor")
            .map(|_| get_numeric_from_config(config, "factor"))
            .transpose()?
            .unwrap_or(1.0);
        let proportion = config
            .get("partial_rotary_factor")
            .map(|_| get_numeric_from_config(config, "partial_rotary_factor"))
            .transpose()?
            .unwrap_or(1.0);
        return Ok(RopeVariant::Proportional(ProportionalRope::new(
            dims,
            traditional,
            base,
            factor,
            proportion,
            stream,
        )?));
    } else if rope_type == FloatOrStr::Str("yarn") {
        let config = scaling_config
            .as_ref()
            .ok_or_else(|| Exception::custom("scaling_config is required for YaRN RoPE"))?;
        let value_or = |key: &str, default: f32| {
            config
                .get(key)
                .map(|_| get_numeric_from_config(config, key))
                .transpose()
                .map(|value| value.unwrap_or(default))
        };
        let truncate = match config.get("truncate") {
            Some(FloatOrString::Bool(value)) => *value,
            Some(_) => {
                return Err(Exception::custom(
                    "YaRN truncate must be a boolean when provided",
                ))
            }
            None => true,
        };
        return Ok(RopeVariant::Yarn(YarnRope::new(
            dims,
            traditional,
            base,
            get_numeric_from_config(config, "factor")?,
            get_numeric_from_config(config, "original_max_position_embeddings")?,
            value_or("beta_fast", 32.0)?,
            value_or("beta_slow", 1.0)?,
            value_or("mscale", 1.0)?,
            value_or("mscale_all_dim", 0.0)?,
            truncate,
        )));
    } else if rope_type == FloatOrStr::Str("longrope") {
        todo!()
    }

    Err(Exception::custom(format!(
        "Unsupported RoPE type {rope_type:?}"
    )))
}

#[cfg(test)]
mod tests {
    use super::{proportional_frequency_values, proportional_rotary_dims, yarn_frequency_values};

    #[test]
    fn proportional_rope_uses_full_half_head_frequency_layout() {
        let (rotary_dims, rope_angles) = proportional_rotary_dims(512, 0.25);
        let freqs = proportional_frequency_values(512, 1_000_000.0, 1.0, 0.25);

        assert_eq!(rotary_dims, 128);
        assert_eq!(rope_angles, 64);
        assert_eq!(freqs.len(), 256);
        assert_eq!(freqs[0], 1.0);
        assert!((freqs[1] - 1_000_000.0_f32.powf(2.0 / 512.0)).abs() < 0.0001);
        assert!(freqs[63] > 0.0);
        assert_eq!(freqs[64], f32::MAX);
        assert_eq!(freqs[255], f32::MAX);
    }

    #[test]
    fn yarn_interpolates_between_original_and_extended_frequencies() {
        let factor = 32.0;
        let base = 150_000.0;
        let freqs = yarn_frequency_values(64, base, factor, 4096.0, 32.0, 1.0, false);
        assert_eq!(freqs.len(), 32);
        assert!((freqs[0] - 1.0).abs() < 1e-6);
        let last_base = base.powf(62.0 / 64.0);
        assert!((freqs[31] / last_base - factor).abs() < 1e-3);
    }
}
