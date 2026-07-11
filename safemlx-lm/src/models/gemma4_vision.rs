use safemlx::{
    error::Exception,
    fast::{scaled_dot_product_attention, ScaledDotProductAttentionMask},
    macros::ModuleParameters,
    module::{Module, Param},
    nn,
    ops::{concatenate_axis, indexing::NewAxis, indexing::TryIndexOp, maximum, mean_axes},
    Array, Dtype, Stream,
};
use serde::Deserialize;

use super::{gemma4::rms_norm_without_scale, gemma4_multimodal::Gemma4ClippedLinear};
use crate::utils::rope::FloatOrString;

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct Gemma4VisionConfig {
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub head_dim: i32,
    pub patch_size: i32,
    pub pooling_kernel_size: i32,
    pub position_embedding_size: i32,
    pub rms_norm_eps: f32,
    #[serde(default = "default_hidden_activation")]
    pub hidden_activation: String,
    #[serde(default)]
    pub standardize: bool,
    #[serde(default)]
    pub rope_parameters: Option<std::collections::HashMap<String, FloatOrString>>,
}

fn default_hidden_activation() -> String {
    "gelu_pytorch_tanh".into()
}

impl Gemma4VisionConfig {
    fn rope_theta(&self) -> f32 {
        self.rope_parameters
            .as_ref()
            .and_then(|parameters| parameters.get("rope_theta"))
            .and_then(|value| match value {
                FloatOrString::Float(value) => Some(*value),
                FloatOrString::String(value) => value.parse().ok(),
            })
            .unwrap_or(100.0)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub(crate) struct VisionPatchEmbedder {
    #[param]
    pub input_proj: nn::Linear,
    #[param]
    pub position_embedding_table: Param<Array>,
}

impl VisionPatchEmbedder {
    fn new(config: &Gemma4VisionConfig, stream: &Stream) -> Result<Self, Exception> {
        Ok(Self {
            input_proj: nn::Linear::unloaded(
                3 * config.patch_size * config.patch_size,
                config.hidden_size,
                false,
                Dtype::Float32,
                stream,
            )?,
            position_embedding_table: Param::<Array>::unloaded(
                &[2, config.position_embedding_size, config.hidden_size],
                Dtype::Float32,
                stream,
            )?,
        })
    }

    fn forward(
        &mut self,
        pixel_values: &Array,
        position_ids: &Array,
        stream: &Stream,
    ) -> Result<(Array, Array), Exception> {
        validate_vision_inputs(pixel_values, position_ids)?;
        let x_positions = position_ids.try_index_device((.., .., 0), stream)?;
        let y_positions = position_ids.try_index_device((.., .., 1), stream)?;
        let padding = x_positions
            .eq(Array::from_int(-1), stream)?
            .logical_and(&y_positions.eq(Array::from_int(-1), stream)?, stream)?;
        let x_indices = maximum(x_positions, Array::from_int(0), stream)?;
        let y_indices = maximum(y_positions, Array::from_int(0), stream)?;
        let x_table = self
            .position_embedding_table
            .try_index_device((0, .., ..), stream)?;
        let y_table = self
            .position_embedding_table
            .try_index_device((1, .., ..), stream)?;
        let positions = x_table
            .try_index_device((&x_indices, ..), stream)?
            .add(y_table.try_index_device((&y_indices, ..), stream)?, stream)?;
        let valid = padding
            .logical_not(stream)?
            .as_dtype(positions.dtype(), stream)?
            .try_index_device((.., .., NewAxis), stream)?;
        let positions = positions.multiply(valid, stream)?;
        let scaled_pixels = pixel_values
            .multiply(Array::from_f32(2.0), stream)?
            .subtract(Array::from_f32(1.0), stream)?
            .as_dtype(self.input_proj.weight.dtype(), stream)?;
        Ok((
            self.input_proj
                .forward(&scaled_pixels, stream)?
                .add(positions, stream)?,
            padding,
        ))
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub(crate) struct VisionAttention {
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    #[param]
    pub q_proj: Gemma4ClippedLinear,
    #[param]
    pub k_proj: Gemma4ClippedLinear,
    #[param]
    pub v_proj: Gemma4ClippedLinear,
    #[param]
    pub o_proj: Gemma4ClippedLinear,
    #[param]
    pub q_norm: nn::RmsNorm,
    #[param]
    pub k_norm: nn::RmsNorm,
    pub norm_eps: f32,
}

impl VisionAttention {
    fn new(config: &Gemma4VisionConfig, stream: &Stream) -> Result<Self, Exception> {
        Ok(Self {
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            q_proj: Gemma4ClippedLinear::new(
                config.hidden_size,
                config.num_attention_heads * config.head_dim,
                false,
                stream,
            )?,
            k_proj: Gemma4ClippedLinear::new(
                config.hidden_size,
                config.num_key_value_heads * config.head_dim,
                false,
                stream,
            )?,
            v_proj: Gemma4ClippedLinear::new(
                config.hidden_size,
                config.num_key_value_heads * config.head_dim,
                false,
                stream,
            )?,
            o_proj: Gemma4ClippedLinear::new(
                config.num_attention_heads * config.head_dim,
                config.hidden_size,
                false,
                stream,
            )?,
            q_norm: nn::RmsNorm::unloaded(
                config.head_dim,
                config.rms_norm_eps,
                Dtype::Float32,
                stream,
            )?,
            k_norm: nn::RmsNorm::unloaded(
                config.head_dim,
                config.rms_norm_eps,
                Dtype::Float32,
                stream,
            )?,
            norm_eps: config.rms_norm_eps,
        })
    }

    fn forward(
        &mut self,
        hidden: &Array,
        padding: &Array,
        cos: &Array,
        sin: &Array,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let batch = hidden.dim(0);
        let sequence = hidden.dim(1);
        let query = self.q_norm.forward(
            &self
                .q_proj
                .forward(hidden, stream)?
                .reshape(&[batch, sequence, self.num_heads, self.head_dim], stream)?,
            stream,
        )?;
        let key = self.k_norm.forward(
            &self
                .k_proj
                .forward(hidden, stream)?
                .reshape(&[batch, sequence, self.num_kv_heads, self.head_dim], stream)?,
            stream,
        )?;
        let value = rms_norm_without_scale(
            &self
                .v_proj
                .forward(hidden, stream)?
                .reshape(&[batch, sequence, self.num_kv_heads, self.head_dim], stream)?,
            self.norm_eps,
            stream,
        )?;
        let query =
            apply_2d_rope(query, cos, sin, stream)?.transpose_axes(&[0, 2, 1, 3], stream)?;
        let key = apply_2d_rope(key, cos, sin, stream)?.transpose_axes(&[0, 2, 1, 3], stream)?;
        let value = value.transpose_axes(&[0, 2, 1, 3], stream)?;
        let key_mask = padding
            .try_index_device((.., NewAxis, NewAxis, ..), stream)?
            .as_dtype(query.dtype(), stream)?
            .multiply(Array::from_f32(-1.0e9), stream)?;
        let output = scaled_dot_product_attention(
            query,
            key,
            value,
            1.0,
            Some(ScaledDotProductAttentionMask::Array(&key_mask)),
            None,
            stream,
        )?
        .transpose_axes(&[0, 2, 1, 3], stream)?
        .reshape(&[batch, sequence, -1], stream)?;
        self.o_proj.forward(&output, stream)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub(crate) struct VisionMlp {
    pub activation: String,
    #[param]
    pub gate_proj: Gemma4ClippedLinear,
    #[param]
    pub up_proj: Gemma4ClippedLinear,
    #[param]
    pub down_proj: Gemma4ClippedLinear,
}

impl VisionMlp {
    fn new(config: &Gemma4VisionConfig, stream: &Stream) -> Result<Self, Exception> {
        Ok(Self {
            activation: config.hidden_activation.clone(),
            gate_proj: Gemma4ClippedLinear::new(
                config.hidden_size,
                config.intermediate_size,
                false,
                stream,
            )?,
            up_proj: Gemma4ClippedLinear::new(
                config.hidden_size,
                config.intermediate_size,
                false,
                stream,
            )?,
            down_proj: Gemma4ClippedLinear::new(
                config.intermediate_size,
                config.hidden_size,
                false,
                stream,
            )?,
        })
    }

    fn forward(&mut self, x: &Array, stream: &Stream) -> Result<Array, Exception> {
        if self.activation != "gelu_pytorch_tanh" && self.activation != "gelu_new" {
            return Err(Exception::custom(format!(
                "Gemma 4 vision activation '{}' is not supported",
                self.activation
            )));
        }
        let gate = nn::gelu_approximate(self.gate_proj.forward(x, stream)?, stream)?;
        let up = self.up_proj.forward(x, stream)?;
        self.down_proj.forward(&gate.multiply(up, stream)?, stream)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub(crate) struct VisionLayer {
    #[param]
    pub self_attn: VisionAttention,
    #[param]
    pub mlp: VisionMlp,
    #[param]
    pub input_layernorm: nn::RmsNorm,
    #[param]
    pub post_attention_layernorm: nn::RmsNorm,
    #[param]
    pub pre_feedforward_layernorm: nn::RmsNorm,
    #[param]
    pub post_feedforward_layernorm: nn::RmsNorm,
}

impl VisionLayer {
    fn new(config: &Gemma4VisionConfig, stream: &Stream) -> Result<Self, Exception> {
        let norm = || {
            nn::RmsNorm::unloaded(
                config.hidden_size,
                config.rms_norm_eps,
                Dtype::Float32,
                stream,
            )
        };
        Ok(Self {
            self_attn: VisionAttention::new(config, stream)?,
            mlp: VisionMlp::new(config, stream)?,
            input_layernorm: norm()?,
            post_attention_layernorm: norm()?,
            pre_feedforward_layernorm: norm()?,
            post_feedforward_layernorm: norm()?,
        })
    }

    fn forward(
        &mut self,
        hidden: &Array,
        padding: &Array,
        cos: &Array,
        sin: &Array,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let attention = self.self_attn.forward(
            &self.input_layernorm.forward(hidden, stream)?,
            padding,
            cos,
            sin,
            stream,
        )?;
        let hidden = hidden.add(
            self.post_attention_layernorm.forward(&attention, stream)?,
            stream,
        )?;
        let feed_forward = self.mlp.forward(
            &self.pre_feedforward_layernorm.forward(&hidden, stream)?,
            stream,
        )?;
        hidden.add(
            self.post_feedforward_layernorm
                .forward(&feed_forward, stream)?,
            stream,
        )
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub(crate) struct VisionEncoder {
    #[param]
    pub layers: Vec<VisionLayer>,
}

#[derive(Debug, Clone, ModuleParameters)]
pub(crate) struct Gemma4VisionTower {
    pub config: Gemma4VisionConfig,
    #[param]
    pub patch_embedder: VisionPatchEmbedder,
    #[param]
    pub encoder: VisionEncoder,
    #[param]
    pub std_bias: Param<Option<Array>>,
    #[param]
    pub std_scale: Param<Option<Array>>,
}

impl Gemma4VisionTower {
    pub(super) fn new(config: Gemma4VisionConfig, stream: &Stream) -> Result<Self, Exception> {
        let patch_embedder = VisionPatchEmbedder::new(&config, stream)?;
        let layers = (0..config.num_hidden_layers)
            .map(|_| VisionLayer::new(&config, stream))
            .collect::<Result<Vec<_>, _>>()?;
        let (std_bias, std_scale) = if config.standardize {
            (
                Param::<Option<Array>>::unloaded_some(
                    &[config.hidden_size],
                    Dtype::Float32,
                    stream,
                )?,
                Param::<Option<Array>>::unloaded_some(
                    &[config.hidden_size],
                    Dtype::Float32,
                    stream,
                )?,
            )
        } else {
            (Param::new(None), Param::new(None))
        };
        Ok(Self {
            config,
            patch_embedder,
            encoder: VisionEncoder { layers },
            std_bias,
            std_scale,
        })
    }

    pub(super) fn forward(
        &mut self,
        pixel_values: &Array,
        position_ids: &Array,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let (mut hidden, padding) =
            self.patch_embedder
                .forward(pixel_values, position_ids, stream)?;
        let working_dtype = hidden.dtype();
        let (cos, sin) = vision_rope(
            position_ids,
            self.config.head_dim,
            self.config.rope_theta(),
            stream,
        )?;
        for layer in &mut self.encoder.layers {
            hidden = layer.forward(&hidden, &padding, &cos, &sin, stream)?;
        }
        let mut hidden = pool_hidden(
            &hidden,
            position_ids,
            self.config.pooling_kernel_size,
            stream,
        )?
        .as_dtype(Dtype::Float32, stream)?
        .multiply(
            Array::from_f32((self.config.hidden_size as f32).sqrt()),
            stream,
        )?;
        if let (Some(bias), Some(scale)) = (self.std_bias.as_ref(), self.std_scale.as_ref()) {
            hidden = hidden.subtract(bias, stream)?.multiply(scale, stream)?;
        }
        hidden.as_dtype(working_dtype, stream)
    }
}

fn validate_vision_inputs(pixel_values: &Array, position_ids: &Array) -> Result<(), Exception> {
    let pixel_shape = pixel_values.shape();
    let position_shape = position_ids.shape();
    if pixel_shape.len() != 3 || position_shape.len() != 3 || position_shape[2] != 2 {
        return Err(Exception::custom(format!(
            "Gemma 4 image tensor must be [batch, patches, patch_dims] with positions [batch, patches, 2], got {pixel_shape:?} and {position_shape:?}"
        )));
    }
    if pixel_shape[..2] != position_shape[..2] {
        return Err(Exception::custom(format!(
            "Gemma 4 image tensor and position IDs disagree on batch/patch dimensions: {pixel_shape:?} vs {position_shape:?}"
        )));
    }
    Ok(())
}

fn vision_rope(
    position_ids: &Array,
    head_dim: i32,
    theta: f32,
    stream: &Stream,
) -> Result<(Array, Array), Exception> {
    let spatial_dim = head_dim / 2;
    let inv_freq = (0..spatial_dim)
        .step_by(2)
        .map(|index| 1.0 / theta.powf(index as f32 / spatial_dim as f32))
        .collect::<Vec<_>>();
    let inv_freq = Array::from_slice(&inv_freq, &[1, 1, inv_freq.len() as i32]);
    let positions =
        maximum(position_ids, Array::from_int(0), stream)?.as_dtype(Dtype::Float32, stream)?;
    let mut dimensions = Vec::with_capacity(2);
    for dimension in 0..2 {
        let angles = positions
            .try_index_device((.., .., dimension), stream)?
            .try_index_device((.., .., NewAxis), stream)?
            .multiply(&inv_freq, stream)?;
        dimensions.push(concatenate_axis(&[angles.clone(), angles], -1, stream)?);
    }
    let angles = concatenate_axis(&[dimensions[0].clone(), dimensions[1].clone()], -1, stream)?;
    Ok((angles.cos(stream)?, angles.sin(stream)?))
}

fn apply_2d_rope(x: Array, cos: &Array, sin: &Array, stream: &Stream) -> Result<Array, Exception> {
    let half = x.dim(3) / 2;
    let mut outputs = Vec::with_capacity(2);
    for dimension in 0..2 {
        let start = dimension * half;
        let end = start + half;
        let part = x.try_index_device((.., .., .., start..end), stream)?;
        let part_cos = cos
            .try_index_device((.., .., start..end), stream)?
            .try_index_device((.., .., NewAxis, ..), stream)?;
        let part_sin = sin
            .try_index_device((.., .., start..end), stream)?
            .try_index_device((.., .., NewAxis, ..), stream)?;
        let quarter = half / 2;
        let left = part.try_index_device((.., .., .., ..quarter), stream)?;
        let right = part.try_index_device((.., .., .., quarter..), stream)?;
        let rotated = concatenate_axis(&[right.negative(stream)?, left], -1, stream)?;
        outputs.push(
            part.multiply(part_cos, stream)?
                .add(rotated.multiply(part_sin, stream)?, stream)?,
        );
    }
    concatenate_axis(&outputs, -1, stream)
}

fn pool_hidden(
    hidden: &Array,
    position_ids: &Array,
    kernel: i32,
    stream: &Stream,
) -> Result<Array, Exception> {
    let x_positions = position_ids.try_index_device((.., .., 0), stream)?;
    let y_positions = position_ids.try_index_device((.., .., 1), stream)?;
    let width = x_positions.max_axis(1, false, stream)?.item::<i32>(stream) + 1;
    let height = y_positions.max_axis(1, false, stream)?.item::<i32>(stream) + 1;
    if width % kernel != 0 || height % kernel != 0 {
        return Err(Exception::custom(format!(
            "Gemma 4 patch grid {height}x{width} is not divisible by pooling kernel {kernel}"
        )));
    }
    let real_patches = width * height;
    let hidden = hidden
        .try_index_device((.., ..real_patches, ..), stream)?
        .reshape(
            &[
                hidden.dim(0),
                height / kernel,
                kernel,
                width / kernel,
                kernel,
                hidden.dim(2),
            ],
            stream,
        )?;
    mean_axes(&hidden, &[2, 4], false, stream)?.reshape(&[hidden.dim(0), -1, hidden.dim(5)], stream)
}
