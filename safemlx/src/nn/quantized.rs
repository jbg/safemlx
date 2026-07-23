use std::iter::once;

use crate::{
    error::Exception,
    module::{Module, ModuleParameters, Param},
    native_quantization::NativeQuantizedTensor,
    ops::indexing::TryIndexOp,
    ops::{
        self, dequantize_with_mode, quantized_matmul_with_mode, quantized_packed_dimension,
        QuantizationMode,
    },
    quantization::Quantizable,
    Array, Dtype, Stream,
};
use safemlx_macros::ModuleParameters;

use crate::nn::{Embedding, Linear};

/// Quantize a module.
///
/// # Params
///
/// - `module`: The module to quantize.
/// - `group_size`: The group size to use for the quantized weight. Default to [`Quantizable::DEFAULT_GROUP_SIZE`]
/// - `bits`: The bit width to use for the quantized weight. Default to [`Quantizable::DEFAULT_BITS`]
pub fn quantize<M>(
    module: M,
    group_size: impl Into<Option<i32>>,
    bits: impl Into<Option<i32>>,
    stream: &crate::Stream,
) -> Result<M::Quantized, M::QuantizationError>
where
    M: Quantizable,
{
    let group_size = group_size.into().unwrap_or(M::DEFAULT_GROUP_SIZE);
    let bits = bits.into().unwrap_or(M::DEFAULT_BITS);
    module.try_into_quantized(group_size, bits, stream)
}

/// Builder for [`QuantizedEmbedding`]
#[derive(Debug, Clone)]
pub struct QuantizedEmbeddingBuilder {
    /// How many possible discrete tokens can we embed. Usually called the vocabulary size.
    pub embedding_count: i32,

    /// The dimensionality of the embeddings.
    pub dimensions: i32,

    /// Quantization group size. Default to [`QuantizedEmbedding::DEFAULT_GROUP_SIZE`]
    pub group_size: i32,

    /// Bits per parameter. Default to [`QuantizedEmbedding::DEFAULT_BITS`]
    pub bits: i32,

    /// Quantized weight encoding.
    pub mode: QuantizationMode,
}

/// The same as ``Embedding`` but with a quantized weight matrix.
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = crate)]
pub struct QuantizedEmbedding {
    /// Quantization group size. Default to [`QuantizedEmbedding::DEFAULT_GROUP_SIZE`]
    pub group_size: i32,

    /// Bits per parameter. Default to [`QuantizedEmbedding::DEFAULT_BITS`]
    pub bits: i32,

    /// Quantized weight encoding.
    pub mode: QuantizationMode,

    /// Optional checkpoint-native storage used instead of affine parameters.
    pub native: Option<NativeQuantizedTensor>,

    /// Scales
    #[param]
    pub scales: Param<Array>,

    /// Biases
    #[param]
    pub biases: Param<Option<Array>>,

    /// Inner embedding
    #[param]
    pub inner: Embedding,
}

impl QuantizedEmbeddingBuilder {
    /// Create a builder for [`QuantizedEmbedding`].
    pub fn new(embedding_count: impl Into<i32>, dimensions: impl Into<i32>) -> Self {
        Self {
            embedding_count: embedding_count.into(),
            dimensions: dimensions.into(),
            group_size: QuantizedEmbedding::DEFAULT_GROUP_SIZE,
            bits: QuantizedEmbedding::DEFAULT_BITS,
            mode: QuantizationMode::Affine,
        }
    }

    /// Set the quantization group size.
    pub fn group_size(mut self, group_size: impl Into<i32>) -> Self {
        self.group_size = group_size.into();
        self
    }

    /// Set the quantization bit width.
    pub fn bits(mut self, bits: impl Into<i32>) -> Self {
        self.bits = bits.into();
        self
    }

    /// Set the quantized weight encoding.
    pub fn mode(mut self, mode: QuantizationMode) -> Self {
        self.mode = mode;
        self
    }

    /// Build a new [`QuantizedEmbedding`] using `stream` for quantization.
    pub fn build(self, stream: &crate::Stream) -> Result<QuantizedEmbedding, Exception> {
        build_quantized_embedding(self, stream)
    }

    /// Convenience method to build a new [`QuantizedEmbedding`] with an existing [`Embedding`]
    pub fn build_with_embedding(
        self,
        embedding: Embedding,
        stream: &crate::Stream,
    ) -> Result<QuantizedEmbedding, Exception> {
        let weight = embedding.weight.value;
        self.build_with_weight(weight, stream)
    }

    /// Convenience method to build a new [`QuantizedEmbedding`] with an existing weight matrix
    pub fn build_with_weight(
        self,
        weight: Array,
        stream: &crate::Stream,
    ) -> Result<QuantizedEmbedding, Exception> {
        let group_size = self.group_size;
        let bits = self.bits;
        build_quantized_embedding_inner(weight, group_size, bits, self.mode, stream)
    }
}

fn build_quantized_embedding_inner(
    weight: Array,
    group_size: i32,
    bits: i32,
    mode: QuantizationMode,
    stream: &crate::Stream,
) -> Result<QuantizedEmbedding, Exception> {
    let arrays = ops::quantize_with_mode(&weight, group_size, bits, mode, stream)?;

    let inner = Embedding {
        weight: Param::new(arrays.weight),
    };

    let mut qe = QuantizedEmbedding {
        group_size,
        bits,
        mode,
        native: None,
        scales: Param::new(arrays.scales),
        biases: Param::new(arrays.biases),
        inner,
    };

    // Freeze all parameters
    qe.freeze_parameters(true);

    Ok(qe)
}

fn build_quantized_embedding(
    builder: QuantizedEmbeddingBuilder,
    stream: &crate::Stream,
) -> Result<QuantizedEmbedding, Exception> {
    let embedding_count = builder.embedding_count;
    let dims = builder.dimensions;

    let scale = f32::sqrt(1.0 / (dims as f32));
    let weight = super::init::uniform(-scale, scale, &[embedding_count, dims]);

    builder.build_with_weight(weight, stream)
}

impl QuantizedEmbedding {
    /// Default group size
    pub const DEFAULT_GROUP_SIZE: i32 = 64;

    /// Default bits
    pub const DEFAULT_BITS: i32 = 4;

    /// Creates a quantized embedding layer whose parameters carry only shape
    /// metadata.
    ///
    /// This is intended for modules that will immediately load real
    /// checkpoint weights before any forward pass.
    pub fn unloaded(
        embedding_count: i32,
        dimensions: i32,
        group_size: i32,
        bits: i32,
        stream: impl AsRef<Stream>,
    ) -> Result<Self, Exception> {
        Self::unloaded_with_mode(
            embedding_count,
            dimensions,
            group_size,
            bits,
            QuantizationMode::Affine,
            stream,
        )
    }

    /// Creates an unloaded quantized embedding with an explicit encoding.
    pub fn unloaded_with_mode(
        embedding_count: i32,
        dimensions: i32,
        group_size: i32,
        bits: i32,
        mode: QuantizationMode,
        stream: impl AsRef<Stream>,
    ) -> Result<Self, Exception> {
        mode.validate(group_size, bits)?;
        let stream = stream.as_ref();
        let scale_dtype = if mode == QuantizationMode::MxFp4 {
            Dtype::Uint8
        } else {
            Dtype::Float32
        };
        let inner = Embedding {
            weight: Param::<Array>::unloaded(
                &[
                    embedding_count,
                    quantized_packed_dimension(dimensions, bits),
                ],
                Dtype::Uint32,
                stream,
            )?,
        };
        let mut qe = Self {
            group_size,
            bits,
            mode,
            native: None,
            scales: Param::<Array>::unloaded(
                &[embedding_count, dimensions / group_size],
                scale_dtype,
                stream,
            )?,
            biases: if mode.has_biases() {
                Param::<Option<Array>>::unloaded_some(
                    &[embedding_count, dimensions / group_size],
                    Dtype::Float32,
                    stream,
                )?
            } else {
                Param::new(None)
            },
            inner,
        };
        qe.freeze_parameters(true);
        Ok(qe)
    }

    /// Create a new quantized embedding using `stream` for quantization.
    pub fn new(
        embedding_count: impl Into<i32>,
        dimensions: impl Into<i32>,
        stream: &crate::Stream,
    ) -> Result<Self, Exception> {
        QuantizedEmbeddingBuilder::new(embedding_count, dimensions).build(stream)
    }

    /// Convert an embedding layer to a quantized embedding layer.
    ///
    /// # Params
    ///
    /// - `embedding`: The embedding layer to convert.
    /// - `group_size`: The group size to use for the quantized weight. Default to [`QuantizedEmbedding::DEFAULT_GROUP_SIZE`]
    /// - `bits`: The bit width to use for the quantized weight. Default to [`QuantizedEmbedding::DEFAULT_BITS`]
    pub fn try_from_embedding(
        embedding: Embedding,
        group_size: impl Into<Option<i32>>,
        bits: impl Into<Option<i32>>,
        stream: &crate::Stream,
    ) -> Result<Self, Exception> {
        let group_size = group_size.into().unwrap_or(Self::DEFAULT_GROUP_SIZE);
        let bits = bits.into().unwrap_or(Self::DEFAULT_BITS);
        build_quantized_embedding_inner(
            embedding.weight.value,
            group_size,
            bits,
            QuantizationMode::Affine,
            stream,
        )
    }

    /// Call the embedding layer as a linear layer.
    ///
    /// Use this for example when input embedding and output projection
    /// weights are tied.
    pub fn as_linear(
        &self,
        x: impl AsRef<Array>,
        stream: &crate::Stream,
    ) -> Result<Array, Exception> {
        if let Some(native) = &self.native {
            return native.linear(x.as_ref(), true, stream);
        }
        quantized_matmul_with_mode(
            x.as_ref(),
            &self.inner.weight,
            &self.scales,
            self.biases.value.as_ref(),
            true,
            self.group_size,
            self.bits,
            self.mode,
            stream,
        )
    }
}

impl Module<&Array> for QuantizedEmbedding {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array, stream: &crate::Stream) -> Result<Array, Self::Error> {
        if let Some(native) = &self.native {
            return native.embedding(x, stream);
        }
        let s = x.shape();
        let x = x.flatten(None, None, stream)?;
        let w = self.inner.weight.try_index_device(&x, stream)?;
        let scales = self.scales.try_index_device(&x, stream)?;
        let biases = self
            .biases
            .value
            .as_ref()
            .map(|biases| biases.try_index_device(&x, stream))
            .transpose()?;

        let out = dequantize_with_mode(
            &w,
            &scales,
            biases.as_ref(),
            self.group_size,
            self.bits,
            self.mode,
            stream,
        )?;

        let ret_shape = s.iter().copied().chain(once(-1)).collect::<Vec<_>>();
        out.reshape(&ret_shape, stream)
    }

    fn training_mode(&mut self, mode: bool) {
        self.inner.training_mode(mode);
    }
}

/// Builder for [`QuantizedLinear`]
#[derive(Debug, Clone)]
pub struct QuantizedLinearBuilder {
    /// The dimensionality of the input features.
    pub input_dims: i32,

    /// The dimensionality of the output features.
    pub output_dims: i32,

    /// Quantization group size. Default to [`QuantizedLinear::DEFAULT_GROUP_SIZE`]
    pub group_size: i32,

    /// Bits per parameter. Default to [`QuantizedLinear::DEFAULT_BITS`]
    pub bits: i32,

    /// Quantized weight encoding.
    pub mode: QuantizationMode,

    /// Whether the linear layer has a bias. Default to [`Linear::DEFAULT_BIAS`]
    pub bias: bool,
}

impl QuantizedLinearBuilder {
    /// Create a builder for [`QuantizedLinear`].
    pub fn new(input_dims: impl Into<i32>, output_dims: impl Into<i32>) -> Self {
        Self {
            input_dims: input_dims.into(),
            output_dims: output_dims.into(),
            group_size: QuantizedLinear::DEFAULT_GROUP_SIZE,
            bits: QuantizedLinear::DEFAULT_BITS,
            mode: QuantizationMode::Affine,
            bias: Linear::DEFAULT_BIAS,
        }
    }

    /// Set the quantization group size.
    pub fn group_size(mut self, group_size: impl Into<i32>) -> Self {
        self.group_size = group_size.into();
        self
    }

    /// Set the quantization bit width.
    pub fn bits(mut self, bits: impl Into<i32>) -> Self {
        self.bits = bits.into();
        self
    }

    /// Set the quantized weight encoding.
    pub fn mode(mut self, mode: QuantizationMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set whether the linear layer has a bias.
    pub fn bias(mut self, bias: impl Into<bool>) -> Self {
        self.bias = bias.into();
        self
    }

    /// Build a new [`QuantizedLinear`] using `stream` for quantization.
    pub fn build(self, stream: &crate::Stream) -> Result<QuantizedLinear, Exception> {
        build_quantized_linear(self, stream)
    }

    /// Convenience method to build a new [`QuantizedLinear`] with an existing [`Linear`]
    pub fn build_with_linear(
        self,
        other: Linear,
        stream: &crate::Stream,
    ) -> Result<QuantizedLinear, Exception> {
        self.build_with_weight_and_bias(other.weight.value, other.bias.value, stream)
    }

    fn build_with_weight_and_bias(
        self,
        weight: Array,
        bias: Option<Array>,
        stream: &crate::Stream,
    ) -> Result<QuantizedLinear, Exception> {
        build_quantized_linear_inner(weight, bias, self.group_size, self.bits, self.mode, stream)
    }
}

fn build_quantized_linear_inner(
    weight: Array,
    bias: Option<Array>,
    group_size: i32,
    bits: i32,
    mode: QuantizationMode,
    stream: &crate::Stream,
) -> Result<QuantizedLinear, Exception> {
    let arrays = ops::quantize_with_mode(&weight, group_size, bits, mode, stream)?;

    let inner = Linear {
        weight: Param::new(arrays.weight),
        bias: Param::new(bias),
    };

    let mut ql = QuantizedLinear {
        group_size,
        bits,
        mode,
        native: None,
        scales: Param::new(arrays.scales),
        biases: Param::new(arrays.biases),
        inner,
    };

    // Freeze all parameters
    ql.freeze_parameters(true);

    Ok(ql)
}

/// Builds a new [`QuantizedLinear`]
pub fn build_quantized_linear(
    builder: QuantizedLinearBuilder,
    stream: &crate::Stream,
) -> Result<QuantizedLinear, Exception> {
    let input_dims = builder.input_dims;
    let output_dims = builder.output_dims;
    let scale = f32::sqrt(1.0 / (input_dims as f32));
    let weight = super::init::uniform(-scale, scale, &[output_dims, input_dims]);

    let bias = if builder.bias {
        Some(super::init::zeros(&[output_dims]))
    } else {
        None
    };

    builder.build_with_weight_and_bias(weight, bias, stream)
}

/// Applies an affine transformation to the input using a quantized weight matrix.
///
/// It is the quantized equivalent of [`Linear`].  For now its
/// parameters are frozen and will not be included in any gradient computation
/// but this will probably change in the future.
///
/// QuantizedLinear also provides several useful static to convert linear
/// layers to QuantizedLinear layers.
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = crate)]
pub struct QuantizedLinear {
    /// Quantization group size. Default to [`QuantizedLinear::DEFAULT_GROUP_SIZE`]
    pub group_size: i32,

    /// Bits per parameter. Default to [`QuantizedLinear::DEFAULT_BITS`]
    pub bits: i32,

    /// Quantized weight encoding.
    pub mode: QuantizationMode,

    /// Optional checkpoint-native storage used instead of affine parameters.
    pub native: Option<NativeQuantizedTensor>,

    /// Scales
    #[param]
    pub scales: Param<Array>,

    /// Biases
    #[param]
    pub biases: Param<Option<Array>>,

    /// Inner linear layer
    #[param]
    pub inner: Linear,
}

impl QuantizedLinear {
    /// Default group size
    pub const DEFAULT_GROUP_SIZE: i32 = 64;

    /// Default bits
    pub const DEFAULT_BITS: i32 = 4;

    /// Creates a quantized linear layer whose parameters carry only shape
    /// metadata.
    ///
    /// This is intended for modules that will immediately load real
    /// checkpoint weights before any forward pass.
    pub fn unloaded(
        input_dims: i32,
        output_dims: i32,
        group_size: i32,
        bits: i32,
        bias: bool,
        stream: impl AsRef<Stream>,
    ) -> Result<Self, Exception> {
        Self::unloaded_with_mode(
            input_dims,
            output_dims,
            group_size,
            bits,
            QuantizationMode::Affine,
            bias,
            stream,
        )
    }

    /// Creates an unloaded quantized linear layer with an explicit encoding.
    #[allow(clippy::too_many_arguments)]
    pub fn unloaded_with_mode(
        input_dims: i32,
        output_dims: i32,
        group_size: i32,
        bits: i32,
        mode: QuantizationMode,
        bias: bool,
        stream: impl AsRef<Stream>,
    ) -> Result<Self, Exception> {
        mode.validate(group_size, bits)?;
        let stream = stream.as_ref();
        let scale_dtype = if mode == QuantizationMode::MxFp4 {
            Dtype::Uint8
        } else {
            Dtype::Float32
        };
        let inner = Linear {
            weight: Param::<Array>::unloaded(
                &[output_dims, quantized_packed_dimension(input_dims, bits)],
                Dtype::Uint32,
                stream,
            )?,
            bias: if bias {
                Param::<Option<Array>>::unloaded_some(&[output_dims], Dtype::Float32, stream)?
            } else {
                Param::new(None)
            },
        };
        let mut ql = Self {
            group_size,
            bits,
            mode,
            native: None,
            scales: Param::<Array>::unloaded(
                &[output_dims, input_dims / group_size],
                scale_dtype,
                stream,
            )?,
            biases: if mode.has_biases() {
                Param::<Option<Array>>::unloaded_some(
                    &[output_dims, input_dims / group_size],
                    Dtype::Float32,
                    stream,
                )?
            } else {
                Param::new(None)
            },
            inner,
        };
        ql.freeze_parameters(true);
        Ok(ql)
    }

    /// Create a new quantized linear layer using `stream` for quantization.
    pub fn new(
        input_dims: impl Into<i32>,
        output_dims: impl Into<i32>,
        stream: &crate::Stream,
    ) -> Result<Self, Exception> {
        QuantizedLinearBuilder::new(input_dims, output_dims).build(stream)
    }

    /// Convert a linear layer to a quantized linear layer.
    ///
    /// # Params
    ///
    /// - `linear`: The linear layer to convert.
    /// - `group_size`: The group size to use for the quantized weight. Default to [`QuantizedLinear::DEFAULT_GROUP_SIZE`]
    /// - `bits`: The bit width to use for the quantized weight. Default to [`QuantizedLinear::DEFAULT_BITS`]
    pub fn try_from_linear(
        linear: Linear,
        group_size: impl Into<Option<i32>>,
        bits: impl Into<Option<i32>>,
        stream: &crate::Stream,
    ) -> Result<Self, Exception> {
        let group_size = group_size.into().unwrap_or(Self::DEFAULT_GROUP_SIZE);
        let bits = bits.into().unwrap_or(Self::DEFAULT_BITS);
        build_quantized_linear_inner(
            linear.weight.value,
            linear.bias.value,
            group_size,
            bits,
            QuantizationMode::Affine,
            stream,
        )
    }
}

impl Module<&Array> for QuantizedLinear {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array, stream: &crate::Stream) -> Result<Array, Self::Error> {
        if let Some(native) = &self.native {
            let mut output = native.linear(x, true, stream)?;
            if let Some(bias) = &self.inner.bias.value {
                output = output.add(bias, stream)?;
            }
            return Ok(output);
        }
        let mut x = quantized_matmul_with_mode(
            x,
            &self.inner.weight,
            &self.scales,
            self.biases.value.as_ref(),
            true,
            self.group_size,
            self.bits,
            self.mode,
            stream,
        )?;
        if let Some(bias) = &self.inner.bias.value {
            x = x.add(bias, stream)?;
        }
        Ok(x)
    }

    fn training_mode(&mut self, mode: bool) {
        self.inner.training_mode(mode);
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        array,
        module::{Module, ModuleParameters},
        ops::QuantizationMode,
        random::{randint, uniform},
        Dtype,
    };

    use super::{
        QuantizedEmbedding, QuantizedEmbeddingBuilder, QuantizedLinear, QuantizedLinearBuilder,
    };

    #[test]
    fn unloaded_parameters_use_mlx_bit_packing() {
        let stream = crate::test_stream();
        let linear = QuantizedLinear::unloaded(1024, 8, 32, 5, false, stream).unwrap();
        assert_eq!(linear.inner.weight.shape(), &[8, 160]);
        assert_eq!(linear.scales.shape(), &[8, 32]);

        let embedding = QuantizedEmbedding::unloaded(16, 1024, 32, 5, stream).unwrap();
        assert_eq!(embedding.inner.weight.shape(), &[16, 160]);
        assert_eq!(embedding.scales.shape(), &[16, 32]);

        let q6_linear = QuantizedLinear::unloaded(1024, 8, 16, 6, false, stream).unwrap();
        assert_eq!(q6_linear.inner.weight.shape(), &[8, 192]);
        assert_eq!(q6_linear.scales.shape(), &[8, 64]);

        let q6_embedding = QuantizedEmbedding::unloaded(16, 1024, 16, 6, stream).unwrap();
        assert_eq!(q6_embedding.inner.weight.shape(), &[16, 192]);
        assert_eq!(q6_embedding.scales.shape(), &[16, 64]);

        let q2_linear = QuantizedLinear::unloaded(1024, 8, 16, 2, false, stream).unwrap();
        assert_eq!(q2_linear.inner.weight.shape(), &[8, 64]);
        assert_eq!(q2_linear.scales.shape(), &[8, 64]);

        let q3_linear = QuantizedLinear::unloaded(1024, 8, 16, 3, false, stream).unwrap();
        assert_eq!(q3_linear.inner.weight.shape(), &[8, 96]);
        assert_eq!(q3_linear.scales.shape(), &[8, 64]);
    }

    #[test]
    fn quantized_linear_new_requires_explicit_stream() {
        let stream = crate::test_stream();
        let mut layer = QuantizedLinear::new(64, 4, stream).unwrap();
        let key = crate::test_key(13, stream);
        let input = uniform::<_, f32>(-1.0, 1.0, &[2, 64], &key, stream).unwrap();

        let output = layer.forward(&input, stream).unwrap();

        assert_eq!(output.shape(), &[2, 4]);
        assert_eq!(output.dtype(), Dtype::Float32);
    }

    #[test]
    fn quantized_embedding_builder_requires_explicit_stream() {
        let stream = crate::test_stream();
        let mut embedding = QuantizedEmbeddingBuilder::new(16, 64)
            .group_size(32)
            .bits(4)
            .build(stream)
            .unwrap();
        let key = crate::test_key(21, stream);
        let input = randint::<_, i32>(array!(0), array!(16), &[2, 3], &key, stream).unwrap();

        let output = embedding.forward(&input, stream).unwrap();

        assert_eq!(output.shape(), &[2, 3, 64]);
        assert_eq!(output.dtype(), Dtype::Float32);
    }

    #[test]
    fn mxfp4_modules_execute_without_quantization_bias_parameters() {
        let stream = crate::test_stream();
        let mut linear = QuantizedLinearBuilder::new(64, 8)
            .group_size(32)
            .bits(4)
            .mode(QuantizationMode::MxFp4)
            .bias(true)
            .build(stream)
            .unwrap();
        let input = crate::Array::ones::<f32>(&[2, 64], stream).unwrap();
        assert_eq!(linear.forward(&input, stream).unwrap().shape(), &[2, 8]);
        let linear_keys = linear.parameters().flatten();
        assert!(linear_keys.contains_key("inner.weight"));
        assert!(linear_keys.contains_key("scales"));
        assert!(!linear_keys.contains_key("biases"));
        assert!(linear_keys.contains_key("inner.bias"));

        let mut embedding = QuantizedEmbeddingBuilder::new(16, 64)
            .group_size(32)
            .bits(4)
            .mode(QuantizationMode::MxFp4)
            .build(stream)
            .unwrap();
        let tokens = crate::Array::from_slice(&[0i32, 3, 7, 15], &[2, 2]);
        assert_eq!(
            embedding.forward(&tokens, stream).unwrap().shape(),
            &[2, 2, 64]
        );
        let embedding_keys = embedding.parameters().flatten();
        assert!(!embedding_keys.contains_key("biases"));
    }
}
