use safemlx::{
    error::Exception,
    macros::{ModuleParameters, Quantizable},
    module::{Module, Param},
    nn,
    ops::clip,
    quantization::MaybeQuantized,
    Array, Dtype, Stream,
};

use super::gemma4::{maybe_quantized_linear_with_bias, rms_norm_without_scale};

#[derive(Debug, Clone, ModuleParameters)]
pub(crate) struct Gemma4ClippedLinear {
    #[param]
    pub linear: nn::Linear,
    #[param]
    pub input_min: Param<Array>,
    #[param]
    pub input_max: Param<Array>,
    #[param]
    pub output_min: Param<Array>,
    #[param]
    pub output_max: Param<Array>,
}

impl Gemma4ClippedLinear {
    pub(crate) fn new(
        input: i32,
        output: i32,
        bias: bool,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        Ok(Self {
            linear: nn::Linear::unloaded(input, output, bias, Dtype::Float32, stream)?,
            input_min: Param::<Array>::unloaded(&[], Dtype::Float32, stream)?,
            input_max: Param::<Array>::unloaded(&[], Dtype::Float32, stream)?,
            output_min: Param::<Array>::unloaded(&[], Dtype::Float32, stream)?,
            output_max: Param::<Array>::unloaded(&[], Dtype::Float32, stream)?,
        })
    }

    pub(crate) fn forward(&mut self, x: &Array, stream: &Stream) -> Result<Array, Exception> {
        let x = clip(x, (&*self.input_min, &*self.input_max), stream)?;
        let output = self.linear.forward(&x, stream)?;
        clip(output, (&*self.output_min, &*self.output_max), stream)
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub(crate) struct Gemma4ModalityEmbedder {
    pub eps: f32,
    #[quantizable]
    #[param]
    pub embedding_projection: MaybeQuantized<nn::Linear>,
}

impl Gemma4ModalityEmbedder {
    pub(crate) fn new(
        input_size: i32,
        output_size: i32,
        eps: f32,
        bias: bool,
        quantization: (bool, i32, i32),
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let (quantized, group_size, bits) = quantization;
        Ok(Self {
            eps,
            embedding_projection: maybe_quantized_linear_with_bias(
                quantized,
                input_size,
                output_size,
                group_size,
                bits,
                bias,
                stream,
            )?,
        })
    }

    pub(crate) fn forward(&mut self, x: &Array, stream: &Stream) -> Result<Array, Exception> {
        self.embedding_projection
            .forward(&rms_norm_without_scale(x, self.eps, stream)?, stream)
    }
}
