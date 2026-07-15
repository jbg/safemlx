//! Activation functions and feed-forward layers.

use safemlx::{
    builder::Builder,
    error::Exception,
    macros::{ModuleParameters, Quantizable},
    module::Module,
    nn,
    ops::{maximum, sigmoid},
    quantization::MaybeQuantized,
    Array, Dtype, Stream,
};

use crate::{inspection::ActivationObserver, quantization::WeightQuantization};

use super::linear::unloaded_maybe_quantized_linear;

/// Applies the SiLU activation function.
pub fn silu(x: Array, stream: &Stream) -> Result<Array, Exception> {
    x.multiply(sigmoid(&x, stream)?, stream)
}

/// Applies the squared ReLU activation used by Nemotron-H dense and MoE MLPs.
pub fn relu2(x: Array, stream: &Stream) -> Result<Array, Exception> {
    maximum(&x, Array::from_f32(0.0), stream)?.square(stream)
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
/// SwiGLU MLP with optionally quantized projections.
pub struct SwiGluMlp {
    #[quantizable]
    #[param]
    /// Gate projection.
    pub gate_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    /// Down projection back to the model hidden size.
    pub down_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    /// Up projection.
    pub up_proj: MaybeQuantized<nn::Linear>,
}

impl SwiGluMlp {
    /// Creates an initialized SwiGLU MLP.
    pub fn new(dim: i32, hidden_dim: i32, bias: bool) -> Result<Self, Exception> {
        Ok(Self {
            gate_proj: MaybeQuantized::Original(
                nn::LinearBuilder::new(dim, hidden_dim).bias(bias).build()?,
            ),
            down_proj: MaybeQuantized::Original(
                nn::LinearBuilder::new(hidden_dim, dim).bias(bias).build()?,
            ),
            up_proj: MaybeQuantized::Original(
                nn::LinearBuilder::new(dim, hidden_dim).bias(bias).build()?,
            ),
        })
    }

    /// Creates an unloaded SwiGLU MLP whose parameters can be populated from weights.
    pub fn unloaded(
        dim: i32,
        hidden_dim: i32,
        bias: bool,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        Self::unloaded_with_quantization(dim, hidden_dim, bias, None, stream)
    }

    /// Creates an unloaded SwiGLU MLP with optional MLX affine projections.
    pub fn unloaded_with_quantization(
        dim: i32,
        hidden_dim: i32,
        bias: bool,
        quantization: Option<WeightQuantization>,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        Ok(Self {
            gate_proj: unloaded_maybe_quantized_linear(
                dim,
                hidden_dim,
                bias,
                quantization,
                stream,
            )?,
            down_proj: unloaded_maybe_quantized_linear(
                hidden_dim,
                dim,
                bias,
                quantization,
                stream,
            )?,
            up_proj: unloaded_maybe_quantized_linear(dim, hidden_dim, bias, quantization, stream)?,
        })
    }

    /// Forward pass that reports intermediate activations to an observer.
    pub fn forward_with_observer(
        &mut self,
        input: &Array,
        stream: &Stream,
        prefix: &str,
        observer: &mut impl ActivationObserver,
    ) -> Result<Array, Exception> {
        let gate = self.gate_proj.forward(input, stream)?;
        observer.observe(&format!("{prefix}.gate_proj"), &gate)?;

        let up = self.up_proj.forward(input, stream)?;
        observer.observe(&format!("{prefix}.up_proj"), &up)?;

        let activated_gate = silu(gate, stream)?;
        observer.observe(&format!("{prefix}.gate_activation"), &activated_gate)?;

        let down_proj_input = activated_gate.multiply(up, stream)?;
        observer.observe(&format!("{prefix}.down_proj_input"), &down_proj_input)?;

        let output = self.down_proj.forward(&down_proj_input, stream)?;
        observer.observe(&format!("{prefix}.down_proj"), &output)?;
        Ok(output)
    }
}

impl Module<&Array> for SwiGluMlp {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: &Array, stream: &Stream) -> Result<Self::Output, Self::Error> {
        let down_proj_input = silu(self.gate_proj.forward(input, stream)?, stream)?
            .multiply(self.up_proj.forward(input, stream)?, stream)?;
        self.down_proj.forward(&down_proj_input, stream)
    }

    fn training_mode(&mut self, mode: bool) {
        self.gate_proj.training_mode(mode);
        self.down_proj.training_mode(mode);
        self.up_proj.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters)]
/// Dense SwiGLU MLP without quantized projection wrappers.
pub struct DenseSwiGluMlp {
    #[param]
    /// Gate projection.
    pub gate_proj: nn::Linear,
    #[param]
    /// Up projection.
    pub up_proj: nn::Linear,
    #[param]
    /// Down projection back to the model hidden size.
    pub down_proj: nn::Linear,
}

impl DenseSwiGluMlp {
    /// Creates an initialized dense SwiGLU MLP.
    pub fn new(dim: i32, hidden_dim: i32, bias: bool) -> Result<Self, Exception> {
        Ok(Self {
            gate_proj: nn::LinearBuilder::new(dim, hidden_dim).bias(bias).build()?,
            up_proj: nn::LinearBuilder::new(dim, hidden_dim).bias(bias).build()?,
            down_proj: nn::LinearBuilder::new(hidden_dim, dim).bias(bias).build()?,
        })
    }

    /// Creates an unloaded dense SwiGLU MLP whose parameters can be populated from weights.
    pub fn unloaded(
        dim: i32,
        hidden_dim: i32,
        bias: bool,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        Ok(Self {
            gate_proj: nn::Linear::unloaded(dim, hidden_dim, bias, Dtype::Float32, stream)?,
            up_proj: nn::Linear::unloaded(dim, hidden_dim, bias, Dtype::Float32, stream)?,
            down_proj: nn::Linear::unloaded(hidden_dim, dim, bias, Dtype::Float32, stream)?,
        })
    }
}

impl Module<&Array> for DenseSwiGluMlp {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: &Array, stream: &Stream) -> Result<Self::Output, Self::Error> {
        let h = silu(self.gate_proj.forward(input, stream)?, stream)?
            .multiply(self.up_proj.forward(input, stream)?, stream)?;
        self.down_proj.forward(&h, stream)
    }

    fn training_mode(&mut self, mode: bool) {
        self.gate_proj.training_mode(mode);
        self.up_proj.training_mode(mode);
        self.down_proj.training_mode(mode);
    }
}
