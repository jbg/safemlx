//! Causal convolution layers and their generation cache.

use safemlx::{
    error::Exception,
    macros::ModuleParameters,
    module::Param,
    ops::{concatenate_axis, conv1d, indexing::TryIndexOp, zeros},
    Array, Dtype, Stream,
};

#[derive(Debug, Clone, ModuleParameters)]
/// PyTorch-layout depthwise convolution parameters shared by recurrent LM blocks.
///
/// Checkpoints store weights as `[channels, 1, kernel]`; MLX convolution expects
/// `[channels, kernel, 1]`, so the forward helper transposes the last two axes.
pub struct DepthwiseConv1d {
    #[param]
    /// Convolution weights shaped `[channels, 1, kernel]`.
    pub weight: Param<Array>,
    #[param]
    /// Optional per-channel bias.
    pub bias: Param<Option<Array>>,
    /// Kernel width.
    pub kernel_size: i32,
}

impl DepthwiseConv1d {
    /// Creates unloaded depthwise-convolution parameters.
    pub fn new(
        channels: i32,
        kernel_size: i32,
        bias: bool,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        Ok(Self {
            weight: Param::<Array>::unloaded(&[channels, 1, kernel_size], Dtype::Float32, stream)?,
            bias: if bias {
                Param::<Option<Array>>::unloaded_some(&[channels], Dtype::Float32, stream)?
            } else {
                Param::new(None)
            },
            kernel_size,
        })
    }

    /// Applies a valid depthwise convolution to an already left-padded NLC tensor.
    pub fn forward_padded(&self, padded: &Array, stream: &Stream) -> Result<Array, Exception> {
        let channels = padded.dim(-1);
        let weight = self.weight.as_ref().swap_axes(1, 2, stream)?;
        let mut output = conv1d(
            padded,
            &weight,
            Some(1),
            Some(0),
            Some(1),
            Some(channels),
            stream,
        )?;
        if let Some(bias) = self.bias.as_ref() {
            output = output.add(bias, stream)?;
        }
        Ok(output)
    }
}

#[derive(Debug, Clone, Default)]
/// State retained by a causal depthwise convolution between generation calls.
pub struct CausalConv1dCache {
    /// Previous `kernel_size - 1` inputs shaped `[batch, state, channels]`.
    pub state: Option<Array>,
    /// Number of consumed tokens.
    pub offset: i32,
}

/// Applies a causal depthwise convolution and updates its bounded state.
pub fn causal_depthwise_conv1d(
    convolution: &DepthwiseConv1d,
    input: &Array,
    cache: Option<&mut CausalConv1dCache>,
    stream: &Stream,
) -> Result<Array, Exception> {
    let batch = input.dim(0);
    let seq_len = input.dim(1);
    let channels = input.dim(2);
    let state_len = convolution.kernel_size - 1;
    let state = cache
        .as_ref()
        .and_then(|cache| cache.state.clone())
        .unwrap_or(zeros::<f32>(&[batch, state_len, channels], stream)?);
    let padded = concatenate_axis(&[state, input.clone()], 1, stream)?;
    if let Some(cache) = cache {
        cache.state = Some(padded.try_index_device((.., seq_len.., ..), stream)?);
        cache.offset += seq_len;
    }
    convolution.forward_padded(&padded, stream)
}
