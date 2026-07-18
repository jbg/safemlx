use safemlx::{
    error::Exception,
    macros::ModuleParameters,
    module::{Module, Param},
    nn,
    ops::{
        concatenate_axis, conv1d, conv2d,
        indexing::{NewAxis, TryIndexOp},
        matmul, pad, softmax_axis, stack_axis, tanh, PadWidth,
    },
    Array, Dtype, Stream,
};
use serde::Deserialize;

use super::gemma4_multimodal::Gemma4ClippedLinear;

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct Gemma4AudioConfig {
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,
    pub output_proj_dims: i32,
    pub conv_kernel_size: i32,
    pub attention_chunk_size: i32,
    pub attention_context_left: i32,
    pub attention_context_right: i32,
    pub attention_invalid_logits_value: f32,
    pub attention_logit_cap: f32,
    pub residual_weight: f32,
    pub rms_norm_eps: f32,
    pub subsampling_conv_channels: Vec<i32>,
}

impl Gemma4AudioConfig {
    fn head_dim(&self) -> i32 {
        self.hidden_size / self.num_attention_heads
    }
}

fn scale_layer_norm(
    dimensions: i32,
    eps: f32,
    stream: &Stream,
) -> Result<nn::LayerNorm, Exception> {
    Ok(nn::LayerNorm {
        dimensions,
        eps,
        weight: Param::<Option<Array>>::unloaded_some(&[dimensions], Dtype::Float32, stream)?,
        bias: Param::new(None),
    })
}

#[derive(Debug, Clone, ModuleParameters)]
pub(crate) struct AudioConvWeight {
    #[param]
    pub weight: Param<Array>,
}

#[derive(Debug, Clone, ModuleParameters)]
pub(crate) struct AudioSubsampleLayer {
    #[param]
    pub conv: AudioConvWeight,
    #[param]
    pub norm: nn::LayerNorm,
}

impl AudioSubsampleLayer {
    fn new(input: i32, output: i32, eps: f32, stream: &Stream) -> Result<Self, Exception> {
        Ok(Self {
            // Gemma's converted checkpoints already store MLX's [out, h, w, in] layout.
            conv: AudioConvWeight {
                weight: Param::<Array>::unloaded(&[output, 3, 3, input], Dtype::Float32, stream)?,
            },
            norm: scale_layer_norm(output, eps, stream)?,
        })
    }

    fn forward(&mut self, x: &Array, stream: &Stream) -> Result<Array, Exception> {
        let x = conv2d(
            x,
            &*self.conv.weight,
            Some((2, 2)),
            Some((1, 1)),
            None,
            None,
            stream,
        )?;
        nn::relu(self.norm.forward(&x, stream)?, stream)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub(crate) struct AudioSubsampleConvProjection {
    #[param]
    pub layer0: AudioSubsampleLayer,
    #[param]
    pub layer1: AudioSubsampleLayer,
    #[param]
    pub input_proj_linear: nn::Linear,
}

impl AudioSubsampleConvProjection {
    pub(crate) fn new(config: &Gemma4AudioConfig, stream: &Stream) -> Result<Self, Exception> {
        if config.attention_context_right != 0 {
            return Err(Exception::custom(
                "Gemma 4 audio currently requires zero right attention context",
            ));
        }
        if config.attention_context_left <= 1 {
            return Err(Exception::custom(
                "Gemma 4 audio requires attention_context_left greater than one",
            ));
        }
        if config.subsampling_conv_channels.len() != 2 {
            return Err(Exception::custom(
                "Gemma 4 audio requires exactly two subsampling convolution channels",
            ));
        }
        let first = config.subsampling_conv_channels[0];
        let second = config.subsampling_conv_channels[1];
        Ok(Self {
            layer0: AudioSubsampleLayer::new(1, first, config.rms_norm_eps, stream)?,
            layer1: AudioSubsampleLayer::new(first, second, config.rms_norm_eps, stream)?,
            input_proj_linear: nn::Linear::unloaded(
                32 * second,
                config.hidden_size,
                false,
                Dtype::Float32,
                stream,
            )?,
        })
    }

    pub(crate) fn forward(
        &mut self,
        features: &Array,
        valid_frames: i32,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        if features.shape().len() != 3 || features.dim(2) != 128 {
            return Err(Exception::custom(format!(
                "Gemma 4 audio features must be [batch, frames, 128], got {:?}",
                features.shape()
            )));
        }
        let input_mask = Array::from_slice(
            &(0..features.dim(1))
                .map(|index| index < valid_frames)
                .collect::<Vec<_>>(),
            &[1, features.dim(1), 1],
        );
        let x = features
            .multiply(input_mask, stream)?
            .try_index_device((.., .., .., NewAxis), stream)?;
        let x = self.layer0.forward(&x, stream)?;
        let first_valid = (valid_frames + 1) / 2;
        let first_mask = Array::from_slice(
            &(0..x.dim(1))
                .map(|index| index < first_valid)
                .collect::<Vec<_>>(),
            &[1, x.dim(1), 1, 1],
        );
        let x = self
            .layer1
            .forward(&x.multiply(first_mask, stream)?, stream)?;
        let x = x.reshape(&[x.dim(0), x.dim(1), x.dim(2) * x.dim(3)], stream)?;
        self.input_proj_linear.forward(&x, stream)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub(crate) struct AudioFeedForward {
    #[param]
    pub pre_layer_norm: nn::RmsNorm,
    #[param]
    pub ffw_layer_1: Gemma4ClippedLinear,
    #[param]
    pub ffw_layer_2: Gemma4ClippedLinear,
    #[param]
    pub post_layer_norm: nn::RmsNorm,
    pub residual_weight: f32,
}

impl AudioFeedForward {
    fn new(config: &Gemma4AudioConfig, stream: &Stream) -> Result<Self, Exception> {
        Ok(Self {
            pre_layer_norm: nn::RmsNorm::unloaded(
                config.hidden_size,
                config.rms_norm_eps,
                Dtype::Float32,
                stream,
            )?,
            ffw_layer_1: Gemma4ClippedLinear::new(
                config.hidden_size,
                4 * config.hidden_size,
                false,
                stream,
            )?,
            ffw_layer_2: Gemma4ClippedLinear::new(
                4 * config.hidden_size,
                config.hidden_size,
                false,
                stream,
            )?,
            post_layer_norm: nn::RmsNorm::unloaded(
                config.hidden_size,
                config.rms_norm_eps,
                Dtype::Float32,
                stream,
            )?,
            residual_weight: config.residual_weight,
        })
    }

    fn forward(&mut self, x: &Array, stream: &Stream) -> Result<Array, Exception> {
        let residual = x.clone();
        let x = self.pre_layer_norm.forward(x, stream)?;
        let x = nn::silu(self.ffw_layer_1.forward(&x, stream)?, stream)?;
        let x = self.ffw_layer_2.forward(&x, stream)?;
        let x = self.post_layer_norm.forward(&x, stream)?;
        residual.add(
            x.multiply(Array::from_f32(self.residual_weight), stream)?,
            stream,
        )
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub(crate) struct AudioLightConv1d {
    #[param]
    pub pre_layer_norm: nn::RmsNorm,
    #[param]
    pub linear_start: Gemma4ClippedLinear,
    #[param]
    pub depthwise_conv1d: AudioConvWeight,
    #[param]
    pub conv_norm: nn::RmsNorm,
    #[param]
    pub linear_end: Gemma4ClippedLinear,
    pub kernel_size: i32,
}

impl AudioLightConv1d {
    fn new(config: &Gemma4AudioConfig, stream: &Stream) -> Result<Self, Exception> {
        let hidden = config.hidden_size;
        Ok(Self {
            pre_layer_norm: nn::RmsNorm::unloaded(
                hidden,
                config.rms_norm_eps,
                Dtype::Float32,
                stream,
            )?,
            linear_start: Gemma4ClippedLinear::new(hidden, 2 * hidden, false, stream)?,
            depthwise_conv1d: AudioConvWeight {
                weight: Param::<Array>::unloaded(
                    &[hidden, config.conv_kernel_size, 1],
                    Dtype::Float32,
                    stream,
                )?,
            },
            conv_norm: nn::RmsNorm::unloaded(hidden, config.rms_norm_eps, Dtype::Float32, stream)?,
            linear_end: Gemma4ClippedLinear::new(hidden, hidden, false, stream)?,
            kernel_size: config.conv_kernel_size,
        })
    }

    fn forward(&mut self, x: &Array, stream: &Stream) -> Result<Array, Exception> {
        let residual = x.clone();
        let projected = self
            .linear_start
            .forward(&self.pre_layer_norm.forward(x, stream)?, stream)?;
        let hidden = projected.dim(2) / 2;
        let left = projected.try_index_device((.., .., ..hidden), stream)?;
        let right = projected.try_index_device((.., .., hidden..), stream)?;
        let gated = left.multiply(nn::sigmoid(right, stream)?, stream)?;
        let padded = pad(
            &gated,
            PadWidth::from(&[(0, 0), (self.kernel_size - 1, 0), (0, 0)][..]),
            None,
            None,
            stream,
        )?;
        let x = conv1d(
            &padded,
            &*self.depthwise_conv1d.weight,
            None,
            None,
            None,
            Some(hidden),
            stream,
        )?;
        let x = nn::silu(self.conv_norm.forward(&x, stream)?, stream)?;
        residual.add(self.linear_end.forward(&x, stream)?, stream)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub(crate) struct AudioAttention {
    #[param]
    pub q_proj: Gemma4ClippedLinear,
    #[param]
    pub k_proj: Gemma4ClippedLinear,
    #[param]
    pub v_proj: Gemma4ClippedLinear,
    #[param]
    pub post: Gemma4ClippedLinear,
    #[param]
    pub relative_k_proj: nn::Linear,
    #[param]
    pub per_dim_scale: Param<Array>,
    pub heads: i32,
    pub head_dim: i32,
    pub chunk_size: i32,
    pub past: i32,
    pub logit_cap: f32,
    pub invalid_logits: f32,
}

impl AudioAttention {
    fn new(config: &Gemma4AudioConfig, stream: &Stream) -> Result<Self, Exception> {
        let hidden = config.hidden_size;
        Ok(Self {
            q_proj: Gemma4ClippedLinear::new(hidden, hidden, false, stream)?,
            k_proj: Gemma4ClippedLinear::new(hidden, hidden, false, stream)?,
            v_proj: Gemma4ClippedLinear::new(hidden, hidden, false, stream)?,
            post: Gemma4ClippedLinear::new(hidden, hidden, false, stream)?,
            relative_k_proj: nn::Linear::unloaded(hidden, hidden, false, Dtype::Float32, stream)?,
            per_dim_scale: Param::<Array>::unloaded(&[config.head_dim()], Dtype::Float32, stream)?,
            heads: config.num_attention_heads,
            head_dim: config.head_dim(),
            chunk_size: config.attention_chunk_size,
            past: config.attention_context_left - 1,
            logit_cap: config.attention_logit_cap,
            invalid_logits: config.attention_invalid_logits_value,
        })
    }

    fn relative_embeddings(&self) -> Array {
        let mut values =
            Vec::with_capacity(((self.past + 1) * self.heads * self.head_dim) as usize);
        let hidden = self.heads * self.head_dim;
        let timescales = hidden / 2;
        let increment = 10_000.0_f32.ln() / (timescales - 1).max(1) as f32;
        for position in (0..=self.past).rev() {
            for index in 0..timescales {
                values.push((position as f32 * (-increment * index as f32).exp()).sin());
            }
            for index in 0..timescales {
                values.push((position as f32 * (-increment * index as f32).exp()).cos());
            }
        }
        Array::from_slice(&values, &[self.past + 1, self.heads * self.head_dim])
    }

    fn forward(&mut self, x: &Array, valid: i32, stream: &Stream) -> Result<Array, Exception> {
        if x.dim(0) != 1 {
            return Err(Exception::custom(
                "Gemma 4 audio currently requires batch size 1",
            ));
        }
        let sequence = x.dim(1);
        let padded_sequence =
            ((sequence + self.chunk_size - 1) / self.chunk_size) * self.chunk_size;
        let high = padded_sequence - sequence;
        let x = if high > 0 {
            pad(
                x,
                PadWidth::from(&[(0, 0), (0, high), (0, 0)][..]),
                None,
                None,
                stream,
            )?
        } else {
            x.clone()
        };
        let query_scale = nn::softplus(&*self.per_dim_scale, stream)?.multiply(
            Array::from_f32((self.head_dim as f32).powf(-0.5) / std::f32::consts::LN_2),
            stream,
        )?;
        let key_scale = 1.0_f32.exp().ln_1p() / std::f32::consts::LN_2;
        let q = self
            .q_proj
            .forward(&x, stream)?
            .reshape(&[1, padded_sequence, self.heads, self.head_dim], stream)?
            .transpose_axes(&[0, 2, 1, 3], stream)?
            .multiply(query_scale, stream)?;
        let k = self
            .k_proj
            .forward(&x, stream)?
            .reshape(&[1, padded_sequence, self.heads, self.head_dim], stream)?
            .transpose_axes(&[0, 2, 1, 3], stream)?
            .multiply(Array::from_f32(key_scale), stream)?;
        let v = self
            .v_proj
            .forward(&x, stream)?
            .reshape(&[1, padded_sequence, self.heads, self.head_dim], stream)?
            .transpose_axes(&[0, 2, 1, 3], stream)?;
        let relative = self
            .relative_k_proj
            .forward(&self.relative_embeddings(), stream)?
            .reshape(&[self.past + 1, self.heads, self.head_dim], stream)?
            .transpose_axes(&[1, 0, 2], stream)?;
        let mut outputs = Vec::new();
        for start in (0..padded_sequence).step_by(self.chunk_size as usize) {
            let q_chunk =
                q.try_index_device((.., .., start..start + self.chunk_size, ..), stream)?;
            let key_start = (start - self.past).max(0);
            let key_end = (start + self.chunk_size).min(padded_sequence);
            let k_chunk = k.try_index_device((.., .., key_start..key_end, ..), stream)?;
            let v_chunk = v.try_index_device((.., .., key_start..key_end, ..), stream)?;
            let mut logits = matmul(&q_chunk, &k_chunk.swap_axes(2, 3, stream)?, stream)?;
            let relative_logits = matmul(
                &q_chunk,
                &relative
                    .try_index_device((NewAxis, .., .., ..), stream)?
                    .swap_axes(2, 3, stream)?,
                stream,
            )?;
            let key_count = key_end - key_start;
            let mut rows = Vec::with_capacity(self.chunk_size as usize);
            let mut mask = Vec::with_capacity((self.chunk_size * key_count) as usize);
            for query in 0..self.chunk_size {
                let absolute_query = start + query;
                let mut columns = Vec::with_capacity(key_count as usize);
                for key in key_start..key_end {
                    let distance = (absolute_query - key).clamp(0, self.past - 1);
                    columns.push(
                        relative_logits
                            .try_index_device((.., .., query, self.past - distance), stream)?,
                    );
                    mask.push(
                        if key <= absolute_query
                            && absolute_query - key < self.past
                            && key < valid
                            && absolute_query < valid
                        {
                            0.0
                        } else {
                            self.invalid_logits
                        },
                    );
                }
                let refs = columns.iter().collect::<Vec<_>>();
                rows.push(stack_axis(&refs, -1, stream)?);
            }
            let refs = rows.iter().collect::<Vec<_>>();
            let relative_logits = stack_axis(&refs, 2, stream)?;
            logits = logits.add(relative_logits, stream)?;
            logits = tanh(
                &logits.divide(Array::from_f32(self.logit_cap), stream)?,
                stream,
            )?
            .multiply(Array::from_f32(self.logit_cap), stream)?
            .add(
                Array::from_slice(&mask, &[1, 1, self.chunk_size, key_count]),
                stream,
            )?;
            let probabilities = softmax_axis(&logits, -1, None, stream)?;
            outputs.push(matmul(probabilities, v_chunk, stream)?);
        }
        let refs = outputs.iter().collect::<Vec<_>>();
        let output = concatenate_axis(&refs, 2, stream)?
            .try_index_device((.., .., ..sequence, ..), stream)?
            .transpose_axes(&[0, 2, 1, 3], stream)?
            .reshape(&[1, sequence, self.heads * self.head_dim], stream)?;
        self.post.forward(&output, stream)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
/// One Gemma 4 audio encoder block.
pub struct AudioLayer {
    #[param]
    pub(crate) feed_forward1: AudioFeedForward,
    #[param]
    pub(crate) norm_pre_attn: nn::RmsNorm,
    #[param]
    pub(crate) self_attn: AudioAttention,
    #[param]
    pub(crate) norm_post_attn: nn::RmsNorm,
    #[param]
    pub(crate) lconv1d: AudioLightConv1d,
    #[param]
    pub(crate) feed_forward2: AudioFeedForward,
    #[param]
    pub(crate) norm_out: nn::RmsNorm,
}

impl AudioLayer {
    pub(crate) fn new(config: &Gemma4AudioConfig, stream: &Stream) -> Result<Self, Exception> {
        let norm = || {
            nn::RmsNorm::unloaded(
                config.hidden_size,
                config.rms_norm_eps,
                Dtype::Float32,
                stream,
            )
        };
        Ok(Self {
            feed_forward1: AudioFeedForward::new(config, stream)?,
            norm_pre_attn: norm()?,
            self_attn: AudioAttention::new(config, stream)?,
            norm_post_attn: norm()?,
            lconv1d: AudioLightConv1d::new(config, stream)?,
            feed_forward2: AudioFeedForward::new(config, stream)?,
            norm_out: norm()?,
        })
    }

    pub(crate) fn forward(
        &mut self,
        x: &Array,
        valid: i32,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let x = self.feed_forward1.forward(x, stream)?;
        let residual = x.clone();
        let attended =
            self.self_attn
                .forward(&self.norm_pre_attn.forward(&x, stream)?, valid, stream)?;
        let x = residual.add(self.norm_post_attn.forward(&attended, stream)?, stream)?;
        let x = self.lconv1d.forward(&x, stream)?;
        let x = self.feed_forward2.forward(&x, stream)?;
        self.norm_out.forward(&x, stream)
    }
}

/// Pinned audio preprocessing and output projection around layerwise blocks.
#[derive(Debug, Clone, ModuleParameters)]
pub(crate) struct Gemma4AudioLayerwiseStatic {
    #[param]
    pub(crate) subsample_conv_projection: AudioSubsampleConvProjection,
    #[param]
    pub(crate) output_proj: nn::Linear,
}

impl Gemma4AudioLayerwiseStatic {
    pub(crate) fn from_tower(tower: Gemma4AudioTower) -> Self {
        Self {
            subsample_conv_projection: tower.subsample_conv_projection,
            output_proj: tower.output_proj,
        }
    }

    pub(crate) fn begin(
        &mut self,
        features: &Array,
        mask: &Array,
        stream: &Stream,
    ) -> Result<(Array, i32), Exception> {
        if mask.shape().len() != 2
            || mask.dim(0) != features.dim(0)
            || mask.dim(1) != features.dim(1)
        {
            return Err(Exception::custom(format!(
                "Gemma 4 audio mask must be [batch, frames], got {:?} for {:?}",
                mask.shape(),
                features.shape()
            )));
        }
        let valid_frames = mask.sum(None, stream)?.item::<i32>(stream);
        let valid = (valid_frames + 3) / 4;
        Ok((
            self.subsample_conv_projection
                .forward(features, valid_frames, stream)?,
            valid,
        ))
    }

    pub(crate) fn finish(
        &mut self,
        hidden: &Array,
        valid: i32,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.output_proj
            .forward(&hidden.try_index_device((.., ..valid, ..), stream)?, stream)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub(crate) struct Gemma4AudioTower {
    #[param]
    pub subsample_conv_projection: AudioSubsampleConvProjection,
    #[param]
    pub layers: Vec<AudioLayer>,
    #[param]
    pub output_proj: nn::Linear,
}

impl Gemma4AudioTower {
    pub(crate) fn new(config: &Gemma4AudioConfig, stream: &Stream) -> Result<Self, Exception> {
        Ok(Self {
            subsample_conv_projection: AudioSubsampleConvProjection::new(config, stream)?,
            layers: (0..config.num_hidden_layers)
                .map(|_| AudioLayer::new(config, stream))
                .collect::<Result<Vec<_>, _>>()?,
            output_proj: nn::Linear::unloaded(
                config.hidden_size,
                config.output_proj_dims,
                true,
                Dtype::Float32,
                stream,
            )?,
        })
    }

    pub(crate) fn forward(
        &mut self,
        features: &Array,
        mask: &Array,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        if mask.shape().len() != 2
            || mask.dim(0) != features.dim(0)
            || mask.dim(1) != features.dim(1)
        {
            return Err(Exception::custom(format!(
                "Gemma 4 audio mask must be [batch, frames], got {:?} for {:?}",
                mask.shape(),
                features.shape()
            )));
        }
        let valid_frames = mask.sum(None, stream)?.item::<i32>(stream);
        let valid = (valid_frames + 3) / 4;
        let mut hidden = self
            .subsample_conv_projection
            .forward(features, valid_frames, stream)?;
        for layer in &mut self.layers {
            hidden = layer.forward(&hidden, valid, stream)?;
        }
        self.output_proj
            .forward(&hidden.try_index_device((.., ..valid, ..), stream)?, stream)
    }
}
