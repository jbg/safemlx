//! Thinking Machines Lab Inkling multimodal model support.
//!
//! The released checkpoint is a multimodal conditional-generation model.  This
//! module owns the decoder, the native dMel audio and hMLP vision towers, and
//! the native safetensors loader. Multi-token-prediction draft layers are not
//! needed for ordinary autoregressive generation and are skipped by the loader.

#![allow(missing_docs)]

use std::path::Path;

use safemlx::{
    error::Exception,
    macros::ModuleParameters,
    module::{Module, ModuleParameters as ModuleParametersTrait, ModuleParametersExt, Param},
    nn,
    ops::{
        arange, argpartition_axis, broadcast_to, clip, concatenate_axis,
        indexing::{take_along_axis, NewAxis, TryIndexOp},
        matmul, r#where, sigmoid, softmax_axis, sum_axis,
    },
    Array, Dtype, Stream,
};
use serde::Deserialize;
use serde_json::Value;
use tokenizers::Tokenizer;

use crate::{
    cache::{ConcatKeyValueCache, KeyValueCache, SlidingKeyValueCache},
    error::Error,
    models::{
        common::{
            self,
            convolution::{causal_depthwise_conv1d, CausalConv1dCache, DepthwiseConv1d},
            generation::CausalLm,
            layers::SwiGluMlp,
            moe::PackedSwiGluExperts,
        },
        input,
    },
    weights::{
        for_each_safetensor_array, load_array_strict, safetensors_files, StrictLoadConfig,
        StrictLoadReport,
    },
};

fn default_model_type() -> String {
    "inkling_mm_model".into()
}

fn default_true() -> bool {
    true
}

fn default_rms_norm_eps() -> f32 {
    1e-6
}

fn default_head_dim() -> i32 {
    128
}

fn default_sconv_kernel_size() -> i32 {
    4
}

fn default_rel_extent() -> i32 {
    1024
}

fn default_sliding_window() -> i32 {
    512
}

fn default_route_scale() -> f32 {
    1.0
}

fn default_logit_scale() -> f32 {
    1.0
}

fn default_image_token_id() -> u32 {
    200_054
}

fn default_audio_token_id() -> u32 {
    200_053
}

#[derive(Debug, Clone, Deserialize)]
/// Decoder fields from Inkling's nested `text_config`.
pub struct TextArgs {
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub vocab_size: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    #[serde(default = "default_head_dim")]
    pub head_dim: i32,
    #[serde(default)]
    pub swa_num_attention_heads: Option<i32>,
    #[serde(default)]
    pub swa_num_key_value_heads: Option<i32>,
    #[serde(default)]
    pub swa_head_dim: Option<i32>,
    #[serde(default = "default_sliding_window", alias = "sliding_window")]
    pub sliding_window_size: i32,
    #[serde(default)]
    pub local_layer_ids: Option<Vec<i32>>,
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,
    #[serde(default)]
    pub dense_mlp_idx: i32,
    #[serde(default)]
    pub mlp_layer_types: Option<Vec<String>>,
    #[serde(default = "default_sconv_kernel_size", alias = "conv_kernel_size")]
    pub sconv_kernel_size: i32,
    #[serde(default = "default_true")]
    pub use_sconv: bool,
    #[serde(default = "default_rel_extent")]
    pub rel_extent: i32,
    pub d_rel: i32,
    #[serde(default)]
    pub log_scaling_n_floor: Option<i32>,
    #[serde(default)]
    pub log_scaling_alpha: f32,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_true")]
    pub use_embed_norm: bool,
    #[serde(default)]
    pub unpadded_vocab_size: Option<i32>,
    #[serde(default = "default_logit_scale")]
    pub logits_mup_width_multiplier: f32,
    #[serde(default)]
    pub final_logit_softcapping: Option<f32>,
    #[serde(default)]
    pub intermediate_size: i32,
    #[serde(default)]
    pub dense_intermediate_size: Option<i32>,
    #[serde(default)]
    pub moe_intermediate_size: Option<i32>,
    #[serde(default, alias = "num_experts")]
    pub n_routed_experts: i32,
    #[serde(default)]
    pub num_experts_per_tok: i32,
    #[serde(default)]
    pub n_shared_experts: i32,
    #[serde(default = "default_route_scale")]
    pub route_scale: f32,
    #[serde(default = "default_true")]
    pub shared_expert_sink: bool,
    #[serde(default = "default_true")]
    pub use_gate_bias: bool,
    #[serde(default = "default_true")]
    pub norm_after_topk: bool,
    #[serde(default = "default_true")]
    pub use_global_scale: bool,
    #[serde(default = "default_gate_activation")]
    pub gate_activation: String,
    #[serde(default = "default_hidden_activation")]
    pub hidden_act: String,
    #[serde(default)]
    pub attention_dropout: f32,
    #[serde(default)]
    pub q_bias: bool,
    #[serde(default)]
    pub o_bias: bool,
    #[serde(default)]
    pub model_max_length: Option<i32>,
}

fn default_gate_activation() -> String {
    "sigmoid".into()
}

fn default_hidden_activation() -> String {
    "silu".into()
}

fn default_audio_mode() -> String {
    "dmel".into()
}

fn default_vision_encoder_type() -> String {
    "hmlp".into()
}

impl TextArgs {
    fn dense_intermediate_size(&self) -> i32 {
        self.dense_intermediate_size
            .unwrap_or(self.intermediate_size)
    }

    pub(crate) fn moe_intermediate_size(&self) -> i32 {
        self.moe_intermediate_size.unwrap_or(self.intermediate_size)
    }

    fn is_local(&self, layer: i32) -> bool {
        if let Some(ids) = &self.local_layer_ids {
            return ids.contains(&layer);
        }
        if let Some(types) = &self.layer_types {
            return types
                .get(layer as usize)
                .is_some_and(|kind| kind.contains("sliding"));
        }
        (layer + 1) % 6 != 0
    }

    pub(crate) fn is_dense(&self, layer: i32) -> bool {
        if let Some(types) = &self.mlp_layer_types {
            return types
                .get(layer as usize)
                .is_some_and(|kind| kind == "dense");
        }
        layer < self.dense_mlp_idx
    }

    fn q_heads(&self, local: bool) -> i32 {
        if local {
            self.swa_num_attention_heads
                .unwrap_or(self.num_attention_heads)
        } else {
            self.num_attention_heads
        }
    }

    fn kv_heads(&self, local: bool) -> i32 {
        if local {
            self.swa_num_key_value_heads
                .unwrap_or(self.num_key_value_heads)
        } else {
            self.num_key_value_heads
        }
    }

    fn attention_head_dim(&self, local: bool) -> i32 {
        if local {
            self.swa_head_dim.unwrap_or(self.head_dim)
        } else {
            self.head_dim
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
/// Released top-level Inkling configuration.
pub struct ModelArgs {
    #[serde(default = "default_model_type")]
    pub model_type: String,
    pub text_config: TextArgs,
    #[serde(default)]
    pub audio_config: Option<AudioArgs>,
    #[serde(default)]
    pub vision_config: Option<VisionArgs>,
    #[serde(default = "default_image_token_id")]
    pub image_token_id: u32,
    #[serde(default = "default_audio_token_id")]
    pub audio_token_id: u32,
    #[serde(default)]
    pub eos_token_id: Option<u32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AudioArgs {
    #[serde(alias = "decoder_dmodel")]
    pub text_hidden_size: i32,
    #[serde(alias = "n_mel_bins")]
    pub num_codebooks: i32,
    #[serde(alias = "mel_vocab_size")]
    pub codebook_size: i32,
    #[serde(default)]
    pub bias: bool,
    #[serde(default = "default_true")]
    pub use_audio_norm: bool,
    #[serde(default = "default_audio_mode")]
    pub audio_mode: String,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VisionArgs {
    #[serde(default = "default_vision_encoder_type")]
    pub vision_encoder_type: String,
    #[serde(alias = "decoder_dmodel")]
    pub text_hidden_size: i32,
    pub patch_size: i32,
    pub temporal_patch_size: i32,
    #[serde(alias = "n_channels")]
    pub num_channels: i32,
    #[serde(alias = "n_layers")]
    pub num_hidden_layers: i32,
    #[serde(default = "default_true")]
    pub use_vision_norm: bool,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
}

#[derive(Debug, Clone)]
/// Global or bounded KV state selected per decoder layer.
pub enum InklingKvCache {
    Global(ConcatKeyValueCache),
    Sliding(SlidingKeyValueCache),
}

impl KeyValueCache for InklingKvCache {
    fn offset(&self) -> i32 {
        match self {
            Self::Global(cache) => cache.offset(),
            Self::Sliding(cache) => cache.offset(),
        }
    }

    fn max_size(&self) -> Option<i32> {
        match self {
            Self::Global(cache) => cache.max_size(),
            Self::Sliding(cache) => cache.max_size(),
        }
    }

    fn retained_arrays(&self) -> Vec<&Array> {
        match self {
            Self::Global(cache) => cache.retained_arrays(),
            Self::Sliding(cache) => cache.retained_arrays(),
        }
    }

    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
        stream: &Stream,
    ) -> Result<(Array, Array), Exception> {
        match self {
            Self::Global(cache) => cache.update_and_fetch(keys, values, stream),
            Self::Sliding(cache) => cache.update_and_fetch(keys, values, stream),
        }
    }
}

#[derive(Debug, Clone)]
/// Incremental state for one Inkling decoder layer.
pub struct LayerCache {
    pub kv: InklingKvCache,
    pub convolutions: [CausalConv1dCache; 4],
}

impl LayerCache {
    fn new(local: bool, window: i32) -> Self {
        Self {
            kv: if local {
                InklingKvCache::Sliding(SlidingKeyValueCache::new(window))
            } else {
                InklingKvCache::Global(ConcatKeyValueCache::new())
            },
            convolutions: std::array::from_fn(|_| CausalConv1dCache::default()),
        }
    }
}

#[derive(Debug, Clone)]
/// Heterogeneous Inkling generation cache.
pub struct Cache {
    pub layers: Vec<LayerCache>,
}

impl Cache {
    pub(crate) fn new(args: &TextArgs) -> Self {
        Self {
            layers: (0..args.num_hidden_layers)
                .map(|layer| LayerCache::new(args.is_local(layer), args.sliding_window_size))
                .collect(),
        }
    }

    pub fn offset(&self) -> i32 {
        self.layers.first().map_or(0, |layer| layer.kv.offset())
    }

    pub(crate) fn reset(&mut self) {
        for layer in &mut self.layers {
            match &mut layer.kv {
                InklingKvCache::Global(cache) => cache.clear(),
                InklingKvCache::Sliding(cache) => cache.clear(),
            }
            layer.convolutions = std::array::from_fn(|_| CausalConv1dCache::default());
        }
    }
}

#[derive(Debug, Clone, ModuleParameters)]
struct InklingAttention {
    n_heads: i32,
    n_kv_heads: i32,
    head_dim: i32,
    d_rel: i32,
    rel_extent: i32,
    local: bool,
    sliding_window: i32,
    log_scaling_n_floor: Option<i32>,
    log_scaling_alpha: f32,
    #[param]
    q_proj: nn::Linear,
    #[param]
    k_proj: nn::Linear,
    #[param]
    v_proj: nn::Linear,
    #[param]
    r_proj: nn::Linear,
    #[param]
    o_proj: nn::Linear,
    #[param]
    q_norm: nn::RmsNorm,
    #[param]
    k_norm: nn::RmsNorm,
    #[param]
    rel_proj: Param<Array>,
    #[param]
    k_sconv: DepthwiseConv1d,
    #[param]
    v_sconv: DepthwiseConv1d,
}

impl InklingAttention {
    fn new(args: &TextArgs, layer: i32, stream: &Stream) -> Result<Self, Exception> {
        let local = args.is_local(layer);
        let n_heads = args.q_heads(local);
        let n_kv_heads = args.kv_heads(local);
        let head_dim = args.attention_head_dim(local);
        let rel_extent = if local {
            args.sliding_window_size
        } else {
            args.rel_extent
        };
        Ok(Self {
            n_heads,
            n_kv_heads,
            head_dim,
            d_rel: args.d_rel,
            rel_extent,
            local,
            sliding_window: args.sliding_window_size,
            log_scaling_n_floor: args.log_scaling_n_floor,
            log_scaling_alpha: args.log_scaling_alpha,
            q_proj: nn::Linear::unloaded(
                args.hidden_size,
                n_heads * head_dim,
                false,
                Dtype::Float32,
                stream,
            )?,
            k_proj: nn::Linear::unloaded(
                args.hidden_size,
                n_kv_heads * head_dim,
                false,
                Dtype::Float32,
                stream,
            )?,
            v_proj: nn::Linear::unloaded(
                args.hidden_size,
                n_kv_heads * head_dim,
                false,
                Dtype::Float32,
                stream,
            )?,
            r_proj: nn::Linear::unloaded(
                args.hidden_size,
                n_heads * args.d_rel,
                false,
                Dtype::Float32,
                stream,
            )?,
            o_proj: nn::Linear::unloaded(
                n_heads * head_dim,
                args.hidden_size,
                false,
                Dtype::Float32,
                stream,
            )?,
            q_norm: nn::RmsNorm::unloaded(head_dim, args.rms_norm_eps, Dtype::Float32, stream)?,
            k_norm: nn::RmsNorm::unloaded(head_dim, args.rms_norm_eps, Dtype::Float32, stream)?,
            rel_proj: Param::<Array>::unloaded(&[args.d_rel, rel_extent], Dtype::Float32, stream)?,
            k_sconv: DepthwiseConv1d::new(
                n_kv_heads * head_dim,
                args.sconv_kernel_size,
                false,
                stream,
            )?,
            v_sconv: DepthwiseConv1d::new(
                n_kv_heads * head_dim,
                args.sconv_kernel_size,
                false,
                stream,
            )?,
        })
    }

    fn repeat_kv(&self, states: &Array, stream: &Stream) -> Result<Array, Exception> {
        if self.n_heads == self.n_kv_heads {
            return Ok(states.clone());
        }
        let shape = states.shape();
        let repeats = self.n_heads / self.n_kv_heads;
        broadcast_to(
            &states.reshape(&[shape[0], self.n_kv_heads, 1, shape[2], shape[3]], stream)?,
            &[shape[0], self.n_kv_heads, repeats, shape[2], shape[3]],
            stream,
        )?
        .reshape(&[shape[0], self.n_heads, shape[2], shape[3]], stream)
    }

    fn forward(
        &mut self,
        hidden: &Array,
        cache: Option<&mut LayerCache>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let batch = hidden.dim(0);
        let seq_len = hidden.dim(1);
        let q_offset = cache.as_ref().map_or(0, |cache| cache.kv.offset());

        let q = self.q_proj.forward(hidden, stream)?;
        let mut k = self.k_proj.forward(hidden, stream)?;
        let mut v = self.v_proj.forward(hidden, stream)?;
        let relative = self.r_proj.forward(hidden, stream)?;

        if let Some(cache) = cache {
            k = short_convolution(&self.k_sconv, &k, Some(&mut cache.convolutions[0]), stream)?;
            v = short_convolution(&self.v_sconv, &v, Some(&mut cache.convolutions[1]), stream)?;
            let q = self
                .q_norm
                .forward(
                    &q.reshape(&[batch, seq_len, self.n_heads, self.head_dim], stream)?,
                    stream,
                )?
                .transpose_axes(&[0, 2, 1, 3], stream)?;
            let k = self
                .k_norm
                .forward(
                    &k.reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim], stream)?,
                    stream,
                )?
                .transpose_axes(&[0, 2, 1, 3], stream)?;
            let v = v
                .reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim], stream)?
                .transpose_axes(&[0, 2, 1, 3], stream)?;
            let (k, v) = cache.kv.update_and_fetch(k, v, stream)?;
            let key_len = k.dim(2);
            let key_offset = q_offset + seq_len - key_len;
            self.attend_chunked(
                q, k, v, &relative, batch, seq_len, q_offset, key_offset, stream,
            )
        } else {
            k = short_convolution(&self.k_sconv, &k, None, stream)?;
            v = short_convolution(&self.v_sconv, &v, None, stream)?;
            let q = self
                .q_norm
                .forward(
                    &q.reshape(&[batch, seq_len, self.n_heads, self.head_dim], stream)?,
                    stream,
                )?
                .transpose_axes(&[0, 2, 1, 3], stream)?;
            let k = self
                .k_norm
                .forward(
                    &k.reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim], stream)?,
                    stream,
                )?
                .transpose_axes(&[0, 2, 1, 3], stream)?;
            let v = v
                .reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim], stream)?
                .transpose_axes(&[0, 2, 1, 3], stream)?;
            self.attend_chunked(q, k, v, &relative, batch, seq_len, 0, 0, stream)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn attend_chunked(
        &mut self,
        q: Array,
        k: Array,
        v: Array,
        relative: &Array,
        batch: i32,
        query_len: i32,
        query_offset: i32,
        key_offset: i32,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        // Bound the eagerly materialized score/bias tensors. Local layers also
        // discard keys outside the earliest query's window for each chunk.
        const TARGET_SCORE_ELEMENTS: i32 = 16 * 1024 * 1024;
        let total_key_len = k.dim(2).max(1);
        let chunk_size = if self.local {
            256
        } else {
            (TARGET_SCORE_ELEMENTS / (self.n_heads * total_key_len)).clamp(1, 256)
        };
        let key_limit = key_offset + k.dim(2);
        let mut outputs = Vec::new();
        let mut start = 0;
        while start < query_len {
            let end = (start + chunk_size).min(query_len);
            let query_abs_start = query_offset + start;
            let query_abs_end = query_offset + end;
            let chunk_key_start = if self.local {
                (query_abs_start - self.sliding_window + 1).max(key_offset)
            } else {
                key_offset
            };
            let chunk_key_end = query_abs_end.min(key_limit);
            let key_start_index = chunk_key_start - key_offset;
            let key_end_index = chunk_key_end - key_offset;
            let mut q_chunk = q.try_index_device((.., .., start..end, ..), stream)?;
            let k_chunk =
                k.try_index_device((.., .., key_start_index..key_end_index, ..), stream)?;
            let v_chunk =
                v.try_index_device((.., .., key_start_index..key_end_index, ..), stream)?;
            let relative_chunk = relative.try_index_device((.., start..end, ..), stream)?;
            let (bias, mask, tau) = self.position_data(
                &relative_chunk,
                batch,
                end - start,
                key_end_index - key_start_index,
                query_abs_start,
                chunk_key_start,
                stream,
            )?;
            if let Some(tau) = tau {
                q_chunk = q_chunk.multiply(tau, stream)?;
            }
            outputs.push(self.attend(
                q_chunk,
                k_chunk,
                v_chunk,
                bias,
                mask,
                batch,
                end - start,
                stream,
            )?);
            start = end;
        }
        concatenate_axis(&outputs, 1, stream)
    }

    #[allow(clippy::too_many_arguments)]
    fn position_data(
        &self,
        relative: &Array,
        batch: i32,
        query_len: i32,
        key_len: i32,
        query_offset: i32,
        key_offset: i32,
        stream: &Stream,
    ) -> Result<(Array, Array, Option<Array>), Exception> {
        let relative = relative.reshape(&[batch, query_len, self.n_heads, self.d_rel], stream)?;
        let profiles = matmul(relative, self.rel_proj.as_ref(), stream)?
            .transpose_axes(&[0, 2, 1, 3], stream)?;
        let q_positions = arange::<i32, i32>(query_offset, query_offset + query_len, 1, stream)?
            .try_index_device((.., NewAxis), stream)?;
        let k_positions = arange::<i32, i32>(key_offset, key_offset + key_len, 1, stream)?
            .try_index_device((NewAxis, ..), stream)?;
        let distances = q_positions.subtract(k_positions, stream)?;
        let mut valid = distances.ge(Array::from_int(0), stream)?;
        if self.local {
            valid = valid.logical_and(
                &distances.lt(Array::from_int(self.sliding_window), stream)?,
                stream,
            )?;
        }
        let gather = clip(&distances, (0, self.rel_extent - 1), stream)?
            .as_dtype(Dtype::Int32, stream)?
            .try_index_device((NewAxis, NewAxis, .., ..), stream)?;
        let gather = broadcast_to(&gather, &[batch, self.n_heads, query_len, key_len], stream)?;
        let mut bias = take_along_axis(&profiles, &gather, -1, stream)?;
        let relative_valid = distances.ge(Array::from_int(0), stream)?.logical_and(
            &distances.lt(Array::from_int(self.rel_extent), stream)?,
            stream,
        )?;
        bias = r#where(&relative_valid, bias, Array::from_f32(0.0), stream)?;
        let tau = if !self.local {
            if let Some(floor) = self.log_scaling_n_floor {
                let positions =
                    arange::<i32, i32>(query_offset + 1, query_offset + query_len + 1, 1, stream)?
                        .as_dtype(Dtype::Float32, stream)?;
                let ratio = positions.divide(Array::from_f32(floor as f32), stream)?;
                let ratio = safemlx::ops::maximum(ratio, Array::from_f32(1.0), stream)?;
                let tau = ratio
                    .log(stream)?
                    .multiply(Array::from_f32(self.log_scaling_alpha), stream)?
                    .add(Array::from_f32(1.0), stream)?
                    .reshape(&[1, 1, query_len, 1], stream)?;
                bias = bias.multiply(&tau, stream)?;
                Some(tau)
            } else {
                None
            }
        } else {
            None
        };
        Ok((bias, valid, tau))
    }

    #[allow(clippy::too_many_arguments)]
    fn attend(
        &mut self,
        q: Array,
        k: Array,
        v: Array,
        bias: Array,
        valid: Array,
        batch: i32,
        seq_len: i32,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let k = self.repeat_kv(&k, stream)?;
        let v = self.repeat_kv(&v, stream)?;
        let mut scores = matmul(
            &q.multiply(Array::from_f32(1.0 / self.head_dim as f32), stream)?,
            &k.swap_axes(-1, -2, stream)?,
            stream,
        )?
        .add(bias, stream)?;
        scores = r#where(&valid, scores, Array::from_f32(f32::NEG_INFINITY), stream)?;
        let probabilities = softmax_axis(scores, -1, true, stream)?;
        let attended = matmul(probabilities, v, stream)?
            .transpose_axes(&[0, 2, 1, 3], stream)?
            .reshape(&[batch, seq_len, self.n_heads * self.head_dim], stream)?;
        self.o_proj.forward(&attended, stream)
    }
}

fn short_convolution(
    convolution: &DepthwiseConv1d,
    input: &Array,
    cache: Option<&mut CausalConv1dCache>,
    stream: &Stream,
) -> Result<Array, Exception> {
    let dtype = input.dtype();
    let input = input.as_dtype(Dtype::Float32, stream)?;
    causal_depthwise_conv1d(convolution, &input, cache, stream)?
        .add(&input, stream)?
        .as_dtype(dtype, stream)
}

#[derive(Debug, Clone, ModuleParameters)]
struct InklingRouter {
    num_routed: i32,
    num_shared: i32,
    top_k: i32,
    route_scale: f32,
    #[param]
    weight: Param<Array>,
    #[param]
    bias: Param<Array>,
    #[param]
    global_scale: Param<Array>,
}

impl InklingRouter {
    fn new(args: &TextArgs, stream: &Stream) -> Result<Self, Exception> {
        Ok(Self {
            num_routed: args.n_routed_experts,
            num_shared: args.n_shared_experts,
            top_k: args.num_experts_per_tok,
            route_scale: args.route_scale,
            weight: Param::<Array>::unloaded(
                &[
                    args.n_routed_experts + args.n_shared_experts,
                    args.hidden_size,
                ],
                Dtype::Float32,
                stream,
            )?,
            bias: Param::<Array>::unloaded(&[args.n_routed_experts], Dtype::Float32, stream)?,
            global_scale: Param::<Array>::unloaded(&[1], Dtype::Float32, stream)?,
        })
    }

    fn forward(&self, hidden: &Array, stream: &Stream) -> Result<(Array, Array, Array), Exception> {
        let flat = hidden.reshape(&[-1, hidden.dim(-1)], stream)?;
        let logits = matmul(&flat, &self.weight.as_ref().transpose(stream)?, stream)?;
        let routed = logits.try_index_device((.., ..self.num_routed), stream)?;
        let shared = logits.try_index_device((.., self.num_routed..), stream)?;
        let choice = sigmoid(&routed, stream)?.add(self.bias.as_ref(), stream)?;
        let indices = argpartition_axis(choice, -self.top_k, -1, stream)?
            .try_index_device((.., -self.top_k..), stream)?;
        let selected_logits = take_along_axis(&routed, &indices, -1, stream)?;
        let all_logits = concatenate_axis(&[selected_logits, shared], -1, stream)?;
        let weights = softmax_axis(nn::log_sigmoid(all_logits, stream)?, -1, true, stream)?
            .multiply(Array::from_f32(self.route_scale), stream)?
            .multiply(self.global_scale.as_ref(), stream)?;
        let routed_weights = weights.try_index_device((.., ..self.top_k), stream)?;
        let shared_weights = weights.try_index_device((.., self.top_k..), stream)?;
        Ok((indices, routed_weights, shared_weights))
    }
}

#[derive(Debug, Clone, ModuleParameters)]
struct InklingMoe {
    #[param]
    router: InklingRouter,
    #[param]
    experts: PackedSwiGluExperts,
    #[param]
    shared_experts: PackedSwiGluExperts,
}

impl InklingMoe {
    fn new(args: &TextArgs, stream: &Stream) -> Result<Self, Exception> {
        let intermediate = args.moe_intermediate_size();
        Ok(Self {
            router: InklingRouter::new(args, stream)?,
            experts: PackedSwiGluExperts::new(
                args.n_routed_experts,
                args.hidden_size,
                intermediate,
                None,
                None,
                stream,
            )?,
            shared_experts: PackedSwiGluExperts::new(
                args.n_shared_experts,
                args.hidden_size,
                intermediate,
                None,
                None,
                stream,
            )?,
        })
    }

    fn forward(&mut self, hidden: &Array, stream: &Stream) -> Result<Array, Exception> {
        let shape = hidden.shape().to_vec();
        let flat = hidden.reshape(&[-1, hidden.dim(-1)], stream)?;
        let (indices, weights, shared_weights) = self.router.forward(hidden, stream)?;
        let routed = self.experts.forward(&flat, &indices, &weights, stream)?;
        let tokens = flat.dim(0);
        let shared_indices = broadcast_to(
            &arange::<i32, i32>(0, self.router.num_shared, 1, stream)?
                .try_index_device((NewAxis, ..), stream)?,
            &[tokens, self.router.num_shared],
            stream,
        )?;
        let shared =
            self.shared_experts
                .forward(&flat, &shared_indices, &shared_weights, stream)?;
        routed.add(shared, stream)?.reshape(&shape, stream)
    }

    fn forward_with_expert_executor<F>(
        &mut self,
        hidden: &Array,
        stream: &Stream,
        execute: F,
    ) -> Result<Array, Exception>
    where
        F: FnOnce(&Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let shape = hidden.shape().to_vec();
        let flat = hidden.reshape(&[-1, hidden.dim(-1)], stream)?;
        let (indices, weights, shared_weights) = self.router.forward(hidden, stream)?;
        let routed = execute(&flat, &indices, &weights, stream)?;
        let tokens = flat.dim(0);
        let shared_indices = broadcast_to(
            &arange::<i32, i32>(0, self.router.num_shared, 1, stream)?
                .try_index_device((NewAxis, ..), stream)?,
            &[tokens, self.router.num_shared],
            stream,
        )?;
        let shared =
            self.shared_experts
                .forward(&flat, &shared_indices, &shared_weights, stream)?;
        routed.add(shared, stream)?.reshape(&shape, stream)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub(crate) struct DecoderLayer {
    #[param]
    input_layernorm: nn::RmsNorm,
    #[param]
    self_attn: InklingAttention,
    #[param]
    attn_sconv: DepthwiseConv1d,
    #[param]
    post_attention_layernorm: nn::RmsNorm,
    #[param]
    dense: Option<SwiGluMlp>,
    #[param]
    dense_global_scale: Param<Option<Array>>,
    #[param]
    moe: Option<InklingMoe>,
    #[param]
    mlp_sconv: DepthwiseConv1d,
}

impl DecoderLayer {
    pub(crate) fn new(args: &TextArgs, layer: i32, stream: &Stream) -> Result<Self, Exception> {
        let dense = args.is_dense(layer);
        Ok(Self {
            input_layernorm: nn::RmsNorm::unloaded(
                args.hidden_size,
                args.rms_norm_eps,
                Dtype::Float32,
                stream,
            )?,
            self_attn: InklingAttention::new(args, layer, stream)?,
            attn_sconv: DepthwiseConv1d::new(
                args.hidden_size,
                args.sconv_kernel_size,
                false,
                stream,
            )?,
            post_attention_layernorm: nn::RmsNorm::unloaded(
                args.hidden_size,
                args.rms_norm_eps,
                Dtype::Float32,
                stream,
            )?,
            dense: if dense {
                Some(SwiGluMlp::unloaded(
                    args.hidden_size,
                    args.dense_intermediate_size(),
                    false,
                    stream,
                )?)
            } else {
                None
            },
            dense_global_scale: if dense {
                Param::<Option<Array>>::unloaded_some(&[1], Dtype::Float32, stream)?
            } else {
                Param::new(None)
            },
            moe: if dense {
                None
            } else {
                Some(InklingMoe::new(args, stream)?)
            },
            mlp_sconv: DepthwiseConv1d::new(
                args.hidden_size,
                args.sconv_kernel_size,
                false,
                stream,
            )?,
        })
    }

    pub(crate) fn forward(
        &mut self,
        hidden: &Array,
        cache: Option<&mut LayerCache>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        match cache {
            Some(cache) => {
                let normalized = self.input_layernorm.forward(hidden, stream)?;
                let attention = self.self_attn.forward(&normalized, Some(cache), stream)?;
                let attention = short_convolution(
                    &self.attn_sconv,
                    &attention,
                    Some(&mut cache.convolutions[2]),
                    stream,
                )?;
                let hidden = hidden.add(attention, stream)?;
                let normalized = self.post_attention_layernorm.forward(&hidden, stream)?;
                let mlp = self.forward_mlp(&normalized, stream)?;
                let mlp = short_convolution(
                    &self.mlp_sconv,
                    &mlp,
                    Some(&mut cache.convolutions[3]),
                    stream,
                )?;
                hidden.add(mlp, stream)
            }
            None => {
                let normalized = self.input_layernorm.forward(hidden, stream)?;
                let attention = self.self_attn.forward(&normalized, None, stream)?;
                let attention = short_convolution(&self.attn_sconv, &attention, None, stream)?;
                let hidden = hidden.add(attention, stream)?;
                let normalized = self.post_attention_layernorm.forward(&hidden, stream)?;
                let mlp = self.forward_mlp(&normalized, stream)?;
                let mlp = short_convolution(&self.mlp_sconv, &mlp, None, stream)?;
                hidden.add(mlp, stream)
            }
        }
    }

    pub(crate) fn forward_with_expert_executor<F>(
        &mut self,
        hidden: &Array,
        cache: Option<&mut LayerCache>,
        stream: &Stream,
        execute: F,
    ) -> Result<Array, Exception>
    where
        F: FnOnce(&Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        match cache {
            Some(cache) => {
                let normalized = self.input_layernorm.forward(hidden, stream)?;
                let attention = self.self_attn.forward(&normalized, Some(cache), stream)?;
                let attention = short_convolution(
                    &self.attn_sconv,
                    &attention,
                    Some(&mut cache.convolutions[2]),
                    stream,
                )?;
                let hidden = hidden.add(attention, stream)?;
                let normalized = self.post_attention_layernorm.forward(&hidden, stream)?;
                let mlp = self.forward_mlp_with_expert_executor(&normalized, stream, execute)?;
                let mlp = short_convolution(
                    &self.mlp_sconv,
                    &mlp,
                    Some(&mut cache.convolutions[3]),
                    stream,
                )?;
                hidden.add(mlp, stream)
            }
            None => {
                let normalized = self.input_layernorm.forward(hidden, stream)?;
                let attention = self.self_attn.forward(&normalized, None, stream)?;
                let attention = short_convolution(&self.attn_sconv, &attention, None, stream)?;
                let hidden = hidden.add(attention, stream)?;
                let normalized = self.post_attention_layernorm.forward(&hidden, stream)?;
                let mlp = self.forward_mlp_with_expert_executor(&normalized, stream, execute)?;
                let mlp = short_convolution(&self.mlp_sconv, &mlp, None, stream)?;
                hidden.add(mlp, stream)
            }
        }
    }

    fn forward_mlp(&mut self, hidden: &Array, stream: &Stream) -> Result<Array, Exception> {
        if let Some(dense) = &mut self.dense {
            let output = dense.forward(hidden, stream)?;
            return match self.dense_global_scale.as_ref() {
                Some(scale) => output.multiply(scale, stream),
                None => Ok(output),
            };
        }
        self.moe
            .as_mut()
            .expect("validated sparse layer")
            .forward(hidden, stream)
    }

    fn forward_mlp_with_expert_executor<F>(
        &mut self,
        hidden: &Array,
        stream: &Stream,
        execute: F,
    ) -> Result<Array, Exception>
    where
        F: FnOnce(&Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        if let Some(dense) = &mut self.dense {
            let output = dense.forward(hidden, stream)?;
            return match self.dense_global_scale.as_ref() {
                Some(scale) => output.multiply(scale, stream),
                None => Ok(output),
            };
        }
        self.moe
            .as_mut()
            .expect("validated sparse layer")
            .forward_with_expert_executor(hidden, stream, execute)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
struct TextModel {
    #[param]
    embed_tokens: nn::Embedding,
    #[param]
    embed_norm: nn::RmsNorm,
    #[param]
    layers: Vec<DecoderLayer>,
    #[param]
    norm: nn::RmsNorm,
}

impl TextModel {
    fn new(args: &TextArgs, stream: &Stream) -> Result<Self, Exception> {
        Ok(Self {
            embed_tokens: nn::Embedding::unloaded(
                args.vocab_size,
                args.hidden_size,
                Dtype::Float32,
                stream,
            )?,
            embed_norm: nn::RmsNorm::unloaded(
                args.hidden_size,
                args.rms_norm_eps,
                Dtype::Float32,
                stream,
            )?,
            layers: (0..args.num_hidden_layers)
                .map(|layer| DecoderLayer::new(args, layer, stream))
                .collect::<Result<Vec<_>, _>>()?,
            norm: nn::RmsNorm::unloaded(
                args.hidden_size,
                args.rms_norm_eps,
                Dtype::Float32,
                stream,
            )?,
        })
    }

    fn embed(&mut self, tokens: &Array, stream: &Stream) -> Result<Array, Exception> {
        let embedded = self.embed_tokens.forward(tokens, stream)?;
        self.embed_norm.forward(&embedded, stream)
    }

    fn forward(
        &mut self,
        tokens: &Array,
        inputs_embeds: Option<&Array>,
        cache: Option<&mut Cache>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let mut hidden = match inputs_embeds {
            Some(embeddings) => embeddings.clone(),
            None => self.embed(tokens, stream)?,
        };
        if let Some(cache) = cache {
            for (layer, cache) in self.layers.iter_mut().zip(cache.layers.iter_mut()) {
                hidden = layer.forward(&hidden, Some(cache), stream)?;
            }
        } else {
            for layer in &mut self.layers {
                hidden = layer.forward(&hidden, None, stream)?;
            }
        }
        self.norm.forward(&hidden, stream)
    }

    fn forward_with_expert_executor<F>(
        &mut self,
        tokens: &Array,
        cache: &mut Cache,
        mut execute: F,
        stream: &Stream,
    ) -> Result<Array, Exception>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let mut hidden = self.embed(tokens, stream)?;
        for (index, (layer, layer_cache)) in self
            .layers
            .iter_mut()
            .zip(cache.layers.iter_mut())
            .enumerate()
        {
            hidden = if layer.moe.is_some() {
                layer.forward_with_expert_executor(
                    &hidden,
                    Some(layer_cache),
                    stream,
                    |flat, ids, weights, stream| execute(index, flat, ids, weights, stream),
                )?
            } else {
                layer.forward(&hidden, Some(layer_cache), stream)?
            };
        }
        self.norm.forward(&hidden, stream)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub(crate) struct AudioModel {
    num_codebooks: i32,
    codebook_size: i32,
    #[param]
    encoder: nn::Embedding,
    #[param]
    final_norm: nn::RmsNorm,
}

impl AudioModel {
    pub(crate) fn new(args: &AudioArgs, stream: &Stream) -> Result<Self, Exception> {
        Ok(Self {
            num_codebooks: args.num_codebooks,
            codebook_size: args.codebook_size,
            encoder: nn::Embedding::unloaded(
                args.num_codebooks * args.codebook_size,
                args.text_hidden_size,
                Dtype::Float32,
                stream,
            )?,
            final_norm: nn::RmsNorm::unloaded(
                args.text_hidden_size,
                args.rms_norm_eps,
                Dtype::Float32,
                stream,
            )?,
        })
    }

    pub(crate) fn forward(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let input_ids = match input_ids.ndim() {
            2 => input_ids.try_index_device(NewAxis, stream)?,
            3 if input_ids.dim(0) == 1 => input_ids.clone(),
            _ => {
                return Err(Exception::custom(format!(
                    "Inkling audio IDs must be [frames, {}] or [1, frames, {}], got {:?}",
                    self.num_codebooks,
                    self.num_codebooks,
                    input_ids.shape()
                )))
            }
        };
        if input_ids.dim(-1) != self.num_codebooks {
            return Err(Exception::custom(format!(
                "Inkling audio IDs require {} dMel codebooks, got {:?}",
                self.num_codebooks,
                input_ids.shape()
            )));
        }
        let offsets = arange::<i32, i32>(
            0,
            self.num_codebooks * self.codebook_size,
            self.codebook_size,
            stream,
        )?
        .reshape(&[1, 1, self.num_codebooks], stream)?;
        let indices = input_ids
            .as_dtype(Dtype::Int32, stream)?
            .add(offsets, stream)?;
        let embedded = self.encoder.forward(&indices, stream)?;
        let mut embedded = sum_axis(&embedded, -2, false, stream)?;
        embedded = self.final_norm.forward(&embedded, stream)?;
        if let Some(mask) = mask {
            if mask.ndim() != 2 || mask.dim(0) != 1 || mask.dim(1) != embedded.dim(1) {
                return Err(Exception::custom(format!(
                    "Inkling audio mask must be [1, frames], got {:?}",
                    mask.shape()
                )));
            }
            let valid = mask.sum(None, stream)?.item::<i32>(stream);
            embedded = embedded.try_index_device((.., ..valid, ..), stream)?;
        }
        Ok(embedded)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub(crate) struct VisionLayer {
    t_fold: i32,
    hw_fold: i32,
    #[param]
    projection: nn::Linear,
    #[param]
    layer_norm: Option<nn::RmsNorm>,
}

impl VisionLayer {
    pub(crate) fn new(
        input_dim: i32,
        output_dim: i32,
        t_fold: i32,
        hw_fold: i32,
        add_norm: bool,
        eps: f32,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        Ok(Self {
            t_fold,
            hw_fold,
            projection: nn::Linear::unloaded(input_dim, output_dim, false, Dtype::Float32, stream)?,
            layer_norm: add_norm
                .then(|| nn::RmsNorm::unloaded(output_dim, eps, Dtype::Float32, stream))
                .transpose()?,
        })
    }

    pub(crate) fn forward(&mut self, hidden: &Array, stream: &Stream) -> Result<Array, Exception> {
        let mut hidden = if self.t_fold > 1 || self.hw_fold > 1 {
            let shape = hidden.shape();
            if shape.len() != 5
                || shape[1] % self.t_fold != 0
                || shape[2] % self.hw_fold != 0
                || shape[3] % self.hw_fold != 0
            {
                return Err(Exception::custom(format!(
                    "Inkling hMLP fold ({}, {}) is incompatible with {:?}",
                    self.t_fold, self.hw_fold, shape
                )));
            }
            let (batch, time, height, width, channels) =
                (shape[0], shape[1], shape[2], shape[3], shape[4]);
            hidden
                .reshape(
                    &[
                        batch,
                        time / self.t_fold,
                        self.t_fold,
                        height / self.hw_fold,
                        self.hw_fold,
                        width / self.hw_fold,
                        self.hw_fold,
                        channels,
                    ],
                    stream,
                )?
                .transpose_axes(&[0, 1, 3, 5, 2, 4, 6, 7], stream)?
                .reshape(
                    &[
                        batch,
                        time / self.t_fold,
                        height / self.hw_fold,
                        width / self.hw_fold,
                        self.t_fold * self.hw_fold * self.hw_fold * channels,
                    ],
                    stream,
                )?
        } else {
            hidden.clone()
        };
        hidden = self.projection.forward(&hidden, stream)?;
        if let Some(norm) = &mut self.layer_norm {
            hidden = norm.forward(&hidden, stream)?;
            hidden = nn::gelu(hidden, stream)?;
        }
        Ok(hidden)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub(crate) struct VisionModel {
    pub(crate) text_hidden_size: i32,
    #[param]
    pub(crate) layers: Vec<VisionLayer>,
    #[param]
    pub(crate) final_norm: nn::RmsNorm,
}

impl VisionModel {
    pub(crate) fn new(args: &VisionArgs, stream: &Stream) -> Result<Self, Error> {
        if (
            args.temporal_patch_size,
            args.patch_size,
            args.num_hidden_layers,
            args.num_channels,
        ) != (2, 40, 4, 3)
        {
            return Err(Error::UnsupportedArchitecture(format!(
                "Inkling hMLP currently supports the released (temporal_patch_size=2, patch_size=40, n_layers=4, channels=3) tower, got ({}, {}, {}, {})",
                args.temporal_patch_size, args.patch_size, args.num_hidden_layers, args.num_channels
            )));
        }
        // `plan_out_scales` for the released tower selects reduction scales
        // [1, 25, 100, 1600, 3200].
        let specs = [
            (75, 128, 1, 5),
            (512, 512, 1, 2),
            (8192, 4800, 1, 4),
            (9600, args.text_hidden_size, 2, 1),
        ];
        let mut layers = Vec::with_capacity(specs.len());
        for (index, (input_dim, output_dim, t_fold, hw_fold)) in specs.into_iter().enumerate() {
            layers.push(VisionLayer::new(
                input_dim,
                output_dim,
                t_fold,
                hw_fold,
                index + 1 != specs.len(),
                args.rms_norm_eps,
                stream,
            )?);
        }
        Ok(Self {
            text_hidden_size: args.text_hidden_size,
            layers,
            final_norm: nn::RmsNorm::unloaded(
                args.text_hidden_size,
                args.rms_norm_eps,
                Dtype::Float32,
                stream,
            )?,
        })
    }

    fn forward(&mut self, pixels: &Array, stream: &Stream) -> Result<Array, Exception> {
        if pixels.ndim() != 5 || pixels.shape()[1..] != [2, 40, 40, 3] {
            return Err(Exception::custom(format!(
                "Inkling image patches must be [patches, 2, 40, 40, 3], got {:?}",
                pixels.shape()
            )));
        }
        let mut hidden = pixels.clone();
        for layer in &mut self.layers {
            hidden = layer.forward(&hidden, stream)?;
        }
        hidden = self.final_norm.forward(&hidden, stream)?;
        hidden
            .reshape(&[-1, self.text_hidden_size], stream)?
            .try_index_device(NewAxis, stream)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
/// Inkling causal language model.
pub struct Model {
    pub args: ModelArgs,
    #[param]
    model: TextModel,
    #[param]
    audio: Option<AudioModel>,
    #[param]
    visual: Option<VisionModel>,
    #[param]
    lm_head: nn::Linear,
}

impl Model {
    pub fn new(args: ModelArgs, stream: &Stream) -> Result<Self, Error> {
        validate_args(&args)?;
        Ok(Self {
            model: TextModel::new(&args.text_config, stream)?,
            audio: args
                .audio_config
                .as_ref()
                .map(|config| AudioModel::new(config, stream))
                .transpose()?,
            visual: args
                .vision_config
                .as_ref()
                .map(|config| VisionModel::new(config, stream))
                .transpose()?,
            lm_head: nn::Linear::unloaded(
                args.text_config.hidden_size,
                args.text_config.vocab_size,
                false,
                Dtype::Float32,
                stream,
            )?,
            args,
        })
    }

    pub fn model_type(&self) -> &str {
        &self.args.model_type
    }

    pub fn new_cache(&self) -> Cache {
        Cache::new(&self.args.text_config)
    }

    pub(crate) fn forward_logits(
        &mut self,
        tokens: &Array,
        inputs_embeds: Option<&Array>,
        cache: Option<&mut Cache>,
        last_token_only: bool,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let mut hidden = self.model.forward(tokens, inputs_embeds, cache, stream)?;
        if last_token_only {
            hidden = hidden.try_index_device((.., -1, ..), stream)?;
        }
        hidden = hidden.divide(
            Array::from_f32(self.args.text_config.logits_mup_width_multiplier),
            stream,
        )?;
        let mut logits = self.lm_head.forward(&hidden, stream)?;
        if let Some(size) = self.args.text_config.unpadded_vocab_size {
            if size < logits.dim(-1) {
                logits = match logits.ndim() {
                    2 => logits.try_index_device((.., ..size), stream)?,
                    3 => logits.try_index_device((.., .., ..size), stream)?,
                    rank => {
                        return Err(Exception::custom(format!(
                            "Inkling logits have unsupported rank {rank}"
                        )))
                    }
                };
            }
        }
        Ok(logits)
    }

    pub(crate) fn forward_cached_expert_parallel<F>(
        &mut self,
        tokens: &Array,
        cache: &mut Cache,
        execute: F,
        stream: &Stream,
    ) -> Result<Array, Exception>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let mut hidden = self
            .model
            .forward_with_expert_executor(tokens, cache, execute, stream)?;
        hidden = hidden.divide(
            Array::from_f32(self.args.text_config.logits_mup_width_multiplier),
            stream,
        )?;
        let mut logits = self.lm_head.forward(&hidden, stream)?;
        if let Some(size) = self.args.text_config.unpadded_vocab_size {
            if size < logits.dim(-1) {
                logits = logits.try_index_device((.., .., ..size), stream)?;
            }
        }
        Ok(logits)
    }

    fn prepare_typed_prefill(
        &mut self,
        input: input::ModelInput<'_>,
        stream: &Stream,
    ) -> Result<input::PreparedPrefill, Exception> {
        let modality_tokens = [
            input::ModalityToken {
                modality: input::Modality::Image,
                token_id: self.args.image_token_id,
            },
            input::ModalityToken {
                modality: input::Modality::Audio,
                token_id: self.args.audio_token_id,
            },
        ];
        let embed_tokens = &mut self.model;
        let audio = &mut self.audio;
        let visual = &mut self.visual;
        input::prepare_decoder_prefill(
            input,
            &modality_tokens,
            self.args.text_config.hidden_size,
            "Inkling",
            stream,
            |tokens, stream| embed_tokens.embed(tokens, stream),
            |part, stream| match (part.modality, part.payload) {
                (_, input::InputPayload::Embeddings(embeddings)) => Ok(vec![embeddings.clone()]),
                (input::Modality::Image, input::InputPayload::Tensor(pixels)) => Ok(vec![visual
                    .as_mut()
                    .ok_or_else(|| {
                        Exception::custom(
                            "Inkling image input requires vision_config and vision weights",
                        )
                    })?
                    .forward(pixels, stream)?]),
                (input::Modality::Audio, input::InputPayload::Tensor(ids)) => Ok(vec![audio
                    .as_mut()
                    .ok_or_else(|| {
                        Exception::custom(
                            "Inkling audio input requires audio_config and audio weights",
                        )
                    })?
                    .forward(ids, part.metadata.audio_mask, stream)?]),
                (modality, input::InputPayload::Tensor(_)) => Err(Exception::custom(format!(
                    "Inkling does not support {} tensor inputs",
                    modality.as_str()
                ))),
                (modality, input::InputPayload::TokenIds(_)) => Err(Exception::custom(format!(
                    "Inkling {} input does not accept token-id payloads",
                    modality.as_str()
                ))),
            },
        )
    }
}

impl CausalLm<Cache> for Model {
    fn prefill_input_logits(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        match self.prepare_typed_prefill(input, stream)? {
            input::PreparedPrefill::Text(tokens) => {
                self.forward_logits(&tokens, None, Some(cache), true, stream)
            }
            input::PreparedPrefill::Embeddings { tokens, embeddings } => {
                self.forward_logits(&tokens, Some(&embeddings), Some(cache), true, stream)
            }
        }
    }

    fn decode_logits(
        &mut self,
        input_tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.forward_logits(input_tokens, None, Some(cache), true, stream)
    }
}

/// Inkling token generation iterator.
pub type Generate<'a, S = crate::sampler::DefaultSampler> =
    common::generation::Generate<'a, Model, Cache, S>;

pub fn load_tokenizer(model_dir: impl AsRef<Path>) -> Result<Tokenizer, Error> {
    Tokenizer::from_file(model_dir.as_ref().join("tokenizer.json")).map_err(Into::into)
}

pub fn get_model_args(model_dir: impl AsRef<Path>) -> Result<ModelArgs, Error> {
    let value: Value =
        serde_json::from_reader(std::fs::File::open(model_dir.as_ref().join("config.json"))?)?;
    validate_model_config_value(&value)?;
    Ok(serde_json::from_value(value)?)
}

pub fn validate_model_config_value(value: &Value) -> Result<(), Error> {
    let args: ModelArgs = serde_json::from_value(value.clone()).map_err(|error| {
        Error::UnsupportedArchitecture(format!("invalid Inkling config: {error}"))
    })?;
    validate_args(&args)
}

fn validate_args(args: &ModelArgs) -> Result<(), Error> {
    let text = &args.text_config;
    if args.model_type != "inkling_mm_model" {
        return Err(Error::UnsupportedArchitecture(format!(
            "expected Inkling model_type inkling_mm_model, got {:?}",
            args.model_type
        )));
    }
    for (name, value) in [
        ("hidden_size", text.hidden_size),
        ("num_hidden_layers", text.num_hidden_layers),
        ("vocab_size", text.vocab_size),
        ("num_attention_heads", text.num_attention_heads),
        ("num_key_value_heads", text.num_key_value_heads),
        ("head_dim", text.head_dim),
        ("d_rel", text.d_rel),
        ("rel_extent", text.rel_extent),
        ("sliding_window_size", text.sliding_window_size),
        ("sconv_kernel_size", text.sconv_kernel_size),
        ("n_routed_experts", text.n_routed_experts),
        ("num_experts_per_tok", text.num_experts_per_tok),
        ("n_shared_experts", text.n_shared_experts),
    ] {
        if value <= 0 {
            return Err(Error::UnsupportedArchitecture(format!(
                "Inkling {name} must be positive, got {value}"
            )));
        }
    }
    if !text.use_sconv
        || !text.use_embed_norm
        || !text.shared_expert_sink
        || !text.use_gate_bias
        || !text.norm_after_topk
        || !text.use_global_scale
        || text.gate_activation != "sigmoid"
        || text.hidden_act != "silu"
        || text.attention_dropout != 0.0
        || text.q_bias
        || text.o_bias
        || text
            .final_logit_softcapping
            .is_some_and(|value| value != 0.0)
    {
        return Err(Error::UnsupportedArchitecture(
            "Inkling config uses an unsupported attention, convolution, routing, or logit variant"
                .into(),
        ));
    }
    if text.num_attention_heads % text.num_key_value_heads != 0
        || text.q_heads(true) % text.kv_heads(true) != 0
        || text.attention_head_dim(true) != text.head_dim
    {
        return Err(Error::UnsupportedArchitecture(
            "Inkling attention head configuration is inconsistent".into(),
        ));
    }
    if text.dense_intermediate_size() <= 0 || text.moe_intermediate_size() <= 0 {
        return Err(Error::UnsupportedArchitecture(
            "Inkling dense and MoE intermediate sizes must be positive".into(),
        ));
    }
    if text.num_experts_per_tok > text.n_routed_experts
        || !(0..=text.num_hidden_layers).contains(&text.dense_mlp_idx)
        || text.local_layer_ids.as_ref().is_some_and(|ids| {
            ids.iter()
                .any(|layer| !(0..text.num_hidden_layers).contains(layer))
        })
        || text
            .layer_types
            .as_ref()
            .is_some_and(|types| types.len() != text.num_hidden_layers as usize)
        || text
            .mlp_layer_types
            .as_ref()
            .is_some_and(|types| types.len() != text.num_hidden_layers as usize)
    {
        return Err(Error::UnsupportedArchitecture(
            "Inkling layer schedule or expert top-k configuration is inconsistent".into(),
        ));
    }
    if let Some(audio) = &args.audio_config {
        if audio.text_hidden_size != text.hidden_size
            || audio.num_codebooks <= 0
            || audio.codebook_size <= 0
            || audio.bias
            || !audio.use_audio_norm
            || audio.audio_mode != "dmel"
        {
            return Err(Error::UnsupportedArchitecture(
                "Inkling audio configuration is inconsistent with the text decoder".into(),
            ));
        }
    }
    if let Some(vision) = &args.vision_config {
        if vision.text_hidden_size != text.hidden_size
            || vision.vision_encoder_type != "hmlp"
            || !vision.use_vision_norm
        {
            return Err(Error::UnsupportedArchitecture(
                "Inkling vision hidden size does not match the text decoder".into(),
            ));
        }
        if (
            vision.temporal_patch_size,
            vision.patch_size,
            vision.num_hidden_layers,
            vision.num_channels,
        ) != (2, 40, 4, 3)
        {
            return Err(Error::UnsupportedArchitecture(
                "Inkling vision configuration is not the released 4-layer 2x40x40 hMLP tower"
                    .into(),
            ));
        }
    }
    Ok(())
}

pub fn load_model(
    model_dir: impl AsRef<Path>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    let args = get_model_args(model_dir)?;
    let mut model = Model::new(args, stream)?;
    let config = StrictLoadConfig::default().allow_unused_prefix("model.mtp.");
    let mut report = StrictLoadReport::default();
    let mut params = model.parameters_mut().flatten();
    for file in safetensors_files(model_dir)? {
        for_each_safetensor_array(file, weights_stream, |key, value| {
            for (key, value) in transform_weight(key, value, stream)? {
                load_array_strict(&mut params, key, value, &config, &mut report);
            }
            Ok(())
        })?;
    }
    drop(params);
    report.finish(&model, &config)?;
    model.copy_to_stream(stream)?;
    Ok(model)
}

pub(crate) fn transform_weight(
    key: String,
    mut value: Array,
    stream: &Stream,
) -> Result<Vec<(String, Array)>, Error> {
    if key.ends_with("_sconv.weight") {
        value = value.as_dtype(Dtype::Float32, stream)?;
    }
    if let Some(suffix) = key.strip_prefix("model.audio.") {
        return Ok(vec![(format!("audio.{suffix}"), value)]);
    }
    if let Some(suffix) = key.strip_prefix("model.visual.") {
        let mut suffix = suffix.to_string();
        for layer in 0..4 {
            suffix = suffix
                .replace(
                    &format!("layers.linear_{layer}.weight"),
                    &format!("layers.{layer}.projection.weight"),
                )
                .replace(
                    &format!("layers.norm_{layer}.weight"),
                    &format!("layers.{layer}.layer_norm.weight"),
                );
        }
        return Ok(vec![(format!("visual.{suffix}"), value)]);
    }
    if !key.starts_with("model.llm.") {
        return Ok(vec![(key, value)]);
    }
    let mut key = key.replacen("model.llm.", "model.", 1);
    key = key
        .replace("model.embed.weight", "model.embed_tokens.weight")
        .replace("model.unembed.weight", "lm_head.weight")
        .replace(".attn_norm.weight", ".input_layernorm.weight")
        .replace(".mlp_norm.weight", ".post_attention_layernorm.weight")
        .replace(".attn.wq_du.weight", ".self_attn.q_proj.weight")
        .replace(".attn.wk_dv.weight", ".self_attn.k_proj.weight")
        .replace(".attn.wv_dv.weight", ".self_attn.v_proj.weight")
        .replace(".attn.wr_du.weight", ".self_attn.r_proj.weight")
        .replace(".attn.wo_ud.weight", ".self_attn.o_proj.weight")
        .replace(".attn.q_norm.weight", ".self_attn.q_norm.weight")
        .replace(".attn.k_norm.weight", ".self_attn.k_norm.weight")
        .replace(".attn.rel_logits_proj.proj", ".self_attn.rel_proj")
        .replace(".attn.k_sconv.weight", ".self_attn.k_sconv.weight")
        .replace(".attn.v_sconv.weight", ".self_attn.v_sconv.weight")
        .replace(".mlp.w2_md.weight", ".dense.down_proj.weight")
        .replace(".mlp.global_scale", ".dense_global_scale")
        .replace(".mlp.gate.weight", ".moe.router.weight")
        .replace(".mlp.gate.bias", ".moe.router.bias")
        .replace(".mlp.gate.global_scale", ".moe.router.global_scale")
        .replace(".mlp.experts.w2_weight", ".moe.experts.down_proj")
        .replace(
            ".mlp.shared_experts.shared_w2_weight",
            ".moe.shared_experts.down_proj",
        );

    if key.ends_with(".mlp.w13_dn.weight") {
        let prefix = key.trim_end_matches(".mlp.w13_dn.weight");
        let (gate, up) = deinterleave_w13(value, stream)?;
        return Ok(vec![
            (format!("{prefix}.dense.gate_proj.weight"), gate),
            (format!("{prefix}.dense.up_proj.weight"), up),
        ]);
    }
    if key.ends_with(".mlp.experts.w13_weight") {
        let prefix = key.trim_end_matches(".mlp.experts.w13_weight");
        let (gate, up) = deinterleave_w13(value, stream)?;
        return Ok(vec![(
            format!("{prefix}.moe.experts.gate_up_proj"),
            concatenate_axis(&[gate, up], -2, stream)?,
        )]);
    }
    if key.ends_with(".mlp.shared_experts.shared_w13_weight") {
        let prefix = key.trim_end_matches(".mlp.shared_experts.shared_w13_weight");
        let (gate, up) = deinterleave_w13(value, stream)?;
        return Ok(vec![(
            format!("{prefix}.moe.shared_experts.gate_up_proj"),
            concatenate_axis(&[gate, up], -2, stream)?,
        )]);
    }
    Ok(vec![(key, value)])
}

fn deinterleave_w13(value: Array, stream: &Stream) -> Result<(Array, Array), Error> {
    let shape = value.shape().to_vec();
    if shape.len() < 2 || shape[shape.len() - 2] % 2 != 0 {
        return Err(Error::UnsupportedArchitecture(format!(
            "Inkling w13 tensor has invalid shape {shape:?}"
        )));
    }
    let rows = shape[shape.len() - 2] / 2;
    let hidden = shape[shape.len() - 1];
    let (reshaped, gate_index, up_index) = if shape.len() == 2 {
        (
            value.reshape(&[rows, 2, hidden], stream)?,
            (.., 0, ..),
            (.., 1, ..),
        )
    } else if shape.len() == 3 {
        let experts = shape[0];
        let reshaped = value.reshape(&[experts, rows, 2, hidden], stream)?;
        let gate = reshaped.try_index_device((.., .., 0, ..), stream)?;
        let up = reshaped.try_index_device((.., .., 1, ..), stream)?;
        return Ok((gate, up));
    } else {
        return Err(Error::UnsupportedArchitecture(format!(
            "Inkling w13 tensor rank {} is unsupported",
            shape.len()
        )));
    };
    Ok((
        reshaped.try_index_device(gate_index, stream)?,
        reshaped.try_index_device(up_index, stream)?,
    ))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    #[test]
    fn released_config_shape_is_accepted() {
        let config = json!({
            "model_type":"inkling_mm_model",
            "eos_token_id":200006,
            "text_config":{
                "hidden_size":32,"num_hidden_layers":3,"vocab_size":64,
                "num_attention_heads":4,"num_key_value_heads":2,"head_dim":8,
                "swa_num_attention_heads":4,"swa_num_key_value_heads":2,"swa_head_dim":8,
                "sliding_window_size":8,"local_layer_ids":[0,1],"dense_mlp_idx":1,
                "sconv_kernel_size":4,"d_rel":4,"rel_extent":16,
                "intermediate_size":24,"dense_intermediate_size":48,
                "n_routed_experts":4,"num_experts_per_tok":2,"n_shared_experts":1,
                "route_scale":8.0,"use_sconv":true,"use_embed_norm":true,
                "shared_expert_sink":true,"use_gate_bias":true,"norm_after_topk":true,
                "use_global_scale":true,"gate_activation":"sigmoid"
            }
        });
        super::validate_model_config_value(&config).unwrap();
        let support = crate::models::check_model_config(&config);
        let crate::models::ModelConfigSupport::Supported(support) = support else {
            panic!("released Inkling metadata did not dispatch")
        };
        assert_eq!(support.kind, crate::models::ModelKind::Inkling);
        assert_eq!(support.effective_model_type, "inkling_mm_model");
    }

    #[test]
    fn cache_schedule_matches_local_and_global_layers() {
        let config = json!({
            "model_type":"inkling_mm_model",
            "image_token_id":200054,
            "audio_token_id":200053,
            "text_config":{
                "hidden_size":32,"num_hidden_layers":3,"vocab_size":64,
                "num_attention_heads":4,"num_key_value_heads":2,"head_dim":8,
                "swa_num_attention_heads":4,"swa_num_key_value_heads":2,"swa_head_dim":8,
                "sliding_window_size":8,"local_layer_ids":[0,1],"dense_mlp_idx":1,
                "sconv_kernel_size":4,"d_rel":4,"rel_extent":16,
                "intermediate_size":24,"dense_intermediate_size":48,
                "n_routed_experts":4,"num_experts_per_tok":2,"n_shared_experts":1,
                "route_scale":8.0,"use_sconv":true,"use_embed_norm":true,
                "shared_expert_sink":true,"use_gate_bias":true,"norm_after_topk":true,
                "use_global_scale":true,"gate_activation":"sigmoid"
            },
            "audio_config":{
                "decoder_dmodel":32,"n_mel_bins":80,"mel_vocab_size":16
            },
            "vision_config":{
                "decoder_dmodel":32,"patch_size":40,"temporal_patch_size":2,
                "n_channels":3,"n_layers":4
            }
        });
        let args: super::ModelArgs = serde_json::from_value(config).unwrap();
        super::validate_args(&args).unwrap();
        let cache = super::Cache::new(&args.text_config);
        assert_eq!(cache.layers.len(), 3);
        assert!(matches!(
            cache.layers[0].kv,
            super::InklingKvCache::Sliding(_)
        ));
        assert!(matches!(
            cache.layers[2].kv,
            super::InklingKvCache::Global(_)
        ));
    }
}
