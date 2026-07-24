//! Shared Qwen vision-language encoder building blocks.

use safemlx::{
    error::Exception,
    macros::ModuleParameters,
    module::{Module, Param},
    nn,
    ops::{
        concatenate_axis,
        indexing::{NewAxis, TryIndexOp},
        matmul,
    },
    Array, Dtype, Stream,
};
use serde::Deserialize;

use crate::{cache::ConcatKeyValueCache, models::common::layers::silu};

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
/// Qwen VL vision encoder configuration.
pub struct VisionConfig {
    #[serde(default = "default_vision_depth")]
    /// Number of vision transformer blocks.
    pub depth: i32,
    #[serde(default = "default_vision_hidden_size")]
    /// Vision transformer hidden size.
    pub hidden_size: i32,
    #[serde(default = "default_vision_hidden_act")]
    /// Vision MLP activation function.
    pub hidden_act: String,
    #[serde(default = "default_vision_intermediate_size")]
    /// Vision MLP intermediate size.
    pub intermediate_size: i32,
    #[serde(default = "default_vision_num_heads")]
    /// Number of vision attention heads.
    pub num_heads: i32,
    #[serde(default = "default_vision_num_position_embeddings")]
    /// Number of learned spatial position embeddings.
    pub num_position_embeddings: i32,
    #[serde(default = "default_vision_in_channels")]
    /// Number of input pixel channels.
    pub in_channels: i32,
    #[serde(default = "default_vision_patch_size")]
    /// Spatial patch size.
    pub patch_size: i32,
    #[serde(default = "default_vision_spatial_merge_size")]
    /// Spatial merge factor used before language-model insertion.
    pub spatial_merge_size: i32,
    #[serde(default = "default_vision_temporal_patch_size")]
    /// Temporal patch size.
    pub temporal_patch_size: i32,
    #[serde(default = "default_vision_window_size")]
    /// Window attention size from the public config.
    pub window_size: i32,
    #[serde(default = "default_vision_out_hidden_size")]
    /// Output hidden size projected into the language model space.
    pub out_hidden_size: i32,
    #[serde(default = "default_vision_fullatt_block_indexes")]
    /// Blocks configured for full attention in the reference implementation.
    pub fullatt_block_indexes: Vec<i32>,
    #[serde(default)]
    /// Vision layers whose merged features are injected into early decoder layers.
    pub deepstack_visual_indexes: Vec<i32>,
}

fn default_vision_depth() -> i32 {
    32
}
fn default_vision_hidden_size() -> i32 {
    3584
}
fn default_vision_hidden_act() -> String {
    "silu".to_string()
}
fn default_vision_intermediate_size() -> i32 {
    3420
}
fn default_vision_num_heads() -> i32 {
    16
}
fn default_vision_num_position_embeddings() -> i32 {
    2304
}
fn default_vision_in_channels() -> i32 {
    3
}
fn default_vision_patch_size() -> i32 {
    14
}
fn default_vision_spatial_merge_size() -> i32 {
    2
}
fn default_vision_temporal_patch_size() -> i32 {
    2
}
fn default_vision_window_size() -> i32 {
    112
}
fn default_vision_out_hidden_size() -> i32 {
    3584
}
fn default_vision_fullatt_block_indexes() -> Vec<i32> {
    vec![7, 15, 23, 31]
}

#[derive(Debug, Clone, ModuleParameters)]
/// Layer normalization used by Qwen vision encoders.
///
/// The public type name is retained for compatibility with the original
/// Qwen3.5 vision implementation this shared module was extracted from.
pub struct QwenVisionRmsNorm {
    #[param]
    /// Learned scale.
    pub weight: Param<Array>,
    #[param]
    /// Learned bias.
    pub bias: Param<Array>,
    /// Numerical epsilon.
    pub eps: f32,
}

impl QwenVisionRmsNorm {
    fn new(dim: i32, eps: f32, stream: &Stream) -> Result<Self, Exception> {
        Ok(Self {
            weight: Param::<Array>::unloaded(&[dim], Dtype::Float32, stream)?,
            bias: Param::<Array>::unloaded(&[dim], Dtype::Float32, stream)?,
            eps,
        })
    }

    fn forward(&self, x: &Array, stream: &Stream) -> Result<Array, Exception> {
        let mean = safemlx::ops::mean_axis(x, -1, true, stream)?;
        let centered = x.subtract(mean, stream)?;
        let variance = safemlx::ops::mean_axis(&centered.square(stream)?, -1, true, stream)?;
        let normalized = centered.multiply(
            safemlx::ops::rsqrt(variance.add(Array::from_f32(self.eps), stream)?, stream)?,
            stream,
        )?;
        normalized
            .multiply(&*self.weight, stream)?
            .add(&*self.bias, stream)
    }

    fn training_mode(&mut self, _mode: bool) {}
}

#[derive(Debug, Clone, ModuleParameters)]
/// Conv3d patch-projection weight for Qwen vision inputs.
pub struct QwenVisionPatchProjection {
    #[param]
    /// Projection weight shaped `[hidden, channels, temporal, height, width]`.
    pub weight: Param<Array>,
    #[param]
    /// Projection bias.
    pub bias: Param<Array>,
}

impl QwenVisionPatchProjection {
    fn new(config: &VisionConfig, stream: &Stream) -> Result<Self, Exception> {
        Ok(Self {
            weight: Param::<Array>::unloaded(
                &[
                    config.hidden_size,
                    config.in_channels,
                    config.temporal_patch_size,
                    config.patch_size,
                    config.patch_size,
                ],
                Dtype::Float32,
                stream,
            )?,
            bias: Param::<Array>::unloaded(&[config.hidden_size], Dtype::Float32, stream)?,
        })
    }

    fn training_mode(&mut self, _mode: bool) {}
}

#[derive(Debug, Clone, ModuleParameters)]
/// Patch embedding layer for preprocessed Qwen vision tensors.
pub struct QwenVisionPatchEmbed {
    /// Input channels.
    pub in_channels: i32,
    /// Temporal patch size.
    pub temporal_patch_size: i32,
    /// Spatial patch size.
    pub patch_size: i32,
    /// Output embedding dimension.
    pub embed_dim: i32,
    #[param]
    /// Conv3d projection represented as a flattened matrix multiply.
    pub proj: QwenVisionPatchProjection,
}

impl QwenVisionPatchEmbed {
    fn new(config: &VisionConfig, stream: &Stream) -> Result<Self, Exception> {
        Ok(Self {
            in_channels: config.in_channels,
            temporal_patch_size: config.temporal_patch_size,
            patch_size: config.patch_size,
            embed_dim: config.hidden_size,
            proj: QwenVisionPatchProjection::new(config, stream)?,
        })
    }

    fn input_dim(&self) -> i32 {
        self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size
    }

    fn forward(&self, pixel_values: &Array, stream: &Stream) -> Result<Array, Exception> {
        let shape = pixel_values.shape();
        if shape.len() != 2 || shape[1] != self.input_dim() {
            return Err(Exception::custom(format!(
                "Qwen VL image tensor must be shaped [patches, {}], got {shape:?}",
                self.input_dim()
            )));
        }
        let weight = self
            .proj
            .weight
            .as_ref()
            .reshape(&[self.embed_dim, self.input_dim()], stream)?;
        let output = matmul(pixel_values, weight.transpose(stream)?, stream)?;
        output.add(&*self.proj.bias, stream)
    }

    pub(crate) fn training_mode(&mut self, mode: bool) {
        self.proj.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters)]
/// Feed-forward block used by Qwen vision transformer layers.
pub struct QwenVisionMlp {
    /// Activation function name.
    pub hidden_act: String,
    #[param]
    /// First projection.
    pub linear_fc1: nn::Linear,
    #[param]
    /// Second projection.
    pub linear_fc2: nn::Linear,
}

impl QwenVisionMlp {
    fn new(config: &VisionConfig, stream: &Stream) -> Result<Self, Exception> {
        Ok(Self {
            hidden_act: config.hidden_act.clone(),
            linear_fc1: nn::Linear::unloaded(
                config.hidden_size,
                config.intermediate_size,
                true,
                Dtype::Float32,
                stream,
            )?,
            linear_fc2: nn::Linear::unloaded(
                config.intermediate_size,
                config.hidden_size,
                true,
                Dtype::Float32,
                stream,
            )?,
        })
    }

    fn activate(hidden_act: &str, x: Array, stream: &Stream) -> Result<Array, Exception> {
        match hidden_act {
            "silu" => silu(x, stream),
            "gelu" => nn::gelu(x, stream),
            "gelu_pytorch_tanh" => nn::gelu_approximate(x, stream),
            other => Err(Exception::custom(format!(
                "Qwen VL vision MLP activation '{other}' is not supported"
            ))),
        }
    }

    fn forward(&mut self, hidden_states: &Array, stream: &Stream) -> Result<Array, Exception> {
        let hidden_act = self.hidden_act.clone();
        let hidden = Self::activate(
            hidden_act.as_str(),
            self.linear_fc1.forward(hidden_states, stream)?,
            stream,
        )?;
        self.linear_fc2.forward(&hidden, stream)
    }

    fn training_mode(&mut self, mode: bool) {
        self.linear_fc1.training_mode(mode);
        self.linear_fc2.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters)]
/// Full attention used inside the Qwen vision encoder.
pub struct QwenVisionAttention {
    /// Number of attention heads.
    pub num_heads: i32,
    /// Per-head dimension.
    pub head_dim: i32,
    /// Attention scale.
    pub scale: f32,
    #[param]
    /// Packed query/key/value projection.
    pub qkv: nn::Linear,
    #[param]
    /// Output projection.
    pub proj: nn::Linear,
}

impl QwenVisionAttention {
    fn new(config: &VisionConfig, stream: &Stream) -> Result<Self, Exception> {
        if config.hidden_size % config.num_heads != 0 {
            return Err(Exception::custom(format!(
                "Qwen VL vision hidden_size {} is not divisible by num_heads {}",
                config.hidden_size, config.num_heads
            )));
        }
        let head_dim = config.hidden_size / config.num_heads;
        Ok(Self {
            num_heads: config.num_heads,
            head_dim,
            scale: (head_dim as f32).sqrt().recip(),
            qkv: nn::Linear::unloaded(
                config.hidden_size,
                config.hidden_size * 3,
                true,
                Dtype::Float32,
                stream,
            )?,
            proj: nn::Linear::unloaded(
                config.hidden_size,
                config.hidden_size,
                true,
                Dtype::Float32,
                stream,
            )?,
        })
    }

    fn forward(
        &mut self,
        hidden_states: &Array,
        chunk_lengths: &[i32],
        cos: &Array,
        sin: &Array,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let seq_len = hidden_states.dim(0);
        let qkv = self
            .qkv
            .forward(hidden_states, stream)?
            .reshape(&[seq_len, 3, self.num_heads, self.head_dim], stream)?;
        let mut query = qkv.try_index_device((.., 0, .., ..), stream)?;
        let mut key = qkv.try_index_device((.., 1, .., ..), stream)?;
        let value = qkv.try_index_device((.., 2, .., ..), stream)?;
        (query, key) = apply_vision_rotary_pos_emb(query, key, cos, sin, stream)?;

        let mut outputs = Vec::with_capacity(chunk_lengths.len());
        let mut start = 0;
        for &len in chunk_lengths {
            let end = start + len;
            let q = query
                .try_index_device((start..end, .., ..), stream)?
                .transpose_axes(&[1, 0, 2], stream)?
                .try_index_device((NewAxis, .., .., ..), stream)?;
            let k = key
                .try_index_device((start..end, .., ..), stream)?
                .transpose_axes(&[1, 0, 2], stream)?
                .try_index_device((NewAxis, .., .., ..), stream)?;
            let v = value
                .try_index_device((start..end, .., ..), stream)?
                .transpose_axes(&[1, 0, 2], stream)?
                .try_index_device((NewAxis, .., .., ..), stream)?;
            let out = crate::utils::scaled_dot_product_attention(
                q,
                k,
                v,
                Option::<ConcatKeyValueCache>::None,
                self.scale,
                None,
                stream,
            )?
            .try_index_device((0, .., .., ..), stream)?
            .transpose_axes(&[1, 0, 2], stream)?
            .reshape(&[len, self.num_heads * self.head_dim], stream)?;
            outputs.push(out);
            start = end;
        }
        let out = concatenate_axis(&outputs, 0, stream)?;
        self.proj.forward(&out, stream)
    }

    fn training_mode(&mut self, mode: bool) {
        self.qkv.training_mode(mode);
        self.proj.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters)]
/// Transformer block used by the Qwen vision encoder.
pub struct QwenVisionBlock {
    #[param]
    /// First layer normalization.
    pub norm1: QwenVisionRmsNorm,
    #[param]
    /// Attention module.
    pub attn: QwenVisionAttention,
    #[param]
    /// Second layer normalization.
    pub norm2: QwenVisionRmsNorm,
    #[param]
    /// Feed-forward module.
    pub mlp: QwenVisionMlp,
}

impl QwenVisionBlock {
    pub(crate) fn new(config: &VisionConfig, stream: &Stream) -> Result<Self, Exception> {
        Ok(Self {
            norm1: QwenVisionRmsNorm::new(config.hidden_size, 1e-6, stream)?,
            attn: QwenVisionAttention::new(config, stream)?,
            norm2: QwenVisionRmsNorm::new(config.hidden_size, 1e-6, stream)?,
            mlp: QwenVisionMlp::new(config, stream)?,
        })
    }

    fn forward(
        &mut self,
        hidden_states: Array,
        chunk_lengths: &[i32],
        cos: &Array,
        sin: &Array,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let normed = self.norm1.forward(&hidden_states, stream)?;
        let attn = self
            .attn
            .forward(&normed, chunk_lengths, cos, sin, stream)?;
        let hidden_states = hidden_states.add(attn, stream)?;
        let normed = self.norm2.forward(&hidden_states, stream)?;
        let mlp = self.mlp.forward(&normed, stream)?;
        hidden_states.add(mlp, stream)
    }

    fn training_mode(&mut self, mode: bool) {
        self.norm1.training_mode(mode);
        self.attn.training_mode(mode);
        self.norm2.training_mode(mode);
        self.mlp.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters)]
/// Patch merger that maps vision features to language-model embeddings.
pub struct QwenVisionPatchMerger {
    /// Number of patch tokens merged per output token.
    pub spatial_merge_unit: i32,
    /// Vision hidden size.
    pub context_dim: i32,
    /// Flattened merger hidden size.
    pub hidden_size: i32,
    /// Whether normalization happens after spatial patches are flattened.
    pub use_postshuffle_norm: bool,
    /// Whether the merger uses tanh-approximated GELU.
    pub approximate_gelu: bool,
    #[param]
    /// Pre-merge layer normalization.
    pub norm: QwenVisionRmsNorm,
    #[param]
    /// First merger projection.
    pub linear_fc1: nn::Linear,
    #[param]
    /// Final projection into language hidden size.
    pub linear_fc2: nn::Linear,
}

impl QwenVisionPatchMerger {
    fn new(
        config: &VisionConfig,
        use_postshuffle_norm: bool,
        approximate_gelu: bool,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let spatial_merge_unit = config.spatial_merge_size * config.spatial_merge_size;
        let hidden_size = config.hidden_size * spatial_merge_unit;
        Ok(Self {
            spatial_merge_unit,
            context_dim: config.hidden_size,
            hidden_size,
            use_postshuffle_norm,
            approximate_gelu,
            norm: QwenVisionRmsNorm::new(
                if use_postshuffle_norm {
                    hidden_size
                } else {
                    config.hidden_size
                },
                1e-6,
                stream,
            )?,
            linear_fc1: nn::Linear::unloaded(
                hidden_size,
                hidden_size,
                true,
                Dtype::Float32,
                stream,
            )?,
            linear_fc2: nn::Linear::unloaded(
                hidden_size,
                config.out_hidden_size,
                true,
                Dtype::Float32,
                stream,
            )?,
        })
    }

    fn forward(&mut self, hidden_states: &Array, stream: &Stream) -> Result<Array, Exception> {
        let seq_len = hidden_states.dim(0);
        if seq_len % self.spatial_merge_unit != 0 {
            return Err(Exception::custom(format!(
                "Qwen VL vision sequence length {seq_len} is not divisible by spatial merge unit {}",
                self.spatial_merge_unit
            )));
        }
        let hidden_states = if self.use_postshuffle_norm {
            let hidden_states = hidden_states.reshape(&[-1, self.hidden_size], stream)?;
            self.norm.forward(&hidden_states, stream)?
        } else {
            let hidden_states = self.norm.forward(hidden_states, stream)?;
            hidden_states.reshape(&[-1, self.hidden_size], stream)?
        };
        let hidden_states = self.linear_fc1.forward(&hidden_states, stream)?;
        let hidden_states = if self.approximate_gelu {
            nn::gelu_approximate(hidden_states, stream)?
        } else {
            nn::gelu(hidden_states, stream)?
        };
        self.linear_fc2.forward(&hidden_states, stream)
    }

    fn training_mode(&mut self, mode: bool) {
        self.norm.training_mode(mode);
        self.linear_fc1.training_mode(mode);
        self.linear_fc2.training_mode(mode);
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum VisionMode {
    Windowed,
    Deepstack,
}

/// Encoded Qwen vision features, including optional DeepStack projections.
pub(crate) struct VisionOutput {
    pub embeddings: Array,
    pub deepstack_features: Vec<Array>,
}

#[derive(Debug, Clone, ModuleParameters)]
/// Pinned Qwen3-VL vision modules surrounding the windowed transformer blocks.
pub(crate) struct QwenVisionLayerwiseStatic {
    pub(crate) config: VisionConfig,
    #[param]
    pub(crate) pos_embed: nn::Embedding,
    #[param]
    pub(crate) patch_embed: QwenVisionPatchEmbed,
    #[param]
    pub(crate) merger: QwenVisionPatchMerger,
    #[param]
    pub(crate) deepstack_merger_list: Vec<QwenVisionPatchMerger>,
}

/// Per-input rotary, window, and DeepStack state for layerwise vision execution.
pub(crate) struct QwenVisionLayerwiseState {
    full_chunk_lengths: Vec<i32>,
    window_chunk_lengths: Vec<i32>,
    window_index: Vec<i32>,
    cos: Array,
    sin: Array,
    deepstack_features: Vec<Array>,
}

impl QwenVisionLayerwiseState {
    pub(crate) fn retained_arrays(&self) -> Vec<&Array> {
        self.deepstack_features.iter().collect()
    }
}

impl QwenVisionLayerwiseStatic {
    pub(crate) fn from_transformer(transformer: QwenVisionTransformer) -> Self {
        Self {
            config: transformer.config,
            pos_embed: transformer.pos_embed,
            patch_embed: transformer.patch_embed,
            merger: transformer.merger,
            deepstack_merger_list: transformer.deepstack_merger_list,
        }
    }

    pub(crate) fn begin(
        &mut self,
        pixel_values: &Array,
        grid_thw: &Array,
        stream: &Stream,
    ) -> Result<(Array, QwenVisionLayerwiseState), Exception> {
        let grid = grid_thw_from_array(grid_thw, stream)?;
        validate_vision_grid(&grid, self.config.spatial_merge_size, pixel_values)?;
        let mut hidden = self.patch_embed.forward(pixel_values, stream)?;
        let seq_len = hidden.dim(0);
        let positions = vision_interpolated_position_embeddings(
            &mut self.pos_embed,
            &grid,
            self.config.num_position_embeddings,
            self.config.spatial_merge_size,
            stream,
        )?;
        hidden = hidden.add(positions.as_dtype(hidden.dtype(), stream)?, stream)?;
        let full_chunk_lengths = vision_attention_chunk_lengths(&grid);
        let total: i32 = full_chunk_lengths.iter().sum();
        if total != seq_len {
            return Err(Exception::custom(format!(
                "Qwen VL vision grid describes {total} patches but image tensor has {seq_len}"
            )));
        }
        let merge_unit = self.config.spatial_merge_size * self.config.spatial_merge_size;
        let window_index = (0..seq_len / merge_unit).collect::<Vec<_>>();
        let window_chunk_lengths = full_chunk_lengths.clone();
        let window_index_array = Array::from_slice(&window_index, &[window_index.len() as i32]);
        hidden = hidden.reshape(&[seq_len / merge_unit, merge_unit, -1], stream)?;
        hidden = hidden.try_index_device((&window_index_array, .., ..), stream)?;
        hidden = hidden.reshape(&[seq_len, -1], stream)?;

        let (cos, sin) = vision_rotary_embeddings(
            &grid,
            self.config.spatial_merge_size,
            self.config.hidden_size / self.config.num_heads,
        );
        let reorder = |array: Array| -> Result<Array, Exception> {
            array
                .reshape(&[seq_len / merge_unit, merge_unit, -1], stream)?
                .try_index_device((&window_index_array, .., ..), stream)?
                .reshape(&[seq_len, -1], stream)
        };
        Ok((
            hidden,
            QwenVisionLayerwiseState {
                full_chunk_lengths,
                window_chunk_lengths,
                window_index,
                cos: reorder(cos)?,
                sin: reorder(sin)?,
                deepstack_features: Vec::new(),
            },
        ))
    }

    pub(crate) fn forward_block(
        &self,
        block: &mut QwenVisionBlock,
        index: usize,
        hidden: Array,
        state: &QwenVisionLayerwiseState,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let chunks = if self.config.fullatt_block_indexes.contains(&(index as i32)) {
            &state.full_chunk_lengths
        } else {
            &state.window_chunk_lengths
        };
        block.forward(hidden, chunks, &state.cos, &state.sin, stream)
    }

    pub(crate) fn capture_deepstack(
        &mut self,
        index: usize,
        hidden: &Array,
        state: &mut QwenVisionLayerwiseState,
        stream: &Stream,
    ) -> Result<(), Exception> {
        if let Some(merger_index) = self
            .config
            .deepstack_visual_indexes
            .iter()
            .position(|&layer| layer == index as i32)
        {
            state.deepstack_features.push(
                self.deepstack_merger_list[merger_index]
                    .forward(hidden, stream)?
                    .try_index_device((NewAxis, .., ..), stream)?,
            );
        }
        Ok(())
    }

    pub(crate) fn finish(
        &mut self,
        hidden: &Array,
        state: &mut QwenVisionLayerwiseState,
        stream: &Stream,
    ) -> Result<VisionOutput, Exception> {
        let hidden = self.merger.forward(hidden, stream)?;
        let reverse_index = reverse_permutation(&state.window_index);
        let reverse_index_array = Array::from_slice(&reverse_index, &[reverse_index.len() as i32]);
        let embeddings = hidden
            .try_index_device((&reverse_index_array, ..), stream)?
            .try_index_device((NewAxis, .., ..), stream)?;
        Ok(VisionOutput {
            embeddings,
            deepstack_features: std::mem::take(&mut state.deepstack_features),
        })
    }
}

#[derive(Debug, Clone, ModuleParameters)]
/// Qwen VL vision transformer used to encode image tensors.
pub struct QwenVisionTransformer {
    /// Vision configuration.
    pub config: VisionConfig,
    #[param]
    /// Learned spatial position embedding table.
    pub pos_embed: nn::Embedding,
    #[param]
    /// Patch embedding.
    pub patch_embed: QwenVisionPatchEmbed,
    #[param]
    /// Vision transformer blocks.
    pub blocks: Vec<QwenVisionBlock>,
    #[param]
    /// Patch merger.
    pub merger: QwenVisionPatchMerger,
    #[param]
    /// Per-layer DeepStack mergers used by Qwen3-VL.
    pub deepstack_merger_list: Vec<QwenVisionPatchMerger>,
    mode: VisionMode,
}

impl QwenVisionTransformer {
    pub(crate) fn new(config: VisionConfig, stream: &Stream) -> Result<Self, Exception> {
        Self::new_with_mode(config, VisionMode::Windowed, stream)
    }

    pub(crate) fn new_deepstack(config: VisionConfig, stream: &Stream) -> Result<Self, Exception> {
        Self::new_with_mode(config, VisionMode::Deepstack, stream)
    }

    fn new_with_mode(
        config: VisionConfig,
        mode: VisionMode,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        if config.spatial_merge_size <= 0 {
            return Err(Exception::custom(
                "Qwen VL vision spatial_merge_size must be positive",
            ));
        }
        let pos_embed = nn::Embedding::unloaded(
            config.num_position_embeddings,
            config.hidden_size,
            Dtype::Float32,
            stream,
        )?;
        let patch_embed = QwenVisionPatchEmbed::new(&config, stream)?;
        let mut blocks = Vec::with_capacity(config.depth as usize);
        for _ in 0..config.depth {
            blocks.push(QwenVisionBlock::new(&config, stream)?);
        }
        let merger =
            QwenVisionPatchMerger::new(&config, false, mode == VisionMode::Windowed, stream)?;
        let deepstack_merger_list = config
            .deepstack_visual_indexes
            .iter()
            .map(|_| QwenVisionPatchMerger::new(&config, true, false, stream))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            config,
            pos_embed,
            patch_embed,
            blocks,
            merger,
            deepstack_merger_list,
            mode,
        })
    }

    pub(crate) fn forward(
        &mut self,
        pixel_values: &Array,
        grid_thw: &Array,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        Ok(self
            .forward_features(pixel_values, grid_thw, stream)?
            .embeddings)
    }

    pub(crate) fn forward_features(
        &mut self,
        pixel_values: &Array,
        grid_thw: &Array,
        stream: &Stream,
    ) -> Result<VisionOutput, Exception> {
        let grid = grid_thw_from_array(grid_thw, stream)?;
        validate_vision_grid(&grid, self.config.spatial_merge_size, pixel_values)?;
        let mut hidden_states = self.patch_embed.forward(pixel_values, stream)?;
        let seq_len = hidden_states.dim(0);
        let position_embeddings = match self.mode {
            VisionMode::Windowed => {
                let indices = vision_position_indices(&grid, self.config.num_position_embeddings)?;
                self.pos_embed.forward(
                    &Array::from_slice(&indices, &[indices.len() as i32]),
                    stream,
                )?
            }
            VisionMode::Deepstack => vision_interpolated_position_embeddings(
                &mut self.pos_embed,
                &grid,
                self.config.num_position_embeddings,
                self.config.spatial_merge_size,
                stream,
            )?,
        };
        hidden_states = hidden_states.add(
            position_embeddings.as_dtype(hidden_states.dtype(), stream)?,
            stream,
        )?;
        let full_chunk_lengths = vision_attention_chunk_lengths(&grid);
        let total: i32 = full_chunk_lengths.iter().sum();
        if total != seq_len {
            return Err(Exception::custom(format!(
                "Qwen VL vision grid describes {total} patches but image tensor has {seq_len}"
            )));
        }
        let (window_index, window_chunk_lengths) = match self.mode {
            VisionMode::Windowed => vision_window_index(
                &grid,
                self.config.spatial_merge_size,
                self.config.window_size,
                self.config.patch_size,
            )?,
            VisionMode::Deepstack => (
                (0..seq_len / (self.config.spatial_merge_size * self.config.spatial_merge_size))
                    .collect(),
                full_chunk_lengths.clone(),
            ),
        };
        let window_index_array = Array::from_slice(&window_index, &[window_index.len() as i32]);
        hidden_states = hidden_states.reshape(
            &[
                seq_len / (self.config.spatial_merge_size * self.config.spatial_merge_size),
                self.config.spatial_merge_size * self.config.spatial_merge_size,
                -1,
            ],
            stream,
        )?;
        hidden_states = hidden_states.try_index_device((&window_index_array, .., ..), stream)?;
        hidden_states = hidden_states.reshape(&[seq_len, -1], stream)?;

        let (cos, sin) = vision_rotary_embeddings(
            &grid,
            self.config.spatial_merge_size,
            self.config.hidden_size / self.config.num_heads,
        );
        let cos = cos.reshape(
            &[
                seq_len / (self.config.spatial_merge_size * self.config.spatial_merge_size),
                self.config.spatial_merge_size * self.config.spatial_merge_size,
                -1,
            ],
            stream,
        )?;
        let cos = cos
            .try_index_device((&window_index_array, .., ..), stream)?
            .reshape(&[seq_len, -1], stream)?;
        let sin = sin.reshape(
            &[
                seq_len / (self.config.spatial_merge_size * self.config.spatial_merge_size),
                self.config.spatial_merge_size * self.config.spatial_merge_size,
                -1,
            ],
            stream,
        )?;
        let sin = sin
            .try_index_device((&window_index_array, .., ..), stream)?
            .reshape(&[seq_len, -1], stream)?;

        let mut deepstack_features = Vec::new();
        for (layer_num, block) in self.blocks.iter_mut().enumerate() {
            let chunk_lengths = if self
                .config
                .fullatt_block_indexes
                .contains(&(layer_num as i32))
            {
                &full_chunk_lengths
            } else {
                &window_chunk_lengths
            };
            hidden_states = block.forward(hidden_states, chunk_lengths, &cos, &sin, stream)?;
            if let Some(index) = self
                .config
                .deepstack_visual_indexes
                .iter()
                .position(|&index| index == layer_num as i32)
            {
                let feature = self.deepstack_merger_list[index]
                    .forward(&hidden_states, stream)?
                    .try_index_device((NewAxis, .., ..), stream)?;
                deepstack_features.push(feature);
            }
        }
        let hidden_states = self.merger.forward(&hidden_states, stream)?;
        let reverse_index = reverse_permutation(&window_index);
        let reverse_index_array = Array::from_slice(&reverse_index, &[reverse_index.len() as i32]);
        let embeddings = hidden_states
            .try_index_device((&reverse_index_array, ..), stream)?
            .try_index_device((NewAxis, .., ..), stream)?;
        Ok(VisionOutput {
            embeddings,
            deepstack_features,
        })
    }

    pub(crate) fn training_mode(&mut self, mode: bool) {
        self.patch_embed.training_mode(mode);
        for block in &mut self.blocks {
            block.training_mode(mode);
        }
        self.merger.training_mode(mode);
        for merger in &mut self.deepstack_merger_list {
            merger.training_mode(mode);
        }
    }
}

fn apply_vision_rotary_pos_emb(
    query: Array,
    key: Array,
    cos: &Array,
    sin: &Array,
    stream: &Stream,
) -> Result<(Array, Array), Exception> {
    let query_dtype = query.dtype();
    let key_dtype = key.dtype();
    let query = query.as_dtype(Dtype::Float32, stream)?;
    let key = key.as_dtype(Dtype::Float32, stream)?;
    let cos = cos.try_index_device((.., NewAxis, ..), stream)?;
    let sin = sin.try_index_device((.., NewAxis, ..), stream)?;
    let query_embed = query.multiply(&cos, stream)?.add(
        rotate_half_vision(&query, stream)?.multiply(&sin, stream)?,
        stream,
    )?;
    let key_embed = key.multiply(cos, stream)?.add(
        rotate_half_vision(&key, stream)?.multiply(sin, stream)?,
        stream,
    )?;
    Ok((
        query_embed.as_dtype(query_dtype, stream)?,
        key_embed.as_dtype(key_dtype, stream)?,
    ))
}

fn rotate_half_vision(x: &Array, stream: &Stream) -> Result<Array, Exception> {
    let half = x.dim(-1) / 2;
    let x1 = x.try_index_device((.., .., ..half), stream)?;
    let x2 = x.try_index_device((.., .., half..), stream)?;
    concatenate_axis(
        &[x2.multiply(Array::from_f32(-1.0), stream)?, x1],
        -1,
        stream,
    )
}

pub(crate) fn grid_thw_from_array(
    grid_thw: &Array,
    stream: &Stream,
) -> Result<Vec<(i32, i32, i32)>, Exception> {
    let shape = grid_thw.shape();
    if shape.len() != 2 || shape[1] != 3 {
        return Err(Exception::custom(format!(
            "qwen3_5_moe qwen_grid_thw must be shaped [items, 3], got {shape:?}"
        )));
    }
    let mut grid = Vec::with_capacity(shape[0] as usize);
    for index in 0..shape[0] {
        let t = grid_thw
            .try_index_device((index, 0), stream)?
            .item::<i32>(stream);
        let h = grid_thw
            .try_index_device((index, 1), stream)?
            .item::<i32>(stream);
        let w = grid_thw
            .try_index_device((index, 2), stream)?
            .item::<i32>(stream);
        grid.push((t, h, w));
    }
    Ok(grid)
}

fn validate_vision_grid(
    grid: &[(i32, i32, i32)],
    spatial_merge_size: i32,
    pixel_values: &Array,
) -> Result<(), Exception> {
    let patches: i32 = grid.iter().map(|(t, h, w)| t * h * w).sum();
    if patches != pixel_values.dim(0) {
        return Err(Exception::custom(format!(
            "qwen3_5_moe qwen_grid_thw describes {patches} patches but image tensor has {}",
            pixel_values.dim(0)
        )));
    }
    for &(t, h, w) in grid {
        if t <= 0 || h <= 0 || w <= 0 {
            return Err(Exception::custom(format!(
                "qwen3_5_moe qwen_grid_thw entries must be positive, got {:?}",
                (t, h, w)
            )));
        }
        if h % spatial_merge_size != 0 || w % spatial_merge_size != 0 {
            return Err(Exception::custom(format!(
                "qwen3_5_moe qwen_grid_thw spatial dimensions must be divisible by spatial_merge_size {spatial_merge_size}, got {:?}",
                (t, h, w)
            )));
        }
    }
    Ok(())
}

fn vision_position_indices(
    grid: &[(i32, i32, i32)],
    num_position_embeddings: i32,
) -> Result<Vec<u32>, Exception> {
    let side = (num_position_embeddings as f64).sqrt() as i32;
    if side * side != num_position_embeddings {
        return Err(Exception::custom(format!(
            "Qwen VL vision num_position_embeddings must be a square, got {num_position_embeddings}"
        )));
    }
    let mut indices = Vec::new();
    for &(t, h, w) in grid {
        if h > side || w > side {
            return Err(Exception::custom(format!(
                "qwen3_5_moe qwen_grid_thw spatial dimensions {:?} exceed learned position table side {side}",
                (h, w)
            )));
        }
        for _ in 0..t {
            for h_pos in 0..h {
                for w_pos in 0..w {
                    indices.push((h_pos * side + w_pos) as u32);
                }
            }
        }
    }
    Ok(indices)
}

fn vision_interpolated_position_embeddings(
    pos_embed: &mut nn::Embedding,
    grid: &[(i32, i32, i32)],
    num_position_embeddings: i32,
    spatial_merge_size: i32,
    stream: &Stream,
) -> Result<Array, Exception> {
    let side = (num_position_embeddings as f64).sqrt() as i32;
    if side * side != num_position_embeddings {
        return Err(Exception::custom(format!(
            "Qwen VL vision num_position_embeddings must be a square, got {num_position_embeddings}"
        )));
    }
    let mut corner_indices = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let mut corner_weights = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    for &(t, h, w) in grid {
        let axis = |position: i32, length: i32| {
            if length == 1 {
                (0, 0, 0.0)
            } else {
                let value = position as f32 * (side - 1) as f32 / (length - 1) as f32;
                let floor = value.floor() as i32;
                (floor, (floor + 1).min(side - 1), value - floor as f32)
            }
        };
        for _ in 0..t {
            for h_block in 0..h / spatial_merge_size {
                for w_block in 0..w / spatial_merge_size {
                    for h_inner in 0..spatial_merge_size {
                        for w_inner in 0..spatial_merge_size {
                            let (h0, h1, hf) = axis(h_block * spatial_merge_size + h_inner, h);
                            let (w0, w1, wf) = axis(w_block * spatial_merge_size + w_inner, w);
                            for (corner, index, weight) in [
                                (0, h0 * side + w0, (1.0 - hf) * (1.0 - wf)),
                                (1, h0 * side + w1, (1.0 - hf) * wf),
                                (2, h1 * side + w0, hf * (1.0 - wf)),
                                (3, h1 * side + w1, hf * wf),
                            ] {
                                corner_indices[corner].push(index as u32);
                                corner_weights[corner].push(weight);
                            }
                        }
                    }
                }
            }
        }
    }
    let seq_len = corner_indices[0].len() as i32;
    let mut output: Option<Array> = None;
    for corner in 0..4 {
        let indices = Array::from_slice(&corner_indices[corner], &[seq_len]);
        let weights = Array::from_slice(&corner_weights[corner], &[seq_len, 1]);
        let weighted = pos_embed
            .forward(&indices, stream)?
            .multiply(weights, stream)?;
        output = Some(match output {
            Some(current) => current.add(weighted, stream)?,
            None => weighted,
        });
    }
    output.ok_or_else(|| Exception::custom("Qwen VL vision grid must not be empty"))
}

fn vision_attention_chunk_lengths(grid: &[(i32, i32, i32)]) -> Vec<i32> {
    let mut lengths = Vec::new();
    for &(t, h, w) in grid {
        for _ in 0..t {
            lengths.push(h * w);
        }
    }
    lengths
}

pub(crate) fn vision_window_index(
    grid: &[(i32, i32, i32)],
    spatial_merge_size: i32,
    window_size: i32,
    patch_size: i32,
) -> Result<(Vec<i32>, Vec<i32>), Exception> {
    let vit_merger_window_size = window_size / spatial_merge_size / patch_size;
    if vit_merger_window_size <= 0 {
        return Err(Exception::custom(format!(
            "Qwen VL vision window_size {window_size} is too small for spatial_merge_size {spatial_merge_size} and patch_size {patch_size}"
        )));
    }
    let spatial_merge_unit = spatial_merge_size * spatial_merge_size;
    let mut window_index = Vec::new();
    let mut cumulative_seqlens = vec![0];
    let mut window_index_id = 0;
    for &(grid_t, grid_h, grid_w) in grid {
        let llm_grid_h = grid_h / spatial_merge_size;
        let llm_grid_w = grid_w / spatial_merge_size;
        let pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size;
        let pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size;
        let num_windows_h = (llm_grid_h + pad_h) / vit_merger_window_size;
        let num_windows_w = (llm_grid_w + pad_w) / vit_merger_window_size;
        for t in 0..grid_t {
            for window_h in 0..num_windows_h {
                for window_w in 0..num_windows_w {
                    let mut window_groups = 0;
                    for inner_h in 0..vit_merger_window_size {
                        for inner_w in 0..vit_merger_window_size {
                            let h = window_h * vit_merger_window_size + inner_h;
                            let w = window_w * vit_merger_window_size + inner_w;
                            if h < llm_grid_h && w < llm_grid_w {
                                let index = t * llm_grid_h * llm_grid_w + h * llm_grid_w + w;
                                window_index.push(window_index_id + index);
                                window_groups += 1;
                            }
                        }
                    }
                    let next = cumulative_seqlens.last().copied().unwrap_or(0)
                        + window_groups * spatial_merge_unit;
                    if cumulative_seqlens.last().copied() != Some(next) {
                        cumulative_seqlens.push(next);
                    }
                }
            }
        }
        window_index_id += grid_t * llm_grid_h * llm_grid_w;
    }
    let chunk_lengths = cumulative_seqlens
        .windows(2)
        .map(|window| window[1] - window[0])
        .collect::<Vec<_>>();
    Ok((window_index, chunk_lengths))
}

pub(crate) fn reverse_permutation(indices: &[i32]) -> Vec<i32> {
    let mut reverse = vec![0; indices.len()];
    for (position, &index) in indices.iter().enumerate() {
        reverse[index as usize] = position as i32;
    }
    reverse
}

fn vision_rotary_embeddings(
    grid: &[(i32, i32, i32)],
    spatial_merge_size: i32,
    head_dim: i32,
) -> (Array, Array) {
    let rotary_dim = head_dim / 2;
    let inv_freq = (0..rotary_dim)
        .step_by(2)
        .map(|idx| 1.0f32 / 10000.0f32.powf(idx as f32 / rotary_dim as f32))
        .collect::<Vec<_>>();
    let mut cos_values = Vec::new();
    let mut sin_values = Vec::new();
    for &(t, h, w) in grid {
        for _ in 0..t {
            for h_block in 0..(h / spatial_merge_size) {
                for w_block in 0..(w / spatial_merge_size) {
                    for h_inner in 0..spatial_merge_size {
                        for w_inner in 0..spatial_merge_size {
                            let h_pos = h_block * spatial_merge_size + h_inner;
                            let w_pos = w_block * spatial_merge_size + w_inner;
                            let mut angles = Vec::with_capacity(rotary_dim as usize);
                            for position in [h_pos, w_pos] {
                                for inv in &inv_freq {
                                    angles.push(position as f32 * inv);
                                }
                            }
                            let full_angles = angles
                                .iter()
                                .chain(angles.iter())
                                .copied()
                                .collect::<Vec<_>>();
                            for angle in full_angles {
                                cos_values.push(angle.cos());
                                sin_values.push(angle.sin());
                            }
                        }
                    }
                }
            }
        }
    }
    let seq_len = (cos_values.len() as i32) / head_dim;
    (
        Array::from_slice(&cos_values, &[seq_len, head_dim]),
        Array::from_slice(&sin_values, &[seq_len, head_dim]),
    )
}
