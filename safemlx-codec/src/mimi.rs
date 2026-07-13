//! Mimi neural audio tokenizer support.
//!
//! Mimi is the neural audio codec used by Moshi-family realtime models. This
//! module implements safemlx-native checkpoint loading, the split residual
//! vector quantizer, and the non-streaming SEANet/transformer encoder and
//! decoder used to map between PCM and Mimi codebook tokens.

use std::{collections::HashMap, fs::File, path::Path, rc::Rc};

use memmap2::MmapOptions;
use safemlx::{
    argmin_axis,
    builder::Builder,
    fast::{scaled_dot_product_attention, ScaledDotProductAttentionMask},
    macros::ModuleParameters,
    module::{Module, ModuleParameters, ModuleParametersExt, Param},
    nn,
    ops::{
        concatenate_axis, conv1d, conv_transpose1d, indexing::TryIndexOp, matmul, maximum, pad,
        stack_axis, sum_axis, PadMode as MlxPadMode,
    },
    Array, Dtype, Stream,
};
use safetensors::SafeTensors;

use crate::{AudioTokenizer, AudioTokenizerConfig, Error};

const EPSILON: f32 = 1e-5;

/// Default released Mimi checkpoint filename used by PersonaPlex.
pub const PERSONAPLEX_MIMI_SAFETENSORS: &str = "tokenizer-e351c8d8-checkpoint125.safetensors";

/// Mimi resampling strategy.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ResampleMethod {
    /// Learned convolutional resampling.
    Conv,
}

/// Mimi codec configuration.
#[derive(Debug, Clone)]
pub struct Config {
    /// Audio channels.
    pub channels: i32,
    /// PCM sample rate.
    pub sample_rate: f64,
    /// Codec frame rate.
    pub frame_rate: f64,
    /// Whether the original training path renormalized audio.
    pub renormalize: bool,
    /// Latent resampling method.
    pub resample_method: ResampleMethod,
    /// Active residual codebooks.
    pub num_codebooks: i32,
    /// Total codebooks available in the released checkpoint.
    pub total_codebooks: i32,
    /// Codebook cardinality.
    pub bins: i32,
    /// Codebook embedding dimension.
    pub quantizer_dim: i32,
    /// Model latent dimension.
    pub latent_dim: i32,
}

impl Config {
    /// Released Mimi v0.1 defaults, with a caller-selected active codebook count.
    pub fn v0_1(num_codebooks: Option<i32>) -> Self {
        Self {
            channels: 1,
            sample_rate: 24_000.0,
            frame_rate: 12.5,
            renormalize: true,
            resample_method: ResampleMethod::Conv,
            num_codebooks: num_codebooks.unwrap_or(16),
            total_codebooks: 32,
            bins: 2_048,
            quantizer_dim: 256,
            latent_dim: 512,
        }
    }

    fn validate(&self) -> Result<(), Error> {
        if self.channels <= 0
            || self.sample_rate <= 0.0
            || self.frame_rate <= 0.0
            || self.num_codebooks <= 0
            || self.num_codebooks > self.total_codebooks
            || self.bins <= 0
            || self.quantizer_dim <= 0
            || self.latent_dim <= 0
        {
            return Err(Error::InvalidShape(format!(
                "invalid Mimi config: channels={}, sample_rate={}, frame_rate={}, num_codebooks={}, total_codebooks={}, bins={}, quantizer_dim={}, latent_dim={}",
                self.channels,
                self.sample_rate,
                self.frame_rate,
                self.num_codebooks,
                self.total_codebooks,
                self.bins,
                self.quantizer_dim,
                self.latent_dim
            )));
        }
        Ok(())
    }
}

/// Mimi audio tokenizer.
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = safemlx)]
pub struct Mimi {
    /// Split residual vector quantizer.
    #[param]
    pub quantizer: SplitResidualVectorQuantizer,
    #[param]
    encoder: SeaNetEncoder,
    #[param]
    encoder_transformer: MimiTransformer,
    #[param]
    downsample: StreamableConv1d,
    #[param]
    upsample: StreamableConvTranspose1d,
    #[param]
    decoder_transformer: MimiTransformer,
    #[param]
    decoder: SeaNetDecoder,
    config: Config,
}

impl Mimi {
    /// Creates an unloaded Mimi tokenizer from config.
    pub fn new(config: Config, stream: &Stream) -> Result<Self, Error> {
        config.validate()?;
        Ok(Self {
            quantizer: SplitResidualVectorQuantizer::unloaded(&config, stream)?,
            encoder: SeaNetEncoder::unloaded(stream)?,
            encoder_transformer: MimiTransformer::unloaded(stream)?,
            downsample: StreamableConv1d::unloaded_with_pad_mode(
                config.latent_dim,
                config.latent_dim,
                4,
                2,
                1,
                1,
                false,
                ConvPadMode::Edge,
                stream,
            )?,
            upsample: StreamableConvTranspose1d::unloaded(
                config.latent_dim,
                config.latent_dim,
                4,
                2,
                config.latent_dim,
                false,
                stream,
            )?,
            decoder_transformer: MimiTransformer::unloaded(stream)?,
            decoder: SeaNetDecoder::unloaded(stream)?,
            config,
        })
    }

    /// Loads a Mimi tokenizer checkpoint.
    pub fn load(
        path: impl AsRef<Path>,
        num_codebooks: Option<i32>,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let mut model = Self::new(Config::v0_1(num_codebooks), stream)?;
        let transformed = load_decoder_safetensors_arrays(path, stream)?
            .into_iter()
            .map(|result| result.map(|(key, value)| (Rc::<str>::from(key), value)))
            .collect::<Result<HashMap<_, _>, _>>()?;
        let params = model.parameters().flatten();
        let mut missing = Vec::new();
        for (key, parameter) in params {
            match transformed.get(key.as_ref()) {
                None => missing.push(key.to_string()),
                Some(value) if value.shape() != parameter.shape() => {
                    return Err(Error::InvalidShape(format!(
                        "Mimi checkpoint tensor {key} has shape {:?}, expected {:?}",
                        value.shape(),
                        parameter.shape()
                    )));
                }
                Some(_) => {}
            }
        }
        if !missing.is_empty() {
            missing.sort();
            return Err(Error::InvalidShape(format!(
                "Mimi checkpoint is missing {} model tensors: {}",
                missing.len(),
                missing.join(", ")
            )));
        }
        model.update_flattened(transformed);
        model.copy_to_stream(stream)?;
        Ok(model)
    }

    /// Returns the Mimi configuration.
    pub fn mimi_config(&self) -> &Config {
        &self.config
    }

    /// Encodes latent frames shaped `[batch, 512, frames]` into Mimi tokens.
    pub fn encode_latent(&mut self, latent: &Array, stream: &Stream) -> Result<Array, Error> {
        self.quantizer.encode(latent, stream)
    }

    /// Encodes PCM shaped `[batch, 1, samples]` into Mimi tokens `[batch, codebooks, frames]`.
    pub fn encode(&mut self, pcm: &Array, stream: &Stream) -> Result<Array, Error> {
        let latent = self.encoder.forward(pcm, stream)?;
        let latent = self.encoder_transformer.forward(&latent, stream)?;
        let latent = self.downsample.forward(&latent, stream)?;
        self.quantizer.encode(&latent, stream)
    }

    /// Resets state used by [`Mimi::encode_step`].
    pub fn reset_encode_state(&mut self) {
        self.encoder.reset_state();
        self.encoder_transformer.reset_state();
        self.downsample.reset_state();
    }

    /// Encodes one PCM frame into the next Mimi token frame.
    ///
    /// Accepts PCM shaped `[batch, 1, samples]`. Returns `None` until the
    /// streaming encoder has enough samples to emit a complete codec frame.
    pub fn encode_step(&mut self, pcm: &Array, stream: &Stream) -> Result<Option<Array>, Error> {
        let latent = match self.encoder.step(pcm, stream)? {
            Some(latent) => latent,
            None => return Ok(None),
        };
        let latent = self.encoder_transformer.step(&latent, stream)?;
        let latent = match self.downsample.step(&latent, stream)? {
            Some(latent) => latent,
            None => return Ok(None),
        };
        Ok(Some(
            self.quantizer
                .encode(&latent, stream)?
                .squeeze_axes(&[2], stream)?,
        ))
    }

    /// Decodes Mimi tokens shaped `[batch, codebooks, frames]` into latent frames.
    pub fn decode_latent(&mut self, codes: &Array, stream: &Stream) -> Result<Array, Error> {
        self.quantizer.decode(codes, stream)
    }

    /// Decodes Mimi tokens shaped `[batch, codebooks, frames]` into PCM `[batch, 1, samples]`.
    pub fn decode(&mut self, codes: &Array, stream: &Stream) -> Result<Array, Error> {
        let latent = self.quantizer.decode(codes, stream)?;
        let latent = self.upsample.forward(&latent, stream)?;
        let latent = self.decoder_transformer.forward(&latent, stream)?;
        self.decoder.forward(&latent, stream)
    }

    /// Resets state used by [`Mimi::decode_step`].
    pub fn reset_decode_state(&mut self) {
        self.upsample.reset_state();
        self.decoder_transformer.reset_state();
        self.decoder.reset_state();
    }

    /// Decodes one Mimi token frame into the next PCM chunk.
    ///
    /// Accepts codes shaped `[batch, codebooks]` or `[batch, codebooks, 1]`.
    pub fn decode_step(&mut self, codes: &Array, stream: &Stream) -> Result<Array, Error> {
        let codes = match codes.shape() {
            [_, _] => codes.expand_dims(2, stream)?,
            [_, _, 1] => codes.clone(),
            _ => {
                return Err(Error::InvalidShape(format!(
                    "Mimi decode_step expects [batch, codebooks] or [batch, codebooks, 1], got {:?}",
                    codes.shape()
                )));
            }
        };
        let latent = self.quantizer.decode(&codes, stream)?;
        let latent = self.upsample.step(&latent, stream)?;
        let latent = self.decoder_transformer.step(&latent, stream)?;
        self.decoder.step(&latent, stream)
    }
}

fn load_decoder_safetensors_arrays(
    path: impl AsRef<Path>,
    stream: &Stream,
) -> Result<impl Iterator<Item = Result<(String, Array), Error>>, Error> {
    let file = File::open(path)?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    let tensors = SafeTensors::deserialize(&mmap).map_err(|err| Error::Other(Box::new(err)))?;
    let mut loaded = Vec::new();
    for (key, view) in tensors.iter() {
        let Some(key) = transform_decoder_key(key) else {
            continue;
        };
        let mut value = Array::try_from(view).map_err(|err| Error::Other(Box::new(err)))?;
        if key.ends_with(".weight") && is_conv_weight_key(&key) {
            value = pytorch_conv_weight_to_mlx(&key, value, stream)?;
        }
        loaded.push(Ok((key, value.copy(stream)?)));
    }
    Ok(loaded.into_iter())
}

fn transform_decoder_key(key: &str) -> Option<String> {
    if key.starts_with("quantizer.") {
        return Some(key.to_string());
    }
    if key == "downsample.conv.conv.conv.weight" {
        return Some("downsample.weight".to_string());
    }
    if let Some(key) = key.strip_prefix("encoder_transformer.transformer.") {
        let key = key
            .replace(".self_attn.in_proj_weight", ".self_attn.in_proj.weight")
            .replace(".linear1.", ".mlp.linear1.")
            .replace(".linear2.", ".mlp.linear2.");
        return Some(format!("encoder_transformer.{key}"));
    }
    if let Some(key) = key.strip_prefix("encoder.model.") {
        return transform_seanet_encoder_key(key);
    }
    if key == "upsample.convtr.convtr.convtr.weight" {
        return Some("upsample.weight".to_string());
    }
    if let Some(key) = key.strip_prefix("decoder_transformer.transformer.") {
        let key = key
            .replace(".self_attn.in_proj_weight", ".self_attn.in_proj.weight")
            .replace(".linear1.", ".mlp.linear1.")
            .replace(".linear2.", ".mlp.linear2.");
        return Some(format!("decoder_transformer.{key}"));
    }
    if let Some(key) = key.strip_prefix("decoder.model.") {
        return transform_seanet_decoder_key(key);
    }
    None
}

fn transform_seanet_encoder_key(key: &str) -> Option<String> {
    let (source, target) = [
        ("0.conv.conv.", "encoder.init_conv1d."),
        (
            "1.block.1.conv.conv.",
            "encoder.layers.0.residuals.0.block.0.",
        ),
        (
            "1.block.3.conv.conv.",
            "encoder.layers.0.residuals.0.block.1.",
        ),
        ("3.conv.conv.", "encoder.layers.0.downsample."),
        (
            "4.block.1.conv.conv.",
            "encoder.layers.1.residuals.0.block.0.",
        ),
        (
            "4.block.3.conv.conv.",
            "encoder.layers.1.residuals.0.block.1.",
        ),
        ("6.conv.conv.", "encoder.layers.1.downsample."),
        (
            "7.block.1.conv.conv.",
            "encoder.layers.2.residuals.0.block.0.",
        ),
        (
            "7.block.3.conv.conv.",
            "encoder.layers.2.residuals.0.block.1.",
        ),
        ("9.conv.conv.", "encoder.layers.2.downsample."),
        (
            "10.block.1.conv.conv.",
            "encoder.layers.3.residuals.0.block.0.",
        ),
        (
            "10.block.3.conv.conv.",
            "encoder.layers.3.residuals.0.block.1.",
        ),
        ("12.conv.conv.", "encoder.layers.3.downsample."),
        ("14.conv.conv.", "encoder.final_conv1d."),
    ]
    .into_iter()
    .find(|(source, _)| key.starts_with(source))?;
    Some(format!("{target}{}", &key[source.len()..]))
}

fn transform_seanet_decoder_key(key: &str) -> Option<String> {
    let (source, target) = [
        ("0.conv.conv.", "decoder.init_conv1d."),
        ("2.convtr.convtr.", "decoder.layers.0.upsample."),
        (
            "3.block.1.conv.conv.",
            "decoder.layers.0.residuals.0.block.0.",
        ),
        (
            "3.block.3.conv.conv.",
            "decoder.layers.0.residuals.0.block.1.",
        ),
        ("5.convtr.convtr.", "decoder.layers.1.upsample."),
        (
            "6.block.1.conv.conv.",
            "decoder.layers.1.residuals.0.block.0.",
        ),
        (
            "6.block.3.conv.conv.",
            "decoder.layers.1.residuals.0.block.1.",
        ),
        ("8.convtr.convtr.", "decoder.layers.2.upsample."),
        (
            "9.block.1.conv.conv.",
            "decoder.layers.2.residuals.0.block.0.",
        ),
        (
            "9.block.3.conv.conv.",
            "decoder.layers.2.residuals.0.block.1.",
        ),
        ("11.convtr.convtr.", "decoder.layers.3.upsample."),
        (
            "12.block.1.conv.conv.",
            "decoder.layers.3.residuals.0.block.0.",
        ),
        (
            "12.block.3.conv.conv.",
            "decoder.layers.3.residuals.0.block.1.",
        ),
        ("14.conv.conv.", "decoder.final_conv1d."),
    ]
    .into_iter()
    .find(|(source, _)| key.starts_with(source))?;
    Some(format!("{target}{}", &key[source.len()..]))
}

fn is_conv_weight_key(key: &str) -> bool {
    key.starts_with("upsample.")
        || key.starts_with("downsample.")
        || key.contains(".upsample.")
        || key.contains(".downsample.")
        || key.contains(".init_conv1d.")
        || key.contains(".final_conv1d.")
        || key.contains(".block.")
}

fn pytorch_conv_weight_to_mlx(key: &str, value: Array, stream: &Stream) -> Result<Array, Error> {
    if value.shape().len() != 3 {
        return Ok(value);
    }
    if key.contains(".upsample.") {
        Ok(value.transpose_axes(&[1, 2, 0], stream)?)
    } else {
        Ok(value.transpose_axes(&[0, 2, 1], stream)?)
    }
}

impl AudioTokenizer for Mimi {
    fn config(&self) -> AudioTokenizerConfig {
        AudioTokenizerConfig {
            sample_rate: self.config.sample_rate,
            frame_rate: self.config.frame_rate,
            channels: self.config.channels,
            codebooks: self.config.num_codebooks,
            cardinality: self.config.bins,
        }
    }

    fn encode(&mut self, pcm: &Array, stream: &Stream) -> Result<Array, Error> {
        self.encode(pcm, stream)
    }

    fn decode(&mut self, _codes: &Array, _stream: &Stream) -> Result<Array, Error> {
        self.decode(_codes, _stream)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
#[module(root = safemlx)]
struct SeaNetEncoder {
    #[param]
    init_conv1d: StreamableConv1d,
    #[param]
    layers: Vec<EncoderLayer>,
    #[param]
    final_conv1d: StreamableConv1d,
}

impl SeaNetEncoder {
    fn unloaded(stream: &Stream) -> Result<Self, Error> {
        let ratios = [4, 5, 6, 8];
        let mut channels = 64;
        let mut layers = Vec::with_capacity(ratios.len());
        for ratio in ratios {
            layers.push(EncoderLayer::unloaded(
                channels,
                channels * 2,
                ratio,
                stream,
            )?);
            channels *= 2;
        }
        Ok(Self {
            init_conv1d: StreamableConv1d::unloaded(1, 64, 7, 1, 1, 1, true, stream)?,
            layers,
            final_conv1d: StreamableConv1d::unloaded(1024, 512, 3, 1, 1, 1, true, stream)?,
        })
    }

    fn forward(&mut self, pcm: &Array, stream: &Stream) -> Result<Array, Error> {
        validate_pcm(pcm)?;
        let mut x = self.init_conv1d.forward(pcm, stream)?;
        for layer in &mut self.layers {
            x = layer.forward(&x, stream)?;
        }
        self.final_conv1d
            .forward(&nn::elu(&x, Some(1.0), stream)?, stream)
    }

    fn reset_state(&mut self) {
        self.init_conv1d.reset_state();
        for layer in &mut self.layers {
            layer.reset_state();
        }
        self.final_conv1d.reset_state();
    }

    fn step(&mut self, pcm: &Array, stream: &Stream) -> Result<Option<Array>, Error> {
        validate_pcm(pcm)?;
        let mut x = match self.init_conv1d.step(pcm, stream)? {
            Some(x) => x,
            None => return Ok(None),
        };
        for layer in &mut self.layers {
            x = match layer.step(&x, stream)? {
                Some(x) => x,
                None => return Ok(None),
            };
        }
        Ok(self
            .final_conv1d
            .step(&nn::elu(&x, Some(1.0), stream)?, stream)?)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
#[module(root = safemlx)]
struct EncoderLayer {
    #[param]
    residuals: Vec<SeaNetResnetBlock>,
    #[param]
    downsample: StreamableConv1d,
}

impl EncoderLayer {
    fn unloaded(
        in_channels: i32,
        out_channels: i32,
        ratio: i32,
        stream: &Stream,
    ) -> Result<Self, Error> {
        Ok(Self {
            residuals: vec![SeaNetResnetBlock::unloaded(in_channels, stream)?],
            downsample: StreamableConv1d::unloaded(
                in_channels,
                out_channels,
                ratio * 2,
                ratio,
                1,
                1,
                true,
                stream,
            )?,
        })
    }

    fn forward(&mut self, x: &Array, stream: &Stream) -> Result<Array, Error> {
        let mut x = x.clone();
        for residual in &mut self.residuals {
            x = residual.forward(&x, stream)?;
        }
        self.downsample
            .forward(&nn::elu(&x, Some(1.0), stream)?, stream)
    }

    fn reset_state(&mut self) {
        for residual in &mut self.residuals {
            residual.reset_state();
        }
        self.downsample.reset_state();
    }

    fn step(&mut self, x: &Array, stream: &Stream) -> Result<Option<Array>, Error> {
        let mut x = x.clone();
        for residual in &mut self.residuals {
            x = residual.step(&x, stream)?;
        }
        self.downsample
            .step(&nn::elu(&x, Some(1.0), stream)?, stream)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
#[module(root = safemlx)]
struct MimiTransformer {
    #[param]
    layers: Vec<MimiTransformerLayer>,
}

impl MimiTransformer {
    fn unloaded(stream: &Stream) -> Result<Self, Error> {
        Ok(Self {
            layers: (0..8)
                .map(|_| MimiTransformerLayer::unloaded(stream))
                .collect::<Result<Vec<_>, _>>()?,
        })
    }

    fn forward(&mut self, latent: &Array, stream: &Stream) -> Result<Array, Error> {
        let mut x = latent.swap_axes(1, 2, stream)?;
        for layer in &mut self.layers {
            x = layer.forward(&x, stream)?;
        }
        Ok(x.swap_axes(1, 2, stream)?)
    }

    fn reset_state(&mut self) {
        for layer in &mut self.layers {
            layer.reset_state();
        }
    }

    fn step(&mut self, latent: &Array, stream: &Stream) -> Result<Array, Error> {
        let mut x = latent.swap_axes(1, 2, stream)?;
        for layer in &mut self.layers {
            x = layer.step(&x, stream)?;
        }
        Ok(x.swap_axes(1, 2, stream)?)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
#[module(root = safemlx)]
struct MimiTransformerLayer {
    #[param]
    norm1: nn::LayerNorm,
    #[param]
    norm2: nn::LayerNorm,
    #[param]
    self_attn: MimiSelfAttention,
    #[param]
    mlp: MimiMlp,
    #[param]
    layer_scale_1: LayerScale,
    #[param]
    layer_scale_2: LayerScale,
}

impl MimiTransformerLayer {
    fn unloaded(stream: &Stream) -> Result<Self, Error> {
        Ok(Self {
            norm1: unloaded_layer_norm(512, stream)?,
            norm2: unloaded_layer_norm(512, stream)?,
            self_attn: MimiSelfAttention::unloaded(stream)?,
            mlp: MimiMlp::unloaded(stream)?,
            layer_scale_1: LayerScale::unloaded(512, stream)?,
            layer_scale_2: LayerScale::unloaded(512, stream)?,
        })
    }

    fn forward(&mut self, x: &Array, stream: &Stream) -> Result<Array, Error> {
        let normed = self.norm1.forward(x, stream)?;
        let attended = self
            .self_attn
            .forward(&normed, stream)?
            .multiply(self.layer_scale_1.scale.as_ref(), stream)?;
        let x = x.add(attended, stream)?;
        let normed = self.norm2.forward(&x, stream)?;
        let mlp = self
            .mlp
            .forward(&normed, stream)?
            .multiply(self.layer_scale_2.scale.as_ref(), stream)?;
        Ok(x.add(mlp, stream)?)
    }

    fn reset_state(&mut self) {
        self.self_attn.reset_state();
    }

    fn step(&mut self, x: &Array, stream: &Stream) -> Result<Array, Error> {
        let normed = self.norm1.forward(x, stream)?;
        let attended = self
            .self_attn
            .step(&normed, stream)?
            .multiply(self.layer_scale_1.scale.as_ref(), stream)?;
        let x = x.add(attended, stream)?;
        let normed = self.norm2.forward(&x, stream)?;
        let mlp = self
            .mlp
            .forward(&normed, stream)?
            .multiply(self.layer_scale_2.scale.as_ref(), stream)?;
        Ok(x.add(mlp, stream)?)
    }
}

fn unloaded_layer_norm(dim: i32, stream: &Stream) -> Result<nn::LayerNorm, Error> {
    Ok(nn::LayerNorm {
        dimensions: dim,
        eps: 1e-5,
        weight: Param::<Option<Array>>::unloaded_some(&[dim], Dtype::Float32, stream)?,
        bias: Param::<Option<Array>>::unloaded_some(&[dim], Dtype::Float32, stream)?,
    })
}

#[derive(Debug, Clone, ModuleParameters)]
#[module(root = safemlx)]
struct LayerScale {
    #[param]
    scale: Param<Array>,
}

impl LayerScale {
    fn unloaded(dim: i32, stream: &Stream) -> Result<Self, Error> {
        Ok(Self {
            scale: Param::<Array>::unloaded(&[dim], Dtype::Float32, stream)?,
        })
    }
}

#[derive(Debug, Clone, ModuleParameters)]
#[module(root = safemlx)]
struct MimiMlp {
    #[param]
    linear1: nn::Linear,
    #[param]
    linear2: nn::Linear,
}

impl MimiMlp {
    fn unloaded(stream: &Stream) -> Result<Self, Error> {
        Ok(Self {
            linear1: nn::Linear::unloaded(512, 2048, false, Dtype::Float32, stream)?,
            linear2: nn::Linear::unloaded(2048, 512, false, Dtype::Float32, stream)?,
        })
    }

    fn forward(&mut self, x: &Array, stream: &Stream) -> Result<Array, Error> {
        let x = self.linear1.forward(x, stream)?;
        let x = nn::gelu(&x, stream)?;
        Ok(self.linear2.forward(&x, stream)?)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
#[module(root = safemlx)]
struct MimiSelfAttention {
    #[param]
    in_proj: nn::Linear,
    #[param]
    out_proj: nn::Linear,
    #[param]
    rope: nn::Rope,
    num_heads: i32,
    head_dim: i32,
    scale: f32,
    context: i32,
    key_cache: Option<Array>,
    value_cache: Option<Array>,
}

impl MimiSelfAttention {
    fn unloaded(stream: &Stream) -> Result<Self, Error> {
        let head_dim = 64;
        Ok(Self {
            in_proj: nn::Linear::unloaded(512, 1536, false, Dtype::Float32, stream)?,
            out_proj: nn::Linear::unloaded(512, 512, false, Dtype::Float32, stream)?,
            rope: nn::RopeBuilder::new(head_dim)
                .traditional(true)
                .base(10_000.0)
                .build()
                .expect("RopeBuilder is infallible"),
            num_heads: 8,
            head_dim,
            scale: (head_dim as f32).sqrt().recip(),
            context: 250,
            key_cache: None,
            value_cache: None,
        })
    }

    fn forward(&mut self, x: &Array, stream: &Stream) -> Result<Array, Error> {
        let shape = x.shape();
        if shape.len() != 3 || shape[2] != 512 {
            return Err(Error::InvalidShape(format!(
                "Mimi decoder transformer expects [batch, frames, 512], got {:?}",
                x.shape()
            )));
        }
        let (batch, seq, dim) = (shape[0], shape[1], shape[2]);
        let qkv = self
            .in_proj
            .forward(x, stream)?
            .reshape(&[batch, seq, 3, self.num_heads, self.head_dim], stream)?;
        let mut q = qkv
            .try_index_device((.., .., 0, .., ..), stream)?
            .transpose_axes(&[0, 2, 1, 3], stream)?;
        let mut k = qkv
            .try_index_device((.., .., 1, .., ..), stream)?
            .transpose_axes(&[0, 2, 1, 3], stream)?;
        let v = qkv
            .try_index_device((.., .., 2, .., ..), stream)?
            .transpose_axes(&[0, 2, 1, 3], stream)?;
        q = self.rope.forward((&q, 0), stream)?;
        k = self.rope.forward((&k, 0), stream)?;
        let attended = scaled_dot_product_attention(
            &q,
            &k,
            &v,
            self.scale,
            Some(ScaledDotProductAttentionMask::Causal),
            None,
            stream,
        )?
        .transpose_axes(&[0, 2, 1, 3], stream)?
        .reshape(&[batch, seq, dim], stream)?;
        Ok(self.out_proj.forward(&attended, stream)?)
    }

    fn reset_state(&mut self) {
        self.key_cache = None;
        self.value_cache = None;
    }

    fn step(&mut self, x: &Array, stream: &Stream) -> Result<Array, Error> {
        let shape = x.shape();
        if shape.len() != 3 || shape[2] != 512 {
            return Err(Error::InvalidShape(format!(
                "Mimi decoder transformer step expects [batch, frames, 512], got {:?}",
                x.shape()
            )));
        }
        let (batch, seq, dim) = (shape[0], shape[1], shape[2]);
        let prev_len = self
            .key_cache
            .as_ref()
            .map(|cache| cache.dim(2))
            .unwrap_or(0);
        let qkv = self
            .in_proj
            .forward(x, stream)?
            .reshape(&[batch, seq, 3, self.num_heads, self.head_dim], stream)?;
        let mut q = qkv
            .try_index_device((.., .., 0, .., ..), stream)?
            .transpose_axes(&[0, 2, 1, 3], stream)?;
        let mut k = qkv
            .try_index_device((.., .., 1, .., ..), stream)?
            .transpose_axes(&[0, 2, 1, 3], stream)?;
        let v = qkv
            .try_index_device((.., .., 2, .., ..), stream)?
            .transpose_axes(&[0, 2, 1, 3], stream)?;
        q = self.rope.forward((&q, prev_len), stream)?;
        k = self.rope.forward((&k, prev_len), stream)?;

        let mut keys = match self.key_cache.take() {
            Some(prev) => concatenate_axis(&[prev, k], 2, stream)?,
            None => k,
        };
        let mut values = match self.value_cache.take() {
            Some(prev) => concatenate_axis(&[prev, v], 2, stream)?,
            None => v,
        };
        let key_len = keys.dim(2);
        if key_len > self.context + seq {
            let start = key_len - (self.context + seq);
            keys = keys.try_index_device((.., .., start.., ..), stream)?;
            values = values.try_index_device((.., .., start.., ..), stream)?;
        }
        let retained_prev_len = keys.dim(2) - seq;
        let mask = streaming_attention_mask(batch, seq, retained_prev_len, self.context, stream)?;
        let attended = scaled_dot_product_attention(
            &q,
            &keys,
            &values,
            self.scale,
            Some(ScaledDotProductAttentionMask::Array(&mask)),
            None,
            stream,
        )?
        .transpose_axes(&[0, 2, 1, 3], stream)?
        .reshape(&[batch, seq, dim], stream)?;
        self.key_cache = Some(keys);
        self.value_cache = Some(values);
        Ok(self.out_proj.forward(&attended, stream)?)
    }
}

fn streaming_attention_mask(
    batch: i32,
    query_len: i32,
    prev_len: i32,
    context: i32,
    stream: &Stream,
) -> Result<Array, Error> {
    let key_len = prev_len + query_len;
    let mut mask = Vec::with_capacity((batch * query_len * key_len) as usize);
    for _ in 0..batch {
        for q in 0..query_len {
            let q_pos = prev_len + q;
            for k in 0..key_len {
                if k <= q_pos && q_pos <= k + context {
                    mask.push(0.0f32);
                } else {
                    mask.push(f32::NEG_INFINITY);
                }
            }
        }
    }
    Ok(Array::from_slice(&mask, &[batch, 1, query_len, key_len]).copy(stream)?)
}

#[derive(Debug, Clone, ModuleParameters)]
#[module(root = safemlx)]
struct SeaNetDecoder {
    #[param]
    init_conv1d: StreamableConv1d,
    #[param]
    layers: Vec<DecoderLayer>,
    #[param]
    final_conv1d: StreamableConv1d,
}

impl SeaNetDecoder {
    fn unloaded(stream: &Stream) -> Result<Self, Error> {
        let ratios = [8, 6, 5, 4];
        let mut channels = 1024;
        let mut layers = Vec::with_capacity(ratios.len());
        for ratio in ratios {
            let out_channels = channels / 2;
            layers.push(DecoderLayer::unloaded(
                channels,
                out_channels,
                ratio,
                stream,
            )?);
            channels = out_channels;
        }
        Ok(Self {
            init_conv1d: StreamableConv1d::unloaded(512, 1024, 7, 1, 1, 1, true, stream)?,
            layers,
            final_conv1d: StreamableConv1d::unloaded(64, 1, 3, 1, 1, 1, true, stream)?,
        })
    }

    fn forward(&mut self, latent: &Array, stream: &Stream) -> Result<Array, Error> {
        let mut x = self.init_conv1d.forward(latent, stream)?;
        for layer in &mut self.layers {
            x = layer.forward(&nn::elu(&x, Some(1.0), stream)?, stream)?;
        }
        self.final_conv1d
            .forward(&nn::elu(&x, Some(1.0), stream)?, stream)
    }

    fn reset_state(&mut self) {
        self.init_conv1d.reset_state();
        for layer in &mut self.layers {
            layer.reset_state();
        }
        self.final_conv1d.reset_state();
    }

    fn step(&mut self, latent: &Array, stream: &Stream) -> Result<Array, Error> {
        let mut x = self.init_conv1d.step(latent, stream)?.ok_or_else(|| {
            Error::InvalidShape("Mimi decoder init conv produced no streaming output".into())
        })?;
        for layer in &mut self.layers {
            x = layer.step(&nn::elu(&x, Some(1.0), stream)?, stream)?;
        }
        self.final_conv1d
            .step(&nn::elu(&x, Some(1.0), stream)?, stream)?
            .ok_or_else(|| Error::InvalidShape("Mimi decoder final conv produced no output".into()))
    }
}

#[derive(Debug, Clone, ModuleParameters)]
#[module(root = safemlx)]
struct DecoderLayer {
    #[param]
    upsample: StreamableConvTranspose1d,
    #[param]
    residuals: Vec<SeaNetResnetBlock>,
}

impl DecoderLayer {
    fn unloaded(
        in_channels: i32,
        out_channels: i32,
        ratio: i32,
        stream: &Stream,
    ) -> Result<Self, Error> {
        Ok(Self {
            upsample: StreamableConvTranspose1d::unloaded(
                in_channels,
                out_channels,
                ratio * 2,
                ratio,
                1,
                true,
                stream,
            )?,
            residuals: vec![SeaNetResnetBlock::unloaded(out_channels, stream)?],
        })
    }

    fn forward(&mut self, x: &Array, stream: &Stream) -> Result<Array, Error> {
        let mut x = self.upsample.forward(x, stream)?;
        for residual in &mut self.residuals {
            x = residual.forward(&x, stream)?;
        }
        Ok(x)
    }

    fn reset_state(&mut self) {
        self.upsample.reset_state();
        for residual in &mut self.residuals {
            residual.reset_state();
        }
    }

    fn step(&mut self, x: &Array, stream: &Stream) -> Result<Array, Error> {
        let mut x = self.upsample.step(x, stream)?;
        for residual in &mut self.residuals {
            x = residual.step(&x, stream)?;
        }
        Ok(x)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
#[module(root = safemlx)]
struct SeaNetResnetBlock {
    #[param]
    block: Vec<StreamableConv1d>,
}

impl SeaNetResnetBlock {
    fn unloaded(channels: i32, stream: &Stream) -> Result<Self, Error> {
        Ok(Self {
            block: vec![
                StreamableConv1d::unloaded(channels, channels / 2, 3, 1, 1, 1, true, stream)?,
                StreamableConv1d::unloaded(channels / 2, channels, 1, 1, 1, 1, true, stream)?,
            ],
        })
    }

    fn forward(&mut self, x: &Array, stream: &Stream) -> Result<Array, Error> {
        let mut y = x.clone();
        for conv in &mut self.block {
            y = conv.forward(&nn::elu(&y, Some(1.0), stream)?, stream)?;
        }
        Ok(y.add(x, stream)?)
    }

    fn reset_state(&mut self) {
        for conv in &mut self.block {
            conv.reset_state();
        }
    }

    fn step(&mut self, x: &Array, stream: &Stream) -> Result<Array, Error> {
        let mut y = x.clone();
        for conv in &mut self.block {
            y = conv
                .step(&nn::elu(&y, Some(1.0), stream)?, stream)?
                .ok_or_else(|| {
                    Error::InvalidShape("Mimi residual conv produced no output".into())
                })?;
        }
        Ok(y.add(x, stream)?)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
#[module(root = safemlx)]
struct StreamableConv1d {
    #[param]
    weight: Param<Array>,
    #[param]
    bias: Param<Option<Array>>,
    stride: i32,
    dilation: i32,
    groups: i32,
    pad_mode: ConvPadMode,
    state_prev_xs: Option<Array>,
    left_pad_applied: bool,
}

impl StreamableConv1d {
    fn unloaded(
        in_channels: i32,
        out_channels: i32,
        kernel_size: i32,
        stride: i32,
        dilation: i32,
        groups: i32,
        bias: bool,
        stream: &Stream,
    ) -> Result<Self, Error> {
        Self::unloaded_with_pad_mode(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation,
            groups,
            bias,
            ConvPadMode::Constant,
            stream,
        )
    }

    fn unloaded_with_pad_mode(
        in_channels: i32,
        out_channels: i32,
        kernel_size: i32,
        stride: i32,
        dilation: i32,
        groups: i32,
        bias: bool,
        pad_mode: ConvPadMode,
        stream: &Stream,
    ) -> Result<Self, Error> {
        Ok(Self {
            weight: Param::<Array>::unloaded(
                &[out_channels, kernel_size, in_channels / groups],
                Dtype::Float32,
                stream,
            )?,
            bias: if bias {
                Param::<Option<Array>>::unloaded_some(&[out_channels], Dtype::Float32, stream)?
            } else {
                Param::new(None)
            },
            stride,
            dilation,
            groups,
            pad_mode,
            state_prev_xs: None,
            left_pad_applied: false,
        })
    }

    fn forward(&mut self, x: &Array, stream: &Stream) -> Result<Array, Error> {
        let kernel_size = self.weight.as_ref().dim(1);
        let effective_kernel = (kernel_size - 1) * self.dilation + 1;
        let padding_total = effective_kernel - self.stride;
        let extra_padding =
            extra_padding_for_conv1d(x.dim(2), effective_kernel, self.stride, padding_total);
        let x = pad_bct(x, padding_total, extra_padding, self.pad_mode, stream)?;
        let x = x.swap_axes(1, 2, stream)?;
        let mut y = conv1d(
            &x,
            self.weight.as_ref(),
            self.stride,
            0,
            self.dilation,
            self.groups,
            stream,
        )?;
        if let Some(bias) = self.bias.as_ref() {
            y = y.add(bias, stream)?;
        }
        Ok(y.swap_axes(1, 2, stream)?)
    }

    fn reset_state(&mut self) {
        self.state_prev_xs = None;
        self.left_pad_applied = false;
    }

    fn step(&mut self, x: &Array, stream: &Stream) -> Result<Option<Array>, Error> {
        let kernel_size = self.weight.as_ref().dim(1);
        let effective_kernel = (kernel_size - 1) * self.dilation + 1;
        let padding_total = effective_kernel - self.stride;
        let x = if self.left_pad_applied {
            x.clone()
        } else {
            self.left_pad_applied = true;
            pad_bct(x, padding_total, 0, self.pad_mode, stream)?
        };
        let x = match self.state_prev_xs.take() {
            Some(prev) => concatenate_axis(&[prev, x], 2, stream)?,
            None => x,
        };
        let seq_len = x.dim(2);
        let num_frames = (seq_len + self.stride).saturating_sub(effective_kernel) / self.stride;
        if num_frames <= 0 {
            self.state_prev_xs = Some(x);
            return Ok(None);
        }
        let offset = num_frames * self.stride;
        self.state_prev_xs = Some(x.try_index_device((.., .., offset..), stream)?);
        let in_len = (num_frames - 1) * self.stride + effective_kernel;
        let x = x.try_index_device((.., .., 0..in_len), stream)?;
        self.forward_unpadded(&x, stream).map(Some)
    }

    fn forward_unpadded(&mut self, x: &Array, stream: &Stream) -> Result<Array, Error> {
        let x = x.swap_axes(1, 2, stream)?;
        let mut y = conv1d(
            &x,
            self.weight.as_ref(),
            self.stride,
            0,
            self.dilation,
            self.groups,
            stream,
        )?;
        if let Some(bias) = self.bias.as_ref() {
            y = y.add(bias, stream)?;
        }
        Ok(y.swap_axes(1, 2, stream)?)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
#[module(root = safemlx)]
struct StreamableConvTranspose1d {
    #[param]
    weight: Param<Array>,
    #[param]
    bias: Param<Option<Array>>,
    kernel_size: i32,
    stride: i32,
    groups: i32,
    state_prev_ys: Option<Array>,
}

impl StreamableConvTranspose1d {
    fn unloaded(
        in_channels: i32,
        out_channels: i32,
        kernel_size: i32,
        stride: i32,
        groups: i32,
        bias: bool,
        stream: &Stream,
    ) -> Result<Self, Error> {
        Ok(Self {
            weight: Param::<Array>::unloaded(
                &[out_channels, kernel_size, in_channels / groups],
                Dtype::Float32,
                stream,
            )?,
            bias: if bias {
                Param::<Option<Array>>::unloaded_some(&[out_channels], Dtype::Float32, stream)?
            } else {
                Param::new(None)
            },
            kernel_size,
            stride,
            groups,
            state_prev_ys: None,
        })
    }

    fn forward(&mut self, x: &Array, stream: &Stream) -> Result<Array, Error> {
        let y = self.forward_untrimmed(x, stream)?;
        let padding_total = self.kernel_size.saturating_sub(self.stride);
        unpad_bct(&y, 0, padding_total, stream)
    }

    fn reset_state(&mut self) {
        self.state_prev_ys = None;
    }

    fn step(&mut self, x: &Array, stream: &Stream) -> Result<Array, Error> {
        let y = self.forward_untrimmed(x, stream)?;
        let out_len = y.dim(2);
        let y = match self.state_prev_ys.take() {
            None => y,
            Some(prev) => {
                let prev_len = prev.dim(2);
                let prev = match self.bias.as_ref() {
                    None => prev,
                    Some(bias) => {
                        prev.subtract(&bias.reshape(&[1, bias.dim(0), 1], stream)?, stream)?
                    }
                };
                let y1 = y
                    .try_index_device((.., .., 0..prev_len), stream)?
                    .add(prev, stream)?;
                let y2 = y.try_index_device((.., .., prev_len..), stream)?;
                concatenate_axis(&[y1, y2], 2, stream)?
            }
        };
        let invalid_steps = self.kernel_size - self.stride;
        let split = out_len - invalid_steps;
        let out = y.try_index_device((.., .., 0..split), stream)?;
        self.state_prev_ys = Some(y.try_index_device((.., .., split..), stream)?);
        Ok(out)
    }

    fn forward_untrimmed(&mut self, x: &Array, stream: &Stream) -> Result<Array, Error> {
        let x = x.swap_axes(1, 2, stream)?;
        let mut y = conv_transpose1d(
            &x,
            self.weight.as_ref(),
            self.stride,
            0,
            1,
            0,
            self.groups,
            stream,
        )?;
        if let Some(bias) = self.bias.as_ref() {
            y = y.add(bias, stream)?;
        }
        Ok(y.swap_axes(1, 2, stream)?)
    }
}

#[derive(Debug, Clone, Copy)]
enum ConvPadMode {
    Constant,
    Edge,
}

impl From<ConvPadMode> for MlxPadMode {
    fn from(value: ConvPadMode) -> Self {
        match value {
            ConvPadMode::Constant => MlxPadMode::Constant,
            ConvPadMode::Edge => MlxPadMode::Edge,
        }
    }
}

fn extra_padding_for_conv1d(len: i32, kernel_size: i32, stride: i32, padding_total: i32) -> i32 {
    let n_frames = (len + padding_total - kernel_size) as f64 / stride as f64 + 1.0;
    let ideal_len = ((n_frames.ceil() as i32 - 1) * stride + kernel_size) - padding_total;
    ideal_len.saturating_sub(len)
}

fn pad_bct(
    x: &Array,
    left: i32,
    right: i32,
    mode: ConvPadMode,
    stream: &Stream,
) -> Result<Array, Error> {
    Ok(pad(
        x,
        &[(0, 0), (0, 0), (left, right)],
        None::<Array>,
        Some(MlxPadMode::from(mode)),
        stream,
    )?)
}

fn unpad_bct(x: &Array, left: i32, right: i32, stream: &Stream) -> Result<Array, Error> {
    let len = x.dim(2);
    if len < left + right {
        return Err(Error::InvalidShape(format!(
            "cannot unpad Mimi tensor of length {len} by {left}+{right}"
        )));
    }
    Ok(x.try_index_device((.., .., left..(len - right)), stream)?)
}

/// Split residual vector quantizer used by Mimi.
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = safemlx)]
pub struct SplitResidualVectorQuantizer {
    /// First semantic codebook branch.
    #[param]
    pub rvq_first: ResidualVectorQuantizer,
    /// Remaining acoustic codebook branch.
    #[param]
    pub rvq_rest: ResidualVectorQuantizer,
    n_q: i32,
}

impl SplitResidualVectorQuantizer {
    fn unloaded(config: &Config, stream: &Stream) -> Result<Self, Error> {
        Ok(Self {
            rvq_first: ResidualVectorQuantizer::unloaded(
                config.latent_dim,
                config.quantizer_dim,
                1,
                config.bins,
                stream,
            )?,
            rvq_rest: ResidualVectorQuantizer::unloaded(
                config.latent_dim,
                config.quantizer_dim,
                config.num_codebooks - 1,
                config.bins,
                stream,
            )?,
            n_q: config.num_codebooks,
        })
    }

    /// Encodes latent frames shaped `[batch, 512, frames]`.
    pub fn encode(&mut self, latent: &Array, stream: &Stream) -> Result<Array, Error> {
        validate_latent(latent)?;
        let first = self.rvq_first.encode(latent, stream)?;
        if self.n_q == 1 {
            Ok(first)
        } else {
            let rest = self.rvq_rest.encode(latent, stream)?;
            Ok(concatenate_axis(&[first, rest], 1, stream)?)
        }
    }

    /// Decodes tokens shaped `[batch, codebooks, frames]`.
    pub fn decode(&mut self, codes: &Array, stream: &Stream) -> Result<Array, Error> {
        validate_codes(codes, self.n_q)?;
        let first_codes = codes.try_index_device((.., 0..1, ..), stream)?;
        let mut quantized = self.rvq_first.decode(&first_codes, stream)?;
        if codes.dim(1) > 1 {
            let rest_codes = codes.try_index_device((.., 1.., ..), stream)?;
            quantized = quantized.add(self.rvq_rest.decode(&rest_codes, stream)?, stream)?;
        }
        Ok(quantized)
    }
}

/// Residual vector quantizer branch.
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = safemlx)]
pub struct ResidualVectorQuantizer {
    /// Input projection from Mimi latent dimension into codebook dimension.
    #[param]
    pub input_proj: Conv1x1NoBias,
    /// Output projection from codebook dimension back into Mimi latent dimension.
    #[param]
    pub output_proj: Conv1x1NoBias,
    /// Residual codebook layers.
    #[param]
    pub vq: ResidualVectorQuantization,
}

impl ResidualVectorQuantizer {
    fn unloaded(
        latent_dim: i32,
        codebook_dim: i32,
        layers: i32,
        bins: i32,
        stream: &Stream,
    ) -> Result<Self, Error> {
        Ok(Self {
            input_proj: Conv1x1NoBias::unloaded(latent_dim, codebook_dim, stream)?,
            output_proj: Conv1x1NoBias::unloaded(codebook_dim, latent_dim, stream)?,
            vq: ResidualVectorQuantization::unloaded(layers, codebook_dim, bins, stream)?,
        })
    }

    fn encode(&mut self, latent: &Array, stream: &Stream) -> Result<Array, Error> {
        self.vq
            .encode(&self.input_proj.forward(latent, stream)?, stream)
    }

    fn decode(&mut self, codes: &Array, stream: &Stream) -> Result<Array, Error> {
        self.output_proj
            .forward(&self.vq.decode(codes, stream)?, stream)
    }
}

/// Residual vector quantization layers.
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = safemlx)]
pub struct ResidualVectorQuantization {
    /// Ordered residual quantization layers.
    #[param]
    pub layers: Vec<VectorQuantization>,
}

impl ResidualVectorQuantization {
    fn unloaded(layers: i32, dim: i32, bins: i32, stream: &Stream) -> Result<Self, Error> {
        Ok(Self {
            layers: (0..layers)
                .map(|_| VectorQuantization::unloaded(dim, bins, stream))
                .collect::<Result<Vec<_>, _>>()?,
        })
    }

    fn encode(&mut self, latent: &Array, stream: &Stream) -> Result<Array, Error> {
        if self.layers.is_empty() {
            return Err(Error::InvalidShape("Mimi RVQ has no layers".into()));
        }
        let mut residual = latent.clone();
        let mut codes = Vec::with_capacity(self.layers.len());
        for layer in &mut self.layers {
            let indices = layer.encode(&residual, stream)?;
            let quantized = layer.decode_one(&indices, stream)?;
            residual = residual.subtract(&quantized, stream)?;
            codes.push(indices);
        }
        Ok(stack_axis(&codes, 1, stream)?)
    }

    fn decode(&mut self, codes: &Array, stream: &Stream) -> Result<Array, Error> {
        if codes.dim(1) != self.layers.len() as i32 {
            return Err(Error::InvalidShape(format!(
                "Mimi RVQ expected {} codebooks, got {:?}",
                self.layers.len(),
                codes.shape()
            )));
        }
        let mut out: Option<Array> = None;
        for (index, layer) in self.layers.iter_mut().enumerate() {
            let code = codes.try_index_device((.., index as i32, ..), stream)?;
            let quantized = layer.decode_one(&code, stream)?;
            out = Some(match out {
                None => quantized,
                Some(prev) => prev.add(&quantized, stream)?,
            });
        }
        out.ok_or_else(|| Error::InvalidShape("Mimi RVQ has no layers".into()))
    }
}

/// Single vector-quantization layer.
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = safemlx)]
pub struct VectorQuantization {
    /// Euclidean codebook.
    #[param]
    pub _codebook: EuclideanCodebook,
}

impl VectorQuantization {
    fn unloaded(dim: i32, bins: i32, stream: &Stream) -> Result<Self, Error> {
        Ok(Self {
            _codebook: EuclideanCodebook::unloaded(dim, bins, stream)?,
        })
    }

    fn encode(&mut self, latent: &Array, stream: &Stream) -> Result<Array, Error> {
        let latent = latent.swap_axes(1, 2, stream)?;
        self._codebook.encode(&latent, stream)
    }

    fn decode_one(&mut self, codes: &Array, stream: &Stream) -> Result<Array, Error> {
        self._codebook
            .decode(codes, stream)?
            .swap_axes(1, 2, stream)
            .map_err(Into::into)
    }
}

/// Euclidean codebook backed by EMA cluster statistics.
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = safemlx)]
pub struct EuclideanCodebook {
    /// Checkpoint initialization flag.
    #[param]
    pub _initialized: Param<Array>,
    /// EMA cluster usage.
    #[param]
    pub cluster_usage: Param<Array>,
    /// EMA embedding sum.
    #[param]
    pub embedding_sum: Param<Array>,
}

impl EuclideanCodebook {
    fn unloaded(dim: i32, bins: i32, stream: &Stream) -> Result<Self, Error> {
        Ok(Self {
            _initialized: Param::<Array>::unloaded(&[1], Dtype::Float32, stream)?,
            cluster_usage: Param::<Array>::unloaded(&[bins], Dtype::Float32, stream)?,
            embedding_sum: Param::<Array>::unloaded(&[bins, dim], Dtype::Float32, stream)?,
        })
    }

    fn embedding(&self, stream: &Stream) -> Result<Array, Error> {
        let usage = maximum(
            self.cluster_usage.as_ref(),
            Array::from_f32(EPSILON),
            stream,
        )?
        .expand_dims(1, stream)?;
        Ok(self.embedding_sum.as_ref().divide(&usage, stream)?)
    }

    fn encode(&self, latent_btd: &Array, stream: &Stream) -> Result<Array, Error> {
        if latent_btd.shape().len() != 3 {
            return Err(Error::InvalidShape(format!(
                "Mimi codebook encode expects [batch, frames, dim], got {:?}",
                latent_btd.shape()
            )));
        }
        let batch = latent_btd.dim(0);
        let frames = latent_btd.dim(1);
        let dim = latent_btd.dim(2);
        let flat = latent_btd.reshape(&[batch * frames, dim], stream)?;
        let embedding = self.embedding(stream)?;
        let x2 = sum_axis(&flat.square(stream)?, -1, true, stream)?;
        let e2 = sum_axis(&embedding.square(stream)?, -1, false, stream)?.expand_dims(0, stream)?;
        let dot = matmul(&flat, &embedding.transpose(stream)?, stream)?;
        let dists = x2
            .add(&e2, stream)?
            .subtract(&dot.multiply(Array::from_f32(2.0), stream)?, stream)?;
        Ok(
            argmin_axis!(&dists, -1, keep_dims = false, stream = stream)?
                .reshape(&[batch, frames], stream)?,
        )
    }

    fn decode(&self, codes: &Array, stream: &Stream) -> Result<Array, Error> {
        if codes.shape().len() != 2 {
            return Err(Error::InvalidShape(format!(
                "Mimi codebook decode expects [batch, frames], got {:?}",
                codes.shape()
            )));
        }
        let batch = codes.dim(0);
        let frames = codes.dim(1);
        let embedding = self.embedding(stream)?;
        let flat = codes.reshape(&[batch * frames], stream)?;
        Ok(embedding
            .try_index_device(flat, stream)?
            .reshape(&[batch, frames, embedding.dim(1)], stream)?)
    }
}

/// Bias-free 1x1 convolution over `[batch, channels, frames]` tensors.
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = safemlx)]
pub struct Conv1x1NoBias {
    /// Weight shaped `[out_channels, in_channels, 1]`.
    #[param]
    pub weight: Param<Array>,
}

impl Conv1x1NoBias {
    fn unloaded(in_channels: i32, out_channels: i32, stream: &Stream) -> Result<Self, Error> {
        Ok(Self {
            weight: Param::<Array>::unloaded(
                &[out_channels, in_channels, 1],
                Dtype::Float32,
                stream,
            )?,
        })
    }

    fn forward(&self, latent: &Array, stream: &Stream) -> Result<Array, Error> {
        if latent.shape().len() != 3 {
            return Err(Error::InvalidShape(format!(
                "Mimi 1x1 projection expects [batch, channels, frames], got {:?}",
                latent.shape()
            )));
        }
        let x = latent.swap_axes(1, 2, stream)?;
        let weight = self.weight.as_ref().squeeze_axes(&[-1], stream)?;
        Ok(matmul(&x, &weight.transpose(stream)?, stream)?.swap_axes(1, 2, stream)?)
    }
}

fn validate_latent(latent: &Array) -> Result<(), Error> {
    if latent.shape().len() != 3 || latent.dim(1) != 512 {
        return Err(Error::InvalidShape(format!(
            "Mimi latent frames must have shape [batch, 512, frames], got {:?}",
            latent.shape()
        )));
    }
    Ok(())
}

fn validate_pcm(pcm: &Array) -> Result<(), Error> {
    if pcm.shape().len() != 3 || pcm.dim(1) != 1 {
        return Err(Error::InvalidShape(format!(
            "Mimi PCM must have shape [batch, 1, samples], got {:?}",
            pcm.shape()
        )));
    }
    Ok(())
}

fn validate_codes(codes: &Array, max_codebooks: i32) -> Result<(), Error> {
    if codes.shape().len() != 3 || codes.dim(1) <= 0 || codes.dim(1) > max_codebooks {
        return Err(Error::InvalidShape(format!(
            "Mimi codes must have shape [batch, 1..={max_codebooks}, frames], got {:?}",
            codes.shape()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{transform_decoder_key, AudioTokenizer, Config, Mimi};
    use safemlx::{
        ops::{concatenate_axis, indexing::TryIndexOp},
        transforms::eval,
        Array, Device, DeviceType, ExecutionContext,
    };

    #[test]
    fn checkpoint_quantizer_keys_keep_the_model_root() {
        let key = "quantizer.rvq_first.vq.layers.0._codebook.embedding_sum";
        assert_eq!(transform_decoder_key(key).as_deref(), Some(key));
    }

    #[test]
    fn v0_1_config_defaults_to_moshi_active_codebooks() {
        let cfg = Config::v0_1(None);
        assert_eq!(cfg.sample_rate, 24_000.0);
        assert_eq!(cfg.frame_rate, 12.5);
        assert_eq!(cfg.num_codebooks, 16);
        assert_eq!(cfg.total_codebooks, 32);
        assert_eq!(cfg.bins, 2_048);
    }

    #[test]
    #[ignore = "requires SAFEMLX_MIMI_PATH with a released Mimi safetensors checkpoint and Metal"]
    fn local_mimi_checkpoint_encode_decode_smoke() {
        let path = std::env::var("SAFEMLX_MIMI_PATH")
            .expect("SAFEMLX_MIMI_PATH must point to a Mimi safetensors checkpoint");
        let ctx = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let stream = ctx.stream();
        let mut mimi = Mimi::load(path, Some(8), stream).unwrap();
        let cfg = mimi.config();
        assert_eq!(cfg.codebooks, 8);
        assert_eq!(cfg.cardinality, 2_048);

        let codes = Array::zeros::<i32>(&[1, 8, 2], stream).unwrap();
        let latent = mimi.decode_latent(&codes, stream).unwrap();
        assert_eq!(latent.shape(), &[1, 512, 2]);
        let recoded = mimi.encode_latent(&latent, stream).unwrap();
        assert_eq!(recoded.shape(), &[1, 8, 2]);
        let pcm = mimi.decode(&codes, stream).unwrap();
        assert_eq!(pcm.shape(), &[1, 1, 3840]);
        let alternate_codes = Array::ones::<i32>(&[1, 8, 2], stream).unwrap();
        let alternate_pcm = mimi.decode(&alternate_codes, stream).unwrap();
        eval([&pcm, &alternate_pcm]).unwrap();
        stream.synchronize().unwrap();
        let pcm_values = pcm.evaluated().unwrap();
        let alternate_values = alternate_pcm.evaluated().unwrap();
        let difference = pcm_values
            .as_slice::<f32>()
            .iter()
            .zip(alternate_values.as_slice::<f32>())
            .map(|(left, right)| (left - right).abs())
            .sum::<f32>();
        assert!(difference > 1e-3, "Mimi decode ignored token values");
        let encoded = mimi.encode(&pcm, stream).unwrap();
        assert_eq!(encoded.shape(), &[1, 8, 2]);

        // PyTorch Mimi oracle for x[n] = ((n mod 17) - 8) / 64. This catches
        // architecture drift that a shape-only checkpoint smoke test cannot.
        let parity_pcm = (0..7680)
            .map(|sample| ((sample % 17) as f32 - 8.0) / 64.0)
            .collect::<Vec<_>>();
        let parity_pcm = Array::from_slice(&parity_pcm, &[1, 1, 7680])
            .copy(stream)
            .unwrap();
        let actual_codes = mimi.encode(&parity_pcm, stream).unwrap();
        let expected_codes = Array::from_slice(
            &[
                1049, 605, 1964, 1964, 74, 712, 712, 712, 1441, 1441, 1441, 1441, 1820, 1820, 1820,
                1820, 1711, 1711, 1711, 1711, 1386, 818, 818, 1418, 127, 755, 755, 127, 130, 1228,
                1228, 1115,
            ],
            &[1, 8, 4],
        )
        .copy(stream)
        .unwrap();
        assert!(
            actual_codes
                .all_close(&expected_codes, 0.0, 0.0, None, stream)
                .unwrap()
                .item::<bool>(stream),
            "Mimi encode tokens differ from the released PyTorch checkpoint oracle"
        );

        mimi.reset_encode_state();
        let encoded_first = mimi
            .encode_step(
                &pcm.try_index_device((.., .., 0..1920), stream).unwrap(),
                stream,
            )
            .unwrap()
            .expect("first PCM frame should encode to one Mimi frame");
        let encoded_second = mimi
            .encode_step(
                &pcm.try_index_device((.., .., 1920..3840), stream).unwrap(),
                stream,
            )
            .unwrap()
            .expect("second PCM frame should encode to one Mimi frame");
        assert_eq!(encoded_first.shape(), &[1, 8]);
        assert_eq!(encoded_second.shape(), &[1, 8]);

        mimi.reset_decode_state();
        let first = mimi
            .decode_step(
                &codes.try_index_device((.., .., 0), stream).unwrap(),
                stream,
            )
            .unwrap();
        let second = mimi
            .decode_step(
                &codes.try_index_device((.., .., 1), stream).unwrap(),
                stream,
            )
            .unwrap();
        assert_eq!(first.shape(), &[1, 1, 1920]);
        assert_eq!(second.shape(), &[1, 1, 1920]);
        let streamed = concatenate_axis(&[first, second], 2, stream).unwrap();
        assert_eq!(streamed.shape(), pcm.shape());
    }
}
