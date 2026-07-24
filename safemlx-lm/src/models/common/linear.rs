//! Linear layers, embeddings, and language-model output heads.

use safemlx::{
    builder::Builder, error::Exception, module::Module, nn, quantization::MaybeQuantized, Array,
    Dtype, Stream,
};

use crate::quantization::WeightQuantization;

/// Builds an initialized untied language-model head.
pub fn build_lm_head(hidden_size: i32, vocab_size: i32) -> Result<nn::Linear, Exception> {
    nn::LinearBuilder::new(hidden_size, vocab_size)
        .bias(false)
        .build()
}

/// Builds an unloaded untied language-model head.
pub fn build_unloaded_lm_head(
    hidden_size: i32,
    vocab_size: i32,
    stream: &Stream,
) -> Result<nn::Linear, Exception> {
    nn::Linear::unloaded(hidden_size, vocab_size, false, Dtype::Float32, stream)
}

/// Builds an initialized language-model head wrapped for optional quantization.
pub fn build_maybe_quantized_lm_head(
    hidden_size: i32,
    vocab_size: i32,
) -> Result<MaybeQuantized<nn::Linear>, Exception> {
    Ok(MaybeQuantized::Original(build_lm_head(
        hidden_size,
        vocab_size,
    )?))
}

/// Builds an unloaded language-model head wrapped for optional quantization.
pub fn build_unloaded_maybe_quantized_lm_head(
    hidden_size: i32,
    vocab_size: i32,
    stream: &Stream,
) -> Result<MaybeQuantized<nn::Linear>, Exception> {
    unloaded_maybe_quantized_linear(hidden_size, vocab_size, false, None, stream)
}

/// Creates an unloaded linear using the standard dense or affine parameter tree.
pub fn unloaded_maybe_quantized_linear(
    input_dims: i32,
    output_dims: i32,
    bias: bool,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
) -> Result<MaybeQuantized<nn::Linear>, Exception> {
    match quantization {
        Some(WeightQuantization::GgufIQuant { ggml_type, endian }) => {
            Ok(MaybeQuantized::Quantized(nn::QuantizedLinear::unloaded_iq(
                input_dims,
                output_dims,
                ggml_type,
                endian,
                bias,
                stream,
            )?))
        }
        Some(config) => Ok(MaybeQuantized::Quantized(
            nn::QuantizedLinear::unloaded_with_mode(
                input_dims,
                output_dims,
                config.group_size(),
                config.bits(),
                config.mode(),
                bias,
                stream,
            )?,
        )),
        None => Ok(MaybeQuantized::Original(nn::Linear::unloaded(
            input_dims,
            output_dims,
            bias,
            Dtype::Float32,
            stream,
        )?)),
    }
}

/// Creates an unloaded embedding using the standard dense or affine parameter tree.
pub fn unloaded_maybe_quantized_embedding(
    embedding_count: i32,
    dimensions: i32,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
) -> Result<MaybeQuantized<nn::Embedding>, Exception> {
    match quantization {
        Some(WeightQuantization::GgufIQuant { ggml_type, endian }) => Ok(
            MaybeQuantized::Quantized(nn::QuantizedEmbedding::unloaded_iq(
                embedding_count,
                dimensions,
                ggml_type,
                endian,
                stream,
            )?),
        ),
        Some(config) => Ok(MaybeQuantized::Quantized(
            nn::QuantizedEmbedding::unloaded_with_mode(
                embedding_count,
                dimensions,
                config.group_size(),
                config.bits(),
                config.mode(),
                stream,
            )?,
        )),
        None => Ok(MaybeQuantized::Original(nn::Embedding::unloaded(
            embedding_count,
            dimensions,
            Dtype::Float32,
            stream,
        )?)),
    }
}

/// Builds an unloaded language-model head with optional affine quantization.
pub fn build_unloaded_maybe_quantized_lm_head_with_quantization(
    hidden_size: i32,
    vocab_size: i32,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
) -> Result<MaybeQuantized<nn::Linear>, Exception> {
    unloaded_maybe_quantized_linear(hidden_size, vocab_size, false, quantization, stream)
}

/// Projects hidden states to logits, using tied embeddings when `lm_head` is absent.
pub fn project_logits_maybe_quantized(
    lm_head: &mut Option<MaybeQuantized<nn::Linear>>,
    embed_tokens: &mut MaybeQuantized<nn::Embedding>,
    hidden_states: &Array,
    stream: &Stream,
) -> Result<Array, Exception> {
    match lm_head.as_mut() {
        Some(lm_head) => lm_head.forward(hidden_states, stream),
        None => match embed_tokens {
            MaybeQuantized::Original(embed_tokens) => embed_tokens.as_linear(hidden_states, stream),
            MaybeQuantized::Quantized(q_embed_tokens) => {
                q_embed_tokens.as_linear(hidden_states, stream)
            }
        },
    }
}

/// Projects hidden states to logits for dense, non-quantized heads.
pub fn project_logits_dense(
    lm_head: &mut Option<nn::Linear>,
    embed_tokens: &nn::Embedding,
    hidden_states: &Array,
    stream: &Stream,
) -> Result<Array, Exception> {
    match lm_head.as_mut() {
        Some(lm_head) => lm_head.forward(hidden_states, stream),
        None => embed_tokens.as_linear(hidden_states, stream),
    }
}
