//! Shared tensor helpers used by model implementations.

use safemlx::{
    arange,
    error::Exception,
    fast::ScaledDotProductAttentionMask,
    ops::{
        expand_dims,
        indexing::{NewAxis, TryIndexOp},
        quantized_matmul, reshape, softmax_axis,
    },
    Array, Dtype, Stream,
};

use crate::cache::KeyValueCache;

/// Rotary position-embedding variants and initialization.
pub mod rope;
/// Tokenizer-related re-exports and helpers.
pub mod tokenizer;

#[allow(unused_macros)]
macro_rules! try_unwrap {
    ($expr:expr) => {
        match $expr {
            core::result::Result::Ok(val) => val,
            core::result::Result::Err(e) => return Some(Err(e.into())),
        }
    };
}

// def quantized_scaled_dot_product_attention(
//     queries: mx.array,
//     q_keys: tuple[mx.array, mx.array, mx.array],
//     q_values: tuple[mx.array, mx.array, mx.array],
//     scale: float,
//     mask: Optional[mx.array],
//     group_size: int = 64,
//     bits: int = 8,
// ) -> mx.array:
//     B, n_q_heads, L, D = queries.shape
//     n_kv_heads = q_keys[0].shape[-3]
//     n_repeats = n_q_heads // n_kv_heads

//     queries *= scale

//     if n_repeats > 1:
//         queries = mx.reshape(queries, (B, n_kv_heads, n_repeats, L, D))
//         q_keys = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_keys)
//         q_values = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_values)

//     scores = mx.quantized_matmul(
//         queries, *q_keys, transpose=True, group_size=group_size, bits=bits
//     )
//     if mask is not None:
//         if isinstance(mask, str):
//             qL, kL = scores.shape[-2:]
//             q_indices = mx.arange(kL - qL, kL)
//             k_indices = mx.arange(kL)
//             mask = q_indices[:, None] >= k_indices[None]
//         if mask.dtype == mx.bool_:
//             scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
//         else:
//             scores += mask
//     scores = mx.softmax(scores, axis=-1, precise=True)
//     out = mx.quantized_matmul(
//         scores, *q_values, transpose=False, group_size=group_size, bits=bits
//     )

//     if n_repeats > 1:
//         out = mx.reshape(out, (B, n_q_heads, L, D))

//     return out

fn index_out_of_bound_exception() -> Exception {
    Exception::custom("index out of bound")
}

#[allow(non_snake_case, clippy::too_many_arguments)]
pub(crate) fn quantized_scaled_dot_product_attention(
    queries: Array,
    mut q_keys: QuantizedKeys,
    mut q_values: QuantizedValues,
    scale: f32,
    mask: Option<&Array>,
    group_size: i32,
    bits: i32,
    stream: &Stream,
) -> Result<Array, Exception> {
    let q_shape = queries.shape();
    let B = *q_shape.first().ok_or_else(index_out_of_bound_exception)?;
    let n_q_heads = *q_shape.get(1).ok_or_else(index_out_of_bound_exception)?;
    let L = *q_shape.get(2).ok_or_else(index_out_of_bound_exception)?;
    let D = *q_shape.get(3).ok_or_else(index_out_of_bound_exception)?;

    let q_keys_shape = q_keys.keys.shape();
    let n_kv_heads = q_keys_shape[q_keys_shape.len() - 3];
    let n_repeats = n_q_heads / n_kv_heads;

    let mut queries = queries.multiply(Array::from_f32(scale), stream)?;

    if n_repeats > 1 {
        queries = reshape(&queries, &[B, n_kv_heads, n_repeats, L, D], stream)?;

        q_keys.keys = expand_dims(q_keys.keys, -3, stream)?;
        q_keys.scales = expand_dims(q_keys.scales, -3, stream)?;
        q_keys.biases = expand_dims(q_keys.biases, -3, stream)?;

        q_values.values = expand_dims(q_values.values, -3, stream)?;
        q_values.scales = expand_dims(q_values.scales, -3, stream)?;
        q_values.biases = expand_dims(q_values.biases, -3, stream)?;
    }

    let mut scores = quantized_matmul(
        &queries,
        q_keys.keys,
        q_keys.scales,
        &q_keys.biases,
        true,
        group_size,
        bits,
        stream,
    )?;

    if let Some(mask) = mask {
        // TODO: handle str type mask

        if mask.dtype() == Dtype::Bool {
            let finfo_min = scores.dtype().finfo_min()?;
            scores =
                safemlx::ops::r#where(mask, scores, Array::from_f32(finfo_min as f32), stream)?;
        } else {
            scores = scores.add(mask, stream)?;
        }
    }
    scores = softmax_axis(scores, -1, true, stream)?;
    let mut out = quantized_matmul(
        scores,
        q_values.values,
        q_values.scales,
        &q_values.biases,
        false,
        group_size,
        bits,
        stream,
    )?;

    if n_repeats > 1 {
        out = reshape(out, &[B, n_q_heads, L, D], stream)?;
    }

    Ok(out)
}

/// Quantized key tensor and its dequantization parameters.
pub struct QuantizedKeys {
    /// Packed quantized keys.
    pub keys: Array,
    /// Per-group quantization scales.
    pub scales: Array,
    /// Per-group quantization biases.
    pub biases: Array,
}

/// Quantized value tensor and its dequantization parameters.
pub struct QuantizedValues {
    /// Packed quantized values.
    pub values: Array,
    /// Per-group quantization scales.
    pub scales: Array,
    /// Per-group quantization biases.
    pub biases: Array,
}

/// Either original or quantized attention keys.
pub enum MaybeQuantizedKeys {
    /// Floating-point keys.
    Original(Array),
    /// Quantized keys plus scale metadata.
    Quantized(QuantizedKeys),
}

impl From<Array> for MaybeQuantizedKeys {
    fn from(value: Array) -> Self {
        Self::Original(value)
    }
}

impl From<QuantizedKeys> for MaybeQuantizedKeys {
    fn from(value: QuantizedKeys) -> Self {
        Self::Quantized(value)
    }
}

/// Either original or quantized attention values.
pub enum MaybeQuantizedValues {
    /// Floating-point values.
    Original(Array),
    /// Quantized values plus scale metadata.
    Quantized(QuantizedValues),
}

impl From<Array> for MaybeQuantizedValues {
    fn from(value: Array) -> Self {
        Self::Original(value)
    }
}

impl From<QuantizedValues> for MaybeQuantizedValues {
    fn from(value: QuantizedValues) -> Self {
        Self::Quantized(value)
    }
}

pub(crate) fn scaled_dot_product_attention<C>(
    queries: Array,
    keys: impl Into<MaybeQuantizedKeys>,
    values: impl Into<MaybeQuantizedValues>,
    cache: Option<C>,
    scale: f32,
    mask: Option<&Array>,
    stream: &Stream,
) -> Result<Array, Exception>
where
    C: KeyValueCache,
{
    let keys = keys.into();
    let values = values.into();

    if let Some(cache) = cache {
        if cache.is_quantized() {
            let group_size = cache
                .group_size()
                .ok_or_else(|| Exception::custom("Cache is quantized but group size is not set"))?;
            let bits = cache
                .bits()
                .ok_or_else(|| Exception::custom("Cache is quantized but bits are not set"))?;

            let (keys, values) = match (keys, values) {
                (MaybeQuantizedKeys::Quantized(keys), MaybeQuantizedValues::Quantized(values)) => {
                    (keys, values)
                }
                _ => {
                    return Err(Exception::custom(
                        "Both keys and values must be quantized when KV cache is quantized",
                    ));
                }
            };

            return quantized_scaled_dot_product_attention(
                queries, keys, values, scale, mask, group_size, bits, stream,
            );
        }
    }

    let (keys, values) = match (keys, values) {
        (MaybeQuantizedKeys::Original(keys), MaybeQuantizedValues::Original(values)) => {
            (keys, values)
        }
        _ => {
            return Err(Exception::custom(
                "Both keys and values must NOT be quantized when KV cache is NOT quantized",
            ));
        }
    };

    safemlx::fast::scaled_dot_product_attention(
        queries,
        keys,
        values,
        scale,
        mask.map(ScaledDotProductAttentionMask::Array),
        None,
        stream,
    )
}

#[derive(Debug, Clone)]
pub(crate) enum AttentionMask {
    Array(Array),
    Causal,
}

impl<'a> From<&'a AttentionMask> for ScaledDotProductAttentionMask<'a> {
    fn from(mask: &'a AttentionMask) -> Self {
        match mask {
            AttentionMask::Array(array) => ScaledDotProductAttentionMask::Array(array),
            AttentionMask::Causal => ScaledDotProductAttentionMask::Causal,
        }
    }
}

#[allow(non_snake_case)]
pub(crate) fn create_causal_mask(
    N: i32,
    offset: Option<i32>,
    window_size: Option<i32>,
    lengths: Option<Array>,
    stream: &Stream,
) -> Result<Array, Exception> {
    let offset = offset.unwrap_or(0);

    let rinds = arange!(stop = offset + N, stream = stream)?;
    let linds = arange!(start = offset, stop = offset + N, stream = stream)?;
    let linds = linds.try_index_device((.., NewAxis), stream)?;
    let rinds = rinds.try_index_device(NewAxis, stream)?;

    let mut mask = linds.ge(&rinds, stream)?;
    if let Some(window_size) = window_size {
        let rinds_window = rinds.add(Array::from_int(window_size), stream)?;
        mask = mask.logical_and(&linds.le(&rinds_window, stream)?, stream)?;
    }

    if let Some(lengths) = lengths {
        let lengths = lengths.try_index_device((.., NewAxis, NewAxis, NewAxis), stream)?;
        mask = mask.logical_and(&linds.lt(&lengths, stream)?, stream)?;
    }

    Ok(mask)
}

#[allow(non_snake_case)]
pub(crate) fn create_attention_mask<C>(
    h: &Array,
    cache: &[Option<C>],
    return_array: Option<bool>,
    stream: &Stream,
) -> Result<Option<AttentionMask>, Exception>
where
    C: KeyValueCache,
{
    let mut return_array = return_array.unwrap_or(false);
    let T = h.shape()[1];
    if T > 1 {
        let mut offset = 0;
        let mut window_size = None;
        if let Some(c) = cache.first().and_then(|c| c.as_ref()) {
            offset = c.offset();
            if let Some(window_size_) = c.max_size() {
                window_size = Some(window_size_);
                offset = offset.min(window_size_);

                return_array = return_array || (offset + T) > window_size_;
            }
        }

        if return_array {
            create_causal_mask(T, Some(offset), window_size, None, stream)
                .map(AttentionMask::Array)
                .map(Some)
        } else {
            Ok(Some(AttentionMask::Causal))
        }
    } else {
        Ok(None)
    }
}

/// Builds an explicit causal mask for a fixed-size attention window.
///
/// `window_size` is the total number of key positions visible to a query,
/// including the query position itself. This matches Hugging Face Mistral's
/// sliding-window convention. Unlike the generic mask helper, this may return
/// a mask for a single decode token once a bounded cache is full, because the
/// cache temporarily exposes one extra old state during the update.
pub(crate) fn create_sliding_attention_mask<C>(
    h: &Array,
    cache: &[Option<C>],
    window_size: i32,
    stream: &Stream,
) -> Result<Option<Array>, Exception>
where
    C: KeyValueCache,
{
    if window_size <= 0 {
        return Err(Exception::custom(
            "sliding attention window must be positive",
        ));
    }

    let sequence = h.shape()[1];
    let retained_prefix = cache
        .first()
        .and_then(|cache| cache.as_ref())
        .map(|cache| {
            cache
                .max_size()
                .map_or_else(|| cache.offset(), |max_size| cache.offset().min(max_size))
        })
        .unwrap_or(0);
    let key_length = retained_prefix + sequence;

    if sequence == 1 && key_length <= window_size {
        return Ok(None);
    }

    create_causal_mask(
        sequence,
        Some(retained_prefix),
        Some(window_size - 1),
        None,
        stream,
    )
    .map(Some)
}

#[cfg(test)]
mod tests {
    use safemlx::{ops::indexing::TryIndexOp, Array, Device, DeviceType, ExecutionContext};

    use crate::cache::{KeyValueCache, SlidingKeyValueCache};

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn mistral_window_counts_the_current_token() {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = context.stream();
        let hidden = Array::from_slice(&[0.0f32; 4], &[1, 4, 1]);
        let cache: Vec<Option<SlidingKeyValueCache>> = Vec::new();
        let mask = super::create_sliding_attention_mask(&hidden, &cache, 2, stream)
            .unwrap()
            .unwrap();

        assert_eq!(mask.shape(), &[4, 4]);
        assert!(mask
            .try_index_device((3, 2), stream)
            .unwrap()
            .item::<bool>(stream));
        assert!(!mask
            .try_index_device((3, 1), stream)
            .unwrap()
            .item::<bool>(stream));
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn full_sliding_cache_masks_the_extra_decode_state() {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = context.stream();
        let mut cache = SlidingKeyValueCache::new(3);
        let states = Array::from_slice(&[0.0f32; 3], &[1, 1, 3, 1]);
        cache
            .update_and_fetch(states.clone(), states, stream)
            .unwrap();
        let hidden = Array::from_slice(&[0.0f32], &[1, 1, 1]);
        let mask = super::create_sliding_attention_mask(&hidden, &[Some(cache)], 3, stream)
            .unwrap()
            .unwrap();

        assert_eq!(mask.shape(), &[1, 4]);
        assert!(!mask
            .try_index_device((0, 0), stream)
            .unwrap()
            .item::<bool>(stream));
        assert!(mask
            .try_index_device((0, 3), stream)
            .unwrap()
            .item::<bool>(stream));
    }
}
