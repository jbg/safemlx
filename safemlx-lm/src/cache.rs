use safemlx::{
    error::Exception,
    ops::{
        concatenate_axis,
        indexing::{TryIndexMutOp, TryIndexOp},
        zeros_dtype,
    },
    Array, Stream,
};

// TODO: somehow move quantized methods to a separate trait?
/// A per-layer attention key/value cache.
pub trait KeyValueCache {
    /// Returns whether this cache stores quantized keys and values.
    fn is_quantized(&self) -> bool {
        false
    }

    /// Returns the group size used for quantization. `None` if not quantized.
    fn group_size(&self) -> Option<i32> {
        None
    }

    /// Returns the number of bits used for quantization. `None` if not quantized.
    fn bits(&self) -> Option<i32> {
        None
    }

    /// Returns the current sequence offset represented by the cache.
    fn offset(&self) -> i32;

    /// Returns the maximum retained sequence length for sliding-window caches.
    fn max_size(&self) -> Option<i32>;

    /// Adds the newest keys and values and returns the full keys and values to attend over.
    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
        stream: &Stream,
    ) -> Result<(Array, Array), Exception>;
}

impl<T> KeyValueCache for &'_ mut T
where
    T: KeyValueCache,
{
    fn is_quantized(&self) -> bool {
        T::is_quantized(self)
    }

    fn group_size(&self) -> Option<i32> {
        T::group_size(self)
    }

    fn bits(&self) -> Option<i32> {
        T::bits(self)
    }

    fn offset(&self) -> i32 {
        T::offset(self)
    }

    fn max_size(&self) -> Option<i32> {
        T::max_size(self)
    }

    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
        stream: &Stream,
    ) -> Result<(Array, Array), Exception> {
        T::update_and_fetch(self, keys, values, stream)
    }
}

/// A cache that appends all key/value states along the sequence axis.
#[derive(Debug, Clone, Default)]
pub struct ConcatKeyValueCache {
    keys: Option<Array>,
    values: Option<Array>,
    offset: i32,
    length: i32,
    capacity: i32,
    step: i32,
    max_size: Option<i32>,
}

impl ConcatKeyValueCache {
    /// Creates an empty concatenating key/value cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates an empty concatenating cache that retains at most `max_size` tokens.
    pub fn new_with_max_size(max_size: i32) -> Self {
        Self {
            max_size: Some(max_size),
            ..Self::default()
        }
    }

    /// Creates a bounded cache whose backing arrays grow in `step`-token chunks.
    ///
    /// Chunked growth avoids rebuilding the full retained cache on every decode
    /// step. The returned key/value arrays are still sliced to the logical
    /// sequence length, so attention semantics are unchanged.
    pub(crate) fn new_with_max_size_and_step(max_size: i32, step: i32) -> Self {
        Self {
            max_size: Some(max_size),
            step: step.max(1),
            ..Self::default()
        }
    }

    /// Truncates the cache to `len` tokens.
    pub fn truncate(&mut self, len: i32, stream: &Stream) -> Result<(), Exception> {
        if let Some(keys) = self.keys.take() {
            self.keys = Some(keys.try_index_device((.., .., ..len, ..), stream)?);
        }
        if let Some(values) = self.values.take() {
            self.values = Some(values.try_index_device((.., .., ..len, ..), stream)?);
        }
        self.offset = len;
        self.length = len;
        self.capacity = len;
        Ok(())
    }

    /// Returns the arrays currently retained by the cache.
    pub fn arrays(&self) -> impl Iterator<Item = &Array> {
        self.keys.iter().chain(self.values.iter())
    }

    /// Clears cached arrays while preserving cache configuration.
    pub fn clear(&mut self) {
        self.keys = None;
        self.values = None;
        self.offset = 0;
        self.length = 0;
        self.capacity = 0;
    }
}

impl ConcatKeyValueCache {
    fn grown_capacity(&self, required: i32) -> i32 {
        let step = self.step.max(1);
        let chunks = (required + step - 1) / step;
        let capacity = chunks * step;
        self.max_size
            .map_or(capacity, |max_size| capacity.min(max_size))
    }

    fn padded(array: &Array, capacity: i32, stream: &Stream) -> Result<Array, Exception> {
        let mut shape = array.shape().to_vec();
        let sequence_axis = shape.len() - 2;
        shape[sequence_axis] = capacity;
        zeros_dtype(&shape, array.dtype(), stream)
    }

    fn logical_arrays(&self, stream: &Stream) -> Result<(Array, Array), Exception> {
        let keys = self
            .keys
            .as_ref()
            .expect("Keys cannot be None")
            .try_index_device((.., .., ..self.length, ..), stream)?;
        let values = self
            .values
            .as_ref()
            .expect("Values cannot be None")
            .try_index_device((.., .., ..self.length, ..), stream)?;
        Ok((keys, values))
    }
}

impl KeyValueCache for ConcatKeyValueCache {
    fn offset(&self) -> i32 {
        self.offset
    }

    fn max_size(&self) -> Option<i32> {
        self.max_size
    }

    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
        stream: &Stream,
    ) -> Result<(Array, Array), Exception> {
        let new_tokens = keys.dim(-2);
        self.offset += new_tokens;

        if self.step <= 1 {
            match (self.keys.take(), self.values.take()) {
                (Some(k), Some(v)) => {
                    self.keys = Some(concatenate_axis(&[k, keys], -2, stream)?);
                    self.values = Some(concatenate_axis(&[v, values], -2, stream)?);
                }
                _ => {
                    self.keys = Some(keys);
                    self.values = Some(values);
                }
            }
            if let Some(max_size) = self.max_size {
                let length = self.keys.as_ref().expect("Keys cannot be None").dim(-2);
                if length > max_size {
                    let start = length - max_size;
                    self.keys = Some(
                        self.keys
                            .take()
                            .expect("Keys cannot be None")
                            .try_index_device((.., .., start.., ..), stream)?,
                    );
                    self.values = Some(
                        self.values
                            .take()
                            .expect("Values cannot be None")
                            .try_index_device((.., .., start.., ..), stream)?,
                    );
                }
            }
            self.length = self.keys.as_ref().expect("Keys cannot be None").dim(-2);
            self.capacity = self.length;
            return Ok((
                self.keys.clone().expect("Keys cannot be None"),
                self.values.clone().expect("Values cannot be None"),
            ));
        }

        let required = self.length + new_tokens;

        if let Some(max_size) = self.max_size {
            if required > max_size {
                if self.keys.is_none() {
                    let start = new_tokens - max_size;
                    self.keys = Some(keys.try_index_device((.., .., start.., ..), stream)?);
                    self.values = Some(values.try_index_device((.., .., start.., ..), stream)?);
                    self.length = max_size;
                    self.capacity = max_size;
                    return self.logical_arrays(stream);
                }
                let (old_keys, old_values) = self.logical_arrays(stream)?;
                let combined_keys = concatenate_axis(&[old_keys, keys], -2, stream)?;
                let combined_values = concatenate_axis(&[old_values, values], -2, stream)?;
                let start = required - max_size;
                self.keys = Some(combined_keys.try_index_device((.., .., start.., ..), stream)?);
                self.values =
                    Some(combined_values.try_index_device((.., .., start.., ..), stream)?);
                self.length = max_size;
                self.capacity = max_size;
                return self.logical_arrays(stream);
            }
        }

        if self.keys.is_none() {
            self.capacity = self.grown_capacity(required);
            if self.capacity == required {
                self.keys = Some(keys);
                self.values = Some(values);
                self.length = required;
                return self.logical_arrays(stream);
            }
            self.keys = Some(Self::padded(&keys, self.capacity, stream)?);
            self.values = Some(Self::padded(&values, self.capacity, stream)?);
        } else if required > self.capacity {
            let new_capacity = self.grown_capacity(required);
            let padding = new_capacity - self.capacity;
            let key_padding = Self::padded(&keys, padding, stream)?;
            let value_padding = Self::padded(&values, padding, stream)?;
            self.keys = Some(concatenate_axis(
                &[self.keys.take().expect("Keys cannot be None"), key_padding],
                -2,
                stream,
            )?);
            self.values = Some(concatenate_axis(
                &[
                    self.values.take().expect("Values cannot be None"),
                    value_padding,
                ],
                -2,
                stream,
            )?);
            self.capacity = new_capacity;
        }

        self.keys
            .as_mut()
            .expect("Keys cannot be None")
            .try_index_mut_device((.., .., self.length..required, ..), &keys, stream)?;
        self.values
            .as_mut()
            .expect("Values cannot be None")
            .try_index_mut_device((.., .., self.length..required, ..), &values, stream)?;
        self.length = required;
        self.logical_arrays(stream)
    }
}

/// Placeholder for a future generic key/value cache implementation.
pub struct DefaultKeyValueCache {}

#[cfg(test)]
mod tests {
    use super::{ConcatKeyValueCache, KeyValueCache};
    use safemlx::{ops::indexing::TryIndexOp, Array, Device, DeviceType, ExecutionContext};

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn chunked_cache_grows_by_steps_and_preserves_sliding_values() {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = context.stream();
        let mut cache = ConcatKeyValueCache::new_with_max_size_and_step(8, 4);

        let mut fetched = None;
        for value in 0..10 {
            let keys =
                Array::full::<f32>(&[1, 1, 1, 2], Array::from_f32(value as f32), stream).unwrap();
            let values = keys.clone();
            fetched = Some(cache.update_and_fetch(keys, values, stream).unwrap().0);
            if value == 0 {
                assert_eq!(cache.capacity, 4);
            } else if value == 4 {
                assert_eq!(cache.capacity, 8);
            }
        }

        let fetched = fetched.unwrap();
        assert_eq!(cache.offset(), 10);
        assert_eq!(fetched.shape(), &[1, 1, 8, 2]);
        assert_eq!(
            fetched
                .try_index_device((0, 0, 0, 0), stream)
                .unwrap()
                .item::<f32>(stream),
            2.0
        );
        assert_eq!(
            fetched
                .try_index_device((0, 0, -1, 0), stream)
                .unwrap()
                .item::<f32>(stream),
            9.0
        );
    }
}
