use safemlx::{
    error::Exception,
    ops::{concatenate_axis, indexing::TryIndexOp},
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
}

impl ConcatKeyValueCache {
    /// Creates an empty concatenating key/value cache.
    pub fn new() -> Self {
        Self::default()
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
        Ok(())
    }

    /// Returns the arrays currently retained by the cache.
    pub fn arrays(&self) -> impl Iterator<Item = &Array> {
        self.keys.iter().chain(self.values.iter())
    }
}

impl KeyValueCache for ConcatKeyValueCache {
    fn offset(&self) -> i32 {
        self.offset
    }

    fn max_size(&self) -> Option<i32> {
        None
    }

    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
        stream: &Stream,
    ) -> Result<(Array, Array), Exception> {
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
        let shape = self.keys.as_ref().expect("Keys cannot be None").shape();
        self.offset = shape[shape.len() - 2];

        Ok((
            self.keys.clone().expect("Keys cannot be None"),
            self.values.clone().expect("Values cannot be None"),
        ))
    }
}

/// Placeholder for a future generic key/value cache implementation.
pub struct DefaultKeyValueCache {}
