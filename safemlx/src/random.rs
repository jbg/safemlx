//! Collection of functions related to random number generation

use crate::ops::indexing::TryIndexOp;
use crate::utils::guard::Guarded;
use crate::utils::IntoOption;
use crate::{error::Exception, error::Result, Array, ArrayElement, Stream};
use mach_sys::mach_time;
use safemlx_internal_macros::generate_macro;
use std::borrow::Cow;

/// Random state for reproducible random number generation.
///
/// This struct holds the PRNG state and can be used with compiled functions
/// to properly track random state across JIT compilation boundaries.
///
/// # Compilation Support
///
/// `RandomState` implements `Updatable`, making it compatible with
/// `compile_with_state`. This is the Rust equivalent of Python's
/// `@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)`.
///
/// # Example
///
/// ```rust,no_run
/// # let stream = safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
/// use safemlx::random::RandomState;
/// use safemlx::transforms::compile::compile_with_state;
/// use safemlx::random::categorical;
/// use safemlx::Array;
///
/// let mut state = RandomState::with_seed(42).unwrap();
/// let logits = Array::zeros::<f32>(&[1, 10], &stream).unwrap();
/// let mut compiled = compile_with_state(
///     |state: &mut RandomState, x: &Array| {
///         let stream = safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
///         let key = state.next_key(&stream)?;
///         categorical(x, None, None, Some(&key), &stream)
///     },
///     None,
/// );
/// let result = compiled(&mut state, &logits).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct RandomState {
    state: Array,
}

impl RandomState {
    /// Create a new random state with a time-based seed.
    pub fn new() -> Result<Self> {
        let now = unsafe { mach_time::mach_approximate_time() };
        Ok(Self { state: key(now)? })
    }

    /// Create a new random state from a specific seed.
    ///
    /// Use this for reproducible random number generation.
    pub fn with_seed(seed: u64) -> Result<Self> {
        Ok(Self { state: key(seed)? })
    }

    /// Create a random state from an existing key array.
    ///
    /// The key must be a valid PRNG key (typically created via `random::key()`).
    pub fn from_key(key: Array) -> Self {
        Self { state: key }
    }

    /// Get the next random key, advancing the state.
    ///
    /// This splits the current state into two keys: one becomes the new state,
    /// and the other is returned for use in random operations.
    pub fn next_key(&mut self, stream: &Stream) -> Result<Array> {
        let next = split(&self.state, 2, stream)?;
        self.state = next.0;
        Ok(next.1)
    }

    /// Reseed the random state.
    pub fn seed(&mut self, seed: u64) -> Result<()> {
        self.state = key(seed)?;
        Ok(())
    }

    /// Get a reference to the underlying state array.
    ///
    /// This is useful for inspection or manual state management.
    pub fn as_array(&self) -> &Array {
        &self.state
    }

    /// Get a mutable reference to the underlying state array.
    ///
    /// # Note
    ///
    /// Modifying the state array directly may break the PRNG invariants.
    /// Prefer using `seed()` or `next_key()` instead.
    pub fn as_array_mut(&mut self) -> &mut Array {
        &mut self.state
    }
}

impl Default for RandomState {
    /// Creates a new `RandomState` with a time-based seed.
    ///
    /// # Panics
    ///
    /// Panics if the underlying PRNG key creation fails, which should not
    /// occur under normal conditions.
    fn default() -> Self {
        Self::new().expect("Failed to create default RandomState")
    }
}

impl crate::utils::Updatable for RandomState {
    fn updatable_states_len(&self) -> usize {
        1
    }

    fn updatable_states(&self) -> impl IntoIterator<Item = &Array> {
        std::iter::once(&self.state)
    }

    fn updatable_states_mut(&mut self) -> impl IntoIterator<Item = &mut Array> {
        std::iter::once(&mut self.state)
    }
}

/// Use the given key.
fn resolve<'a>(key: impl Into<Option<&'a Array>>) -> Result<Cow<'a, Array>> {
    key.into()
        .map(Cow::Borrowed)
        .ok_or_else(|| Exception::custom("random operations require an explicit PRNG key"))
}

/// Get a PRNG key from a seed.
///
/// Return a value that can be used as a PRNG key.  All ``random::*``
/// functions take an optional key -- this will let you control the
/// random number generation.
pub fn key(seed: u64) -> Result<Array> {
    Array::try_from_op(|res| unsafe { safemlx_sys::mlx_random_key(res, seed) })
}

/// Split a PRNG key into two keys and return a tuple.
pub fn split(
    key: impl AsRef<Array>,
    num: i32,
    stream: impl AsRef<Stream>,
) -> Result<(Array, Array)> {
    let stream = stream.as_ref();
    let keys = Array::try_from_op(|res| unsafe {
        safemlx_sys::mlx_random_split_num(res, key.as_ref().as_ptr(), num, stream.as_ptr())
    })?;

    Ok((
        keys.try_index_device(0, stream)?,
        keys.try_index_device(1, stream)?,
    ))
}

/// Generate uniformly distributed random numbers.
/// The values are sampled uniformly in the half-open interval `[lower, upper)`.
/// The lower and upper bound can be scalars or arrays and must be broadcastable to `shape`.
///
/// # Params
///
/// - `lower`: Lower bound of the distribution.
/// - `upper`: Upper bound of the distribution.
/// - `shape` (optional): Shape of the output. Default is `&[]`.
/// - `key` (optional): A PRNG key.
///
/// ```rust
/// # let stream = safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
/// let key = safemlx::random::key(0).unwrap();
///
/// // create an array of shape `[50]` type f32 values in the range [0, 10)
/// let array = safemlx::random::uniform::<_, f32>(0, 10, &[50], &key, &stream);
///
/// // same, but in range [0.5, 1)
/// let array = safemlx::random::uniform::<_, f32>(0.5f32, 1f32, &[50], &key, &stream);
/// ```
#[generate_macro(customize(root = "$crate::random"))]
pub fn uniform<'a, E: Into<Array>, T: ArrayElement>(
    lower: E,
    upper: E,
    #[optional] shape: impl IntoOption<&'a [i32]>,
    #[optional] key: impl Into<Option<&'a Array>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let lb: Array = lower.into();
    let ub: Array = upper.into();
    let shape = shape.into_option().unwrap_or(&[]);
    let stream = stream.as_ref();
    let key = resolve(key)?;

    Array::try_from_op(|res| unsafe {
        safemlx_sys::mlx_random_uniform(
            res,
            lb.as_ptr(),
            ub.as_ptr(),
            shape.as_ptr(),
            shape.len(),
            T::DTYPE.into(),
            key.as_ptr(),
            stream.as_ptr(),
        )
    })
}

/// Generate normally distributed random numbers.
///
/// Generate an array of random numbers using the optional shape. The result
/// will be of the given `T`. `T` must be a floating point type.
///
/// # Params
///
///  - shape: shape of the output, if `None` a single value is returned
///  - loc: mean of the distribution, default is `0.0`
///  - scale: standard deviation of the distribution, default is `1.0`
///  - key: PRNG key
///
/// # Example
///
/// ```rust
/// # let stream = safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
/// let key = safemlx::random::key(0).unwrap();
///
/// // generate a single f32 with normal distribution
/// let value = safemlx::random::normal::<f32>(None, None, None, &key, &stream).unwrap().item::<f32>(&stream);
///
/// // generate an array of f32 with normal distribution in shape [10, 5]
/// let array = safemlx::random::normal::<f32>(&[10, 5], None, None, &key, &stream);
/// ```
#[generate_macro(customize(root = "$crate::random"))]
pub fn normal<'a, T: ArrayElement>(
    #[optional] shape: impl IntoOption<&'a [i32]>,
    #[optional] loc: impl Into<Option<f32>>,
    #[optional] scale: impl Into<Option<f32>>,
    #[optional] key: impl Into<Option<&'a Array>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let shape = shape.into_option().unwrap_or(&[]);
    let stream = stream.as_ref();
    let key = resolve(key)?;

    Array::try_from_op(|res| unsafe {
        safemlx_sys::mlx_random_normal(
            res,
            shape.as_ptr(),
            shape.len(),
            T::DTYPE.into(),
            loc.into().unwrap_or(0.0),
            scale.into().unwrap_or(1.0),
            key.as_ptr(),
            stream.as_ptr(),
        )
    })
}

/// Generate jointly-normal random samples given a mean and covariance.
///
/// The matrix `covariance` must be positive semi-definite. The behavior is
/// undefined if it is not.  The only supported output type is f32.
///
/// # Params
/// - `mean`: array of shape `[..., n]`, the mean of the distribution.
/// - `covariance`: array  of shape `[..., n, n]`, the covariance matrix of the distribution. The batch shape `...` must be broadcast-compatible with that of `mean`.
/// - `shape`: The output shape must be broadcast-compatible with `&mean.shape[..mean.shape.len()-1]` and `&covariance.shape[..covariance.shape.len()-2]`. If empty, the result shape is determined by broadcasting the batch shapes of `mean` and `covariance`.
/// - `key`: PRNG key.
#[generate_macro(customize(root = "$crate::random"))]
// TODO: not supported on GPU yet
pub fn multivariate_normal<'a, T: ArrayElement>(
    mean: impl AsRef<Array>,
    covariance: impl AsRef<Array>,
    #[optional] shape: impl IntoOption<&'a [i32]>,
    #[optional] key: impl Into<Option<&'a Array>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let shape = shape.into_option().unwrap_or(&[]);
    let stream = stream.as_ref();
    let key = resolve(key)?;

    Array::try_from_op(|res| unsafe {
        safemlx_sys::mlx_random_multivariate_normal(
            res,
            mean.as_ref().as_ptr(),
            covariance.as_ref().as_ptr(),
            shape.as_ptr(),
            shape.len(),
            T::DTYPE.into(),
            key.as_ptr(),
            stream.as_ptr(),
        )
    })
}

/// Generate random integers from the given interval (`lower:` and `upper:`).
///
/// The values are sampled with equal probability from the integers in
/// half-open interval `[lb, ub)`. The lower and upper bound can be
/// scalars or arrays and must be roadcastable to `shape`.
///
/// ```rust
/// # let stream = safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
/// use safemlx::{array, random};
///
/// let key = random::key(0).unwrap();
///
/// // generate an array of Int values, one in the range [0, 20) and one in the range [10, 100)
/// let array = random::randint::<_, i32>(array!([0, 20]), array!([10, 100]), None, &key, &stream);
/// ```
#[generate_macro(customize(root = "$crate::random"))]
pub fn randint<'a, E: Into<Array>, T: ArrayElement>(
    lower: E,
    upper: E,
    #[optional] shape: impl IntoOption<&'a [i32]>,
    #[optional] key: impl Into<Option<&'a Array>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let lb: Array = lower.into();
    let ub: Array = upper.into();
    let shape = shape.into_option().unwrap_or(lb.shape());
    let stream = stream.as_ref();
    let key = resolve(key)?;

    Array::try_from_op(|res| unsafe {
        safemlx_sys::mlx_random_randint(
            res,
            lb.as_ptr(),
            ub.as_ptr(),
            shape.as_ptr(),
            shape.len(),
            T::DTYPE.into(),
            key.as_ptr(),
            stream.as_ptr(),
        )
    })
}

/// Generate Bernoulli random values with a given `p` value.
///
/// The values are sampled from the bernoulli distribution with parameter
/// `p`. The parameter `p` must have a floating point type and
/// must be broadcastable to `shape`.
///
/// ```rust
/// # let stream = safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
/// use safemlx::{array, Array, random};
///
/// let key = random::key(0).unwrap();
///
/// // generate a single random Bool with p = 0.8
/// let p: Array = 0.8.into();
/// let value = random::bernoulli(&p, None, &key, &stream);
///
/// // generate an array of shape [50, 2] of random Bool with p = 0.8
/// let array = random::bernoulli(&p, &[50, 2], &key, &stream);
///
/// // generate an array of [3] Bool with the given p values
/// let array = random::bernoulli(&array!([0.1, 0.5, 0.8]), None, &key, &stream);
/// ```
#[generate_macro(customize(root = "$crate::random"))]
pub fn bernoulli<'a>(
    #[optional] p: impl Into<Option<&'a Array>>,
    #[optional] shape: impl IntoOption<&'a [i32]>,
    #[optional] key: impl Into<Option<&'a Array>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let default_array = Array::from_f32(0.5);
    let p = p.into().unwrap_or(&default_array);

    let shape = shape.into_option().unwrap_or(p.shape());
    let stream = stream.as_ref();
    let key = resolve(key)?;

    Array::try_from_op(|res| unsafe {
        safemlx_sys::mlx_random_bernoulli(
            res,
            p.as_ptr(),
            shape.as_ptr(),
            shape.len(),
            key.as_ptr(),
            stream.as_ptr(),
        )
    })
}

/// Generate values from a truncated normal distribution between `low` and `high`.
///
/// The values are sampled from the truncated normal distribution
/// on the domain `(lower, upper)`. The bounds `lower` and `upper`
/// can be scalars or arrays and must be broadcastable to `shape`.
///
/// ```rust
/// # let stream = safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
/// use safemlx::{array, random};
///
/// let key = random::key(0).unwrap();
///
/// // generate an array of two Float values, one in the range 0 ..< 10
/// // and one in the range 10 ..< 100
/// let value = random::truncated_normal::<_, f32>(array!([0, 10]), array!([10, 100]), None, &key, &stream);
/// ```
#[generate_macro(customize(root = "$crate::random"))]
pub fn truncated_normal<'a, E: Into<Array>, T: ArrayElement>(
    lower: E,
    upper: E,
    #[optional] shape: impl IntoOption<&'a [i32]>,
    #[optional] key: impl Into<Option<&'a Array>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let lb: Array = lower.into();
    let ub: Array = upper.into();
    let shape = shape.into_option().unwrap_or(lb.shape());
    let stream = stream.as_ref();
    let key = resolve(key)?;

    Array::try_from_op(|res| unsafe {
        safemlx_sys::mlx_random_truncated_normal(
            res,
            lb.as_ptr(),
            ub.as_ptr(),
            shape.as_ptr(),
            shape.len(),
            T::DTYPE.into(),
            key.as_ptr(),
            stream.as_ptr(),
        )
    })
}

/// Sample from the standard Gumbel distribution.
///
/// The values are sampled from a standard Gumbel distribution
/// which CDF `exp(-exp(-x))`.
///
/// ```rust
/// # let stream = safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
/// let key = safemlx::random::key(0).unwrap();
///
/// // generate a single Float with Gumbel distribution
/// let value = safemlx::random::gumbel::<f32>(None, &key, &stream).unwrap().item::<f32>(&stream);
///
/// // generate an array of Float with Gumbel distribution in shape [10, 5]
/// let array = safemlx::random::gumbel::<f32>(&[10, 5], &key, &stream);
/// ```
#[generate_macro(customize(root = "$crate::random"))]
pub fn gumbel<'a, T: ArrayElement>(
    #[optional] shape: impl IntoOption<&'a [i32]>,
    #[optional] key: impl Into<Option<&'a Array>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let shape = shape.into_option().unwrap_or(&[]);
    let stream = stream.as_ref();
    let key = resolve(key)?;

    Array::try_from_op(|res| unsafe {
        safemlx_sys::mlx_random_gumbel(
            res,
            shape.as_ptr(),
            shape.len(),
            T::DTYPE.into(),
            key.as_ptr(),
            stream.as_ptr(),
        )
    })
}

/// Shape or count for the categorical distribution.
#[derive(Debug, Clone, Copy)]
pub enum ShapeOrCount<'a> {
    /// Shape
    Shape(&'a [i32]),

    /// Count
    Count(i32),
}

/// Sample from a categorical distribution.
///
/// The values are sampled from the categorical distribution specified by
/// the unnormalized values in `logits`.   If the `shape` is not specified
/// the result shape will be the same shape as `logits` with the `axis`
/// dimension removed.
///
/// /// # Params
/// # Params
///
/// - `logits`: The *unnormalized* categorical distribution(s).
/// - `axis`(optional): The axis which specifies the distribution. Default is `-1`.
/// - `shape_or_count`(optional):
/// - - `Shape`: The shape of the output. This must be broadcast compatible with `logits.shape` with the `axis` dimension removed.
/// - - `Count`: The number of samples to draw from each of the categorical distributions in `logits`. The output will have the number of samples in the last dimension.
/// - `key` (optional): A PRNG key.
///
/// # Example
///
/// ```rust
/// # let stream = safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
/// let key = safemlx::random::key(0).unwrap();
///
/// let logits = safemlx::Array::zeros::<u32>(&[5, 20], &stream).unwrap();
///
/// // produces Array of u32 shape &[5]
/// let result = safemlx::random::categorical(&logits, None, None, &key, &stream);
/// ```
#[generate_macro(customize(root = "$crate::random"))]
pub fn categorical<'a>(
    logits: impl AsRef<Array>,
    #[optional] axis: impl Into<Option<i32>>,
    #[optional] shape_or_count: impl Into<Option<ShapeOrCount<'a>>>,
    #[optional] key: impl Into<Option<&'a Array>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let axis = axis.into().unwrap_or(-1);
    let stream = stream.as_ref();
    let key = resolve(key)?;

    match shape_or_count.into() {
        Some(ShapeOrCount::Shape(shape)) => Array::try_from_op(|res| unsafe {
            safemlx_sys::mlx_random_categorical_shape(
                res,
                logits.as_ref().as_ptr(),
                axis,
                shape.as_ptr(),
                shape.len(),
                key.as_ptr(),
                stream.as_ptr(),
            )
        }),
        Some(ShapeOrCount::Count(num_samples)) => Array::try_from_op(|res| unsafe {
            safemlx_sys::mlx_random_categorical_num_samples(
                res,
                logits.as_ref().as_ptr(),
                axis,
                num_samples,
                key.as_ptr(),
                stream.as_ptr(),
            )
        }),
        None => Array::try_from_op(|res| unsafe {
            safemlx_sys::mlx_random_categorical(
                res,
                logits.as_ref().as_ptr(),
                axis,
                key.as_ptr(),
                stream.as_ptr(),
            )
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{array, assert_array_eq};
    use float_eq::{assert_float_eq, float_eq};

    #[test]
    fn test_explicit_random_state_is_deterministic() {
        let stream = crate::test_stream();
        let mut state = RandomState::with_seed(3).unwrap();
        let a_key = state.next_key(stream).unwrap();
        let b_key = state.next_key(stream).unwrap();
        let a = uniform::<_, f32>(0, 1, None, &a_key, stream).unwrap();
        let b = uniform::<_, f32>(0, 1, None, &b_key, stream).unwrap();

        let mut state = RandomState::with_seed(3).unwrap();
        let x_key = state.next_key(stream).unwrap();
        let y_key = state.next_key(stream).unwrap();
        let x = uniform::<_, f32>(0, 1, None, &x_key, stream).unwrap();
        let y = uniform::<_, f32>(0, 1, None, &y_key, stream).unwrap();

        assert_array_eq!(a, x, 0.01, stream = stream);
        assert_array_eq!(b, y, 0.01, stream = stream);
    }

    #[test]
    fn test_key() {
        let k1 = key(0).unwrap();
        let k2 = key(0).unwrap();
        assert!(crate::array::eval_equal_values(&k1, &k2));

        let k2 = key(1).unwrap();
        assert!(!crate::array::eval_equal_values(&k1, &k2));
    }

    #[test]
    fn test_split() {
        let stream = crate::test_stream();
        let key = key(0).unwrap();

        let (k1, k2) = split(&key, 2, stream).unwrap();
        assert!(!crate::array::eval_equal_values(&k1, &k2));

        let (r1, r2) = split(&key, 2, stream).unwrap();
        assert!(crate::array::eval_equal_values(&r1, &k1));
        assert!(crate::array::eval_equal_values(&r2, &k2));
    }

    #[test]
    fn test_uniform_requires_key() {
        let stream = crate::test_stream();
        let value = uniform::<_, f32>(0, 10, &[3], None, stream);
        assert!(value.is_err());
    }

    #[test]
    fn test_uniform_single() {
        let stream = crate::test_stream();
        let key = key(0).unwrap();
        let value = uniform::<_, f32>(0, 10, None, &key, stream).unwrap();
        float_eq!(value.item::<f32>(&stream), 4.18, abs <= 0.01);
    }

    #[test]
    fn test_uniform_multiple() {
        let stream = crate::test_stream();
        let key = key(0).unwrap();
        let value = uniform::<_, f32>(0, 10, &[3], &key, stream).unwrap();
        let expected = Array::from_slice(&[9.65, 3.14, 6.33], &[3]);

        assert_array_eq!(value, expected, 0.01, stream = stream);
    }

    #[test]
    fn test_uniform_multiple_array() {
        let stream = crate::test_stream();
        let key = key(0).unwrap();
        let value = uniform::<_, f32>(&[0, 10], &[10, 100], &[2], &key, stream).unwrap();
        let expected = Array::from_slice(&[2.16, 82.37], &[2]);

        assert_array_eq!(value, expected, 0.01, stream = stream);
    }

    #[test]
    fn test_uniform_non_float() {
        let stream = crate::test_stream();
        let key = key(0).unwrap();
        let value = uniform::<_, i32>(&[0, 10], &[10, 100], &[2], &key, stream);
        assert!(value.is_err());
    }

    #[test]
    fn test_normal() {
        let stream = crate::test_stream();
        let key = key(0).unwrap();
        let value = normal::<f32>(None, None, None, &key, stream).unwrap();
        float_eq!(value.item::<f32>(&stream), -0.20, abs <= 0.01);
    }

    #[test]
    fn test_normal_non_float() {
        let stream = crate::test_stream();
        let key = key(0).unwrap();
        let value = normal::<i32>(None, None, None, &key, stream);
        assert!(value.is_err());
    }

    #[test]
    fn test_multivariate_normal() {
        let stream = crate::test_stream();
        let key = key(0).unwrap();
        let mean = Array::from_slice(&[0.0, 0.0], &[2]);
        let covariance = Array::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);

        let a = multivariate_normal::<f32>(&mean, &covariance, &[3], &key, stream).unwrap();
        assert!(a.shape() == [3, 2]);
    }

    #[test]
    fn test_randint_single() {
        let stream = crate::test_stream();
        let key = key(0).unwrap();
        let value = randint::<_, i32>(0, 100, None, &key, stream).unwrap();
        assert_eq!(value.item::<i32>(&stream), 41);
    }

    #[test]
    fn test_randint_multiple() {
        let stream = crate::test_stream();
        let key = key(0).unwrap();
        let value =
            randint::<_, i32>(array!([0, 10]), array!([10, 100]), None, &key, stream).unwrap();
        let expected = Array::from_slice(&[2, 82], &[2]);

        assert_array_eq!(value, expected, 0.01, stream = stream);
    }

    #[test]
    fn test_randint_non_int() {
        let stream = crate::test_stream();
        let key = key(0).unwrap();
        let value = randint::<_, f32>(array!([0, 10]), array!([10, 100]), None, &key, stream);
        assert!(value.is_err());
    }

    #[test]
    fn test_bernoulli_single() {
        let stream = crate::test_stream();
        let key = key(0).unwrap();
        let value = bernoulli(None, None, &key, stream).unwrap();
        assert!(value.item::<bool>(&stream));
    }

    #[test]
    fn test_bernoulli_multiple() {
        let stream = crate::test_stream();
        let key = key(0).unwrap();
        let value = bernoulli(None, &[4], &key, stream).unwrap();
        let expected = Array::from_slice(&[false, true, false, true], &[4]);

        assert_array_eq!(value, expected, 0.01, stream = stream);
    }

    #[test]
    fn test_bernoulli_p() {
        let stream = crate::test_stream();
        let key = key(0).unwrap();
        let p: Array = 0.8.into();
        let value = bernoulli(&p, &[4], &key, stream).unwrap();
        let expected = Array::from_slice(&[false, true, true, true], &[4]);

        assert_array_eq!(value, expected, 0.01, stream = stream);
    }

    #[test]
    fn test_bernoulli_p_array() {
        let stream = crate::test_stream();
        let key = key(0).unwrap();
        let value = bernoulli(&array!([0.1, 0.5, 0.8]), None, &key, stream).unwrap();
        let expected = Array::from_slice(&[false, true, true], &[3]);

        assert_array_eq!(value, expected, 0.01, stream = stream);
    }

    #[test]
    fn test_truncated_normal_single() {
        let stream = crate::test_stream();
        let key = key(0).unwrap();
        let value = truncated_normal::<_, f32>(0, 10, None, &key, stream).unwrap();
        assert_array_eq!(value, Array::from_f32(0.55), 0.01, stream = stream);
    }

    #[test]
    fn test_truncated_normal_multiple() {
        let stream = crate::test_stream();
        let key = key(0).unwrap();
        let value = truncated_normal::<_, f32>(0.0, 0.5, &[3], &key, stream).unwrap();
        let expected = Array::from_slice(&[0.48, 0.15, 0.30], &[3]);

        assert_array_eq!(value, expected, 0.01, stream = stream);
    }

    #[test]
    fn test_truncated_normal_multiple_array() {
        let stream = crate::test_stream();
        let key = key(0).unwrap();
        let value =
            truncated_normal::<_, f32>(array!([0.0, 0.5]), array!([0.5, 1.0]), None, &key, stream)
                .unwrap();
        let expected = Array::from_slice(&[0.10, 0.88], &[2]);

        assert_array_eq!(value, expected, 0.01, stream = stream);
    }

    #[test]
    fn test_gumbel() {
        let stream = crate::test_stream();
        let key = key(0).unwrap();
        let value = gumbel::<f32>(None, &key, stream).unwrap();
        assert_array_eq!(value, Array::from_f32(0.13), 0.01, stream = stream);
    }

    #[test]
    fn test_logits() {
        let stream = crate::test_stream();
        let key = key(0).unwrap();
        let logits = Array::zeros::<u32>(&[5, 20], stream).unwrap();
        let result = categorical(&logits, None, None, &key, stream).unwrap();

        assert_eq!(result.shape(), [5]);

        let expected = Array::from_slice(&[1, 1, 17, 17, 17], &[5]);
        assert_array_eq!(result, expected, 0.01, stream = stream);
    }

    #[test]
    fn test_logits_count() {
        let stream = crate::test_stream();
        let key = key(0).unwrap();
        let logits = Array::zeros::<u32>(&[5, 20], stream).unwrap();
        let result = categorical(&logits, None, ShapeOrCount::Count(2), &key, stream).unwrap();

        assert_eq!(result.shape(), [5, 2]);

        let expected = Array::from_slice(&[16, 3, 14, 10, 17, 7, 6, 8, 12, 8], &[5, 2]);
        assert_array_eq!(result, expected, 0.01, stream = stream);
    }

    #[test]
    fn test_random_state_new() {
        let state = RandomState::new().unwrap();
        assert_eq!(state.as_array().shape(), &[2]);
    }

    #[test]
    fn test_random_state_with_seed_deterministic() {
        let s1 = RandomState::with_seed(42).unwrap();
        let s2 = RandomState::with_seed(42).unwrap();
        assert!(crate::array::eval_equal_values(
            s1.as_array(),
            s2.as_array()
        ));
    }

    #[test]
    fn test_random_state_next_key_advances() {
        let stream = crate::test_stream();
        let mut state = RandomState::with_seed(0).unwrap();
        let k1 = state.next_key(stream).unwrap();
        let k2 = state.next_key(stream).unwrap();
        assert!(!crate::array::eval_equal_values(&k1, &k2));
    }

    #[test]
    fn test_random_state_from_key_roundtrip() {
        let original = RandomState::with_seed(99).unwrap();
        let arr = original.as_array().clone();
        let restored = RandomState::from_key(arr);
        assert!(crate::array::eval_equal_values(
            original.as_array(),
            restored.as_array()
        ));
    }

    #[test]
    fn test_random_state_updatable() {
        use crate::utils::Updatable;
        let state = RandomState::with_seed(0).unwrap();
        assert_eq!(state.updatable_states_len(), 1);
        assert_eq!(state.updatable_states().into_iter().count(), 1);
    }

    #[test]
    fn test_random_state_default() {
        let state = RandomState::default();
        assert_eq!(state.as_array().shape(), &[2]);
    }

    #[test]
    fn test_random_seed_same() {
        let stream = crate::test_stream();
        // Same random seed should produce the same results
        let seed = 23;
        let mut results = Vec::new();
        for _ in 0..10 {
            let mut state = RandomState::new().unwrap();
            state.seed(seed).unwrap();
            let draw_key = state.next_key(stream).unwrap();
            let result = uniform::<_, f32>(0.0, 1.0, &[10, 10], &draw_key, stream)
                .unwrap()
                .sum(None, stream)
                .unwrap()
                .try_item::<f32>(&stream)
                .unwrap();
            results.push(result);
        }

        // Check that all results are the same within a small tolerance
        let first = results[0];
        for result in &results[1..] {
            assert_float_eq!(
                first,
                *result,
                abs <= 0.01,
                "Results should be equal for the same seed"
            );
        }
    }
}
