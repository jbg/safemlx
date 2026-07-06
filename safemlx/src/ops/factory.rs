use crate::array::Array;
use crate::array::ArrayElement;
use crate::error::Result;
use crate::utils::guard::Guarded;
use crate::{Dtype, Stream};
use num_traits::NumCast;
use safemlx_internal_macros::generate_macro;

impl Array {
    /// Construct an array of zeros returning an error if shape is invalid.
    ///
    /// # Params
    ///
    /// - shape: Desired shape
    ///
    /// # Example
    ///
    /// ```rust
    /// # let stream = safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    /// use safemlx::{Array, Stream};
    /// Array::zeros::<f32>(&[5, 10], Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0))).unwrap();
    /// ```
    pub fn zeros<T: ArrayElement>(shape: &[i32], stream: impl AsRef<Stream>) -> Result<Array> {
        let dtype = T::DTYPE;
        zeros_dtype(shape, dtype, stream)
    }

    /// Construct an array of ones returning an error if shape is invalid.
    ///
    /// # Params
    ///
    /// - shape: Desired shape
    ///
    /// # Example
    ///
    /// ```rust
    /// # let stream = safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    /// use safemlx::{Array, Stream};
    /// Array::ones::<f32>(&[5, 10], Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0))).unwrap();
    /// ```
    pub fn ones<T: ArrayElement>(shape: &[i32], stream: impl AsRef<Stream>) -> Result<Array> {
        let dtype = T::DTYPE;
        ones_dtype(shape, dtype, stream)
    }

    /// Create an identity matrix or a general diagonal matrix returning an error if params are invalid.
    ///
    /// # Params
    ///
    /// - n: number of rows in the output
    /// - m: number of columns in the output -- equal to `n` if not specified
    /// - k: index of the diagonal - defaults to 0 if not specified
    ///
    /// # Example
    ///
    /// ```rust
    /// # let stream = safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    /// use safemlx::{Array, Stream};
    /// //  create [10, 10] array with 1's on the diagonal.
    /// let r = Array::eye::<f32>(10, None, None, Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0))).unwrap();
    /// ```
    pub fn eye<T: ArrayElement>(
        n: i32,
        m: Option<i32>,
        k: Option<i32>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            safemlx_sys::mlx_eye(
                res,
                n,
                m.unwrap_or(n),
                k.unwrap_or(0),
                T::DTYPE.into(),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Construct an array with the given value returning an error if shape is invalid.
    ///
    /// Constructs an array of size `shape` filled with `values`. If `values`
    /// is an [Array] it must be [broadcasting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/broadcasting) to the given `shape`.
    ///
    /// # Params
    ///
    /// - shape: shape of the output array
    /// - values: values to be broadcast into the array
    ///
    /// # Example
    ///
    /// ```rust
    /// # let stream = safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    /// use safemlx::{Array, Stream, array};
    /// //  create [5, 4] array filled with 7
    /// let r = Array::full::<f32>(&[5, 4], array!(7.0f32), Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0))).unwrap();
    /// ```
    pub fn full<T: ArrayElement>(
        shape: &[i32],
        values: impl AsRef<Array>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            safemlx_sys::mlx_full(
                res,
                shape.as_ptr(),
                shape.len(),
                values.as_ref().as_ptr(),
                T::DTYPE.into(),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Create a square identity matrix returning an error if params are invalid.
    ///
    /// # Params
    ///
    /// - n: number of rows and columns in the output
    ///
    /// # Example
    ///
    /// ```rust
    /// # let stream = safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    /// use safemlx::{Array, Stream};
    /// //  create [10, 10] array with 1's on the diagonal.
    /// let r = Array::identity::<f32>(10, Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0))).unwrap();
    /// ```
    pub fn identity<T: ArrayElement>(n: i32, stream: impl AsRef<Stream>) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            safemlx_sys::mlx_identity(res, n, T::DTYPE.into(), stream.as_ref().as_ptr())
        })
    }

    /// Generates ranges of numbers.
    ///
    /// Generate numbers in the half-open interval `[start, stop)` in increments of `step`.
    ///
    /// # Params
    ///
    /// - `start`: Starting value which defaults to `0`.
    /// - `stop`: Stopping value.
    /// - `step`: Increment which defaults to `1`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # let stream = safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    /// use safemlx::{Array, Stream};
    ///
    /// // Create a 1-D array with values from 0 to 50
    /// let r = Array::arange::<_, f32>(None, 50, None, &stream);
    /// ```
    pub fn arange<U, T>(
        start: impl Into<Option<U>>,
        stop: U,
        step: impl Into<Option<U>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array>
    where
        U: NumCast,
        T: ArrayElement,
    {
        let start: f64 = start.into().and_then(NumCast::from).unwrap_or(0.0);
        let stop: f64 = NumCast::from(stop).unwrap();
        let step: f64 = step.into().and_then(NumCast::from).unwrap_or(1.0);

        Array::try_from_op(|res| unsafe {
            safemlx_sys::mlx_arange(
                res,
                start,
                stop,
                step,
                T::DTYPE.into(),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Generate `num` evenly spaced numbers over interval `[start, stop]` returning an error if params are invalid.
    ///
    /// # Params
    ///
    /// - start: start value
    /// - stop: stop value
    /// - count: number of samples -- defaults to 50 if not specified
    ///
    /// # Example
    ///
    /// ```rust
    /// # let stream = safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    /// use safemlx::{Array, Stream};
    /// // Create a 50 element 1-D array with values from 0 to 50
    /// let r = Array::linspace::<_, f32>(0, 50, None, Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0))).unwrap();
    /// ```
    pub fn linspace<U, T>(
        start: U,
        stop: U,
        count: impl Into<Option<i32>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array>
    where
        U: NumCast,
        T: ArrayElement,
    {
        let count = count.into().unwrap_or(50);
        let start_f32 = NumCast::from(start).unwrap();
        let stop_f32 = NumCast::from(stop).unwrap();

        Array::try_from_op(|res| unsafe {
            safemlx_sys::mlx_linspace(
                res,
                start_f32,
                stop_f32,
                count,
                T::DTYPE.into(),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Repeat an array along a specified axis returning an error if params are invalid.
    ///
    /// # Params
    ///
    /// - array: array to repeat
    /// - count: number of times to repeat
    /// - axis: axis to repeat along
    ///
    /// # Example
    ///
    /// ```rust
    /// # let stream = safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    /// use safemlx::{Array, Stream};
    /// // repeat a [2, 2] array 4 times along axis 1
    /// let source = Array::from_slice(&[0, 1, 2, 3], &[2, 2]);
    /// let r = Array::repeat_axis::<i32>(source, 4, 1, &stream).unwrap();
    /// ```
    pub fn repeat_axis<T: ArrayElement>(
        array: Array,
        count: i32,
        axis: i32,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            safemlx_sys::mlx_repeat_axis(res, array.as_ptr(), count, axis, stream.as_ref().as_ptr())
        })
    }

    /// Repeat a flattened array along axis 0 returning an error if params are invalid.
    ///
    /// # Params
    ///
    /// - array: array to repeat
    /// - count: number of times to repeat
    ///
    /// # Example
    ///
    /// ```rust
    /// # let stream = safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    /// use safemlx::{Array, Stream};
    /// // repeat a 4 element array 4 times along axis 0
    /// let source = Array::from_slice(&[0, 1, 2, 3], &[2, 2]);
    /// let r = Array::repeat::<i32>(source, 4, &stream).unwrap();
    /// ```
    pub fn repeat<T: ArrayElement>(
        array: Array,
        count: i32,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            safemlx_sys::mlx_repeat(res, array.as_ptr(), count, stream.as_ref().as_ptr())
        })
    }

    /// An array with ones at and below the given diagonal and zeros elsewhere.
    ///
    /// # Params
    ///
    /// - n: number of rows in the output
    /// - m: number of columns in the output -- equal to `n` if not specified
    /// - k: index of the diagonal -- defaults to 0 if not specified
    ///
    /// # Example
    ///
    /// ```rust
    /// # let stream = safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    /// use safemlx::{Array, Stream};
    /// // [5, 5] array with the lower triangle filled with 1s
    /// let r = Array::tri::<f32>(5, None, None, Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0)));
    /// ```
    pub fn tri<T: ArrayElement>(
        n: i32,
        m: Option<i32>,
        k: Option<i32>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            safemlx_sys::mlx_tri(
                res,
                n,
                m.unwrap_or(n),
                k.unwrap_or(0),
                T::DTYPE.into(),
                stream.as_ref().as_ptr(),
            )
        })
    }
}

/// See [`Array::zeros`]
#[generate_macro]
pub fn zeros<T: ArrayElement>(
    shape: &[i32],
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::zeros::<T>(shape, stream)
}

/// An array of zeros like the input.
#[generate_macro]
pub fn zeros_like(
    input: impl AsRef<Array>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = input.as_ref();
    let shape = a.shape();
    let dtype = a.dtype();
    zeros_dtype(shape, dtype, stream)
}

/// Similar to [`Array::zeros`] but with a specified dtype.
#[generate_macro]
pub fn zeros_dtype(
    shape: &[i32],
    dtype: Dtype,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        safemlx_sys::mlx_zeros(
            res,
            shape.as_ptr(),
            shape.len(),
            dtype.into(),
            stream.as_ref().as_ptr(),
        )
    })
}

/// See [`Array::ones`]
#[generate_macro]
pub fn ones<T: ArrayElement>(
    shape: &[i32],
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::ones::<T>(shape, stream)
}

/// An array of ones like the input.
#[generate_macro]
pub fn ones_like(
    input: impl AsRef<Array>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = input.as_ref();
    let shape = a.shape();
    let dtype = a.dtype();
    ones_dtype(shape, dtype, stream)
}

/// An array filled with the given value, with the same shape as the input.
///
/// # Params
///
/// - `input`: Input array to take shape from
/// - `values`: Value(s) to fill the array with
/// - `dtype`: Optional dtype for the output array. Defaults to the dtype of the input array.
/// - `stream`: Stream to run the operation on
///
/// # Example
///
/// ```rust
/// # let stream = safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
/// use safemlx::{Array, Dtype, ops::full_like};
///
/// let a = Array::from_slice(&[1i32, 2, 3], &[3]);
/// // Fill with same dtype as input
/// let b = full_like(&a, &Array::from_f32(7.0), None, &stream).unwrap();
/// assert_eq!(b.dtype(), Dtype::Int32);
///
/// // Fill with specified dtype
/// let c = full_like(&a, &Array::from_f32(7.5), Some(Dtype::Float32), &stream).unwrap();
/// assert_eq!(c.dtype(), Dtype::Float32);
/// ```
#[generate_macro]
pub fn full_like(
    input: impl AsRef<Array>,
    values: impl AsRef<Array>,
    #[optional] dtype: impl Into<Option<Dtype>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = input.as_ref();
    let dtype = dtype.into().unwrap_or_else(|| a.dtype());
    Array::try_from_op(|res| unsafe {
        safemlx_sys::mlx_full_like(
            res,
            a.as_ptr(),
            values.as_ref().as_ptr(),
            dtype.into(),
            stream.as_ref().as_ptr(),
        )
    })
}

/// Similar to [`Array::ones`] but with a specified dtype.
#[generate_macro]
pub fn ones_dtype(
    shape: &[i32],
    dtype: Dtype,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        safemlx_sys::mlx_ones(
            res,
            shape.as_ptr(),
            shape.len(),
            dtype.into(),
            stream.as_ref().as_ptr(),
        )
    })
}

/// See [`Array::eye`]
#[generate_macro]
pub fn eye<T: ArrayElement>(
    n: i32,
    #[optional] m: Option<i32>,
    #[optional] k: Option<i32>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::eye::<T>(n, m, k, stream)
}

/// See [`Array::full`]
#[generate_macro]
pub fn full<T: ArrayElement>(
    shape: &[i32],
    values: impl AsRef<Array>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::full::<T>(shape, values, stream)
}

/// See [`Array::identity`]
#[generate_macro]
pub fn identity<T: ArrayElement>(n: i32, #[optional] stream: impl AsRef<Stream>) -> Result<Array> {
    Array::identity::<T>(n, stream)
}

/// See [`Array::arange`]
#[generate_macro]
pub fn arange<U, T>(
    #[optional] start: impl Into<Option<U>>,
    #[named] stop: U,
    #[optional] step: impl Into<Option<U>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array>
where
    U: NumCast,
    T: ArrayElement,
{
    Array::arange::<U, T>(start, stop, step, stream)
}

/// See [`Array::linspace`]
#[generate_macro]
pub fn linspace<U, T>(
    start: U,
    stop: U,
    #[optional] count: impl Into<Option<i32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array>
where
    U: NumCast,
    T: ArrayElement,
{
    Array::linspace::<U, T>(start, stop, count, stream)
}

/// See [`Array::repeat`]
#[generate_macro]
pub fn repeat_axis<T: ArrayElement>(
    array: Array,
    count: i32,
    axis: i32,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::repeat_axis::<T>(array, count, axis, stream)
}

/// See [`Array::repeat`]
#[generate_macro]
pub fn repeat<T: ArrayElement>(
    array: Array,
    count: i32,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::repeat::<T>(array, count, stream)
}

/// See [`Array::tri`]
#[generate_macro]
pub fn tri<T: ArrayElement>(
    n: i32,
    #[optional] m: Option<i32>,
    #[optional] k: Option<i32>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::tri::<T>(n, m, k, stream)
}

/// Zeros the array above the given diagonal
///
/// # Params
///
/// - `a`: input array
/// - `k`: diagonal of the 2D array. Default to `0`
/// - `stream`: stream to execute on
#[generate_macro]
pub fn tril(
    a: impl AsRef<Array>,
    #[optional] k: impl Into<Option<i32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let k = k.into().unwrap_or(0);
    Array::try_from_op(|res| unsafe {
        safemlx_sys::mlx_tril(res, a.as_ptr(), k, stream.as_ref().as_ptr())
    })
}

/// Zeros the array below the given diagonal
///
/// # Params
///
/// - `a`: input array
/// - `k`: diagonal of the 2D array. Default to `0`
#[generate_macro]
pub fn triu(
    a: impl AsRef<Array>,
    #[optional] k: impl Into<Option<i32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let k = k.into().unwrap_or(0);
    Array::try_from_op(|res| unsafe {
        safemlx_sys::mlx_triu(res, a.as_ptr(), k, stream.as_ref().as_ptr())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{array, dtype::Dtype, Stream};
    use half::f16;

    #[test]
    fn test_zeros() {
        let stream = crate::test_stream();
        let array = Array::zeros::<f32>(&[2, 3], stream).unwrap();
        assert_eq!(array.shape(), &[2, 3]);
        assert_eq!(array.dtype(), Dtype::Float32);

        let data: Vec<f32> = crate::array::eval_vec(&array);
        assert_eq!(data, &[0.0; 6]);
    }

    #[test]
    fn test_zeros_try() {
        let stream = crate::test_stream();
        let array = Array::zeros::<f32>(&[2, 3], stream);
        assert!(array.is_ok());

        let array = Array::zeros::<f32>(&[-1, 3], stream);
        assert!(array.is_err());
    }

    #[test]
    fn test_ones() {
        let stream = crate::test_stream();
        let array = Array::ones::<f16>(&[2, 3], stream).unwrap();
        assert_eq!(array.shape(), &[2, 3]);
        assert_eq!(array.dtype(), Dtype::Float16);

        let data: Vec<f16> = crate::array::eval_vec(&array);
        assert_eq!(data, &[f16::from_f32(1.0); 6]);
    }

    #[test]
    fn test_eye() {
        let stream = crate::test_stream();
        let array = Array::eye::<f32>(3, None, None, stream).unwrap();
        assert_eq!(array.shape(), &[3, 3]);
        assert_eq!(array.dtype(), Dtype::Float32);

        let data: Vec<f32> = crate::array::eval_vec(&array);
        assert_eq!(data, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_full_scalar() {
        let stream = crate::test_stream();
        let array = Array::full::<f32>(&[2, 3], array!(7f32), stream).unwrap();
        assert_eq!(array.shape(), &[2, 3]);
        assert_eq!(array.dtype(), Dtype::Float32);

        let data: Vec<f32> = crate::array::eval_vec(&array);
        assert_eq!(data, &[7.0; 6]);
    }

    #[test]
    fn test_full_array() {
        let stream = crate::test_stream();
        let source = Array::zeros::<f32>(
            &[1, 3],
            Stream::new_with_device(&crate::Device::new(crate::DeviceType::Cpu, 0)),
        )
        .unwrap();
        let array = Array::full::<f32>(&[2, 3], source, stream).unwrap();
        assert_eq!(array.shape(), &[2, 3]);
        assert_eq!(array.dtype(), Dtype::Float32);

        let data: Vec<f32> = crate::array::eval_vec(&array);
        float_eq::float_eq!(*data, [0.0; 6], abs <= [1e-6; 6]);
    }

    #[test]
    fn test_full_try() {
        let stream = crate::test_stream();
        let source = Array::zeros::<f32>(
            &[1, 3],
            Stream::new_with_device(&crate::Device::new(crate::DeviceType::Gpu, 0)),
        )
        .unwrap();
        let array = Array::full::<f32>(&[2, 3], source, stream);
        assert!(array.is_ok());

        let source = Array::zeros::<f32>(
            &[1, 3],
            Stream::new_with_device(&crate::Device::new(crate::DeviceType::Gpu, 0)),
        )
        .unwrap();
        let array = Array::full::<f32>(&[-1, 3], source, stream);
        assert!(array.is_err());
    }

    #[test]
    fn test_identity() {
        let stream = crate::test_stream();
        let array = Array::identity::<f32>(3, stream).unwrap();
        assert_eq!(array.shape(), &[3, 3]);
        assert_eq!(array.dtype(), Dtype::Float32);

        let data: Vec<f32> = crate::array::eval_vec(&array);
        assert_eq!(data, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_arange() {
        let stream = crate::test_stream();
        let array = Array::arange::<_, f32>(None, 50, None, stream).unwrap();
        assert_eq!(array.shape(), &[50]);
        assert_eq!(array.dtype(), Dtype::Float32);

        let data: Vec<f32> = crate::array::eval_vec(&array);
        let expected: Vec<f32> = (0..50).map(|x| x as f32).collect();
        assert_eq!(data, expected);

        let array = Array::arange::<_, i32>(0, 50, None, stream).unwrap();
        assert_eq!(array.shape(), &[50]);
        assert_eq!(array.dtype(), Dtype::Int32);

        let data: Vec<i32> = crate::array::eval_vec(&array);
        let expected: Vec<i32> = (0..50).collect();
        assert_eq!(data, expected);

        let result = Array::arange::<_, bool>(None, 50, None, stream);
        assert!(result.is_err());

        let result = Array::arange::<_, f32>(f64::NEG_INFINITY, 50.0, None, stream);
        assert!(result.is_err());

        let result = Array::arange::<_, f32>(0.0, f64::INFINITY, None, stream);
        assert!(result.is_err());

        let result = Array::arange::<_, f32>(0.0, 50.0, f32::NAN, stream);
        assert!(result.is_err());

        let result = Array::arange::<_, f32>(f32::NAN, 50.0, None, stream);
        assert!(result.is_err());

        let result = Array::arange::<_, f32>(0.0, f32::NAN, None, stream);
        assert!(result.is_err());

        let result = Array::arange::<_, f32>(0, i32::MAX as i64 + 1, None, stream);
        assert!(result.is_err());
    }

    #[test]
    fn test_linspace_int() {
        let stream = crate::test_stream();
        let array = Array::linspace::<_, f32>(0, 50, None, stream).unwrap();
        assert_eq!(array.shape(), &[50]);
        assert_eq!(array.dtype(), Dtype::Float32);

        let expected_data: Vec<f32> = (0..50).map(|x| x as f32 * (50.0 / 49.0)).collect();
        let expected = Array::from_slice(&expected_data, &[50]);
        assert_eq!(array.shape(), expected.shape());
        assert_array_all_close!(array, expected, stream = stream);
    }

    #[test]
    fn test_linspace_float() {
        let stream = crate::test_stream();
        let array = Array::linspace::<_, f32>(0., 50., None, stream).unwrap();
        assert_eq!(array.shape(), &[50]);
        assert_eq!(array.dtype(), Dtype::Float32);

        let expected_data: Vec<f32> = (0..50).map(|x| x as f32 * (50.0 / 49.0)).collect();
        let expected = Array::from_slice(&expected_data, &[50]);
        assert_eq!(array.shape(), expected.shape());
        assert_array_all_close!(array, expected, stream = stream);
    }

    #[test]
    fn test_linspace_try() {
        let stream = crate::test_stream();
        let array = Array::linspace::<_, f32>(0, 50, None, stream);
        assert!(array.is_ok());

        let array = Array::linspace::<_, f32>(0, 50, Some(-1), stream);
        assert!(array.is_err());
    }

    #[test]
    fn test_repeat() {
        let stream = crate::test_stream();
        let source = Array::from_slice(&[0, 1, 2, 3], &[2, 2]);
        let array = Array::repeat_axis::<i32>(source, 4, 1, stream).unwrap();
        assert_eq!(array.shape(), &[2, 8]);
        assert_eq!(array.dtype(), Dtype::Int32);

        let data: Vec<i32> = crate::array::eval_vec(&array);
        assert_eq!(data, [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]);
    }

    #[test]
    fn test_repeat_try() {
        let stream = crate::test_stream();
        let source = Array::from_slice(&[0, 1, 2, 3], &[2, 2]);
        let array = Array::repeat_axis::<i32>(source, 4, 1, stream);
        assert!(array.is_ok());

        let source = Array::from_slice(&[0, 1, 2, 3], &[2, 2]);
        let array = Array::repeat_axis::<i32>(source, -1, 1, stream);
        assert!(array.is_err());
    }

    #[test]
    fn test_repeat_all() {
        let stream = crate::test_stream();
        let source = Array::from_slice(&[0, 1, 2, 3], &[2, 2]);
        let array = Array::repeat::<i32>(source, 4, stream).unwrap();
        assert_eq!(array.shape(), &[16]);
        assert_eq!(array.dtype(), Dtype::Int32);

        let data: Vec<i32> = crate::array::eval_vec(&array);
        assert_eq!(data, [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]);
    }

    #[test]
    fn test_repeat_all_try() {
        let stream = crate::test_stream();
        let source = Array::from_slice(&[0, 1, 2, 3], &[2, 2]);
        let array = Array::repeat::<i32>(source, 4, stream);
        assert!(array.is_ok());

        let source = Array::from_slice(&[0, 1, 2, 3], &[2, 2]);
        let array = Array::repeat::<i32>(source, -1, stream);
        assert!(array.is_err());
    }

    #[test]
    fn test_tri() {
        let stream = crate::test_stream();
        let array = Array::tri::<f32>(3, None, None, stream).unwrap();
        assert_eq!(array.shape(), &[3, 3]);
        assert_eq!(array.dtype(), Dtype::Float32);

        let data: Vec<f32> = crate::array::eval_vec(&array);
        assert_eq!(data, &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]);
    }

    // The tests below are adapted from the C++ unit test `ops_tests.cpp/test full_like`
    #[test]
    fn test_full_like() {
        let stream = crate::test_stream();
        // Test with explicit dtype (different from input)
        let base_int = Array::from_slice(&[1i16, 2, 3], &[3]);
        let from_array_with_dtype =
            full_like(&base_int, &array!(7.5f32), Some(Dtype::Float16), stream).unwrap();
        assert_eq!(from_array_with_dtype.dtype(), Dtype::Float16);
        assert_eq!(from_array_with_dtype.shape(), &[3]);

        let expected_f16: Vec<f16> = vec![f16::from_f32(7.5); 3];
        let data: Vec<f16> = crate::array::eval_vec::<f16>(&from_array_with_dtype).to_vec();
        assert_eq!(data, expected_f16);

        // Test with default dtype (inherits from input)
        let from_array_default_dtype = full_like(&base_int, &array!(4.0f32), None, stream).unwrap();
        assert_eq!(from_array_default_dtype.dtype(), Dtype::Int16);
        let data: Vec<i16> = crate::array::eval_vec(&from_array_default_dtype);
        assert_eq!(data, &[4, 4, 4]);

        // Test with explicit dtype float32
        let from_scalar_with_dtype =
            full_like(&base_int, &array!(3.25f32), Some(Dtype::Float32), stream).unwrap();
        assert_eq!(from_scalar_with_dtype.dtype(), Dtype::Float32);
        let data: Vec<f32> = crate::array::eval_vec(&from_scalar_with_dtype);
        assert_eq!(data, &[3.25f32, 3.25f32, 3.25f32]);

        // Test with float base and int value - uses base dtype
        let base_float = Array::from_slice(&[1.0f32, 2.0f32], &[2]);
        let from_scalar_default_dtype =
            full_like(&base_float, &array!(2i32), None, stream).unwrap();
        assert_eq!(from_scalar_default_dtype.dtype(), Dtype::Float32);
        let data: Vec<f32> = crate::array::eval_vec(&from_scalar_default_dtype);
        assert_eq!(data, &[2.0f32, 2.0f32]);
    }
}
