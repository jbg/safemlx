//! Macros for creating arrays.

/// A helper macro to create an array with up to 3 dimensions.
///
/// # Examples
///
/// ```rust
/// # let stream = safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
/// use safemlx::array;
///
/// // Create an empty array
/// // Note that an empty array defaults to f32 and one dimension
/// let empty = array!();
///
/// // Create a scalar array
/// let s = array!(1);
/// // Scalar array has 0 dimension
/// assert_eq!(s.ndim(), 0);
///
/// // Create a one-element array (singleton matrix)
/// let s = array!([1]);
/// // Singleton array has 1 dimension
/// assert!(s.ndim() == 1);
///
/// // Create a 1D array
/// let a1 = array!([1, 2, 3]);
///
/// // Create a 2D array
/// let a2 = array!([
///     [1, 2, 3],
///     [4, 5, 6]
/// ]);
///
/// // Create a 3D array
/// let a3 = array!([
///     [
///         [1, 2, 3],
///         [4, 5, 6]
///     ],
///     [
///         [7, 8, 9],
///         [10, 11, 12]
///     ]
/// ]);
///
/// // Create a 2x2 array by specifying the shape
/// let a = array!([1, 2, 3, 4], shape=[2, 2]);
/// ```
#[macro_export]
macro_rules! array {
    ([$($x:expr),*], shape=[$($s:expr),*]) => {
        {
            let data = [$($x,)*];
            let shape = [$($s,)*];
            $crate::Array::from_slice(&data, &shape)
        }
    };
    ([$([$([$($x:expr),*]),*]),*]) => {
        {
            let arr = [$([$([$($x,)*],)*],)*];
            <$crate::Array as $crate::FromNested<_>>::from_nested(arr)
        }
    };
    ([$([$($x:expr),*]),*]) => {
        {
            let arr = [$([$($x,)*],)*];
            <$crate::Array as $crate::FromNested<_>>::from_nested(arr)
        }
    };
    ([$($x:expr),*]) => {
        {
            let arr = [$($x,)*];
            <$crate::Array as $crate::FromNested<_>>::from_nested(arr)
        }
    };
    ($x:expr) => {
        {
            <$crate::Array as $crate::FromScalar<_>>::from_scalar($x)
        }
    };
    // Empty array default to f32
    () => {
        $crate::Array::from_slice::<f32>(&[], &[0])
    };
}

#[cfg(test)]
mod tests {
    use crate::ops::indexing::IndexOp;

    #[test]
    fn test_scalar_array() {
        let stream = crate::test_stream();
        let arr = array!(1);

        // Scalar array has 0 dimension
        assert_eq!(arr.ndim(), 0);
        // Scalar array has empty shape
        assert!(arr.shape().is_empty());
        assert_eq!(arr.item::<i32>(&stream), 1);
    }

    #[test]
    fn test_array_1d() {
        let stream = crate::test_stream();
        let arr = array!([1, 2, 3]);

        // One element array has 1 dimension
        assert_eq!(arr.ndim(), 1);
        assert_eq!(arr.shape(), &[3]);
        assert_eq!(arr.index_device(0, stream).item::<i32>(&stream), 1);
        assert_eq!(arr.index_device(1, stream).item::<i32>(&stream), 2);
        assert_eq!(arr.index_device(2, stream).item::<i32>(&stream), 3);
    }

    #[test]
    fn test_array_2d() {
        let stream = crate::test_stream();
        let a = array!([[1, 2, 3], [4, 5, 6]]);

        assert_eq!(a.ndim(), 2);
        assert_eq!(a.shape(), &[2, 3]);
        assert_eq!(a.index_device((0, 0), stream).item::<i32>(&stream), 1);
        assert_eq!(a.index_device((0, 1), stream).item::<i32>(&stream), 2);
        assert_eq!(a.index_device((0, 2), stream).item::<i32>(&stream), 3);
        assert_eq!(a.index_device((1, 0), stream).item::<i32>(&stream), 4);
        assert_eq!(a.index_device((1, 1), stream).item::<i32>(&stream), 5);
        assert_eq!(a.index_device((1, 2), stream).item::<i32>(&stream), 6);
    }

    #[test]
    fn test_array_3d() {
        let stream = crate::test_stream();
        let a = array!([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]);

        assert!(a.ndim() == 3);
        assert_eq!(a.shape(), &[2, 2, 3]);
        assert_eq!(a.index_device((0, 0, 0), stream).item::<i32>(&stream), 1);
        assert_eq!(a.index_device((0, 0, 1), stream).item::<i32>(&stream), 2);
        assert_eq!(a.index_device((0, 0, 2), stream).item::<i32>(&stream), 3);
        assert_eq!(a.index_device((0, 1, 0), stream).item::<i32>(&stream), 4);
        assert_eq!(a.index_device((0, 1, 1), stream).item::<i32>(&stream), 5);
        assert_eq!(a.index_device((0, 1, 2), stream).item::<i32>(&stream), 6);
        assert_eq!(a.index_device((1, 0, 0), stream).item::<i32>(&stream), 7);
        assert_eq!(a.index_device((1, 0, 1), stream).item::<i32>(&stream), 8);
        assert_eq!(a.index_device((1, 0, 2), stream).item::<i32>(&stream), 9);
        assert_eq!(a.index_device((1, 1, 0), stream).item::<i32>(&stream), 10);
        assert_eq!(a.index_device((1, 1, 1), stream).item::<i32>(&stream), 11);
        assert_eq!(a.index_device((1, 1, 2), stream).item::<i32>(&stream), 12);
    }

    #[test]
    fn test_array_with_shape() {
        let stream = crate::test_stream();
        let a = array!([1, 2, 3, 4], shape = [2, 2]);

        assert_eq!(a.ndim(), 2);
        assert_eq!(a.shape(), &[2, 2]);
        assert_eq!(a.index_device((0, 0), stream).item::<i32>(&stream), 1);
        assert_eq!(a.index_device((0, 1), stream).item::<i32>(&stream), 2);
        assert_eq!(a.index_device((1, 0), stream).item::<i32>(&stream), 3);
        assert_eq!(a.index_device((1, 1), stream).item::<i32>(&stream), 4);
    }
}
