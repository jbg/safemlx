use crate::error::Result;
use crate::utils::guard::Guarded;
use crate::{Array, Stream};
use safemlx_internal_macros::generate_macro;

impl Array {
    /// Return the cumulative maximum of the elements along the given axis returning an error if the inputs are invalid.
    ///
    /// # Params
    ///
    /// - axis: Optional axis to compute the cumulative maximum over. If unspecified the cumulative maximum of the flattened array is returned.
    /// - reverse: If true, the cumulative maximum is computed in reverse - defaults to false if unspecified.
    /// - inclusive: If true, the i-th element of the output includes the i-th element of the input - defaults to true if unspecified.
    ///
    /// # Example
    ///
    /// ```rust
    /// # let stream = safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    /// use safemlx::Array;
    /// let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
    ///
    /// // result is [[5, 8], [5, 9]] -- cumulative max along the columns
    /// let result = array.cummax(0, None, None, &stream).unwrap();
    /// ```
    pub fn cummax(
        &self,
        axis: impl Into<Option<i32>>,
        reverse: impl Into<Option<bool>>,
        inclusive: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        let stream = stream.as_ref();

        match axis.into() {
            Some(axis) => Array::try_from_op(|res| unsafe {
                safemlx_sys::mlx_cummax(
                    res,
                    self.as_ptr(),
                    axis,
                    reverse.into().unwrap_or(false),
                    inclusive.into().unwrap_or(true),
                    stream.as_ptr(),
                )
            }),
            None => {
                let shape = &[-1];
                let flat = self.reshape(shape, stream)?;
                Array::try_from_op(|res| unsafe {
                    safemlx_sys::mlx_cummax(
                        res,
                        flat.as_ptr(),
                        0,
                        reverse.into().unwrap_or(false),
                        inclusive.into().unwrap_or(true),
                        stream.as_ptr(),
                    )
                })
            }
        }
    }

    /// Return the cumulative minimum of the elements along the given axis returning an error if the inputs are invalid.
    ///
    /// # Params
    ///
    /// - axis: Optional axis to compute the cumulative minimum over. If unspecified the cumulative maximum of the flattened array is returned.
    /// - reverse: If true, the cumulative minimum is computed in reverse - defaults to false if unspecified.
    /// - inclusive: If true, the i-th element of the output includes the i-th element of the input - defaults to true if unspecified.
    ///
    /// # Example
    ///
    /// ```rust
    /// # let stream = safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    /// use safemlx::Array;
    /// let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
    ///
    /// // result is [[5, 8], [4, 8]] -- cumulative min along the columns
    /// let result = array.cummin(0, None, None, &stream).unwrap();
    /// ```
    pub fn cummin(
        &self,
        axis: impl Into<Option<i32>>,
        reverse: impl Into<Option<bool>>,
        inclusive: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        let stream = stream.as_ref();

        match axis.into() {
            Some(axis) => Array::try_from_op(|res| unsafe {
                safemlx_sys::mlx_cummin(
                    res,
                    self.as_ptr(),
                    axis,
                    reverse.into().unwrap_or(false),
                    inclusive.into().unwrap_or(true),
                    stream.as_ptr(),
                )
            }),
            None => {
                let shape = &[-1];
                let flat = self.reshape(shape, stream)?;
                Array::try_from_op(|res| unsafe {
                    safemlx_sys::mlx_cummin(
                        res,
                        flat.as_ptr(),
                        0,
                        reverse.into().unwrap_or(false),
                        inclusive.into().unwrap_or(true),
                        stream.as_ptr(),
                    )
                })
            }
        }
    }

    /// Return the cumulative product of the elements along the given axis returning an error if the inputs are invalid.
    ///
    /// # Params
    ///
    /// - axis: Optional axis to compute the cumulative product over. If unspecified the cumulative maximum of the flattened array is returned.
    /// - reverse: If true, the cumulative product is computed in reverse - defaults to false if unspecified.
    /// - inclusive: If true, the i-th element of the output includes the i-th element of the input - defaults to true if unspecified.
    ///
    /// # Example
    ///
    /// ```rust
    /// # let stream = safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    /// use safemlx::Array;
    /// let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
    ///
    /// // result is [[5, 8], [20, 72]] -- cumulative min along the columns
    /// let result = array.cumprod(0, None, None, &stream).unwrap();
    /// ```
    pub fn cumprod(
        &self,
        axis: impl Into<Option<i32>>,
        reverse: impl Into<Option<bool>>,
        inclusive: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        let stream = stream.as_ref();

        match axis.into() {
            Some(axis) => Array::try_from_op(|res| unsafe {
                safemlx_sys::mlx_cumprod(
                    res,
                    self.as_ptr(),
                    axis,
                    reverse.into().unwrap_or(false),
                    inclusive.into().unwrap_or(true),
                    stream.as_ptr(),
                )
            }),
            None => {
                let shape = &[-1];
                let flat = self.reshape(shape, stream)?;
                Array::try_from_op(|res| unsafe {
                    safemlx_sys::mlx_cumprod(
                        res,
                        flat.as_ptr(),
                        0,
                        reverse.into().unwrap_or(false),
                        inclusive.into().unwrap_or(true),
                        stream.as_ptr(),
                    )
                })
            }
        }
    }

    /// Return the cumulative sum of the elements along the given axis returning an error if the inputs are invalid.
    ///
    /// # Params
    ///
    /// - axis: Optional axis to compute the cumulative sum over. If unspecified the cumulative maximum of the flattened array is returned.
    /// - reverse: If true, the cumulative sum is computed in reverse - defaults to false if unspecified.
    /// - inclusive: If true, the i-th element of the output includes the i-th element of the input - defaults to true if unspecified.
    ///
    /// # Example
    ///
    /// ```rust
    /// # let stream = safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    /// use safemlx::Array;
    /// let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
    ///
    /// // result is [[5, 8], [9, 17]] -- cumulative min along the columns
    /// let result = array.cumsum(0, None, None, &stream).unwrap();
    /// ```
    pub fn cumsum(
        &self,
        axis: impl Into<Option<i32>>,
        reverse: impl Into<Option<bool>>,
        inclusive: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        let stream = stream.as_ref();

        match axis.into() {
            Some(axis) => Array::try_from_op(|res| unsafe {
                safemlx_sys::mlx_cumsum(
                    res,
                    self.as_ptr(),
                    axis,
                    reverse.into().unwrap_or(false),
                    inclusive.into().unwrap_or(true),
                    stream.as_ptr(),
                )
            }),
            None => {
                let shape = &[-1];
                let flat = self.reshape(shape, stream)?;
                Array::try_from_op(|res| unsafe {
                    safemlx_sys::mlx_cumsum(
                        res,
                        flat.as_ptr(),
                        0,
                        reverse.into().unwrap_or(false),
                        inclusive.into().unwrap_or(true),
                        stream.as_ptr(),
                    )
                })
            }
        }
    }
}

/// See [`Array::cummax`]
#[generate_macro]
pub fn cummax(
    a: impl AsRef<Array>,
    #[optional] axis: impl Into<Option<i32>>,
    #[optional] reverse: impl Into<Option<bool>>,
    #[optional] inclusive: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    a.as_ref().cummax(axis, reverse, inclusive, stream)
}

/// See [`Array::cummin`]
#[generate_macro]
pub fn cummin(
    a: impl AsRef<Array>,
    #[optional] axis: impl Into<Option<i32>>,
    #[optional] reverse: impl Into<Option<bool>>,
    #[optional] inclusive: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    a.as_ref().cummin(axis, reverse, inclusive, stream)
}

/// See [`Array::cumprod`]
#[generate_macro]
pub fn cumprod(
    a: impl AsRef<Array>,
    #[optional] axis: impl Into<Option<i32>>,
    #[optional] reverse: impl Into<Option<bool>>,
    #[optional] inclusive: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    a.as_ref().cumprod(axis, reverse, inclusive, stream)
}

/// See [`Array::cumsum`]
#[generate_macro]
pub fn cumsum(
    a: impl AsRef<Array>,
    #[optional] axis: impl Into<Option<i32>>,
    #[optional] reverse: impl Into<Option<bool>>,
    #[optional] inclusive: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    a.as_ref().cumsum(axis, reverse, inclusive, stream)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_cummax() {
        let stream = crate::test_stream();
        let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);

        let result = array.cummax(0, None, None, stream).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(crate::array::eval_vec::<i32>(&result), &[5, 8, 5, 9]);

        let result = array.cummax(1, None, None, stream).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(crate::array::eval_vec::<i32>(&result), &[5, 8, 4, 9]);

        let result = array.cummax(None, None, None, stream).unwrap();
        assert_eq!(result.shape(), &[4]);
        assert_eq!(crate::array::eval_vec::<i32>(&result), &[5, 8, 8, 9]);

        let result = array.cummax(0, Some(true), None, stream).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(crate::array::eval_vec::<i32>(&result), &[5, 9, 4, 9]);

        let result = array.cummax(0, None, Some(true), stream).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(crate::array::eval_vec::<i32>(&result), &[5, 8, 5, 9]);
    }

    #[test]
    fn test_cummax_out_of_bounds() {
        let stream = crate::test_stream();
        let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
        let result = array.cummax(2, None, None, stream);
        assert!(result.is_err());
    }

    #[test]
    fn test_cummin() {
        let stream = crate::test_stream();
        let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);

        let result = array.cummin(0, None, None, stream).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(crate::array::eval_vec::<i32>(&result), &[5, 8, 4, 8]);

        let result = array.cummin(1, None, None, stream).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(crate::array::eval_vec::<i32>(&result), &[5, 5, 4, 4]);

        let result = array.cummin(None, None, None, stream).unwrap();
        assert_eq!(result.shape(), &[4]);
        assert_eq!(crate::array::eval_vec::<i32>(&result), &[5, 5, 4, 4]);

        let result = array.cummin(0, Some(true), None, stream).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(crate::array::eval_vec::<i32>(&result), &[4, 8, 4, 9]);

        let result = array.cummin(0, None, Some(true), stream).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(crate::array::eval_vec::<i32>(&result), &[5, 8, 4, 8]);
    }

    #[test]
    fn test_cummin_out_of_bounds() {
        let stream = crate::test_stream();
        let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
        let result = array.cummin(2, None, None, stream);
        assert!(result.is_err());
    }

    #[test]
    fn test_cumprod() {
        let stream = crate::test_stream();
        let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);

        let result = array.cumprod(0, None, None, stream).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(crate::array::eval_vec::<i32>(&result), &[5, 8, 20, 72]);

        let result = array.cumprod(1, None, None, stream).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(crate::array::eval_vec::<i32>(&result), &[5, 40, 4, 36]);

        let result = array.cumprod(None, None, None, stream).unwrap();
        assert_eq!(result.shape(), &[4]);
        assert_eq!(crate::array::eval_vec::<i32>(&result), &[5, 40, 160, 1440]);

        let result = array.cumprod(0, Some(true), None, stream).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(crate::array::eval_vec::<i32>(&result), &[20, 72, 4, 9]);

        let result = array.cumprod(0, None, Some(true), stream).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(crate::array::eval_vec::<i32>(&result), &[5, 8, 20, 72]);
    }

    #[test]
    fn test_cumprod_out_of_bounds() {
        let stream = crate::test_stream();
        let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
        let result = array.cumprod(2, None, None, stream);
        assert!(result.is_err());
    }

    #[test]
    fn test_cumsum() {
        let stream = crate::test_stream();
        let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);

        let result = array.cumsum(0, None, None, stream).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(crate::array::eval_vec::<i32>(&result), &[5, 8, 9, 17]);

        let result = array.cumsum(1, None, None, stream).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(crate::array::eval_vec::<i32>(&result), &[5, 13, 4, 13]);

        let result = array.cumsum(None, None, None, stream).unwrap();
        assert_eq!(result.shape(), &[4]);
        assert_eq!(crate::array::eval_vec::<i32>(&result), &[5, 13, 17, 26]);

        let result = array.cumsum(0, Some(true), None, stream).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(crate::array::eval_vec::<i32>(&result), &[9, 17, 4, 9]);

        let result = array.cumsum(0, None, Some(true), stream).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(crate::array::eval_vec::<i32>(&result), &[5, 8, 9, 17]);
    }

    #[test]
    fn test_cumsum_out_of_bounds() {
        let stream = crate::test_stream();
        let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
        let result = array.cumsum(2, None, None, stream);
        assert!(result.is_err());
    }
}
