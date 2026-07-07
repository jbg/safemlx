use std::borrow::Cow;

use crate::array::Array;
use crate::error::{Exception, Result};
use crate::utils::axes_or_default_to_all;
use crate::utils::guard::Guarded;
use crate::Stream;
use safemlx_internal_macros::generate_macro;

use super::{factory::zeros_dtype, indexing::scatter_add};

fn resolve_axis(axis: i32, ndim: usize) -> Result<usize> {
    let resolved = if axis < 0 { axis + ndim as i32 } else { axis };
    if resolved < 0 || resolved as usize >= ndim {
        return Err(Exception::custom(format!(
            "axis {axis} is out of bounds for array with {ndim} dimensions"
        )));
    }
    Ok(resolved as usize)
}

impl Array {
    /// An `and` reduction over the given axes returning an error if the axes are invalid.
    ///
    /// # Params
    ///
    /// - axes: The axes to reduce over -- defaults to all axes if not provided
    /// - keep_dims: Whether to keep the reduced dimensions -- defaults to false if not provided
    ///
    /// # Example
    ///
    /// ```rust
    /// # let stream = safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    /// use safemlx::Array;
    /// let a = Array::from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], &[3, 4]);
    /// let mut b = a.all_axes(&[0], None, &stream).unwrap();
    ///
    /// let b = b.evaluated().unwrap();
    /// let results: &[bool] = b.as_slice();
    /// // results == [false, true, true, true]
    /// ```
    pub fn all_axes(
        &self,
        axes: &[i32],
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            safemlx_sys::mlx_all_axes(
                res,
                self.as_ptr(),
                axes.as_ptr(),
                axes.len(),
                keep_dims.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Similar to [`Array::all_axes`] but only reduces over a single axis.
    pub fn all_axis(
        &self,
        axis: i32,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            safemlx_sys::mlx_all_axis(
                res,
                self.as_ptr(),
                axis,
                keep_dims.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Similar to [`Array::all_axes`] but reduces over all axes.
    pub fn all(
        &self,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            safemlx_sys::mlx_all(
                res,
                self.as_ptr(),
                keep_dims.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// A `product` reduction over the given axes returning an error if the axes are invalid.
    ///
    /// # Params
    ///
    /// - axes: axes to reduce over
    /// - keep_dims: Whether to keep the reduced dimensions -- defaults to false if not provided
    ///
    /// # Example
    ///
    /// ```rust
    /// # let stream = safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    /// use safemlx::Array;
    /// let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
    ///
    /// // result is [20, 72]
    /// let result = array.prod_axes(&[0], None, &stream).unwrap();
    /// ```
    pub fn prod_axes(
        &self,
        axes: &[i32],
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            safemlx_sys::mlx_prod_axes(
                res,
                self.as_ptr(),
                axes.as_ptr(),
                axes.len(),
                keep_dims.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Similar to [`Array::prod_axes`] but only reduces over a single axis.
    pub fn prod_axis(
        &self,
        axis: i32,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            safemlx_sys::mlx_prod_axis(
                res,
                self.as_ptr(),
                axis,
                keep_dims.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Similar to [`Array::prod_axes`] but reduces over all axes.
    pub fn prod(
        &self,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            safemlx_sys::mlx_prod(
                res,
                self.as_ptr(),
                keep_dims.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// A `max` reduction over the given axes returning an error if the axes are invalid.
    ///
    /// # Params
    ///
    /// - axes: axes to reduce over
    /// - keep_dims: Whether to keep the reduced dimensions -- defaults to false if not provided
    ///
    /// # Example
    ///
    /// ```rust
    /// # let stream = safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    /// use safemlx::Array;
    /// let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
    ///
    /// // result is [5, 9]
    /// let result = array.max_axes(&[0], None, &stream).unwrap();
    /// ```
    pub fn max_axes(
        &self,
        axes: &[i32],
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            safemlx_sys::mlx_max_axes(
                res,
                self.as_ptr(),
                axes.as_ptr(),
                axes.len(),
                keep_dims.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Similar to [`Array::max_axes`] but only reduces over a single axis.
    pub fn max_axis(
        &self,
        axis: i32,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            safemlx_sys::mlx_max_axis(
                res,
                self.as_ptr(),
                axis,
                keep_dims.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Similar to [`Array::max_axes`] but reduces over all axes.
    pub fn max(
        &self,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            safemlx_sys::mlx_max(
                res,
                self.as_ptr(),
                keep_dims.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Sum reduce the array over the given axes returning an error if the axes are invalid.
    ///
    /// # Params
    ///
    /// - axes: axes to reduce over
    /// - keep_dims: if `true`, keep the reduces axes as singleton dimensions
    ///
    /// # Example
    ///
    /// ```rust
    /// # let stream = safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    /// use safemlx::Array;
    /// let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
    ///
    /// // result is [9, 17]
    /// let result = array.sum_axes(&[0], None, &stream).unwrap();
    /// ```
    pub fn sum_axes(
        &self,
        axes: &[i32],
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            safemlx_sys::mlx_sum_axes(
                res,
                self.as_ptr(),
                axes.as_ptr(),
                axes.len(),
                keep_dims.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Similar to [`Array::sum_axes`] but only reduces over a single axis.
    pub fn sum_axis(
        &self,
        axis: i32,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            safemlx_sys::mlx_sum_axis(
                res,
                self.as_ptr(),
                axis,
                keep_dims.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Similar to [`Array::sum_axes`] but reduces over all axes.
    pub fn sum(
        &self,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            safemlx_sys::mlx_sum(
                res,
                self.as_ptr(),
                keep_dims.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Sum values by integer segment ids along one axis.
    ///
    /// This creates an output with `shape[axis] == num_segments` and accumulates `self` into that
    /// output using `segment_ids` along `axis`. Duplicate segment ids are summed. The output dtype
    /// is the dtype of `self`.
    pub fn segment_sum(
        &self,
        segment_ids: impl AsRef<Array>,
        num_segments: i32,
        axis: i32,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        let axis = resolve_axis(axis, self.ndim())?;
        let mut shape = self.shape().to_vec();
        shape[axis] = num_segments;
        let segment_ids = segment_ids.as_ref();
        let segment_ids = if segment_ids.ndim() == 1 && self.ndim() > 1 {
            let mut index_shape = vec![1; self.ndim()];
            index_shape[axis] = self.dim(axis as i32);
            Cow::Owned(segment_ids.reshape(&index_shape, &stream)?)
        } else {
            Cow::Borrowed(segment_ids)
        };
        let base = zeros_dtype(&shape, self.dtype(), &stream)?;
        scatter_add(base, segment_ids.as_ref(), self, axis as i32, stream)
    }

    /// A `mean` reduction over the given axes returning an error if the axes are invalid.
    ///
    /// # Params
    ///
    /// - axes: axes to reduce over
    /// - keep_dims: Whether to keep the reduced dimensions -- defaults to false if not provided
    ///
    /// # Example
    ///
    /// ```rust
    /// # let stream = safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    /// use safemlx::Array;
    /// let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
    ///
    /// // result is [4.5, 8.5]
    /// let result = array.mean_axes(&[0], None, &stream).unwrap();
    /// ```
    pub fn mean_axes(
        &self,
        axes: &[i32],
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        let axes = axes_or_default_to_all(axes, self.ndim() as i32);
        Array::try_from_op(|res| unsafe {
            safemlx_sys::mlx_mean_axes(
                res,
                self.as_ptr(),
                axes.as_ptr(),
                axes.len(),
                keep_dims.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Similar to [`Array::mean_axes`] but only reduces over a single axis.
    pub fn mean_axis(
        &self,
        axis: i32,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            safemlx_sys::mlx_mean_axis(
                res,
                self.as_ptr(),
                axis,
                keep_dims.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Similar to [`Array::mean_axes`] but reduces over all axes.
    pub fn mean(
        &self,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            safemlx_sys::mlx_mean(
                res,
                self.as_ptr(),
                keep_dims.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// A `min` reduction over the given axes returning an error if the axes are invalid.
    ///
    /// # Params
    ///
    /// - axes: axes to reduce over
    /// - keep_dims: Whether to keep the reduced dimensions -- defaults to false if not provided
    ///
    /// # Example
    ///
    /// ```rust
    /// # let stream = safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    /// use safemlx::Array;
    /// let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
    ///
    /// // result is [4, 8]
    /// let result = array.min_axes(&[0], None, &stream).unwrap();
    /// ```
    pub fn min_axes(
        &self,
        axes: &[i32],
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            safemlx_sys::mlx_min_axes(
                res,
                self.as_ptr(),
                axes.as_ptr(),
                axes.len(),
                keep_dims.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Similar to [`Array::min_axes`] but only reduces over a single axis.
    pub fn min_axis(
        &self,
        axis: i32,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            safemlx_sys::mlx_min_axis(
                res,
                self.as_ptr(),
                axis,
                keep_dims.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Similar to [`Array::min_axes`] but reduces over all axes.
    pub fn min(
        &self,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            safemlx_sys::mlx_min(
                res,
                self.as_ptr(),
                keep_dims.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Compute the variance(s) over the given axes returning an error if the axes are invalid.
    ///
    /// # Params
    ///
    /// - axes: axes to reduce over
    /// - keep_dims: if `true`, keep the reduces axes as singleton dimensions
    /// - ddof: the divisor to compute the variance is `N - ddof`
    pub fn var_axes(
        &self,
        axes: &[i32],
        keep_dims: impl Into<Option<bool>>,
        ddof: impl Into<Option<i32>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            safemlx_sys::mlx_var_axes(
                res,
                self.as_ptr(),
                axes.as_ptr(),
                axes.len(),
                keep_dims.into().unwrap_or(false),
                ddof.into().unwrap_or(0),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Similar to [`Array::var_axes`] but only reduces over a single axis.
    pub fn var_axis(
        &self,
        axis: i32,
        keep_dims: impl Into<Option<bool>>,
        ddof: impl Into<Option<i32>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            safemlx_sys::mlx_var_axis(
                res,
                self.as_ptr(),
                axis,
                keep_dims.into().unwrap_or(false),
                ddof.into().unwrap_or(0),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Similar to [`Array::var_axes`] but reduces over all axes.
    pub fn var(
        &self,
        keep_dims: impl Into<Option<bool>>,
        ddof: impl Into<Option<i32>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            safemlx_sys::mlx_var(
                res,
                self.as_ptr(),
                keep_dims.into().unwrap_or(false),
                ddof.into().unwrap_or(0),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Compute the median over the given axes.
    ///
    /// # Params
    ///
    /// - axes: axes to reduce over
    /// - keep_dims: Whether to keep the reduced dimensions -- defaults to false if not provided
    pub fn median_axes(
        &self,
        axes: &[i32],
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            safemlx_sys::mlx_median(
                res,
                self.as_ptr(),
                axes.as_ptr(),
                axes.len(),
                keep_dims.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Similar to [`Array::median_axes`] but only reduces over a single axis.
    pub fn median_axis(
        &self,
        axis: i32,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        self.median_axes(&[axis], keep_dims, stream)
    }

    /// Compute the median over all axes.
    pub fn median(
        &self,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        let axes: Vec<i32> = (0..self.ndim() as i32).collect();
        self.median_axes(&axes, keep_dims, stream)
    }

    /// A `log-sum-exp` reduction over the given axes returning an error if the axes are invalid.
    ///
    /// The log-sum-exp reduction is a numerically stable version of using the individual operations.
    ///
    /// # Params
    ///
    /// - axes: axes to reduce over
    /// - keep_dims: Whether to keep the reduced dimensions -- defaults to false if not provided
    pub fn logsumexp_axes(
        &self,
        axes: &[i32],
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            safemlx_sys::mlx_logsumexp_axes(
                res,
                self.as_ptr(),
                axes.as_ptr(),
                axes.len(),
                keep_dims.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Similar to [`Array::logsumexp_axes`] but only reduces over a single axis.
    pub fn logsumexp_axis(
        &self,
        axis: i32,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            safemlx_sys::mlx_logsumexp_axis(
                res,
                self.as_ptr(),
                axis,
                keep_dims.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Similar to [`Array::logsumexp_axes`] but reduces over all axes.
    pub fn logsumexp(
        &self,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            safemlx_sys::mlx_logsumexp(
                res,
                self.as_ptr(),
                keep_dims.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            )
        })
    }
}

/// See [`Array::all_axes`]
#[generate_macro]
pub fn all_axes(
    array: impl AsRef<Array>,
    axes: &[i32],
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    array.as_ref().all_axes(axes, keep_dims, stream)
}

/// See [`Array::all_axis`]
#[generate_macro]
pub fn all_axis(
    array: impl AsRef<Array>,
    axis: i32,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    array.as_ref().all_axis(axis, keep_dims, stream)
}

/// See [`Array::all`]
#[generate_macro]
pub fn all(
    array: impl AsRef<Array>,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    array.as_ref().all(keep_dims, stream)
}

/// See [`Array::prod_axes`]
#[generate_macro]
pub fn prod_axes(
    array: impl AsRef<Array>,
    axes: &[i32],
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    array.as_ref().prod_axes(axes, keep_dims, stream)
}

/// See [`Array::prod_axis`]
#[generate_macro]
pub fn prod_axis(
    array: impl AsRef<Array>,
    axis: i32,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    array.as_ref().prod_axis(axis, keep_dims, stream)
}

/// See [`Array::prod`]
#[generate_macro]
pub fn prod(
    array: impl AsRef<Array>,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    array.as_ref().prod(keep_dims, stream)
}

/// See [`Array::max_axes`]
#[generate_macro]
pub fn max_axes(
    array: impl AsRef<Array>,
    axes: &[i32],
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    array.as_ref().max_axes(axes, keep_dims, stream)
}

/// See [`Array::max_axis`]
#[generate_macro]
pub fn max_axis(
    array: impl AsRef<Array>,
    axis: i32,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    array.as_ref().max_axis(axis, keep_dims, stream)
}

/// See [`Array::max`]
#[generate_macro]
pub fn max(
    array: impl AsRef<Array>,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    array.as_ref().max(keep_dims, stream)
}

/// Compute the standard deviation(s) over the given axes.
///
/// # Params
///
/// - `a`: Input array
/// - `axes`: Optional axis or axes to reduce over. If unspecified this defaults to reducing over
///   the entire array.
/// - `keep_dims`: Keep reduced axes as singleton dimensions, defaults to False.
/// - `ddof`: The divisor to compute the variance is `N - ddof`, defaults to `0`.
#[generate_macro]
pub fn std_axes(
    a: impl AsRef<Array>,
    axes: &[i32],
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] ddof: impl Into<Option<i32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let keep_dims = keep_dims.into().unwrap_or(false);
    let ddof = ddof.into().unwrap_or(0);
    Array::try_from_op(|res| unsafe {
        safemlx_sys::mlx_std_axes(
            res,
            a.as_ptr(),
            axes.as_ptr(),
            axes.len(),
            keep_dims,
            ddof,
            stream.as_ref().as_ptr(),
        )
    })
}

/// Similar to [`std_axes()`] but only reduces over a single axis.
#[generate_macro]
pub fn std_axis(
    a: impl AsRef<Array>,
    axis: i32,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] ddof: impl Into<Option<i32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let keep_dims = keep_dims.into().unwrap_or(false);
    let ddof = ddof.into().unwrap_or(0);
    Array::try_from_op(|res| unsafe {
        safemlx_sys::mlx_std_axis(
            res,
            a.as_ptr(),
            axis,
            keep_dims,
            ddof,
            stream.as_ref().as_ptr(),
        )
    })
}

/// Similar to [`std_axes()`] but reduces over all axes.
#[generate_macro]
pub fn std(
    a: impl AsRef<Array>,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] ddof: impl Into<Option<i32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let keep_dims = keep_dims.into().unwrap_or(false);
    let ddof = ddof.into().unwrap_or(0);
    Array::try_from_op(|res| unsafe {
        safemlx_sys::mlx_std(res, a.as_ptr(), keep_dims, ddof, stream.as_ref().as_ptr())
    })
}

/// See [`Array::sum_axes`]
#[generate_macro]
pub fn sum_axes(
    array: impl AsRef<Array>,
    axes: &[i32],
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    array.as_ref().sum_axes(axes, keep_dims, stream)
}

/// See [`Array::sum_axis`]
#[generate_macro]
pub fn sum_axis(
    array: impl AsRef<Array>,
    axis: i32,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    array.as_ref().sum_axis(axis, keep_dims, stream)
}

/// See [`Array::sum`]
#[generate_macro]
pub fn sum(
    array: impl AsRef<Array>,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    array.as_ref().sum(keep_dims, stream)
}

/// See [`Array::segment_sum`]
#[generate_macro]
pub fn segment_sum(
    array: impl AsRef<Array>,
    segment_ids: impl AsRef<Array>,
    num_segments: i32,
    axis: i32,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    array
        .as_ref()
        .segment_sum(segment_ids, num_segments, axis, stream)
}

/// See [`Array::mean_axes`]
#[generate_macro]
pub fn mean_axes(
    array: impl AsRef<Array>,
    axes: &[i32],
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    array.as_ref().mean_axes(axes, keep_dims, stream)
}

/// See [`Array::mean_axis`]
#[generate_macro]
pub fn mean_axis(
    array: impl AsRef<Array>,
    axis: i32,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    array.as_ref().mean_axis(axis, keep_dims, stream)
}

/// See [`Array::mean`]
#[generate_macro]
pub fn mean(
    array: impl AsRef<Array>,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    array.as_ref().mean(keep_dims, stream)
}

/// See [`Array::min`]
#[generate_macro]
pub fn min_axes(
    array: impl AsRef<Array>,
    axes: &[i32],
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    array.as_ref().min_axes(axes, keep_dims, stream)
}

/// See [`Array::min_axis`]
#[generate_macro]
pub fn min_axis(
    array: impl AsRef<Array>,
    axis: i32,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    array.as_ref().min_axis(axis, keep_dims, stream)
}

/// See [`Array::min`]
#[generate_macro]
pub fn min(
    array: impl AsRef<Array>,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    array.as_ref().min(keep_dims, stream)
}

/// See [`Array::var_axes`]
#[generate_macro]
pub fn var_axes(
    array: impl AsRef<Array>,
    axes: &[i32],
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] ddof: impl Into<Option<i32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    array.as_ref().var_axes(axes, keep_dims, ddof, stream)
}

/// See [`Array::var_axis`]
#[generate_macro]
pub fn var_axis(
    array: impl AsRef<Array>,
    axis: i32,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] ddof: impl Into<Option<i32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    array.as_ref().var_axis(axis, keep_dims, ddof, stream)
}

/// See [`Array::var`]
#[generate_macro]
pub fn var(
    array: impl AsRef<Array>,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] ddof: impl Into<Option<i32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    array.as_ref().var(keep_dims, ddof, stream)
}

/// See [`Array::median_axes`]
#[generate_macro]
pub fn median_axes(
    array: impl AsRef<Array>,
    axes: &[i32],
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    array.as_ref().median_axes(axes, keep_dims, stream)
}

/// See [`Array::median_axis`]
#[generate_macro]
pub fn median_axis(
    array: impl AsRef<Array>,
    axis: i32,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    array.as_ref().median_axis(axis, keep_dims, stream)
}

/// See [`Array::median`]
#[generate_macro]
pub fn median(
    array: impl AsRef<Array>,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    array.as_ref().median(keep_dims, stream)
}

/// See [`Array::logsumexp_axes`]
#[generate_macro]
pub fn logsumexp_axes(
    array: impl AsRef<Array>,
    axes: &[i32],
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    array.as_ref().logsumexp_axes(axes, keep_dims, stream)
}

/// See [`Array::logsumexp_axis`]
#[generate_macro]
pub fn logsumexp_axis(
    array: impl AsRef<Array>,
    axis: i32,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    array.as_ref().logsumexp_axis(axis, keep_dims, stream)
}

/// See [`Array::logsumexp`]
#[generate_macro]
pub fn logsumexp(
    array: impl AsRef<Array>,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    array.as_ref().logsumexp(keep_dims, stream)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_all() {
        let stream = crate::test_stream();
        let array = Array::from_slice(&[true, false, true, false], &[2, 2]);

        assert_eq!(
            array.all(None, stream).unwrap().item::<bool>(&stream),
            false
        );
        assert_eq!(array.all(true, stream).unwrap().shape(), &[1, 1]);
        assert_eq!(
            array
                .all_axes(&[0, 1], None, stream)
                .unwrap()
                .item::<bool>(&stream),
            false
        );

        let result = array.all_axis(0, None, stream).unwrap();
        assert_eq!(crate::array::eval_vec::<bool>(&result), &[true, false]);

        let result = array.all_axis(1, None, stream).unwrap();
        assert_eq!(crate::array::eval_vec::<bool>(&result), &[false, false]);
    }

    #[test]
    fn test_all_empty_axes() {
        let stream = crate::test_stream();
        let array = Array::from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], &[3, 4]);
        let all = array.all_axes(&[], None, stream).unwrap();

        let results: Vec<bool> = crate::array::eval_vec(&all);
        assert_eq!(
            results,
            &[false, true, true, true, true, true, true, true, true, true, true, true]
        );
    }

    #[test]
    fn test_prod() {
        let stream = crate::test_stream();
        let x = Array::from_slice(&[1, 2, 3, 3], &[2, 2]);
        assert_eq!(x.prod(None, stream).unwrap().item::<i32>(&stream), 18);

        let y = x.prod(true, stream).unwrap();
        assert_eq!(y.clone().item::<i32>(&stream), 18);
        assert_eq!(y.shape(), &[1, 1]);

        let result = x.prod_axis(0, None, stream).unwrap();
        assert_eq!(crate::array::eval_vec::<i32>(&result), &[3, 6]);

        let result = x.prod_axis(1, None, stream).unwrap();
        assert_eq!(crate::array::eval_vec::<i32>(&result), &[2, 9])
    }

    #[test]
    fn test_prod_empty_axes() {
        let stream = crate::test_stream();
        let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
        let result = array.prod_axes(&[], None, stream).unwrap();

        let results: Vec<i32> = crate::array::eval_vec(&result);
        assert_eq!(results, &[5, 8, 4, 9]);
    }

    #[test]
    fn test_max() {
        let stream = crate::test_stream();
        let x = Array::from_slice(&[1, 2, 3, 4], &[2, 2]);
        assert_eq!(x.max(None, stream).unwrap().item::<i32>(&stream), 4);
        let y = x.max(true, stream).unwrap();
        assert_eq!(y.clone().item::<i32>(&stream), 4);
        assert_eq!(y.shape(), &[1, 1]);

        let result = x.max_axis(0, None, stream).unwrap();
        assert_eq!(crate::array::eval_vec::<i32>(&result), &[3, 4]);

        let result = x.max_axis(1, None, stream).unwrap();
        assert_eq!(crate::array::eval_vec::<i32>(&result), &[2, 4]);
    }

    #[test]
    fn test_max_empty_axes() {
        let stream = crate::test_stream();
        let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
        let result = array.max_axes(&[], None, stream).unwrap();

        let results: Vec<i32> = crate::array::eval_vec(&result);
        assert_eq!(results, &[5, 8, 4, 9]);
    }

    #[test]
    fn test_sum() {
        let stream = crate::test_stream();
        let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
        let result = array.sum_axis(0, None, stream).unwrap();

        let results: Vec<i32> = crate::array::eval_vec(&result);
        assert_eq!(results, &[9, 17]);
    }

    #[test]
    fn test_sum_empty_axes() {
        let stream = crate::test_stream();
        let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
        let result = array.sum_axes(&[], None, stream).unwrap();

        let results: Vec<i32> = crate::array::eval_vec(&result);
        assert_eq!(results, &[5, 8, 4, 9]);
    }

    #[test]
    fn test_mean() {
        let stream = crate::test_stream();
        let x = Array::from_slice(&[1, 2, 3, 4], &[2, 2]);
        assert_eq!(x.mean(None, stream).unwrap().item::<f32>(&stream), 2.5);
        let y = x.mean(true, stream).unwrap();
        assert_eq!(y.clone().item::<f32>(&stream), 2.5);
        assert_eq!(y.shape(), &[1, 1]);

        let result = x.mean_axis(0, None, stream).unwrap();
        assert_eq!(crate::array::eval_vec::<f32>(&result), &[2.0, 3.0]);

        let result = x.mean_axis(1, None, stream).unwrap();
        assert_eq!(crate::array::eval_vec::<f32>(&result), &[1.5, 3.5]);
    }

    #[test]
    fn test_mean_empty_axes() {
        let stream = crate::test_stream();
        let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
        let result = array.mean_axes(&[], None, stream).unwrap();

        let results: Vec<f32> = crate::array::eval_vec(&result);
        assert_eq!(results, &[5.0, 8.0, 4.0, 9.0]);
    }

    #[test]
    fn test_mean_out_of_bounds() {
        let stream = crate::test_stream();
        let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
        let result = array.mean_axis(2, None, stream);
        assert!(result.is_err());
    }

    #[test]
    fn test_min() {
        let stream = crate::test_stream();
        let x = Array::from_slice(&[1, 2, 3, 4], &[2, 2]);
        assert_eq!(x.min(None, stream).unwrap().item::<i32>(&stream), 1);
        let y = x.min(true, stream).unwrap();
        assert_eq!(y.clone().item::<i32>(&stream), 1);
        assert_eq!(y.shape(), &[1, 1]);

        let result = x.min_axis(0, None, stream).unwrap();
        assert_eq!(crate::array::eval_vec::<i32>(&result), &[1, 2]);

        let result = x.min_axis(1, None, stream).unwrap();
        assert_eq!(crate::array::eval_vec::<i32>(&result), &[1, 3]);
    }

    #[test]
    fn test_min_empty_axes() {
        let stream = crate::test_stream();
        let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
        let result = array.min_axes(&[], None, stream).unwrap();

        let results: Vec<i32> = crate::array::eval_vec(&result);
        assert_eq!(results, &[5, 8, 4, 9]);
    }

    #[test]
    fn test_var() {
        let stream = crate::test_stream();
        let x = Array::from_slice(&[1, 2, 3, 4], &[2, 2]);
        assert_eq!(
            x.var(None, None, stream).unwrap().item::<f32>(&stream),
            1.25
        );
        let y = x.var(true, None, stream).unwrap();
        assert_eq!(y.clone().item::<f32>(&stream), 1.25);
        assert_eq!(y.shape(), &[1, 1]);

        let result = x.var_axis(0, None, None, stream).unwrap();
        assert_eq!(crate::array::eval_vec::<f32>(&result), &[1.0, 1.0]);

        let result = x.var_axis(1, None, None, stream).unwrap();
        assert_eq!(crate::array::eval_vec::<f32>(&result), &[0.25, 0.25]);

        let x = Array::from_slice(&[1.0, 2.0], &[2]);
        let out = x.var(None, Some(3), stream).unwrap();
        assert_eq!(out.item::<f32>(&stream), f32::INFINITY);
    }

    #[test]
    fn test_var_empty_axes() {
        let stream = crate::test_stream();
        let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
        let result = array.var_axes(&[], None, 0, stream).unwrap();

        let results: Vec<f32> = crate::array::eval_vec(&result);
        assert_eq!(results, &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_log_sum_exp() {
        let stream = crate::test_stream();
        let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
        let result = array.logsumexp_axis(0, None, stream).unwrap();

        let results: Vec<f32> = crate::array::eval_vec(&result);
        assert_eq!(results, &[5.3132615, 9.313262]);
    }

    #[test]
    fn test_log_sum_exp_empty_axes() {
        let stream = crate::test_stream();
        let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
        let result = array.logsumexp_axes(&[], None, stream).unwrap();

        let results: Vec<f32> = crate::array::eval_vec(&result);
        assert_eq!(results, &[5.0, 8.0, 4.0, 9.0]);
    }

    #[test]
    fn test_segment_sum() {
        let stream = crate::test_stream();
        let values = Array::from_slice(&[1.0f32, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0], &[4, 2]);
        let segment_ids = Array::from_slice(&[0u32, 1, 0, 2], &[4]);
        let out = segment_sum(&values, &segment_ids, 4, 0, stream).unwrap();
        let expected = Array::from_slice(&[4.0f32, 40.0, 2.0, 20.0, 4.0, 40.0, 0.0, 0.0], &[4, 2]);
        assert!(out
            .all_close(&expected, 1e-5, 1e-5, None, stream)
            .unwrap()
            .item::<bool>(&stream));
        assert_eq!(out.dtype(), values.dtype());

        let values = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let segment_ids = Array::from_slice(&[0u32, 1, 0], &[3]);
        let out = values.segment_sum(&segment_ids, 2, -1, stream).unwrap();
        let expected = Array::from_slice(&[4.0f32, 2.0, 10.0, 5.0], &[2, 2]);
        assert!(out
            .all_close(&expected, 1e-5, 1e-5, None, stream)
            .unwrap()
            .item::<bool>(&stream));
    }

    // Tests adapted from Python test `test_ops.py/test_median`
    #[test]
    fn test_median() {
        let stream = crate::test_stream();
        // Test basic median over all elements (odd count)
        let x = Array::from_slice(&[0, 1, 2, 3, 4], &[5]);
        let out = x.median(None, stream).unwrap();
        assert_eq!(out.shape(), &[] as &[i32]);
        assert_eq!(out.item::<i32>(&stream), 2);

        // Test keepdims
        let out = x.median(true, stream).unwrap();
        assert_eq!(out.shape(), &[1]);

        // Test median with even count (should be average of two middle values)
        let x = Array::from_slice(&[0, 1, 2, 3, 4, 5], &[6]);
        let out = x.median(None, stream).unwrap();
        assert!((out.item::<f32>(&stream) - 2.5).abs() < 1e-5);

        // Test median over specific axes
        use crate::random;
        let key = random::key(0).unwrap();
        let x = random::normal::<f32>(&[5, 5, 5, 5], None, None, &key, stream).unwrap();

        let out = x.median_axes(&[0, 2], true, stream).unwrap();
        assert_eq!(out.shape(), &[1, 5, 1, 5]);

        let out = x.median_axes(&[1, 3], true, stream).unwrap();
        assert_eq!(out.shape(), &[5, 1, 5, 1]);

        // Test single axis
        let x = Array::from_slice(&[1, 5, 2, 4, 3, 6], &[2, 3]);
        let out = x.median_axis(0, None, stream).unwrap();
        assert_eq!(out.shape(), &[3]);

        let out = x.median_axis(1, None, stream).unwrap();
        assert_eq!(out.shape(), &[2]);
    }
}
