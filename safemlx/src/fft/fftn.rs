use safemlx_internal_macros::generate_macro;

use crate::{
    array::Array,
    error::Result,
    utils::{guard::Guarded, IntoOption},
    Stream,
};

use super::utils::{resolve_size_and_axis_unchecked, resolve_sizes_and_axes_unchecked};

const DEFAULT_NORM: safemlx_sys::mlx_fft_norm = safemlx_sys::mlx_fft_norm__MLX_FFT_NORM_BACKWARD;

/// One dimensional discrete Fourier Transform.
///
/// # Params
///
/// - `a`: The input array.
/// - `n`: Size of the transformed axis. The corresponding axis in the input is truncated or padded
///   with zeros to match `n`. The default value is `a.shape[axis]`.
/// - `axis`: Axis along which to perform the FFT. The default is -1.
#[generate_macro(customize(root = "$crate::fft"))]
pub fn fft(
    a: impl AsRef<Array>,
    #[optional] n: impl Into<Option<i32>>,
    #[optional] axis: impl Into<Option<i32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let (n, axis) = resolve_size_and_axis_unchecked(a, n.into(), axis.into());
    Array::try_from_op(|res| unsafe {
        safemlx_sys::mlx_fft_fft(
            res,
            a.as_ptr(),
            n,
            axis,
            DEFAULT_NORM,
            stream.as_ref().as_ptr(),
        )
    })
}

/// Two dimensional discrete Fourier Transform.
///
/// # Params
///
/// - `a`: The input array.
/// - `s`: Size of the transformed axes. The corresponding axes in the input are truncated or padded
/// with zeros to match `s`. The default value is the sizes of `a` along `axes`.
/// - `axes`: Axes along which to perform the FFT. The default is `[-2, -1]`.
#[generate_macro(customize(root = "$crate::fft"))]
pub fn fft2<'a>(
    a: impl AsRef<Array>,
    #[optional] s: impl IntoOption<&'a [i32]>,
    #[optional] axes: impl IntoOption<&'a [i32]>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let axes = axes.into_option().unwrap_or(&[-2, -1]);
    let (s, axes) = resolve_sizes_and_axes_unchecked(a, s.into_option(), Some(axes));

    let num_s = s.len();
    let num_axes = axes.len();

    let s_ptr = s.as_ptr();
    let axes_ptr = axes.as_ptr();

    Array::try_from_op(|res| unsafe {
        safemlx_sys::mlx_fft_fft2(
            res,
            a.as_ptr(),
            s_ptr,
            num_s,
            axes_ptr,
            num_axes,
            DEFAULT_NORM,
            stream.as_ref().as_ptr(),
        )
    })
}

/// n-dimensional discrete Fourier Transform.
///
/// # Params
///
/// - `a`: The input array.
/// - `s`: Sizes of the transformed axes. The corresponding axes in the input are truncated or
/// padded with zeros to match the sizes in `s`. The default value is the sizes of `a` along `axes`
/// if not specified.
/// - `axes`: Axes along which to perform the FFT. The default is `None` in which case the FFT is
/// over the last `len(s)` axes are or all axes if `s` is also `None`.
#[generate_macro(customize(root = "$crate::fft"))]
pub fn fftn<'a>(
    a: impl AsRef<Array>,
    #[optional] s: impl IntoOption<&'a [i32]>,
    #[optional] axes: impl IntoOption<&'a [i32]>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let (s, axes) = resolve_sizes_and_axes_unchecked(a, s.into_option(), axes.into_option());
    let num_s = s.len();
    let num_axes = axes.len();

    let s_ptr = s.as_ptr();
    let axes_ptr = axes.as_ptr();

    Array::try_from_op(|res| unsafe {
        safemlx_sys::mlx_fft_fftn(
            res,
            a.as_ptr(),
            s_ptr,
            num_s,
            axes_ptr,
            num_axes,
            DEFAULT_NORM,
            stream.as_ref().as_ptr(),
        )
    })
}

/// One dimensional inverse discrete Fourier Transform.
///
/// # Params
///
/// - `a`: Input array.
/// - `n`: Size of the transformed axis. The corresponding axis in the input is truncated or padded
///  with zeros to match `n`. The default value is `a.shape[axis]` if not specified.
/// - `axis`: Axis along which to perform the FFT. The default is `-1` if not specified.
#[generate_macro(customize(root = "$crate::fft"))]
pub fn ifft(
    a: impl AsRef<Array>,
    #[optional] n: impl Into<Option<i32>>,
    #[optional] axis: impl Into<Option<i32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let (n, axis) = resolve_size_and_axis_unchecked(a, n.into(), axis.into());

    Array::try_from_op(|res| unsafe {
        safemlx_sys::mlx_fft_ifft(
            res,
            a.as_ptr(),
            n,
            axis,
            DEFAULT_NORM,
            stream.as_ref().as_ptr(),
        )
    })
}

/// Two dimensional inverse discrete Fourier Transform.
///
/// # Params
///
/// - `a`: The input array.
/// - `s`: Size of the transformed axes. The corresponding axes in the input are truncated or padded
/// with zeros to match `s`. The default value is the sizes of `a` along `axes`.
/// - `axes`: Axes along which to perform the FFT. The default is `[-2, -1]`.
#[generate_macro(customize(root = "$crate::fft"))]
pub fn ifft2<'a>(
    a: impl AsRef<Array>,
    #[optional] s: impl IntoOption<&'a [i32]>,
    #[optional] axes: impl IntoOption<&'a [i32]>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let axes = axes.into_option().unwrap_or(&[-2, -1]);
    let (s, axes) = resolve_sizes_and_axes_unchecked(a, s.into_option(), Some(axes));

    let num_s = s.len();
    let num_axes = axes.len();

    let s_ptr = s.as_ptr();
    let axes_ptr = axes.as_ptr();

    Array::try_from_op(|res| unsafe {
        safemlx_sys::mlx_fft_ifft2(
            res,
            a.as_ptr(),
            s_ptr,
            num_s,
            axes_ptr,
            num_axes,
            DEFAULT_NORM,
            stream.as_ref().as_ptr(),
        )
    })
}

/// n-dimensional inverse discrete Fourier Transform.
///
/// # Params
///
/// - `a`: The input array.
/// - `s`: Sizes of the transformed axes. The corresponding axes in the input are truncated or
/// padded with zeros to match the sizes in `s`. The default value is the sizes of `a` along `axes`
/// if not specified.
/// - `axes`: Axes along which to perform the FFT. The default is `None` in which case the FFT is
/// over the last `len(s)` axes are or all axes if `s` is also `None`.
#[generate_macro(customize(root = "$crate::fft"))]
pub fn ifftn<'a>(
    a: impl AsRef<Array>,
    #[optional] s: impl IntoOption<&'a [i32]>,
    #[optional] axes: impl IntoOption<&'a [i32]>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let (s, axes) = resolve_sizes_and_axes_unchecked(a, s.into_option(), axes.into_option());
    let num_s = s.len();
    let num_axes = axes.len();

    let s_ptr = s.as_ptr();
    let axes_ptr = axes.as_ptr();

    Array::try_from_op(|res| unsafe {
        safemlx_sys::mlx_fft_ifftn(
            res,
            a.as_ptr(),
            s_ptr,
            num_s,
            axes_ptr,
            num_axes,
            DEFAULT_NORM,
            stream.as_ref().as_ptr(),
        )
    })
}

#[cfg(test)]
mod tests {
    use crate::{complex64, fft::*, Array, Dtype};

    #[test]
    fn test_fft() {
        let stream = crate::test_stream();
        const FFT_DATA: &[f32] = &[1.0, 2.0, 3.0, 4.0];
        const FFT_SHAPE: &[i32] = &[4];
        const FFT_EXPECTED: &[complex64; 4] = &[
            complex64::new(10.0, 0.0),
            complex64::new(-2.0, 2.0),
            complex64::new(-2.0, 0.0),
            complex64::new(-2.0, -2.0),
        ];

        let array = Array::from_slice(FFT_DATA, FFT_SHAPE);
        let fft = fft(&array, None, None, stream).unwrap();

        assert_eq!(fft.dtype(), Dtype::Complex64);
        assert_eq!(crate::array::eval_vec::<complex64>(&fft), FFT_EXPECTED);

        let ifft = ifft(&fft, None, None, stream).unwrap();

        assert_eq!(ifft.dtype(), Dtype::Complex64);
        assert_eq!(
            crate::array::eval_vec::<complex64>(&ifft),
            FFT_DATA
                .iter()
                .map(|&x| complex64::new(x, 0.0))
                .collect::<Vec<_>>()
        );

        // The original array is not modified and valid
        let data: Vec<f32> = crate::array::eval_vec(&array);
        assert_eq!(data, FFT_DATA);
    }

    #[test]
    fn test_fft2() {
        let stream = crate::test_stream();
        const FFT2_DATA: &[f32] = &[1.0, 1.0, 1.0, 1.0];
        const FFT2_SHAPE: &[i32] = &[2, 2];
        const FFT2_EXPECTED: &[complex64; 4] = &[
            complex64::new(4.0, 0.0),
            complex64::new(0.0, 0.0),
            complex64::new(0.0, 0.0),
            complex64::new(0.0, 0.0),
        ];

        let array = Array::from_slice(FFT2_DATA, FFT2_SHAPE);
        let fft2 = fft2(&array, None, None, stream).unwrap();

        assert_eq!(fft2.dtype(), Dtype::Complex64);
        assert_eq!(crate::array::eval_vec::<complex64>(&fft2), FFT2_EXPECTED);

        let ifft2 = ifft2(&fft2, None, None, stream).unwrap();

        assert_eq!(ifft2.dtype(), Dtype::Complex64);
        assert_eq!(
            crate::array::eval_vec::<complex64>(&ifft2),
            FFT2_DATA
                .iter()
                .map(|&x| complex64::new(x, 0.0))
                .collect::<Vec<_>>()
        );

        // test that previous array is not modified and valid
        let data: Vec<f32> = crate::array::eval_vec(&array);
        assert_eq!(data, FFT2_DATA);
    }

    #[test]
    fn test_fftn() {
        let stream = crate::test_stream();
        const FFTN_DATA: &[f32] = &[1.0; 8];
        const FFTN_SHAPE: &[i32] = &[2, 2, 2];
        const FFTN_EXPECTED: &[complex64; 8] = &[
            complex64::new(8.0, 0.0),
            complex64::new(0.0, 0.0),
            complex64::new(0.0, 0.0),
            complex64::new(0.0, 0.0),
            complex64::new(0.0, 0.0),
            complex64::new(0.0, 0.0),
            complex64::new(0.0, 0.0),
            complex64::new(0.0, 0.0),
        ];

        let array = Array::from_slice(FFTN_DATA, FFTN_SHAPE);
        let fftn = fftn(&array, None, None, stream).unwrap();

        assert_eq!(fftn.dtype(), Dtype::Complex64);
        assert_eq!(crate::array::eval_vec::<complex64>(&fftn), FFTN_EXPECTED);

        let ifftn = ifftn(&fftn, FFTN_SHAPE, &[0, 1, 2], stream).unwrap();

        assert_eq!(ifftn.dtype(), Dtype::Complex64);
        assert_eq!(
            crate::array::eval_vec::<complex64>(&ifftn),
            FFTN_DATA
                .iter()
                .map(|&x| complex64::new(x, 0.0))
                .collect::<Vec<_>>()
        );

        // test that previous array is not modified and valid
        let data: Vec<f32> = crate::array::eval_vec(&array);
        assert_eq!(data, FFTN_DATA);
    }
}
