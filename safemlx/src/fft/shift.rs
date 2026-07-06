use safemlx_internal_macros::generate_macro;
use smallvec::SmallVec;

use crate::{
    array::Array, constants::DEFAULT_STACK_VEC_LEN, error::Result, utils::guard::Guarded,
    utils::IntoOption, Stream,
};

/// Resolve axes for shift operations - when None, returns all axes
fn resolve_axes(a: &Array, axes: Option<&[i32]>) -> SmallVec<[i32; DEFAULT_STACK_VEC_LEN]> {
    match axes {
        Some(axes) => SmallVec::from_slice(axes),
        None => (0..a.ndim() as i32).collect(),
    }
}

/// Shift the zero-frequency component to the center of the spectrum.
///
/// This function swaps half-spaces for all axes listed (defaults to all).
/// Note that `y[0]` is the Nyquist component only if `len(x)` is even.
///
/// # Params
///
/// - `a`: The input array.
/// - `axes`: Axes over which to shift. The default is `None` which shifts all axes.
///
/// # Example
///
/// ```rust
/// # let stream = safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
/// use safemlx::{Array, fft::*};
///
/// let a = Array::from_slice(&[0.0f32, 1.0, 2.0, 3.0, 4.0, -4.0, -3.0, -2.0, -1.0], &[9]);
/// let shifted = fftshift(&a, None, &stream).unwrap();
/// // shifted contains: [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
/// ```
#[generate_macro(customize(root = "$crate::fft"))]
pub fn fftshift<'a>(
    a: impl AsRef<Array>,
    #[optional] axes: impl IntoOption<&'a [i32]>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let axes = resolve_axes(a, axes.into_option());

    Array::try_from_op(|res| unsafe {
        safemlx_sys::mlx_fft_fftshift(
            res,
            a.as_ptr(),
            axes.as_ptr(),
            axes.len(),
            stream.as_ref().as_ptr(),
        )
    })
}

/// The inverse of `fftshift`.
///
/// Although identical for even-length `x`, the functions differ by one sample for odd-length `x`.
///
/// # Params
///
/// - `a`: The input array.
/// - `axes`: Axes over which to calculate. The default is `None` which shifts all axes.
///
/// # Example
///
/// ```rust
/// # let stream = safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
/// use safemlx::{Array, fft::*};
///
/// let a = Array::from_slice(&[-4.0f32, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0], &[9]);
/// let unshifted = ifftshift(&a, None, &stream).unwrap();
/// // unshifted contains: [0.0, 1.0, 2.0, 3.0, 4.0, -4.0, -3.0, -2.0, -1.0]
/// ```
#[generate_macro(customize(root = "$crate::fft"))]
pub fn ifftshift<'a>(
    a: impl AsRef<Array>,
    #[optional] axes: impl IntoOption<&'a [i32]>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let axes = resolve_axes(a, axes.into_option());

    Array::try_from_op(|res| unsafe {
        safemlx_sys::mlx_fft_ifftshift(
            res,
            a.as_ptr(),
            axes.as_ptr(),
            axes.len(),
            stream.as_ref().as_ptr(),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random;

    // Helper to check fftshift matches expected behavior
    fn check_fftshift(a: &Array, axes: Option<&[i32]>, stream: &Stream) {
        let shifted = fftshift(a, axes, stream).unwrap();
        let unshifted = ifftshift(&shifted, axes, stream).unwrap();
        assert!(
            unshifted
                .all_close(a, 1e-5, 1e-6, None, stream)
                .unwrap()
                .item::<bool>(&stream),
            "ifftshift(fftshift(x)) should equal x"
        );
    }

    #[test]
    fn test_fftshift_1d() {
        // Test 1D arrays (matches Python test)
        let stream = crate::test_stream();
        let key = random::key(42).unwrap();
        let r = random::uniform::<_, f32>(0.0, 1.0, &[100], &key, stream).unwrap();
        check_fftshift(&r, None, stream);
    }

    #[test]
    fn test_fftshift_with_axes() {
        // Test with specific axis (matches Python test)
        let stream = crate::test_stream();
        let key = random::key(42).unwrap();
        let r = random::uniform::<_, f32>(0.0, 1.0, &[4, 6], &key, stream).unwrap();
        check_fftshift(&r, Some(&[0]), stream);
        check_fftshift(&r, Some(&[1]), stream);
        check_fftshift(&r, Some(&[0, 1]), stream);
    }

    #[test]
    fn test_fftshift_negative_axes() {
        // Test with negative axes (matches Python test)
        let stream = crate::test_stream();
        let key = random::key(42).unwrap();
        let r = random::uniform::<_, f32>(0.0, 1.0, &[4, 6], &key, stream).unwrap();
        check_fftshift(&r, Some(&[-1]), stream);
    }

    #[test]
    fn test_fftshift_odd_lengths() {
        // Test with odd lengths (matches Python test)
        let stream = crate::test_stream();
        let key = random::key(42).unwrap();
        let r = random::uniform::<_, f32>(0.0, 1.0, &[5, 7], &key, stream).unwrap();
        check_fftshift(&r, None, stream);
        check_fftshift(&r, Some(&[0]), stream);
    }

    #[test]
    fn test_ifftshift_1d() {
        // Test 1D arrays (matches Python test)
        let stream = crate::test_stream();
        let key = random::key(42).unwrap();
        let r = random::uniform::<_, f32>(0.0, 1.0, &[100], &key, stream).unwrap();

        let shifted = ifftshift(&r, None, stream).unwrap();
        let unshifted = fftshift(&shifted, None, stream).unwrap();
        assert!(
            unshifted
                .all_close(&r, 1e-5, 1e-6, None, stream)
                .unwrap()
                .item::<bool>(&stream),
            "fftshift(ifftshift(x)) should equal x"
        );
    }

    #[test]
    fn test_ifftshift_with_axes() {
        // Test with specific axis (matches Python test)
        let stream = crate::test_stream();
        let key = random::key(42).unwrap();
        let r = random::uniform::<_, f32>(0.0, 1.0, &[4, 6], &key, stream).unwrap();

        for axes in [&[0][..], &[1][..], &[0, 1][..]] {
            let shifted = ifftshift(&r, axes, stream).unwrap();
            let unshifted = fftshift(&shifted, axes, stream).unwrap();
            assert!(
                unshifted
                    .all_close(&r, 1e-5, 1e-6, None, stream)
                    .unwrap()
                    .item::<bool>(&stream),
                "fftshift(ifftshift(x)) should equal x for axes {:?}",
                axes
            );
        }
    }

    #[test]
    fn test_ifftshift_negative_axes() {
        // Test with negative axes (matches Python test)
        let stream = crate::test_stream();
        let key = random::key(42).unwrap();
        let r = random::uniform::<_, f32>(0.0, 1.0, &[4, 6], &key, stream).unwrap();

        let shifted = ifftshift(&r, &[-1], stream).unwrap();
        let unshifted = fftshift(&shifted, &[-1], stream).unwrap();
        assert!(unshifted
            .all_close(&r, 1e-5, 1e-6, None, stream)
            .unwrap()
            .item::<bool>(&stream),);
    }

    #[test]
    fn test_ifftshift_odd_lengths() {
        // Test with odd lengths (matches Python test)
        let stream = crate::test_stream();
        let key = random::key(42).unwrap();
        let r = random::uniform::<_, f32>(0.0, 1.0, &[5, 7], &key, stream).unwrap();

        let shifted = ifftshift(&r, None, stream).unwrap();
        let unshifted = fftshift(&shifted, None, stream).unwrap();
        assert!(unshifted
            .all_close(&r, 1e-5, 1e-6, None, stream)
            .unwrap()
            .item::<bool>(&stream),);

        let shifted = ifftshift(&r, &[0], stream).unwrap();
        let unshifted = fftshift(&shifted, &[0], stream).unwrap();
        assert!(unshifted
            .all_close(&r, 1e-5, 1e-6, None, stream)
            .unwrap()
            .item::<bool>(&stream),);
    }

    #[test]
    fn test_fftshift_empty_array() {
        // Test empty array (matches Python test)
        let stream = crate::test_stream();
        let x = Array::from_slice::<f32>(&[], &[0]);
        let shifted = fftshift(&x, None, stream).unwrap();
        assert!(shifted
            .array_eq(&x, None, stream)
            .unwrap()
            .item::<bool>(&stream));
    }
}
