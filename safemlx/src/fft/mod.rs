//! Fast Fourier Transform (FFT) and its inverse (IFFT) for one, two, and `N` dimensions.
//!
//! Like all other functions in `mlx-rs`, three variants are provided for each FFT function.
//!
//! The difference are explained below using `fftn` as an example:
//!
//! 1. `fftn_unchecked`/`fftn_device_unchecked`: This function is simply a wrapper around the C API
//!    and does not perform any checks on the input. It may panic or get an fatal error that cannot
//!    be caught by the rust runtime if the input is invalid.
//! 2. `try_fftn`/`try_fftn_device`: This function performs checks on the input and returns a
//!    `Result` instead of panicking.
//! 3. `fftn`/`fftn`: This function is a wrapper around `try_fftn` and unwraps the result. It
//!    panics if the input is invalid.
//!
//! Each operation takes an explicit [`Stream`], which determines where the operation is scheduled.
//!
//! # Examples
//!
//! ## One dimension
//!
//! ```rust
//! # let stream = safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
//! use safemlx::{Dtype, Array, Stream, complex64, fft::*};
//!
//! let src = [1.0f32, 2.0, 3.0, 4.0];
//! let mut array = Array::from_slice(&src[..], &[4]);
//!
//! let mut fft_result = fft(&array, 4, 0, &stream).unwrap();
//! assert_eq!(fft_result.dtype(), Dtype::Complex64);
//!
//! let expected = &[
//!     complex64::new(10.0, 0.0),
//!     complex64::new(-2.0, 2.0),
//!     complex64::new(-2.0, 0.0),
//!     complex64::new(-2.0, -2.0),
//! ];
//! assert_eq!(fft_result.evaluated().unwrap().as_slice::<complex64>(), &expected[..]);
//!
//! let mut ifft_result = ifft(&fft_result, 4, 0, &stream).unwrap();
//! assert_eq!(ifft_result.dtype(), Dtype::Complex64);
//!
//! let expected = &[
//!    complex64::new(1.0, 0.0),
//!    complex64::new(2.0, 0.0),
//!    complex64::new(3.0, 0.0),
//!    complex64::new(4.0, 0.0),
//! ];
//! assert_eq!(ifft_result.evaluated().unwrap().as_slice::<complex64>(), &expected[..]);
//!
//! let mut rfft_result = rfft(&array, 4, 0, &stream).unwrap();
//! assert_eq!(rfft_result.dtype(), Dtype::Complex64);
//!
//! let expected = &[
//!    complex64::new(10.0, 0.0),
//!    complex64::new(-2.0, 2.0),
//!    complex64::new(-2.0, 0.0),
//! ];
//! assert_eq!(rfft_result.evaluated().unwrap().as_slice::<complex64>(), &expected[..]);
//!
//! let mut irfft_result = irfft(&rfft_result, 4, 0, &stream).unwrap();
//! assert_eq!(irfft_result.dtype(), Dtype::Float32);
//! assert_eq!(irfft_result.evaluated().unwrap().as_slice::<f32>(), &src[..]);
//!
//! // The original array is not modified
//! let array = array.evaluated().unwrap();
//! let data: &[f32] = array.as_slice();
//! assert_eq!(data, &src[..]);
//! ```
//!
//! ## Two dimensions
//!
//! ```rust
//! # let stream = safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
//! use safemlx::{Dtype, Array, Stream, complex64, fft::*};
//!
//! let src = [1.0f32, 1.0, 1.0, 1.0];
//! let mut array = Array::from_slice(&src[..], &[2, 2]);
//!
//! let mut fft2_result = fft2(&array, None, None, &stream).unwrap();
//! assert_eq!(fft2_result.dtype(), Dtype::Complex64);
//! let expected = &[
//!     complex64::new(4.0, 0.0),
//!     complex64::new(0.0, 0.0),
//!     complex64::new(0.0, 0.0),
//!     complex64::new(0.0, 0.0),
//! ];
//! assert_eq!(fft2_result.evaluated().unwrap().as_slice::<complex64>(), &expected[..]);
//!
//! let mut ifft2_result = ifft2(&fft2_result, None, None, &stream).unwrap();
//! assert_eq!(ifft2_result.dtype(), Dtype::Complex64);
//!
//! let expected = &[
//!    complex64::new(1.0, 0.0),
//!    complex64::new(1.0, 0.0),
//!    complex64::new(1.0, 0.0),
//!    complex64::new(1.0, 0.0),
//! ];
//! assert_eq!(ifft2_result.evaluated().unwrap().as_slice::<complex64>(), &expected[..]);
//!
//! let mut rfft2_result = rfft2(&array, None, None, &stream).unwrap();
//! assert_eq!(rfft2_result.dtype(), Dtype::Complex64);
//!
//! let expected = &[
//!     complex64::new(4.0, 0.0),
//!     complex64::new(0.0, 0.0),
//!     complex64::new(0.0, 0.0),
//!     complex64::new(0.0, 0.0),
//! ];
//! assert_eq!(rfft2_result.evaluated().unwrap().as_slice::<complex64>(), &expected[..]);
//!
//! let mut irfft2_result = irfft2(&rfft2_result, None, None, &stream).unwrap();
//! assert_eq!(irfft2_result.dtype(), Dtype::Float32);
//! assert_eq!(irfft2_result.evaluated().unwrap().as_slice::<f32>(), &src[..]);
//!
//! // The original array is not modified
//! let array = array.evaluated().unwrap();
//! let data: &[f32] = array.as_slice();
//! assert_eq!(data, &[1.0, 1.0, 1.0, 1.0]);
//! ```
//!
//! ## `N` dimensions
//!
//! ```rust
//! # let stream = safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
//! use safemlx::{Dtype, Array, Stream, complex64, fft::*};
//!
//! let mut array = Array::ones::<f32>(&[2, 2, 2], &stream).unwrap();
//! let mut fftn_result = fftn(&array, None, None, &stream).unwrap();
//! assert_eq!(fftn_result.dtype(), Dtype::Complex64);
//!
//! let mut expected = [complex64::new(0.0, 0.0); 8];
//! expected[0] = complex64::new(8.0, 0.0);
//! assert_eq!(fftn_result.evaluated().unwrap().as_slice::<complex64>(), &expected[..]);
//!
//! let mut ifftn_result = ifftn(&fftn_result, None, None, &stream).unwrap();
//! assert_eq!(ifftn_result.dtype(), Dtype::Complex64);
//!
//! let expected = [complex64::new(1.0, 0.0); 8];
//! assert_eq!(ifftn_result.evaluated().unwrap().as_slice::<complex64>(), &expected[..]);
//!
//! let mut rfftn_result = rfftn(&array, None, None, &stream).unwrap();
//! assert_eq!(rfftn_result.dtype(), Dtype::Complex64);
//!
//! let mut expected = [complex64::new(0.0, 0.0); 8];
//! expected[0] = complex64::new(8.0, 0.0);
//! assert_eq!(rfftn_result.evaluated().unwrap().as_slice::<complex64>(), &expected[..]);
//!
//! let mut irfftn_result = irfftn(&rfftn_result, None, None, &stream).unwrap();
//! assert_eq!(irfftn_result.dtype(), Dtype::Float32);
//!
//! let expected = [1.0; 8];
//! assert_eq!(irfftn_result.evaluated().unwrap().as_slice::<f32>(), &expected[..]);
//!
//! // The original array is not modified
//! let array = array.evaluated().unwrap();
//! let data: &[f32] = array.as_slice();
//! assert_eq!(data, &[1.0; 8]);
//! ```

mod fftn;
mod rfftn;
mod shift;
mod utils;

pub use self::{fftn::*, rfftn::*, shift::*};

/* -------------------------------------------------------------------------- */
/*                              Helper functions                              */
/* -------------------------------------------------------------------------- */
