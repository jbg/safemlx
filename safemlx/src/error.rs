//! Custom error types and handler for the c ffi

use crate::Dtype;
use libc::strdup;
use std::convert::Infallible;
use std::ffi::NulError;
use std::panic::Location;
use std::sync::Once;
use std::{cell::Cell, ffi::c_char};
use thiserror::Error;

/// Type alias for a `Result` with an `Exception` error type.
pub type Result<T> = std::result::Result<T, Exception>;

/// Error with io operations
#[derive(Error, PartialEq, Debug)]
pub enum IoError {
    /// Path must point to a local file
    #[error("Path must point to a local file")]
    NotFile,

    /// Path contains invalid UTF-8
    #[error("Path contains invalid UTF-8")]
    InvalidUtf8,

    /// Path contains null bytes
    #[error("Path contains null bytes")]
    NullBytes,

    /// No file extension found
    #[error("No file extension found")]
    NoExtension,

    /// Unsupported file format
    #[error("Unsupported file format")]
    UnsupportedFormat,

    /// Invalid or unsupported GGUF data
    #[error("invalid GGUF file: {0}")]
    InvalidGguf(String),

    /// Unable to open file
    #[error("Unable to open file")]
    UnableToOpenFile,

    /// Unable to allocate memory
    #[error("Unable to allocate memory")]
    AllocationError,

    /// Null error
    #[error(transparent)]
    NulError(#[from] NulError),

    /// Error with unfalttening the loaded optimizer state
    #[error(transparent)]
    Unflatten(#[from] UnflattenError),

    /// Exception
    #[error(transparent)]
    Exception(#[from] Exception),
}

impl From<Infallible> for IoError {
    fn from(_: Infallible) -> Self {
        unreachable!()
    }
}

impl From<RawException> for IoError {
    #[track_caller]
    fn from(e: RawException) -> Self {
        let exception = Exception {
            what: e.what,
            location: Location::caller(),
        };
        Self::Exception(exception)
    }
}

/// Error associated with `Array::try_as_slice()`
#[derive(Debug, PartialEq, Error)]
pub enum AsSliceError {
    /// The underlying data pointer is null.
    ///
    /// This is likely because the array has not been evaluated yet.
    #[error("The data pointer is null.")]
    Null,

    /// The output dtype does not match the data type of the array.
    #[error("dtype mismatch: expected {expecting:?}, found {found:?}")]
    DtypeMismatch {
        /// The expected data type.
        expecting: Dtype,

        /// The actual data type
        found: Dtype,
    },

    /// Exception
    #[error(transparent)]
    Exception(#[from] Exception),
}

/// Error with unflattening a loaded optimizer state
#[derive(Debug, PartialEq, Error)]
pub enum UnflattenError {
    /// Expecting next (key, value) pair, found none
    #[error("Expecting next (key, value) pair, found none")]
    ExpectingNextPair,

    /// The key is not in a valid format
    #[error("Invalid key")]
    InvalidKey,
}

/// Error with loading an optimizer state
#[derive(Debug, PartialEq, Error)]
pub enum OptimizerStateLoadError {
    /// Error with io operations
    #[error(transparent)]
    Io(#[from] IoError),

    /// Error with unflattening the optimizer state
    #[error(transparent)]
    Unflatten(#[from] UnflattenError),
}

impl From<Infallible> for OptimizerStateLoadError {
    fn from(_: Infallible) -> Self {
        unreachable!()
    }
}

cfg_safetensors! {
    /// Error associated with conversion between `safetensors::tensor::TensorView` and `Array`
    /// when the data type is not supported.
    #[derive(Debug, Error)]
    pub enum ConversionError {
        /// The safetensors data type that is not supported.
        ///
        /// This is the error type for conversions from `safetensors::tensor::TensorView` to `Array`.
        #[error("The safetensors data type {0:?} is not supported.")]
        SafeTensorDtype(safetensors::tensor::Dtype),

        /// The mlx data type that is not supported.
        ///
        /// This is the error type for conversions from `Array` to `safetensors::tensor::TensorView`.
        #[error("The mlx data type {0:?} is not supported.")]
        MlxDtype(crate::Dtype),

        /// Error casting the data buffer to `&[u8]`.
        #[error(transparent)]
        PodCastError(#[from] bytemuck::PodCastError),

        /// Error with creating a `safetensors::tensor::TensorView`.
        #[error(transparent)]
        SafeTensorError(#[from] safetensors::tensor::SafeTensorError),
    }
}

pub(crate) struct RawException {
    pub(crate) what: String,
}

/// Exception. Most will come from the C API.
#[derive(Debug, PartialEq, Error)]
#[error("{what:?} at {location}")]
pub struct Exception {
    pub(crate) what: String,
    pub(crate) location: &'static Location<'static>,
}

impl Exception {
    /// The error message.
    pub fn what(&self) -> &str {
        &self.what
    }

    /// The location of the error.
    ///
    /// The location is obtained from `std::panic::Location::caller()` and points
    /// to the location in the code where the error was created and not where it was
    /// propagated.
    pub fn location(&self) -> &'static Location<'static> {
        self.location
    }

    /// Creates a new exception with the given message.
    #[track_caller]
    pub fn custom(what: impl Into<String>) -> Self {
        Self {
            what: what.into(),
            location: Location::caller(),
        }
    }
}

impl From<RawException> for Exception {
    #[track_caller]
    fn from(e: RawException) -> Self {
        Self {
            what: e.what,
            location: Location::caller(),
        }
    }
}

impl From<&str> for Exception {
    #[track_caller]
    fn from(what: &str) -> Self {
        Self {
            what: what.to_string(),
            location: Location::caller(),
        }
    }
}

impl From<Infallible> for Exception {
    fn from(_: Infallible) -> Self {
        unreachable!()
    }
}

impl From<Exception> for String {
    fn from(e: Exception) -> Self {
        e.what
    }
}

thread_local! {
    static CLOSURE_ERROR: Cell<Option<Exception>> = const { Cell::new(None) };
    static LAST_MLX_ERROR: Cell<*mut c_char> = const { Cell::new(std::ptr::null_mut()) };
}

static INIT_ERR_HANDLER: Once = Once::new();

#[no_mangle]
extern "C" fn default_mlx_error_handler(msg: *const c_char, _data: *mut std::ffi::c_void) {
    unsafe {
        LAST_MLX_ERROR.with(|last_error| {
            let previous = last_error.replace(strdup(msg));
            if !previous.is_null() {
                libc::free(previous as *mut libc::c_void);
            }
        });
    }
}

fn setup_mlx_error_handler() {
    let handler = default_mlx_error_handler;
    unsafe {
        safemlx_sys::mlx_set_error_handler(Some(handler), std::ptr::null_mut(), None);
    }
}

pub(crate) fn ensure_mlx_error_handler() {
    INIT_ERR_HANDLER.call_once(setup_mlx_error_handler);
}

pub(crate) fn set_closure_error(err: Exception) {
    CLOSURE_ERROR.with(|closure_error| closure_error.set(Some(err)));
}

pub(crate) fn get_and_clear_closure_error() -> Option<Exception> {
    CLOSURE_ERROR.with(|closure_error| closure_error.replace(None))
}

#[track_caller]
pub(crate) fn get_and_clear_last_mlx_error() -> Option<RawException> {
    LAST_MLX_ERROR.with(|last_error| {
        let last_err_ptr = last_error.replace(std::ptr::null_mut());
        if last_err_ptr.is_null() {
            return None;
        }

        let last_err = unsafe {
            std::ffi::CStr::from_ptr(last_err_ptr)
                .to_string_lossy()
                .into_owned()
        };
        unsafe {
            libc::free(last_err_ptr as *mut libc::c_void);
        }

        Some(RawException { what: last_err })
    })
}

/// Error with building a cross-entropy loss function
#[derive(Debug, Clone, PartialEq, Error)]
pub enum CrossEntropyBuildError {
    /// Label smoothing factor must be in the range [0, 1)
    #[error("Label smoothing factor must be in the range [0, 1)")]
    InvalidLabelSmoothingFactor,
}

impl From<CrossEntropyBuildError> for Exception {
    fn from(value: CrossEntropyBuildError) -> Self {
        Exception::custom(format!("{value}"))
    }
}

/// Error with building a RmsProp optimizer
#[derive(Debug, Clone, PartialEq, Error)]
pub enum RmsPropBuildError {
    /// Alpha must be non-negative
    #[error("alpha must be non-negative")]
    NegativeAlpha,

    /// Epsilon must be non-negative
    #[error("epsilon must be non-negative")]
    NegativeEpsilon,
}

/// Error with building an AdaDelta optimizer
#[derive(Debug, Clone, PartialEq, Error)]
pub enum AdaDeltaBuildError {
    /// Rho must be non-negative
    #[error("rho must be non-negative")]
    NegativeRho,

    /// Epsilon must be non-negative
    #[error("epsilon must be non-negative")]
    NegativeEps,
}

/// Error with building an Adafactor optimizer.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum AdafactorBuildError {
    /// Either learning rate is provided or relative step is set to true.
    #[error("Either learning rate is provided or relative step is set to true")]
    LrIsNoneAndRelativeStepIsFalse,
}

/// Error with building a dropout layer
#[derive(Debug, Clone, PartialEq, Error)]
pub enum DropoutBuildError {
    /// Dropout probability must be in the range [0, 1)
    #[error("Dropout probability must be in the range [0, 1)")]
    InvalidProbability,
}

/// Error with building a MultiHeadAttention module
#[derive(Debug, PartialEq, Error)]
pub enum MultiHeadAttentionBuildError {
    /// Invalid number of heads
    #[error("Invalid number of heads: {0}")]
    InvalidNumHeads(i32),

    /// Exceptions
    #[error(transparent)]
    Exception(#[from] Exception),
}

/// Error with building a transformer
#[derive(Debug, PartialEq, Error)]
pub enum TransformerBulidError {
    /// Dropout probability must be in the range [0, 1)
    #[error("Dropout probability must be in the range [0, 1)")]
    InvalidProbability,

    /// Invalid number of heads
    #[error("Invalid number of heads: {0}")]
    InvalidNumHeads(i32),

    /// Exceptions
    #[error(transparent)]
    Exception(#[from] Exception),
}

impl From<DropoutBuildError> for TransformerBulidError {
    fn from(e: DropoutBuildError) -> Self {
        match e {
            DropoutBuildError::InvalidProbability => Self::InvalidProbability,
        }
    }
}

impl From<MultiHeadAttentionBuildError> for TransformerBulidError {
    fn from(e: MultiHeadAttentionBuildError) -> Self {
        match e {
            MultiHeadAttentionBuildError::InvalidNumHeads(n) => Self::InvalidNumHeads(n),
            MultiHeadAttentionBuildError::Exception(e) => Self::Exception(e),
        }
    }
}

/// The dtype is not a float-point type
#[derive(Debug, Error)]
#[error("[finfo] dtype {:?} is not inexact", .0)]
pub struct InexactDtypeError(pub Dtype);

impl From<InexactDtypeError> for Exception {
    #[track_caller]
    fn from(value: InexactDtypeError) -> Self {
        Exception::custom(value.to_string())
    }
}

#[cfg(test)]
mod tests {
    use crate::{array, Array};

    #[test]
    fn test_exception() {
        let stream = crate::test_stream();
        let a = array!([1.0, 2.0, 3.0]);
        let b = array!([4.0, 5.0]);

        let result = a.add(&b, stream);
        let error = result.expect_err("Expected error");

        // The full error message would also contain the full path to the original c++ file,
        // so we just check for a substring
        assert!(error
            .what()
            .contains("Shapes (3) and (2) cannot be broadcast."))
    }

    #[test]
    fn mlx_errors_are_thread_local() {
        let threads = (0..8)
            .map(|thread_index| {
                std::thread::spawn(move || {
                    let stream = crate::Stream::new_with_device(&crate::Device::new(
                        crate::DeviceType::Gpu,
                        0,
                    ));
                    for iteration in 0..64 {
                        let lhs_len = 3 + thread_index;
                        let rhs_len = lhs_len + 1 + iteration % 3;
                        let lhs = Array::from_slice(&vec![0.0f32; lhs_len], &[lhs_len as i32]);
                        let rhs = Array::from_slice(&vec![0.0f32; rhs_len], &[rhs_len as i32]);
                        let error = lhs.add(&rhs, &stream).expect_err("add should fail");
                        let expected = format!("Shapes ({lhs_len}) and ({rhs_len})");
                        assert!(
                            error.what().contains(&expected),
                            "expected thread-local error containing {expected:?}, got {:?}",
                            error.what()
                        );
                    }
                })
            })
            .collect::<Vec<_>>();

        for thread in threads {
            thread.join().expect("worker thread panicked");
        }
    }
}
