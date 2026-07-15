//! Metal backend configuration.

use crate::error::{self, Exception, Result};
use std::{ffi::CString, path::Path};

/// Path to the `mlx.metallib` exported by `safemlx-sys` for this build.
///
/// This is a build-machine path. Add the file to the Xcode application's Copy
/// Bundle Resources phase under the name `mlx.metallib`; it is not a path that
/// can be used directly on an iOS, tvOS, or visionOS device.
pub const BUILT_METALLIB_PATH: Option<&str> = option_env!("SAFEMLX_METALLIB_PATH");

fn check_status(status: i32) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(error::get_and_clear_last_mlx_error()
            .expect("MLX Metal operation failed but no error was set")
            .into())
    }
}

/// Returns whether the MLX Metal backend is available on the current device.
pub fn is_available() -> Result<bool> {
    error::ensure_mlx_error_handler();
    let mut available = false;
    check_status(unsafe { safemlx_sys::mlx_metal_is_available(&mut available) })?;
    Ok(available)
}

/// Overrides the path from which MLX loads its default Metal library.
///
/// Call this before creating arrays or performing any other MLX operation. An
/// application that copies `mlx.metallib` into the root of its bundle normally
/// does not need this override because MLX discovers that location itself.
pub fn set_metallib_path(path: impl AsRef<Path>) -> Result<()> {
    error::ensure_mlx_error_handler();
    let path = path
        .as_ref()
        .to_str()
        .ok_or_else(|| Exception::custom("metallib path is not valid UTF-8"))?;
    let path = CString::new(path).map_err(|error| Exception::custom(error.to_string()))?;
    check_status(unsafe { safemlx_sys::mlx_metal_set_metallib_path(path.as_ptr()) })
}
