//! CUDA backend configuration.

use crate::error::{self, Result};

fn check_status(status: i32) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(error::get_and_clear_last_mlx_error()
            .expect("MLX CUDA operation failed but no error was set")
            .into())
    }
}

/// Returns whether the MLX CUDA backend is available on the current device.
pub fn is_available() -> Result<bool> {
    error::ensure_mlx_error_handler();
    let mut available = false;
    check_status(unsafe { safemlx_sys::mlx_cuda_is_available(&mut available) })?;
    Ok(available)
}
