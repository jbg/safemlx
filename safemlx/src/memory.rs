//! MLX allocator memory statistics.
//!
//! These counters cover memory managed by MLX. They do not include all process
//! resident memory, memory-mapped checkpoint pages, or allocations owned by
//! unrelated libraries.

use crate::error::{self, Result};

fn check_status(status: i32) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(error::get_and_clear_last_mlx_error()
            .expect("MLX memory operation failed but no error was set")
            .into())
    }
}

/// Returns bytes currently held by active MLX allocations.
pub fn active_memory() -> Result<usize> {
    error::ensure_mlx_error_handler();
    let mut bytes = 0;
    check_status(unsafe { safemlx_sys::mlx_get_active_memory(&mut bytes) })?;
    Ok(bytes)
}

/// Returns bytes currently retained by the MLX allocation cache.
pub fn cache_memory() -> Result<usize> {
    error::ensure_mlx_error_handler();
    let mut bytes = 0;
    check_status(unsafe { safemlx_sys::mlx_get_cache_memory(&mut bytes) })?;
    Ok(bytes)
}

/// Returns the peak number of active MLX allocation bytes since the last reset.
pub fn peak_memory() -> Result<usize> {
    error::ensure_mlx_error_handler();
    let mut bytes = 0;
    check_status(unsafe { safemlx_sys::mlx_get_peak_memory(&mut bytes) })?;
    Ok(bytes)
}

/// Resets the MLX peak-active-memory counter to the current active allocation.
pub fn reset_peak_memory() -> Result<()> {
    error::ensure_mlx_error_handler();
    check_status(unsafe { safemlx_sys::mlx_reset_peak_memory() })
}
