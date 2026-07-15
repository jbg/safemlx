#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(clippy::all)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

/// Path to the `mlx.metallib` produced for this Cargo target and profile.
///
/// Add this file to the application's Copy Bundle Resources phase. Set
/// `SAFEMLX_METALLIB_OUTPUT_DIR` while building to export it directly to a
/// different resource-staging directory.
#[cfg(all(feature = "metal", target_vendor = "apple"))]
pub const MLX_METALLIB_PATH: &str = match option_env!("SAFEMLX_METALLIB_PATH") {
    Some(path) => path,
    None => "",
};
