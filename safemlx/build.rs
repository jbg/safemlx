use std::env;

fn main() {
    println!("cargo:rerun-if-env-changed=DEP_MLX_METALLIB_PATH");
    if let Some(path) = env::var_os("DEP_MLX_METALLIB_PATH") {
        println!(
            "cargo:rustc-env=SAFEMLX_METALLIB_PATH={}",
            path.to_string_lossy()
        );
    }
}
