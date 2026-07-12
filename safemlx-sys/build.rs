extern crate cmake;

use cmake::Config;
use std::{env, path::PathBuf, process::Command};

fn is_docs_rs() -> bool {
    env::var_os("DOCS_RS").is_some()
}

fn should_generate_bindings() -> bool {
    env::var_os("SAFEMLX_SYS_GENERATE_BINDINGS").is_some()
}

fn copy_pregenerated_bindings(out_path: PathBuf) {
    println!("cargo:rerun-if-changed=src/bindings.rs");
    std::fs::copy("src/bindings.rs", out_path.join("bindings.rs"))
        .expect("Couldn't copy pregenerated bindings!");
}

/// Find the clang runtime library path dynamically using xcrun
fn find_clang_rt_path() -> Option<String> {
    // Use xcrun to find the active toolchain path
    let output = Command::new("xcrun")
        .args(["--show-sdk-platform-path"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    // Get the developer directory which contains the toolchain
    let output = Command::new("xcode-select")
        .args(["--print-path"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let developer_dir = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let toolchain_base = format!(
        "{}/Toolchains/XcodeDefault.xctoolchain/usr/lib/clang",
        developer_dir
    );

    // Find the clang version directory (it varies by Xcode version)
    let clang_dir = std::fs::read_dir(&toolchain_base).ok()?;
    for entry in clang_dir.flatten() {
        let darwin_path = entry.path().join("lib/darwin");
        let clang_rt_lib = darwin_path.join("libclang_rt.osx.a");
        if clang_rt_lib.exists() {
            return Some(darwin_path.to_string_lossy().to_string());
        }
    }

    None
}

fn build_and_link_mlx_c() {
    let mut config = Config::new("src/mlx-c");
    config.very_verbose(true);
    config.define("CMAKE_INSTALL_PREFIX", ".");

    // Use Xcode's clang to ensure compatibility with the macOS SDK
    config.define("CMAKE_C_COMPILER", "/usr/bin/cc");
    config.define("CMAKE_CXX_COMPILER", "/usr/bin/c++");

    #[cfg(debug_assertions)]
    {
        config.define("CMAKE_BUILD_TYPE", "Debug");
    }

    #[cfg(not(debug_assertions))]
    {
        config.define("CMAKE_BUILD_TYPE", "Release");
    }

    config.define("MLX_BUILD_METAL", "OFF");
    config.define("MLX_BUILD_ACCELERATE", "OFF");

    #[cfg(feature = "metal")]
    {
        config.define("MLX_BUILD_METAL", "ON");
    }

    #[cfg(feature = "accelerate")]
    {
        config.define("MLX_BUILD_ACCELERATE", "ON");
    }

    // build the mlx-c project
    let dst = config.build();

    println!("cargo:rustc-link-search=native={}/build/lib", dst.display());
    println!("cargo:rustc-link-lib=static=mlx");
    println!("cargo:rustc-link-lib=static=mlxc");

    println!("cargo:rustc-link-lib=c++");
    println!("cargo:rustc-link-lib=dylib=objc");
    println!("cargo:rustc-link-lib=framework=Foundation");

    #[cfg(feature = "metal")]
    {
        println!("cargo:rustc-link-lib=framework=Metal");
    }

    #[cfg(feature = "accelerate")]
    {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }

    // Link against Xcode's clang runtime for ___isPlatformVersionAtLeast symbol
    // This is needed on macOS 26+ where the bundled LLVM runtime may be outdated
    // See: https://github.com/conda-forge/llvmdev-feedstock/issues/244
    if let Some(clang_rt_path) = find_clang_rt_path() {
        println!("cargo:rustc-link-search={}", clang_rt_path);
        println!("cargo:rustc-link-lib=static=clang_rt.osx");
    }
}

#[cfg(feature = "generate-bindings")]
fn generate_bindings(out_path: PathBuf) {
    let bindings = bindgen::Builder::default()
        .rust_target("1.73.0".parse().expect("rust-version"))
        .header("src/mlx-c/mlx/c/mlx.h")
        .header("src/mlx-c/mlx/c/fast.h")
        .header("src/mlx-c/mlx/c/linalg.h")
        .header("src/mlx-c/mlx/c/error.h")
        .header("src/mlx-c/mlx/c/transforms_impl.h")
        .clang_arg("-Isrc/mlx-c")
        .allowlist_function("^mlx_.*")
        .blocklist_function("^mlx_export_to_dot$")
        .blocklist_function("^mlx_print_graph$")
        .allowlist_type("^mlx_.*")
        .allowlist_type("^float16_t$")
        .allowlist_type("^bfloat16_t$")
        .blocklist_type("^FILE$")
        .blocklist_type("^__int64_t$")
        .blocklist_type("^__sbuf$")
        .blocklist_type("^__sFILE.*")
        .blocklist_type("^fpos_t$")
        .blocklist_type("^__darwin_.*")
        .allowlist_var("^mlx_.*")
        .allowlist_var("^MLX_.*")
        .layout_tests(false)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}

fn main() {
    println!("cargo:rerun-if-env-changed=DOCS_RS");
    println!("cargo:rerun-if-env-changed=SAFEMLX_SYS_GENERATE_BINDINGS");
    println!("cargo:rerun-if-changed=src/mlx-c");
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());

    if is_docs_rs() {
        println!("cargo:warning=Using pregenerated bindings on docs.rs");
        copy_pregenerated_bindings(out_path);
        return;
    }

    build_and_link_mlx_c();

    if should_generate_bindings() {
        #[cfg(feature = "generate-bindings")]
        generate_bindings(out_path);

        #[cfg(not(feature = "generate-bindings"))]
        panic!("enable the safemlx-sys `generate-bindings` feature to regenerate bindings");
    } else {
        copy_pregenerated_bindings(out_path);
    }
}
