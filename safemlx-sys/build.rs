extern crate cmake;

mod build_support;

use build_support::apple_mobile_target;
use cmake::Config;
use std::{env, path::Path, path::PathBuf};

#[cfg(feature = "metal")]
fn find_metal_compiler() -> PathBuf {
    for arguments in [
        &["--toolchain", "Metal", "-find", "metal"][..],
        &["-find", "metal"][..],
    ] {
        let output = std::process::Command::new("xcrun")
            .args(arguments)
            // An SDKROOT for a mobile SDK can make xcrun select the placeholder
            // compiler in XcodeDefault.xctoolchain instead of the separately
            // installed Metal toolchain.
            .env_remove("SDKROOT")
            .output()
            .expect("Couldn't run xcrun to locate the Metal compiler");
        if output.status.success() {
            let path = String::from_utf8(output.stdout)
                .expect("xcrun returned a non-UTF-8 Metal compiler path");
            let path = PathBuf::from(path.trim());
            if path.is_file() {
                return path;
            }
        }
    }

    panic!(
        "Couldn't locate the Metal compiler. Install it with `xcodebuild -downloadComponent MetalToolchain`."
    );
}

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

#[cfg(feature = "metal")]
fn profile_output_dir(out_path: &Path) -> PathBuf {
    out_path
        .ancestors()
        .nth(3)
        .expect("Cargo OUT_DIR did not have the expected layout")
        .join("safemlx-resources")
}

#[cfg(feature = "metal")]
fn export_metallib(dst: &Path, out_path: &Path) -> PathBuf {
    let source_candidates = [
        dst.join("lib/mlx.metallib"),
        dst.join("build/lib/mlx.metallib"),
        dst.join("build/_deps/mlx-build/mlx/backend/metal/kernels/mlx.metallib"),
    ];
    let source = source_candidates
        .iter()
        .find(|path| path.is_file())
        .unwrap_or_else(|| {
            panic!(
                "MLX built with Metal but did not produce mlx.metallib; checked: {}",
                source_candidates
                    .iter()
                    .map(|path| path.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        });
    let output_dir = env::var_os("SAFEMLX_METALLIB_OUTPUT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| profile_output_dir(out_path));
    std::fs::create_dir_all(&output_dir).expect("Couldn't create Metal resource output dir");
    let output = output_dir.join("mlx.metallib");
    std::fs::copy(source, &output).expect("Couldn't export mlx.metallib");
    output
}

fn build_and_link_mlx_c(out_path: &Path) {
    #[cfg(not(feature = "metal"))]
    let _ = out_path;
    let target = env::var("TARGET").expect("TARGET was not set by Cargo");
    let target_os = env::var("CARGO_CFG_TARGET_OS").expect("target OS was not set by Cargo");
    let target_arch =
        env::var("CARGO_CFG_TARGET_ARCH").expect("target architecture was not set by Cargo");
    let mobile_target = apple_mobile_target(&target, &target_os, &target_arch)
        .unwrap_or_else(|error| panic!("{error}"));

    let mut config = Config::new("src/mlx-c");
    config.very_verbose(true);
    config.define("CMAKE_INSTALL_PREFIX", ".");
    config.define(
        "CMAKE_BUILD_TYPE",
        if env::var("PROFILE").as_deref() == Ok("release") {
            "Release"
        } else {
            "Debug"
        },
    );
    config.define("CMAKE_TRY_COMPILE_TARGET_TYPE", "STATIC_LIBRARY");
    config.define("BUILD_SHARED_LIBS", "OFF");
    config.define("MLX_C_BUILD_EXAMPLES", "OFF");

    if let Some(platform) = mobile_target {
        println!(
            "cargo:rerun-if-env-changed={}",
            platform.deployment_target_env
        );
        let deployment_target = env::var(platform.deployment_target_env)
            .unwrap_or_else(|_| platform.default_deployment_target.into());
        if env::var_os(platform.deployment_target_env).is_none() {
            // The cmake crate asks cc-rs for the target compiler and flags.
            // Make its deployment target agree with the CMake and Metal builds.
            env::set_var(platform.deployment_target_env, &deployment_target);
        }
        config.define("CMAKE_OSX_SYSROOT", platform.sdk);
        config.define("CMAKE_OSX_ARCHITECTURES", platform.cmake_architecture);
        config.define("CMAKE_OSX_DEPLOYMENT_TARGET", &deployment_target);
        config.define("MLX_METAL_SDK", platform.sdk);
        config.define(
            "MLX_METAL_MIN_VERSION_FLAG",
            platform.metal_minimum_version_arg(&deployment_target),
        );
    }

    config.define("MLX_BUILD_METAL", "OFF");
    config.define("MLX_BUILD_ACCELERATE", "OFF");

    #[cfg(feature = "metal")]
    {
        config.define("MLX_METAL_COMPILER", find_metal_compiler());
        config.define("MLX_BUILD_METAL", "ON");
    }

    #[cfg(feature = "accelerate")]
    {
        config.define("MLX_BUILD_ACCELERATE", "ON");
    }

    // build the mlx-c project
    let dst = config.build();

    println!("cargo:rustc-link-search=native={}/build/lib", dst.display());
    println!(
        "cargo:rustc-link-search=native={}/build/_deps/mlx-build/mlx/io",
        dst.display()
    );
    println!("cargo:rustc-link-lib=static=mlx");
    println!("cargo:rustc-link-lib=static=mlxc");
    // MLX links its GGUF parser privately. Static Rust consumers must name the
    // parser archive as well because CMake's transitive link interface is not
    // preserved when Cargo links the installed archives directly.
    println!("cargo:rustc-link-lib=static=gguflib");

    println!("cargo:rustc-link-lib=c++");
    println!("cargo:rustc-link-lib=dylib=objc");
    println!("cargo:rustc-link-lib=framework=Foundation");

    #[cfg(feature = "metal")]
    {
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=QuartzCore");
    }

    #[cfg(feature = "accelerate")]
    {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }

    #[cfg(feature = "metal")]
    {
        let metallib = export_metallib(&dst, out_path);
        println!("cargo:metadata=metallib_path={}", metallib.display());
        println!(
            "cargo:rustc-env=SAFEMLX_METALLIB_PATH={}",
            metallib.display()
        );
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
    println!("cargo:rerun-if-env-changed=SAFEMLX_METALLIB_OUTPUT_DIR");
    println!("cargo:rerun-if-env-changed=DEVELOPER_DIR");
    println!("cargo:rerun-if-changed=src/mlx-c");
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());

    if is_docs_rs() {
        println!("cargo:warning=Using pregenerated bindings on docs.rs");
        copy_pregenerated_bindings(out_path);
        return;
    }

    build_and_link_mlx_c(&out_path);

    if should_generate_bindings() {
        #[cfg(feature = "generate-bindings")]
        generate_bindings(out_path);

        #[cfg(not(feature = "generate-bindings"))]
        panic!("enable the safemlx-sys `generate-bindings` feature to regenerate bindings");
    } else {
        copy_pregenerated_bindings(out_path);
    }
}
