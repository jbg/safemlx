# safemlx-sys

Rust bindings to the mlx-c API. Generated using bindgen.

## Linux and CUDA

CPU-only Linux builds require Git, a C++20 compiler, CMake 3.25 or newer, and
the BLAS, LAPACK, and LAPACKE development packages. The first build fetches the
pinned MLX source. On Ubuntu these dependencies can be installed with:

```sh
sudo apt-get install git cmake build-essential libblas-dev liblapack-dev liblapacke-dev
```

The optional `cuda` feature additionally requires a CUDA 12 or supported CUDA
13 toolkit and cuDNN 9 development files. The NVIDIA GPU must have compute
capability 7.5 or newer. Follow the pinned MLX version's
[source-build requirements](https://ml-explore.github.io/mlx/build/html/install.html#cuda),
then build with:

```sh
cargo build --release -p safemlx --features cuda
```

MLX detects the current GPU architecture during configuration. For builds on a
machine without an accessible GPU, or to produce a binary for a different GPU,
set a semicolon-separated CMake architecture list explicitly:

```sh
SAFEMLX_CUDA_ARCHITECTURES=80 cargo build --release -p safemlx --features cuda
```

The build also honors `CMAKE_CUDA_COMPILER`, `CUDAToolkit_ROOT`,
`CUDNN_INCLUDE_PATH`, and `CUDNN_LIBRARY_PATH`. CUDA and cuDNN shared libraries
must remain discoverable by the system dynamic loader when the Rust executable
runs.

NCCL is disabled by default so installing it cannot silently change the native
link requirements. Enable the `nccl` feature and, if necessary, set
`NCCL_ROOT_DIR`, `NCCL_INCLUDE_DIR`, and `NCCL_LIB_DIR`.

## Apple platform targets

The crate builds MLX with Accelerate and Metal for these Rust targets on a
macOS host with Xcode installed:

| Platform | Device target | Apple Silicon simulator target | Minimum OS |
| --- | --- | --- | --- |
| iOS / iPadOS | `aarch64-apple-ios` | `aarch64-apple-ios-sim` | 17.0 |
| tvOS | `aarch64-apple-tvos` | `aarch64-apple-tvos-sim` | 17.0 |
| visionOS | `aarch64-apple-visionos` | `aarch64-apple-visionos-sim` | 1.0 |

Install a target and build in the usual way:

```sh
rustup target add aarch64-apple-ios
cargo build -p safemlx --release --target aarch64-apple-ios
```

On Xcode versions which ship Metal as a separately downloadable component,
install it once with:

```sh
xcodebuild -downloadComponent MetalToolchain
```

The build exports `mlx.metallib` to
`target/<rust-target>/<profile>/safemlx-resources/mlx.metallib`. Add that file
to the Xcode target's **Copy Bundle Resources** phase, preserving the name
`mlx.metallib`. MLX automatically searches the application bundle for it.

An Xcode Run Script phase can instead make Cargo stage the file directly in
the product's resource directory:

```sh
export SAFEMLX_METALLIB_OUTPUT_DIR="$TARGET_BUILD_DIR/$UNLOCALIZED_RESOURCES_FOLDER_PATH"
cargo build --manifest-path "$SRCROOT/path/to/Cargo.toml" \
  --release --target "$SAFEMLX_RUST_TARGET"
```

Set `SAFEMLX_RUST_TARGET` in the Xcode configuration to the appropriate device
or simulator triple. The standard `IPHONEOS_DEPLOYMENT_TARGET`,
`TVOS_DEPLOYMENT_TARGET`, and `XROS_DEPLOYMENT_TARGET` settings are honored;
the minimum versions in the table are used when they are absent.

Mac Catalyst and watchOS are not currently supported.
