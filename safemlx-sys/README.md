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

## Native Windows x86-64 CUDA

Native Windows builds use the stable Rust MSVC target, Visual Studio 2022, MLX
v0.32.0, and cuDNN 9. The pinned MLX/CUTLASS combination is covered in CI with
CUDA 12.9 and CUDA 13.0. CUDA 12.6 is not supported on Windows by this MLX
version. Install these prerequisites:

- 64-bit Windows and the `x86_64-pc-windows-msvc` Rust toolchain.
- Visual Studio 2022 Build Tools with the Desktop development with C++ workload.
- Git, CMake 3.25 or newer, and Ninja.
- CUDA Toolkit 12.9 or 13.0 and the corresponding cuDNN 9 development archive.

From a Visual Studio x64 developer PowerShell, point CMake at the toolkit and
cuDNN installation and build a final executable:

```powershell
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
$env:CUDA_HOME = $env:CUDA_PATH
$env:CUDAToolkit_ROOT = $env:CUDA_PATH
$env:CMAKE_CUDA_COMPILER = "$env:CUDA_PATH\bin\nvcc.exe"
$env:CUDNN_INCLUDE_PATH = "C:\tools\cudnn\include"
$env:CUDNN_LIBRARY_PATH = "C:\tools\cudnn\lib\x64"
$env:SAFEMLX_CUDA_ARCHITECTURES = "75"
$env:CMAKE_GENERATOR = "Ninja"
$env:PATH = "$env:CUDA_PATH\bin;C:\tools\cudnn\bin\x64;$env:PATH"

cargo build --release -p safemlx --features cuda --example cuda_smoke --no-default-features
cargo build --release -p safemlx-lm-cli --features cuda
```

`CUDAToolkit_ROOT`, `CUDA_HOME`, and `CUDA_PATH` are searched in that order
after an explicit `CMAKE_CUDA_COMPILER`; standard versioned CUDA installations
and `nvcc.exe` on `PATH` are also detected. `MLX_CUDA_ARCHITECTURES` is honored,
and `SAFEMLX_CUDA_ARCHITECTURES` overrides it. Use a semicolon-separated list
for a multi-architecture binary.

On Windows, safemlx builds `mlx` and `mlxc` as DLLs. This keeps CMake in charge
of the CUDA, cuDNN, OpenBLAS, dlfcn-win32, MSVC runtime, `delayimp`, and
`/DELAYLOAD` dependency graph. The build stages `mlx.dll`, `mlxc.dll`, and the
fetched OpenBLAS DLL next to Cargo binaries, examples, and test executables.
Upstream MLX's delay-load hook records the configured CUDA and cuDNN binary
directories; keep those installations at the same paths. When relocating a
build, add the CUDA and cuDNN `bin` directories to `PATH` before starting it.

Common configuration failures are usually resolved as follows:

- Missing `nvcc.exe`: set `CMAKE_CUDA_COMPILER` and `CUDAToolkit_ROOT` to the
  same toolkit installation.
- Missing CUDA import libraries: install the cuBLAS, cuFFT, NVRTC, and runtime
  development components, and verify `CUDA_PATH\lib\x64` exists.
- Missing cuDNN headers or `.lib` files: set `CUDNN_INCLUDE_PATH` to the
  directory containing `cudnn.h` and `CUDNN_LIBRARY_PATH` to `lib\x64`.
- Startup DLL errors: restore the build-time CUDA/cuDNN locations or add their
  `bin` directories to `PATH`; do not copy only the import libraries.
- Unsupported compute capability: set `SAFEMLX_CUDA_ARCHITECTURES` to the GPU's
  architecture. MLX v0.32.0 requires compute capability 7.5 or newer.

Windows ARM CUDA and Windows NCCL are intentionally unsupported. Metal and
Accelerate remain Apple-only. Hosted CI proves MSVC compilation and final
linkage without a GPU; Windows CUDA runtime behavior remains experimental until
the optional self-hosted Windows NVIDIA workflow has passed on real hardware.

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
