# safemlx-sys

Rust bindings to the mlx-c API. Generated using bindgen.

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
