# safemlx

Rust bindings for Apple's MLX machine learning framework.

`safemlx` provides a safe, idiomatic Rust interface over the low-level
`safemlx-sys` bindings. It includes array operations, neural-network building
blocks, transforms, optimizers, quantization helpers, and optional
SafeTensors support.

This crate targets Apple platforms supported by MLX. The default feature set
enables both Accelerate and Metal support.

## Features

- `accelerate`: enables Accelerate-backed MLX operations.
- `metal`: enables Metal-backed MLX operations.
- `safetensors`: enables conversion between `Array` and
  `safetensors::TensorView`.

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
safemlx = "0.1"
```

## Important Notes on Automatic Differentiation

When using automatic differentiation in `safemlx`, there is an important
difference in how closures work compared to Python's MLX. In Python, variables
are implicitly captured and properly traced in the compute graph. In Rust, pass
all arrays that should be traced as explicit inputs.

```rust
// Don't do this
let x = random::normal::<f32>(&[num_examples, num_features], None, None, None)?;
let y = x.matmul(&w_star)? + eps;

let loss_fn = |w: &Array| -> Result<Array, Exception> {
    let y_pred = x.matmul(w)?;  // x and y are captured from outer scope
    let loss = Array::from_f32(0.5) * ops::mean(&ops::square(&(y_pred - &y))?, None, None)?;
    Ok(loss)
};

let grad_fn = transforms::grad_with_argnums(loss_fn, &[0], stream);
```

Instead, pass all required arrays as inputs:

```rust
let loss_fn = |inputs: &[Array]| -> Result<Array, Exception> {
    let w = &inputs[0];
    let x = &inputs[1];
    let y = &inputs[2];

    let y_pred = x.matmul(w)?;
    let loss = Array::from_f32(0.5) * ops::mean(&ops::square(y_pred - y)?, None, None)?;
    Ok(loss)
};
let argnums = &[0];

let mut inputs = vec![w, x, y];
let grad = transforms::grad_with_argnums(loss_fn, argnums, stream)(&inputs)?;
```

When using gradients in training loops, remember to update the appropriate array in your inputs:

```rust
let mut inputs = vec![w, x, y];

for _ in 0..num_iterations {
    let grad = transforms::grad_with_argnums(loss_fn, argnums, stream)(&inputs)?;
    inputs[0] = &inputs[0] - Array::from_f32(learning_rate) * grad;
    inputs[0].eval()?;
}
```

For now, explicitly passing all required arrays as shown above is the
recommended approach.

## Versioning

The `safemlx` crates use normal Rust semantic versioning. The initial
crates.io release is `0.1.0`.

## Status

`safemlx` is in active development.

## MSRV

The minimum supported Rust version is 1.85.0.

Each published crate declares its MSRV in `Cargo.toml`.

## License

Licensed under either MIT or Apache-2.0.
