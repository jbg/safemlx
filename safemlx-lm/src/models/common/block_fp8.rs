//! Architecture-neutral block-scaled FP8 projections.
//!
//! Checkpoints store E4M3 bytes together with one inverse scale per 128x128
//! weight block. Conventional dense and routed-expert projections dynamically
//! quantize each 128-value activation block to E4M3 using the checkpoint's
//! declared DeepSeek/Qwen FP8 execution scheme. GPU operations consume both
//! packed representations directly, including rank-3 expert banks, without
//! expanding a complete weight bank. CPU execution uses a deliberately slow
//! dequantized reference path for correctness tests and functional fallback.

use std::cell::RefCell;

#[cfg(feature = "cuda")]
use safemlx::fast::CudaKernel;
#[cfg(not(feature = "cuda"))]
use safemlx::fast::MetalKernel;
use safemlx::fast::MetalKernelConfig;
use safemlx::{
    error::Exception,
    ops::{concatenate_axis, grouped_matmul, indexing::TryIndexOp, matmul},
    transforms::eval,
    Array, DeviceType, Dtype, Stream,
};

#[cfg(not(feature = "cuda"))]
thread_local! {
    static ACT_QUANT_KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static LINEAR_KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static LINEAR_SCALAR_KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static GROUPED_LINEAR_KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static GROUPED_LINEAR_SCALAR_KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static SEGMENTED_LINEAR_KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static SEGMENTED_TRANSPOSED_LINEAR_KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
}

#[cfg(feature = "cuda")]
thread_local! {
    static ACT_QUANT_KERNEL: RefCell<Option<CudaKernel>> = const { RefCell::new(None) };
    static LINEAR_KERNEL: RefCell<Option<CudaKernel>> = const { RefCell::new(None) };
    static GROUPED_LINEAR_KERNEL: RefCell<Option<CudaKernel>> = const { RefCell::new(None) };
    static SEGMENTED_LINEAR_KERNEL: RefCell<Option<CudaKernel>> = const { RefCell::new(None) };
    static SEGMENTED_TRANSPOSED_LINEAR_KERNEL: RefCell<Option<CudaKernel>> = const { RefCell::new(None) };
}

const OUT_TILE: i32 = 16;
const REDUCTION_TILE: i32 = 16;
const SCALE_BLOCK: i32 = 128;
#[cfg(not(feature = "cuda"))]
const TILED_ROW_THRESHOLD: i32 = 8;

fn ceil_div(lhs: i32, rhs: i32) -> i32 {
    (lhs + rhs - 1) / rhs
}

fn linear_tiled_config(rows: i32, in_dim: i32, out_dim: i32, scale_cols: i32) -> MetalKernelConfig {
    let out_grid = ceil_div(out_dim, OUT_TILE) * OUT_TILE;
    MetalKernelConfig::new()
        .with_template_arg_int("IN_DIM", in_dim)
        .with_template_arg_int("OUT_DIM", out_dim)
        .with_template_arg_int("OUT_TILE", OUT_TILE)
        .with_template_arg_int("REDUCTION_TILE", REDUCTION_TILE)
        .with_template_arg_int("SCALE_BLOCK", SCALE_BLOCK)
        .with_template_arg_int("SCALE_COLS", scale_cols)
        .with_grid([out_grid, rows * REDUCTION_TILE, 1])
        .with_thread_group([OUT_TILE, REDUCTION_TILE, 1])
        .with_output_arg([rows, out_dim], Dtype::Float32)
}

fn grouped_tiled_config(
    routes: i32,
    in_dim: i32,
    out_dim: i32,
    scale_out: i32,
    scale_cols: i32,
) -> MetalKernelConfig {
    linear_tiled_config(routes, in_dim, out_dim, scale_cols)
        .with_template_arg_int("SCALE_OUT", scale_out)
}

fn activation_dtype(input: &Array) -> Result<Dtype, Exception> {
    let dtype = input.dtype();
    if !dtype.is_float() {
        return Err(Exception::custom(format!(
            "block-FP8 activation input must be floating point, got {dtype:?}"
        )));
    }
    Ok(dtype)
}

fn restore_activation_dtype(
    output: Array,
    dtype: Dtype,
    stream: &Stream,
) -> Result<Array, Exception> {
    if dtype == Dtype::Float32 {
        Ok(output)
    } else {
        output.as_dtype(dtype, stream)
    }
}

fn is_cpu_stream(stream: &Stream) -> Result<bool, Exception> {
    Ok(stream.get_device()?.get_type()? == DeviceType::Cpu)
}

fn dequantize_grouped(weight: &Array, scale: &Array, stream: &Stream) -> Result<Array, Exception> {
    let experts = weight.dim(0);
    let out_dim = weight.dim(1);
    let in_dim = weight.dim(2);
    let scale = Array::repeat_axis::<f32>(scale.clone(), 128, 1, stream)?;
    let scale = Array::repeat_axis::<f32>(scale, 128, 2, stream)?;
    weight.from_fp8(Dtype::Float32, stream)?.multiply(
        scale.try_index_device((..experts, ..out_dim, ..in_dim), stream)?,
        stream,
    )
}

#[allow(clippy::too_many_arguments)]
fn segmented_reference(
    input: &Array,
    weight: &Array,
    scale: &Array,
    group_ids: &Array,
    group_stride: i32,
    row_offset: i32,
    output_dims: i32,
    transpose: bool,
    stream: &Stream,
) -> Result<Array, Exception> {
    let weight = dequantize(weight, scale, stream)?;
    let group_ids = group_ids.as_dtype(Dtype::Uint32, stream)?;
    eval([&group_ids])?;
    let evaluated = group_ids.evaluated()?;
    let mut outputs = Vec::with_capacity(input.dim(0) as usize);
    for (route, group) in evaluated.as_slice::<u32>().iter().copied().enumerate() {
        let start = group as i32 * group_stride + row_offset;
        let input_row = input.try_index_device(route as i32..route as i32 + 1, stream)?;
        let segment = weight.try_index_device(start..start + output_dims, stream)?;
        outputs.push(if transpose {
            matmul(&input_row, &segment, stream)?
        } else {
            matmul(&input_row, &segment.transpose(stream)?, stream)?
        });
    }
    let refs = outputs.iter().collect::<Vec<_>>();
    concatenate_axis(&refs, 0, stream)
}

struct QuantizedActivations {
    values: Array,
    scales: Array,
}

fn quantize_activations(
    input: &Array,
    rows: i32,
    in_dim: i32,
    stream: &Stream,
) -> Result<QuantizedActivations, Exception> {
    let scale_cols = ceil_div(in_dim, SCALE_BLOCK);
    let config = MetalKernelConfig::new()
        .with_template_arg_int("IN_DIM", in_dim)
        .with_template_arg_int("SCALE_COLS", scale_cols)
        .with_template_arg_int("SCALE_BLOCK", SCALE_BLOCK)
        .with_grid([rows * scale_cols * SCALE_BLOCK, 1, 1])
        .with_thread_group([SCALE_BLOCK, 1, 1])
        .with_output_arg([rows, in_dim], Dtype::Uint8)
        .with_output_arg([rows, scale_cols], Dtype::Float32);

    #[cfg(feature = "cuda")]
    let mut outputs = ACT_QUANT_KERNEL.with(|cell| -> Result<_, Exception> {
        if cell.borrow().is_none() {
            *cell.borrow_mut() = Some(activation_quantization_kernel_cuda()?);
        }
        cell.borrow()
            .as_ref()
            .expect("CUDA activation quantization kernel initialized")
            .apply_device([input], &config, stream)
    })?;

    #[cfg(not(feature = "cuda"))]
    let mut outputs = ACT_QUANT_KERNEL.with(|cell| -> Result<_, Exception> {
        if cell.borrow().is_none() {
            *cell.borrow_mut() = Some(activation_quantization_kernel_metal()?);
        }
        cell.borrow()
            .as_ref()
            .expect("Metal activation quantization kernel initialized")
            .apply_device([input], &config, stream)
    })?;

    if outputs.len() != 2 {
        return Err(Exception::custom(format!(
            "block-FP8 activation quantization returned {} outputs, expected 2",
            outputs.len()
        )));
    }
    let scales = outputs.pop().expect("activation scale output");
    let values = outputs.pop().expect("quantized activation output");
    Ok(QuantizedActivations { values, scales })
}

#[cfg(not(feature = "cuda"))]
fn activation_quantization_kernel_metal() -> Result<MetalKernel, Exception> {
    MetalKernel::new(
        "block_fp8_activation_quantization",
        ["input"],
        ["quantized", "activation_scale"],
        concat!(
            "uint block = thread_position_in_grid.x / SCALE_BLOCK;",
            "uint lane = thread_position_in_grid.x % SCALE_BLOCK;",
            "uint row = block / SCALE_COLS;",
            "uint scale_col = block % SCALE_COLS;",
            "uint col = scale_col * SCALE_BLOCK + lane;",
            "bool valid = col < IN_DIM;",
            "float value = valid ? float(input[row * IN_DIM + col]) : 0.0f;",
            "threadgroup float maxima[SCALE_BLOCK];",
            "threadgroup float block_scale[1];",
            "maxima[lane] = abs(value);",
            "threadgroup_barrier(mem_flags::mem_threadgroup);",
            "for (uint stride = SCALE_BLOCK / 2; stride > 0; stride /= 2) {",
            " if (lane < stride) maxima[lane] = max(maxima[lane], maxima[lane + stride]);",
            " threadgroup_barrier(mem_flags::mem_threadgroup);",
            "}",
            "if (lane == 0) {",
            " block_scale[0] = max(maxima[0], 1.0e-4f) / 448.0f;",
            " activation_scale[block] = block_scale[0];",
            "}",
            "threadgroup_barrier(mem_flags::mem_threadgroup);",
            "if (valid) quantized[row * IN_DIM + col] = float_to_fp8_e4m3(value / block_scale[0]);"
        ),
        METAL_HEADER,
        true,
        false,
    )
}

#[cfg(feature = "cuda")]
fn activation_quantization_kernel_cuda() -> Result<CudaKernel, Exception> {
    CudaKernel::new(
        "block_fp8_activation_quantization",
        ["input"],
        ["quantized", "activation_scale"],
        concat!(
            "uint32_t block = blockIdx.x;",
            "uint32_t lane = threadIdx.x;",
            "uint32_t row = block / SCALE_COLS;",
            "uint32_t scale_col = block % SCALE_COLS;",
            "uint32_t col = scale_col * SCALE_BLOCK + lane;",
            "bool valid = col < IN_DIM;",
            "float value = valid ? float(input[row * IN_DIM + col]) : 0.0f;",
            "__shared__ float maxima[SCALE_BLOCK];",
            "__shared__ float block_scale;",
            "maxima[lane] = fabsf(value);",
            "__syncthreads();",
            "for (uint32_t stride = SCALE_BLOCK / 2; stride > 0; stride /= 2) {",
            " if (lane < stride) maxima[lane] = fmaxf(maxima[lane], maxima[lane + stride]);",
            " __syncthreads();",
            "}",
            "if (lane == 0) {",
            " block_scale = fmaxf(maxima[0], 1.0e-4f) / 448.0f;",
            " activation_scale[block] = block_scale;",
            "}",
            "__syncthreads();",
            "if (valid) quantized[row * IN_DIM + col] = float_to_fp8_e4m3(value / block_scale);"
        ),
        CUDA_HEADER,
        true,
        0,
    )
}

/// Expands one rank-2 block-scaled E4M3 matrix to floating point.
///
/// This is intended for algorithms that must absorb or slice a projection
/// matrix rather than apply it as a conventional linear operation. Callers
/// should keep the result transient.
pub fn dequantize(weight: &Array, scale: &Array, stream: &Stream) -> Result<Array, Exception> {
    if weight.ndim() != 2 || scale.ndim() != 2 {
        return Err(Exception::custom(
            "block-FP8 dequantization expects rank-2 weight and scale arrays",
        ));
    }
    let out_dim = weight.dim(0);
    let in_dim = weight.dim(1);
    let scale = Array::repeat_axis::<f32>(scale.clone(), 128, 0, stream)?;
    let scale = Array::repeat_axis::<f32>(scale, 128, 1, stream)?;
    weight.from_fp8(Dtype::Float32, stream)?.multiply(
        scale.try_index_device((..out_dim, ..in_dim), stream)?,
        stream,
    )
}

/// Applies a rank-2 block-scaled E4M3 weight matrix.
pub fn linear(
    input: &Array,
    weight: &Array,
    scale: &Array,
    stream: &Stream,
) -> Result<Array, Exception> {
    if input.ndim() < 1 || weight.ndim() != 2 || scale.ndim() != 2 {
        return Err(Exception::custom(
            "block-FP8 linear expects an input with at least one dimension and rank-2 weight/scale arrays",
        ));
    }
    let input_shape = input.shape();
    let output_dtype = activation_dtype(input)?;
    let in_dim = input.dim(-1);
    let out_dim = weight.dim(0);
    if in_dim <= 0
        || out_dim <= 0
        || weight.dim(1) != in_dim
        || scale.dim(0) != ceil_div(out_dim, SCALE_BLOCK)
        || scale.dim(1) != ceil_div(in_dim, SCALE_BLOCK)
    {
        return Err(Exception::custom(
            "invalid block-FP8 linear weight or scale dimensions",
        ));
    }
    let rows = (input.size() as i32) / in_dim;
    if is_cpu_stream(stream)? {
        let weight = dequantize(weight, scale, stream)?;
        let output = matmul(input, &weight.transpose(stream)?, stream)?;
        return restore_activation_dtype(output, output_dtype, stream);
    }
    let input = input.reshape(&[rows, in_dim], stream)?;
    let input = quantize_activations(&input, rows, in_dim, stream)?;
    let scale_cols = scale.dim(1);

    #[cfg(feature = "cuda")]
    let out = linear_tiled_cuda(
        &input.values,
        &input.scales,
        weight,
        scale,
        rows,
        in_dim,
        out_dim,
        scale_cols,
        stream,
    )?;

    #[cfg(not(feature = "cuda"))]
    let out = if rows <= TILED_ROW_THRESHOLD {
        linear_tiled(
            &input.values,
            &input.scales,
            weight,
            scale,
            rows,
            in_dim,
            out_dim,
            scale_cols,
            stream,
        )?
    } else {
        linear_scalar(
            &input.values,
            &input.scales,
            weight,
            scale,
            rows,
            in_dim,
            out_dim,
            scale_cols,
            stream,
        )?
    };

    let mut output_shape = input_shape.to_vec();
    *output_shape.last_mut().expect("linear input rank") = out_dim;
    let out = out.reshape(&output_shape, stream)?;
    restore_activation_dtype(out, output_dtype, stream)
}

#[allow(clippy::too_many_arguments)]
#[cfg(feature = "cuda")]
fn linear_tiled_cuda(
    input: &Array,
    input_scale: &Array,
    weight: &Array,
    scale: &Array,
    rows: i32,
    in_dim: i32,
    out_dim: i32,
    scale_cols: i32,
    stream: &Stream,
) -> Result<Array, Exception> {
    LINEAR_KERNEL.with(|cell| -> Result<_, Exception> {
        if cell.borrow().is_none() {
            *cell.borrow_mut() = Some(linear_kernel_cuda()?);
        }
        let config = linear_tiled_config(rows, in_dim, out_dim, scale_cols);
        cell.borrow()
            .as_ref()
            .expect("CUDA FP8 linear kernel initialized")
            .apply_one_device([input, input_scale, weight, scale], &config, stream)
    })
}

#[cfg(feature = "cuda")]
fn linear_kernel_cuda() -> Result<CudaKernel, Exception> {
    CudaKernel::new(
        "block_fp8_linear_k16",
        ["input", "input_scale", "weight", "scale"],
        ["out"],
        concat!(
            "uint32_t out_col = blockIdx.x * blockDim.x + threadIdx.x;",
            "uint32_t row = blockIdx.y;",
            "uint32_t lane_k = threadIdx.y;",
            "__shared__ float partial[REDUCTION_TILE][OUT_TILE];",
            "float acc = 0.0f;",
            "if (out_col < OUT_DIM) {",
            " for (uint32_t k = lane_k; k < IN_DIM; k += REDUCTION_TILE) {",
            "  uint8_t raw = weight[out_col * IN_DIM + k];",
            "  float x = fp8_e4m3_to_float(input[row * IN_DIM + k]);",
            "  uint32_t scale_col = k / SCALE_BLOCK;",
            "  float xs = float(input_scale[row * SCALE_COLS + scale_col]);",
            "  float ws = float(scale[(out_col / SCALE_BLOCK) * SCALE_COLS + scale_col]);",
            "  acc += x * fp8_e4m3_to_float(raw) * xs * ws;",
            " }",
            "}",
            "partial[threadIdx.y][threadIdx.x] = acc;",
            "__syncthreads();",
            "if (lane_k == 0 && out_col < OUT_DIM) {",
            " float sum = 0.0f;",
            " for (uint32_t lane = 0; lane < REDUCTION_TILE; ++lane) sum += partial[lane][threadIdx.x];",
            " out[row * OUT_DIM + out_col] = sum;",
            "}"
        ),
        CUDA_HEADER,
        true,
        0,
    )
}

#[allow(clippy::too_many_arguments)]
#[cfg(not(feature = "cuda"))]
fn linear_tiled(
    input: &Array,
    input_scale: &Array,
    weight: &Array,
    scale: &Array,
    rows: i32,
    in_dim: i32,
    out_dim: i32,
    scale_cols: i32,
    stream: &Stream,
) -> Result<Array, Exception> {
    LINEAR_KERNEL.with(|cell| -> Result<_, Exception> {
        if cell.borrow().is_none() {
            *cell.borrow_mut() = Some(linear_kernel()?);
        }
        let config = linear_tiled_config(rows, in_dim, out_dim, scale_cols);
        cell.borrow()
            .as_ref()
            .expect("FP8 linear kernel initialized")
            .apply_one_device([input, input_scale, weight, scale], &config, stream)
    })
}

#[allow(clippy::too_many_arguments)]
#[cfg(not(feature = "cuda"))]
fn linear_scalar(
    input: &Array,
    input_scale: &Array,
    weight: &Array,
    scale: &Array,
    rows: i32,
    in_dim: i32,
    out_dim: i32,
    scale_cols: i32,
    stream: &Stream,
) -> Result<Array, Exception> {
    LINEAR_SCALAR_KERNEL.with(|cell| -> Result<_, Exception> {
        if cell.borrow().is_none() {
            *cell.borrow_mut() = Some(linear_scalar_kernel()?);
        }
        let config = MetalKernelConfig::new()
            .with_template_arg_int("IN_DIM", in_dim)
            .with_template_arg_int("OUT_DIM", out_dim)
            .with_template_arg_int("SCALE_BLOCK", SCALE_BLOCK)
            .with_template_arg_int("SCALE_COLS", scale_cols)
            .with_grid([rows * out_dim, 1, 1])
            .with_thread_group([256, 1, 1])
            .with_output_arg([rows, out_dim], Dtype::Float32);
        cell.borrow()
            .as_ref()
            .expect("scalar FP8 linear kernel initialized")
            .apply_one_device([input, input_scale, weight, scale], &config, stream)
    })
}

#[cfg(not(feature = "cuda"))]
fn linear_kernel() -> Result<MetalKernel, Exception> {
    MetalKernel::new(
        "block_fp8_linear_k16",
        ["input", "input_scale", "weight", "scale"],
        ["out"],
        concat!(
            "uint out_col = thread_position_in_grid.x;",
            "uint row = thread_position_in_grid.y / 16;",
            "uint lane_k = thread_position_in_grid.y % 16;",
            "uint local_col = thread_position_in_grid.x % 16;",
            "uint input_base = row * IN_DIM;",
            "threadgroup float partial[REDUCTION_TILE][OUT_TILE];",
            "float acc = 0.0f;",
            "if (out_col < OUT_DIM) {",
            " for (uint k = lane_k; k < IN_DIM; k += REDUCTION_TILE) {",
            "  uint8_t raw = weight[out_col * IN_DIM + k];",
            "  float x = fp8_e4m3_to_float(input[input_base + k]);",
            "  uint scale_col = k / SCALE_BLOCK;",
            "  float xs = float(input_scale[row * SCALE_COLS + scale_col]);",
            "  float ws = float(scale[(out_col / SCALE_BLOCK) * SCALE_COLS + scale_col]);",
            "  acc += x * fp8_e4m3_to_float(raw) * xs * ws;",
            " }",
            "}",
            "partial[lane_k][local_col] = acc;",
            "threadgroup_barrier(mem_flags::mem_threadgroup);",
            "if (lane_k == 0 && out_col < OUT_DIM) {",
            " float sum = 0.0f;",
            " for (uint lane = 0; lane < REDUCTION_TILE; ++lane) sum += partial[lane][local_col];",
            " out[row * OUT_DIM + out_col] = sum;",
            "}"
        ),
        METAL_HEADER,
        true,
        false,
    )
}

#[cfg(not(feature = "cuda"))]
fn linear_scalar_kernel() -> Result<MetalKernel, Exception> {
    MetalKernel::new(
        "block_fp8_linear_scalar",
        ["input", "input_scale", "weight", "scale"],
        ["out"],
        concat!(
            "uint elem = thread_position_in_grid.x;",
            "uint out_col = elem % OUT_DIM;",
            "uint row = elem / OUT_DIM;",
            "float acc = 0.0f;",
            "uint weight_base = out_col * IN_DIM;",
            "uint input_base = row * IN_DIM;",
            "uint scale_row = out_col / SCALE_BLOCK;",
            "for (uint k = 0; k < IN_DIM; ++k) {",
            " float w = fp8_e4m3_to_float(weight[weight_base + k]);",
            " uint scale_col = k / SCALE_BLOCK;",
            " float xs = float(input_scale[row * SCALE_COLS + scale_col]);",
            " float ws = float(scale[scale_row * SCALE_COLS + scale_col]);",
            " acc += fp8_e4m3_to_float(input[input_base + k]) * w * xs * ws;",
            "}",
            "out[elem] = acc;"
        ),
        METAL_HEADER,
        true,
        false,
    )
}

/// Applies a rank-3 block-scaled E4M3 expert bank to expert-major rows.
pub fn grouped_linear(
    input: &Array,
    weight: &Array,
    scale: &Array,
    group_ids: &Array,
    stream: &Stream,
) -> Result<Array, Exception> {
    if input.ndim() != 2 || weight.ndim() != 3 || scale.ndim() != 3 || group_ids.ndim() != 1 {
        return Err(Exception::custom(
            "grouped block-FP8 linear expects rank-2 input, rank-3 weight/scale, and rank-1 group ids",
        ));
    }
    let output_dtype = activation_dtype(input)?;
    let routes = input.dim(0);
    let in_dim = input.dim(1);
    let experts = weight.dim(0);
    let out_dim = weight.dim(1);
    if routes != group_ids.dim(0)
        || routes <= 0
        || experts <= 0
        || out_dim <= 0
        || in_dim <= 0
        || weight.dim(2) != in_dim
        || scale.dim(0) != experts
        || scale.dim(1) != ceil_div(out_dim, SCALE_BLOCK)
        || scale.dim(2) != ceil_div(in_dim, SCALE_BLOCK)
    {
        return Err(Exception::custom(
            "invalid grouped block-FP8 linear weight, scale, or route dimensions",
        ));
    }
    let scale_cols = scale.dim(2);
    if is_cpu_stream(stream)? {
        let weight = dequantize_grouped(weight, scale, stream)?;
        let output = grouped_matmul(
            input,
            &weight.swap_axes(-1, -2, stream)?,
            group_ids,
            true,
            stream,
        )?;
        return restore_activation_dtype(output, output_dtype, stream);
    }
    let input = quantize_activations(input, routes, in_dim, stream)?;

    #[cfg(feature = "cuda")]
    let out = grouped_linear_tiled_cuda(
        &input.values,
        &input.scales,
        weight,
        scale,
        group_ids,
        routes,
        in_dim,
        out_dim,
        scale_cols,
        stream,
    )?;

    #[cfg(not(feature = "cuda"))]
    let out = if routes <= TILED_ROW_THRESHOLD {
        grouped_linear_tiled(
            &input.values,
            &input.scales,
            weight,
            scale,
            group_ids,
            routes,
            in_dim,
            out_dim,
            scale_cols,
            stream,
        )?
    } else {
        grouped_linear_scalar(
            &input.values,
            &input.scales,
            weight,
            scale,
            group_ids,
            routes,
            in_dim,
            out_dim,
            scale_cols,
            stream,
        )?
    };

    restore_activation_dtype(out, output_dtype, stream)
}

#[allow(clippy::too_many_arguments)]
#[cfg(feature = "cuda")]
fn grouped_linear_tiled_cuda(
    input: &Array,
    input_scale: &Array,
    weight: &Array,
    scale: &Array,
    group_ids: &Array,
    routes: i32,
    in_dim: i32,
    out_dim: i32,
    scale_cols: i32,
    stream: &Stream,
) -> Result<Array, Exception> {
    GROUPED_LINEAR_KERNEL.with(|cell| -> Result<_, Exception> {
        if cell.borrow().is_none() {
            *cell.borrow_mut() = Some(grouped_linear_kernel_cuda()?);
        }
        let config = grouped_tiled_config(routes, in_dim, out_dim, scale.dim(1), scale_cols);
        cell.borrow()
            .as_ref()
            .expect("CUDA grouped FP8 linear kernel initialized")
            .apply_one_device(
                [input, input_scale, weight, scale, group_ids],
                &config,
                stream,
            )
    })
}

#[cfg(feature = "cuda")]
fn grouped_linear_kernel_cuda() -> Result<CudaKernel, Exception> {
    CudaKernel::new(
        "block_fp8_grouped_linear_k16",
        ["input", "input_scale", "weight", "scale", "group_ids"],
        ["out"],
        concat!(
            "uint32_t out_col = blockIdx.x * blockDim.x + threadIdx.x;",
            "uint32_t route = blockIdx.y;",
            "uint32_t lane_k = threadIdx.y;",
            "uint32_t expert = uint32_t(group_ids[route]);",
            "__shared__ float partial[REDUCTION_TILE][OUT_TILE];",
            "float acc = 0.0f;",
            "if (out_col < OUT_DIM) {",
            " for (uint32_t k = lane_k; k < IN_DIM; k += REDUCTION_TILE) {",
            "  uint32_t wi = (expert * OUT_DIM + out_col) * IN_DIM + k;",
            "  uint32_t si = (expert * SCALE_OUT + out_col / SCALE_BLOCK) * SCALE_COLS + k / SCALE_BLOCK;",
            "  float xs = float(input_scale[route * SCALE_COLS + k / SCALE_BLOCK]);",
            "  acc += fp8_e4m3_to_float(input[route * IN_DIM + k]) * fp8_e4m3_to_float(weight[wi]) * xs * float(scale[si]);",
            " }",
            "}",
            "partial[threadIdx.y][threadIdx.x] = acc;",
            "__syncthreads();",
            "if (lane_k == 0 && out_col < OUT_DIM) {",
            " float sum = 0.0f;",
            " for (uint32_t lane = 0; lane < REDUCTION_TILE; ++lane) sum += partial[lane][threadIdx.x];",
            " out[route * OUT_DIM + out_col] = sum;",
            "}"
        ),
        CUDA_HEADER,
        true,
        0,
    )
}

#[allow(clippy::too_many_arguments)]
#[cfg(not(feature = "cuda"))]
fn grouped_linear_tiled(
    input: &Array,
    input_scale: &Array,
    weight: &Array,
    scale: &Array,
    group_ids: &Array,
    routes: i32,
    in_dim: i32,
    out_dim: i32,
    scale_cols: i32,
    stream: &Stream,
) -> Result<Array, Exception> {
    GROUPED_LINEAR_KERNEL.with(|cell| -> Result<_, Exception> {
        if cell.borrow().is_none() {
            *cell.borrow_mut() = Some(grouped_linear_kernel()?);
        }
        let config = grouped_tiled_config(routes, in_dim, out_dim, scale.dim(1), scale_cols);
        cell.borrow()
            .as_ref()
            .expect("grouped FP8 linear kernel initialized")
            .apply_one_device(
                [input, input_scale, weight, scale, group_ids],
                &config,
                stream,
            )
    })
}

#[allow(clippy::too_many_arguments)]
#[cfg(not(feature = "cuda"))]
fn grouped_linear_scalar(
    input: &Array,
    input_scale: &Array,
    weight: &Array,
    scale: &Array,
    group_ids: &Array,
    routes: i32,
    in_dim: i32,
    out_dim: i32,
    scale_cols: i32,
    stream: &Stream,
) -> Result<Array, Exception> {
    GROUPED_LINEAR_SCALAR_KERNEL.with(|cell| -> Result<_, Exception> {
        if cell.borrow().is_none() {
            *cell.borrow_mut() = Some(grouped_linear_scalar_kernel()?);
        }
        let config = MetalKernelConfig::new()
            .with_template_arg_int("IN_DIM", in_dim)
            .with_template_arg_int("OUT_DIM", out_dim)
            .with_template_arg_int("SCALE_OUT", scale.dim(1))
            .with_template_arg_int("SCALE_BLOCK", SCALE_BLOCK)
            .with_template_arg_int("SCALE_COLS", scale_cols)
            .with_grid([routes * out_dim, 1, 1])
            .with_thread_group([256, 1, 1])
            .with_output_arg([routes, out_dim], Dtype::Float32);
        cell.borrow()
            .as_ref()
            .expect("scalar grouped FP8 linear kernel initialized")
            .apply_one_device(
                [input, input_scale, weight, scale, group_ids],
                &config,
                stream,
            )
    })
}

#[cfg(not(feature = "cuda"))]
fn grouped_linear_kernel() -> Result<MetalKernel, Exception> {
    MetalKernel::new(
        "block_fp8_grouped_linear_k16",
        ["input", "input_scale", "weight", "scale", "group_ids"],
        ["out"],
        concat!(
            "uint out_col = thread_position_in_grid.x;",
            "uint route = thread_position_in_grid.y / 16;",
            "uint lane_k = thread_position_in_grid.y % 16;",
            "uint local_col = thread_position_in_grid.x % 16;",
            "uint expert = uint(group_ids[route]);",
            "uint input_base = route * IN_DIM;",
            "threadgroup float partial[REDUCTION_TILE][OUT_TILE];",
            "float acc = 0.0f;",
            "if (out_col < OUT_DIM) {",
            " for (uint k = lane_k; k < IN_DIM; k += REDUCTION_TILE) {",
            "  uint wi = (expert * OUT_DIM + out_col) * IN_DIM + k;",
            "  uint si = (expert * SCALE_OUT + out_col / SCALE_BLOCK) * SCALE_COLS + k / SCALE_BLOCK;",
            "  float xs = float(input_scale[route * SCALE_COLS + k / SCALE_BLOCK]);",
            "  acc += fp8_e4m3_to_float(input[input_base + k]) * fp8_e4m3_to_float(weight[wi]) * xs * float(scale[si]);",
            " }",
            "}",
            "partial[lane_k][local_col] = acc;",
            "threadgroup_barrier(mem_flags::mem_threadgroup);",
            "if (lane_k == 0 && out_col < OUT_DIM) {",
            " float sum = 0.0f;",
            " for (uint lane = 0; lane < REDUCTION_TILE; ++lane) sum += partial[lane][local_col];",
            " out[route * OUT_DIM + out_col] = sum;",
            "}"
        ),
        METAL_HEADER,
        true,
        false,
    )
}

#[cfg(not(feature = "cuda"))]
fn grouped_linear_scalar_kernel() -> Result<MetalKernel, Exception> {
    MetalKernel::new(
        "block_fp8_grouped_linear_scalar",
        ["input", "input_scale", "weight", "scale", "group_ids"],
        ["out"],
        concat!(
            "uint elem = thread_position_in_grid.x;",
            "uint out_col = elem % OUT_DIM;",
            "uint route = elem / OUT_DIM;",
            "uint expert = uint(group_ids[route]);",
            "float acc = 0.0f;",
            "uint weight_base = (expert * OUT_DIM + out_col) * IN_DIM;",
            "uint input_base = route * IN_DIM;",
            "uint scale_base = (expert * SCALE_OUT + out_col / SCALE_BLOCK) * SCALE_COLS;",
            "for (uint k = 0; k < IN_DIM; ++k) {",
            " float w = fp8_e4m3_to_float(weight[weight_base + k]);",
            " uint scale_col = k / SCALE_BLOCK;",
            " float xs = float(input_scale[route * SCALE_COLS + scale_col]);",
            " float ws = float(scale[scale_base + scale_col]);",
            " acc += fp8_e4m3_to_float(input[input_base + k]) * w * xs * ws;",
            "}",
            "out[elem] = acc;"
        ),
        METAL_HEADER,
        true,
        false,
    )
}

/// Applies one row segment per group from a rank-2 block-FP8 matrix.
///
/// `weight` is laid out as `[groups * group_stride, input_dims]`. Each input
/// row selects a group through `group_ids`; the output uses
/// `weight[group * group_stride + row_offset .. + output_dims]`. This is useful
/// for fused projections whose logical per-group matrices are concatenated in
/// the checkpoint output dimension.
///
/// This absorbed-MLA operation deliberately keeps floating-point activations:
/// DeepSeek's released absorb path dequantizes this weight and applies an
/// einsum rather than invoking its dynamically activation-quantized linear.
#[allow(clippy::too_many_arguments)]
pub fn segmented_linear(
    input: &Array,
    weight: &Array,
    scale: &Array,
    group_ids: &Array,
    group_stride: i32,
    row_offset: i32,
    output_dims: i32,
    stream: &Stream,
) -> Result<Array, Exception> {
    if input.ndim() != 2 || weight.ndim() != 2 || scale.ndim() != 2 || group_ids.ndim() != 1 {
        return Err(Exception::custom(
            "segmented block-FP8 linear expects rank-2 input/weight/scale and rank-1 group ids",
        ));
    }
    if input.dim(0) != group_ids.dim(0)
        || input.dim(1) != weight.dim(1)
        || group_stride <= 0
        || row_offset < 0
        || output_dims <= 0
        || row_offset + output_dims > group_stride
        || weight.dim(0) % group_stride != 0
    {
        return Err(Exception::custom(
            "invalid segmented block-FP8 linear dimensions",
        ));
    }
    if is_cpu_stream(stream)? {
        return segmented_reference(
            input,
            weight,
            scale,
            group_ids,
            group_stride,
            row_offset,
            output_dims,
            false,
            stream,
        );
    }
    let routes = input.dim(0);
    SEGMENTED_LINEAR_KERNEL.with(|cell| -> Result<_, Exception> {
        if cell.borrow().is_none() {
            *cell.borrow_mut() = Some(segmented_linear_kernel()?);
        }
        let config = MetalKernelConfig::new()
            .with_template_arg_int("IN_DIM", input.dim(1))
            .with_template_arg_int("OUT_DIM", output_dims)
            .with_template_arg_int("GROUP_STRIDE", group_stride)
            .with_template_arg_int("ROW_OFFSET", row_offset)
            .with_template_arg_int("SCALE_COLS", scale.dim(1))
            .with_template_arg_int("N_ELEMS", routes * output_dims)
            .with_grid([routes * output_dims, 1, 1])
            .with_thread_group([256, 1, 1])
            .with_output_arg([routes, output_dims], Dtype::Float32);
        cell.borrow()
            .as_ref()
            .expect("segmented FP8 linear kernel initialized")
            .apply_one_device([input, weight, scale, group_ids], &config, stream)
    })
}

/// Applies the transpose of one row segment per group from a rank-2
/// block-FP8 matrix without dequantizing the complete matrix.
///
/// `input` has `segment_rows` columns and the result has `weight.dim(1)`
/// columns. Weight rows are selected with the same grouped segment layout as
/// [`segmented_linear`].
#[allow(clippy::too_many_arguments)]
pub fn segmented_transposed_linear(
    input: &Array,
    weight: &Array,
    scale: &Array,
    group_ids: &Array,
    group_stride: i32,
    row_offset: i32,
    stream: &Stream,
) -> Result<Array, Exception> {
    if input.ndim() != 2 || weight.ndim() != 2 || scale.ndim() != 2 || group_ids.ndim() != 1 {
        return Err(Exception::custom(
            "segmented transposed block-FP8 linear expects rank-2 input/weight/scale and rank-1 group ids",
        ));
    }
    if input.dim(0) != group_ids.dim(0)
        || group_stride <= 0
        || row_offset < 0
        || input.dim(1) <= 0
        || row_offset + input.dim(1) > group_stride
        || weight.dim(0) % group_stride != 0
    {
        return Err(Exception::custom(
            "invalid segmented transposed block-FP8 linear dimensions",
        ));
    }
    if is_cpu_stream(stream)? {
        return segmented_reference(
            input,
            weight,
            scale,
            group_ids,
            group_stride,
            row_offset,
            input.dim(1),
            true,
            stream,
        );
    }
    let routes = input.dim(0);
    let output_dims = weight.dim(1);
    SEGMENTED_TRANSPOSED_LINEAR_KERNEL.with(|cell| -> Result<_, Exception> {
        if cell.borrow().is_none() {
            *cell.borrow_mut() = Some(segmented_transposed_linear_kernel()?);
        }
        let config = MetalKernelConfig::new()
            .with_template_arg_int("SEGMENT_ROWS", input.dim(1))
            .with_template_arg_int("OUT_DIM", output_dims)
            .with_template_arg_int("GROUP_STRIDE", group_stride)
            .with_template_arg_int("ROW_OFFSET", row_offset)
            .with_template_arg_int("SCALE_COLS", scale.dim(1))
            .with_template_arg_int("N_ELEMS", routes * output_dims)
            .with_grid([routes * output_dims, 1, 1])
            .with_thread_group([256, 1, 1])
            .with_output_arg([routes, output_dims], Dtype::Float32);
        cell.borrow()
            .as_ref()
            .expect("segmented transposed FP8 linear kernel initialized")
            .apply_one_device([input, weight, scale, group_ids], &config, stream)
    })
}

#[cfg(not(feature = "cuda"))]
fn segmented_linear_kernel() -> Result<MetalKernel, Exception> {
    MetalKernel::new(
        "block_fp8_segmented_linear",
        ["input", "weight", "scale", "group_ids"],
        ["out"],
        concat!(
            "uint elem = thread_position_in_grid.x;",
            "if (elem >= N_ELEMS) return;",
            "uint out_col = elem % OUT_DIM;",
            "uint route = elem / OUT_DIM;",
            "uint group = uint(group_ids[route]);",
            "uint weight_row = group * GROUP_STRIDE + ROW_OFFSET + out_col;",
            "uint weight_base = weight_row * IN_DIM;",
            "uint input_base = route * IN_DIM;",
            "uint scale_base = (weight_row / 128) * SCALE_COLS;",
            "float acc = 0.0f;",
            "for (uint k = 0; k < IN_DIM; ++k) {",
            " float w = fp8_e4m3_to_float(weight[weight_base + k]);",
            " float s = float(scale[scale_base + k / 128]);",
            " acc += float(input[input_base + k]) * w * s;",
            "}",
            "out[elem] = acc;"
        ),
        METAL_HEADER,
        true,
        false,
    )
}

#[cfg(not(feature = "cuda"))]
fn segmented_transposed_linear_kernel() -> Result<MetalKernel, Exception> {
    MetalKernel::new(
        "block_fp8_segmented_transposed_linear",
        ["input", "weight", "scale", "group_ids"],
        ["out"],
        concat!(
            "uint elem = thread_position_in_grid.x;",
            "if (elem >= N_ELEMS) return;",
            "uint out_col = elem % OUT_DIM;",
            "uint route = elem / OUT_DIM;",
            "uint group = uint(group_ids[route]);",
            "uint input_base = route * SEGMENT_ROWS;",
            "uint first_weight_row = group * GROUP_STRIDE + ROW_OFFSET;",
            "float acc = 0.0f;",
            "for (uint k = 0; k < SEGMENT_ROWS; ++k) {",
            " uint weight_row = first_weight_row + k;",
            " uint weight_idx = weight_row * OUT_DIM + out_col;",
            " uint scale_idx = (weight_row / 128) * SCALE_COLS + out_col / 128;",
            " acc += float(input[input_base + k]) * fp8_e4m3_to_float(weight[weight_idx]) * float(scale[scale_idx]);",
            "}",
            "out[elem] = acc;"
        ),
        METAL_HEADER,
        true,
        false,
    )
}

#[cfg(feature = "cuda")]
fn segmented_linear_kernel() -> Result<CudaKernel, Exception> {
    CudaKernel::new(
        "block_fp8_segmented_linear",
        ["input", "weight", "scale", "group_ids"],
        ["out"],
        concat!(
            "auto elem = cooperative_groups::this_grid().thread_rank();",
            "if (elem >= N_ELEMS) return;",
            "uint32_t out_col = elem % OUT_DIM;",
            "uint32_t route = elem / OUT_DIM;",
            "uint32_t group = uint32_t(group_ids[route]);",
            "uint32_t weight_row = group * GROUP_STRIDE + ROW_OFFSET + out_col;",
            "uint32_t weight_base = weight_row * IN_DIM;",
            "uint32_t input_base = route * IN_DIM;",
            "uint32_t scale_base = (weight_row / 128) * SCALE_COLS;",
            "float acc = 0.0f;",
            "for (uint32_t k = 0; k < IN_DIM; ++k) {",
            " float w = fp8_e4m3_to_float(weight[weight_base + k]);",
            " float s = float(scale[scale_base + k / 128]);",
            " acc += float(input[input_base + k]) * w * s;",
            "}",
            "out[elem] = acc;"
        ),
        CUDA_HEADER,
        true,
        0,
    )
}

#[cfg(feature = "cuda")]
fn segmented_transposed_linear_kernel() -> Result<CudaKernel, Exception> {
    CudaKernel::new(
        "block_fp8_segmented_transposed_linear",
        ["input", "weight", "scale", "group_ids"],
        ["out"],
        concat!(
            "auto elem = cooperative_groups::this_grid().thread_rank();",
            "if (elem >= N_ELEMS) return;",
            "uint32_t out_col = elem % OUT_DIM;",
            "uint32_t route = elem / OUT_DIM;",
            "uint32_t group = uint32_t(group_ids[route]);",
            "uint32_t input_base = route * SEGMENT_ROWS;",
            "uint32_t first_weight_row = group * GROUP_STRIDE + ROW_OFFSET;",
            "float acc = 0.0f;",
            "for (uint32_t k = 0; k < SEGMENT_ROWS; ++k) {",
            " uint32_t weight_row = first_weight_row + k;",
            " uint32_t weight_idx = weight_row * OUT_DIM + out_col;",
            " uint32_t scale_idx = (weight_row / 128) * SCALE_COLS + out_col / 128;",
            " acc += float(input[input_base + k]) * fp8_e4m3_to_float(weight[weight_idx]) * float(scale[scale_idx]);",
            "}",
            "out[elem] = acc;"
        ),
        CUDA_HEADER,
        true,
        0,
    )
}

#[cfg(not(feature = "cuda"))]
const METAL_HEADER: &str = concat!(
    "float fp8_e4m3_to_float(uint8_t bits) {",
    " if ((bits & 127) == 127) return as_type<float>(0x7fc00000u);",
    " uint16_t v = uint16_t(bits & 127) << 7;",
    " half converted = as_type<half>(v);",
    " converted *= 256.0h;",
    " return (bits & 128) ? -float(converted) : float(converted);",
    "}\n",
    "uint8_t float_to_fp8_e4m3(float value) {",
    " if (isnan(value)) return uint8_t(0x7f);",
    " uint sign = (as_type<uint>(value) >> 24) & 0x80u;",
    " float magnitude = min(abs(value), 448.0f);",
    " if (magnitude == 0.0f) return uint8_t(sign);",
    " uint code;",
    " if (magnitude < 0.015625f) {",
    "  int mantissa = int(rint(magnitude * 512.0f));",
    "  if (mantissa < 0) mantissa = 0;",
    "  code = mantissa >= 8 ? 8u : uint(mantissa);",
    " } else {",
    "  int exponent = int(floor(log2(magnitude)));",
    "  float base = exp2(float(exponent));",
    "  int mantissa = int(rint((magnitude / base - 1.0f) * 8.0f));",
    "  if (mantissa == 8) { exponent += 1; mantissa = 0; }",
    "  code = uint((exponent + 7) * 8 + mantissa);",
    "  if (code > 0x7eu) code = 0x7eu;",
    " }",
    " return uint8_t(sign | code);",
    "}\n",
);

#[cfg(feature = "cuda")]
const CUDA_HEADER: &str = concat!(
    "#include <cooperative_groups.h>\n",
    "#include <cuda_fp16.h>\n",
    "#include <math.h>\n",
    "#include <stdint.h>\n",
    "__device__ __forceinline__ float fp8_e4m3_to_float(uint8_t bits) {",
    " if ((bits & 127) == 127) return NAN;",
    " uint16_t v = uint16_t(bits & 127) << 7;",
    " float converted = __half2float(__ushort_as_half(v)) * 256.0f;",
    " return (bits & 128) ? -converted : converted;",
    "}\n",
    "__device__ __forceinline__ uint8_t float_to_fp8_e4m3(float value) {",
    " if (isnan(value)) return uint8_t(0x7f);",
    " uint32_t sign = (__float_as_uint(value) >> 24) & 0x80u;",
    " float magnitude = fminf(fabsf(value), 448.0f);",
    " if (magnitude == 0.0f) return uint8_t(sign);",
    " uint32_t code;",
    " if (magnitude < 0.015625f) {",
    "  int mantissa = int(nearbyintf(magnitude * 512.0f));",
    "  if (mantissa < 0) mantissa = 0;",
    "  code = mantissa >= 8 ? 8u : uint32_t(mantissa);",
    " } else {",
    "  int exponent = int(floorf(log2f(magnitude)));",
    "  float base = exp2f(float(exponent));",
    "  int mantissa = int(nearbyintf((magnitude / base - 1.0f) * 8.0f));",
    "  if (mantissa == 8) { exponent += 1; mantissa = 0; }",
    "  code = uint32_t((exponent + 7) * 8 + mantissa);",
    "  if (code > 0x7eu) code = 0x7eu;",
    " }",
    " return uint8_t(sign | code);",
    "}\n",
);

#[cfg(test)]
mod tests {
    use super::{
        grouped_linear, linear, quantize_activations, segmented_linear, segmented_transposed_linear,
    };
    use safemlx::{ops::indexing::TryIndexOp, Array, Device, DeviceType, Dtype, ExecutionContext};

    fn assert_block_fp8_dense_and_grouped_projections(device_type: DeviceType) {
        let context = ExecutionContext::new(Device::new(device_type, 0));
        let stream = context.stream();
        // E4M3 0x38 represents 1.0; inverse block scale 2.0 makes
        // every effective weight 2.0.
        let input = Array::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[1, 4]);
        let weight = Array::from_slice(&[0x38u8; 16], &[4, 4]);
        let scale = Array::from_slice(&[2.0f32], &[1, 1]);
        let output = linear(&input, &weight, &scale, stream).unwrap();
        assert_eq!(output.shape(), &[1, 4]);
        assert_eq!(
            output
                .try_index_device((0, 0), stream)
                .unwrap()
                .item::<f32>(stream),
            8.0
        );

        let bf16_input = input.as_dtype(Dtype::Bfloat16, stream).unwrap();
        let bf16_output = linear(&bf16_input, &weight, &scale, stream).unwrap();
        assert_eq!(bf16_output.dtype(), Dtype::Bfloat16);
        assert_eq!(
            bf16_output
                .as_dtype(Dtype::Float32, stream)
                .unwrap()
                .try_index_device((0, 0), stream)
                .unwrap()
                .item::<f32>(stream),
            8.0
        );

        let grouped_weight = Array::from_slice(&[0x38u8; 32], &[2, 4, 4]);
        let grouped_scale = Array::from_slice(&[2.0f32, 3.0], &[2, 1, 1]);
        let rows = Array::from_slice(&[1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], &[2, 4]);
        let groups = Array::from_slice(&[0u32, 1], &[2]);
        let output =
            grouped_linear(&rows, &grouped_weight, &grouped_scale, &groups, stream).unwrap();
        assert_eq!(output.shape(), &[2, 4]);
        assert_eq!(
            output
                .try_index_device((0, 0), stream)
                .unwrap()
                .item::<f32>(stream),
            8.0
        );
        assert_eq!(
            output
                .try_index_device((1, 0), stream)
                .unwrap()
                .item::<f32>(stream),
            12.0
        );

        // Exercise grouped row segments at real 128-row block boundaries,
        // including the transposed path used by absorbed MLA queries.
        let segmented_weight = Array::from_slice(&vec![0x38u8; 512 * 128], &[512, 128]);
        let segmented_scale = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4, 1]);
        let segmented_groups = Array::from_slice(&[0u32, 1], &[2]);
        let query = Array::from_slice(&vec![1.0f32; 2 * 128], &[2, 128]);
        let absorbed_query = segmented_transposed_linear(
            &query,
            &segmented_weight,
            &segmented_scale,
            &segmented_groups,
            256,
            0,
            stream,
        )
        .unwrap();
        assert_eq!(absorbed_query.shape(), &[2, 128]);
        assert_eq!(
            absorbed_query
                .try_index_device((0, 0), stream)
                .unwrap()
                .item::<f32>(stream),
            128.0
        );
        assert_eq!(
            absorbed_query
                .try_index_device((1, 0), stream)
                .unwrap()
                .item::<f32>(stream),
            384.0
        );

        let context = Array::from_slice(&vec![1.0f32; 2 * 128], &[2, 128]);
        let absorbed_output = segmented_linear(
            &context,
            &segmented_weight,
            &segmented_scale,
            &segmented_groups,
            256,
            128,
            128,
            stream,
        )
        .unwrap();
        assert_eq!(absorbed_output.shape(), &[2, 128]);
        assert_eq!(
            absorbed_output
                .try_index_device((0, 0), stream)
                .unwrap()
                .item::<f32>(stream),
            256.0
        );
        assert_eq!(
            absorbed_output
                .try_index_device((1, 0), stream)
                .unwrap()
                .item::<f32>(stream),
            512.0
        );
    }

    #[test]
    fn block_fp8_dense_and_grouped_projections() {
        assert_block_fp8_dense_and_grouped_projections(DeviceType::Gpu);
    }

    #[test]
    fn block_fp8_cpu_reference_projections() {
        assert_block_fp8_dense_and_grouped_projections(DeviceType::Cpu);
    }

    #[test]
    fn dynamic_activation_quantization_matches_e4m3_and_clamped_scale() {
        let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let stream = context.stream();
        let mut input = vec![0.0f32; 256];
        for (index, value) in [448.0, -448.0, 1.0, 0.5, 0.015625, 0.001953125]
            .into_iter()
            .enumerate()
        {
            input[index] = value;
        }
        let input = Array::from_slice(&input, &[1, 256]);
        let quantized = quantize_activations(&input, 1, 256, stream).unwrap();

        for (index, expected) in [0x7eu8, 0xfe, 0x38, 0x30, 0x08, 0x01]
            .into_iter()
            .enumerate()
        {
            assert_eq!(
                quantized
                    .values
                    .try_index_device((0, index as i32), stream)
                    .unwrap()
                    .item::<u8>(stream),
                expected
            );
        }
        assert_eq!(
            quantized
                .scales
                .try_index_device((0, 0), stream)
                .unwrap()
                .item::<f32>(stream),
            1.0
        );
        let zero_scale = quantized
            .scales
            .try_index_device((0, 1), stream)
            .unwrap()
            .item::<f32>(stream);
        assert!((zero_scale - 1.0e-4 / 448.0).abs() < 1.0e-12);

        // Compare every finite positive and negative E4M3FN encoding against
        // MLX's native conversion. Each 128-value block includes magnitude
        // 448, so the dynamic scale is exactly one.
        let encoded = (0u8..=126)
            .chain(std::iter::once(126))
            .chain(128u8..=254)
            .chain(std::iter::once(254))
            .collect::<Vec<_>>();
        let decoded = Array::from_slice(&encoded, &[2, 128])
            .from_fp8(Dtype::Float32, stream)
            .unwrap();
        let quantized = quantize_activations(&decoded, 2, 128, stream).unwrap();
        for (index, expected) in encoded.into_iter().enumerate() {
            assert_eq!(
                quantized
                    .values
                    .try_index_device((index as i32 / 128, index as i32 % 128), stream)
                    .unwrap()
                    .item::<u8>(stream),
                expected
            );
        }
    }

    #[test]
    fn block_fp8_dense_and_grouped_scale_block_boundaries() {
        let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let stream = context.stream();
        let in_dim = 129;
        let out_dim = 130;

        // Every packed weight is 1.0, making the expected result depend only
        // on the four independently scaled 128x128 weight blocks.
        let input = Array::from_slice(
            &[vec![1.0f32; in_dim as usize], vec![2.0f32; in_dim as usize]].concat(),
            &[2, in_dim],
        );
        let weight = Array::from_slice(
            &vec![0x38u8; (out_dim * in_dim) as usize],
            &[out_dim, in_dim],
        );
        let scale = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]);
        let output = linear(&input, &weight, &scale, stream).unwrap();
        for (row, low, high) in [(0, 130.0f32, 388.0f32), (1, 260.0, 776.0)] {
            assert_eq!(
                output
                    .try_index_device((row, 0), stream)
                    .unwrap()
                    .item::<f32>(stream),
                low
            );
            assert_eq!(
                output
                    .try_index_device((row, 129), stream)
                    .unwrap()
                    .item::<f32>(stream),
                high
            );
        }

        let grouped_weight = Array::from_slice(
            &vec![0x38u8; (2 * out_dim * in_dim) as usize],
            &[2, out_dim, in_dim],
        );
        let grouped_scale =
            Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]);
        let grouped_input = Array::from_slice(&vec![1.0f32; (2 * in_dim) as usize], &[2, in_dim]);
        let groups = Array::from_slice(&[1u32, 0], &[2]);
        let output = grouped_linear(
            &grouped_input,
            &grouped_weight,
            &grouped_scale,
            &groups,
            stream,
        )
        .unwrap();
        for (route, low, high) in [(0, 646.0f32, 904.0f32), (1, 130.0, 388.0)] {
            assert_eq!(
                output
                    .try_index_device((route, 0), stream)
                    .unwrap()
                    .item::<f32>(stream),
                low
            );
            assert_eq!(
                output
                    .try_index_device((route, 129), stream)
                    .unwrap()
                    .item::<f32>(stream),
                high
            );
        }
    }
}
