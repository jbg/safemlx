//! Checkpoint-native quantized storage and device execution.
//!
//! Native tensors keep their physical checkpoint blocks intact and describe
//! logical matrices as zero-copy row views. Backends select kernels from the
//! format, operation, shape, and device; callers remain independent of the
//! originating model architecture. Unsupported operations use a transient
//! dequantized fallback and never require a second persistent affine copy.
//!
//! The bundled MLX C API exposes managed host buffers, but its public contract
//! still describes their input as copied and it exposes no API that wraps an
//! mmap region as a guaranteed Metal shared buffer. `Array::from_slice` also
//! documents and performs a copy. Native loading therefore makes one
//! persistent MLX-owned raw-byte copy today. [`NativeStorageKind`] is the seam
//! for a future mmap/external-buffer owner once the C API can guarantee buffer
//! identity and device accessibility; logical views already share ownership
//! through `Arc` and will not need to change.

use std::{cell::RefCell, sync::Arc};

use crate::{
    error::Exception,
    fast::{MetalKernel, MetalKernelConfig},
    ops::{grouped_matmul, indexing::TryIndexOp, matmul, sum_axis},
    transforms::eval,
    Array, DeviceType, Dtype, Stream,
};

const Q4_K_BLOCK_VALUES: i32 = 256;
const Q4_K_BLOCK_BYTES: i32 = 144;
const Q5_1_BLOCK_VALUES: i32 = 32;
const Q5_1_BLOCK_BYTES: i32 = 24;
const OUT_TILE: i32 = 4;
const REDUCTION_TILE: i32 = 32;

thread_local! {
    static Q4K_LINEAR_KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static Q4K_GROUPED_KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static Q4K_EMBEDDING_KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static Q4K_GATE_UP_KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static Q4K_DOWN_REDUCE_KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static Q5_1_GROUPED_KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static Q5_1_DOWN_REDUCE_KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
}

/// Physical quantization encoding retained from a checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NativeQuantizationFormat {
    /// GGUF/GGML Q4_K blocks: 256 weights in 144 bytes.
    GgufQ4K,
    /// GGUF/GGML Q5_1 blocks: 32 weights in 24 bytes.
    GgufQ5_1,
}

/// Persistent native and generic-fallback quantization storage loaded by a model.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NativeQuantizationStats {
    /// Physical tensors retained in checkpoint-native representation.
    pub native_tensor_count: u64,
    /// Persistent checkpoint-native bytes.
    pub native_bytes: u64,
    /// Quantized physical tensors using the generic converted representation.
    pub fallback_tensor_count: u64,
    /// Original checkpoint bytes represented by fallback tensors.
    pub fallback_checkpoint_bytes: u64,
    /// Native Q4_K physical tensors.
    pub q4k_tensor_count: u64,
    /// Native Q4_K bytes.
    pub q4k_bytes: u64,
    /// Native Q5_1 physical tensors.
    pub q5_1_tensor_count: u64,
    /// Native Q5_1 bytes.
    pub q5_1_bytes: u64,
}

impl NativeQuantizationStats {
    /// Records a quantized checkpoint tensor as initially using fallback storage.
    pub fn record_fallback(&mut self, bytes: u64) {
        self.fallback_tensor_count += 1;
        self.fallback_checkpoint_bytes += bytes;
    }

    /// Moves one previously counted fallback tensor to native storage.
    pub fn promote_native(&mut self, format: NativeQuantizationFormat, bytes: u64) {
        self.fallback_tensor_count = self.fallback_tensor_count.saturating_sub(1);
        self.fallback_checkpoint_bytes = self.fallback_checkpoint_bytes.saturating_sub(bytes);
        self.native_tensor_count += 1;
        self.native_bytes += bytes;
        match format {
            NativeQuantizationFormat::GgufQ4K => {
                self.q4k_tensor_count += 1;
                self.q4k_bytes += bytes;
            }
            NativeQuantizationFormat::GgufQ5_1 => {
                self.q5_1_tensor_count += 1;
                self.q5_1_bytes += bytes;
            }
        }
    }
}

/// How the persistent raw bytes are owned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeStorageKind {
    /// MLX owns one minimally copied raw byte buffer.
    ///
    /// The current C custom-kernel path cannot portably promise that an mmap
    /// pointer becomes a no-copy Metal buffer. Keeping this distinction in the
    /// public representation allows a managed external/mmap owner to be added
    /// without changing logical views or operation dispatch.
    MlxOwnedCopy,
}

/// Native operation classes used for backend capability selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeOperation {
    /// Matrix multiplication with the stored matrix transposed.
    Linear,
    /// Matrix multiplication in the stored matrix direction.
    LinearUntransposed,
    /// Lookup and dequantize selected physical rows.
    Embedding,
    /// Expert-major projection selected by one expert id per input row.
    GroupedLinear,
    /// Direct selected-expert fused gate/up projection.
    SelectedGateUp,
    /// Direct selected-expert down projection and weighted reduction.
    SelectedDownReduce,
}

/// Device execution backend selected independently of model architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeExecutionBackend {
    /// Custom kernels evaluated through [`MetalKernel`].
    Metal,
    /// Transient dequantization followed by portable MLX operations.
    GenericFallback,
}

/// Device-independent capability description for a native format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeCapabilities {
    /// Whether transposed linear execution is available.
    pub linear: bool,
    /// Whether untransposed linear execution is available without conversion.
    pub linear_untransposed: bool,
    /// Whether native embedding row lookup is available.
    pub embedding: bool,
    /// Whether expert-major grouped execution is available.
    pub grouped_linear: bool,
    /// Whether direct selected gate/up execution is available.
    pub selected_gate_up: bool,
    /// Whether direct selected down/reduction execution is available.
    pub selected_down_reduce: bool,
}

impl NativeCapabilities {
    /// Returns whether one operation is supported.
    pub const fn supports(self, operation: NativeOperation) -> bool {
        match operation {
            NativeOperation::Linear => self.linear,
            NativeOperation::LinearUntransposed => self.linear_untransposed,
            NativeOperation::Embedding => self.embedding,
            NativeOperation::GroupedLinear => self.grouped_linear,
            NativeOperation::SelectedGateUp => self.selected_gate_up,
            NativeOperation::SelectedDownReduce => self.selected_down_reduce,
        }
    }
}

/// Persistent physical byte storage shared by one or more logical views.
#[derive(Debug)]
pub struct NativeStorage {
    format: NativeQuantizationFormat,
    bytes: Array,
    byte_len: usize,
    kind: NativeStorageKind,
}

impl NativeStorage {
    /// Physical native format.
    pub fn format(&self) -> NativeQuantizationFormat {
        self.format
    }

    /// Number of persistent raw checkpoint bytes.
    pub fn byte_len(&self) -> usize {
        self.byte_len
    }

    /// Raw-storage ownership mode.
    pub fn kind(&self) -> NativeStorageKind {
        self.kind
    }

    /// Raw MLX byte array consumed by device kernels.
    pub fn bytes(&self) -> &Array {
        &self.bytes
    }
}

/// A zero-copy logical matrix-bank view over physical Q4_K rows.
///
/// Physical rows are `[matrix, physical_row, input]`. `row_start..row_start +
/// rows` selects the logical rows inside every matrix, which represents fused
/// gate/up expert banks without splitting or repacking their bytes.
#[derive(Debug, Clone)]
pub struct NativeQuantizedTensor {
    storage: Arc<NativeStorage>,
    matrix_count: i32,
    physical_rows: i32,
    row_start: i32,
    rows: i32,
    columns: i32,
}

impl NativeQuantizedTensor {
    /// Copies one little-endian Q4_K physical matrix bank into raw MLX storage.
    ///
    /// `shape` may be `[rows, columns]` or `[matrices, rows, columns]`.
    pub fn from_q4k_bytes(data: &[u8], shape: &[i32], stream: &Stream) -> Result<Self, Exception> {
        Self::from_native_bytes(
            data,
            shape,
            NativeQuantizationFormat::GgufQ4K,
            Q4_K_BLOCK_VALUES,
            Q4_K_BLOCK_BYTES,
            stream,
        )
    }

    /// Copies one little-endian Q5_1 physical matrix bank into raw MLX storage.
    ///
    /// `shape` may be `[rows, columns]` or `[matrices, rows, columns]`.
    pub fn from_q5_1_bytes(data: &[u8], shape: &[i32], stream: &Stream) -> Result<Self, Exception> {
        Self::from_native_bytes(
            data,
            shape,
            NativeQuantizationFormat::GgufQ5_1,
            Q5_1_BLOCK_VALUES,
            Q5_1_BLOCK_BYTES,
            stream,
        )
    }

    fn from_native_bytes(
        data: &[u8],
        shape: &[i32],
        format: NativeQuantizationFormat,
        block_values: i32,
        block_bytes: i32,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let (matrix_count, physical_rows, columns) = match shape {
            [rows, columns] => (1, *rows, *columns),
            [matrices, rows, columns] => (*matrices, *rows, *columns),
            _ => {
                return Err(Exception::custom(format!(
                    "native {format:?} expects rank-2 or rank-3 shape, got {shape:?}"
                )))
            }
        };
        if matrix_count <= 0 || physical_rows <= 0 || columns <= 0 || columns % block_values != 0 {
            return Err(Exception::custom(format!(
                "invalid native {format:?} shape {shape:?}; input columns must be a positive multiple of {block_values}"
            )));
        }
        let expected = i64::from(matrix_count)
            * i64::from(physical_rows)
            * i64::from(columns / block_values)
            * i64::from(block_bytes);
        if expected != data.len() as i64 {
            return Err(Exception::custom(format!(
                "native {format:?} payload has {} bytes, expected {expected} for shape {shape:?}",
                data.len()
            )));
        }
        let len = i32::try_from(data.len())
            .map_err(|_| Exception::custom("native tensor exceeds MLX i32 array limits"))?;
        let source = Array::from_slice(data, &[len]);
        let bytes = source.copy(stream)?;
        eval([&bytes])?;
        let storage = Arc::new(NativeStorage {
            format,
            bytes,
            byte_len: data.len(),
            kind: NativeStorageKind::MlxOwnedCopy,
        });
        Ok(Self {
            storage,
            matrix_count,
            physical_rows,
            row_start: 0,
            rows: physical_rows,
            columns,
        })
    }

    /// Physical storage shared by this logical view.
    pub fn storage(&self) -> &Arc<NativeStorage> {
        &self.storage
    }

    /// Native encoding.
    pub fn format(&self) -> NativeQuantizationFormat {
        self.storage.format
    }

    /// Logical shape, including an expert/matrix dimension when present.
    pub fn shape(&self) -> Vec<i32> {
        if self.matrix_count == 1 {
            vec![self.rows, self.columns]
        } else {
            vec![self.matrix_count, self.rows, self.columns]
        }
    }

    /// Number of matrices in the physical bank.
    pub fn matrix_count(&self) -> i32 {
        self.matrix_count
    }

    /// Number of logical output rows per matrix.
    pub fn rows(&self) -> i32 {
        self.rows
    }

    /// Number of input columns per matrix.
    pub fn columns(&self) -> i32 {
        self.columns
    }

    /// Logical starting row within every physical matrix.
    pub fn row_start(&self) -> i32 {
        self.row_start
    }

    /// Number of physical rows per matrix.
    pub fn physical_rows(&self) -> i32 {
        self.physical_rows
    }

    /// Creates a zero-copy logical row segment inside every physical matrix.
    pub fn row_view(&self, row_start: i32, rows: i32) -> Result<Self, Exception> {
        if row_start < 0 || rows <= 0 || row_start + rows > self.rows {
            return Err(Exception::custom(format!(
                "native row view {row_start}..{} exceeds {} logical rows",
                row_start + rows,
                self.rows
            )));
        }
        Ok(Self {
            storage: Arc::clone(&self.storage),
            matrix_count: self.matrix_count,
            physical_rows: self.physical_rows,
            row_start: self.row_start + row_start,
            rows,
            columns: self.columns,
        })
    }

    /// Capabilities implemented by the native Q4_K representation.
    pub fn capabilities(&self) -> NativeCapabilities {
        match self.format() {
            NativeQuantizationFormat::GgufQ4K => NativeCapabilities {
                linear: self.matrix_count == 1,
                linear_untransposed: false,
                embedding: self.matrix_count == 1,
                grouped_linear: true,
                selected_gate_up: true,
                selected_down_reduce: true,
            },
            NativeQuantizationFormat::GgufQ5_1 => NativeCapabilities {
                linear: false,
                linear_untransposed: false,
                embedding: false,
                grouped_linear: true,
                selected_gate_up: false,
                selected_down_reduce: true,
            },
        }
    }

    /// Native capabilities available on the selected device backend.
    pub fn capabilities_on(&self, stream: &Stream) -> Result<NativeCapabilities, Exception> {
        if native_execution_backend(stream)? == NativeExecutionBackend::Metal {
            Ok(self.capabilities())
        } else {
            Ok(NativeCapabilities {
                linear: false,
                linear_untransposed: false,
                embedding: false,
                grouped_linear: false,
                selected_gate_up: false,
                selected_down_reduce: false,
            })
        }
    }

    /// Applies this native matrix to `input`.
    ///
    /// Metal uses direct block dequantization. CPU and unsupported devices use
    /// a transient float matrix without retaining affine companions.
    pub fn linear(
        &self,
        input: &Array,
        transpose: bool,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        if self.matrix_count != 1 {
            return Err(Exception::custom(
                "native linear expects a single logical matrix",
            ));
        }
        let operation = if transpose {
            NativeOperation::Linear
        } else {
            NativeOperation::LinearUntransposed
        };
        if !self.capabilities_on(stream)?.supports(operation) {
            return self.linear_fallback(input, transpose, stream);
        }
        if native_execution_backend(stream)? == NativeExecutionBackend::Metal
            && transpose
            && self.format() == NativeQuantizationFormat::GgufQ4K
        {
            return q4k_linear_metal(input, self, stream);
        }
        self.linear_fallback(input, transpose, stream)
    }

    /// Looks up and dequantizes selected rows from a native Q4_K matrix.
    pub fn embedding(&self, indices: &Array, stream: &Stream) -> Result<Array, Exception> {
        if self.matrix_count != 1 {
            return Err(Exception::custom(
                "native embedding expects a single logical matrix",
            ));
        }
        if native_execution_backend(stream)? == NativeExecutionBackend::Metal
            && self.format() == NativeQuantizationFormat::GgufQ4K
        {
            return q4k_embedding_metal(indices, self, stream);
        }
        let dense = self.dequantize(stream)?;
        dense.try_index_device(indices, stream)
    }

    /// Transiently dequantizes this logical view to float32.
    pub fn dequantize(&self, stream: &Stream) -> Result<Array, Exception> {
        let evaluated = self.storage.bytes.evaluated()?;
        let raw = evaluated.as_slice::<u8>();
        let values = match self.format() {
            NativeQuantizationFormat::GgufQ4K => decode_q4k_view(raw, self)?,
            NativeQuantizationFormat::GgufQ5_1 => decode_q5_1_view(raw, self)?,
        };
        let shape = if self.matrix_count == 1 {
            vec![self.rows, self.columns]
        } else {
            vec![self.matrix_count, self.rows, self.columns]
        };
        let dense = Array::from_slice(&values, &shape).copy(stream)?;
        eval([&dense])?;
        Ok(dense)
    }

    fn linear_fallback(
        &self,
        input: &Array,
        transpose: bool,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let dense = self.dequantize(stream)?;
        if transpose {
            matmul(input, dense.transpose(stream)?, stream)
        } else {
            matmul(input, dense, stream)
        }
    }
}

/// Applies an expert-major native Q4_K matrix bank.
pub fn native_grouped_linear(
    input: &Array,
    weight: &NativeQuantizedTensor,
    group_ids: &Array,
    stream: &Stream,
) -> Result<Array, Exception> {
    if input.ndim() != 2
        || group_ids.ndim() != 1
        || input.dim(0) != group_ids.dim(0)
        || input.dim(1) != weight.columns
    {
        return Err(Exception::custom(format!(
            "native grouped linear shape mismatch: input {:?}, ids {:?}, weight {:?}",
            input.shape(),
            group_ids.shape(),
            weight.shape()
        )));
    }
    if native_execution_backend(stream)? == NativeExecutionBackend::Metal {
        return match weight.format() {
            NativeQuantizationFormat::GgufQ4K => {
                q4k_grouped_metal(input, weight, group_ids, stream)
            }
            NativeQuantizationFormat::GgufQ5_1 => {
                q5_1_grouped_metal(input, weight, group_ids, stream)
            }
        };
    }
    let dense = weight.dequantize(stream)?;
    grouped_matmul(
        input,
        &dense.swap_axes(-1, -2, stream)?,
        group_ids,
        true,
        stream,
    )
}

/// Runs direct selected-expert fused gate/up projection and approximate GELU.
///
/// `fused_gate_up` must contain `2 * intermediate` physical rows per expert,
/// with gate rows followed by up rows.
pub fn native_selected_gate_up(
    hidden: &Array,
    fused_gate_up: &NativeQuantizedTensor,
    expert_ids: &Array,
    intermediate: i32,
    stream: &Stream,
) -> Result<Array, Exception> {
    if hidden.ndim() != 2
        || hidden.dim(0) != 1
        || hidden.dim(1) != fused_gate_up.columns
        || expert_ids.ndim() != 1
        || fused_gate_up.row_start != 0
        || fused_gate_up.rows != 2 * intermediate
    {
        return Err(Exception::custom(format!(
            "native selected gate/up shape mismatch: hidden {:?}, ids {:?}, weight {:?}, intermediate {intermediate}",
            hidden.shape(),
            expert_ids.shape(),
            fused_gate_up.shape()
        )));
    }
    if native_execution_backend(stream)? == NativeExecutionBackend::Metal {
        return q4k_gate_up_metal(hidden, fused_gate_up, expert_ids, intermediate, stream);
    }
    let top_k = expert_ids.dim(0);
    let repeated = Array::repeat_axis::<f32>(hidden.clone(), top_k, 0, stream)?;
    let gate = native_grouped_linear(
        &repeated,
        &fused_gate_up.row_view(0, intermediate)?,
        expert_ids,
        stream,
    )?;
    let up = native_grouped_linear(
        &repeated,
        &fused_gate_up.row_view(intermediate, intermediate)?,
        expert_ids,
        stream,
    )?;
    crate::nn::gelu_approximate(gate, stream)?.multiply(up, stream)
}

/// Runs selected-expert native down projection, route weighting, and reduction.
pub fn native_selected_down_reduce(
    activated: &Array,
    down: &NativeQuantizedTensor,
    expert_ids: &Array,
    route_weights: &Array,
    stream: &Stream,
) -> Result<Array, Exception> {
    if activated.ndim() != 2
        || expert_ids.ndim() != 1
        || route_weights.size() as i32 != expert_ids.dim(0)
        || activated.dim(0) != expert_ids.dim(0)
        || activated.dim(1) != down.columns
    {
        return Err(Exception::custom(format!(
            "native selected down shape mismatch: activated {:?}, ids {:?}, route weights {:?}, down {:?}",
            activated.shape(),
            expert_ids.shape(),
            route_weights.shape(),
            down.shape()
        )));
    }
    if native_execution_backend(stream)? == NativeExecutionBackend::Metal {
        return match down.format() {
            NativeQuantizationFormat::GgufQ4K => {
                q4k_down_reduce_metal(activated, down, expert_ids, route_weights, stream)
            }
            NativeQuantizationFormat::GgufQ5_1 => {
                q5_1_down_reduce_metal(activated, down, expert_ids, route_weights, stream)
            }
        };
    }
    let projected = native_grouped_linear(activated, down, expert_ids, stream)?;
    sum_axis(
        projected.multiply(route_weights.reshape(&[-1, 1], stream)?, stream)?,
        0,
        true,
        stream,
    )
}

fn is_gpu(stream: &Stream) -> Result<bool, Exception> {
    Ok(stream.get_device()?.get_type()? == DeviceType::Gpu)
}

/// Selects a native execution backend without consulting model architecture.
pub fn native_execution_backend(stream: &Stream) -> Result<NativeExecutionBackend, Exception> {
    #[cfg(feature = "cuda")]
    {
        let _ = stream;
        Ok(NativeExecutionBackend::GenericFallback)
    }
    #[cfg(not(feature = "cuda"))]
    {
        Ok(if is_gpu(stream)? {
            NativeExecutionBackend::Metal
        } else {
            NativeExecutionBackend::GenericFallback
        })
    }
}

fn q4k_config(
    rows: i32,
    view: &NativeQuantizedTensor,
    output_rows: i32,
    output_cols: i32,
    dtype: Dtype,
) -> MetalKernelConfig {
    let out_grid = ((output_cols + OUT_TILE - 1) / OUT_TILE) * OUT_TILE;
    MetalKernelConfig::new()
        .with_template_arg_dtype("T", dtype)
        .with_template_arg_int("IN_DIM", view.columns)
        .with_template_arg_int("OUT_DIM", output_cols)
        .with_template_arg_int("OUT_GRID", out_grid)
        .with_template_arg_int("BLOCKS", view.columns / Q4_K_BLOCK_VALUES)
        .with_template_arg_int("PHYSICAL_ROWS", view.physical_rows)
        .with_template_arg_int("ROW_START", view.row_start)
        .with_template_arg_int("REDUCTION_TILE", REDUCTION_TILE)
        .with_template_arg_int("OUT_TILE", OUT_TILE)
        .with_grid([REDUCTION_TILE, rows * out_grid, 1])
        .with_thread_group([REDUCTION_TILE, OUT_TILE, 1])
        .with_output_arg([output_rows, output_cols], dtype)
}

fn validate_activation_dtype(input: &Array) -> Result<Dtype, Exception> {
    let dtype = input.dtype();
    if !matches!(dtype, Dtype::Float16 | Dtype::Float32 | Dtype::Bfloat16) {
        return Err(Exception::custom(format!(
            "native Q4_K activation must be float16, bfloat16, or float32, got {dtype:?}"
        )));
    }
    Ok(dtype)
}

fn q4k_linear_metal(
    input: &Array,
    view: &NativeQuantizedTensor,
    stream: &Stream,
) -> Result<Array, Exception> {
    if input.dim(-1) != view.columns {
        return Err(Exception::custom(format!(
            "native Q4_K linear expected input dimension {}, got {:?}",
            view.columns,
            input.shape()
        )));
    }
    let dtype = validate_activation_dtype(input)?;
    let outer = input.size() as i32 / view.columns;
    let flat = input.reshape(&[outer, view.columns], stream)?;
    let config = q4k_config(outer, view, outer, view.rows, dtype);
    let output = Q4K_LINEAR_KERNEL.with(|cell| -> Result<_, Exception> {
        if cell.borrow().is_none() {
            *cell.borrow_mut() = Some(q4k_linear_kernel()?);
        }
        cell.borrow()
            .as_ref()
            .expect("Q4_K linear kernel initialized")
            .apply_one_device([&flat, view.storage.bytes()], &config, stream)
    })?;
    let mut shape = input.shape()[..input.ndim() as usize - 1].to_vec();
    shape.push(view.rows);
    output.reshape(&shape, stream)
}

fn q4k_grouped_metal(
    input: &Array,
    view: &NativeQuantizedTensor,
    group_ids: &Array,
    stream: &Stream,
) -> Result<Array, Exception> {
    let dtype = validate_activation_dtype(input)?;
    let routes = input.dim(0);
    let config = q4k_config(routes, view, routes, view.rows, dtype)
        .with_template_arg_int("MATRIX_COUNT", view.matrix_count);
    Q4K_GROUPED_KERNEL.with(|cell| -> Result<_, Exception> {
        if cell.borrow().is_none() {
            *cell.borrow_mut() = Some(q4k_grouped_kernel()?);
        }
        cell.borrow()
            .as_ref()
            .expect("Q4_K grouped kernel initialized")
            .apply_one_device([input, view.storage.bytes(), group_ids], &config, stream)
    })
}

fn q5_1_config(
    logical_rows: i32,
    view: &NativeQuantizedTensor,
    output_rows: i32,
    output_cols: i32,
    dtype: Dtype,
) -> MetalKernelConfig {
    let out_grid = ((output_cols + OUT_TILE - 1) / OUT_TILE) * OUT_TILE;
    MetalKernelConfig::new()
        .with_template_arg_dtype("T", dtype)
        .with_template_arg_int("IN_DIM", view.columns)
        .with_template_arg_int("OUT_DIM", output_cols)
        .with_template_arg_int("OUT_GRID", out_grid)
        .with_template_arg_int("BLOCKS", view.columns / Q5_1_BLOCK_VALUES)
        .with_template_arg_int("PHYSICAL_ROWS", view.physical_rows)
        .with_template_arg_int("ROW_START", view.row_start)
        .with_template_arg_int("REDUCTION_TILE", REDUCTION_TILE)
        .with_template_arg_int("OUT_TILE", OUT_TILE)
        .with_grid([REDUCTION_TILE, logical_rows * out_grid, 1])
        .with_thread_group([REDUCTION_TILE, OUT_TILE, 1])
        .with_output_arg([output_rows, output_cols], dtype)
}

fn q5_1_grouped_metal(
    input: &Array,
    view: &NativeQuantizedTensor,
    group_ids: &Array,
    stream: &Stream,
) -> Result<Array, Exception> {
    let dtype = validate_activation_dtype(input)?;
    let routes = input.dim(0);
    let config = q5_1_config(routes, view, routes, view.rows, dtype)
        .with_template_arg_int("MATRIX_COUNT", view.matrix_count);
    Q5_1_GROUPED_KERNEL.with(|cell| -> Result<_, Exception> {
        if cell.borrow().is_none() {
            *cell.borrow_mut() = Some(q5_1_grouped_kernel()?);
        }
        cell.borrow()
            .as_ref()
            .expect("Q5_1 grouped kernel initialized")
            .apply_one_device([input, view.storage.bytes(), group_ids], &config, stream)
    })
}

fn q4k_embedding_metal(
    indices: &Array,
    view: &NativeQuantizedTensor,
    stream: &Stream,
) -> Result<Array, Exception> {
    let count = indices.size() as i32;
    let config = MetalKernelConfig::new()
        .with_template_arg_int("IN_DIM", view.columns)
        .with_template_arg_int("ROWS", view.rows)
        .with_template_arg_int("BLOCKS", view.columns / Q4_K_BLOCK_VALUES)
        .with_template_arg_int("PHYSICAL_ROWS", view.physical_rows)
        .with_template_arg_int("ROW_START", view.row_start)
        .with_grid([count * view.columns, 1, 1])
        .with_thread_group([256, 1, 1])
        .with_output_arg([count, view.columns], Dtype::Float32);
    let output = Q4K_EMBEDDING_KERNEL.with(|cell| -> Result<_, Exception> {
        if cell.borrow().is_none() {
            *cell.borrow_mut() = Some(q4k_embedding_kernel()?);
        }
        cell.borrow()
            .as_ref()
            .expect("Q4_K embedding kernel initialized")
            .apply_one_device([view.storage.bytes(), indices], &config, stream)
    })?;
    let mut shape = indices.shape().to_vec();
    shape.push(view.columns);
    output.reshape(&shape, stream)
}

fn q4k_gate_up_metal(
    hidden: &Array,
    view: &NativeQuantizedTensor,
    expert_ids: &Array,
    intermediate: i32,
    stream: &Stream,
) -> Result<Array, Exception> {
    let dtype = validate_activation_dtype(hidden)?;
    let top_k = expert_ids.dim(0);
    let config = q4k_config(top_k, view, top_k, intermediate, dtype)
        .with_template_arg_int("INTERMEDIATE", intermediate)
        .with_template_arg_int("MATRIX_COUNT", view.matrix_count);
    Q4K_GATE_UP_KERNEL.with(|cell| -> Result<_, Exception> {
        if cell.borrow().is_none() {
            *cell.borrow_mut() = Some(q4k_gate_up_kernel()?);
        }
        cell.borrow()
            .as_ref()
            .expect("Q4_K gate/up kernel initialized")
            .apply_one_device([hidden, view.storage.bytes(), expert_ids], &config, stream)
    })
}

fn q4k_down_reduce_metal(
    activated: &Array,
    view: &NativeQuantizedTensor,
    expert_ids: &Array,
    route_weights: &Array,
    stream: &Stream,
) -> Result<Array, Exception> {
    let dtype = validate_activation_dtype(activated)?;
    let top_k = expert_ids.dim(0);
    let config = q4k_config(1, view, 1, view.rows, dtype)
        .with_template_arg_int("TOP_K", top_k)
        .with_template_arg_int("MATRIX_COUNT", view.matrix_count);
    Q4K_DOWN_REDUCE_KERNEL.with(|cell| -> Result<_, Exception> {
        if cell.borrow().is_none() {
            *cell.borrow_mut() = Some(q4k_down_reduce_kernel()?);
        }
        cell.borrow()
            .as_ref()
            .expect("Q4_K down/reduce kernel initialized")
            .apply_one_device(
                [activated, view.storage.bytes(), expert_ids, route_weights],
                &config,
                stream,
            )
    })
}

fn q5_1_down_reduce_metal(
    activated: &Array,
    view: &NativeQuantizedTensor,
    expert_ids: &Array,
    route_weights: &Array,
    stream: &Stream,
) -> Result<Array, Exception> {
    let dtype = validate_activation_dtype(activated)?;
    let top_k = expert_ids.dim(0);
    let config = q5_1_config(1, view, 1, view.rows, dtype)
        .with_template_arg_int("TOP_K", top_k)
        .with_template_arg_int("MATRIX_COUNT", view.matrix_count);
    Q5_1_DOWN_REDUCE_KERNEL.with(|cell| -> Result<_, Exception> {
        if cell.borrow().is_none() {
            *cell.borrow_mut() = Some(q5_1_down_reduce_kernel()?);
        }
        cell.borrow()
            .as_ref()
            .expect("Q5_1 down/reduce kernel initialized")
            .apply_one_device(
                [activated, view.storage.bytes(), expert_ids, route_weights],
                &config,
                stream,
            )
    })
}

fn q4k_linear_kernel() -> Result<MetalKernel, Exception> {
    MetalKernel::new(
        "native_q4k_linear",
        ["input", "weight"],
        ["out"],
        [
            Q4K_TILED_PROLOGUE,
            "uint physical_row = ROW_START + out_col;",
            "uint matrix_base = physical_row * BLOCKS * 144;",
            Q4K_ACCUMULATE,
            Q4K_TILED_EPILOGUE,
        ]
        .concat(),
        Q4K_METAL_HEADER,
        true,
        false,
    )
}

fn q4k_grouped_kernel() -> Result<MetalKernel, Exception> {
    MetalKernel::new(
        "native_q4k_grouped",
        ["input", "weight", "group_ids"],
        ["out"],
        [
            Q4K_TILED_PROLOGUE,
            "uint expert = uint(group_ids[row]);",
            "uint physical_row = expert * PHYSICAL_ROWS + ROW_START + out_col;",
            "uint matrix_base = physical_row * BLOCKS * 144;",
            Q4K_ACCUMULATE,
            Q4K_TILED_EPILOGUE,
        ]
        .concat(),
        Q4K_METAL_HEADER,
        true,
        false,
    )
}

fn q5_1_grouped_kernel() -> Result<MetalKernel, Exception> {
    MetalKernel::new(
        "native_q5_1_grouped",
        ["input", "weight", "group_ids"],
        ["out"],
        concat!(
            "uint lane = thread_position_in_grid.x;",
            "uint row = thread_position_in_grid.y / OUT_GRID;",
            "uint out_col = thread_position_in_grid.y % OUT_GRID;",
            "float acc = 0.0f;",
            "if (out_col < OUT_DIM) {",
            " uint expert = uint(group_ids[row]);",
            " uint physical_row = expert * PHYSICAL_ROWS + ROW_START + out_col;",
            " uint matrix_base = physical_row * BLOCKS * 24;",
            " for (uint block = lane; block < BLOCKS; block += REDUCTION_TILE) {",
            "  uint base = matrix_base + block * 24;",
            "  uint input_block = row * IN_DIM + block * 32;",
            "  for (uint i = 0; i < 32; ++i) {",
            "   acc += float(input[input_block + i]) * q5_1_value(weight, base, i);",
            "  }",
            " }",
            "}",
            "float total = simd_sum(acc);",
            "if (lane == 0 && out_col < OUT_DIM) out[row * OUT_DIM + out_col] = T(total);"
        ),
        Q5_1_METAL_HEADER,
        true,
        false,
    )
}

fn q4k_embedding_kernel() -> Result<MetalKernel, Exception> {
    MetalKernel::new(
        "native_q4k_embedding",
        ["weight", "indices"],
        ["out"],
        concat!(
            "uint elem = thread_position_in_grid.x;",
            "uint col = elem % IN_DIM;",
            "uint output_row = elem / IN_DIM;",
            "uint row = uint(indices[output_row]);",
            "if (row >= ROWS) { out[elem] = 0.0f; return; }",
            "uint physical_row = ROW_START + row;",
            "uint block = col / 256;",
            "uint within = col % 256;",
            "uint g = within / 32;",
            "uint i = within % 32;",
            "uint base = (physical_row * BLOCKS + block) * 144;",
            "out[elem] = q4k_value(weight, base, g, i);"
        ),
        Q4K_METAL_HEADER,
        true,
        false,
    )
}

fn q4k_gate_up_kernel() -> Result<MetalKernel, Exception> {
    MetalKernel::new(
        "native_q4k_selected_gate_up",
        ["input", "weight", "expert_ids"],
        ["out"],
        concat!(
            "uint lane = thread_position_in_grid.x;",
            "uint route = thread_position_in_grid.y / OUT_GRID;",
            "uint out_col = thread_position_in_grid.y % OUT_GRID;",
            "uint local_col = out_col % OUT_TILE;",
            "float gate_acc = 0.0f;",
            "float up_acc = 0.0f;",
            "if (out_col < INTERMEDIATE) {",
            " uint expert = uint(expert_ids[route]);",
            " uint gate_row = expert * PHYSICAL_ROWS + out_col;",
            " uint up_row = gate_row + INTERMEDIATE;",
            " uint gate_base = gate_row * BLOCKS * 144;",
            " uint up_base = up_row * BLOCKS * 144;",
            " for (uint block = 0; block < BLOCKS; ++block) {",
            "  uint gb = gate_base + block * 144;",
            "  uint ub = up_base + block * 144;",
            "  uint input_block = block * 256;",
            "  for (uint g = 0; g < 8; ++g) {",
            "   for (uint i = lane; i < 32; i += REDUCTION_TILE) {",
            "    float x = float(input[input_block + g * 32 + i]);",
            "    gate_acc += x * q4k_value(weight, gb, g, i);",
            "    up_acc += x * q4k_value(weight, ub, g, i);",
            "   }",
            "  }",
            " }",
            "}",
            "float gate = simd_sum(gate_acc);",
            "float up = simd_sum(up_acc);",
            "if (lane == 0 && out_col < INTERMEDIATE) {",
            " float c = 0.7978845608028654f;",
            " float activated = 0.5f * gate * (1.0f + metal::tanh(c * (gate + 0.044715f * gate * gate * gate)));",
            " out[route * INTERMEDIATE + out_col] = T(activated * up);",
            "}"
        ),
        Q4K_METAL_HEADER,
        true,
        false,
    )
}

fn q4k_down_reduce_kernel() -> Result<MetalKernel, Exception> {
    MetalKernel::new(
        "native_q4k_selected_down_reduce",
        ["input", "weight", "expert_ids", "route_weights"],
        ["out"],
        concat!(
            "uint lane = thread_position_in_grid.x;",
            "uint out_col = thread_position_in_grid.y % OUT_GRID;",
            "uint local_col = out_col % OUT_TILE;",
            "float acc = 0.0f;",
            "if (out_col < OUT_DIM) {",
            " for (uint route = 0; route < TOP_K; ++route) {",
            "  uint expert = uint(expert_ids[route]);",
            "  uint physical_row = expert * PHYSICAL_ROWS + ROW_START + out_col;",
            "  uint matrix_base = physical_row * BLOCKS * 144;",
            "  float route_acc = 0.0f;",
            "  for (uint block = 0; block < BLOCKS; ++block) {",
            "   uint base = matrix_base + block * 144;",
            "   uint input_block = route * IN_DIM + block * 256;",
            "   for (uint g = 0; g < 8; ++g) {",
            "    for (uint i = lane; i < 32; i += REDUCTION_TILE) {",
            "     route_acc += float(input[input_block + g * 32 + i]) * q4k_value(weight, base, g, i);",
            "    }",
            "   }",
            "  }",
            "  acc += route_acc * float(route_weights[route]);",
            " }",
            "}",
            "float total = simd_sum(acc);",
            "if (lane == 0 && out_col < OUT_DIM) {",
            " out[out_col] = T(total);",
            "}"
        ),
        Q4K_METAL_HEADER,
        true,
        false,
    )
}

fn q5_1_down_reduce_kernel() -> Result<MetalKernel, Exception> {
    MetalKernel::new(
        "native_q5_1_selected_down_reduce",
        ["input", "weight", "expert_ids", "route_weights"],
        ["out"],
        concat!(
            "uint lane = thread_position_in_grid.x;",
            "uint out_col = thread_position_in_grid.y % OUT_GRID;",
            "float acc = 0.0f;",
            "if (out_col < OUT_DIM) {",
            " for (uint route = 0; route < TOP_K; ++route) {",
            "  uint expert = uint(expert_ids[route]);",
            "  uint physical_row = expert * PHYSICAL_ROWS + ROW_START + out_col;",
            "  uint matrix_base = physical_row * BLOCKS * 24;",
            "  float route_acc = 0.0f;",
            "  for (uint block = lane; block < BLOCKS; block += REDUCTION_TILE) {",
            "   uint base = matrix_base + block * 24;",
            "   uint input_block = route * IN_DIM + block * 32;",
            "   for (uint i = 0; i < 32; ++i) {",
            "    route_acc += float(input[input_block + i]) * q5_1_value(weight, base, i);",
            "   }",
            "  }",
            "  acc += route_acc * float(route_weights[route]);",
            " }",
            "}",
            "float total = simd_sum(acc);",
            "if (lane == 0 && out_col < OUT_DIM) out[out_col] = T(total);"
        ),
        Q5_1_METAL_HEADER,
        true,
        false,
    )
}

const Q4K_TILED_PROLOGUE: &str = concat!(
    "uint lane = thread_position_in_grid.x;",
    "uint row = thread_position_in_grid.y / OUT_GRID;",
    "uint out_col = thread_position_in_grid.y % OUT_GRID;",
    "uint local_col = out_col % OUT_TILE;",
    "float acc = 0.0f;",
    "if (out_col < OUT_DIM) {"
);

const Q4K_ACCUMULATE: &str = concat!(
    " for (uint block = 0; block < BLOCKS; ++block) {",
    "  uint base = matrix_base + block * 144;",
    "  uint input_block = row * IN_DIM + block * 256;",
    "  for (uint g = 0; g < 8; ++g) {",
    "   for (uint i = lane; i < 32; i += REDUCTION_TILE) {",
    "    acc += float(input[input_block + g * 32 + i]) * q4k_value(weight, base, g, i);",
    "   }",
    "  }",
    " }",
    "}"
);

const Q4K_TILED_EPILOGUE: &str = concat!(
    "float total = simd_sum(acc);",
    "if (lane == 0 && out_col < OUT_DIM) {",
    " out[row * OUT_DIM + out_col] = T(total);",
    "}"
);

// Q4_K layout and scale unpacking follow llama.cpp's ggml-quants reference
// implementation (MIT) and MLX's GGUF converter (MIT).
const Q4K_METAL_HEADER: &str = concat!(
    "float q4k_value(const device uint8_t* weight, uint base, uint g, uint i) {",
    " uint d_bits = uint(weight[base]) | (uint(weight[base + 1]) << 8);",
    " uint dm_bits = uint(weight[base + 2]) | (uint(weight[base + 3]) << 8);",
    " float d = float(as_type<half>(ushort(d_bits)));",
    " float dm = float(as_type<half>(ushort(dm_bits)));",
    " uint sc;",
    " uint m;",
    " if (g < 4) {",
    "  sc = uint(weight[base + 4 + g]) & 63u;",
    "  m = uint(weight[base + 8 + g]) & 63u;",
    " } else {",
    "  sc = (uint(weight[base + 8 + g]) & 15u) | ((uint(weight[base + g]) >> 6) << 4);",
    "  m = (uint(weight[base + 8 + g]) >> 4) | ((uint(weight[base + 4 + g]) >> 6) << 4);",
    " }",
    " uint packed = uint(weight[base + 16 + (g / 2) * 32 + i]);",
    " uint q = (g & 1u) == 0u ? (packed & 15u) : (packed >> 4);",
    " return d * float(sc) * float(q) - dm * float(m);",
    "}\n"
);

// Q5_1 layout follows llama.cpp's ggml-quants reference implementation
// (MIT) and MLX's GGUF converter (MIT).
const Q5_1_METAL_HEADER: &str = concat!(
    "float q5_1_value(const device uint8_t* weight, uint base, uint i) {",
    " uint d_bits = uint(weight[base]) | (uint(weight[base + 1]) << 8);",
    " uint m_bits = uint(weight[base + 2]) | (uint(weight[base + 3]) << 8);",
    " float d = float(as_type<half>(ushort(d_bits)));",
    " float m = float(as_type<half>(ushort(m_bits)));",
    " uint qh = uint(weight[base + 4]) | (uint(weight[base + 5]) << 8) |",
    "           (uint(weight[base + 6]) << 16) | (uint(weight[base + 7]) << 24);",
    " uint packed = uint(weight[base + 8 + (i & 15u)]);",
    " uint low = i < 16 ? (packed & 15u) : (packed >> 4);",
    " uint q = low | (((qh >> i) & 1u) << 4);",
    " return d * float(q) + m;",
    "}\n"
);

fn decode_q4k_view(raw: &[u8], view: &NativeQuantizedTensor) -> Result<Vec<f32>, Exception> {
    let blocks = view.columns as usize / Q4_K_BLOCK_VALUES as usize;
    let matrix_stride = view.physical_rows as usize * blocks * Q4_K_BLOCK_BYTES as usize;
    let expected = view.matrix_count as usize * matrix_stride;
    if raw.len() != expected {
        return Err(Exception::custom(format!(
            "native Q4_K storage has {} bytes, expected {expected}",
            raw.len()
        )));
    }
    let mut output =
        Vec::with_capacity(view.matrix_count as usize * view.rows as usize * view.columns as usize);
    for matrix in 0..view.matrix_count as usize {
        for logical_row in 0..view.rows as usize {
            let physical_row = view.row_start as usize + logical_row;
            let row_base =
                matrix * matrix_stride + physical_row * blocks * Q4_K_BLOCK_BYTES as usize;
            for block in 0..blocks {
                decode_q4k_block(
                    &raw[row_base + block * 144..row_base + (block + 1) * 144],
                    &mut output,
                );
            }
        }
    }
    Ok(output)
}

fn decode_q4k_block(block: &[u8], output: &mut Vec<f32>) {
    let d = half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
    let dm = half::f16::from_bits(u16::from_le_bytes([block[2], block[3]])).to_f32();
    let scales = &block[4..16];
    let quants = &block[16..144];
    for group in 0..8 {
        let (scale, min) = if group < 4 {
            (scales[group] & 63, scales[group + 4] & 63)
        } else {
            (
                (scales[group + 4] & 15) | ((scales[group - 4] >> 6) << 4),
                (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4),
            )
        };
        for index in 0..32 {
            let packed = quants[(group / 2) * 32 + index];
            let quant = if group % 2 == 0 {
                packed & 15
            } else {
                packed >> 4
            };
            output.push(d * f32::from(scale) * f32::from(quant) - dm * f32::from(min));
        }
    }
}

fn decode_q5_1_view(raw: &[u8], view: &NativeQuantizedTensor) -> Result<Vec<f32>, Exception> {
    let blocks = view.columns as usize / Q5_1_BLOCK_VALUES as usize;
    let matrix_stride = view.physical_rows as usize * blocks * Q5_1_BLOCK_BYTES as usize;
    let expected = view.matrix_count as usize * matrix_stride;
    if raw.len() != expected {
        return Err(Exception::custom(format!(
            "native Q5_1 storage has {} bytes, expected {expected}",
            raw.len()
        )));
    }
    let mut output =
        Vec::with_capacity(view.matrix_count as usize * view.rows as usize * view.columns as usize);
    for matrix in 0..view.matrix_count as usize {
        for logical_row in 0..view.rows as usize {
            let physical_row = view.row_start as usize + logical_row;
            let row_base =
                matrix * matrix_stride + physical_row * blocks * Q5_1_BLOCK_BYTES as usize;
            for block in 0..blocks {
                decode_q5_1_block(
                    &raw[row_base + block * 24..row_base + (block + 1) * 24],
                    &mut output,
                );
            }
        }
    }
    Ok(output)
}

fn decode_q5_1_block(block: &[u8], output: &mut Vec<f32>) {
    let d = half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
    let min = half::f16::from_bits(u16::from_le_bytes([block[2], block[3]])).to_f32();
    let high = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
    for index in 0..32 {
        let packed = block[8 + index % 16];
        let low = if index < 16 { packed & 15 } else { packed >> 4 };
        let quant = low | ((((high >> index) & 1) as u8) << 4);
        output.push(d * f32::from(quant) + min);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::indexing::TryIndexOp;

    fn sample_block() -> Vec<u8> {
        let mut block = vec![0u8; 144];
        block[0..2].copy_from_slice(&half::f16::from_f32(0.125).to_bits().to_le_bytes());
        block[2..4].copy_from_slice(&half::f16::from_f32(0.25).to_bits().to_le_bytes());
        for (index, value) in block[4..16].iter_mut().enumerate() {
            *value = (index as u8).wrapping_mul(17).wrapping_add(3);
        }
        for (index, value) in block[16..].iter_mut().enumerate() {
            *value = (index as u8).wrapping_mul(29).wrapping_add(11);
        }
        block
    }

    fn sample_q5_1_block() -> Vec<u8> {
        let mut block = vec![0u8; 24];
        block[0..2].copy_from_slice(&half::f16::from_f32(0.125).to_bits().to_le_bytes());
        block[2..4].copy_from_slice(&half::f16::from_f32(-0.75).to_bits().to_le_bytes());
        block[4..8].copy_from_slice(&0xa5c3_781fu32.to_le_bytes());
        for (index, value) in block[8..].iter_mut().enumerate() {
            *value = (index as u8).wrapping_mul(23).wrapping_add(5);
        }
        block
    }

    #[test]
    fn q4k_views_share_storage_and_decode_segments() {
        let stream = crate::Stream::new_with_device(&crate::Device::new(DeviceType::Cpu, 0));
        let mut raw = Vec::new();
        for _ in 0..8 {
            raw.extend(sample_block());
        }
        let fused = NativeQuantizedTensor::from_q4k_bytes(&raw, &[2, 4, 256], &stream).unwrap();
        let gate = fused.row_view(0, 2).unwrap();
        let up = fused.row_view(2, 2).unwrap();
        assert!(Arc::ptr_eq(gate.storage(), up.storage()));
        assert_eq!(gate.shape(), vec![2, 2, 256]);
        assert_eq!(up.row_start(), 2);
        drop(fused);
        // Logical views retain the backing allocation after the physical
        // parent object is gone.
        assert_eq!(gate.dequantize(&stream).unwrap().shape(), &[2, 2, 256]);
    }

    #[test]
    fn q4k_decode_matches_affine_reference() {
        let stream = crate::Stream::new_with_device(&crate::Device::new(DeviceType::Cpu, 0));
        let raw = sample_block();
        let native = NativeQuantizedTensor::from_q4k_bytes(&raw, &[1, 256], &stream).unwrap();
        let actual = native.dequantize(&stream).unwrap();
        let actual = actual.evaluated().unwrap();
        let actual = actual.as_slice::<f32>();

        let descriptor = safemlx_gguf::TensorDescriptor {
            name: "test.weight".into(),
            dimensions: vec![256, 1],
            ggml_type: safemlx_gguf::GgmlType::Q4K,
            relative_offset: 0,
            data_offset: 0,
            byte_len: 144,
        };
        let mut file = Vec::new();
        safemlx_gguf::Writer::default()
            .write(
                std::io::Cursor::new(&mut file),
                &std::collections::BTreeMap::new(),
                &[safemlx_gguf::TensorInput {
                    name: &descriptor.name,
                    dimensions: &descriptor.dimensions,
                    ggml_type: descriptor.ggml_type,
                    data: &raw,
                }],
            )
            .unwrap();
        let mut reader = safemlx_gguf::Reader::new(std::io::Cursor::new(file)).unwrap();
        let desc = reader.tensors()[0].clone();
        let safemlx_gguf::ConvertedTensor::Affine(reference) = reader.read_tensor(&desc).unwrap()
        else {
            panic!("expected affine reference")
        };
        let expected = reference.dequantize();
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() <= 1e-6, "{actual} != {expected}");
        }
    }

    #[test]
    fn q4k_cpu_linear_embedding_and_grouped_fallbacks() {
        let stream = crate::Stream::new_with_device(&crate::Device::new(DeviceType::Cpu, 0));
        let mut raw = Vec::new();
        raw.extend(sample_block());
        raw.extend(sample_block());
        let matrix = NativeQuantizedTensor::from_q4k_bytes(&raw, &[2, 256], &stream).unwrap();
        let input = Array::from_slice(&vec![0.5f32; 512], &[2, 256]);
        let output = matrix.linear(&input, true, &stream).unwrap();
        assert_eq!(output.shape(), &[2, 2]);
        let untransposed_input = Array::from_slice(&[0.25f32, -0.5], &[1, 2]);
        let untransposed = matrix.linear(&untransposed_input, false, &stream).unwrap();
        assert_eq!(untransposed.shape(), &[1, 256]);
        let ids = Array::from_slice(&[1i32, 0], &[2]);
        let embedded = matrix.embedding(&ids, &stream).unwrap();
        assert_eq!(embedded.shape(), &[2, 256]);

        let mut expert_raw = Vec::new();
        for _ in 0..4 {
            expert_raw.extend(sample_block());
        }
        let experts =
            NativeQuantizedTensor::from_q4k_bytes(&expert_raw, &[2, 2, 256], &stream).unwrap();
        let group_ids = Array::from_slice(&[1i32, 0], &[2]);
        let grouped = native_grouped_linear(&input, &experts, &group_ids, &stream).unwrap();
        assert_eq!(grouped.shape(), &[2, 2]);
    }

    #[test]
    fn q5_1_decode_matches_affine_reference() {
        let stream = crate::Stream::new_with_device(&crate::Device::new(DeviceType::Cpu, 0));
        let raw = sample_q5_1_block();
        let native = NativeQuantizedTensor::from_q5_1_bytes(&raw, &[1, 32], &stream).unwrap();
        let actual = native.dequantize(&stream).unwrap();
        let actual = actual.evaluated().unwrap();

        let mut file = Vec::new();
        safemlx_gguf::Writer::default()
            .write(
                std::io::Cursor::new(&mut file),
                &std::collections::BTreeMap::new(),
                &[safemlx_gguf::TensorInput {
                    name: "test.weight",
                    dimensions: &[32, 1],
                    ggml_type: safemlx_gguf::GgmlType::Q5_1,
                    data: &raw,
                }],
            )
            .unwrap();
        let mut reader = safemlx_gguf::Reader::new(std::io::Cursor::new(file)).unwrap();
        let descriptor = reader.tensors()[0].clone();
        let safemlx_gguf::ConvertedTensor::Affine(reference) =
            reader.read_tensor(&descriptor).unwrap()
        else {
            panic!("expected affine reference")
        };
        for (actual, expected) in actual.as_slice::<f32>().iter().zip(reference.dequantize()) {
            assert!((actual - expected).abs() <= 1e-6, "{actual} != {expected}");
        }
    }

    fn repeated_blocks(count: usize) -> Vec<u8> {
        let block = sample_block();
        let mut raw = Vec::with_capacity(count * block.len());
        for index in 0..count {
            let mut block = block.clone();
            block[16] = block[16].wrapping_add(index as u8);
            raw.extend(block);
        }
        raw
    }

    #[test]
    #[ignore = "requires an accessible Metal device"]
    fn q4k_metal_linear_prefill_embedding_and_partial_tiles_match_float() {
        let stream = crate::Stream::new_with_device(&crate::Device::new(DeviceType::Gpu, 0));
        let raw = repeated_blocks(5 * 2);
        let native = NativeQuantizedTensor::from_q4k_bytes(&raw, &[5, 512], &stream).unwrap();
        let input = Array::from_slice(
            &(0..3 * 512)
                .map(|index| (index as f32 % 37.0 - 18.0) / 19.0)
                .collect::<Vec<_>>(),
            &[3, 512],
        )
        .copy(&stream)
        .unwrap();
        let actual = native.linear(&input, true, &stream).unwrap();
        let dense = native.dequantize(&stream).unwrap();
        let expected = matmul(&input, dense.transpose(&stream).unwrap(), &stream).unwrap();
        assert!(actual
            .all_close(&expected, Some(2e-3), Some(2e-3), None, &stream)
            .unwrap()
            .item::<bool>(&stream));

        let ids = Array::from_slice(&[4i32, 1, 4], &[3])
            .copy(&stream)
            .unwrap();
        let actual = native.embedding(&ids, &stream).unwrap();
        let expected = dense.try_index_device(&ids, &stream).unwrap();
        assert!(actual
            .all_close(&expected, Some(1e-6), Some(1e-6), None, &stream)
            .unwrap()
            .item::<bool>(&stream));
    }

    #[test]
    #[ignore = "requires an accessible Metal device"]
    fn q4k_metal_selected_experts_match_dequantized_reference() {
        let stream = crate::Stream::new_with_device(&crate::Device::new(DeviceType::Gpu, 0));
        let experts = 3;
        let intermediate = 256;
        let hidden = 256;
        let output = 5;
        let gate_up_raw = repeated_blocks((experts * 2 * intermediate) as usize);
        let down_raw = repeated_blocks((experts * output) as usize);
        let gate_up = NativeQuantizedTensor::from_q4k_bytes(
            &gate_up_raw,
            &[experts, 2 * intermediate, hidden],
            &stream,
        )
        .unwrap();
        let down = NativeQuantizedTensor::from_q4k_bytes(
            &down_raw,
            &[experts, output, intermediate],
            &stream,
        )
        .unwrap();
        let hidden_state = Array::from_slice(
            &(0..hidden)
                .map(|index| (index as f32 % 23.0 - 11.0) / 12.0)
                .collect::<Vec<_>>(),
            &[1, hidden],
        )
        .copy(&stream)
        .unwrap();
        let ids = Array::from_slice(&[2i32, 0, 2], &[3])
            .copy(&stream)
            .unwrap();
        let weights = Array::from_slice(&[0.2f32, 0.3, 0.5], &[3])
            .copy(&stream)
            .unwrap();

        let activated =
            native_selected_gate_up(&hidden_state, &gate_up, &ids, intermediate, &stream).unwrap();
        let actual =
            native_selected_down_reduce(&activated, &down, &ids, &weights, &stream).unwrap();

        let selected = gate_up
            .dequantize(&stream)
            .unwrap()
            .try_index_device(&ids, &stream)
            .unwrap();
        let gate_weight = selected
            .try_index_device((.., ..intermediate, ..), &stream)
            .unwrap();
        let up_weight = selected
            .try_index_device((.., intermediate.., ..), &stream)
            .unwrap();
        let repeated = Array::repeat_axis::<f32>(hidden_state, ids.dim(0), 0, &stream).unwrap();
        let batched_input = repeated.reshape(&[-1, 1, hidden], &stream).unwrap();
        let gate = matmul(
            &batched_input,
            gate_weight.swap_axes(-1, -2, &stream).unwrap(),
            &stream,
        )
        .unwrap()
        .reshape(&[-1, intermediate], &stream)
        .unwrap();
        let up = matmul(
            &batched_input,
            up_weight.swap_axes(-1, -2, &stream).unwrap(),
            &stream,
        )
        .unwrap()
        .reshape(&[-1, intermediate], &stream)
        .unwrap();
        let reference_activated = crate::nn::gelu_approximate(gate, &stream)
            .unwrap()
            .multiply(up, &stream)
            .unwrap();
        let selected_down = down
            .dequantize(&stream)
            .unwrap()
            .try_index_device(&ids, &stream)
            .unwrap();
        let projected = matmul(
            &reference_activated
                .reshape(&[-1, 1, intermediate], &stream)
                .unwrap(),
            selected_down.swap_axes(-1, -2, &stream).unwrap(),
            &stream,
        )
        .unwrap()
        .reshape(&[-1, output], &stream)
        .unwrap();
        let expected = sum_axis(
            projected
                .multiply(weights.reshape(&[-1, 1], &stream).unwrap(), &stream)
                .unwrap(),
            0,
            true,
            &stream,
        )
        .unwrap();
        assert!(actual
            .all_close(&expected, Some(5e-3), Some(5e-3), None, &stream)
            .unwrap()
            .item::<bool>(&stream));
    }

    #[test]
    #[ignore = "requires an accessible Metal device"]
    fn mixed_q4k_gate_up_q5_1_down_matches_dequantized_reference() {
        let stream = crate::Stream::new_with_device(&crate::Device::new(DeviceType::Gpu, 0));
        let experts = 3;
        let intermediate = 64;
        let hidden = 256;
        let output = 5;
        let gate_up_raw = repeated_blocks((experts * 2 * intermediate) as usize);
        let mut down_raw = Vec::new();
        for index in 0..(experts * output * intermediate / 32) {
            let mut block = sample_q5_1_block();
            block[8] = block[8].wrapping_add(index as u8);
            down_raw.extend(block);
        }
        let gate_up = NativeQuantizedTensor::from_q4k_bytes(
            &gate_up_raw,
            &[experts, 2 * intermediate, hidden],
            &stream,
        )
        .unwrap();
        let down = NativeQuantizedTensor::from_q5_1_bytes(
            &down_raw,
            &[experts, output, intermediate],
            &stream,
        )
        .unwrap();
        let hidden_state = Array::from_slice(&vec![0.0001f32; hidden as usize], &[1, hidden])
            .copy(&stream)
            .unwrap();
        let ids = Array::from_slice(&[2i32, 0, 2], &[3])
            .copy(&stream)
            .unwrap();
        let weights = Array::from_slice(&[0.2f32, 0.3, 0.5], &[3])
            .copy(&stream)
            .unwrap();
        let activated =
            native_selected_gate_up(&hidden_state, &gate_up, &ids, intermediate, &stream).unwrap();
        let actual =
            native_selected_down_reduce(&activated, &down, &ids, &weights, &stream).unwrap();
        let selected_down = down
            .dequantize(&stream)
            .unwrap()
            .try_index_device(&ids, &stream)
            .unwrap();
        let projected = matmul(
            &activated.reshape(&[-1, 1, intermediate], &stream).unwrap(),
            selected_down.swap_axes(-1, -2, &stream).unwrap(),
            &stream,
        )
        .unwrap()
        .reshape(&[-1, output], &stream)
        .unwrap();
        let expected = sum_axis(
            projected
                .multiply(weights.reshape(&[-1, 1], &stream).unwrap(), &stream)
                .unwrap(),
            0,
            true,
            &stream,
        )
        .unwrap();
        assert!(actual
            .all_close(&expected, Some(5e-3), Some(5e-3), None, &stream)
            .unwrap()
            .item::<bool>(&stream));
    }
}
