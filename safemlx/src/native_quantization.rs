//! Checkpoint-native quantized storage and device execution.
//!
//! Native tensors keep their physical checkpoint blocks intact and describe
//! logical matrices as zero-copy row views. Backends select kernels from the
//! format, operation, shape, and device; callers remain independent of the
//! originating model architecture. Metal linear kernels reuse each decoded
//! block across a small activation-row tile, and routed-expert kernels fuse
//! gate/up activation and down-projection reduction where the representation
//! permits it. CPU execution decodes one packed weight row at a time into
//! bounded scratch space; it never materializes the complete dense matrix or a
//! second persistent affine copy.
//!
//! The bundled MLX C API exposes managed host buffers, but its public contract
//! still describes their input as copied and it exposes no API that wraps an
//! mmap region as a guaranteed Metal shared buffer. `Array::from_slice` also
//! documents and performs a copy. Native loading therefore makes one
//! persistent MLX-owned raw-byte copy today. [`NativeStorageKind`] is the seam
//! for a future mmap/external-buffer owner once the C API can guarantee buffer
//! identity and device accessibility; logical views already share ownership
//! through `Arc` and will not need to change.

use std::{cell::RefCell, collections::HashMap, fmt::Write, sync::Arc};

use crate::{
    error::Exception,
    fast::{MetalKernel, MetalKernelConfig},
    ops::{matmul, stack_axis, sum_axis},
    transforms::eval,
    Array, DeviceType, Dtype, Stream,
};
use safemlx_gguf::{Endian as GgufEndian, GgmlType};

const Q4_K_BLOCK_VALUES: i32 = 256;
const Q4_K_BLOCK_BYTES: i32 = 144;
const Q5_1_BLOCK_VALUES: i32 = 32;
const Q5_1_BLOCK_BYTES: i32 = 24;
const Q8_0_BLOCK_VALUES: i32 = 32;
const Q8_0_BLOCK_BYTES: i32 = 34;
const OUT_TILE: i32 = 4;
const REDUCTION_TILE: i32 = 32;
const NATIVE_BATCH_TILE: i32 = 8;
const Q8_LARGE_OUT_TILE: i32 = 8;
const Q8_LARGE_OUTPUT_ROWS: i32 = 65_536;

thread_local! {
    static Q4K_LINEAR_KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static Q4K_BATCH_KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static Q4K_GROUPED_KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static Q4K_EMBEDDING_KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static Q4K_GATE_UP_KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static Q4K_DOWN_REDUCE_KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static Q5_1_LINEAR_KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static Q5_1_BATCH_KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static Q5_1_GROUPED_KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static Q5_1_EMBEDDING_KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static Q5_1_DOWN_REDUCE_KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static Q8_0_LINEAR_KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static Q8_0_BATCH_KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static Q8_0_GROUPED_KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static Q8_0_EMBEDDING_KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static Q8_0_DOWN_REDUCE_KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static IQ_LINEAR_KERNEL: RefCell<HashMap<(NativeQuantizationFormat, bool), MetalKernel>> =
        RefCell::new(HashMap::new());
    static IQ_BATCH_KERNEL: RefCell<HashMap<(NativeQuantizationFormat, bool), MetalKernel>> =
        RefCell::new(HashMap::new());
    static IQ_GROUPED_KERNEL: RefCell<HashMap<(NativeQuantizationFormat, bool), MetalKernel>> =
        RefCell::new(HashMap::new());
    static IQ_EMBEDDING_KERNEL: RefCell<HashMap<(NativeQuantizationFormat, bool), MetalKernel>> =
        RefCell::new(HashMap::new());
    static IQ_GATE_UP_KERNEL: RefCell<HashMap<(NativeQuantizationFormat, bool, i32), MetalKernel>> =
        RefCell::new(HashMap::new());
    static IQ_DOWN_REDUCE_KERNEL: RefCell<HashMap<(NativeQuantizationFormat, bool), MetalKernel>> =
        RefCell::new(HashMap::new());
}

/// Physical quantization encoding retained from a checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NativeQuantizationFormat {
    /// GGUF/GGML Q4_K blocks: 256 weights in 144 bytes.
    GgufQ4K,
    /// GGUF/GGML Q5_1 blocks: 32 weights in 24 bytes.
    GgufQ5_1,
    /// GGUF/GGML Q8_0 blocks: 32 signed weights and one FP16 scale in 34 bytes.
    GgufQ8_0,
    /// GGUF/GGML IQ2_XXS codebook blocks.
    GgufIQ2XXS,
    /// GGUF/GGML IQ2_XS codebook blocks.
    GgufIQ2XS,
    /// GGUF/GGML IQ3_XXS codebook blocks.
    GgufIQ3XXS,
    /// GGUF/GGML IQ1_S codebook blocks.
    GgufIQ1S,
    /// GGUF/GGML IQ4_NL nonlinear blocks.
    GgufIQ4NL,
    /// GGUF/GGML IQ3_S codebook blocks.
    GgufIQ3S,
    /// GGUF/GGML IQ2_S codebook blocks.
    GgufIQ2S,
    /// GGUF/GGML IQ4_XS nonlinear blocks.
    GgufIQ4XS,
    /// GGUF/GGML IQ1_M codebook blocks.
    GgufIQ1M,
}

impl NativeQuantizationFormat {
    /// Maps a canonical GGML IQ type to native execution metadata.
    pub fn from_ggml_type(ty: GgmlType) -> Option<Self> {
        Some(match ty {
            GgmlType::IQ2XXS => Self::GgufIQ2XXS,
            GgmlType::IQ2XS => Self::GgufIQ2XS,
            GgmlType::IQ3XXS => Self::GgufIQ3XXS,
            GgmlType::IQ1S => Self::GgufIQ1S,
            GgmlType::IQ4NL => Self::GgufIQ4NL,
            GgmlType::IQ3S => Self::GgufIQ3S,
            GgmlType::IQ2S => Self::GgufIQ2S,
            GgmlType::IQ4XS => Self::GgufIQ4XS,
            GgmlType::IQ1M => Self::GgufIQ1M,
            _ => return None,
        })
    }

    /// Returns the GGML type for an IQ format.
    pub fn ggml_type(self) -> Option<GgmlType> {
        Some(match self {
            Self::GgufIQ2XXS => GgmlType::IQ2XXS,
            Self::GgufIQ2XS => GgmlType::IQ2XS,
            Self::GgufIQ3XXS => GgmlType::IQ3XXS,
            Self::GgufIQ1S => GgmlType::IQ1S,
            Self::GgufIQ4NL => GgmlType::IQ4NL,
            Self::GgufIQ3S => GgmlType::IQ3S,
            Self::GgufIQ2S => GgmlType::IQ2S,
            Self::GgufIQ4XS => GgmlType::IQ4XS,
            Self::GgufIQ1M => GgmlType::IQ1M,
            _ => return None,
        })
    }

    /// Returns `(values_per_block, bytes_per_block)`.
    pub fn block_geometry(self) -> (i32, i32) {
        match self {
            Self::GgufQ4K => (Q4_K_BLOCK_VALUES, Q4_K_BLOCK_BYTES),
            Self::GgufQ5_1 => (Q5_1_BLOCK_VALUES, Q5_1_BLOCK_BYTES),
            Self::GgufQ8_0 => (Q8_0_BLOCK_VALUES, Q8_0_BLOCK_BYTES),
            iq => {
                let (values, bytes) = iq
                    .ggml_type()
                    .expect("IQ format")
                    .block_and_bytes()
                    .expect("canonical IQ geometry");
                (values as i32, bytes as i32)
            }
        }
    }
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
    /// Native Q8_0 physical tensors.
    pub q8_0_tensor_count: u64,
    /// Native Q8_0 bytes.
    pub q8_0_bytes: u64,
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
            NativeQuantizationFormat::GgufQ8_0 => {
                self.q8_0_tensor_count += 1;
                self.q8_0_bytes += bytes;
            }
            _ => {}
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
    endian: GgufEndian,
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

    /// Byte order of multibyte fields in the retained blocks.
    pub fn endian(&self) -> GgufEndian {
        self.endian
    }
}

/// A zero-copy logical matrix-bank view over checkpoint-native physical rows.
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
    /// Copies the packed storage to another execution stream while preserving
    /// this tensor's logical matrix and row view.
    pub fn copy_to_stream(&self, stream: &Stream) -> Result<Self, Exception> {
        let bytes = self.storage.bytes.copy(stream)?;
        eval([&bytes])?;
        Ok(Self {
            storage: Arc::new(NativeStorage {
                format: self.storage.format,
                endian: self.storage.endian,
                bytes,
                byte_len: self.storage.byte_len,
                kind: NativeStorageKind::MlxOwnedCopy,
            }),
            matrix_count: self.matrix_count,
            physical_rows: self.physical_rows,
            row_start: self.row_start,
            rows: self.rows,
            columns: self.columns,
        })
    }

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
            GgufEndian::Little,
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
            GgufEndian::Little,
            stream,
        )
    }

    /// Copies one little-endian Q8_0 physical matrix bank into raw MLX storage.
    ///
    /// `shape` may be `[rows, columns]` or `[matrices, rows, columns]`.
    pub fn from_q8_0_bytes(data: &[u8], shape: &[i32], stream: &Stream) -> Result<Self, Exception> {
        Self::from_native_bytes(
            data,
            shape,
            NativeQuantizationFormat::GgufQ8_0,
            Q8_0_BLOCK_VALUES,
            Q8_0_BLOCK_BYTES,
            GgufEndian::Little,
            stream,
        )
    }

    /// Retains an already materialized MLX byte array as an IQ tensor.
    pub fn from_iq_array(
        bytes: Array,
        shape: &[i32],
        ty: GgmlType,
        endian: GgufEndian,
    ) -> Result<Self, Exception> {
        let format = NativeQuantizationFormat::from_ggml_type(ty)
            .ok_or_else(|| Exception::custom(format!("{ty:?} is not an IQ encoding")))?;
        let (block_values, block_bytes) = format.block_geometry();
        Self::from_native_array(bytes, shape, format, block_values, block_bytes, endian)
    }

    fn from_native_bytes(
        data: &[u8],
        shape: &[i32],
        format: NativeQuantizationFormat,
        block_values: i32,
        block_bytes: i32,
        endian: GgufEndian,
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
            endian,
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

    fn from_native_array(
        bytes: Array,
        shape: &[i32],
        format: NativeQuantizationFormat,
        block_values: i32,
        block_bytes: i32,
        endian: GgufEndian,
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
        if bytes.dtype() != Dtype::Uint8 {
            return Err(Exception::custom(format!(
                "native {format:?} storage must be uint8, got {:?}",
                bytes.dtype()
            )));
        }
        let expected = i64::from(matrix_count)
            * i64::from(physical_rows)
            * i64::from(columns / block_values)
            * i64::from(block_bytes);
        if matrix_count <= 0
            || physical_rows <= 0
            || columns <= 0
            || columns % block_values != 0
            || expected != bytes.size() as i64
        {
            return Err(Exception::custom(format!(
                "native {format:?} storage shape mismatch: logical {shape:?}, {} bytes",
                bytes.size()
            )));
        }
        Ok(Self {
            storage: Arc::new(NativeStorage {
                format,
                endian,
                byte_len: bytes.size(),
                bytes,
                kind: NativeStorageKind::MlxOwnedCopy,
            }),
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

    /// Capabilities implemented by this native representation.
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
                linear: self.matrix_count == 1,
                linear_untransposed: false,
                embedding: self.matrix_count == 1,
                grouped_linear: true,
                selected_gate_up: false,
                selected_down_reduce: true,
            },
            NativeQuantizationFormat::GgufQ8_0 => NativeCapabilities {
                linear: self.matrix_count == 1,
                linear_untransposed: false,
                embedding: self.matrix_count == 1,
                grouped_linear: true,
                selected_gate_up: false,
                selected_down_reduce: true,
            },
            _ => NativeCapabilities {
                linear: self.matrix_count == 1,
                linear_untransposed: false,
                embedding: self.matrix_count == 1,
                grouped_linear: true,
                selected_gate_up: true,
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
        if native_execution_backend(stream)? == NativeExecutionBackend::Metal && transpose {
            return match self.format() {
                NativeQuantizationFormat::GgufQ4K => q4k_linear_metal(input, self, stream),
                NativeQuantizationFormat::GgufQ8_0 => q8_0_linear_metal(input, self, stream),
                NativeQuantizationFormat::GgufQ5_1 => q5_1_linear_metal(input, self, stream),
                _ => iq_linear_metal(input, self, stream),
            };
        }
        self.linear_fallback(input, transpose, stream)
    }

    /// Looks up and dequantizes selected rows from a native matrix.
    pub fn embedding(&self, indices: &Array, stream: &Stream) -> Result<Array, Exception> {
        if self.matrix_count != 1 {
            return Err(Exception::custom(
                "native embedding expects a single logical matrix",
            ));
        }
        if native_execution_backend(stream)? == NativeExecutionBackend::Metal {
            return match self.format() {
                NativeQuantizationFormat::GgufQ4K => q4k_embedding_metal(indices, self, stream),
                NativeQuantizationFormat::GgufQ8_0 => q8_0_embedding_metal(indices, self, stream),
                NativeQuantizationFormat::GgufQ5_1 => q5_1_embedding_metal(indices, self, stream),
                _ => iq_embedding_metal(indices, self, stream),
            };
        }
        self.embedding_cpu_streaming(indices, stream)
    }

    /// Transiently dequantizes this logical view to float32.
    pub fn dequantize(&self, stream: &Stream) -> Result<Array, Exception> {
        let evaluated = self.storage.bytes.evaluated()?;
        let raw = evaluated.as_slice::<u8>();
        let values = match self.format() {
            NativeQuantizationFormat::GgufQ4K => decode_q4k_view(raw, self)?,
            NativeQuantizationFormat::GgufQ5_1 => decode_q5_1_view(raw, self)?,
            NativeQuantizationFormat::GgufQ8_0 => decode_q8_0_view(raw, self)?,
            format => {
                let ty = format.ggml_type().expect("IQ format");
                let physical_shape = if self.matrix_count == 1 {
                    vec![self.physical_rows as u64, self.columns as u64]
                } else {
                    vec![
                        self.matrix_count as u64,
                        self.physical_rows as u64,
                        self.columns as u64,
                    ]
                };
                let tensor = safemlx_gguf::IQuantTensor {
                    shape: physical_shape,
                    ggml_type: ty,
                    endian: self.storage.endian,
                    data: raw.to_vec(),
                };
                let all = tensor
                    .dequantize_f32()
                    .map_err(|error| Exception::custom(error.to_string()))?;
                if self.row_start == 0 && self.rows == self.physical_rows {
                    all
                } else {
                    let mut selected = Vec::with_capacity(
                        self.matrix_count as usize * self.rows as usize * self.columns as usize,
                    );
                    let matrix_stride = self.physical_rows as usize * self.columns as usize;
                    for matrix in 0..self.matrix_count as usize {
                        let start = matrix * matrix_stride
                            + self.row_start as usize * self.columns as usize;
                        let end = start + self.rows as usize * self.columns as usize;
                        selected.extend_from_slice(&all[start..end]);
                    }
                    selected
                }
            }
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
        if native_execution_backend(stream)? == NativeExecutionBackend::GenericFallback {
            return self.linear_cpu_streaming(input, transpose, stream);
        }
        let dense = self.dequantize(stream)?;
        if transpose {
            matmul(input, dense.transpose(stream)?, stream)
        } else {
            matmul(input, dense, stream)
        }
    }

    fn linear_cpu_streaming(
        &self,
        input: &Array,
        transpose: bool,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let expected = if transpose { self.columns } else { self.rows };
        if input.dim(-1) != expected {
            return Err(Exception::custom(format!(
                "native CPU linear expected trailing dimension {expected}, got {:?}",
                input.shape()
            )));
        }
        let outer = input.size() as i32 / expected;
        let input = input.as_dtype(Dtype::Float32, stream)?;
        eval([&input, self.storage.bytes()])?;
        let evaluated_input = input.evaluated()?;
        let input_values = evaluated_input.as_slice::<f32>();
        let evaluated_storage = self.storage.bytes.evaluated()?;
        let raw = evaluated_storage.as_slice::<u8>();

        let output_width = if transpose { self.rows } else { self.columns };
        let mut output = vec![0.0f32; outer as usize * output_width as usize];
        if transpose {
            for output_row in 0..self.rows {
                let weights = decode_native_row(raw, self, 0, output_row)?;
                for input_row in 0..outer as usize {
                    let input_start = input_row * self.columns as usize;
                    output[input_row * self.rows as usize + output_row as usize] = dot_f32(
                        &input_values[input_start..input_start + self.columns as usize],
                        &weights,
                    );
                }
            }
        } else {
            for weight_row in 0..self.rows {
                let weights = decode_native_row(raw, self, 0, weight_row)?;
                for input_row in 0..outer as usize {
                    let scale = input_values[input_row * self.rows as usize + weight_row as usize];
                    let output_row = &mut output[input_row * self.columns as usize
                        ..(input_row + 1) * self.columns as usize];
                    for (value, weight) in output_row.iter_mut().zip(&weights) {
                        *value += scale * weight;
                    }
                }
            }
        }
        let mut shape = input.shape()[..input.ndim() - 1].to_vec();
        shape.push(output_width);
        Array::from_slice(&output, &shape).copy(stream)
    }

    fn embedding_cpu_streaming(
        &self,
        indices: &Array,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let indices = indices.as_dtype(Dtype::Int32, stream)?;
        eval([&indices, self.storage.bytes()])?;
        let evaluated_indices = indices.evaluated()?;
        let index_values = evaluated_indices.as_slice::<i32>();
        let evaluated_storage = self.storage.bytes.evaluated()?;
        let raw = evaluated_storage.as_slice::<u8>();
        let mut output = Vec::with_capacity(index_values.len() * self.columns as usize);
        for &index in index_values {
            if index < 0 || index >= self.rows {
                return Err(Exception::custom(format!(
                    "native embedding index {index} is outside 0..{}",
                    self.rows
                )));
            }
            output.extend(decode_native_row(raw, self, 0, index)?);
        }
        let mut shape = indices.shape().to_vec();
        shape.push(self.columns);
        Array::from_slice(&output, &shape).copy(stream)
    }
}

fn dot_f32(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter()
        .zip(rhs)
        .fold(0.0f32, |sum, (&left, &right)| left.mul_add(right, sum))
}

fn decode_native_row(
    raw: &[u8],
    view: &NativeQuantizedTensor,
    matrix: i32,
    logical_row: i32,
) -> Result<Vec<f32>, Exception> {
    if matrix < 0 || matrix >= view.matrix_count || logical_row < 0 || logical_row >= view.rows {
        return Err(Exception::custom(format!(
            "native row ({matrix}, {logical_row}) is outside {:?}",
            view.shape()
        )));
    }
    let (block_values, block_bytes) = view.format().block_geometry();
    let blocks = view.columns / block_values;
    let row_bytes = blocks as usize * block_bytes as usize;
    let physical_row =
        matrix as usize * view.physical_rows as usize + (view.row_start + logical_row) as usize;
    let start = physical_row * row_bytes;
    let end = start + row_bytes;
    let row = raw
        .get(start..end)
        .ok_or_else(|| Exception::custom("native packed row exceeds storage"))?;
    let mut values = Vec::with_capacity(view.columns as usize);
    match view.format() {
        NativeQuantizationFormat::GgufQ4K => {
            for block in row.chunks_exact(Q4_K_BLOCK_BYTES as usize) {
                decode_q4k_block(block, &mut values);
            }
        }
        NativeQuantizationFormat::GgufQ5_1 => {
            for block in row.chunks_exact(Q5_1_BLOCK_BYTES as usize) {
                decode_q5_1_block(block, &mut values);
            }
        }
        NativeQuantizationFormat::GgufQ8_0 => {
            for block in row.chunks_exact(Q8_0_BLOCK_BYTES as usize) {
                decode_q8_0_block(block, &mut values);
            }
        }
        format => {
            values = safemlx_gguf::IQuantTensor {
                shape: vec![1, view.columns as u64],
                ggml_type: format.ggml_type().expect("IQ format"),
                endian: view.storage.endian,
                data: row.to_vec(),
            }
            .dequantize_f32()
            .map_err(|error| Exception::custom(error.to_string()))?;
        }
    }
    debug_assert_eq!(values.len(), view.columns as usize);
    Ok(values)
}

fn native_grouped_linear_cpu(
    input: &Array,
    weight: &NativeQuantizedTensor,
    group_ids: &Array,
    stream: &Stream,
) -> Result<Array, Exception> {
    let input = input.as_dtype(Dtype::Float32, stream)?;
    let group_ids = group_ids.as_dtype(Dtype::Int32, stream)?;
    eval([&input, &group_ids, weight.storage.bytes()])?;
    let evaluated_input = input.evaluated()?;
    let input_values = evaluated_input.as_slice::<f32>();
    let evaluated_ids = group_ids.evaluated()?;
    let ids = evaluated_ids.as_slice::<i32>();
    let evaluated_storage = weight.storage.bytes.evaluated()?;
    let raw = evaluated_storage.as_slice::<u8>();
    let routes = input.dim(0);
    let mut output = vec![0.0f32; routes as usize * weight.rows as usize];
    for route in 0..routes as usize {
        let expert = ids[route];
        if expert < 0 || expert >= weight.matrix_count {
            return Err(Exception::custom(format!(
                "native grouped expert {expert} is outside 0..{}",
                weight.matrix_count
            )));
        }
        let input_row =
            &input_values[route * weight.columns as usize..(route + 1) * weight.columns as usize];
        for output_row in 0..weight.rows {
            let weights = decode_native_row(raw, weight, expert, output_row)?;
            output[route * weight.rows as usize + output_row as usize] =
                dot_f32(input_row, &weights);
        }
    }
    Array::from_slice(&output, &[routes, weight.rows]).copy(stream)
}

fn native_selected_gate_up_cpu(
    hidden: &Array,
    weight: &NativeQuantizedTensor,
    expert_ids: &Array,
    intermediate: i32,
    stream: &Stream,
) -> Result<Array, Exception> {
    let hidden = hidden.as_dtype(Dtype::Float32, stream)?;
    let expert_ids = expert_ids.as_dtype(Dtype::Int32, stream)?;
    eval([&hidden, &expert_ids, weight.storage.bytes()])?;
    let evaluated_hidden = hidden.evaluated()?;
    let hidden_values = evaluated_hidden.as_slice::<f32>();
    let evaluated_ids = expert_ids.evaluated()?;
    let ids = evaluated_ids.as_slice::<i32>();
    let evaluated_storage = weight.storage.bytes.evaluated()?;
    let raw = evaluated_storage.as_slice::<u8>();
    let mut output = vec![0.0f32; ids.len() * intermediate as usize];
    let gelu_coefficient = 0.797_884_6f32;
    for (route, &expert) in ids.iter().enumerate() {
        if expert < 0 || expert >= weight.matrix_count {
            return Err(Exception::custom(format!(
                "native selected expert {expert} is outside 0..{}",
                weight.matrix_count
            )));
        }
        for row in 0..intermediate {
            let gate = dot_f32(hidden_values, &decode_native_row(raw, weight, expert, row)?);
            let up = dot_f32(
                hidden_values,
                &decode_native_row(raw, weight, expert, intermediate + row)?,
            );
            let activated = 0.5
                * gate
                * (1.0 + (gelu_coefficient * (gate + 0.044_715 * gate * gate * gate)).tanh());
            output[route * intermediate as usize + row as usize] = activated * up;
        }
    }
    Array::from_slice(&output, &[ids.len() as i32, intermediate]).copy(stream)
}

/// Applies an expert-major native quantized matrix bank.
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
            NativeQuantizationFormat::GgufQ8_0 => {
                q8_0_grouped_metal(input, weight, group_ids, stream)
            }
            _ => iq_grouped_metal(input, weight, group_ids, stream),
        };
    }
    native_grouped_linear_cpu(input, weight, group_ids, stream)
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
        match fused_gate_up.format() {
            NativeQuantizationFormat::GgufQ4K => {
                return q4k_gate_up_metal(hidden, fused_gate_up, expert_ids, intermediate, stream);
            }
            format if format.ggml_type().is_some() => {
                return iq_gate_up_metal(hidden, fused_gate_up, expert_ids, intermediate, stream);
            }
            _ => {}
        }
    }
    native_selected_gate_up_cpu(hidden, fused_gate_up, expert_ids, intermediate, stream)
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
            NativeQuantizationFormat::GgufQ8_0 => {
                q8_0_down_reduce_metal(activated, down, expert_ids, route_weights, stream)
            }
            _ => iq_down_reduce_metal(activated, down, expert_ids, route_weights, stream),
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
            "native quantized activation must be float16, bfloat16, or float32, got {dtype:?}"
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
    let out_grid = ((view.rows + OUT_TILE - 1) / OUT_TILE) * OUT_TILE;
    let mut config = q4k_config(outer, view, outer, view.rows, dtype)
        .with_template_arg_int("ROWS", outer)
        .with_template_arg_int("BATCH_TILE", NATIVE_BATCH_TILE);
    let output = if outer == 1 {
        Q4K_LINEAR_KERNEL.with(|cell| -> Result<_, Exception> {
            if cell.borrow().is_none() {
                *cell.borrow_mut() = Some(q4k_linear_kernel()?);
            }
            cell.borrow()
                .as_ref()
                .expect("Q4_K linear kernel initialized")
                .apply_one_device([&flat, view.storage.bytes()], &config, stream)
        })?
    } else {
        let batch_tiles = (outer + NATIVE_BATCH_TILE - 1) / NATIVE_BATCH_TILE;
        config = config.with_grid([REDUCTION_TILE, out_grid, batch_tiles]);
        Q4K_BATCH_KERNEL.with(|cell| -> Result<_, Exception> {
            if cell.borrow().is_none() {
                *cell.borrow_mut() = Some(q4k_batch_kernel()?);
            }
            cell.borrow()
                .as_ref()
                .expect("Q4_K batch kernel initialized")
                .apply_one_device([&flat, view.storage.bytes()], &config, stream)
        })?
    };
    let mut shape = input.shape()[..input.ndim() - 1].to_vec();
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

fn iq_linear_metal(
    input: &Array,
    view: &NativeQuantizedTensor,
    stream: &Stream,
) -> Result<Array, Exception> {
    let dtype = validate_activation_dtype(input)?;
    if input.dim(-1) != view.columns {
        return Err(Exception::custom(format!(
            "IQ linear input {:?} does not match {} columns",
            input.shape(),
            view.columns
        )));
    }
    let rows = input.size() as i32 / view.columns;
    let big_endian = view.storage.endian == GgufEndian::Big;
    let kernel_key = (view.format(), big_endian);
    let flat = input.reshape(&[rows, view.columns], stream)?;
    let (_, block_bytes) = view.format().block_geometry();
    let mut config = MetalKernelConfig::new()
        .with_template_arg_dtype("T", dtype)
        .with_template_arg_int("ROWS", rows)
        .with_template_arg_int("IN_DIM", view.columns)
        .with_template_arg_int("OUT_DIM", view.rows)
        .with_template_arg_int("BLOCKS", view.columns / view.format().block_geometry().0)
        .with_template_arg_int("BLOCK_BYTES", block_bytes)
        .with_template_arg_int("PHYSICAL_ROWS", view.physical_rows)
        .with_template_arg_int("ROW_START", view.row_start)
        .with_template_arg_int("BATCH_TILE", NATIVE_BATCH_TILE)
        .with_grid([32, rows * view.rows, 1])
        .with_thread_group([32, 1, 1])
        .with_output_arg([rows, view.rows], dtype);
    let output = if rows == 1 {
        IQ_LINEAR_KERNEL.with(|cell| -> Result<_, Exception> {
            let mut kernels = cell.borrow_mut();
            if let std::collections::hash_map::Entry::Vacant(entry) = kernels.entry(kernel_key) {
                entry.insert(iq_linear_kernel(view.format(), big_endian)?);
            }
            kernels
                .get(&kernel_key)
                .expect("IQ linear kernel initialized")
                .apply_one_device([&flat, view.storage.bytes()], &config, stream)
        })?
    } else {
        let batch_tiles = (rows + NATIVE_BATCH_TILE - 1) / NATIVE_BATCH_TILE;
        config = config.with_grid([32, view.rows, batch_tiles]);
        IQ_BATCH_KERNEL.with(|cell| -> Result<_, Exception> {
            let mut kernels = cell.borrow_mut();
            if let std::collections::hash_map::Entry::Vacant(entry) = kernels.entry(kernel_key) {
                entry.insert(iq_batch_kernel(view.format(), big_endian)?);
            }
            kernels
                .get(&kernel_key)
                .expect("IQ batch kernel initialized")
                .apply_one_device([&flat, view.storage.bytes()], &config, stream)
        })?
    };
    let mut shape = input.shape()[..input.ndim() - 1].to_vec();
    shape.push(view.rows);
    output.reshape(&shape, stream)
}

fn iq_embedding_metal(
    indices: &Array,
    view: &NativeQuantizedTensor,
    stream: &Stream,
) -> Result<Array, Exception> {
    let count = indices.size() as i32;
    let big_endian = view.storage.endian == GgufEndian::Big;
    let kernel_key = (view.format(), big_endian);
    let (block_values, block_bytes) = view.format().block_geometry();
    let config = MetalKernelConfig::new()
        .with_template_arg_int("IN_DIM", view.columns)
        .with_template_arg_int("ROWS", view.rows)
        .with_template_arg_int("BLOCKS", view.columns / block_values)
        .with_template_arg_int("BLOCK_BYTES", block_bytes)
        .with_template_arg_int("PHYSICAL_ROWS", view.physical_rows)
        .with_template_arg_int("ROW_START", view.row_start)
        .with_grid([count * view.columns, 1, 1])
        .with_thread_group([256, 1, 1])
        .with_output_arg([count, view.columns], Dtype::Float32);
    let output = IQ_EMBEDDING_KERNEL.with(|cell| -> Result<_, Exception> {
        let mut kernels = cell.borrow_mut();
        if let std::collections::hash_map::Entry::Vacant(entry) = kernels.entry(kernel_key) {
            entry.insert(iq_embedding_kernel(view.format(), big_endian)?);
        }
        kernels
            .get(&kernel_key)
            .expect("IQ embedding kernel initialized")
            .apply_one_device([view.storage.bytes(), indices], &config, stream)
    })?;
    let mut shape = indices.shape().to_vec();
    shape.push(view.columns);
    output.reshape(&shape, stream)
}

fn iq_grouped_metal(
    input: &Array,
    view: &NativeQuantizedTensor,
    group_ids: &Array,
    stream: &Stream,
) -> Result<Array, Exception> {
    let dtype = validate_activation_dtype(input)?;
    let rows = input.dim(0);
    let big_endian = view.storage.endian == GgufEndian::Big;
    let kernel_key = (view.format(), big_endian);
    let (block_values, block_bytes) = view.format().block_geometry();
    let config = MetalKernelConfig::new()
        .with_template_arg_dtype("T", dtype)
        .with_template_arg_int("IN_DIM", view.columns)
        .with_template_arg_int("OUT_DIM", view.rows)
        .with_template_arg_int("BLOCKS", view.columns / block_values)
        .with_template_arg_int("BLOCK_BYTES", block_bytes)
        .with_template_arg_int("PHYSICAL_ROWS", view.physical_rows)
        .with_template_arg_int("ROW_START", view.row_start)
        .with_grid([32, rows * view.rows, 1])
        .with_thread_group([32, 1, 1])
        .with_output_arg([rows, view.rows], dtype);
    IQ_GROUPED_KERNEL.with(|cell| -> Result<_, Exception> {
        let mut kernels = cell.borrow_mut();
        if let std::collections::hash_map::Entry::Vacant(entry) = kernels.entry(kernel_key) {
            entry.insert(iq_grouped_kernel(view.format(), big_endian)?);
        }
        kernels
            .get(&kernel_key)
            .expect("IQ grouped kernel initialized")
            .apply_one_device([input, view.storage.bytes(), group_ids], &config, stream)
    })
}

fn q8_0_config(
    view: &NativeQuantizedTensor,
    output_rows: i32,
    output_cols: i32,
    dtype: Dtype,
) -> MetalKernelConfig {
    let out_tile = q8_0_out_tile(output_cols);
    let out_grid = ((output_cols + out_tile - 1) / out_tile) * out_tile;
    MetalKernelConfig::new()
        .with_template_arg_dtype("T", dtype)
        .with_template_arg_int("ROWS", output_rows)
        .with_template_arg_int("IN_DIM", view.columns)
        .with_template_arg_int("OUT_DIM", output_cols)
        .with_template_arg_int("OUT_GRID", out_grid)
        .with_template_arg_int("BLOCKS", view.columns / Q8_0_BLOCK_VALUES)
        .with_template_arg_int("PHYSICAL_ROWS", view.physical_rows)
        .with_template_arg_int("ROW_START", view.row_start)
        .with_template_arg_int("OUT_TILE", out_tile)
        .with_template_arg_int("BATCH_TILE", NATIVE_BATCH_TILE)
        .with_thread_group([REDUCTION_TILE, out_tile, 1])
        .with_output_arg([output_rows, output_cols], dtype)
}

fn q8_0_out_tile(output_cols: i32) -> i32 {
    if output_cols >= Q8_LARGE_OUTPUT_ROWS {
        Q8_LARGE_OUT_TILE
    } else {
        OUT_TILE
    }
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

fn q5_1_linear_metal(
    input: &Array,
    view: &NativeQuantizedTensor,
    stream: &Stream,
) -> Result<Array, Exception> {
    if input.dim(-1) != view.columns {
        return Err(Exception::custom(format!(
            "native Q5_1 linear expected input dimension {}, got {:?}",
            view.columns,
            input.shape()
        )));
    }
    let dtype = validate_activation_dtype(input)?;
    let outer = input.size() as i32 / view.columns;
    let flat = input.reshape(&[outer, view.columns], stream)?;
    let out_grid = ((view.rows + OUT_TILE - 1) / OUT_TILE) * OUT_TILE;
    let mut config = q5_1_config(outer, view, outer, view.rows, dtype)
        .with_template_arg_int("ROWS", outer)
        .with_template_arg_int("BATCH_TILE", NATIVE_BATCH_TILE);
    let output = if outer == 1 {
        Q5_1_LINEAR_KERNEL.with(|cell| -> Result<_, Exception> {
            if cell.borrow().is_none() {
                *cell.borrow_mut() = Some(q5_1_linear_kernel()?);
            }
            cell.borrow()
                .as_ref()
                .expect("Q5_1 linear kernel initialized")
                .apply_one_device([&flat, view.storage.bytes()], &config, stream)
        })?
    } else {
        let batch_tiles = (outer + NATIVE_BATCH_TILE - 1) / NATIVE_BATCH_TILE;
        config = config.with_grid([REDUCTION_TILE, out_grid, batch_tiles]);
        Q5_1_BATCH_KERNEL.with(|cell| -> Result<_, Exception> {
            if cell.borrow().is_none() {
                *cell.borrow_mut() = Some(q5_1_batch_kernel()?);
            }
            cell.borrow()
                .as_ref()
                .expect("Q5_1 batch kernel initialized")
                .apply_one_device([&flat, view.storage.bytes()], &config, stream)
        })?
    };
    let mut shape = input.shape()[..input.ndim() - 1].to_vec();
    shape.push(view.rows);
    output.reshape(&shape, stream)
}

fn q5_1_embedding_metal(
    indices: &Array,
    view: &NativeQuantizedTensor,
    stream: &Stream,
) -> Result<Array, Exception> {
    let count = indices.size() as i32;
    let config = q5_1_config(count, view, count, view.columns, Dtype::Float32)
        .with_template_arg_int("ROWS", view.rows)
        .with_grid([count * view.columns, 1, 1])
        .with_thread_group([256, 1, 1]);
    let output = Q5_1_EMBEDDING_KERNEL.with(|cell| -> Result<_, Exception> {
        if cell.borrow().is_none() {
            *cell.borrow_mut() = Some(q5_1_embedding_kernel()?);
        }
        cell.borrow()
            .as_ref()
            .expect("Q5_1 embedding kernel initialized")
            .apply_one_device([view.storage.bytes(), indices], &config, stream)
    })?;
    let mut shape = indices.shape().to_vec();
    shape.push(view.columns);
    output.reshape(&shape, stream)
}

fn q8_0_linear_metal(
    input: &Array,
    view: &NativeQuantizedTensor,
    stream: &Stream,
) -> Result<Array, Exception> {
    if input.dim(-1) != view.columns {
        return Err(Exception::custom(format!(
            "native Q8_0 linear expected input dimension {}, got {:?}",
            view.columns,
            input.shape()
        )));
    }
    let dtype = validate_activation_dtype(input)?;
    let outer = input.size() as i32 / view.columns;
    let flat = input.reshape(&[outer, view.columns], stream)?;
    let out_tile = q8_0_out_tile(view.rows);
    let out_grid = ((view.rows + out_tile - 1) / out_tile) * out_tile;
    let mut config = q8_0_config(view, outer, view.rows, dtype);
    let output = if outer == 1 {
        config = config.with_grid([REDUCTION_TILE, out_grid, 1]);
        Q8_0_LINEAR_KERNEL.with(|cell| -> Result<_, Exception> {
            if cell.borrow().is_none() {
                *cell.borrow_mut() = Some(q8_0_linear_kernel()?);
            }
            cell.borrow()
                .as_ref()
                .expect("Q8_0 linear kernel initialized")
                .apply_one_device([&flat, view.storage.bytes()], &config, stream)
        })?
    } else {
        let batch_tiles = (outer + NATIVE_BATCH_TILE - 1) / NATIVE_BATCH_TILE;
        config = config.with_grid([REDUCTION_TILE, out_grid, batch_tiles]);
        Q8_0_BATCH_KERNEL.with(|cell| -> Result<_, Exception> {
            if cell.borrow().is_none() {
                *cell.borrow_mut() = Some(q8_0_batch_kernel()?);
            }
            cell.borrow()
                .as_ref()
                .expect("Q8_0 batch kernel initialized")
                .apply_one_device([&flat, view.storage.bytes()], &config, stream)
        })?
    };
    let mut shape = input.shape()[..input.ndim() - 1].to_vec();
    shape.push(view.rows);
    output.reshape(&shape, stream)
}

fn q8_0_grouped_metal(
    input: &Array,
    view: &NativeQuantizedTensor,
    group_ids: &Array,
    stream: &Stream,
) -> Result<Array, Exception> {
    let dtype = validate_activation_dtype(input)?;
    let routes = input.dim(0);
    let out_tile = q8_0_out_tile(view.rows);
    let out_grid = ((view.rows + out_tile - 1) / out_tile) * out_tile;
    let config = q8_0_config(view, routes, view.rows, dtype)
        .with_template_arg_int("MATRIX_COUNT", view.matrix_count)
        .with_grid([REDUCTION_TILE, routes * out_grid, 1]);
    Q8_0_GROUPED_KERNEL.with(|cell| -> Result<_, Exception> {
        if cell.borrow().is_none() {
            *cell.borrow_mut() = Some(q8_0_grouped_kernel()?);
        }
        cell.borrow()
            .as_ref()
            .expect("Q8_0 grouped kernel initialized")
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

fn q8_0_embedding_metal(
    indices: &Array,
    view: &NativeQuantizedTensor,
    stream: &Stream,
) -> Result<Array, Exception> {
    let count = indices.size() as i32;
    let config = MetalKernelConfig::new()
        .with_template_arg_int("IN_DIM", view.columns)
        .with_template_arg_int("ROWS", view.rows)
        .with_template_arg_int("BLOCKS", view.columns / Q8_0_BLOCK_VALUES)
        .with_template_arg_int("PHYSICAL_ROWS", view.physical_rows)
        .with_template_arg_int("ROW_START", view.row_start)
        .with_grid([count * view.columns, 1, 1])
        .with_thread_group([256, 1, 1])
        .with_output_arg([count, view.columns], Dtype::Float32);
    let output = Q8_0_EMBEDDING_KERNEL.with(|cell| -> Result<_, Exception> {
        if cell.borrow().is_none() {
            *cell.borrow_mut() = Some(q8_0_embedding_kernel()?);
        }
        cell.borrow()
            .as_ref()
            .expect("Q8_0 embedding kernel initialized")
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

fn q8_0_down_reduce_metal(
    activated: &Array,
    view: &NativeQuantizedTensor,
    expert_ids: &Array,
    route_weights: &Array,
    stream: &Stream,
) -> Result<Array, Exception> {
    let dtype = validate_activation_dtype(activated)?;
    let top_k = expert_ids.dim(0);
    let out_tile = q8_0_out_tile(view.rows);
    let out_grid = ((view.rows + out_tile - 1) / out_tile) * out_tile;
    let config = q8_0_config(view, 1, view.rows, dtype)
        .with_template_arg_int("TOP_K", top_k)
        .with_template_arg_int("MATRIX_COUNT", view.matrix_count)
        .with_grid([REDUCTION_TILE, out_grid, 1]);
    Q8_0_DOWN_REDUCE_KERNEL.with(|cell| -> Result<_, Exception> {
        if cell.borrow().is_none() {
            *cell.borrow_mut() = Some(q8_0_down_reduce_kernel()?);
        }
        cell.borrow()
            .as_ref()
            .expect("Q8_0 down/reduce kernel initialized")
            .apply_one_device(
                [activated, view.storage.bytes(), expert_ids, route_weights],
                &config,
                stream,
            )
    })
}

fn iq_gate_up_metal(
    hidden: &Array,
    view: &NativeQuantizedTensor,
    expert_ids: &Array,
    intermediate: i32,
    stream: &Stream,
) -> Result<Array, Exception> {
    let dtype = validate_activation_dtype(hidden)?;
    let top_k = expert_ids.dim(0);
    let big_endian = view.storage.endian == GgufEndian::Big;
    let (block_values, block_bytes) = view.format().block_geometry();
    let base_config = MetalKernelConfig::new()
        .with_template_arg_dtype("T", dtype)
        .with_template_arg_int("IN_DIM", view.columns)
        .with_template_arg_int("INTERMEDIATE", intermediate)
        .with_template_arg_int("BLOCKS", view.columns / block_values)
        .with_template_arg_int("BLOCK_BYTES", block_bytes)
        .with_template_arg_int("PHYSICAL_ROWS", view.physical_rows)
        .with_grid([32, intermediate, 1])
        .with_thread_group([32, 1, 1])
        .with_output_arg([intermediate], dtype);
    let outputs = IQ_GATE_UP_KERNEL.with(|cell| -> Result<Vec<Array>, Exception> {
        let mut outputs = Vec::with_capacity(top_k as usize);
        for route in 0..top_k {
            let config = base_config.clone();
            let mut kernels = cell.borrow_mut();
            let key = (view.format(), big_endian, route);
            if let std::collections::hash_map::Entry::Vacant(entry) = kernels.entry(key) {
                entry.insert(iq_gate_up_kernel(view.format(), big_endian, route)?);
            }
            let output = kernels
                .get(&key)
                .expect("IQ gate/up kernel initialized")
                .apply_one_device([hidden, view.storage.bytes(), expert_ids], &config, stream)?;
            outputs.push(output);
        }
        Ok(outputs)
    })?;
    stack_axis(&outputs, 0, stream)
}

fn iq_down_reduce_metal(
    activated: &Array,
    view: &NativeQuantizedTensor,
    expert_ids: &Array,
    route_weights: &Array,
    stream: &Stream,
) -> Result<Array, Exception> {
    let dtype = validate_activation_dtype(activated)?;
    let top_k = expert_ids.dim(0);
    let big_endian = view.storage.endian == GgufEndian::Big;
    let kernel_key = (view.format(), big_endian);
    let (block_values, block_bytes) = view.format().block_geometry();
    let config = MetalKernelConfig::new()
        .with_template_arg_dtype("T", dtype)
        .with_template_arg_int("TOP_K", top_k)
        .with_template_arg_int("IN_DIM", view.columns)
        .with_template_arg_int("OUT_DIM", view.rows)
        .with_template_arg_int("BLOCKS", view.columns / block_values)
        .with_template_arg_int("BLOCK_BYTES", block_bytes)
        .with_template_arg_int("PHYSICAL_ROWS", view.physical_rows)
        .with_template_arg_int("ROW_START", view.row_start)
        .with_grid([32, view.rows, 1])
        .with_thread_group([32, 1, 1])
        .with_output_arg([1, view.rows], dtype);
    IQ_DOWN_REDUCE_KERNEL.with(|cell| -> Result<_, Exception> {
        let mut kernels = cell.borrow_mut();
        if let std::collections::hash_map::Entry::Vacant(entry) = kernels.entry(kernel_key) {
            entry.insert(iq_down_reduce_kernel(view.format(), big_endian)?);
        }
        kernels
            .get(&kernel_key)
            .expect("IQ down/reduce kernel initialized")
            .apply_one_device(
                [activated, view.storage.bytes(), expert_ids, route_weights],
                &config,
                stream,
            )
    })
}

fn iq_linear_kernel(
    format: NativeQuantizationFormat,
    big_endian: bool,
) -> Result<MetalKernel, Exception> {
    MetalKernel::new(
        format!("native_{format:?}_linear_be_{big_endian}").to_lowercase(),
        ["input", "weight"],
        ["out"],
        concat!(
            "uint lane = thread_position_in_grid.x;",
            "uint item = thread_position_in_grid.y;",
            "uint row = item / OUT_DIM;",
            "uint out_col = item % OUT_DIM;",
            "uint physical_row = ROW_START + out_col;",
            "uint row_base = physical_row * BLOCKS * BLOCK_BYTES;",
            "float acc = 0.0f;",
            "for (uint col = lane; col < IN_DIM; col += 32) {",
            " acc += float(input[row * IN_DIM + col]) * iq_value(weight, row_base, col);",
            "}",
            "float total = simd_sum(acc);",
            "if (lane == 0) out[row * OUT_DIM + out_col] = T(total);"
        ),
        iq_metal_header(format, big_endian),
        true,
        false,
    )
}

fn iq_batch_kernel(
    format: NativeQuantizationFormat,
    big_endian: bool,
) -> Result<MetalKernel, Exception> {
    MetalKernel::new(
        format!("native_{format:?}_batch_be_{big_endian}").to_lowercase(),
        ["input", "weight"],
        ["out"],
        concat!(
            "uint lane = thread_position_in_grid.x;",
            "uint out_col = thread_position_in_grid.y;",
            "uint first_row = thread_position_in_grid.z * BATCH_TILE;",
            "uint physical_row = ROW_START + out_col;",
            "uint row_base = physical_row * BLOCKS * BLOCK_BYTES;",
            "float acc[BATCH_TILE];",
            "for (uint r = 0; r < BATCH_TILE; ++r) acc[r] = 0.0f;",
            "for (uint col = lane; col < IN_DIM; col += 32) {",
            " float w = iq_value(weight, row_base, col);",
            " for (uint r = 0; r < BATCH_TILE; ++r) {",
            "  uint row = first_row + r;",
            "  if (row < ROWS) acc[r] += float(input[row * IN_DIM + col]) * w;",
            " }",
            "}",
            "for (uint r = 0; r < BATCH_TILE; ++r) {",
            " float total = simd_sum(acc[r]);",
            " uint row = first_row + r;",
            " if (lane == 0 && row < ROWS) out[row * OUT_DIM + out_col] = T(total);",
            "}"
        ),
        iq_metal_header(format, big_endian),
        true,
        false,
    )
}

fn iq_embedding_kernel(
    format: NativeQuantizationFormat,
    big_endian: bool,
) -> Result<MetalKernel, Exception> {
    MetalKernel::new(
        format!("native_{format:?}_embedding_be_{big_endian}").to_lowercase(),
        ["weight", "indices"],
        ["out"],
        concat!(
            "uint elem = thread_position_in_grid.x;",
            "uint col = elem % IN_DIM;",
            "uint output_row = elem / IN_DIM;",
            "uint row = uint(indices[output_row]);",
            "if (row >= ROWS) { out[elem] = 0.0f; return; }",
            "uint physical_row = ROW_START + row;",
            "uint row_base = physical_row * BLOCKS * BLOCK_BYTES;",
            "out[elem] = iq_value(weight, row_base, col);"
        ),
        iq_metal_header(format, big_endian),
        true,
        false,
    )
}

fn iq_grouped_kernel(
    format: NativeQuantizationFormat,
    big_endian: bool,
) -> Result<MetalKernel, Exception> {
    MetalKernel::new(
        format!("native_{format:?}_grouped_be_{big_endian}").to_lowercase(),
        ["input", "weight", "group_ids"],
        ["out"],
        concat!(
            "uint lane = thread_position_in_grid.x;",
            "uint item = thread_position_in_grid.y;",
            "uint row = item / OUT_DIM;",
            "uint out_col = item % OUT_DIM;",
            "uint expert = uint(group_ids[row]);",
            "uint physical_row = expert * PHYSICAL_ROWS + ROW_START + out_col;",
            "uint row_base = physical_row * BLOCKS * BLOCK_BYTES;",
            "float acc = 0.0f;",
            "for (uint col = lane; col < IN_DIM; col += 32) {",
            " acc += float(input[row * IN_DIM + col]) * iq_value(weight, row_base, col);",
            "}",
            "float total = simd_sum(acc);",
            "if (lane == 0) out[row * OUT_DIM + out_col] = T(total);"
        ),
        iq_metal_header(format, big_endian),
        true,
        false,
    )
}

fn iq_gate_up_kernel(
    format: NativeQuantizationFormat,
    big_endian: bool,
    route: i32,
) -> Result<MetalKernel, Exception> {
    let source = concat!(
        "uint lane = thread_position_in_grid.x;",
        "uint out_col = thread_position_in_grid.y;",
        "uint expert = uint(expert_ids[ROUTE_INDEX]);",
        " uint gate_row = expert * PHYSICAL_ROWS + out_col;",
        " uint up_row = gate_row + INTERMEDIATE;",
        " uint gate_base = gate_row * BLOCKS * BLOCK_BYTES;",
        " uint up_base = up_row * BLOCKS * BLOCK_BYTES;",
        " float gate_acc = 0.0f;",
        " float up_acc = 0.0f;",
        " for (uint col = lane; col < IN_DIM; col += 32) {",
        "  float x = float(input[col]);",
        "  gate_acc += x * iq_value(weight, gate_base, col);",
        "  up_acc += x * iq_value(weight, up_base, col);",
        " }",
        " float gate = simd_sum(gate_acc);",
        " float up = simd_sum(up_acc);",
        "if (lane == 0) {",
        "  float c = 0.7978845608028654f;",
        "  float activated = 0.5f * gate * (1.0f + metal::tanh(c * (gate + 0.044715f * gate * gate * gate)));",
        "  out[out_col] = T(activated * up);",
        "}"
    )
    .replace("ROUTE_INDEX", &route.to_string());
    MetalKernel::new(
        format!("native_{format:?}_selected_gate_up_be_{big_endian}_route_{route}").to_lowercase(),
        ["input", "weight", "expert_ids"],
        ["out"],
        source,
        iq_metal_header(format, big_endian),
        true,
        false,
    )
}

fn iq_down_reduce_kernel(
    format: NativeQuantizationFormat,
    big_endian: bool,
) -> Result<MetalKernel, Exception> {
    MetalKernel::new(
        format!("native_{format:?}_selected_down_reduce_be_{big_endian}").to_lowercase(),
        ["input", "weight", "expert_ids", "route_weights"],
        ["out"],
        concat!(
            "uint lane = thread_position_in_grid.x;",
            "uint out_col = thread_position_in_grid.y;",
            "float acc = 0.0f;",
            "for (uint route = 0; route < TOP_K; ++route) {",
            " uint expert = uint(expert_ids[route]);",
            " uint physical_row = expert * PHYSICAL_ROWS + ROW_START + out_col;",
            " uint row_base = physical_row * BLOCKS * BLOCK_BYTES;",
            " float route_acc = 0.0f;",
            " for (uint col = lane; col < IN_DIM; col += 32) {",
            "  route_acc += float(input[route * IN_DIM + col]) * iq_value(weight, row_base, col);",
            " }",
            " acc += route_acc * float(route_weights[route]);",
            "}",
            "float total = simd_sum(acc);",
            "if (lane == 0) out[out_col] = T(total);"
        ),
        iq_metal_header(format, big_endian),
        true,
        false,
    )
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

fn q4k_batch_kernel() -> Result<MetalKernel, Exception> {
    MetalKernel::new(
        "native_q4k_batch",
        ["input", "weight"],
        ["out"],
        concat!(
            "uint lane = thread_position_in_grid.x;",
            "uint out_col = thread_position_in_grid.y;",
            "uint first_row = thread_position_in_grid.z * BATCH_TILE;",
            "float acc[BATCH_TILE];",
            "for (uint r = 0; r < BATCH_TILE; ++r) acc[r] = 0.0f;",
            "if (out_col < OUT_DIM) {",
            " uint physical_row = ROW_START + out_col;",
            " uint matrix_base = physical_row * BLOCKS * 144;",
            " for (uint block = 0; block < BLOCKS; ++block) {",
            "  uint base = matrix_base + block * 144;",
            "  uint input_block = block * 256;",
            "  for (uint g = 0; g < 8; ++g) {",
            "   float w = q4k_value(weight, base, g, lane);",
            "   uint col = input_block + g * 32 + lane;",
            "   for (uint r = 0; r < BATCH_TILE; ++r) {",
            "    uint row = first_row + r;",
            "    if (row < ROWS) acc[r] += float(input[row * IN_DIM + col]) * w;",
            "   }",
            "  }",
            " }",
            "}",
            "for (uint r = 0; r < BATCH_TILE; ++r) {",
            " float total = simd_sum(acc[r]);",
            " uint row = first_row + r;",
            " if (lane == 0 && row < ROWS && out_col < OUT_DIM) out[row * OUT_DIM + out_col] = T(total);",
            "}"
        ),
        Q4K_METAL_HEADER,
        true,
        false,
    )
}

fn q5_1_linear_kernel() -> Result<MetalKernel, Exception> {
    MetalKernel::new(
        "native_q5_1_linear",
        ["input", "weight"],
        ["out"],
        concat!(
            "uint lane = thread_position_in_grid.x;",
            "uint row = thread_position_in_grid.y / OUT_GRID;",
            "uint out_col = thread_position_in_grid.y % OUT_GRID;",
            "float acc = 0.0f;",
            "if (out_col < OUT_DIM) {",
            " uint physical_row = ROW_START + out_col;",
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

fn q5_1_batch_kernel() -> Result<MetalKernel, Exception> {
    MetalKernel::new(
        "native_q5_1_batch",
        ["input", "weight"],
        ["out"],
        concat!(
            "uint lane = thread_position_in_grid.x;",
            "uint out_col = thread_position_in_grid.y;",
            "uint first_row = thread_position_in_grid.z * BATCH_TILE;",
            "float acc[BATCH_TILE];",
            "for (uint r = 0; r < BATCH_TILE; ++r) acc[r] = 0.0f;",
            "if (out_col < OUT_DIM) {",
            " uint physical_row = ROW_START + out_col;",
            " uint matrix_base = physical_row * BLOCKS * 24;",
            " for (uint block = lane; block < BLOCKS; block += REDUCTION_TILE) {",
            "  uint base = matrix_base + block * 24;",
            "  for (uint i = 0; i < 32; ++i) {",
            "   float w = q5_1_value(weight, base, i);",
            "   uint col = block * 32 + i;",
            "   for (uint r = 0; r < BATCH_TILE; ++r) {",
            "    uint row = first_row + r;",
            "    if (row < ROWS) acc[r] += float(input[row * IN_DIM + col]) * w;",
            "   }",
            "  }",
            " }",
            "}",
            "for (uint r = 0; r < BATCH_TILE; ++r) {",
            " float total = simd_sum(acc[r]);",
            " uint row = first_row + r;",
            " if (lane == 0 && row < ROWS && out_col < OUT_DIM) out[row * OUT_DIM + out_col] = T(total);",
            "}"
        ),
        Q5_1_METAL_HEADER,
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

fn q5_1_embedding_kernel() -> Result<MetalKernel, Exception> {
    MetalKernel::new(
        "native_q5_1_embedding",
        ["weight", "indices"],
        ["out"],
        concat!(
            "uint elem = thread_position_in_grid.x;",
            "uint col = elem % IN_DIM;",
            "uint output_row = elem / IN_DIM;",
            "uint row = uint(indices[output_row]);",
            "if (row >= ROWS) { out[elem] = 0.0f; return; }",
            "uint physical_row = ROW_START + row;",
            "uint block = col / 32;",
            "uint within = col % 32;",
            "uint base = (physical_row * BLOCKS + block) * 24;",
            "out[elem] = q5_1_value(weight, base, within);"
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

fn q8_0_linear_kernel() -> Result<MetalKernel, Exception> {
    MetalKernel::new(
        "native_q8_0_linear",
        ["input", "weight"],
        ["out"],
        concat!(
            "uint lane = thread_position_in_grid.x;",
            "uint out_col = thread_position_in_grid.y;",
            "float acc = 0.0f;",
            "if (out_col < OUT_DIM) {",
            " uint physical_row = ROW_START + out_col;",
            " uint matrix_base = physical_row * BLOCKS * 34;",
            " for (uint block = 0; block < BLOCKS; ++block) {",
            "  uint base = matrix_base + block * 34;",
            "  float d = q8_0_scale(weight, base);",
            "  int q = q8_0_quant(weight, base + 2 + lane);",
            "  acc += float(input[block * 32 + lane]) * d * float(q);",
            " }",
            "}",
            "float total = simd_sum(acc);",
            "if (lane == 0 && out_col < OUT_DIM) out[out_col] = T(total);"
        ),
        Q8_0_METAL_HEADER,
        true,
        false,
    )
}

fn q8_0_batch_kernel() -> Result<MetalKernel, Exception> {
    MetalKernel::new(
        "native_q8_0_batch",
        ["input", "weight"],
        ["out"],
        concat!(
            "uint lane = thread_position_in_grid.x;",
            "uint out_col = thread_position_in_grid.y;",
            "uint first_row = thread_position_in_grid.z * BATCH_TILE;",
            "float acc0 = 0.0f;",
            "float acc1 = 0.0f;",
            "float acc2 = 0.0f;",
            "float acc3 = 0.0f;",
            "float acc4 = 0.0f;",
            "float acc5 = 0.0f;",
            "float acc6 = 0.0f;",
            "float acc7 = 0.0f;",
            "if (out_col < OUT_DIM) {",
            " uint physical_row = ROW_START + out_col;",
            " uint matrix_base = physical_row * BLOCKS * 34;",
            " for (uint block = 0; block < BLOCKS; ++block) {",
            "  uint base = matrix_base + block * 34;",
            "  float w = q8_0_scale(weight, base) * float(q8_0_quant(weight, base + 2 + lane));",
            "  uint input_col = block * 32 + lane;",
            "  if (first_row < ROWS) acc0 += float(input[first_row * IN_DIM + input_col]) * w;",
            "  if (first_row + 1 < ROWS) acc1 += float(input[(first_row + 1) * IN_DIM + input_col]) * w;",
            "  if (first_row + 2 < ROWS) acc2 += float(input[(first_row + 2) * IN_DIM + input_col]) * w;",
            "  if (first_row + 3 < ROWS) acc3 += float(input[(first_row + 3) * IN_DIM + input_col]) * w;",
            "  if (first_row + 4 < ROWS) acc4 += float(input[(first_row + 4) * IN_DIM + input_col]) * w;",
            "  if (first_row + 5 < ROWS) acc5 += float(input[(first_row + 5) * IN_DIM + input_col]) * w;",
            "  if (first_row + 6 < ROWS) acc6 += float(input[(first_row + 6) * IN_DIM + input_col]) * w;",
            "  if (first_row + 7 < ROWS) acc7 += float(input[(first_row + 7) * IN_DIM + input_col]) * w;",
            " }",
            "}",
            "float total0 = simd_sum(acc0);",
            "float total1 = simd_sum(acc1);",
            "float total2 = simd_sum(acc2);",
            "float total3 = simd_sum(acc3);",
            "float total4 = simd_sum(acc4);",
            "float total5 = simd_sum(acc5);",
            "float total6 = simd_sum(acc6);",
            "float total7 = simd_sum(acc7);",
            "if (lane == 0 && out_col < OUT_DIM) {",
            " if (first_row < ROWS) out[first_row * OUT_DIM + out_col] = T(total0);",
            " if (first_row + 1 < ROWS) out[(first_row + 1) * OUT_DIM + out_col] = T(total1);",
            " if (first_row + 2 < ROWS) out[(first_row + 2) * OUT_DIM + out_col] = T(total2);",
            " if (first_row + 3 < ROWS) out[(first_row + 3) * OUT_DIM + out_col] = T(total3);",
            " if (first_row + 4 < ROWS) out[(first_row + 4) * OUT_DIM + out_col] = T(total4);",
            " if (first_row + 5 < ROWS) out[(first_row + 5) * OUT_DIM + out_col] = T(total5);",
            " if (first_row + 6 < ROWS) out[(first_row + 6) * OUT_DIM + out_col] = T(total6);",
            " if (first_row + 7 < ROWS) out[(first_row + 7) * OUT_DIM + out_col] = T(total7);",
            "}"
        ),
        Q8_0_METAL_HEADER,
        true,
        false,
    )
}

fn q8_0_grouped_kernel() -> Result<MetalKernel, Exception> {
    MetalKernel::new(
        "native_q8_0_grouped",
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
            " uint matrix_base = physical_row * BLOCKS * 34;",
            " for (uint block = 0; block < BLOCKS; ++block) {",
            "  uint base = matrix_base + block * 34;",
            "  float d = q8_0_scale(weight, base);",
            "  int q = q8_0_quant(weight, base + 2 + lane);",
            "  acc += float(input[row * IN_DIM + block * 32 + lane]) * d * float(q);",
            " }",
            "}",
            "float total = simd_sum(acc);",
            "if (lane == 0 && out_col < OUT_DIM) out[row * OUT_DIM + out_col] = T(total);"
        ),
        Q8_0_METAL_HEADER,
        true,
        false,
    )
}

fn q8_0_embedding_kernel() -> Result<MetalKernel, Exception> {
    MetalKernel::new(
        "native_q8_0_embedding",
        ["weight", "indices"],
        ["out"],
        concat!(
            "uint elem = thread_position_in_grid.x;",
            "uint col = elem % IN_DIM;",
            "uint output_row = elem / IN_DIM;",
            "uint row = uint(indices[output_row]);",
            "if (row >= ROWS) { out[elem] = 0.0f; return; }",
            "uint physical_row = ROW_START + row;",
            "uint block = col / 32;",
            "uint lane = col % 32;",
            "uint base = (physical_row * BLOCKS + block) * 34;",
            "out[elem] = q8_0_scale(weight, base) * float(q8_0_quant(weight, base + 2 + lane));"
        ),
        Q8_0_METAL_HEADER,
        true,
        false,
    )
}

fn q8_0_down_reduce_kernel() -> Result<MetalKernel, Exception> {
    MetalKernel::new(
        "native_q8_0_selected_down_reduce",
        ["input", "weight", "expert_ids", "route_weights"],
        ["out"],
        concat!(
            "uint lane = thread_position_in_grid.x;",
            "uint out_col = thread_position_in_grid.y;",
            "float acc = 0.0f;",
            "if (out_col < OUT_DIM) {",
            " for (uint route = 0; route < TOP_K; ++route) {",
            "  uint expert = uint(expert_ids[route]);",
            "  uint physical_row = expert * PHYSICAL_ROWS + ROW_START + out_col;",
            "  uint matrix_base = physical_row * BLOCKS * 34;",
            "  float route_acc = 0.0f;",
            "  for (uint block = 0; block < BLOCKS; ++block) {",
            "   uint base = matrix_base + block * 34;",
            "   float d = q8_0_scale(weight, base);",
            "   int q = q8_0_quant(weight, base + 2 + lane);",
            "   route_acc += float(input[route * IN_DIM + block * 32 + lane]) * d * float(q);",
            "  }",
            "  acc += route_acc * float(route_weights[route]);",
            " }",
            "}",
            "float total = simd_sum(acc);",
            "if (lane == 0 && out_col < OUT_DIM) out[out_col] = T(total);"
        ),
        Q8_0_METAL_HEADER,
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

// Q8_0 layout follows llama.cpp's ggml-quants reference implementation
// (MIT) and MLX's GGUF converter (MIT).
const Q8_0_METAL_HEADER: &str = concat!(
    "float q8_0_scale(const device uint8_t* weight, uint base) {",
    " uint d_bits = uint(weight[base]) | (uint(weight[base + 1]) << 8);",
    " return float(as_type<half>(ushort(d_bits)));",
    "}\n",
    "int q8_0_quant(const device uint8_t* weight, uint offset) {",
    " return int(as_type<char>(weight[offset]));",
    "}\n"
);

fn iq_metal_array<T: std::fmt::LowerHex>(
    output: &mut String,
    metal_type: &str,
    name: &str,
    values: &[T],
) {
    let _ = write!(output, "constant {metal_type} {name}[{}]={{", values.len());
    for value in values {
        let _ = write!(output, "0x{value:x},");
    }
    output.push_str("};\n");
}

fn iq_metal_header(format: NativeQuantizationFormat, big_endian: bool) -> String {
    use safemlx_gguf::iquant_tables::{
        IQ1S_GRID, IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID, KSIGNS_IQ2XS,
        KVALUES_IQ4NL,
    };

    let mut header = format!(
        "constant bool BIG_ENDIAN = {};\n",
        if big_endian { "true" } else { "false" }
    );
    header.push_str(concat!(
        "ushort iq_u16(const device uint8_t* w,uint p){",
        "return BIG_ENDIAN ? ushort((uint(w[p])<<8)|uint(w[p+1])) : ushort(uint(w[p])|(uint(w[p+1])<<8));}\n",
        "uint iq_u32(const device uint8_t* w,uint p){",
        "return BIG_ENDIAN ? ((uint(w[p])<<24)|(uint(w[p+1])<<16)|(uint(w[p+2])<<8)|uint(w[p+3]))",
        ": (uint(w[p])|(uint(w[p+1])<<8)|(uint(w[p+2])<<16)|(uint(w[p+3])<<24));}\n",
        "float iq_half(const device uint8_t* w,uint p){return float(as_type<half>(iq_u16(w,p)));}\n",
        "uint iq_grid8(const constant ulong* t,uint i,uint j){return uint((t[i]>>(8*j))&255ul);}\n",
        "uint iq_grid4(const constant uint* t,uint i,uint j){return (t[i]>>(8*j))&255u;}\n",
        "float iq_sign(float v,uint signs,uint j){return ((signs>>j)&1u)!=0u ? -v:v;}\n"
    ));

    match format {
        NativeQuantizationFormat::GgufIQ2XXS => {
            iq_metal_array(&mut header, "uchar", "IQ_SIGNS", &KSIGNS_IQ2XS);
            iq_metal_array(&mut header, "ulong", "IQ_GRID", &IQ2XXS_GRID);
            header.push_str(concat!(
                "float iq_value(const device uint8_t* w,uint r,uint c){",
                "uint b=c/256u;uint x=c%256u;uint g=x/32u;uint l=(x%32u)/8u;uint j=x%8u;",
                "uint p=r+b*66u;float d=iq_half(w,p);uint q=p+2u+8u*g;",
                "uint w0=uint(iq_u16(w,q+2u*(l/2u)));",
                "uint aux=uint(iq_u16(w,q+4u))|(uint(iq_u16(w,q+6u))<<16);",
                "uint index=(l&1u)==0u?(w0&255u):(w0>>8);",
                "float db=d*(0.5f+float(aux>>28))*0.25f;",
                "return iq_sign(db*float(iq_grid8(IQ_GRID,index,j)),uint(IQ_SIGNS[(aux>>(7u*l))&127u]),j);}\n"
            ));
        }
        NativeQuantizationFormat::GgufIQ2XS => {
            iq_metal_array(&mut header, "uchar", "IQ_SIGNS", &KSIGNS_IQ2XS);
            iq_metal_array(&mut header, "ulong", "IQ_GRID", &IQ2XS_GRID);
            header.push_str(concat!(
                "float iq_value(const device uint8_t* w,uint r,uint c){",
                "uint b=c/256u;uint x=c%256u;uint g=x/32u;uint l=(x%32u)/8u;uint j=x%8u;",
                "uint p=r+b*74u;float d=iq_half(w,p);uint q=uint(iq_u16(w,p+2u+2u*(4u*g+l)));",
                "uint s=uint(w[p+66u+g]);uint nib=(l/2u)==0u?(s&15u):(s>>4);",
                "float db=d*(0.5f+float(nib))*0.25f;",
                "return iq_sign(db*float(iq_grid8(IQ_GRID,q&511u,j)),uint(IQ_SIGNS[q>>9]),j);}\n"
            ));
        }
        NativeQuantizationFormat::GgufIQ2S => {
            iq_metal_array(&mut header, "ulong", "IQ_GRID", &IQ2S_GRID);
            header.push_str(concat!(
                "float iq_value(const device uint8_t* w,uint r,uint c){",
                "uint b=c/256u;uint x=c%256u;uint g=x/32u;uint l=(x%32u)/8u;uint j=x%8u;",
                "uint p=r+b*82u;float d=iq_half(w,p);uint qh=uint(w[p+66u+g]);",
                "uint index=uint(w[p+2u+4u*g+l])|((qh<<(8u-2u*l))&0x300u);",
                "uint s=uint(w[p+74u+g]);uint nib=(l/2u)==0u?(s&15u):(s>>4);",
                "float db=d*(0.5f+float(nib))*0.25f;",
                "return iq_sign(db*float(iq_grid8(IQ_GRID,index,j)),uint(w[p+34u+4u*g+l]),j);}\n"
            ));
        }
        NativeQuantizationFormat::GgufIQ3XXS => {
            iq_metal_array(&mut header, "uchar", "IQ_SIGNS", &KSIGNS_IQ2XS);
            iq_metal_array(&mut header, "uint", "IQ_GRID", &IQ3XXS_GRID);
            header.push_str(concat!(
                "float iq_value(const device uint8_t* w,uint r,uint c){",
                "uint b=c/256u;uint x=c%256u;uint g=x/32u;uint l=(x%32u)/8u;uint j=x%8u;",
                "uint p=r+b*98u;float d=iq_half(w,p);uint aux=iq_u32(w,p+66u+4u*g);",
                "uint qi=uint(w[p+2u+8u*g+2u*l+(j/4u)]);",
                "float db=d*(0.5f+float(aux>>28))*0.5f;",
                "return iq_sign(db*float(iq_grid4(IQ_GRID,qi,j%4u)),uint(IQ_SIGNS[(aux>>(7u*l))&127u]),j);}\n"
            ));
        }
        NativeQuantizationFormat::GgufIQ3S => {
            iq_metal_array(&mut header, "uint", "IQ_GRID", &IQ3S_GRID);
            header.push_str(concat!(
                "float iq_value(const device uint8_t* w,uint r,uint c){",
                "uint b=c/256u;uint x=c%256u;uint pair=x/64u;uint side=(x%64u)/32u;",
                "uint l=(x%32u)/8u;uint j=x%8u;uint p=r+b*110u;float d=iq_half(w,p);",
                "uint scale=uint(w[p+106u+pair]);uint nib=side==0u?(scale&15u):(scale>>4);",
                "float db=d*float(1u+2u*nib);uint high=uint(w[p+66u+2u*pair+side]);",
                "uint qoff=p+2u+16u*pair+8u*side+2u*l+(j/4u);",
                "uint index=uint(w[qoff])|((high<<(8u-2u*l-(j/4u)))&256u);",
                "uint signs=uint(w[p+74u+8u*pair+4u*side+l]);",
                "return iq_sign(db*float(iq_grid4(IQ_GRID,index,j%4u)),signs,j);}\n"
            ));
        }
        NativeQuantizationFormat::GgufIQ1S => {
            iq_metal_array(&mut header, "ulong", "IQ_GRID", &IQ1S_GRID);
            header.push_str(concat!(
                "float iq_value(const device uint8_t* w,uint r,uint c){",
                "uint b=c/256u;uint x=c%256u;uint g=x/32u;uint l=(x%32u)/8u;uint j=x%8u;",
                "uint p=r+b*50u;float d=iq_half(w,p);uint hi=uint(iq_u16(w,p+34u+2u*g));",
                "float dl=d*float(2u*((hi>>12)&7u)+1u);float delta=(hi&0x8000u)!=0u?-0.125f:0.125f;",
                "uint index=uint(w[p+2u+4u*g+l])|(((hi>>(3u*l))&7u)<<8);",
                "int q=int(as_type<char>(uchar(iq_grid8(IQ_GRID,index,j))));return dl*(float(q)+delta);}\n"
            ));
        }
        NativeQuantizationFormat::GgufIQ1M => {
            iq_metal_array(&mut header, "ulong", "IQ_GRID", &IQ1S_GRID);
            header.push_str(concat!(
                "float iq_value(const device uint8_t* w,uint r,uint c){",
                "uint b=c/256u;uint x=c%256u;uint g=x/32u;uint l=(x%32u)/8u;uint j=x%8u;",
                "uint p=r+b*56u;uint s0=uint(iq_u16(w,p+48u));uint s1=uint(iq_u16(w,p+50u));",
                "uint s2=uint(iq_u16(w,p+52u));uint s3=uint(iq_u16(w,p+54u));",
                "uint dbits=(s0>>12)|((s1>>8)&0xf0u)|((s2>>4)&0xf00u)|(s3&0xf000u);",
                "float d=float(as_type<half>(ushort(dbits)));uint sw=g<2u?s0:(g<4u?s1:(g<6u?s2:s3));",
                "uint shift=6u*(g&1u)+3u*(l/2u);float dl=d*float(2u*((sw>>shift)&7u)+1u);",
                "uint h=uint(w[p+32u+2u*g+l/2u]);uint index=uint(w[p+4u*g+l])|",
                "(((h>>((l&1u)*4u))&7u)<<8);uint negbit=(l&1u)==0u?8u:128u;",
                "float delta=(h&negbit)!=0u?-0.125f:0.125f;",
                "int q=int(as_type<char>(uchar(iq_grid8(IQ_GRID,index,j))));return dl*(float(q)+delta);}\n"
            ));
        }
        NativeQuantizationFormat::GgufIQ4NL => {
            iq_metal_array(&mut header, "uchar", "IQ4", &KVALUES_IQ4NL);
            header.push_str(concat!(
                "float iq_value(const device uint8_t* w,uint r,uint c){",
                "uint b=c/32u;uint x=c%32u;uint p=r+b*18u;uint q=uint(w[p+2u+(x%16u)]);",
                "uint code=x<16u?(q&15u):(q>>4);return iq_half(w,p)*float(as_type<char>(IQ4[code]));}\n"
            ));
        }
        NativeQuantizationFormat::GgufIQ4XS => {
            iq_metal_array(&mut header, "uchar", "IQ4", &KVALUES_IQ4NL);
            header.push_str(concat!(
                "float iq_value(const device uint8_t* w,uint r,uint c){",
                "uint b=c/256u;uint x=c%256u;uint g=x/32u;uint z=x%32u;uint p=r+b*136u;",
                "uint sh=uint(iq_u16(w,p+2u));uint sl=uint(w[p+4u+g/2u]);",
                "uint low=(sl>>(4u*(g&1u)))&15u;uint high=(sh>>(2u*g))&3u;",
                "float dl=iq_half(w,p)*float(int(low|(high<<4))-32);",
                "uint q=uint(w[p+8u+16u*g+(z%16u)]);uint code=z<16u?(q&15u):(q>>4);",
                "return dl*float(as_type<char>(IQ4[code]));}\n"
            ));
        }
        _ => unreachable!("IQ Metal header requested for non-IQ format"),
    }
    header
}

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

fn decode_q8_0_view(raw: &[u8], view: &NativeQuantizedTensor) -> Result<Vec<f32>, Exception> {
    let blocks = view.columns as usize / Q8_0_BLOCK_VALUES as usize;
    let matrix_stride = view.physical_rows as usize * blocks * Q8_0_BLOCK_BYTES as usize;
    let expected = view.matrix_count as usize * matrix_stride;
    if raw.len() != expected {
        return Err(Exception::custom(format!(
            "native Q8_0 storage has {} bytes, expected {expected}",
            raw.len()
        )));
    }
    let mut output =
        Vec::with_capacity(view.matrix_count as usize * view.rows as usize * view.columns as usize);
    for matrix in 0..view.matrix_count as usize {
        for logical_row in 0..view.rows as usize {
            let physical_row = view.row_start as usize + logical_row;
            let row_base =
                matrix * matrix_stride + physical_row * blocks * Q8_0_BLOCK_BYTES as usize;
            for block in 0..blocks {
                decode_q8_0_block(
                    &raw[row_base + block * 34..row_base + (block + 1) * 34],
                    &mut output,
                );
            }
        }
    }
    Ok(output)
}

fn decode_q8_0_block(block: &[u8], output: &mut Vec<f32>) {
    let d = half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
    output.extend(block[2..].iter().map(|&quant| d * f32::from(quant as i8)));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::indexing::TryIndexOp;

    fn unhex(value: &str) -> Vec<u8> {
        value
            .as_bytes()
            .chunks_exact(2)
            .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
            .collect()
    }

    #[test]
    fn every_iq_format_executes_direct_packed_linear_embedding_and_grouped() {
        let stream = crate::test_stream();
        for line in
            include_str!("../../safemlx-gguf/tests/fixtures/llama-c0bc8591-iq.oracle").lines()
        {
            let mut fields = line.split('|');
            let ty = GgmlType::from_code(fields.next().unwrap().parse().unwrap());
            let all_raw = unhex(fields.next().unwrap());
            let _oracle_f16 = fields.next().unwrap();
            let (block_values, block_bytes) = ty.block_and_bytes().unwrap();
            let raw = &all_raw[..block_bytes as usize];
            let canonical = safemlx_gguf::IQuantTensor {
                shape: vec![1, block_values],
                ggml_type: ty,
                endian: GgufEndian::Little,
                data: raw.to_vec(),
            }
            .dequantize_f32()
            .unwrap();
            let packed = Array::from_slice(raw, &[1, block_bytes as i32])
                .copy(stream)
                .unwrap();
            let native = NativeQuantizedTensor::from_iq_array(
                packed,
                &[1, block_values as i32],
                ty,
                GgufEndian::Little,
            )
            .unwrap();

            let input = (0..block_values)
                .map(|index| ((index % 17) as f32 - 8.0) / 16.0)
                .collect::<Vec<_>>();
            let expected = input
                .iter()
                .zip(&canonical)
                .map(|(lhs, rhs)| lhs * rhs)
                .sum::<f32>();
            let actual = native
                .linear(
                    &Array::from_slice(&input, &[1, block_values as i32]),
                    true,
                    stream,
                )
                .unwrap();
            eval([&actual]).unwrap();
            let actual = actual.evaluated().unwrap().as_slice::<f32>()[0];
            let tolerance = 2e-4 * expected.abs().max(1.0);
            assert!(
                (actual - expected).abs() <= tolerance,
                "{ty:?}: packed linear {actual} != {expected}"
            );

            let prefill_values = (0..9)
                .flat_map(|row| input.iter().map(move |value| value + row as f32 * 0.01))
                .collect::<Vec<_>>();
            let prefill = Array::from_slice(&prefill_values, &[9, block_values as i32]);
            let actual = native.linear(&prefill, true, stream).unwrap();
            let dense = Array::from_slice(&canonical, &[1, block_values as i32]);
            let expected = matmul(&prefill, dense.transpose(stream).unwrap(), stream).unwrap();
            assert!(actual
                .all_close(&expected, Some(3e-4), Some(3e-4), None, stream)
                .unwrap()
                .item::<bool>(stream));

            let embedded = native
                .embedding(&Array::from_slice(&[0i32], &[1]), stream)
                .unwrap();
            eval([&embedded]).unwrap();
            let evaluated = embedded.evaluated().unwrap();
            let actual = evaluated.as_slice::<f32>();
            for (index, (&actual, &expected)) in actual.iter().zip(&canonical).enumerate() {
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "{ty:?} embedding element {index}"
                );
            }

            let bank = Array::from_slice(&all_raw, &[2, 1, block_bytes as i32])
                .copy(stream)
                .unwrap();
            let bank = NativeQuantizedTensor::from_iq_array(
                bank,
                &[2, 1, block_values as i32],
                ty,
                GgufEndian::Little,
            )
            .unwrap();
            let grouped_input = Array::from_slice(
                &[input.as_slice(), input.as_slice()].concat(),
                &[2, block_values as i32],
            );
            let grouped = native_grouped_linear(
                &grouped_input,
                &bank,
                &Array::from_slice(&[1i32, 0], &[2]),
                stream,
            )
            .unwrap();
            let dense = bank.dequantize(stream).unwrap();
            let selected = dense
                .try_index_device(&Array::from_slice(&[1i32, 0], &[2]), stream)
                .unwrap();
            let expected = matmul(
                grouped_input
                    .reshape(&[2, 1, block_values as i32], stream)
                    .unwrap(),
                selected.swap_axes(-1, -2, stream).unwrap(),
                stream,
            )
            .unwrap()
            .reshape(&[2, 1], stream)
            .unwrap();
            assert!(grouped
                .all_close(&expected, Some(2e-4), Some(2e-4), None, stream)
                .unwrap()
                .item::<bool>(stream));

            let intermediate = block_values as i32;
            let gate_up_raw = raw.repeat(4 * block_values as usize);
            let gate_up = NativeQuantizedTensor::from_iq_array(
                Array::from_slice(&gate_up_raw, &[2, 2 * intermediate, block_bytes as i32])
                    .copy(stream)
                    .unwrap(),
                &[2, 2 * intermediate, block_values as i32],
                ty,
                GgufEndian::Little,
            )
            .unwrap();
            let down_raw = raw.repeat(2);
            let down = NativeQuantizedTensor::from_iq_array(
                Array::from_slice(&down_raw, &[2, 1, block_bytes as i32])
                    .copy(stream)
                    .unwrap(),
                &[2, 1, block_values as i32],
                ty,
                GgufEndian::Little,
            )
            .unwrap();
            let ids = Array::from_slice(&[1i32, 0], &[2]);
            let route_weights = Array::from_slice(&[0.25f32, 0.75], &[2]);
            let hidden = Array::from_slice(&input, &[1, block_values as i32]);
            let activated =
                native_selected_gate_up(&hidden, &gate_up, &ids, intermediate, stream).unwrap();
            let actual =
                native_selected_down_reduce(&activated, &down, &ids, &route_weights, stream)
                    .unwrap();
            let selected_gate_up = gate_up
                .dequantize(stream)
                .unwrap()
                .try_index_device(&ids, stream)
                .unwrap();
            let repeated_hidden =
                crate::ops::broadcast_to(&hidden, &[2, block_values as i32], stream).unwrap();
            let gate = matmul(
                repeated_hidden.reshape(&[2, 1, -1], stream).unwrap(),
                selected_gate_up
                    .try_index_device((.., ..intermediate, ..), stream)
                    .unwrap()
                    .swap_axes(-1, -2, stream)
                    .unwrap(),
                stream,
            )
            .unwrap()
            .reshape(&[2, intermediate], stream)
            .unwrap();
            let up = matmul(
                repeated_hidden.reshape(&[2, 1, -1], stream).unwrap(),
                selected_gate_up
                    .try_index_device((.., intermediate.., ..), stream)
                    .unwrap()
                    .swap_axes(-1, -2, stream)
                    .unwrap(),
                stream,
            )
            .unwrap()
            .reshape(&[2, intermediate], stream)
            .unwrap();
            let activated_ref = crate::nn::gelu_approximate(gate, stream)
                .unwrap()
                .multiply(up, stream)
                .unwrap();
            let activated_close = activated
                .all_close(&activated_ref, Some(4e-4), Some(4e-4), None, stream)
                .unwrap()
                .item::<bool>(stream);
            if !activated_close {
                eval([&activated, &activated_ref]).unwrap();
                let actual_values = activated.evaluated().unwrap();
                let actual_values = actual_values.as_slice::<f32>();
                let expected_values = activated_ref.evaluated().unwrap();
                let expected_values = expected_values.as_slice::<f32>();
                let mismatch =
                    actual_values
                        .iter()
                        .zip(expected_values)
                        .position(|(&actual, &expected)| {
                            (actual - expected).abs() > 4e-4 * expected.abs().max(1.0)
                        });
                panic!(
                    "{ty:?} fused gate/up mismatch {mismatch:?}: actual {:?}, expected {:?}",
                    mismatch.map(|index| actual_values[index]),
                    mismatch.map(|index| expected_values[index]),
                );
            }
            let selected_down = down
                .dequantize(stream)
                .unwrap()
                .try_index_device(&ids, stream)
                .unwrap();
            let projected = matmul(
                activated_ref
                    .reshape(&[2, 1, intermediate], stream)
                    .unwrap(),
                selected_down.swap_axes(-1, -2, stream).unwrap(),
                stream,
            )
            .unwrap()
            .reshape(&[2, 1], stream)
            .unwrap();
            let expected = sum_axis(
                projected
                    .multiply(route_weights.reshape(&[2, 1], stream).unwrap(), stream)
                    .unwrap(),
                0,
                true,
                stream,
            )
            .unwrap();
            let close = actual
                .all_close(&expected, Some(4e-4), Some(4e-4), None, stream)
                .unwrap()
                .item::<bool>(stream);
            if !close {
                eval([&actual, &expected]).unwrap();
                panic!(
                    "{ty:?} fused expert output {:?} != {:?}",
                    actual.evaluated().unwrap().as_slice::<f32>(),
                    expected.evaluated().unwrap().as_slice::<f32>()
                );
            }
        }
    }

    #[test]
    #[ignore = "requires an accessible Metal device"]
    fn every_iq_format_executes_batched_and_fused_metal_kernels() {
        let stream = crate::Stream::new_with_device(&crate::Device::new(DeviceType::Gpu, 0));
        for line in
            include_str!("../../safemlx-gguf/tests/fixtures/llama-c0bc8591-iq.oracle").lines()
        {
            let mut fields = line.split('|');
            let ty = GgmlType::from_code(fields.next().unwrap().parse().unwrap());
            let all_raw = unhex(fields.next().unwrap());
            let _oracle_f16 = fields.next().unwrap();
            let (block_values, block_bytes) = ty.block_and_bytes().unwrap();
            let raw = &all_raw[..block_bytes as usize];
            let input_values = (0..9 * block_values as usize)
                .map(|index| ((index % 17) as f32 - 8.0) / 16.0)
                .collect::<Vec<_>>();
            let input = Array::from_slice(&input_values, &[9, block_values as i32])
                .copy(&stream)
                .unwrap();
            let native = NativeQuantizedTensor::from_iq_array(
                Array::from_slice(raw, &[1, block_bytes as i32])
                    .copy(&stream)
                    .unwrap(),
                &[1, block_values as i32],
                ty,
                GgufEndian::Little,
            )
            .unwrap();
            let dense = native.dequantize(&stream).unwrap();
            let actual = native.linear(&input, true, &stream).unwrap();
            let expected = matmul(&input, dense.transpose(&stream).unwrap(), &stream).unwrap();
            assert!(
                actual
                    .all_close(&expected, Some(4e-4), Some(4e-4), None, &stream)
                    .unwrap()
                    .item::<bool>(&stream),
                "{ty:?} batched Metal linear"
            );

            let intermediate = block_values as i32;
            let gate_up = NativeQuantizedTensor::from_iq_array(
                Array::from_slice(
                    &raw.repeat(4 * block_values as usize),
                    &[2, 2 * intermediate, block_bytes as i32],
                )
                .copy(&stream)
                .unwrap(),
                &[2, 2 * intermediate, block_values as i32],
                ty,
                GgufEndian::Little,
            )
            .unwrap();
            let down = NativeQuantizedTensor::from_iq_array(
                Array::from_slice(&raw.repeat(2), &[2, 1, block_bytes as i32])
                    .copy(&stream)
                    .unwrap(),
                &[2, 1, block_values as i32],
                ty,
                GgufEndian::Little,
            )
            .unwrap();
            let ids = Array::from_slice(&[1i32, 0], &[2]).copy(&stream).unwrap();
            let route_weights = Array::from_slice(&[0.25f32, 0.75], &[2])
                .copy(&stream)
                .unwrap();
            let hidden = Array::from_slice(
                &input_values[..block_values as usize]
                    .iter()
                    .map(|value| value / 1024.0)
                    .collect::<Vec<_>>(),
                &[1, block_values as i32],
            )
            .copy(&stream)
            .unwrap();
            let activated =
                native_selected_gate_up(&hidden, &gate_up, &ids, intermediate, &stream).unwrap();
            let actual =
                native_selected_down_reduce(&activated, &down, &ids, &route_weights, &stream)
                    .unwrap();

            let selected_gate_up = gate_up
                .dequantize(&stream)
                .unwrap()
                .try_index_device(&ids, &stream)
                .unwrap();
            let repeated =
                crate::ops::broadcast_to(&hidden, &[2, block_values as i32], &stream).unwrap();
            let gate = matmul(
                repeated.reshape(&[2, 1, -1], &stream).unwrap(),
                selected_gate_up
                    .try_index_device((.., ..intermediate, ..), &stream)
                    .unwrap()
                    .swap_axes(-1, -2, &stream)
                    .unwrap(),
                &stream,
            )
            .unwrap()
            .reshape(&[2, intermediate], &stream)
            .unwrap();
            let up = matmul(
                repeated.reshape(&[2, 1, -1], &stream).unwrap(),
                selected_gate_up
                    .try_index_device((.., intermediate.., ..), &stream)
                    .unwrap()
                    .swap_axes(-1, -2, &stream)
                    .unwrap(),
                &stream,
            )
            .unwrap()
            .reshape(&[2, intermediate], &stream)
            .unwrap();
            let activated_ref = crate::nn::gelu_approximate(gate, &stream)
                .unwrap()
                .multiply(up, &stream)
                .unwrap();
            let gate_up_close = activated
                .all_close(&activated_ref, Some(6e-4), Some(6e-4), None, &stream)
                .unwrap()
                .item::<bool>(&stream);
            if !gate_up_close {
                eval([&activated, &activated_ref]).unwrap();
                let actual_evaluated = activated.evaluated().unwrap();
                let actual_values = actual_evaluated.as_slice::<f32>();
                let expected_evaluated = activated_ref.evaluated().unwrap();
                let expected_values = expected_evaluated.as_slice::<f32>();
                let mismatch =
                    actual_values
                        .iter()
                        .zip(expected_values)
                        .position(|(&actual, &expected)| {
                            (actual - expected).abs() > 6e-4 * expected.abs().max(1.0)
                        });
                panic!(
                    "{ty:?} fused Metal gate/up mismatch {mismatch:?}: {:?} != {:?}",
                    mismatch.map(|index| actual_values[index]),
                    mismatch.map(|index| expected_values[index])
                );
            }
            let selected_down = down
                .dequantize(&stream)
                .unwrap()
                .try_index_device(&ids, &stream)
                .unwrap();
            let projected = matmul(
                activated_ref.reshape(&[2, 1, -1], &stream).unwrap(),
                selected_down.swap_axes(-1, -2, &stream).unwrap(),
                &stream,
            )
            .unwrap()
            .reshape(&[2, 1], &stream)
            .unwrap();
            let expected = sum_axis(
                projected
                    .multiply(route_weights.reshape(&[2, 1], &stream).unwrap(), &stream)
                    .unwrap(),
                0,
                true,
                &stream,
            )
            .unwrap();
            assert!(
                actual
                    .all_close(&expected, Some(6e-4), Some(6e-4), None, &stream)
                    .unwrap()
                    .item::<bool>(&stream),
                "{ty:?} fused Metal down/reduce"
            );
        }
    }

    #[test]
    #[ignore = "requires an accessible Metal device"]
    fn iq_metal_execution_honors_big_endian_block_fields() {
        let stream = crate::Stream::new_with_device(&crate::Device::new(DeviceType::Gpu, 0));
        for ty in [GgmlType::IQ4NL, GgmlType::IQ2XS] {
            let (values, bytes) = ty.block_and_bytes().unwrap();
            let mut little = (0..bytes)
                .map(|index| (index * 29 + 7) as u8)
                .collect::<Vec<_>>();
            little[..2].copy_from_slice(&half::f16::from_f32(0.75).to_bits().to_le_bytes());
            let mut big = little.clone();
            match ty {
                GgmlType::IQ4NL => big[..2].reverse(),
                GgmlType::IQ2XS => {
                    for pair in big[..66].chunks_exact_mut(2) {
                        pair.reverse();
                    }
                }
                _ => unreachable!(),
            }
            let input = Array::from_slice(&vec![1.0f32; values as usize], &[1, values as i32])
                .copy(&stream)
                .unwrap();
            let execute = |raw: &[u8], endian| {
                let packed = Array::from_slice(raw, &[1, bytes as i32])
                    .copy(&stream)
                    .unwrap();
                let native =
                    NativeQuantizedTensor::from_iq_array(packed, &[1, values as i32], ty, endian)
                        .unwrap();
                let output = native.linear(&input, true, &stream).unwrap();
                eval([&output]).unwrap();
                output.evaluated().unwrap().as_slice::<f32>()[0]
            };
            assert_eq!(
                execute(&little, GgufEndian::Little).to_bits(),
                execute(&big, GgufEndian::Big).to_bits(),
                "{ty:?}"
            );
        }
    }

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

    fn sample_q8_0_block() -> Vec<u8> {
        let mut block = vec![0u8; 34];
        block[0..2].copy_from_slice(&half::f16::from_f32(0.125).to_bits().to_le_bytes());
        for (index, value) in block[2..].iter_mut().enumerate() {
            *value = (index as i8).wrapping_mul(11).wrapping_sub(97) as u8;
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
    fn native_view_copy_preserves_logical_slice() {
        let source = crate::Stream::new_with_device(&crate::Device::new(DeviceType::Cpu, 0));
        let destination = crate::Stream::new_with_device(&crate::Device::new(DeviceType::Cpu, 0));
        let mut raw = Vec::new();
        for _ in 0..8 {
            raw.extend(sample_block());
        }
        let source_view = NativeQuantizedTensor::from_q4k_bytes(&raw, &[2, 4, 256], &source)
            .unwrap()
            .row_view(1, 2)
            .unwrap();
        let copied = source_view.copy_to_stream(&destination).unwrap();

        assert!(!Arc::ptr_eq(source_view.storage(), copied.storage()));
        assert_eq!(copied.shape(), source_view.shape());
        assert_eq!(copied.row_start(), source_view.row_start());
        assert_eq!(copied.physical_rows(), source_view.physical_rows());
        assert_eq!(
            copied
                .dequantize(&destination)
                .unwrap()
                .evaluated()
                .unwrap()
                .as_slice::<f32>(),
            source_view
                .dequantize(&source)
                .unwrap()
                .evaluated()
                .unwrap()
                .as_slice::<f32>()
        );
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
        let dense = matrix.dequantize(&stream).unwrap();
        let input = Array::from_slice(&vec![0.5f32; 512], &[2, 256]);
        let output = matrix.linear(&input, true, &stream).unwrap();
        assert_eq!(output.shape(), &[2, 2]);
        let expected = matmul(&input, dense.transpose(&stream).unwrap(), &stream).unwrap();
        assert!(output
            .all_close(&expected, Some(1e-5), Some(1e-5), None, &stream)
            .unwrap()
            .item::<bool>(&stream));
        let untransposed_input = Array::from_slice(&[0.25f32, -0.5], &[1, 2]);
        let untransposed = matrix.linear(&untransposed_input, false, &stream).unwrap();
        assert_eq!(untransposed.shape(), &[1, 256]);
        let expected = matmul(&untransposed_input, &dense, &stream).unwrap();
        assert!(untransposed
            .all_close(&expected, Some(1e-5), Some(1e-5), None, &stream)
            .unwrap()
            .item::<bool>(&stream));
        let ids = Array::from_slice(&[1i32, 0], &[2]);
        let embedded = matrix.embedding(&ids, &stream).unwrap();
        assert_eq!(embedded.shape(), &[2, 256]);
        let expected = dense.try_index_device(&ids, &stream).unwrap();
        assert!(embedded
            .all_close(&expected, Some(1e-6), Some(1e-6), None, &stream)
            .unwrap()
            .item::<bool>(&stream));

        let mut expert_raw = Vec::new();
        for _ in 0..4 {
            expert_raw.extend(sample_block());
        }
        let experts =
            NativeQuantizedTensor::from_q4k_bytes(&expert_raw, &[2, 2, 256], &stream).unwrap();
        let group_ids = Array::from_slice(&[1i32, 0], &[2]);
        let grouped = native_grouped_linear(&input, &experts, &group_ids, &stream).unwrap();
        assert_eq!(grouped.shape(), &[2, 2]);
        let selected = experts
            .dequantize(&stream)
            .unwrap()
            .try_index_device(&group_ids, &stream)
            .unwrap();
        let expected = matmul(
            input.reshape(&[2, 1, 256], &stream).unwrap(),
            selected.swap_axes(-1, -2, &stream).unwrap(),
            &stream,
        )
        .unwrap()
        .reshape(&[2, 2], &stream)
        .unwrap();
        assert!(grouped
            .all_close(&expected, Some(1e-5), Some(1e-5), None, &stream)
            .unwrap()
            .item::<bool>(&stream));
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

    #[test]
    fn q5_1_direct_linear_prefill_and_embedding_match_dequantized_reference() {
        let stream = crate::test_stream();
        let mut raw = sample_q5_1_block();
        let mut second = sample_q5_1_block();
        second[8] ^= 0x5a;
        raw.extend(second);
        let native = NativeQuantizedTensor::from_q5_1_bytes(&raw, &[2, 32], stream).unwrap();
        let input_values = (0..9 * 32)
            .map(|index| (index as f32 % 29.0 - 14.0) / 20.0)
            .collect::<Vec<_>>();
        let input = Array::from_slice(&input_values, &[9, 32]);
        let actual = native.linear(&input, true, stream).unwrap();
        let dense = native.dequantize(stream).unwrap();
        let expected = matmul(&input, dense.transpose(stream).unwrap(), stream).unwrap();
        assert!(actual
            .all_close(&expected, Some(2e-4), Some(2e-4), None, stream)
            .unwrap()
            .item::<bool>(stream));

        let ids = Array::from_slice(&[1i32, 0, 1], &[3]);
        let actual = native.embedding(&ids, stream).unwrap();
        let expected = dense.try_index_device(&ids, stream).unwrap();
        assert!(actual
            .all_close(&expected, Some(1e-6), Some(1e-6), None, stream)
            .unwrap()
            .item::<bool>(stream));
    }

    #[test]
    #[ignore = "requires an accessible Metal device"]
    fn q5_1_metal_linear_prefill_and_embedding_match_dequantized_reference() {
        let stream = crate::Stream::new_with_device(&crate::Device::new(DeviceType::Gpu, 0));
        let mut raw = sample_q5_1_block();
        let mut second = sample_q5_1_block();
        second[8] ^= 0x5a;
        raw.extend(second);
        let native = NativeQuantizedTensor::from_q5_1_bytes(&raw, &[2, 32], &stream).unwrap();
        let input = Array::from_slice(
            &(0..9 * 32)
                .map(|index| (index as f32 % 29.0 - 14.0) / 20.0)
                .collect::<Vec<_>>(),
            &[9, 32],
        )
        .copy(&stream)
        .unwrap();
        let dense = native.dequantize(&stream).unwrap();
        let actual = native.linear(&input, true, &stream).unwrap();
        let expected = matmul(&input, dense.transpose(&stream).unwrap(), &stream).unwrap();
        assert!(actual
            .all_close(&expected, Some(2e-4), Some(2e-4), None, &stream)
            .unwrap()
            .item::<bool>(&stream));

        let ids = Array::from_slice(&[1i32, 0, 1], &[3])
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
    fn q8_0_decode_and_cpu_operations_match_affine_reference() {
        let stream = crate::Stream::new_with_device(&crate::Device::new(DeviceType::Cpu, 0));
        let mut raw = Vec::new();
        raw.extend(sample_q8_0_block());
        raw.extend(sample_q8_0_block());
        let native = NativeQuantizedTensor::from_q8_0_bytes(&raw, &[2, 32], &stream).unwrap();
        let actual = native.dequantize(&stream).unwrap();
        let actual = actual.evaluated().unwrap();

        let mut file = Vec::new();
        safemlx_gguf::Writer::default()
            .write(
                std::io::Cursor::new(&mut file),
                &std::collections::BTreeMap::new(),
                &[safemlx_gguf::TensorInput {
                    name: "test.weight",
                    dimensions: &[32, 2],
                    ggml_type: safemlx_gguf::GgmlType::Q8_0,
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

        let input = Array::from_slice(&vec![0.01f32; 3 * 32], &[3, 32]);
        assert_eq!(
            native.linear(&input, true, &stream).unwrap().shape(),
            &[3, 2]
        );
        let ids = Array::from_slice(&[1i32, 0], &[2]);
        assert_eq!(native.embedding(&ids, &stream).unwrap().shape(), &[2, 32]);
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

        let ids = Array::from_slice(&[4u32, 1, 4], &[3])
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
    fn q8_0_metal_linear_prefill_embedding_and_partial_tiles_match_float() {
        let stream = crate::Stream::new_with_device(&crate::Device::new(DeviceType::Gpu, 0));
        let mut raw = Vec::new();
        for index in 0..(5 * 3) {
            let mut block = sample_q8_0_block();
            block[2] = block[2].wrapping_add(index as u8);
            raw.extend(block);
        }
        let native = NativeQuantizedTensor::from_q8_0_bytes(&raw, &[5, 96], &stream).unwrap();
        let input = Array::from_slice(
            &(0..7 * 96)
                .map(|index| (index as f32 % 37.0 - 18.0) / 190.0)
                .collect::<Vec<_>>(),
            &[7, 96],
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
        for dtype in [Dtype::Float16, Dtype::Bfloat16] {
            let typed_input = input.as_dtype(dtype, &stream).unwrap();
            let typed_dense = dense.as_dtype(dtype, &stream).unwrap();
            let actual = native.linear(&typed_input, true, &stream).unwrap();
            let expected = matmul(
                &typed_input,
                typed_dense.transpose(&stream).unwrap(),
                &stream,
            )
            .unwrap();
            assert!(
                actual
                    .all_close(&expected, Some(2e-2), Some(2e-2), None, &stream)
                    .unwrap()
                    .item::<bool>(&stream),
                "Q8_0 {dtype:?} linear disagrees with float reference"
            );
        }

        let decode = input.try_index_device((0..1, ..), &stream).unwrap();
        let actual = native.linear(&decode, true, &stream).unwrap();
        let expected = matmul(&decode, dense.transpose(&stream).unwrap(), &stream).unwrap();
        assert!(actual
            .all_close(&expected, Some(2e-3), Some(2e-3), None, &stream)
            .unwrap()
            .item::<bool>(&stream));

        let ids = Array::from_slice(&[4u32, 1, 4], &[3])
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
    fn q8_0_metal_grouped_and_selected_down_match_float_with_repeated_ids() {
        let stream = crate::Stream::new_with_device(&crate::Device::new(DeviceType::Gpu, 0));
        let experts = 3;
        let output = 5;
        let input_dim = 64;
        let routes = 4;
        let mut raw = Vec::new();
        for index in 0..(experts * output * input_dim / 32) {
            let mut block = sample_q8_0_block();
            block[2] = block[2].wrapping_add(index as u8);
            raw.extend(block);
        }
        let native =
            NativeQuantizedTensor::from_q8_0_bytes(&raw, &[experts, output, input_dim], &stream)
                .unwrap();
        let input = Array::from_slice(
            &(0..routes * input_dim)
                .map(|index| (index as f32 % 37.0 - 18.0) / 190.0)
                .collect::<Vec<_>>(),
            &[routes, input_dim],
        )
        .copy(&stream)
        .unwrap();
        let ids = Array::from_slice(&[2i32, 0, 2, 1], &[routes])
            .copy(&stream)
            .unwrap();
        let weights = Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4], &[routes])
            .copy(&stream)
            .unwrap();

        let selected = native
            .dequantize(&stream)
            .unwrap()
            .try_index_device(&ids, &stream)
            .unwrap();
        let expected = matmul(
            input.reshape(&[-1, 1, input_dim], &stream).unwrap(),
            selected.swap_axes(-1, -2, &stream).unwrap(),
            &stream,
        )
        .unwrap()
        .reshape(&[routes, output], &stream)
        .unwrap();
        let actual = native_grouped_linear(&input, &native, &ids, &stream).unwrap();
        assert!(actual
            .all_close(&expected, Some(2e-3), Some(2e-3), None, &stream)
            .unwrap()
            .item::<bool>(&stream));

        let actual = native_selected_down_reduce(&input, &native, &ids, &weights, &stream).unwrap();
        let expected = sum_axis(
            expected
                .multiply(weights.reshape(&[-1, 1], &stream).unwrap(), &stream)
                .unwrap(),
            0,
            true,
            &stream,
        )
        .unwrap();
        assert!(actual
            .all_close(&expected, Some(2e-3), Some(2e-3), None, &stream)
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
            reference_activated
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
            activated.reshape(&[-1, 1, intermediate], &stream).unwrap(),
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
