//! Fast implementations of commonly used multi-op functions.

use std::{
    ffi::{c_char, CStr, CString},
    fmt,
};

use crate::error::{Exception, Result};
use crate::utils::guard::Guarded;
use crate::utils::{IntoOption, VectorArray, SUCCESS};
use crate::{Array, Dtype, Stream};
use safemlx_internal_macros::generate_macro;

/// A compiled custom Metal kernel.
///
/// The kernel owns the underlying MLX fast-metal handle and can be applied
/// repeatedly with different inputs and [`MetalKernelConfig`] values.
pub struct MetalKernel {
    c_kernel: safemlx_sys::mlx_fast_metal_kernel,
    name: String,
    input_names: Vec<String>,
    output_names: Vec<String>,
}

impl fmt::Debug for MetalKernel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MetalKernel")
            .field("name", &self.name)
            .field("input_names", &self.input_names)
            .field("output_names", &self.output_names)
            .finish_non_exhaustive()
    }
}

impl MetalKernel {
    /// Create a new custom Metal kernel.
    ///
    /// `input_names` and `output_names` must match the argument names used by
    /// `source` and `header`. `ensure_row_contiguous` asks MLX to make inputs
    /// row-contiguous before dispatch. `atomic_outputs` marks outputs as using
    /// atomic writes.
    pub fn new<Name, Inputs, InputName, Outputs, OutputName, Source, Header>(
        name: Name,
        input_names: Inputs,
        output_names: Outputs,
        source: Source,
        header: Header,
        ensure_row_contiguous: bool,
        atomic_outputs: bool,
    ) -> Result<Self>
    where
        Name: Into<String>,
        Inputs: IntoIterator<Item = InputName>,
        InputName: Into<String>,
        Outputs: IntoIterator<Item = OutputName>,
        OutputName: Into<String>,
        Source: Into<String>,
        Header: Into<String>,
    {
        crate::error::ensure_mlx_error_handler();

        let name = name.into();
        let input_names: Vec<String> = input_names.into_iter().map(Into::into).collect();
        let output_names: Vec<String> = output_names.into_iter().map(Into::into).collect();
        let source = source.into();
        let header = header.into();

        let c_name = cstring(&name)?;
        let c_source = cstring(&source)?;
        let c_header = cstring(&header)?;
        let c_input_names = VectorString::try_from_strings(&input_names)?;
        let c_output_names = VectorString::try_from_strings(&output_names)?;

        let c_kernel = unsafe {
            safemlx_sys::mlx_fast_metal_kernel_new(
                c_name.as_ptr(),
                c_input_names.as_ptr(),
                c_output_names.as_ptr(),
                c_source.as_ptr(),
                c_header.as_ptr(),
                ensure_row_contiguous,
                atomic_outputs,
            )
        };

        if c_kernel.ctx.is_null() {
            let what = crate::error::get_and_clear_last_mlx_error()
                .map(|e| e.what)
                .unwrap_or_else(|| "failed to create Metal kernel".to_string());
            return Err(Exception::custom(what));
        }

        Ok(Self {
            c_kernel,
            name,
            input_names,
            output_names,
        })
    }

    /// Apply the kernel on `stream`.
    ///
    /// Returns one [`Array`] for each output declared in `config`.
    pub fn apply_device<I, A>(
        &self,
        inputs: I,
        config: &MetalKernelConfig,
        stream: impl AsRef<Stream>,
    ) -> Result<Vec<Array>>
    where
        I: IntoIterator<Item = A>,
        A: AsRef<Array>,
    {
        let inputs = VectorArray::try_from_iter(inputs.into_iter())?;
        let raw_config = RawMetalKernelConfig::try_from_config(config)?;
        let outputs = Vec::<Array>::try_from_op(|outputs| unsafe {
            safemlx_sys::mlx_fast_metal_kernel_apply(
                outputs,
                self.c_kernel,
                inputs.as_ptr(),
                raw_config.as_ptr(),
                stream.as_ref().as_ptr(),
            )
        })?;

        if outputs.len() != config.output_count() {
            return Err(Exception::custom(format!(
                "Metal kernel returned {} outputs, expected {}",
                outputs.len(),
                config.output_count()
            )));
        }

        Ok(outputs)
    }

    /// Apply the kernel on `stream` and require exactly one output.
    pub fn apply_one_device<I, A>(
        &self,
        inputs: I,
        config: &MetalKernelConfig,
        stream: impl AsRef<Stream>,
    ) -> Result<Array>
    where
        I: IntoIterator<Item = A>,
        A: AsRef<Array>,
    {
        let mut outputs = self.apply_device(inputs, config, stream)?;
        match outputs.len() {
            1 => Ok(outputs.remove(0)),
            n => Err(Exception::custom(format!(
                "Metal kernel returned {n} outputs, expected 1"
            ))),
        }
    }
}

impl Drop for MetalKernel {
    fn drop(&mut self) {
        unsafe {
            safemlx_sys::mlx_fast_metal_kernel_free(self.c_kernel);
        }
    }
}

/// Result returned by stateful recurrent kernels.
///
/// The first value is the emitted sequence, including the single-token
/// sequence produced by decode kernels. The second value is the state that
/// should be carried into the next recurrent call.
#[derive(Debug)]
pub struct StatefulKernelOutput {
    /// Emitted output sequence.
    pub output_sequence: Array,

    /// Updated recurrent state.
    pub new_state: Array,
}

impl StatefulKernelOutput {
    /// Create a stateful kernel output from its two arrays.
    pub fn new(output_sequence: Array, new_state: Array) -> Self {
        Self {
            output_sequence,
            new_state,
        }
    }

    /// Split into `(output_sequence, new_state)`.
    pub fn into_tuple(self) -> (Array, Array) {
        (self.output_sequence, self.new_state)
    }
}

/// A custom Metal kernel that returns `(output_sequence, new_state)`.
///
/// This is a light wrapper around [`MetalKernel`] for recurrent/stateful
/// kernels where callers want the API to reflect state threading rather than
/// manually indexing a `Vec<Array>`.
pub struct StatefulMetalKernel {
    kernel: MetalKernel,
}

impl fmt::Debug for StatefulMetalKernel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StatefulMetalKernel")
            .field("kernel", &self.kernel)
            .finish()
    }
}

impl StatefulMetalKernel {
    /// Create a stateful custom Metal kernel.
    ///
    /// `output_names` should name exactly two outputs in this order:
    /// output sequence, then new recurrent state.
    pub fn new<Name, Inputs, InputName, Source, Header>(
        name: Name,
        input_names: Inputs,
        output_names: [&str; 2],
        source: Source,
        header: Header,
        ensure_row_contiguous: bool,
        atomic_outputs: bool,
    ) -> Result<Self>
    where
        Name: Into<String>,
        Inputs: IntoIterator<Item = InputName>,
        InputName: Into<String>,
        Source: Into<String>,
        Header: Into<String>,
    {
        Ok(Self {
            kernel: MetalKernel::new(
                name,
                input_names,
                output_names,
                source,
                header,
                ensure_row_contiguous,
                atomic_outputs,
            )?,
        })
    }

    /// Apply the kernel on `stream`.
    pub fn apply_device<I, A>(
        &self,
        inputs: I,
        config: &MetalKernelConfig,
        stream: impl AsRef<Stream>,
    ) -> Result<StatefulKernelOutput>
    where
        I: IntoIterator<Item = A>,
        A: AsRef<Array>,
    {
        if config.output_count() != 2 {
            return Err(Exception::custom(format!(
                "stateful kernel config declares {} outputs, expected 2",
                config.output_count()
            )));
        }

        let mut outputs = self.kernel.apply_device(inputs, config, stream)?;
        match outputs.len() {
            2 => {
                let new_state = outputs.remove(1);
                let output_sequence = outputs.remove(0);
                Ok(StatefulKernelOutput::new(output_sequence, new_state))
            }
            n => Err(Exception::custom(format!(
                "stateful kernel returned {n} outputs, expected 2"
            ))),
        }
    }
}

/// Paired recurrent scan kernels for prefill and decode.
///
/// The decode kernel handles a single recurrent step and the prefill kernel
/// scans a full sequence while carrying state internally inside the custom
/// kernel. Both kernels return [`StatefulKernelOutput`].
pub struct RecurrentScanKernel {
    decode: StatefulMetalKernel,
    prefill: StatefulMetalKernel,
}

impl fmt::Debug for RecurrentScanKernel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RecurrentScanKernel")
            .field("decode", &self.decode)
            .field("prefill", &self.prefill)
            .finish()
    }
}

impl RecurrentScanKernel {
    /// Create a recurrent scan kernel pair from decode and prefill kernels.
    pub fn new(decode: StatefulMetalKernel, prefill: StatefulMetalKernel) -> Self {
        Self { decode, prefill }
    }

    /// Apply the single-token decode kernel on `stream`.
    pub fn decode_device<I, A>(
        &self,
        inputs: I,
        config: &MetalKernelConfig,
        stream: impl AsRef<Stream>,
    ) -> Result<StatefulKernelOutput>
    where
        I: IntoIterator<Item = A>,
        A: AsRef<Array>,
    {
        self.decode.apply_device(inputs, config, stream)
    }

    /// Apply the full-sequence prefill kernel on `stream`.
    pub fn prefill_device<I, A>(
        &self,
        inputs: I,
        config: &MetalKernelConfig,
        stream: impl AsRef<Stream>,
    ) -> Result<StatefulKernelOutput>
    where
        I: IntoIterator<Item = A>,
        A: AsRef<Array>,
    {
        self.prefill.apply_device(inputs, config, stream)
    }
}

/// Output declaration for a custom Metal kernel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetalKernelOutput {
    /// Output shape.
    pub shape: Vec<i32>,

    /// Output dtype.
    pub dtype: Dtype,
}

impl MetalKernelOutput {
    /// Create an output declaration.
    pub fn new(shape: impl Into<Vec<i32>>, dtype: Dtype) -> Self {
        Self {
            shape: shape.into(),
            dtype,
        }
    }
}

/// Template argument for a custom Metal kernel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MetalKernelTemplateArg {
    /// Dtype template argument.
    Dtype {
        /// Template parameter name.
        name: String,

        /// Template dtype value.
        dtype: Dtype,
    },

    /// Integer template argument.
    Int {
        /// Template parameter name.
        name: String,

        /// Template integer value.
        value: i32,
    },

    /// Boolean template argument.
    Bool {
        /// Template parameter name.
        name: String,

        /// Template boolean value.
        value: bool,
    },
}

/// Dispatch configuration for a custom Metal kernel.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MetalKernelConfig {
    outputs: Vec<MetalKernelOutput>,
    template_args: Vec<MetalKernelTemplateArg>,
    grid: Option<[i32; 3]>,
    thread_group: Option<[i32; 3]>,
    init_value: Option<f32>,
    verbose: bool,
}

impl MetalKernelConfig {
    /// Create an empty dispatch configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Return the declared output count.
    pub fn output_count(&self) -> usize {
        self.outputs.len()
    }

    /// Return the output declarations.
    pub fn outputs(&self) -> &[MetalKernelOutput] {
        &self.outputs
    }

    /// Return the template arguments.
    pub fn template_args(&self) -> &[MetalKernelTemplateArg] {
        &self.template_args
    }

    /// Add an output declaration.
    pub fn add_output_arg(&mut self, shape: impl Into<Vec<i32>>, dtype: Dtype) -> &mut Self {
        self.outputs.push(MetalKernelOutput::new(shape, dtype));
        self
    }

    /// Add an output declaration and return the updated config.
    pub fn with_output_arg(mut self, shape: impl Into<Vec<i32>>, dtype: Dtype) -> Self {
        self.add_output_arg(shape, dtype);
        self
    }

    /// Set the dispatch grid dimensions.
    pub fn set_grid(&mut self, grid: [i32; 3]) -> &mut Self {
        self.grid = Some(grid);
        self
    }

    /// Set the dispatch grid dimensions and return the updated config.
    pub fn with_grid(mut self, grid: [i32; 3]) -> Self {
        self.set_grid(grid);
        self
    }

    /// Set the thread-group dimensions.
    pub fn set_thread_group(&mut self, thread_group: [i32; 3]) -> &mut Self {
        self.thread_group = Some(thread_group);
        self
    }

    /// Set the thread-group dimensions and return the updated config.
    pub fn with_thread_group(mut self, thread_group: [i32; 3]) -> Self {
        self.set_thread_group(thread_group);
        self
    }

    /// Set the output initialization value used by MLX.
    pub fn set_init_value(&mut self, value: f32) -> &mut Self {
        self.init_value = Some(value);
        self
    }

    /// Set the output initialization value and return the updated config.
    pub fn with_init_value(mut self, value: f32) -> Self {
        self.set_init_value(value);
        self
    }

    /// Enable or disable verbose MLX kernel logging.
    pub fn set_verbose(&mut self, verbose: bool) -> &mut Self {
        self.verbose = verbose;
        self
    }

    /// Enable or disable verbose MLX kernel logging and return the updated config.
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.set_verbose(verbose);
        self
    }

    /// Add a dtype template argument.
    pub fn add_template_arg_dtype(&mut self, name: impl Into<String>, dtype: Dtype) -> &mut Self {
        self.template_args.push(MetalKernelTemplateArg::Dtype {
            name: name.into(),
            dtype,
        });
        self
    }

    /// Add a dtype template argument and return the updated config.
    pub fn with_template_arg_dtype(mut self, name: impl Into<String>, dtype: Dtype) -> Self {
        self.add_template_arg_dtype(name, dtype);
        self
    }

    /// Add an integer template argument.
    pub fn add_template_arg_int(&mut self, name: impl Into<String>, value: i32) -> &mut Self {
        self.template_args.push(MetalKernelTemplateArg::Int {
            name: name.into(),
            value,
        });
        self
    }

    /// Add an integer template argument and return the updated config.
    pub fn with_template_arg_int(mut self, name: impl Into<String>, value: i32) -> Self {
        self.add_template_arg_int(name, value);
        self
    }

    /// Add a boolean template argument.
    pub fn add_template_arg_bool(&mut self, name: impl Into<String>, value: bool) -> &mut Self {
        self.template_args.push(MetalKernelTemplateArg::Bool {
            name: name.into(),
            value,
        });
        self
    }

    /// Add a boolean template argument and return the updated config.
    pub fn with_template_arg_bool(mut self, name: impl Into<String>, value: bool) -> Self {
        self.add_template_arg_bool(name, value);
        self
    }
}

struct RawMetalKernelConfig {
    c_config: safemlx_sys::mlx_fast_metal_kernel_config,
}

impl RawMetalKernelConfig {
    fn try_from_config(config: &MetalKernelConfig) -> Result<Self> {
        crate::error::ensure_mlx_error_handler();

        let c_config = unsafe { safemlx_sys::mlx_fast_metal_kernel_config_new() };
        if c_config.ctx.is_null() {
            let what = crate::error::get_and_clear_last_mlx_error()
                .map(|e| e.what)
                .unwrap_or_else(|| "failed to create Metal kernel config".to_string());
            return Err(Exception::custom(what));
        }

        let raw = Self { c_config };
        raw.populate(config)?;
        Ok(raw)
    }

    fn as_ptr(&self) -> safemlx_sys::mlx_fast_metal_kernel_config {
        self.c_config
    }

    fn populate(&self, config: &MetalKernelConfig) -> Result<()> {
        for output in &config.outputs {
            check_status(unsafe {
                safemlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
                    self.c_config,
                    output.shape.as_ptr(),
                    output.shape.len(),
                    output.dtype.into(),
                )
            })?;
        }

        if let Some([x, y, z]) = config.grid {
            check_status(unsafe {
                safemlx_sys::mlx_fast_metal_kernel_config_set_grid(self.c_config, x, y, z)
            })?;
        }

        if let Some([x, y, z]) = config.thread_group {
            check_status(unsafe {
                safemlx_sys::mlx_fast_metal_kernel_config_set_thread_group(self.c_config, x, y, z)
            })?;
        }

        if let Some(value) = config.init_value {
            check_status(unsafe {
                safemlx_sys::mlx_fast_metal_kernel_config_set_init_value(self.c_config, value)
            })?;
        }

        check_status(unsafe {
            safemlx_sys::mlx_fast_metal_kernel_config_set_verbose(self.c_config, config.verbose)
        })?;

        for template_arg in &config.template_args {
            match template_arg {
                MetalKernelTemplateArg::Dtype { name, dtype } => {
                    let name = cstring(name)?;
                    check_status(unsafe {
                        safemlx_sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(
                            self.c_config,
                            name.as_ptr(),
                            (*dtype).into(),
                        )
                    })?;
                }
                MetalKernelTemplateArg::Int { name, value } => {
                    let name = cstring(name)?;
                    check_status(unsafe {
                        safemlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
                            self.c_config,
                            name.as_ptr(),
                            *value,
                        )
                    })?;
                }
                MetalKernelTemplateArg::Bool { name, value } => {
                    let name = cstring(name)?;
                    check_status(unsafe {
                        safemlx_sys::mlx_fast_metal_kernel_config_add_template_arg_bool(
                            self.c_config,
                            name.as_ptr(),
                            *value,
                        )
                    })?;
                }
            }
        }

        Ok(())
    }
}

impl Drop for RawMetalKernelConfig {
    fn drop(&mut self) {
        unsafe {
            safemlx_sys::mlx_fast_metal_kernel_config_free(self.c_config);
        }
    }
}

struct VectorString {
    c_vec: safemlx_sys::mlx_vector_string,
    _strings: Vec<CString>,
}

impl VectorString {
    fn try_from_strings(strings: &[String]) -> Result<Self> {
        let mut c_strings = Vec::with_capacity(strings.len());
        for string in strings {
            c_strings.push(cstring(string)?);
        }

        let mut c_ptrs: Vec<*const c_char> = c_strings.iter().map(|s| s.as_ptr()).collect();
        let c_vec =
            unsafe { safemlx_sys::mlx_vector_string_new_data(c_ptrs.as_mut_ptr(), c_ptrs.len()) };

        Ok(Self {
            c_vec,
            _strings: c_strings,
        })
    }

    fn as_ptr(&self) -> safemlx_sys::mlx_vector_string {
        self.c_vec
    }
}

impl Drop for VectorString {
    fn drop(&mut self) {
        let status = unsafe { safemlx_sys::mlx_vector_string_free(self.c_vec) };
        debug_assert_eq!(status, SUCCESS);
    }
}

fn cstring(value: &str) -> Result<CString> {
    CString::new(value).map_err(|e| Exception::custom(format!("{e}")))
}

fn check_status(status: i32) -> Result<()> {
    match status {
        SUCCESS => Ok(()),
        _ => {
            let what = crate::error::get_and_clear_last_mlx_error()
                .map(|e| e.what)
                .unwrap_or_else(|| "MLX operation failed but no error was set".to_string());
            Err(Exception::custom(what))
        }
    }
}

/// Optimized implementation of `NN.RoPE`.
#[allow(clippy::too_many_arguments)]
#[generate_macro(customize(root = "$crate::fast"))]
pub fn rope<'a>(
    #[named] array: impl AsRef<Array>,
    #[named] dimensions: i32,
    #[named] traditional: bool,
    #[optional] base: impl Into<Option<f32>>,
    #[named] scale: f32,
    #[named] offset: i32,
    #[optional] freqs: impl Into<Option<&'a Array>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let base = base.into();
    let base = safemlx_sys::mlx_optional_float {
        value: base.unwrap_or(0.0),
        has_value: base.is_some(),
    };
    let freqs = freqs.into();
    Array::try_from_op(|res| unsafe {
        safemlx_sys::mlx_fast_rope(
            res,
            array.as_ref().as_ptr(),
            dimensions,
            traditional,
            base,
            scale,
            offset,
            freqs
                .map(|a| a.as_ptr())
                .unwrap_or(safemlx_sys::mlx_array_new()),
            stream.as_ref().as_ptr(),
        )
    })
}

/// Optimized implementation of `NN.RoPE` with dynamic (array) offset.
///
/// This variant allows specifying the offset as an array, enabling different
/// offsets for different positions in the input.
///
/// # Params
///
/// - `array`: Input array
/// - `dimensions`: The feature dimensions to apply rope to
/// - `traditional`: If true, uses the traditional rope implementation
/// - `base`: The base used to compute angular frequency for each dimension
/// - `scale`: The scale to apply to the positions
/// - `offset`: An array of position offsets
/// - `freqs`: Optional precomputed frequencies
/// - `stream`: Stream to evaluate on
#[allow(clippy::too_many_arguments)]
#[generate_macro(customize(root = "$crate::fast"))]
pub fn rope_dynamic<'a>(
    #[named] array: impl AsRef<Array>,
    #[named] dimensions: i32,
    #[named] traditional: bool,
    #[optional] base: impl Into<Option<f32>>,
    #[named] scale: f32,
    #[named] offset: impl AsRef<Array>,
    #[optional] freqs: impl Into<Option<&'a Array>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let base = base.into();
    let base = safemlx_sys::mlx_optional_float {
        value: base.unwrap_or(0.0),
        has_value: base.is_some(),
    };
    let freqs = freqs.into();
    Array::try_from_op(|res| unsafe {
        safemlx_sys::mlx_fast_rope_dynamic(
            res,
            array.as_ref().as_ptr(),
            dimensions,
            traditional,
            base,
            scale,
            offset.as_ref().as_ptr(),
            freqs
                .map(|a| a.as_ptr())
                .unwrap_or(safemlx_sys::mlx_array_new()),
            stream.as_ref().as_ptr(),
        )
    })
}

const DEFAULT_MASK_MODE: &CStr = c"";
const CAUSAL_MASK_MODE: &CStr = c"causal";

/// Mask modes for scaled dot product attention.
#[derive(Debug)]
pub enum ScaledDotProductAttentionMask<'a> {
    /// A single mask array
    Array(&'a Array),

    /// Causal masking (no explicit mask array needed)
    Causal,
}

impl<'a> From<&'a Array> for ScaledDotProductAttentionMask<'a> {
    fn from(mask: &'a Array) -> Self {
        ScaledDotProductAttentionMask::Array(mask)
    }
}

impl<'a> IntoOption<ScaledDotProductAttentionMask<'a>> for &'a Array {
    fn into_option(self) -> Option<ScaledDotProductAttentionMask<'a>> {
        Some(ScaledDotProductAttentionMask::Array(self))
    }
}

impl ScaledDotProductAttentionMask<'_> {
    fn as_mode_and_mask(&self) -> (&'static CStr, safemlx_sys::mlx_array) {
        match self {
            ScaledDotProductAttentionMask::Array(mask) => (DEFAULT_MASK_MODE, mask.as_ptr()),
            ScaledDotProductAttentionMask::Causal => {
                (CAUSAL_MASK_MODE, unsafe { safemlx_sys::mlx_array_new() })
            }
        }
    }
}

/// A fast implementation of multi-head attention: `O = softmax(Q @ K.T, dim=-1) @ V`
///
/// Supports [Multi-Head Attention](https://arxiv.org/abs/1706.03762), [Grouped Query Attention](https://arxiv.org/abs/2305.13245), and [Multi-Query Attention](https://arxiv.org/abs/1911.02150).
///
/// This function will dispatch to an optimized Metal kernel when the query sequence length is 1. It handles other cases with regular MLX operations.
///
/// > Note: The softmax operation is performed in float32 precision regardless of input precision (float16 or float32).
///
/// > Note: For Grouped Query Attention and Multi-Query Attention, the input arrays for `key` and `value` should not be pre-tiled to match the `query` array.
#[generate_macro(customize(root = "$crate::fast"))]
pub fn scaled_dot_product_attention<'a>(
    queries: impl AsRef<Array>,
    keys: impl AsRef<Array>,
    values: impl AsRef<Array>,
    scale: f32,
    #[optional] mask: impl IntoOption<ScaledDotProductAttentionMask<'a>>,
    #[optional] sinks: impl Into<Option<&'a Array>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let (mask_mode, mask_arr) = mask.into_option().map_or_else(
        || (DEFAULT_MASK_MODE, unsafe { safemlx_sys::mlx_array_new() }),
        |m| m.as_mode_and_mask(),
    );

    Array::try_from_op(|res| unsafe {
        safemlx_sys::mlx_fast_scaled_dot_product_attention(
            res,
            queries.as_ref().as_ptr(),
            keys.as_ref().as_ptr(),
            values.as_ref().as_ptr(),
            scale,
            mask_mode.as_ptr(),
            mask_arr,
            sinks
                .into()
                .map(|a| a.as_ptr())
                .unwrap_or(safemlx_sys::mlx_array_new()),
            stream.as_ref().as_ptr(),
        )
    })
}

/// Root Mean Square normalization (RMS norm).
///
/// The normalization is with respect to the last axis of the input `x`.
///
/// # Params
///
/// - x: input array
/// - weight: A multiplicative weight to scale the result by. The `weight` should be one-dimensional with the same size as the last axis of `x`.
/// - eps: A small additive constant for numerical stability
/// - stream: stream or device to evaluate on
#[generate_macro(customize(root = "$crate::fast"))]
pub fn rms_norm(
    x: impl AsRef<Array>,
    weight: impl AsRef<Array>,
    eps: f32,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        safemlx_sys::mlx_fast_rms_norm(
            res,
            x.as_ref().as_ptr(),
            weight.as_ref().as_ptr(),
            eps,
            stream.as_ref().as_ptr(),
        )
    })
}

/// Layer normalization.
///
/// The normalization is with respect to the last axis of the input `x`.
///
/// # Params
///
/// - x: input array
/// - weight: A multiplicative weight to scale the result by. The `weight` should be one-dimensional
///   with the same size as the last axis of `x`.  If not given no scaling will occur.
/// - bias: An additive offset to be added to the result. The `bias` should be one-dimensional
///   with the same size as the last axis of `x`.  It not given no offset will occur.
/// - eps: A small additive constant for numerical stability
/// - stream: stream or device to evaluate on
#[generate_macro(customize(root = "$crate::fast"))]
pub fn layer_norm<'a>(
    #[named] x: impl AsRef<Array>,
    #[optional] weight: impl Into<Option<&'a Array>>,
    #[optional] bias: impl Into<Option<&'a Array>>,
    #[named] eps: f32,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        safemlx_sys::mlx_fast_layer_norm(
            res,
            x.as_ref().as_ptr(),
            weight
                .into()
                .map(|a| a.as_ptr())
                .unwrap_or(safemlx_sys::mlx_array_new()),
            bias.into()
                .map(|a| a.as_ptr())
                .unwrap_or(safemlx_sys::mlx_array_new()),
            eps,
            stream.as_ref().as_ptr(),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ops::indexing::{ArrayIndexOp, IndexOp},
        random::normal,
        Stream,
    };
    use float_eq::assert_float_eq;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_metal_kernel_config_builder() {
        let config = MetalKernelConfig::new()
            .with_output_arg([2, 3], Dtype::Float32)
            .with_grid([6, 1, 1])
            .with_thread_group([32, 1, 1])
            .with_init_value(0.0)
            .with_verbose(true)
            .with_template_arg_dtype("T", Dtype::Float32)
            .with_template_arg_int("N", 6)
            .with_template_arg_bool("DO_SCALE", true);

        assert_eq!(config.output_count(), 1);
        assert_eq!(
            config.outputs()[0],
            MetalKernelOutput::new([2, 3], Dtype::Float32)
        );
        assert_eq!(config.template_args().len(), 3);
    }

    #[test]
    #[ignore = "requires an accessible Metal device"]
    fn test_custom_metal_kernel_multiple_outputs() {
        let input = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4]);
        let kernel = MetalKernel::new(
            "copy_and_double",
            ["inp"],
            ["out0", "out1"],
            concat!(
                "uint elem = thread_position_in_grid.x;",
                "T value = inp[elem];",
                "out0[elem] = value;",
                "out1[elem] = value + value;"
            ),
            "",
            true,
            false,
        )
        .unwrap();

        let config = MetalKernelConfig::new()
            .with_template_arg_dtype("T", Dtype::Float32)
            .with_grid([input.size() as i32, 1, 1])
            .with_thread_group([256, 1, 1])
            .with_output_arg(input.shape(), input.dtype())
            .with_output_arg(input.shape(), input.dtype());

        let outputs = kernel
            .apply_device(
                [&input],
                &config,
                Stream::new_with_device(&crate::Device::new(crate::DeviceType::Gpu, 0)),
            )
            .unwrap();

        assert_eq!(outputs.len(), 2);
        assert_eq!(
            crate::array::eval_vec::<f32>(&outputs[0]),
            &[1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(
            crate::array::eval_vec::<f32>(&outputs[1]),
            &[2.0, 4.0, 6.0, 8.0]
        );
    }

    #[test]
    #[ignore = "requires an accessible Metal device"]
    fn test_stateful_and_recurrent_metal_kernels() {
        let state = Array::from_slice(&[10.0f32, 20.0], &[2]);
        let token = Array::from_slice(&[1.0f32, 2.0], &[2]);
        let sequence = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);

        let decode = StatefulMetalKernel::new(
            "stateful_decode_test",
            ["state", "token"],
            ["out", "state_out"],
            concat!(
                "uint elem = thread_position_in_grid.x;",
                "float updated = float(state[elem]) + float(token[elem]);",
                "out[elem] = updated;",
                "state_out[elem] = updated;"
            ),
            "",
            true,
            false,
        )
        .unwrap();
        let prefill = StatefulMetalKernel::new(
            "stateful_prefill_test",
            ["state", "sequence"],
            ["out", "state_out"],
            concat!(
                "uint elem = thread_position_in_grid.x;",
                "float acc = float(state[elem]);",
                "for (uint t = 0; t < L; ++t) {",
                "  acc += float(sequence[t * D + elem]);",
                "  out[t * D + elem] = acc;",
                "}",
                "state_out[elem] = acc;"
            ),
            "",
            true,
            false,
        )
        .unwrap();
        let recurrent = RecurrentScanKernel::new(decode, prefill);

        let decode_config = MetalKernelConfig::new()
            .with_grid([2, 1, 1])
            .with_thread_group([32, 1, 1])
            .with_output_arg([2], Dtype::Float32)
            .with_output_arg([2], Dtype::Float32);
        let decode = recurrent
            .decode_device(
                [&state, &token],
                &decode_config,
                Stream::new_with_device(&crate::Device::new(crate::DeviceType::Gpu, 0)),
            )
            .unwrap();
        assert_eq!(
            crate::array::eval_vec::<f32>(&decode.output_sequence),
            &[11.0, 22.0]
        );
        assert_eq!(
            crate::array::eval_vec::<f32>(&decode.new_state),
            &[11.0, 22.0]
        );

        let prefill_config = MetalKernelConfig::new()
            .with_template_arg_int("L", 3)
            .with_template_arg_int("D", 2)
            .with_grid([2, 1, 1])
            .with_thread_group([32, 1, 1])
            .with_output_arg([3, 2], Dtype::Float32)
            .with_output_arg([2], Dtype::Float32);
        let prefill = recurrent
            .prefill_device(
                [&state, &sequence],
                &prefill_config,
                Stream::new_with_device(&crate::Device::new(crate::DeviceType::Gpu, 0)),
            )
            .unwrap();
        assert_eq!(
            crate::array::eval_vec::<f32>(&prefill.output_sequence),
            &[11.0, 22.0, 14.0, 26.0, 19.0, 32.0]
        );
        assert_eq!(
            crate::array::eval_vec::<f32>(&prefill.new_state),
            &[19.0, 32.0]
        );
    }

    #[test]
    fn test_rope() {
        let stream = crate::test_stream();
        let key = crate::test_key(71, stream);
        let a = crate::random::uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], &key, stream).unwrap();
        assert_eq!(a.shape(), [2, 8, 16]);
        assert_eq!(a.dtype(), crate::Dtype::Float32);

        let result = rope(a, 8, false, 10000., 1.0, 0, None, stream).unwrap();
        assert_eq!(result.shape(), [2, 8, 16]);
        assert_eq!(result.dtype(), crate::Dtype::Float32);
        assert_float_eq!(
            result.mean(None, stream).unwrap().item::<f32>(&stream),
            0.456_253_77,
            abs <= 0.009_125_075
        );
        assert_float_eq!(
            result.sum(None, stream).unwrap().item::<f32>(&stream),
            116.800_964,
            abs <= 2.336_019_3
        );
    }

    // Test adapted from Python test_fast.py/test_rope - the Python test accepts both
    // int offset and array offset, which in C/Rust are separate functions
    #[test]
    fn test_rope_dynamic() {
        let stream = crate::test_stream();
        let key = crate::test_key(71, stream);
        let a = crate::random::uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], &key, stream).unwrap();
        assert_eq!(a.shape(), [2, 8, 16]);
        assert_eq!(a.dtype(), crate::Dtype::Float32);

        // Test with array offset - should produce similar results to int offset of 3
        let offset = crate::Array::from_int(3);
        let result = rope_dynamic(&a, 8, false, 10000., 1.0, &offset, None, stream).unwrap();
        assert_eq!(result.shape(), [2, 8, 16]);
        assert_eq!(result.dtype(), crate::Dtype::Float32);

        // Compare with regular rope using int offset=3
        let result_int_offset = rope(&a, 8, false, 10000., 1.0, 3, None, stream).unwrap();
        assert_eq!(result_int_offset.shape(), [2, 8, 16]);

        // The results should be close
        let diff = result.subtract(&result_int_offset, stream).unwrap();
        let max_diff = diff
            .abs(stream)
            .unwrap()
            .max(None, stream)
            .unwrap()
            .item::<f32>(&stream);
        assert!(max_diff < 1e-5, "Max difference was {}", max_diff);
    }

    #[test]
    fn test_rms_norm() {
        let stream = crate::test_stream();
        let key = crate::test_key(103, stream);
        let a = crate::random::uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], &key, stream).unwrap();
        assert_eq!(a.shape(), [2, 8, 16]);
        assert_eq!(a.dtype(), crate::Dtype::Float32);

        let weight = Array::ones::<f32>(&[16], stream).unwrap();
        let result = rms_norm(a, weight, 1e-5, stream).unwrap();
        assert_eq!(result.shape(), [2, 8, 16]);
        assert_eq!(result.dtype(), crate::Dtype::Float32);
        assert_float_eq!(
            result.mean(None, stream).unwrap().item::<f32>(&stream),
            0.872_938_75,
            abs <= 0.017_458_774
        );
        assert_float_eq!(
            result.sum(None, stream).unwrap().item::<f32>(&stream),
            223.472_32,
            abs <= 4.469_446
        );
    }

    #[test]
    pub fn test_layer_norm_affine() {
        let stream = crate::test_stream();
        let key = crate::test_key(635, stream);
        let a = crate::random::uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], &key, stream).unwrap();
        assert_eq!(a.shape(), [2, 8, 16]);
        assert_eq!(a.dtype(), crate::Dtype::Float32);

        let weight = Array::ones::<f32>(&[16], stream).unwrap();
        let bias = Array::zeros::<f32>(&[16], stream).unwrap();
        let result = layer_norm(a, &weight, &bias, 1e-5, stream).unwrap();
        let result = result.index_device((ArrayIndexOp::Ellipsis, 0), stream);
        assert_eq!(result.shape(), [2, 8]);
        assert_eq!(result.dtype(), crate::Dtype::Float32);
        assert_float_eq!(
            result.mean(None, stream).unwrap().item::<f32>(&stream),
            0.290_990_38,
            abs <= 0.005_819_807_8
        );
        assert_float_eq!(
            result.sum(None, stream).unwrap().item::<f32>(&stream),
            4.655_846,
            abs <= 0.093_116_924
        );
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_fast_sdpa() {
        let stream = crate::test_stream();
        // This test just makes sure that `scaled_dot_product_attention` is callable
        // in the various cases, based on the Python test `test_fast_sdpa`.

        let Dk = 64;
        let scale = 1.0 / (Dk as f32).sqrt();
        for seq_len in [63, 129, 400] {
            for dtype in [crate::Dtype::Float32, crate::Dtype::Float16] {
                let B = 2;
                let H = 24;
                let q_key = crate::test_key((seq_len + Dk) as u64, stream);
                let k_key = crate::test_key((seq_len + Dk + 1) as u64, stream);
                let v_key = crate::test_key((seq_len + Dk + 2) as u64, stream);
                let q = normal::<f32>(&[B, H, seq_len, Dk], None, None, &q_key, stream)
                    .unwrap()
                    .as_dtype(dtype, stream)
                    .unwrap();
                let k = normal::<f32>(&[B, H, seq_len, Dk], None, None, &k_key, stream)
                    .unwrap()
                    .as_dtype(dtype, stream)
                    .unwrap();
                let v = normal::<f32>(&[B, H, seq_len, Dk], None, None, &v_key, stream)
                    .unwrap()
                    .as_dtype(dtype, stream)
                    .unwrap();

                let result =
                    scaled_dot_product_attention(q, k, v, scale, None, None, stream).unwrap();
                assert_eq!(result.shape(), [B, H, seq_len, Dk]);
                assert_eq!(result.dtype(), dtype);
            }
        }
    }

    // Test adapted from Python test `test_fast_sdpa.py/test_sdpa_attention_sinks`
    #[test]
    fn test_fast_sdpa_with_sinks() {
        let stream = crate::test_stream();
        let b = 2;
        let n_q = 8;
        let t_q = 128;
        let t_kv = 128;
        let d = 64;

        let q_key = crate::test_key(0, stream);
        let k_key = crate::test_key(1, stream);
        let v_key = crate::test_key(2, stream);
        let sinks_key = crate::test_key(3, stream);
        let q = normal::<f32>(&[b, n_q, t_q, d], None, None, &q_key, stream).unwrap();
        let k = normal::<f32>(&[b, n_q, t_kv, d], None, None, &k_key, stream).unwrap();
        let v = normal::<f32>(&[b, n_q, t_kv, d], None, None, &v_key, stream).unwrap();
        let scale = (d as f32).powf(-0.5);

        // Test with sinks parameter
        let sinks = normal::<f32>(&[n_q], None, None, &sinks_key, stream)
            .unwrap()
            .multiply(Array::from_f32(10.0), stream)
            .unwrap();

        let result = scaled_dot_product_attention(&q, &k, &v, scale, None, &sinks, stream).unwrap();
        assert_eq!(result.shape(), &[b, n_q, t_q, d]);
    }
}
