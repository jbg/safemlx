use std::ffi::CStr;

use safemlx_internal_macros::generate_macro;

use crate::{
    error::Result,
    utils::{guard::Guarded, VectorArray},
    Array, Stream,
};

const DEFAULT_GROUP_SIZE: i32 = 64;
const DEFAULT_BITS: i32 = 4;

/// Quantized matrix-multiplication weight encoding.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum QuantizationMode {
    /// Per-group scale and bias used by MLX's general-purpose quantizer.
    Affine,
    /// Microscaling FP4 with one E8M0 scale for every 32 values.
    MxFp4,
}

impl QuantizationMode {
    fn as_c_str(self) -> &'static CStr {
        match self {
            Self::Affine => c"affine",
            Self::MxFp4 => c"mxfp4",
        }
    }

    /// Returns whether this encoding stores a per-group quantization bias.
    pub const fn has_biases(self) -> bool {
        matches!(self, Self::Affine)
    }

    /// Validates the group size and bit width required by this encoding.
    pub fn validate(self, group_size: i32, bits: i32) -> Result<()> {
        if self == Self::MxFp4 && (group_size != 32 || bits != 4) {
            return Err(crate::error::Exception::custom(format!(
                "MXFP4 requires group_size=32 and bits=4, got group_size={group_size} bits={bits}"
            )));
        }
        Ok(())
    }
}

/// Arrays returned by quantization. MXFP4 has no quantization bias tensor.
#[derive(Debug, Clone)]
pub struct QuantizedArrays {
    /// Packed quantized values.
    pub weight: Array,
    /// Per-group scales.
    pub scales: Array,
    /// Per-group affine biases, absent for MXFP4.
    pub biases: Option<Array>,
}

/// Returns the number of `u32` values used to store one packed quantized row.
///
/// MLX packs power-of-two widths directly and packs 3-, 5-, and 6-bit values
/// into byte groups. Both layouts occupy `dimension * bits` bits overall.
pub const fn quantized_packed_dimension(dimension: i32, bits: i32) -> i32 {
    dimension * bits / 32
}

/// Helper to convert Option<i32> to mlx_optional_int
fn optional_int(value: Option<i32>, default: i32) -> safemlx_sys::mlx_optional_int {
    safemlx_sys::mlx_optional_int {
        value: value.unwrap_or(default),
        has_value: value.is_some(),
    }
}

/// Helper to create a "no value" optional dtype
fn optional_dtype_none() -> safemlx_sys::mlx_optional_dtype {
    safemlx_sys::mlx_optional_dtype {
        value: safemlx_sys::mlx_dtype__MLX_FLOAT32, // default value, ignored when has_value is false
        has_value: false,
    }
}

/// Quantize the matrix `w` using `bits` bits per element.
///
/// Note, every `group_size` elements in a row of `w` are quantized together. Hence, number of
/// columns of `w` should be divisible by `group_size`. In particular, the rows of `w` are divided
/// into groups of size `group_size` which are quantized together.
///
/// > `quantized` currently only supports 2D inputs with dimensions which are multiples of 32
///
/// For details, please see [this
/// documentation](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html)
///
/// # Params
///
/// - `w`: The input matrix
/// - `group_size`: The size of the group in `w` that shares a scale and bias. (default: `64`)
/// - `bits`: The number of bits occupied by each element of w in the returned quantized matrix.
///   (default: 4)
#[generate_macro]
pub fn quantize(
    w: impl AsRef<Array>,
    #[optional] group_size: impl Into<Option<i32>>,
    #[optional] bits: impl Into<Option<i32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<(Array, Array, Array)> {
    let arrays = quantize_with_mode(
        w,
        group_size.into().unwrap_or(DEFAULT_GROUP_SIZE),
        bits.into().unwrap_or(DEFAULT_BITS),
        QuantizationMode::Affine,
        stream,
    )?;
    Ok((
        arrays.weight,
        arrays.scales,
        arrays.biases.expect("affine quantization returns biases"),
    ))
}

/// Quantizes `w` using an explicit weight encoding.
pub fn quantize_with_mode(
    w: impl AsRef<Array>,
    group_size: i32,
    bits: i32,
    mode: QuantizationMode,
    stream: impl AsRef<Stream>,
) -> Result<QuantizedArrays> {
    mode.validate(group_size, bits)?;
    let group_size = optional_int(Some(group_size), DEFAULT_GROUP_SIZE);
    let bits = optional_int(Some(bits), DEFAULT_BITS);

    let result = VectorArray::try_from_op(|res| unsafe {
        safemlx_sys::mlx_quantize(
            res,
            w.as_ref().as_ptr(),
            group_size,
            bits,
            mode.as_c_str().as_ptr(),
            safemlx_sys::mlx_array_new(),
            stream.as_ref().as_ptr(),
        )
    })?;

    let arrays: Vec<Array> = result.try_into_values()?;
    let expected = if mode.has_biases() { 3 } else { 2 };
    if arrays.len() != expected {
        return Err(crate::error::Exception::custom(format!(
            "Expected {expected} arrays from {mode:?} quantize, got {}",
            arrays.len()
        )));
    }
    let mut iter = arrays.into_iter();
    Ok(QuantizedArrays {
        weight: iter.next().unwrap(),
        scales: iter.next().unwrap(),
        biases: iter.next(),
    })
}

/// Perform the matrix multiplication with the quantized matrix `w`. The quantization uses one
/// floating point scale and bias per `group_size` of elements. Each element in `w` takes `bits`
/// bits and is packed in an unsigned 32 bit integer.
#[allow(clippy::too_many_arguments)]
#[generate_macro]
pub fn quantized_matmul<'a>(
    x: impl AsRef<Array>,
    w: impl AsRef<Array>,
    scales: impl AsRef<Array>,
    #[optional] biases: impl Into<Option<&'a Array>>,
    #[optional] transpose: impl Into<Option<bool>>,
    #[optional] group_size: impl Into<Option<i32>>,
    #[optional] bits: impl Into<Option<i32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    quantized_matmul_with_mode(
        x,
        w,
        scales,
        biases.into(),
        transpose.into().unwrap_or(false),
        group_size.into().unwrap_or(DEFAULT_GROUP_SIZE),
        bits.into().unwrap_or(DEFAULT_BITS),
        QuantizationMode::Affine,
        stream,
    )
}

/// Performs quantized matrix multiplication using an explicit weight encoding.
#[allow(clippy::too_many_arguments)]
pub fn quantized_matmul_with_mode(
    x: impl AsRef<Array>,
    w: impl AsRef<Array>,
    scales: impl AsRef<Array>,
    biases: Option<&Array>,
    transpose: bool,
    group_size: i32,
    bits: i32,
    mode: QuantizationMode,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let group_size = optional_int(Some(group_size), DEFAULT_GROUP_SIZE);
    let bits = optional_int(Some(bits), DEFAULT_BITS);

    <Array as Guarded>::try_from_op(|res| unsafe {
        safemlx_sys::mlx_quantized_matmul(
            res,
            x.as_ref().as_ptr(),
            w.as_ref().as_ptr(),
            scales.as_ref().as_ptr(),
            biases
                .map(|a| a.as_ptr())
                .unwrap_or(safemlx_sys::mlx_array_new()),
            transpose,
            group_size,
            bits,
            mode.as_c_str().as_ptr(),
            stream.as_ref().as_ptr(),
        )
    })
}

/// Dequantize the matrix `w` using the provided `scales` and `biases` and the `group_size` and
/// `bits` configuration.
///
/// For details, please see [this
/// documentation](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.dequantize.html)
#[generate_macro]
pub fn dequantize<'a>(
    w: impl AsRef<Array>,
    scales: impl AsRef<Array>,
    #[optional] biases: impl Into<Option<&'a Array>>,
    #[optional] group_size: impl Into<Option<i32>>,
    #[optional] bits: impl Into<Option<i32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    dequantize_with_mode(
        w,
        scales,
        biases.into(),
        group_size.into().unwrap_or(DEFAULT_GROUP_SIZE),
        bits.into().unwrap_or(DEFAULT_BITS),
        QuantizationMode::Affine,
        stream,
    )
}

/// Dequantizes `w` using an explicit weight encoding.
pub fn dequantize_with_mode(
    w: impl AsRef<Array>,
    scales: impl AsRef<Array>,
    biases: Option<&Array>,
    group_size: i32,
    bits: i32,
    mode: QuantizationMode,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    mode.validate(group_size, bits)?;
    let group_size = optional_int(Some(group_size), DEFAULT_GROUP_SIZE);
    let bits = optional_int(Some(bits), DEFAULT_BITS);

    <Array as Guarded>::try_from_op(|res| unsafe {
        safemlx_sys::mlx_dequantize(
            res,
            w.as_ref().as_ptr(),
            scales.as_ref().as_ptr(),
            biases
                .map(|a| a.as_ptr())
                .unwrap_or(safemlx_sys::mlx_array_new()),
            group_size,
            bits,
            mode.as_c_str().as_ptr(),
            safemlx_sys::mlx_array_new(),
            optional_dtype_none(),
            stream.as_ref().as_ptr(),
        )
    })
}

/// Perform quantized matrix multiplication with gathered indices.
///
/// This combines the functionality of `gather_mm` and `quantized_matmul`, allowing
/// matrix multiplication with quantized weights and index gathering along batch dimensions.
///
/// # Params
///
/// - `x`: Input array
/// - `w`: Quantized weight matrix
/// - `scales`: Quantization scales
/// - `biases`: Optional quantization biases (required for affine mode)
/// - `lhs_indices`: Optional indices to gather from `x`'s batch dimensions
/// - `rhs_indices`: Optional indices to gather from `w`'s batch dimensions
/// - `transpose`: If true, transpose the weight matrix (default: true)
/// - `group_size`: The quantization group size (default: 64)
/// - `bits`: The number of bits per element (default: 4)
/// - `sorted_indices`: If true, indicates the indices are sorted (default: false)
#[allow(clippy::too_many_arguments)]
#[generate_macro]
pub fn gather_qmm<'b, 'lhs, 'rhs>(
    x: impl AsRef<Array>,
    w: impl AsRef<Array>,
    scales: impl AsRef<Array>,
    #[optional] biases: impl Into<Option<&'b Array>>,
    #[optional] lhs_indices: impl Into<Option<&'lhs Array>>,
    #[optional] rhs_indices: impl Into<Option<&'rhs Array>>,
    #[optional] transpose: impl Into<Option<bool>>,
    #[optional] group_size: impl Into<Option<i32>>,
    #[optional] bits: impl Into<Option<i32>>,
    #[optional] sorted_indices: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let transpose = transpose.into().unwrap_or(true);
    let group_size = optional_int(group_size.into(), DEFAULT_GROUP_SIZE);
    let bits = optional_int(bits.into(), DEFAULT_BITS);
    let sorted = sorted_indices.into().unwrap_or(false);

    gather_qmm_with_mode(
        x,
        w,
        scales,
        biases.into(),
        lhs_indices.into(),
        rhs_indices.into(),
        transpose,
        group_size.value,
        bits.value,
        sorted,
        QuantizationMode::Affine,
        stream.as_ref(),
    )
}

/// Performs a gathered quantized matrix multiplication using an explicit
/// weight encoding. This is the common primitive for affine routed experts and
/// checkpoint-native MXFP4 routed experts.
#[allow(clippy::too_many_arguments)]
pub fn gather_qmm_with_mode(
    x: impl AsRef<Array>,
    w: impl AsRef<Array>,
    scales: impl AsRef<Array>,
    biases: Option<&Array>,
    lhs_indices: Option<&Array>,
    rhs_indices: Option<&Array>,
    transpose: bool,
    group_size: i32,
    bits: i32,
    sorted_indices: bool,
    mode: QuantizationMode,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    mode.validate(group_size, bits)?;
    unsafe {
        let biases_ptr = biases
            .map(|a| a.as_ptr())
            .unwrap_or(safemlx_sys::mlx_array_new());
        let lhs_ptr = lhs_indices
            .map(|i| i.as_ptr())
            .unwrap_or(safemlx_sys::mlx_array_new());
        let rhs_ptr = rhs_indices
            .map(|i| i.as_ptr())
            .unwrap_or(safemlx_sys::mlx_array_new());

        <Array as Guarded>::try_from_op(|res| {
            safemlx_sys::mlx_gather_qmm(
                res,
                x.as_ref().as_ptr(),
                w.as_ref().as_ptr(),
                scales.as_ref().as_ptr(),
                biases_ptr,
                lhs_ptr,
                rhs_ptr,
                transpose,
                optional_int(Some(group_size), DEFAULT_GROUP_SIZE),
                optional_int(Some(bits), DEFAULT_BITS),
                mode.as_c_str().as_ptr(),
                sorted_indices,
                stream.as_ref().as_ptr(),
            )
        })
    }
}

/// Quantized matrix multiplication with quantization of both inputs.
///
/// Performs matrix multiplication where `x` is dynamically quantized and `w` is pre-quantized.
/// This function supports `nvfp4` and `mxfp8` quantization modes.
///
/// Note: This function is only supported on GPU with the CUDA backend (Linux with NVIDIA GPU).
/// It is not available on macOS.
///
/// # Params
///
/// - `x`: Input matrix to be dynamically quantized
/// - `w`: Pre-quantized weight matrix
/// - `w_scales`: Optional scales for the quantized weights (required if `w` is already quantized)
/// - `group_size`: The quantization group size (default depends on mode: 16 for nvfp4, 32 for mxfp8)
/// - `bits`: The number of bits per element (default depends on mode: 4 for nvfp4, 8 for mxfp8)
/// - `mode`: Quantization mode - either "nvfp4" or "mxfp8" (default: "nvfp4")
#[cfg(not(target_os = "macos"))]
#[allow(clippy::too_many_arguments)]
#[generate_macro]
pub fn qqmm<'a>(
    x: impl AsRef<Array>,
    w: impl AsRef<Array>,
    #[optional] w_scales: impl Into<Option<&'a Array>>,
    #[optional] group_size: impl Into<Option<i32>>,
    #[optional] bits: impl Into<Option<i32>>,
    #[optional] mode: impl Into<Option<&'a str>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let mode_str = mode.into().unwrap_or("nvfp4");
    let mode_cstr = std::ffi::CString::new(mode_str).expect("Invalid mode string");

    // Defaults depend on mode
    let (default_group_size, default_bits) = match mode_str {
        "nvfp4" => (16, 4),
        "mxfp8" => (32, 8),
        _ => (16, 4), // fallback to nvfp4 defaults
    };

    let group_size = optional_int(group_size.into(), default_group_size);
    let bits = optional_int(bits.into(), default_bits);

    <Array as Guarded>::try_from_op(|res| unsafe {
        safemlx_sys::mlx_qqmm(
            res,
            x.as_ref().as_ptr(),
            w.as_ref().as_ptr(),
            w_scales
                .into()
                .map(|a| a.as_ptr())
                .unwrap_or(safemlx_sys::mlx_array_new()),
            group_size,
            bits,
            mode_cstr.as_ptr(),
            stream.as_ref().as_ptr(),
        )
    })
}

#[cfg(test)]
mod tests {
    use crate::{
        array,
        ops::{
            dequantize, dequantize_with_mode, expand_dims, quantize, quantize_with_mode,
            quantized_matmul, quantized_matmul_with_mode, QuantizationMode,
        },
        random, Array, Stream,
    };

    #[test]
    fn test_quantize_dequantize() {
        let stream = crate::test_stream();
        let x1 = Array::ones::<f32>(&[128, 1], stream).unwrap();
        let x2 = expand_dims(
            Array::arange::<_, f32>(0, 512, None, stream).unwrap(),
            0,
            stream,
        )
        .unwrap();
        let x = x1.multiply(&x2, stream).unwrap();

        for i in [2, 4, 8].iter() {
            let el_per_int = 32 / i;
            let (x_q, scales, biases) = quantize(&x, 128, *i, stream).unwrap();
            assert_eq!(x_q.shape(), [128, 512 / el_per_int]);
            assert_eq!(scales.shape(), [128, 4]);
            assert_eq!(biases.shape(), [128, 4]);

            let x_hat = dequantize(&x_q, &scales, &biases, 128, *i, stream).unwrap();
            let max_diff = x
                .subtract(&x_hat, stream)
                .unwrap()
                .abs(stream)
                .unwrap()
                .max(None, stream)
                .unwrap()
                .item::<f32>(&stream);
            assert!(max_diff <= 127.0 / (1 << i) as f32);
        }
    }

    #[test]
    fn test_mxfp4_quantize_dequantize_and_matmul() {
        let stream = crate::test_stream();
        let weight = Array::arange::<_, f32>(-64, 64, None, stream)
            .unwrap()
            .reshape(&[2, 64], stream)
            .unwrap()
            .divide(array!(16.0), stream)
            .unwrap();
        let arrays = quantize_with_mode(&weight, 32, 4, QuantizationMode::MxFp4, stream).unwrap();
        assert_eq!(arrays.weight.shape(), &[2, 8]);
        assert_eq!(arrays.scales.shape(), &[2, 2]);
        assert!(arrays.biases.is_none());

        let restored = dequantize_with_mode(
            &arrays.weight,
            &arrays.scales,
            None,
            32,
            4,
            QuantizationMode::MxFp4,
            stream,
        )
        .unwrap();
        let max_error = weight
            .subtract(&restored, stream)
            .unwrap()
            .abs(stream)
            .unwrap()
            .max(None, stream)
            .unwrap()
            .item::<f32>(stream);
        assert!(max_error <= 1.0, "MXFP4 max error was {max_error}");

        let input = Array::ones::<f32>(&[3, 64], stream).unwrap();
        let actual = quantized_matmul_with_mode(
            &input,
            &arrays.weight,
            &arrays.scales,
            None,
            true,
            32,
            4,
            QuantizationMode::MxFp4,
            stream,
        )
        .unwrap();
        let expected = input
            .matmul(&restored.transpose(stream).unwrap(), stream)
            .unwrap();
        let max_difference = actual
            .subtract(&expected, stream)
            .unwrap()
            .abs(stream)
            .unwrap()
            .max(None, stream)
            .unwrap()
            .item::<f32>(stream);
        assert!(
            max_difference < 1e-3,
            "MXFP4 qmm difference was {max_difference}"
        );
    }

    #[test]
    fn test_mxfp4_rejects_nonstandard_settings() {
        let stream = crate::test_stream();
        let weight = Array::zeros::<f32>(&[2, 64], stream).unwrap();
        assert!(quantize_with_mode(&weight, 64, 4, QuantizationMode::MxFp4, stream).is_err());
        assert!(quantize_with_mode(&weight, 32, 8, QuantizationMode::MxFp4, stream).is_err());
    }

    // Test adapted from Python test `test_quantized.py/test_qmm`
    #[test]
    fn test_quantized_matmul() {
        let stream = crate::test_stream();
        let mut random_state = random::RandomState::with_seed(0).unwrap();

        let group_size = 64;
        let bits = 4;
        let m = 32;
        let n = 128;
        let k = 128;

        let scale = 1.0 / (k as f32).sqrt();
        let key = random_state.next_key(stream).unwrap();
        let x = random::normal::<f32>(&[m, k], None, None, &key, stream)
            .unwrap()
            .multiply(array!(scale), stream)
            .unwrap();
        let key = random_state.next_key(stream).unwrap();
        let w = random::normal::<f32>(&[k, n], None, None, &key, stream)
            .unwrap()
            .multiply(array!(scale), stream)
            .unwrap();

        let (w_q, scales, biases) = quantize(&w, group_size, bits, stream).unwrap();
        let w_hat = dequantize(&w_q, &scales, &biases, group_size, bits, stream).unwrap();

        // Test with biases
        let y_q =
            quantized_matmul(&x, &w_q, &scales, &biases, false, group_size, bits, stream).unwrap();
        let y_hat = x.matmul(&w_hat, stream).unwrap();

        assert_eq!(y_q.shape(), y_hat.shape());
        let max_diff = y_q
            .subtract(&y_hat, stream)
            .unwrap()
            .abs(stream)
            .unwrap()
            .max(None, stream)
            .unwrap()
            .item::<f32>(&stream);
        assert!(max_diff < 1e-3, "max_diff: {}", max_diff);
    }

    // Test adapted from Python test `test_quantized.py/test_gather_qmm`
    #[test]
    fn test_gather_qmm() {
        let stream = crate::test_stream();
        use crate::ops::{gather_mm, gather_qmm, swap_axes};

        let mut random_state = random::RandomState::with_seed(0).unwrap();

        let group_size = 64;
        let bits = 4;

        // Helper to quantize with transpose option
        fn quantize_with_transpose(
            w: &Array,
            transpose: bool,
            group_size: i32,
            bits: i32,
            stream: &Stream,
        ) -> (Array, Array, Array, Array) {
            let (w_q, scales, biases) = quantize(w, group_size, bits, stream).unwrap();
            let mut w_hat = dequantize(&w_q, &scales, &biases, group_size, bits, stream).unwrap();
            if transpose {
                w_hat = swap_axes(&w_hat, -1, -2, stream).unwrap();
            }
            (w_hat, w_q, scales, biases)
        }

        // Test case 1: batch_A=(1,), lhs_indices=(0,), batch_B=(3,), rhs_indices=(2, 1)
        let m = 32;
        let n = 64;
        let k = 64;

        let key = random_state.next_key(stream).unwrap();
        let x = random::normal::<f32>(&[1, m, k], None, None, &key, stream).unwrap();
        let key = random_state.next_key(stream).unwrap();
        let w = random::normal::<f32>(&[3, n, k], None, None, &key, stream).unwrap(); // transpose=true shape
        let (w_hat, w_q, scales, biases) =
            quantize_with_transpose(&w, true, group_size, bits, stream);

        let lhs_indices = Array::from_slice(&[0u32], &[1]);
        let rhs_indices = Array::from_slice(&[2u32, 1], &[2]);

        // Compare gather_mm on dequantized weights vs gather_qmm
        let c1 = gather_mm(&x, &w_hat, &lhs_indices, &rhs_indices, None, stream).unwrap();
        let c2 = gather_qmm(
            &x,
            &w_q,
            &scales,
            &biases,
            &lhs_indices,
            &rhs_indices,
            true,
            group_size,
            bits,
            None,
            stream,
        )
        .unwrap();
        assert!(
            c1.all_close(&c2, 1e-4, 1e-4, None, stream)
                .unwrap()
                .item::<bool>(&stream),
            "gather_qmm test case 1 failed"
        );

        // Test case 2: batch_A=(5,), lhs_indices=(0, 2), batch_B=(3,), rhs_indices=(2, 1)
        let key = random_state.next_key(stream).unwrap();
        let x = random::normal::<f32>(&[5, m, k], None, None, &key, stream).unwrap();
        let lhs_indices = Array::from_slice(&[0u32, 2], &[2]);

        let c1 = gather_mm(&x, &w_hat, &lhs_indices, &rhs_indices, None, stream).unwrap();
        let c2 = gather_qmm(
            &x,
            &w_q,
            &scales,
            &biases,
            &lhs_indices,
            &rhs_indices,
            true,
            group_size,
            bits,
            None,
            stream,
        )
        .unwrap();
        assert!(
            c1.all_close(&c2, 1e-4, 1e-4, None, stream)
                .unwrap()
                .item::<bool>(&stream),
            "gather_qmm test case 2 failed"
        );
    }

    #[test]
    fn test_gather_qmm_mxfp4_mode() {
        use crate::ops::{gather_qmm_with_mode, QuantizationMode};

        let stream = crate::test_stream();
        let x = Array::ones::<f32>(&[1, 1, 32], stream).unwrap();
        // Four u32 values pack 32 FP4 nibbles. Code zero represents 0.0;
        // E8M0 scale 127 represents 2^0.
        let weights = Array::zeros::<u32>(&[1, 1, 4], stream).unwrap();
        let scales = Array::from_slice(&[127u8], &[1, 1, 1]);
        let indices = Array::from_slice(&[0u32], &[1]);
        let output = gather_qmm_with_mode(
            &x,
            &weights,
            &scales,
            None,
            Some(&indices),
            Some(&indices),
            true,
            32,
            4,
            true,
            QuantizationMode::MxFp4,
            stream,
        )
        .unwrap();
        assert_eq!(output.shape(), &[1, 1, 1]);
        assert_eq!(output.item::<f32>(&stream), 0.0);
    }
}
