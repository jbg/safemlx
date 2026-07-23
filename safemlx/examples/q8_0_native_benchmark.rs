//! Isolated native-Q8_0 versus converted-affine Metal benchmark.
//!
//! The deterministic synthetic matrices use Gemma 4 projection dimensions so
//! this can remain as a regression benchmark without requiring a checkpoint.

use std::time::{Duration, Instant};

use half::f16;
use safemlx::{
    memory,
    native_quantization::NativeQuantizedTensor,
    ops::{quantized_matmul_with_mode, QuantizationMode},
    transforms::eval,
    Array, Device, DeviceType, Dtype, Stream,
};
use safemlx_gguf::{ConvertedTensor, GgmlType, Reader, TensorInput, Writer};

fn q8_0_matrix(rows: i32, columns: i32) -> Vec<u8> {
    let blocks = (rows * columns / 32) as usize;
    let mut raw = Vec::with_capacity(blocks * 34);
    for block in 0..blocks {
        let scale = 0.001 + (block % 17) as f32 * 0.0002;
        raw.extend(f16::from_f32(scale).to_bits().to_le_bytes());
        for index in 0..32 {
            let quant = ((block * 31 + index * 17 + 7) % 255) as i16 - 127;
            raw.push((quant as i8) as u8);
        }
    }
    raw
}

fn converted_affine(
    raw: &[u8],
    rows: i32,
    columns: i32,
    stream: &Stream,
) -> (Array, Array, Array, usize) {
    let mut file = Vec::new();
    Writer::default()
        .write(
            std::io::Cursor::new(&mut file),
            &std::collections::BTreeMap::new(),
            &[TensorInput {
                name: "weight",
                dimensions: &[columns as u64, rows as u64],
                ggml_type: GgmlType::Q8_0,
                data: raw,
            }],
        )
        .unwrap();
    let mut reader = Reader::new(std::io::Cursor::new(file)).unwrap();
    let descriptor = reader.tensors()[0].clone();
    let ConvertedTensor::Affine(affine) = reader.read_tensor(&descriptor).unwrap() else {
        panic!("Q8_0 conversion did not return affine storage")
    };
    let packed = affine.weights.len() * 4 + affine.scales.len() * 2 + affine.biases.len() * 2;
    let weights = Array::from_slice(
        &affine.weights,
        &[
            rows,
            i32::try_from(affine.weight_shape[1]).expect("packed dimension"),
        ],
    )
    .copy(stream)
    .unwrap();
    let scales = Array::from_slice(
        &affine
            .scales
            .iter()
            .copied()
            .map(f16::from_bits)
            .collect::<Vec<_>>(),
        &[rows, columns / 32],
    )
    .copy(stream)
    .unwrap();
    let biases = Array::from_slice(
        &affine
            .biases
            .iter()
            .copied()
            .map(f16::from_bits)
            .collect::<Vec<_>>(),
        &[rows, columns / 32],
    )
    .copy(stream)
    .unwrap();
    eval([&weights, &scales, &biases]).unwrap();
    (weights, scales, biases, packed)
}

fn time_average(
    iterations: usize,
    stream: &Stream,
    mut operation: impl FnMut() -> Array,
) -> Duration {
    for _ in 0..3 {
        let output = operation();
        eval([&output]).unwrap();
    }
    stream.synchronize().unwrap();
    let started = Instant::now();
    for _ in 0..iterations {
        let output = operation();
        eval([&output]).unwrap();
    }
    stream.synchronize().unwrap();
    started.elapsed() / iterations as u32
}

fn benchmark_shape(rows: i32, columns: i32, batch: i32, iterations: usize, stream: &Stream) {
    let raw = q8_0_matrix(rows, columns);
    let native = NativeQuantizedTensor::from_q8_0_bytes(&raw, &[rows, columns], stream).unwrap();
    let (affine_weight, affine_scales, affine_biases, affine_bytes) =
        converted_affine(&raw, rows, columns, stream);
    let input = Array::from_slice(
        &(0..batch * columns)
            .map(|index| (index as f32 % 41.0 - 20.0) / 21.0)
            .collect::<Vec<_>>(),
        &[batch, columns],
    )
    .copy(stream)
    .unwrap();

    let native_output = native.linear(&input, true, stream).unwrap();
    let affine_output = quantized_matmul_with_mode(
        &input,
        &affine_weight,
        &affine_scales,
        Some(&affine_biases),
        true,
        32,
        8,
        QuantizationMode::Affine,
        stream,
    )
    .unwrap();
    eval([&native_output, &affine_output]).unwrap();
    let native_values = native_output.evaluated().unwrap();
    let affine_float = affine_output.as_dtype(Dtype::Float32, stream).unwrap();
    let affine_values = affine_float.evaluated().unwrap();
    let (max_abs, rms) = native_values
        .as_slice::<f32>()
        .iter()
        .zip(affine_values.as_slice::<f32>())
        .fold((0.0f32, 0.0f64), |(max_abs, sum_sq), (native, affine)| {
            let error = (native - affine).abs();
            (max_abs.max(error), sum_sq + f64::from(error * error))
        });
    let rms = (rms / native_output.size() as f64).sqrt();

    let native_time = time_average(iterations, stream, || {
        native.linear(&input, true, stream).unwrap()
    });
    let affine_time = time_average(iterations, stream, || {
        quantized_matmul_with_mode(
            &input,
            &affine_weight,
            &affine_scales,
            Some(&affine_biases),
            true,
            32,
            8,
            QuantizationMode::Affine,
            stream,
        )
        .unwrap()
    });
    println!(
        "{rows:6}x{columns:5} batch={batch:2} native={:7.3} ms {:6.1} GB/s affine={:7.3} ms {:6.1} GB/s speedup={:5.2}x raw={:9} affine={:9} max_abs={max_abs:.6} rms={rms:.6}",
        native_time.as_secs_f64() * 1e3,
        raw.len() as f64 / native_time.as_secs_f64() / 1e9,
        affine_time.as_secs_f64() * 1e3,
        affine_bytes as f64 / affine_time.as_secs_f64() / 1e9,
        affine_time.as_secs_f64() / native_time.as_secs_f64(),
        raw.len(),
        affine_bytes,
    );
}

fn main() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    memory::clear_cache().unwrap();
    memory::reset_peak_memory().unwrap();
    println!("shape              native latency/bandwidth       affine latency/bandwidth");
    for (rows, columns, batch, iterations) in [
        (4096, 2816, 1, 50),
        (2816, 4096, 1, 50),
        (2112, 2816, 1, 60),
        (2816, 2112, 1, 60),
        (8192, 2816, 1, 30),
        (262144, 2816, 1, 10),
        (4096, 2816, 16, 12),
        (2112, 2816, 16, 15),
    ] {
        benchmark_shape(rows, columns, batch, iterations, &stream);
    }
    stream.synchronize().unwrap();
    println!(
        "mlx_active_bytes={} mlx_peak_bytes={}",
        memory::active_memory().unwrap(),
        memory::peak_memory().unwrap()
    );
}
