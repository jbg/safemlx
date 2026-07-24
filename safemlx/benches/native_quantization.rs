use std::time::{Duration, Instant};

use safemlx::{
    native_quantization::NativeQuantizedTensor, ops::matmul, transforms::eval, Array, Device,
    DeviceType, Dtype, Stream,
};
use safemlx_gguf::{Endian, GgmlType};

fn env_i32(name: &str, default: i32) -> i32 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn sample_q4k_block() -> Vec<u8> {
    let mut block = vec![0u8; 144];
    block[..2].copy_from_slice(&half::f16::from_f32(0.125).to_bits().to_le_bytes());
    block[2..4].copy_from_slice(&half::f16::from_f32(0.25).to_bits().to_le_bytes());
    for (index, byte) in block[4..].iter_mut().enumerate() {
        *byte = (index as u8).wrapping_mul(29).wrapping_add(11);
    }
    block
}

fn sample_q5_1_block() -> Vec<u8> {
    let mut block = vec![0u8; 24];
    block[..2].copy_from_slice(&half::f16::from_f32(0.0625).to_bits().to_le_bytes());
    block[2..4].copy_from_slice(&half::f16::from_f32(-0.5).to_bits().to_le_bytes());
    for (index, byte) in block[4..].iter_mut().enumerate() {
        *byte = (index as u8).wrapping_mul(37).wrapping_add(7);
    }
    block
}

fn sample_q8_0_block() -> Vec<u8> {
    let mut block = vec![0u8; 34];
    block[..2].copy_from_slice(&half::f16::from_f32(0.015625).to_bits().to_le_bytes());
    for (index, byte) in block[2..].iter_mut().enumerate() {
        *byte = (index as u8).wrapping_mul(13).wrapping_add(3);
    }
    block
}

fn sample_iq4_nl_block() -> Vec<u8> {
    let mut block = vec![0u8; 18];
    block[..2].copy_from_slice(&half::f16::from_f32(0.125).to_bits().to_le_bytes());
    for (index, byte) in block[2..].iter_mut().enumerate() {
        *byte = (index as u8).wrapping_mul(17).wrapping_add(5);
    }
    block
}

fn repeated_blocks(block: &[u8], count: usize) -> Vec<u8> {
    let mut raw = Vec::with_capacity(block.len() * count);
    for index in 0..count {
        let mut block = block.to_vec();
        let last = block.len() - 1;
        block[last] = block[last].wrapping_add(index as u8);
        raw.extend(block);
    }
    raw
}

fn measure(mut operation: impl FnMut() -> Array, stream: &Stream, iterations: usize) -> Duration {
    for _ in 0..3 {
        let output = operation();
        eval([&output]).unwrap();
        stream.synchronize().unwrap();
    }
    let mut samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        let output = operation();
        eval([&output]).unwrap();
        stream.synchronize().unwrap();
        std::hint::black_box(output);
        samples.push(start.elapsed());
    }
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn main() {
    let rows = env_i32("SAFEMLX_BENCH_ROWS", 512);
    let columns = env_i32("SAFEMLX_BENCH_COLUMNS", 2048);
    let iterations = env_i32("SAFEMLX_BENCH_ITERS", 10).max(1) as usize;
    assert!(columns % 256 == 0, "columns must be divisible by 256");

    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let q4_raw = repeated_blocks(&sample_q4k_block(), (rows * columns / 256) as usize);
    let q5_raw = repeated_blocks(&sample_q5_1_block(), (rows * columns / 32) as usize);
    let q8_raw = repeated_blocks(&sample_q8_0_block(), (rows * columns / 32) as usize);
    let iq4_raw = repeated_blocks(&sample_iq4_nl_block(), (rows * columns / 32) as usize);
    let iq4_bytes = Array::from_slice(
        &iq4_raw,
        &[rows, columns / 32 * sample_iq4_nl_block().len() as i32],
    )
    .copy(&stream)
    .unwrap();
    let cases = vec![
        (
            "Q4_K",
            NativeQuantizedTensor::from_q4k_bytes(&q4_raw, &[rows, columns], &stream).unwrap(),
            q4_raw.len(),
        ),
        (
            "Q5_1",
            NativeQuantizedTensor::from_q5_1_bytes(&q5_raw, &[rows, columns], &stream).unwrap(),
            q5_raw.len(),
        ),
        (
            "Q8_0",
            NativeQuantizedTensor::from_q8_0_bytes(&q8_raw, &[rows, columns], &stream).unwrap(),
            q8_raw.len(),
        ),
        (
            "IQ4_NL",
            NativeQuantizedTensor::from_iq_array(
                iq4_bytes,
                &[rows, columns],
                GgmlType::IQ4NL,
                Endian::Little,
            )
            .unwrap(),
            iq4_raw.len(),
        ),
    ];
    let decode_input = Array::from_slice(
        &(0..columns)
            .map(|index| (index as f32 % 31.0 - 15.0) / 32.0)
            .collect::<Vec<_>>(),
        &[1, columns],
    )
    .as_dtype(Dtype::Float16, &stream)
    .unwrap();
    let prefill_rows = 32;
    let prefill_input = Array::from_slice(
        &(0..prefill_rows * columns)
            .map(|index| (index as f32 % 37.0 - 18.0) / 40.0)
            .collect::<Vec<_>>(),
        &[prefill_rows, columns],
    )
    .as_dtype(Dtype::Float16, &stream)
    .unwrap();

    println!(
        "native quantization: rows={rows}, columns={columns}, iterations={iterations} (median)"
    );
    for (name, native, packed_bytes) in cases {
        let dense = native
            .dequantize(&stream)
            .unwrap()
            .as_dtype(Dtype::Float16, &stream)
            .unwrap();
        eval([&dense]).unwrap();
        stream.synchronize().unwrap();
        let dense_transposed = dense.transpose(&stream).unwrap();
        let native_decode = measure(
            || native.linear(&decode_input, true, &stream).unwrap(),
            &stream,
            iterations,
        );
        let dense_decode = measure(
            || matmul(&decode_input, &dense_transposed, &stream).unwrap(),
            &stream,
            iterations,
        );
        let native_prefill = measure(
            || native.linear(&prefill_input, true, &stream).unwrap(),
            &stream,
            iterations,
        );
        let dense_prefill = measure(
            || matmul(&prefill_input, &dense_transposed, &stream).unwrap(),
            &stream,
            iterations,
        );
        let dense_bytes = rows as usize * columns as usize * 2;
        println!(
            "{name:7} packed={:7.2} MiB dense-f16={:7.2} MiB | decode packed={:8.3} ms dense={:8.3} ms | prefill32 packed={:8.3} ms dense={:8.3} ms",
            packed_bytes as f64 / 1_048_576.0,
            dense_bytes as f64 / 1_048_576.0,
            native_decode.as_secs_f64() * 1e3,
            dense_decode.as_secs_f64() * 1e3,
            native_prefill.as_secs_f64() * 1e3,
            dense_prefill.as_secs_f64() * 1e3,
        );
    }
}
