//! This contains the tests for some of the exported macros.
//!
//! This is mainly a sanity check to ensure that the exported macros are working as expected.

use safemlx::{
    array, complex64,
    error::Exception,
    ops::{arange, reshape},
    random, Array, Dtype,
};

mod common;

// Try two functions that don't have any optional arguments.

#[test]
fn test_ops_arithmetic_abs() {
    let data = array!([1i32, 2, -3, -4, -5]);
    let stream =
        safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let result = safemlx::abs!(&data, stream = &stream).unwrap();

    assert_eq!(common::eval_vec::<i32>(&result), &[1, 2, 3, 4, 5]);

    let result = safemlx::abs!(data, stream = &stream).unwrap();

    assert_eq!(common::eval_vec::<i32>(&result), &[1, 2, 3, 4, 5]);
}

#[test]
fn test_ops_arithmetic_add() {
    let data1 = array!([1i32, 2, 3, 4, 5]);
    let data2 = array!([1i32, 2, 3, 4, 5]);
    let stream =
        safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let result = safemlx::add!(&data1, &data2, stream = &stream).unwrap();

    assert_eq!(common::eval_vec::<i32>(&result), &[2, 4, 6, 8, 10]);

    let result = safemlx::add!(data1, data2, stream = &stream).unwrap();

    assert_eq!(common::eval_vec::<i32>(&result), &[2, 4, 6, 8, 10]);
}

// Try a function that has optional arguments.

#[test]
fn test_ops_arithmetic_tensordot() {
    let stream =
        safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let x = reshape(
        arange::<_, f32>(None, 60.0, None, &stream).unwrap(),
        &[3, 4, 5],
        &stream,
    )
    .unwrap();
    let y = reshape(
        arange::<_, f32>(None, 24.0, None, &stream).unwrap(),
        &[4, 3, 2],
        &stream,
    )
    .unwrap();
    let axes_x = [1, 0];
    let axes_y = [0, 1];
    let z = safemlx::tensordot_axes!(&x, &y, &axes_x, &axes_y, stream = &stream).unwrap();
    let expected = Array::from_slice(
        &[
            4400.0, 4730.0, 4532.0, 4874.0, 4664.0, 5018.0, 4796.0, 5162.0, 4928.0, 5306.0,
        ],
        &[5, 2],
    );
    assert!(common::eval_equal_values(&z, &expected));

    let z = safemlx::tensordot_axes!(&x, &y, &axes_x, &axes_y, stream = &stream).unwrap();
    assert!(common::eval_equal_values(&z, &expected));
}

// Test functions defined in `safemlx::ops` module.

#[test]
fn test_ops_convolution_conv1d() {
    let input = array!(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        shape = [1, 5, 2]
    );
    let weight = array!(
        [0.5, 0.0, -0.5, 1.0, 0.0, 1.5, 2.0, 0.0, -2.0, 1.5, 0.0, 1.0],
        shape = [2, 3, 2]
    );

    let stream =
        safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let result = safemlx::conv1d!(
        &input,
        &weight,
        stride = 1,
        padding = 0,
        dilation = 1,
        groups = 1,
        stream = &stream
    )
    .unwrap();

    let expected = array!([12.0, 8.0, 17.0, 13.0, 22.0, 18.0], shape = [1, 3, 2]);
    assert!(common::eval_equal_values(&result, &expected));
}

#[test]
fn test_ops_factory_arange() {
    // Without specifying start and step
    let stream =
        safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let array = safemlx::arange!(stop = 50, stream = &stream).unwrap();
    assert_eq!(array.shape(), &[50]);
    assert_eq!(array.dtype(), Dtype::Float32);

    let data: Vec<f32> = common::eval_vec(&array);
    let expected: Vec<f32> = (0..50).map(|x| x as f32).collect();
    assert_eq!(data, expected);

    // With specifying start and step
    let array = safemlx::arange!(start = 1.0, stop = 50.0, step = 2.0, stream = &stream).unwrap();
    assert_eq!(array.shape(), &[25]);
    assert_eq!(array.dtype(), Dtype::Float32);

    let data: Vec<f32> = common::eval_vec(&array);
    let expected: Vec<f32> = (1..50).step_by(2).map(|x| x as f32).collect();
    assert_eq!(data, expected);

    let array = safemlx::arange!(start = 1.0, stop = 50.0, step = 2.0, stream = &stream).unwrap();
    assert_eq!(array.shape(), &[25]);
    assert_eq!(array.dtype(), Dtype::Float32);

    let data: Vec<f32> = common::eval_vec(&array);
    let expected: Vec<f32> = (1..50).step_by(2).map(|x| x as f32).collect();
    assert_eq!(data, expected);
}

// Test functions defined in `safemlx::fft` module.

#[test]
fn test_fft_fft() {
    const FFT_EXPECTED: &[complex64; 4] = &[
        complex64::new(10.0, 0.0),
        complex64::new(-2.0, 2.0),
        complex64::new(-2.0, 0.0),
        complex64::new(-2.0, -2.0),
    ];

    let data = array!([1.0, 2.0, 3.0, 4.0]);
    let stream =
        safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let fft = safemlx::fft!(&data, stream = &stream).unwrap();

    assert_eq!(fft.dtype(), Dtype::Complex64);
    assert_eq!(common::eval_vec::<complex64>(&fft), FFT_EXPECTED);
}

// Test functions defined in `safemlx::linalg` module.

#[test]
fn test_linalg_norm() {
    let stream =
        safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let a = array!([1.0, 2.0, 3.0, 4.0])
        .reshape(&[2, 2], &stream)
        .unwrap();
    let norm = safemlx::norm_l2!(&a, stream = &stream).unwrap();
    assert_eq!(norm.item::<f32>(&stream), 5.477_226);
}

// Test functions defined in `safemlx::random` module.

#[test]
fn test_random_uniform() {
    let stream =
        safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let key = random::key(0).unwrap();
    let value = safemlx::uniform!(0.0, 1.0, shape = &[1], key = &key, stream = &stream).unwrap();
    assert_eq!(value.shape(), &[1]);
    let value = value.item::<f32>(&stream);
    assert!((0.0..=1.0).contains(&value));
}

#[test]
fn test_random_normal() {
    let stream =
        safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let key = random::key(1).unwrap();
    let value = safemlx::normal!(shape = &[1], key = &key, stream = &stream).unwrap();
    assert_eq!(value.shape(), &[1]);
    let value = value.item::<f32>(&stream);
    assert!((-10.0..=10.0).contains(&value));
}

// Test functions defined in `safemlx::fast` module.

#[test]
#[allow(non_snake_case)]
fn test_fast_sdpa_using_macros() -> Result<(), Exception> {
    // This test just makes sure that `scaled_dot_product_attention` is callable
    // in the various cases, based on the Python test `test_fast_sdpa`.

    let Dk = 64;
    let scale = 1.0 / (Dk as f32).sqrt();
    let stream =
        safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    for seq_len in [63, 129, 400] {
        for dtype in [crate::Dtype::Float32, crate::Dtype::Float16] {
            let B = 2;
            let H = 24;
            let key_q = random::key(seq_len as u64).unwrap();
            let key_k = random::key(seq_len as u64 + 1).unwrap();
            let key_v = random::key(seq_len as u64 + 2).unwrap();
            let q = safemlx::normal!(shape = &[B, H, seq_len, Dk], key = &key_q, stream = &stream)?
                .as_dtype(dtype, &stream)?;
            let k = safemlx::normal!(shape = &[B, H, seq_len, Dk], key = &key_k, stream = &stream)?
                .as_dtype(dtype, &stream)?;
            let v = safemlx::normal!(shape = &[B, H, seq_len, Dk], key = &key_v, stream = &stream)?
                .as_dtype(dtype, &stream)?;

            let result = safemlx::scaled_dot_product_attention!(q, k, v, scale, stream = &stream)?;
            assert_eq!(result.shape(), [B, H, seq_len, Dk]);
            assert_eq!(result.dtype(), dtype);
        }
    }

    Ok(())
}
