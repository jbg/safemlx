use crate::Array;

pub(super) fn zeros(shape: &[i32]) -> Array {
    let len = element_count(shape);
    Array::from_slice(&vec![0.0f32; len], shape)
}

pub(super) fn ones(shape: &[i32]) -> Array {
    let len = element_count(shape);
    Array::from_slice(&vec![1.0f32; len], shape)
}

pub(super) fn uniform(low: f32, high: f32, shape: &[i32]) -> Array {
    let len = element_count(shape);
    let span = high - low;
    let mut state = seed_from_shape(shape, low, high);
    let mut values = Vec::with_capacity(len);
    for _ in 0..len {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let unit = ((state >> 40) as f32) / ((1u64 << 24) as f32);
        values.push(low + span * unit);
    }
    Array::from_slice(&values, shape)
}

fn element_count(shape: &[i32]) -> usize {
    shape
        .iter()
        .map(|&dim| usize::try_from(dim).expect("negative dimensions are not supported"))
        .product()
}

fn seed_from_shape(shape: &[i32], low: f32, high: f32) -> u64 {
    let mut seed = 0x9e3779b97f4a7c15u64;
    for &dim in shape {
        seed ^= dim as u64;
        seed = seed.rotate_left(27).wrapping_mul(0x94d049bb133111ebu64);
    }
    seed ^ u64::from(low.to_bits()) ^ (u64::from(high.to_bits()) << 1)
}
