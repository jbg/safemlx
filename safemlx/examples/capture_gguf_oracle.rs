//! Regenerate Stage 0 fixtures while checked out before the Rust integration.
//! The legacy `Array::load_gguf` call below must still route through patched MLX.
use safemlx::{Array, Device, DeviceType, Dtype, Stream};
use safemlx_gguf::{GgmlType, TensorInput, Writer};
use std::collections::BTreeMap;
use std::fs::File;

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
fn bytes(array: &Array) -> Vec<u8> {
    let evaluated = array.evaluated().unwrap();
    match evaluated.as_array().dtype() {
        Dtype::Uint32 => evaluated
            .as_slice::<u32>()
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        Dtype::Float16 => evaluated
            .as_slice::<half::f16>()
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect(),
        Dtype::Float32 => evaluated
            .as_slice::<f32>()
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect(),
        other => panic!("unexpected oracle dtype {other:?}"),
    }
}
fn put_half(raw: &mut [u8], offset: usize, bits: u16) {
    raw[offset..offset + 2].copy_from_slice(&bits.to_le_bytes())
}
fn main() {
    let formats = [
        GgmlType::Q4_0,
        GgmlType::Q4_1,
        GgmlType::Q8_0,
        GgmlType::Q2K,
        GgmlType::Q3K,
        GgmlType::Q4K,
        GgmlType::Q5K,
        GgmlType::Q6K,
        GgmlType::Q5_0,
        GgmlType::Q5_1,
    ];
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    for ty in formats {
        let (block, size) = ty.block_and_bytes().unwrap();
        let mut raw = vec![0u8; (size * 2) as usize];
        for (i, b) in raw.iter_mut().enumerate() {
            *b = ((i * 37 + ty.code() as usize * 11) & 255) as u8;
        }
        for n in 0..2 {
            let base = n * size as usize;
            let a = if n == 0 { 0x3c00 } else { 0xb800 };
            put_half(&mut raw, base, a);
            match ty {
                GgmlType::Q4_1 | GgmlType::Q5_1 => {
                    put_half(&mut raw, base + 2, if n == 0 { 0x3400 } else { 0xbc00 })
                }
                GgmlType::Q2K => {
                    put_half(&mut raw, base + 80, a);
                    put_half(&mut raw, base + 82, 0x3800)
                }
                GgmlType::Q3K => put_half(&mut raw, base + 108, a),
                GgmlType::Q4K | GgmlType::Q5K => put_half(&mut raw, base + 2, 0x3800),
                GgmlType::Q6K => put_half(&mut raw, base + 208, a),
                _ => {}
            }
        }
        let path = std::env::temp_dir().join(format!("safemlx-oracle-{}.gguf", ty.code()));
        Writer::default()
            .write(
                File::create(&path).unwrap(),
                &BTreeMap::new(),
                &[TensorInput {
                    name: "oracle.weight",
                    dimensions: &[block, 2],
                    ggml_type: ty,
                    data: &raw,
                }],
            )
            .unwrap();
        let arrays = Array::load_gguf(&path, &stream).unwrap();
        print!("{}|{}", ty.code(), hex(&raw));
        for name in ["oracle.weight", "oracle.scales", "oracle.biases"] {
            if let Some(a) = arrays.get(name) {
                print!(
                    "|{}:{:?}:{:?}:{}",
                    name,
                    a.shape(),
                    a.dtype(),
                    hex(&bytes(a))
                );
            }
        }
        if let (Some(w), Some(s), Some(b)) = (
            arrays.get("oracle.weight"),
            arrays.get("oracle.scales"),
            arrays.get("oracle.biases"),
        ) {
            let (bits, group) = match ty {
                GgmlType::Q2K => (2, 16),
                GgmlType::Q3K => (3, 16),
                GgmlType::Q4_0 | GgmlType::Q4_1 | GgmlType::Q4K => (4, 32),
                GgmlType::Q5K => (5, 32),
                GgmlType::Q6K => (6, 16),
                GgmlType::Q8_0 => (8, 32),
                _ => unreachable!(),
            };
            let d =
                safemlx::ops::dequantize(w, s, Some(b), Some(group), Some(bits), &stream).unwrap();
            print!(
                "|oracle.dequantized:{:?}:{:?}:{}",
                d.shape(),
                d.dtype(),
                hex(&bytes(&d))
            );
        }
        println!();
    }
}
