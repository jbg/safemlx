use safemlx_gguf::{ConvertedTensor, GgmlType, Reader, TensorInput, Writer};
use std::collections::BTreeMap;
use std::io::Cursor;

fn unhex(s: &str) -> Vec<u8> {
    assert_eq!(s.len() % 2, 0);
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}
fn bytes_u32(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn bytes_u16(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn parse_field(field: &str) -> (&str, &str, &str) {
    let mut p = field.rsplitn(3, ':');
    let data = p.next().unwrap();
    let dtype = p.next().unwrap();
    let prefix = p.next().unwrap();
    (prefix, dtype, data)
}

#[test]
fn matches_patched_mlx_v032_oracle_byte_for_byte() {
    for line in include_str!("fixtures/mlx-v0.32.0.oracle").lines() {
        let fields: Vec<_> = line.split('|').collect();
        let code: u32 = fields[0].parse().unwrap();
        let ty = GgmlType::from_code(code);
        let raw = unhex(fields[1]);
        let (block, _) = ty.block_and_bytes().unwrap();
        let mut encoded = Cursor::new(Vec::new());
        Writer::default()
            .write(
                &mut encoded,
                &BTreeMap::new(),
                &[TensorInput {
                    name: "oracle.weight",
                    dimensions: &[block, 2],
                    ggml_type: ty,
                    data: &raw,
                }],
            )
            .unwrap();
        let mut reader = Reader::new(Cursor::new(encoded.into_inner())).unwrap();
        let desc = reader.tensors()[0].clone();
        let converted = reader.read_tensor(&desc).unwrap();
        match converted {
            ConvertedTensor::Affine(a) => {
                if matches!(ty, GgmlType::Q5_0 | GgmlType::Q5_1) {
                    assert_eq!(fields.len(), 3, "format {code}");
                    assert_eq!(a.bits, 5);
                    assert_eq!(a.group_size, 32);
                    let (name, dtype, data) = parse_field(fields[2]);
                    assert_eq!(name, "oracle.weight:[2, 32]");
                    assert_eq!(dtype, "Float16");
                    let expected: Vec<f32> = unhex(data)
                        .chunks_exact(2)
                        .map(|b| {
                            half::f16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32()
                        })
                        .collect();
                    for (i, (actual, expected)) in
                        a.dequantize().into_iter().zip(expected).enumerate()
                    {
                        let actual = half::f16::from_f32(actual).to_f32();
                        assert_eq!(actual, expected, "legacy Q5 dequant at index {i}");
                    }
                    continue;
                }
                assert_eq!(fields.len(), 6, "format {code}");
                let (name, dtype, w) = parse_field(fields[2]);
                assert!(name.starts_with("oracle.weight:[2, "));
                assert_eq!(dtype, "Uint32");
                assert_eq!(bytes_u32(&a.weights), unhex(w), "weights for {code}");
                let (name, dtype, s) = parse_field(fields[3]);
                assert!(name.starts_with("oracle.scales:[2, "));
                assert_eq!(dtype, "Float16");
                assert_eq!(bytes_u16(&a.scales), unhex(s), "scales for {code}");
                let (name, dtype, b) = parse_field(fields[4]);
                assert!(name.starts_with("oracle.biases:[2, "));
                assert_eq!(dtype, "Float16");
                assert_eq!(bytes_u16(&a.biases), unhex(b), "biases for {code}");
                let (name, dtype, d) = parse_field(fields[5]);
                assert_eq!(
                    name,
                    "oracle.dequantized:[2, 32]".replace("32", &block.to_string())
                );
                assert_eq!(dtype, "Float16");
                let expected: Vec<f32> = unhex(d)
                    .chunks_exact(2)
                    .map(|b| {
                        half::f16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32()
                    })
                    .collect();
                let actual = a.dequantize();
                assert_eq!(actual.len(), expected.len());
                for (i, (x, y)) in actual.iter().zip(expected).enumerate() {
                    let rounded = half::f16::from_f32(*x).to_f32(); /* MLX's f16 kernel may round the multiply and add separately; allow four f16 ULPs at this fixture's largest magnitudes. */
                    let tolerance = 4.0f32.max(y.abs() * 0.002);
                    assert!(
                        (rounded - y).abs() <= tolerance,
                        "dequant {code} index {i}: {rounded} != {y} (tol {tolerance})"
                    );
                }
            }
            ConvertedTensor::Dense(d) => {
                assert_eq!(fields.len(), 3);
                let (name, dtype, data) = parse_field(fields[2]);
                assert_eq!(name, "oracle.weight:[2, 32]");
                assert_eq!(dtype, "Float16");
                assert_eq!(d.data, unhex(data), "legacy dense output for {code}");
            }
            ConvertedTensor::IQuant(_) => {
                panic!("legacy affine oracle does not contain IQ tensors")
            }
        }
    }
}

fn q5_block(ty: GgmlType, scale: f32, bias: f32, codes: &[u8; 32]) -> Vec<u8> {
    let is_q5_0 = ty == GgmlType::Q5_0;
    let mut block = vec![0u8; if is_q5_0 { 22 } else { 24 }];
    block[..2].copy_from_slice(&half::f16::from_f32(scale).to_bits().to_le_bytes());
    if !is_q5_0 {
        block[2..4].copy_from_slice(&half::f16::from_f32(bias).to_bits().to_le_bytes());
    }
    let qh_offset = if is_q5_0 { 2 } else { 4 };
    let qs_offset = if is_q5_0 { 6 } else { 8 };
    let mut qh = 0u32;
    for j in 0..16 {
        assert!(codes[j] < 32 && codes[j + 16] < 32);
        qh |= u32::from(codes[j] >> 4) << j;
        qh |= u32::from(codes[j + 16] >> 4) << (j + 16);
        block[qs_offset + j] = (codes[j] & 15) | ((codes[j + 16] & 15) << 4);
    }
    block[qh_offset..qh_offset + 4].copy_from_slice(&qh.to_le_bytes());
    block
}

fn read_q5(ty: GgmlType, blocks: &[Vec<u8>]) -> safemlx_gguf::AffineTensor {
    let raw: Vec<u8> = blocks.iter().flatten().copied().collect();
    let mut file = Cursor::new(Vec::new());
    Writer::default()
        .write(
            &mut file,
            &BTreeMap::new(),
            &[TensorInput {
                name: "q5.weight",
                dimensions: &[32, blocks.len() as u64],
                ggml_type: ty,
                data: &raw,
            }],
        )
        .unwrap();
    let mut reader = Reader::new(Cursor::new(file.into_inner())).unwrap();
    let descriptor = reader.tensors()[0].clone();
    let ConvertedTensor::Affine(affine) = reader.read_tensor(&descriptor).unwrap() else {
        panic!("Q5 tensors must remain packed affine tensors");
    };
    affine
}

fn unpack_codes(weights: &[u32], count: usize) -> Vec<u8> {
    (0..count)
        .map(|index| {
            let offset = index * 5;
            let word = offset / 32;
            let shift = offset % 32;
            let mut code = weights[word] >> shift;
            if shift + 5 > 32 {
                code |= weights[word + 1] << (32 - shift);
            }
            (code & 31) as u8
        })
        .collect()
}

fn assert_q5_repacked(ty: GgmlType, scales: [f32; 2], biases: [f32; 2]) {
    let codes0 = std::array::from_fn(|i| i as u8);
    let codes1 = std::array::from_fn(|i| ((i * 13 + 7) & 31) as u8);
    let codes = [codes0, codes1];
    let blocks: Vec<_> = codes
        .iter()
        .enumerate()
        .map(|(i, codes)| q5_block(ty, scales[i], biases[i], codes))
        .collect();
    let affine = read_q5(ty, &blocks);

    assert_eq!(affine.bits, 5);
    assert_eq!(affine.group_size, 32);
    assert_eq!(affine.weight_shape, [2, 5]);
    assert_eq!(affine.scale_shape, [2, 1]);
    assert_eq!(affine.weights.len(), 10);
    assert_eq!(affine.scales.len(), 2);
    assert_eq!(affine.biases.len(), 2);
    assert_eq!(
        unpack_codes(&affine.weights, 64),
        codes.into_iter().flatten().collect::<Vec<_>>()
    );

    let actual = affine.dequantize();
    for block in 0..2 {
        let d = half::f16::from_f32(scales[block]).to_f32();
        let m = half::f16::from_f32(biases[block]).to_f32();
        for (index, code) in codes[block].iter().copied().enumerate() {
            let code = i32::from(code);
            let expected = if ty == GgmlType::Q5_0 {
                d * (code - 16) as f32
            } else {
                d * code as f32 + m
            };
            let position = block * 32 + index;
            assert_eq!(actual[position], expected, "block {block} code {index}");
        }
    }
}

#[test]
fn q5_0_repacking_preserves_signed_values_and_gguf_bit_order() {
    assert_q5_repacked(GgmlType::Q5_0, [0.5, -1.25], [0.0; 2]);
}

#[test]
fn q5_1_repacking_preserves_bias_and_gguf_bit_order() {
    assert_q5_repacked(GgmlType::Q5_1, [0.25, 1.75], [1.5, -0.75]);
}

#[test]
fn boundary_blocks_cover_zero_and_extreme_codes() {
    for (ty, size, header, word) in [
        (GgmlType::Q4_0, 18, 2, u32::MAX),
        (GgmlType::Q8_0, 34, 2, 0x7f7f7f7f),
    ] {
        let (block, _) = ty.block_and_bytes().unwrap();
        let mut raw = vec![0u8; size];
        raw[..2].copy_from_slice(&half::f16::ONE.to_bits().to_le_bytes());
        for b in &mut raw[header..] {
            *b = 0xff;
        }
        let mut file = Cursor::new(Vec::new());
        Writer::default()
            .write(
                &mut file,
                &BTreeMap::new(),
                &[TensorInput {
                    name: "edge.weight",
                    dimensions: &[block],
                    ggml_type: ty,
                    data: &raw,
                }],
            )
            .unwrap();
        let mut r = Reader::new(Cursor::new(file.into_inner())).unwrap();
        let d = r.tensors()[0].clone();
        let ConvertedTensor::Affine(a) = r.read_tensor(&d).unwrap() else {
            panic!()
        };
        assert!(a.weights.iter().all(|&w| w == word));
        assert_eq!(a.dequantize().len(), block as usize);
    }
}
