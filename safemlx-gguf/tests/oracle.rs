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
        }
    }
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
