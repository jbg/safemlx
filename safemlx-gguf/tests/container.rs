use safemlx_gguf::{
    ConvertedTensor, DenseDtype, Endian, GgmlType, Limits, MetadataArray as A, MetadataValue as V,
    Reader, TensorInput, Writer, WriterOptions,
};
use std::collections::BTreeMap;
use std::io::{self, Cursor, Seek, SeekFrom, Write};

fn metadata() -> BTreeMap<String, V> {
    BTreeMap::from([
        ("a.u8".into(), V::Uint8(255)),
        ("b.i8".into(), V::Int8(-3)),
        ("c.u16".into(), V::Uint16(500)),
        ("d.i16".into(), V::Int16(-500)),
        ("e.u32".into(), V::Uint32(90_000)),
        ("f.i32".into(), V::Int32(-90_000)),
        ("g.f32".into(), V::Float32(1.25)),
        ("h.bool".into(), V::Bool(true)),
        ("i.string".into(), V::String("héllo".into())),
        ("j.u64".into(), V::Uint64(u32::MAX as u64 + 2)),
        ("k.i64".into(), V::Int64(-9_000_000_000)),
        ("l.f64".into(), V::Float64(-2.5)),
        ("m.array".into(), V::Array(A::Int16(vec![-1, 0, 2]))),
        (
            "n.nested".into(),
            V::Array(A::Array(vec![
                A::String(vec!["x".into()]),
                A::String(vec!["y".into(), "z".into()]),
            ])),
        ),
        ("o.empty".into(), V::Array(A::Uint8(vec![]))),
        ("p.au8".into(), V::Array(A::Uint8(vec![0, 255]))),
        ("q.ai8".into(), V::Array(A::Int8(vec![-1, 2]))),
        ("r.au16".into(), V::Array(A::Uint16(vec![1, 500]))),
        ("s.au32".into(), V::Array(A::Uint32(vec![1, 90_000]))),
        ("t.ai32".into(), V::Array(A::Int32(vec![-1, 90_000]))),
        ("u.af32".into(), V::Array(A::Float32(vec![-1.5, 2.25]))),
        ("v.abool".into(), V::Array(A::Bool(vec![false, true]))),
        ("w.au64".into(), V::Array(A::Uint64(vec![1, u64::MAX]))),
        ("x.ai64".into(), V::Array(A::Int64(vec![i64::MIN, 1]))),
        ("y.af64".into(), V::Array(A::Float64(vec![-1.5, 2.25]))),
    ])
}

fn file(version: u32, endian: Endian, alignment: u64) -> Vec<u8> {
    let dense = match endian {
        Endian::Little => 1.5f32.to_le_bytes(),
        Endian::Big => 1.5f32.to_be_bytes(),
    };
    let q = vec![0u8; 36];
    let mut out = Cursor::new(Vec::new());
    Writer::new(WriterOptions {
        version,
        endian,
        alignment,
    })
    .unwrap()
    .write(
        &mut out,
        &metadata(),
        &[
            TensorInput {
                name: "dense",
                dimensions: &[1],
                ggml_type: GgmlType::F32,
                data: &dense,
            },
            TensorInput {
                name: "zero",
                dimensions: &[0],
                ggml_type: GgmlType::F16,
                data: &[],
            },
            TensorInput {
                name: "raw.weight",
                dimensions: &[64],
                ggml_type: GgmlType::Q4_0,
                data: &q,
            },
        ],
    )
    .unwrap();
    out.into_inner()
}

#[test]
fn versions_endian_metadata_dense_and_alignment() {
    for version in 1..=3 {
        for endian in [Endian::Little, Endian::Big] {
            let bytes = file(version, endian, 64);
            let mut r = Reader::new(Cursor::new(bytes)).unwrap();
            assert_eq!(r.version(), version);
            assert_eq!(r.endian(), endian);
            assert_eq!(r.alignment(), 64);
            let mut expected = metadata();
            expected.insert("general.alignment".into(), V::Uint32(64));
            assert_eq!(r.metadata(), &expected);
            assert_eq!(r.tensors().len(), 3);
            assert!(r.tensors().iter().all(|t| t.data_offset % 64 == 0));
            let d = r.tensors()[0].clone();
            match r.read_tensor(&d).unwrap() {
                ConvertedTensor::Dense(d) => {
                    assert_eq!(d.dtype, DenseDtype::F32);
                    assert_eq!(d.data, 1.5f32.to_ne_bytes());
                }
                _ => panic!(),
            }
        }
    }
}

#[test]
fn metadata_only_file_does_not_require_tensor_section_padding() {
    let mut bytes = b"GGUF".to_vec();
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());

    let reader = Reader::new(Cursor::new(bytes)).unwrap();
    assert!(reader.tensors().is_empty());
    assert!(reader.metadata().is_empty());
}

#[test]
fn deterministic_and_raw_quantized_roundtrip() {
    let a = file(3, Endian::Little, 32);
    let b = file(3, Endian::Little, 32);
    assert_eq!(a, b);
    let mut r = Reader::new(Cursor::new(a)).unwrap();
    let t = r.tensors()[2].clone();
    assert_eq!(r.read_raw(&t).unwrap(), vec![0; 36]);
}

#[test]
fn every_supported_raw_type_has_checked_size() {
    for ty in [
        GgmlType::Q4_0,
        GgmlType::Q4_1,
        GgmlType::Q5_0,
        GgmlType::Q5_1,
        GgmlType::Q8_0,
        GgmlType::Q2K,
        GgmlType::Q3K,
        GgmlType::Q4K,
        GgmlType::Q5K,
        GgmlType::Q6K,
    ] {
        let (block, size) = ty.block_and_bytes().unwrap();
        let data = vec![0; size as usize];
        let mut out = Cursor::new(Vec::new());
        Writer::default()
            .write(
                &mut out,
                &BTreeMap::new(),
                &[TensorInput {
                    name: "x.weight",
                    dimensions: &[block],
                    ggml_type: ty,
                    data: &data,
                }],
            )
            .unwrap();
        let mut r = Reader::new(Cursor::new(out.into_inner())).unwrap();
        let d = r.tensors()[0].clone();
        assert_eq!(r.read_raw(&d).unwrap(), data);
        assert!(r.read_tensor(&d).is_ok());
    }
}

#[test]
fn all_truncations_are_errors_not_panics() {
    let bytes = file(3, Endian::Little, 32);
    for n in 0..bytes.len() {
        let result = std::panic::catch_unwind(|| Reader::new(Cursor::new(&bytes[..n])));
        assert!(result.is_ok(), "panic at {n}");
        assert!(result.unwrap().is_err(), "accepted truncation at {n}");
    }
}

#[test]
fn limits_and_invalid_inputs_are_rejected() {
    let bytes = file(3, Endian::Little, 32);
    let limits = Limits {
        max_metadata_entries: 1,
        ..Limits::default()
    };
    let error = match Reader::with_limits(Cursor::new(bytes), limits) {
        Ok(_) => panic!("limit accepted"),
        Err(e) => e,
    };
    assert!(error.to_string().contains("metadata entries"));
    assert!(Writer::new(WriterOptions {
        alignment: 3,
        ..WriterOptions::default()
    })
    .is_err());
    let mut out = Cursor::new(Vec::new());
    let err = Writer::default()
        .write(
            &mut out,
            &BTreeMap::new(),
            &[TensorInput {
                name: "bad",
                dimensions: &[31],
                ggml_type: GgmlType::Q4_0,
                data: &[],
            }],
        )
        .unwrap_err();
    assert!(err.to_string().contains("block size"));
}

#[test]
fn duplicate_names_and_bad_payloads_are_rejected() {
    let data = [0u8; 4];
    let t = TensorInput {
        name: "x",
        dimensions: &[1],
        ggml_type: GgmlType::F32,
        data: &data,
    };
    let mut out = Cursor::new(Vec::new());
    assert!(Writer::default()
        .write(&mut out, &BTreeMap::new(), &[t, t])
        .unwrap_err()
        .to_string()
        .contains("duplicate"));
    let bad = TensorInput {
        name: "x",
        dimensions: &[2],
        ggml_type: GgmlType::F32,
        data: &data,
    };
    assert!(Writer::default()
        .write(Cursor::new(Vec::new()), &BTreeMap::new(), &[bad])
        .is_err());
}

#[derive(Default)]
struct SparseSink {
    pos: u64,
    len: u64,
    written: u64,
}
impl Write for SparseSink {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.pos = self
            .pos
            .checked_add(buf.len() as u64)
            .ok_or_else(|| io::Error::other("overflow"))?;
        self.len = self.len.max(self.pos);
        self.written += buf.len() as u64;
        Ok(buf.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
impl Seek for SparseSink {
    fn seek(&mut self, from: SeekFrom) -> io::Result<u64> {
        let next: i128 = match from {
            SeekFrom::Start(v) => v.into(),
            SeekFrom::Current(v) => self.pos as i128 + v as i128,
            SeekFrom::End(v) => self.len as i128 + v as i128,
        };
        self.pos = u64::try_from(next).map_err(|_| io::Error::other("invalid seek"))?;
        Ok(self.pos)
    }
}

#[test]
fn large_offsets_use_sparse_seeks_and_checked_arithmetic() {
    let data = 1f32.to_le_bytes();
    let tensors = [
        TensorInput {
            name: "a",
            dimensions: &[1],
            ggml_type: GgmlType::F32,
            data: &data,
        },
        TensorInput {
            name: "b",
            dimensions: &[1],
            ggml_type: GgmlType::F32,
            data: &data,
        },
    ];
    let mut sink = SparseSink::default();
    Writer::new(WriterOptions {
        alignment: 1 << 40,
        ..WriterOptions::default()
    })
    .unwrap()
    .write(&mut sink, &BTreeMap::new(), &tensors)
    .unwrap();
    assert!(sink.len >= (2 << 40) + 4);
    assert!(sink.written < 1024, "padding was physically buffered");
}

fn locate(bytes: &[u8], needle: &[u8]) -> usize {
    bytes
        .windows(needle.len())
        .position(|w| w == needle)
        .unwrap()
}
fn two_dense() -> Vec<u8> {
    let data = 1f32.to_le_bytes();
    let mut out = Cursor::new(Vec::new());
    Writer::default()
        .write(
            &mut out,
            &BTreeMap::new(),
            &[
                TensorInput {
                    name: "aaaa",
                    dimensions: &[1],
                    ggml_type: GgmlType::F32,
                    data: &data,
                },
                TensorInput {
                    name: "bbbb",
                    dimensions: &[1],
                    ggml_type: GgmlType::F32,
                    data: &data,
                },
            ],
        )
        .unwrap();
    out.into_inner()
}

#[test]
fn malformed_descriptor_corpus_is_rejected_without_panics() {
    let base = two_dense();
    let first = locate(&base, b"aaaa");
    let second = locate(&base, b"bbbb");
    let mut cases = Vec::new();
    let mut b = base.clone();
    b[second..second + 4].copy_from_slice(b"aaaa");
    cases.push(b);
    let mut b = base.clone();
    b[second + 20..second + 28].copy_from_slice(&0u64.to_le_bytes());
    cases.push(b);
    let mut b = base.clone();
    b[first + 20..first + 28].copy_from_slice(&1u64.to_le_bytes());
    cases.push(b);
    let mut b = base.clone();
    b[first + 16..first + 20].copy_from_slice(&999u32.to_le_bytes());
    cases.push(b);
    let mut b = base.clone();
    b[first + 4..first + 8].copy_from_slice(&99u32.to_le_bytes());
    cases.push(b);
    let mut b = base.clone();
    b[first + 8..first + 16].copy_from_slice(&u64::MAX.to_le_bytes());
    cases.push(b);
    let mut b = base.clone();
    b[first + 20..first + 28].copy_from_slice(&(u64::MAX - 31).to_le_bytes());
    cases.push(b);
    for bytes in cases {
        let result = std::panic::catch_unwind(|| Reader::new(Cursor::new(bytes)));
        assert!(result.is_ok());
        assert!(result.unwrap().is_err());
    }
}

#[test]
fn duplicate_metadata_invalid_boolean_and_alignment_are_rejected() {
    let mut bytes = file(3, Endian::Little, 32);
    let b = locate(&bytes, b"b.i8");
    bytes[b..b + 4].copy_from_slice(b"a.u8");
    assert!(Reader::new(Cursor::new(bytes)).is_err());
    let mut bytes = file(3, Endian::Little, 32);
    let key = locate(&bytes, b"h.bool");
    bytes[key + 6 + 4] = 2;
    assert!(Reader::new(Cursor::new(bytes)).is_err());
    let mut bytes = file(3, Endian::Little, 64);
    let key = locate(&bytes, b"general.alignment");
    bytes[key + 17 + 4..key + 17 + 8].copy_from_slice(&3u32.to_le_bytes());
    assert!(Reader::new(Cursor::new(bytes)).is_err());
}
