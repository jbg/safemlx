use safemlx_gguf::{
    Checkpoint, ConvertedTensor, Endian, GgmlType, LogicalDtype, MetadataValue, TensorInput,
    Writer, WriterOptions,
};
use std::collections::BTreeMap;
use std::io::Cursor;

const IQ_TYPES: [GgmlType; 9] = [
    GgmlType::IQ2XXS,
    GgmlType::IQ2XS,
    GgmlType::IQ3XXS,
    GgmlType::IQ1S,
    GgmlType::IQ4NL,
    GgmlType::IQ3S,
    GgmlType::IQ2S,
    GgmlType::IQ4XS,
    GgmlType::IQ1M,
];

fn unhex(value: &str) -> Vec<u8> {
    value
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}

fn converted(ty: GgmlType, blocks: u64, raw: &[u8], endian: Endian) -> Vec<u8> {
    let (block, _) = ty.block_and_bytes().unwrap();
    let mut file = Cursor::new(Vec::new());
    Writer::new(WriterOptions {
        endian,
        ..WriterOptions::default()
    })
    .unwrap()
    .write(
        &mut file,
        &BTreeMap::new(),
        &[TensorInput {
            name: "oracle.weight",
            dimensions: &[block, blocks],
            ggml_type: ty,
            data: raw,
        }],
    )
    .unwrap();
    let mut reader = safemlx_gguf::Reader::new(Cursor::new(file.into_inner())).unwrap();
    let descriptor = reader.tensors()[0].clone();
    let ConvertedTensor::IQuant(iquant) = reader.read_tensor(&descriptor).unwrap() else {
        panic!("IQ tensors must remain packed");
    };
    assert_eq!(iquant.shape, [blocks, block]);
    assert_eq!(iquant.data, raw);
    iquant
        .dequantize_f32()
        .unwrap()
        .into_iter()
        .flat_map(|value| half::f16::from_f32(value).to_bits().to_ne_bytes())
        .collect()
}

#[test]
fn canonical_codes_and_exact_block_geometry_match_pinned_upstream() {
    let expected = [
        (16, GgmlType::IQ2XXS, 256, 66),
        (17, GgmlType::IQ2XS, 256, 74),
        (18, GgmlType::IQ3XXS, 256, 98),
        (19, GgmlType::IQ1S, 256, 50),
        (20, GgmlType::IQ4NL, 32, 18),
        (21, GgmlType::IQ3S, 256, 110),
        (22, GgmlType::IQ2S, 256, 82),
        (23, GgmlType::IQ4XS, 256, 136),
        (29, GgmlType::IQ1M, 256, 56),
    ];
    for (code, ty, block, bytes) in expected {
        assert_eq!(GgmlType::from_code(code), ty);
        assert_eq!(ty.code(), code);
        assert_eq!(ty.block_and_bytes().unwrap(), (block, bytes));
        assert!(ty.is_iq());
    }
}

#[test]
fn removed_runtime_repack_codes_are_known_but_not_accepted_as_gguf_encodings() {
    for (code, ty) in [
        (36, GgmlType::RemovedIQ4NL4_4),
        (37, GgmlType::RemovedIQ4NL4_8),
        (38, GgmlType::RemovedIQ4NL8_8),
    ] {
        assert_eq!(GgmlType::from_code(code), ty);
        assert_eq!(ty.code(), code);
        assert!(!ty.is_iq());
        assert!(ty.block_and_bytes().is_err());
    }
}

#[test]
fn differential_vectors_match_pinned_llama_cpp_f16_outputs_exactly() {
    for line in include_str!("fixtures/llama-c0bc8591-iq.oracle").lines() {
        let mut fields = line.split('|');
        let code = fields.next().unwrap().parse().unwrap();
        let raw = unhex(fields.next().unwrap());
        let expected = unhex(fields.next().unwrap());
        assert!(fields.next().is_none());
        let ty = GgmlType::from_code(code);
        assert_eq!(
            converted(ty, 2, &raw, Endian::Little),
            expected,
            "GGML type {code}"
        );
    }
}

#[test]
fn zero_blocks_have_no_nonzero_decoded_values() {
    for ty in IQ_TYPES {
        let (_, bytes) = ty.block_and_bytes().unwrap();
        let decoded = converted(ty, 1, &vec![0; bytes as usize], Endian::Little);
        assert!(
            decoded.chunks_exact(2).all(|pair| {
                half::f16::from_bits(u16::from_ne_bytes(pair.try_into().unwrap())).to_f32() == 0.0
            }),
            "{ty:?}"
        );
    }
}

#[test]
fn iq_multibyte_fields_obey_declared_gguf_byte_order() {
    let mut little = vec![0xff; 18];
    little[..2].copy_from_slice(&half::f16::from_f32(-0.75).to_bits().to_le_bytes());
    let mut big = little.clone();
    big[..2].reverse();
    assert_eq!(
        converted(GgmlType::IQ4NL, 1, &little, Endian::Little),
        converted(GgmlType::IQ4NL, 1, &big, Endian::Big)
    );

    let mut little = (0..74)
        .map(|index| (index * 29 + 7) as u8)
        .collect::<Vec<_>>();
    little[..2].copy_from_slice(&half::f16::from_f32(1.25).to_bits().to_le_bytes());
    let mut big = little.clone();
    for pair in big[..66].chunks_exact_mut(2) {
        pair.reverse();
    }
    assert_eq!(
        converted(GgmlType::IQ2XS, 1, &little, Endian::Little),
        converted(GgmlType::IQ2XS, 1, &big, Endian::Big)
    );
}

#[test]
fn writer_preserves_every_iq_payload_byte_for_byte() {
    for ty in IQ_TYPES {
        let (block, bytes) = ty.block_and_bytes().unwrap();
        let raw = (0..bytes)
            .map(|index| (index * 37 + 11) as u8)
            .collect::<Vec<_>>();
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("raw.gguf");
        Writer::default()
            .write(
                std::fs::File::create(&path).unwrap(),
                &BTreeMap::new(),
                &[TensorInput {
                    name: "raw.weight",
                    dimensions: &[block],
                    ggml_type: ty,
                    data: &raw,
                }],
            )
            .unwrap();
        let checkpoint = Checkpoint::open(path).unwrap();
        let mut materializer = checkpoint.materializer();
        assert_eq!(materializer.raw_tensor("raw.weight").unwrap().data(), raw);
    }
}

#[test]
fn catalog_exposes_iq_as_one_packed_u8_logical_tensor() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("catalog.gguf");
    let payloads = IQ_TYPES
        .iter()
        .map(|ty| vec![0; ty.block_and_bytes().unwrap().1 as usize])
        .collect::<Vec<_>>();
    let dimensions = IQ_TYPES
        .iter()
        .map(|ty| [ty.block_and_bytes().unwrap().0])
        .collect::<Vec<_>>();
    let names = IQ_TYPES
        .iter()
        .map(|ty| format!("{ty:?}.weight"))
        .collect::<Vec<_>>();
    let inputs = IQ_TYPES
        .iter()
        .enumerate()
        .map(|(index, ty)| TensorInput {
            name: &names[index],
            dimensions: &dimensions[index],
            ggml_type: *ty,
            data: &payloads[index],
        })
        .collect::<Vec<_>>();
    Writer::default()
        .write(
            std::fs::File::create(&path).unwrap(),
            &BTreeMap::new(),
            &inputs,
        )
        .unwrap();
    let checkpoint = Checkpoint::open(path).unwrap();
    for tensor in checkpoint.tensors() {
        assert!(tensor.descriptor().ggml_type.is_iq());
        assert_eq!(tensor.affine(), None);
        assert_eq!(tensor.outputs().len(), 1);
        assert_eq!(tensor.outputs()[0].name, tensor.descriptor().name);
        let (block_values, block_bytes) = tensor.descriptor().ggml_type.block_and_bytes().unwrap();
        assert_eq!(tensor.outputs()[0].shape, [block_bytes]);
        assert_eq!(tensor.descriptor().mlx_shape(), [block_values]);
        assert_eq!(tensor.outputs()[0].dtype, LogicalDtype::U8);
    }
}

#[test]
fn iq_shapes_and_truncation_are_rejected_during_container_validation() {
    let ty = GgmlType::IQ2XXS;
    let (block, bytes) = ty.block_and_bytes().unwrap();
    let payload = vec![0; bytes as usize];
    assert!(Writer::default()
        .write(
            Cursor::new(Vec::new()),
            &BTreeMap::new(),
            &[TensorInput {
                name: "bad.weight",
                dimensions: &[block - 1],
                ggml_type: ty,
                data: &payload,
            }],
        )
        .is_err());

    let mut file = Cursor::new(Vec::new());
    Writer::default()
        .write(
            &mut file,
            &BTreeMap::new(),
            &[TensorInput {
                name: "truncated.weight",
                dimensions: &[block],
                ggml_type: ty,
                data: &payload,
            }],
        )
        .unwrap();
    let mut bytes = file.into_inner();
    bytes.pop();
    assert!(safemlx_gguf::Reader::new(Cursor::new(bytes)).is_err());
}

#[test]
fn unsloth_dynamic_recipe_families_are_tensor_format_compatible() {
    let recipes = [
        ("UD-IQ2_XXS", GgmlType::IQ2XXS, None),
        ("UD-IQ2_M", GgmlType::IQ2XS, Some(GgmlType::IQ4NL)),
        ("UD-Q2_K_XL", GgmlType::Q2K, Some(GgmlType::IQ4NL)),
        ("UD-IQ3_XXS", GgmlType::IQ3XXS, None),
        ("UD-IQ3_S", GgmlType::IQ3S, None),
        ("UD-Q3_K_M", GgmlType::Q3K, Some(GgmlType::IQ4NL)),
        ("UD-Q3_K_XL", GgmlType::Q3K, Some(GgmlType::IQ4NL)),
        ("UD-IQ4_XS", GgmlType::IQ4XS, None),
        ("UD-IQ4_NL", GgmlType::IQ4NL, None),
    ];
    for (recipe, primary, dynamic) in recipes {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join(format!("{recipe}.gguf"));
        let mut types = vec![primary];
        if let Some(dynamic) = dynamic {
            types.push(dynamic);
        }
        let payloads = types
            .iter()
            .map(|ty| vec![0; ty.block_and_bytes().unwrap().1 as usize])
            .collect::<Vec<_>>();
        let dimensions = types
            .iter()
            .map(|ty| [ty.block_and_bytes().unwrap().0])
            .collect::<Vec<_>>();
        let names = (0..types.len())
            .map(|index| format!("blk.0.recipe_{index}.weight"))
            .collect::<Vec<_>>();
        let inputs = types
            .iter()
            .enumerate()
            .map(|(index, ty)| TensorInput {
                name: &names[index],
                dimensions: &dimensions[index],
                ggml_type: *ty,
                data: &payloads[index],
            })
            .collect::<Vec<_>>();
        Writer::default()
            .write(
                std::fs::File::create(&path).unwrap(),
                &BTreeMap::from([("general.name".into(), MetadataValue::String(recipe.into()))]),
                &inputs,
            )
            .unwrap();
        let checkpoint = Checkpoint::open(path).unwrap();
        assert_eq!(checkpoint.physical_tensor_count(), types.len(), "{recipe}");
        assert_eq!(checkpoint.metadata()["general.name"].as_str(), Some(recipe));
        assert_eq!(
            checkpoint
                .tensors()
                .map(|tensor| tensor.descriptor().ggml_type)
                .collect::<Vec<_>>(),
            types,
            "{recipe}"
        );
        assert_eq!(
            checkpoint
                .converted_tensors()
                .collect::<Result<Vec<_>, _>>()
                .unwrap()
                .len(),
            types.len(),
            "{recipe}"
        );
    }
}
