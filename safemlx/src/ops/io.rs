use crate::error::IoError;
use crate::ops::GgufMetadataValue;
use crate::utils::guard::Guarded;
use crate::utils::io::SafeTensors;
use crate::utils::SUCCESS;
use crate::{Array, Dtype, Stream};
use std::collections::HashMap;
use std::ffi::CString;
use std::path::Path;

fn check_file_extension(path: &Path, expected: &str) -> Result<(), IoError> {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some(ext) if ext == expected => Ok(()),
        _ => Err(IoError::UnsupportedFormat),
    }
}

impl Array {
    /// Load array from a binary file in `.npy` format.
    ///
    /// # Params
    ///
    /// - path: path of file to load
    /// - stream: stream or device to evaluate on
    pub fn load_numpy(
        path: impl AsRef<Path>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array, IoError> {
        let path = path.as_ref();
        if !path.is_file() {
            return Err(IoError::NotFile);
        }
        let c_path = CString::new(path.to_str().ok_or(IoError::InvalidUtf8)?)?;
        check_file_extension(path, "npy")?;

        Array::try_from_op(|res| unsafe {
            safemlx_sys::mlx_load(res, c_path.as_ptr(), stream.as_ref().as_ptr())
        })
        .map_err(Into::into)
    }

    /// Load dictionary of ``MLXArray`` from a `safetensors` file.
    ///
    /// # Params
    ///
    /// - path: path of file to load
    /// - stream: stream or device to load on
    pub fn load_safetensors(
        path: impl AsRef<Path>,
        stream: impl AsRef<Stream>,
    ) -> Result<HashMap<String, Array>, IoError> {
        let safetensors = SafeTensors::load_device(path.as_ref(), stream)?;
        let data = safetensors.data()?;
        Ok(data)
    }

    /// Load dictionary of ``MLXArray`` and metadata `[String:String]` from a `safetensors` file.
    ///
    /// # Params
    ///
    /// - path: path of file to load
    /// - stream: stream or device to load on
    #[allow(clippy::type_complexity)]
    pub fn load_safetensors_with_metadata(
        path: impl AsRef<Path>,
        stream: impl AsRef<Stream>,
    ) -> Result<(HashMap<String, Array>, HashMap<String, String>), IoError> {
        let safetensors = SafeTensors::load_device(path.as_ref(), stream)?;
        let data = safetensors.data()?;
        let metadata = safetensors.metadata()?;

        Ok((data, metadata))
    }

    /// Save dense arrays and typed metadata as a deterministic GGUF v3 file.
    ///
    /// This preserves the former MLX writer's dense F32/F16/I8/I16/I32
    /// support and additionally accepts BF16, I64, and F64. Affine arrays are
    /// deliberately rejected because their three-array MLX representation does
    /// not contain enough information for a lossless inverse to GGML blocks.
    pub fn save_gguf<'a, I, S, V>(
        arrays: I,
        metadata: impl Into<Option<&'a HashMap<String, GgufMetadataValue>>>,
        path: impl AsRef<Path>,
    ) -> Result<(), IoError>
    where
        I: IntoIterator<Item = (S, V)>,
        S: AsRef<str>,
        V: AsRef<Array>,
    {
        let path = path.as_ref();
        check_file_extension(path, "gguf")?;
        let mut owned = Vec::new();
        for (name, array) in arrays {
            let evaluated = array.as_ref().evaluated()?;
            let a = evaluated.as_array();
            let (ty, data) = gguf_dense_bytes(&evaluated)?;
            let dimensions = a
                .shape()
                .iter()
                .rev()
                .map(|&v| {
                    u64::try_from(v).map_err(|_| {
                        IoError::InvalidGguf(format!(
                            "negative dimension in tensor {:?}",
                            name.as_ref()
                        ))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            owned.push((name.as_ref().to_owned(), dimensions, ty, data));
        }
        let inputs = owned
            .iter()
            .map(
                |(name, dimensions, ggml_type, data)| safemlx_gguf::TensorInput {
                    name,
                    dimensions,
                    ggml_type: *ggml_type,
                    data,
                },
            )
            .collect::<Vec<_>>();
        let typed = metadata
            .into()
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .collect::<std::collections::BTreeMap<_, _>>();
        let file = std::fs::File::create(path).map_err(|_| IoError::UnableToOpenFile)?;
        safemlx_gguf::Writer::default().write(file, &typed, &inputs)?;
        Ok(())
    }

    /// Save array to a binary file in `.npy`format.
    ///
    /// # Params
    ///
    /// - array: array to save
    /// - url: URL of file to load
    pub fn save_numpy(&self, path: impl AsRef<Path>) -> Result<(), IoError> {
        let path = path.as_ref();
        check_file_extension(path, "npy")?;
        let c_path = CString::new(path.to_str().ok_or(IoError::InvalidUtf8)?)?;

        unsafe { safemlx_sys::mlx_save(c_path.as_ptr(), self.as_ptr()) };

        Ok(())
    }

    /// Save dictionary of arrays in `safetensors` format.
    ///
    /// # Params
    ///
    /// - arrays: arrays to save
    /// - metadata: metadata to save
    /// - path: path of file to save
    pub fn save_safetensors<'a, I, S, V>(
        arrays: I,
        metadata: impl Into<Option<&'a HashMap<String, String>>>,
        path: impl AsRef<Path>,
    ) -> Result<(), IoError>
    where
        I: IntoIterator<Item = (S, V)>,
        S: AsRef<str>,
        V: AsRef<Array>,
    {
        crate::error::ensure_mlx_error_handler();

        let path = path.as_ref();

        check_file_extension(path, "safetensors")?;

        let entries = arrays.into_iter().collect::<Vec<_>>();
        crate::transforms::eval(entries.iter().map(|(_, array)| array.as_ref()))?;

        let arrays = unsafe {
            let data = safemlx_sys::mlx_map_string_to_array_new();
            for (key, array) in entries.iter() {
                let key = CString::new(key.as_ref())?;

                let status = safemlx_sys::mlx_map_string_to_array_insert(
                    data,
                    key.as_ptr(),
                    array.as_ref().as_ptr(),
                );

                if status != SUCCESS {
                    safemlx_sys::mlx_map_string_to_array_free(data);
                    return Err(crate::error::get_and_clear_last_mlx_error()
                        .expect("A non-success status was returned, but no error was set.")
                        .into());
                }
            }
            data
        };

        let default_metadata = HashMap::new();
        let metadata_ref = metadata.into().unwrap_or(&default_metadata);

        let metadata = unsafe {
            let data = safemlx_sys::mlx_map_string_to_string_new();
            for (key, value) in metadata_ref.iter() {
                let key = CString::new(key.as_str())?;
                let value = CString::new(value.as_str())?;

                let status = safemlx_sys::mlx_map_string_to_string_insert(
                    data,
                    key.as_ptr(),
                    value.as_ptr(),
                );

                if status != SUCCESS {
                    safemlx_sys::mlx_map_string_to_string_free(data);
                    return Err(crate::error::get_and_clear_last_mlx_error()
                        .expect("A non-success status was returned, but no error was set.")
                        .into());
                }
            }
            data
        };

        let c_path = CString::new(path.to_str().ok_or(IoError::InvalidUtf8)?)?;

        unsafe {
            let status = safemlx_sys::mlx_save_safetensors(c_path.as_ptr(), arrays, metadata);

            let last_error = match status {
                SUCCESS => None,
                _ => Some(
                    crate::error::get_and_clear_last_mlx_error()
                        .expect("A non-success status was returned, but no error was set."),
                ),
            };

            safemlx_sys::mlx_map_string_to_array_free(arrays);
            safemlx_sys::mlx_map_string_to_string_free(metadata);

            if let Some(error) = last_error {
                return Err(error.into());
            }
        };

        Ok(())
    }
}

fn gguf_dense_bytes(
    array: &crate::EvaluatedArray<'_>,
) -> Result<(safemlx_gguf::GgmlType, Vec<u8>), IoError> {
    macro_rules! bytes {
        ($ty:ty) => {
            array
                .as_slice::<$ty>()
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect()
        };
    }
    Ok(match array.as_array().dtype() {
        Dtype::Float32 => (safemlx_gguf::GgmlType::F32, bytes!(f32)),
        Dtype::Float16 => (
            safemlx_gguf::GgmlType::F16,
            array
                .as_slice::<half::f16>()
                .iter()
                .flat_map(|v| v.to_bits().to_le_bytes())
                .collect(),
        ),
        Dtype::Bfloat16 => (
            safemlx_gguf::GgmlType::Bf16,
            array
                .as_slice::<half::bf16>()
                .iter()
                .flat_map(|v| v.to_bits().to_le_bytes())
                .collect(),
        ),
        Dtype::Int8 => (
            safemlx_gguf::GgmlType::I8,
            array.as_slice::<i8>().iter().map(|v| *v as u8).collect(),
        ),
        Dtype::Int16 => (safemlx_gguf::GgmlType::I16, bytes!(i16)),
        Dtype::Int32 => (safemlx_gguf::GgmlType::I32, bytes!(i32)),
        Dtype::Int64 => (safemlx_gguf::GgmlType::I64, bytes!(i64)),
        Dtype::Float64 => (safemlx_gguf::GgmlType::F64, bytes!(f64)),
        dtype => {
            return Err(IoError::InvalidGguf(format!(
                "dtype {dtype:?} cannot be represented losslessly by GGUF"
            )))
        }
    })
}

#[cfg(test)]
mod tests {
    use crate::{
        ops::{GgufCheckpoint, GgufMetadataValue},
        Array,
    };
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};

    fn io_test_dir() -> (tempfile::TempDir, PathBuf) {
        let temp_dir = tempfile::tempdir().unwrap();
        let test_dir = temp_dir.path().join("formats with spaces");
        std::fs::create_dir(&test_dir).unwrap();
        (temp_dir, test_dir)
    }

    fn unicode_gguf_test_dir() -> (tempfile::TempDir, PathBuf) {
        let temp_dir = tempfile::tempdir().unwrap();
        let test_dir = temp_dir.path().join("GGUF with spaces üñîçødé");
        std::fs::create_dir(&test_dir).unwrap();
        (temp_dir, test_dir)
    }

    fn gguf_test_stream() -> crate::Stream {
        crate::Stream::new_with_device(&crate::Device::new(crate::DeviceType::Cpu, 0))
    }

    fn collect_gguf(path: &Path) -> (HashMap<String, Array>, GgufCheckpoint) {
        let checkpoint = GgufCheckpoint::open(path).unwrap();
        let mut arrays = HashMap::new();
        checkpoint
            .for_each_converted_tensor(|tensor| {
                for (name, array) in tensor.into_arrays() {
                    assert!(arrays.insert(name, array).is_none());
                }
                Ok(())
            })
            .unwrap();
        (arrays, checkpoint)
    }

    #[allow(clippy::too_many_arguments)]
    fn save_gguf_shard(
        path: &Path,
        tensor_name: &str,
        tensor_value: f32,
        split_no: u16,
        split_count: u16,
        total_tensors: i32,
        name: &str,
        stream: &crate::Stream,
    ) {
        let tensor =
            Array::arange::<_, f32>(Some(tensor_value), tensor_value + 1.0, None, stream).unwrap();
        let metadata = std::collections::HashMap::from([
            ("split.no".into(), GgufMetadataValue::Uint16(split_no)),
            ("split.count".into(), GgufMetadataValue::Uint16(split_count)),
            (
                "split.tensors.count".into(),
                GgufMetadataValue::Int32(total_tensors),
            ),
            (
                "general.name".into(),
                GgufMetadataValue::String(name.into()),
            ),
        ]);
        Array::save_gguf([(tensor_name, &tensor)], Some(&metadata), path).unwrap();
    }

    #[test]
    fn test_save_arrays() {
        let stream = crate::test_stream();
        let (_tmp_dir, test_dir) = io_test_dir();
        let path = test_dir.join("test tensors.safetensors");

        let mut arrays = std::collections::HashMap::new();
        arrays.insert(
            "foo".to_string(),
            Array::ones::<i32>(&[1, 2], stream).unwrap(),
        );
        arrays.insert(
            "bar".to_string(),
            Array::zeros::<i32>(&[2, 1], stream).unwrap(),
        );

        Array::save_safetensors(&arrays, None, &path).unwrap();

        let loaded_arrays = Array::load_safetensors(&path, stream).unwrap();

        // compare values
        let mut loaded_keys: Vec<_> = loaded_arrays.keys().cloned().collect();
        let mut original_keys: Vec<_> = arrays.keys().cloned().collect();
        loaded_keys.sort();
        original_keys.sort();
        assert_eq!(loaded_keys, original_keys);

        for key in loaded_keys {
            let loaded_array = loaded_arrays.get(&key).unwrap();
            let original_array = arrays.get(&key).unwrap();
            assert!(loaded_array
                .all_close(original_array, None, None, None, stream)
                .unwrap()
                .item::<bool>(&stream));
        }
    }

    #[test]
    fn test_save_array() {
        let stream = crate::test_stream();
        let (_tmp_dir, test_dir) = io_test_dir();
        let path = test_dir.join("test array.npy");

        let a = Array::ones::<i32>(&[2, 4], stream).unwrap();
        a.save_numpy(&path).unwrap();

        let b = Array::load_numpy(&path, stream).unwrap();
        assert!(a
            .all_close(&b, None, None, None, stream)
            .unwrap()
            .item::<bool>(&stream));
    }

    #[test]
    fn test_stream_gguf_with_metadata() {
        let stream = crate::Stream::new_with_device(&crate::Device::new(crate::DeviceType::Cpu, 0));
        let (_tmp_dir, test_dir) = unicode_gguf_test_dir();
        let path = test_dir.join("test metadata.gguf");
        let tensor = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2])
            .copy(&stream)
            .unwrap();
        let metadata = std::collections::HashMap::from([
            ("answer".into(), GgufMetadataValue::Int32(42)),
            (
                "general.name".into(),
                GgufMetadataValue::String("tiny model".into()),
            ),
            (
                "general.tags".into(),
                GgufMetadataValue::Array(crate::ops::GgufMetadataArray::String(vec![
                    "one".into(),
                    "two".into(),
                ])),
            ),
        ]);
        Array::save_gguf([("tensor", &tensor)], Some(&metadata), &path).unwrap();

        let (arrays, checkpoint) = collect_gguf(&path);
        let metadata = checkpoint.metadata();
        assert_eq!(arrays["tensor"].shape(), &[2, 2]);
        assert!(arrays["tensor"].clone().try_item::<f32>(&stream).is_err());
        match &metadata["answer"] {
            GgufMetadataValue::Int32(value) => assert_eq!(*value, 42),
            value => panic!("unexpected answer metadata: {value:?}"),
        }
        match &metadata["general.name"] {
            GgufMetadataValue::String(value) => assert_eq!(value, "tiny model"),
            value => panic!("unexpected name metadata: {value:?}"),
        }
        match &metadata["general.tags"] {
            GgufMetadataValue::Array(value) => {
                assert_eq!(value.as_strings().unwrap(), &["one", "two"])
            }
            value => panic!("unexpected tags metadata: {value:?}"),
        }
    }

    #[test]
    fn test_load_quantized_gguf_without_mlx_gguf() {
        let (_tmp_dir, test_dir) = unicode_gguf_test_dir();
        let path = test_dir.join("quantized weights.gguf");
        let formats = [
            ("q4_0.weight", safemlx_gguf::GgmlType::Q4_0),
            ("q4_1.weight", safemlx_gguf::GgmlType::Q4_1),
            ("q8_0.weight", safemlx_gguf::GgmlType::Q8_0),
            ("q2_k.weight", safemlx_gguf::GgmlType::Q2K),
            ("q3_k.weight", safemlx_gguf::GgmlType::Q3K),
            ("q4_k.weight", safemlx_gguf::GgmlType::Q4K),
            ("q5_k.weight", safemlx_gguf::GgmlType::Q5K),
            ("q6_k.weight", safemlx_gguf::GgmlType::Q6K),
            ("q5_0.weight", safemlx_gguf::GgmlType::Q5_0),
            ("q5_1.weight", safemlx_gguf::GgmlType::Q5_1),
            ("iq2_xxs.weight", safemlx_gguf::GgmlType::IQ2XXS),
            ("iq2_xs.weight", safemlx_gguf::GgmlType::IQ2XS),
            ("iq3_xxs.weight", safemlx_gguf::GgmlType::IQ3XXS),
            ("iq1_s.weight", safemlx_gguf::GgmlType::IQ1S),
            ("iq4_nl.weight", safemlx_gguf::GgmlType::IQ4NL),
            ("iq3_s.weight", safemlx_gguf::GgmlType::IQ3S),
            ("iq2_s.weight", safemlx_gguf::GgmlType::IQ2S),
            ("iq4_xs.weight", safemlx_gguf::GgmlType::IQ4XS),
            ("iq1_m.weight", safemlx_gguf::GgmlType::IQ1M),
        ];
        let payloads = formats
            .iter()
            .map(|(_, ty)| vec![0; ty.block_and_bytes().unwrap().1 as usize])
            .collect::<Vec<_>>();
        let dimensions = formats
            .iter()
            .map(|(_, ty)| [ty.block_and_bytes().unwrap().0])
            .collect::<Vec<_>>();
        let inputs = formats
            .iter()
            .zip(&payloads)
            .zip(&dimensions)
            .map(
                |(((name, ty), data), dimensions)| safemlx_gguf::TensorInput {
                    name,
                    dimensions,
                    ggml_type: *ty,
                    data,
                },
            )
            .collect::<Vec<_>>();
        safemlx_gguf::Writer::default()
            .write(
                std::fs::File::create(&path).unwrap(),
                &std::collections::BTreeMap::new(),
                &inputs,
            )
            .unwrap();
        let (arrays, _) = collect_gguf(&path);
        for (name, ty) in &formats {
            let prefix = name.strip_suffix(".weight").unwrap();
            if ty.is_iq() {
                assert_eq!(arrays[*name].dtype(), crate::Dtype::Uint8);
                assert_eq!(
                    arrays[*name].size(),
                    ty.block_and_bytes().unwrap().1 as usize
                );
                assert!(!arrays.contains_key(&format!("{prefix}.scales")));
                assert!(!arrays.contains_key(&format!("{prefix}.biases")));
            } else {
                assert_eq!(arrays[*name].dtype(), crate::Dtype::Uint32);
                assert_eq!(
                    arrays[&format!("{prefix}.scales")].dtype(),
                    crate::Dtype::Float16
                );
                assert_eq!(
                    arrays[&format!("{prefix}.biases")].dtype(),
                    crate::Dtype::Float16
                );
            }
        }
        assert_eq!(arrays.len(), 39);

        #[cfg(feature = "metal")]
        if crate::metal::is_available().unwrap_or(false) {
            let metal =
                crate::Stream::new_with_device(&crate::Device::new(crate::DeviceType::Gpu, 0));
            let copies = arrays
                .values()
                .map(|array| array.copy(&metal).unwrap())
                .collect::<Vec<_>>();
            crate::transforms::eval(&copies).unwrap();
            for (source, copied) in arrays.values().zip(&copies) {
                assert_eq!(copied.dtype(), source.dtype());
                assert_eq!(copied.shape(), source.shape());
            }
        }
    }

    #[test]
    fn test_load_sharded_gguf() {
        let stream = gguf_test_stream();
        let tmp_dir = tempfile::tempdir().unwrap();
        let first = tmp_dir.path().join("model-00001-of-00002.gguf");
        let second = tmp_dir.path().join("model-00002-of-00002.gguf");
        save_gguf_shard(&first, "first.weight", 1.0, 0, 2, 2, "first", &stream);
        save_gguf_shard(&second, "second.weight", 2.0, 1, 2, 2, "second", &stream);

        let (arrays, checkpoint) = collect_gguf(&first);
        assert_eq!(arrays.len(), 2);
        assert_eq!(arrays["first.weight"].clone().item::<f32>(&stream), 1.0);
        assert_eq!(arrays["second.weight"].clone().item::<f32>(&stream), 2.0);

        let metadata = checkpoint.metadata();
        match &metadata["general.name"] {
            GgufMetadataValue::String(value) => assert_eq!(value, "first"),
            value => panic!("unexpected name metadata: {value:?}"),
        }
    }

    #[test]
    fn test_sharded_gguf_reports_missing_shard() {
        let stream = gguf_test_stream();
        let tmp_dir = tempfile::tempdir().unwrap();
        let first = tmp_dir.path().join("model-00001-of-00002.gguf");
        save_gguf_shard(&first, "weight", 1.0, 0, 2, 2, "first", &stream);

        let error = GgufCheckpoint::open(&first).unwrap_err().to_string();
        assert!(error.contains("missing GGUF shard"), "{error}");
        assert!(error.contains("model-00002-of-00002.gguf"), "{error}");
    }

    #[test]
    fn test_sharded_gguf_rejects_non_first_shard() {
        let stream = gguf_test_stream();
        let tmp_dir = tempfile::tempdir().unwrap();
        let second = tmp_dir.path().join("model-00002-of-00002.gguf");
        save_gguf_shard(&second, "weight", 1.0, 1, 2, 2, "second", &stream);

        let error = GgufCheckpoint::open(&second).unwrap_err().to_string();
        assert!(
            error.contains("must be loaded from its first shard"),
            "{error}"
        );
    }

    #[test]
    fn test_sharded_gguf_rejects_filename_count_mismatch() {
        let stream = gguf_test_stream();
        let tmp_dir = tempfile::tempdir().unwrap();
        let first = tmp_dir.path().join("model-00001-of-00002.gguf");
        save_gguf_shard(&first, "weight", 1.0, 0, 3, 3, "first", &stream);

        let error = GgufCheckpoint::open(&first).unwrap_err().to_string();
        assert!(
            error.contains("filename declares 2 shards, but split.count=3"),
            "{error}"
        );
    }

    #[test]
    fn test_sharded_gguf_rejects_wrong_shard_index() {
        let stream = gguf_test_stream();
        let tmp_dir = tempfile::tempdir().unwrap();
        let first = tmp_dir.path().join("model-00001-of-00002.gguf");
        let second = tmp_dir.path().join("model-00002-of-00002.gguf");
        save_gguf_shard(&first, "first", 1.0, 0, 2, 2, "first", &stream);
        save_gguf_shard(&second, "second", 2.0, 0, 2, 2, "second", &stream);

        let error = GgufCheckpoint::open(&first).unwrap_err().to_string();
        assert!(error.contains("split.no=0, expected 1"), "{error}");
    }

    #[test]
    fn test_sharded_gguf_rejects_duplicate_tensors() {
        let stream = gguf_test_stream();
        let tmp_dir = tempfile::tempdir().unwrap();
        let first = tmp_dir.path().join("model-00001-of-00002.gguf");
        let second = tmp_dir.path().join("model-00002-of-00002.gguf");
        save_gguf_shard(&first, "weight", 1.0, 0, 2, 2, "first", &stream);
        save_gguf_shard(&second, "weight", 2.0, 1, 2, 2, "second", &stream);

        let error = GgufCheckpoint::open(&first).unwrap_err().to_string();
        assert!(
            error.contains("is duplicated across GGUF shards"),
            "{error}"
        );
    }

    #[test]
    fn test_sharded_gguf_rejects_tensor_count_mismatch() {
        let stream = gguf_test_stream();
        let tmp_dir = tempfile::tempdir().unwrap();
        let first = tmp_dir.path().join("model-00001-of-00002.gguf");
        let second = tmp_dir.path().join("model-00002-of-00002.gguf");
        save_gguf_shard(&first, "first", 1.0, 0, 2, 3, "first", &stream);
        save_gguf_shard(&second, "second", 2.0, 1, 2, 3, "second", &stream);

        let error = GgufCheckpoint::open(&first).unwrap_err().to_string();
        assert!(
            error.contains("declares 3 tensors in split.tensors.count, but 2 were cataloged"),
            "{error}"
        );
    }
}
