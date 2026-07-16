use crate::error::{Exception, IoError};
use crate::ops::{GgufMetadata, GgufMetadataValue};
use crate::utils::guard::Guarded;
use crate::utils::io::SafeTensors;
use crate::utils::SUCCESS;
use crate::{Array, Dtype, Stream};
use std::collections::HashMap;
use std::ffi::CString;
use std::path::{Path, PathBuf};

const GGUF_SPLIT_NO: &str = "split.no";
const GGUF_SPLIT_COUNT: &str = "split.count";
const GGUF_SPLIT_TENSORS_COUNT: &str = "split.tensors.count";

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

    /// Loads all tensors from a GGUF checkpoint.
    ///
    /// MLX preserves Q2_K, Q3_K, Q4_0, Q4_1, Q4_K, Q5_K, Q6_K, and Q8_0 tensors in its packed
    /// affine representation. Some other GGUF quantization formats are
    /// converted to floating point by MLX while loading; formats unsupported
    /// by MLX's bundled GGUF converter return an error.
    ///
    /// Canonically named sharded checkpoints are loaded automatically when
    /// `path` points to the first `-00001-of-NNNNN.gguf` shard.
    pub fn load_gguf(
        path: impl AsRef<Path>,
        stream: impl AsRef<Stream>,
    ) -> Result<HashMap<String, Array>, IoError> {
        Ok(load_gguf_shards(path.as_ref(), stream.as_ref(), false)?.0)
    }

    /// Loads all tensors and typed metadata from a GGUF checkpoint.
    ///
    /// Canonically named sharded checkpoints are loaded automatically when
    /// `path` points to the first `-00001-of-NNNNN.gguf` shard. Metadata is
    /// returned from that first shard.
    ///
    /// Use [`GgufMetadata::from_file`] when only metadata is needed; that API
    /// parses on the host and does not require an MLX stream.
    #[allow(clippy::type_complexity)]
    pub fn load_gguf_with_metadata(
        path: impl AsRef<Path>,
        stream: impl AsRef<Stream>,
    ) -> Result<(HashMap<String, Array>, HashMap<String, GgufMetadataValue>), IoError> {
        let (data, metadata) = load_gguf_shards(path.as_ref(), stream.as_ref(), true)?;
        Ok((data, metadata.expect("metadata was requested")))
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

type GgufData = (
    HashMap<String, Array>,
    Option<HashMap<String, GgufMetadataValue>>,
);

fn load_gguf_shards(
    path: &Path,
    _stream: &Stream,
    with_metadata: bool,
) -> Result<GgufData, IoError> {
    let (first, first_metadata) = load_one_gguf(path)?;
    let split_count = gguf_split_value(&first_metadata, GGUF_SPLIT_COUNT)?.unwrap_or(0);
    if split_count <= 1 {
        let data = first;
        let metadata = with_metadata.then(|| first_metadata.into_inner());
        return Ok((data, metadata));
    }

    let split_no = required_gguf_split_value(&first_metadata, GGUF_SPLIT_NO, path)?;
    if split_no != 0 {
        return Err(invalid_gguf_shards(format!(
            "sharded GGUF must be loaded from its first shard, but {:?} has {GGUF_SPLIT_NO}={split_no}",
            path.display()
        )));
    }
    let expected_tensors =
        required_gguf_split_value(&first_metadata, GGUF_SPLIT_TENSORS_COUNT, path)?;
    let shard_paths = gguf_shard_paths(path, split_count)?;

    let mut data = first;
    for (split_no, shard_path) in shard_paths.into_iter().enumerate().skip(1) {
        if !shard_path.is_file() {
            return Err(invalid_gguf_shards(format!(
                "missing GGUF shard {:?}",
                shard_path.display()
            )));
        }
        let (shard, shard_metadata) = load_one_gguf(&shard_path)?;
        let actual_split_no =
            required_gguf_split_value(&shard_metadata, GGUF_SPLIT_NO, &shard_path)?;
        if actual_split_no != split_no {
            return Err(invalid_gguf_shards(format!(
                "GGUF shard {:?} has {GGUF_SPLIT_NO}={actual_split_no}, expected {split_no}",
                shard_path.display()
            )));
        }
        if let Some(actual_count) = gguf_split_value(&shard_metadata, GGUF_SPLIT_COUNT)? {
            if actual_count != split_count {
                return Err(invalid_gguf_shards(format!(
                    "GGUF shard {:?} has {GGUF_SPLIT_COUNT}={actual_count}, expected {split_count}",
                    shard_path.display()
                )));
            }
        }
        for (name, value) in shard {
            if data.insert(name.clone(), value).is_some() {
                return Err(invalid_gguf_shards(format!(
                    "tensor {name:?} is duplicated across GGUF shards"
                )));
            }
        }
    }

    let loaded_tensors = gguf_tensor_count(&data);
    if loaded_tensors != expected_tensors {
        return Err(invalid_gguf_shards(format!(
            "sharded GGUF declares {expected_tensors} tensors in {GGUF_SPLIT_TENSORS_COUNT}, but {loaded_tensors} were loaded"
        )));
    }

    let metadata = with_metadata.then(|| first_metadata.into_inner());
    Ok((data, metadata))
}

fn load_one_gguf(path: &Path) -> Result<(HashMap<String, Array>, GgufMetadata), IoError> {
    if !path.is_file() {
        return Err(IoError::NotFile);
    }
    if !path
        .extension()
        .and_then(|v| v.to_str())
        .is_some_and(|v| v.eq_ignore_ascii_case("gguf"))
    {
        return Err(IoError::UnsupportedFormat);
    }
    let mut reader = safemlx_gguf::Reader::open(path)
        .map_err(|e| IoError::InvalidGguf(format!("{}: {e}", path.display())))?;
    let metadata = reader
        .metadata()
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let descriptors = reader.tensors().to_vec();
    let mut arrays = HashMap::new();
    for descriptor in descriptors {
        let converted = reader.read_tensor(&descriptor).map_err(|e| {
            IoError::InvalidGguf(format!(
                "{} tensor {:?}: {e}",
                path.display(),
                descriptor.name
            ))
        })?;
        match converted {
            safemlx_gguf::ConvertedTensor::Dense(dense) => {
                let shape = mlx_shape_i32(&descriptor.name, &dense.shape)?;
                let dtype = match dense.dtype {
                    safemlx_gguf::DenseDtype::F32 => Dtype::Float32,
                    safemlx_gguf::DenseDtype::F16 => Dtype::Float16,
                    safemlx_gguf::DenseDtype::Bf16 => Dtype::Bfloat16,
                    safemlx_gguf::DenseDtype::I8 => Dtype::Int8,
                    safemlx_gguf::DenseDtype::I16 => Dtype::Int16,
                    safemlx_gguf::DenseDtype::I32 => Dtype::Int32,
                    safemlx_gguf::DenseDtype::I64 => Dtype::Int64,
                    safemlx_gguf::DenseDtype::F64 => Dtype::Float64,
                };
                let array =
                    unsafe { Array::from_raw_data(dense.data.as_ptr().cast(), &shape, dtype) };
                if arrays.insert(descriptor.name.clone(), array).is_some() {
                    return Err(IoError::InvalidGguf(format!(
                        "duplicate tensor {:?}",
                        descriptor.name
                    )));
                }
            }
            safemlx_gguf::ConvertedTensor::Affine(affine) => {
                let weight_shape = mlx_shape_i32(&descriptor.name, &affine.weight_shape)?;
                let scale_shape = mlx_shape_i32(&descriptor.name, &affine.scale_shape)?;
                let weight = unsafe {
                    Array::from_raw_data(
                        affine.weights.as_ptr().cast(),
                        &weight_shape,
                        Dtype::Uint32,
                    )
                };
                let scales = unsafe {
                    Array::from_raw_data(
                        affine.scales.as_ptr().cast(),
                        &scale_shape,
                        Dtype::Float16,
                    )
                };
                let biases = unsafe {
                    Array::from_raw_data(
                        affine.biases.as_ptr().cast(),
                        &scale_shape,
                        Dtype::Float16,
                    )
                };
                let prefix = descriptor.name.strip_suffix(".weight").ok_or_else(|| {
                    IoError::InvalidGguf(format!(
                        "quantized tensor {:?} must end in .weight",
                        descriptor.name
                    ))
                })?;
                for (name, array) in [
                    (descriptor.name.clone(), weight),
                    (format!("{prefix}.scales"), scales),
                    (format!("{prefix}.biases"), biases),
                ] {
                    if arrays.insert(name.clone(), array).is_some() {
                        return Err(IoError::InvalidGguf(format!(
                            "generated tensor name {name:?} is duplicated"
                        )));
                    }
                }
            }
        }
    }
    Ok((arrays, metadata))
}

fn mlx_shape_i32(name: &str, shape: &[u64]) -> Result<Vec<i32>, IoError> {
    shape
        .iter()
        .map(|&v| {
            i32::try_from(v).map_err(|_| {
                IoError::InvalidGguf(format!(
                    "tensor {name:?} dimension {v} exceeds MLX i32 shape limits"
                ))
            })
        })
        .collect()
}

fn gguf_tensor_count(data: &HashMap<String, Array>) -> usize {
    data.keys()
        .filter(|name| {
            let Some((prefix, suffix)) = name.rsplit_once('.') else {
                return true;
            };
            if !matches!(suffix, "scales" | "biases") {
                return true;
            }
            !(data.contains_key(&format!("{prefix}.weight"))
                && data.contains_key(&format!(
                    "{prefix}.{}",
                    if suffix == "scales" {
                        "biases"
                    } else {
                        "scales"
                    }
                )))
        })
        .count()
}

fn gguf_split_value(metadata: &GgufMetadata, key: &str) -> Result<Option<usize>, IoError> {
    metadata
        .get(key)
        .map(|value| {
            value
                .as_i64()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| {
                    invalid_gguf_shards(format!(
                        "GGUF metadata key {key:?} must be a non-negative integer scalar"
                    ))
                })
        })
        .transpose()
}

fn required_gguf_split_value(
    metadata: &GgufMetadata,
    key: &str,
    path: &Path,
) -> Result<usize, IoError> {
    gguf_split_value(metadata, key)?.ok_or_else(|| {
        invalid_gguf_shards(format!(
            "GGUF shard {:?} is missing required metadata key {key:?}",
            path.display()
        ))
    })
}

fn gguf_shard_paths(path: &Path, split_count: usize) -> Result<Vec<PathBuf>, IoError> {
    let extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .ok_or(IoError::InvalidUtf8)?;
    let stem = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .ok_or(IoError::InvalidUtf8)?;
    let (prefix_and_no, filename_count) = stem.rsplit_once("-of-").ok_or_else(|| {
        invalid_gguf_shards(format!(
            "sharded GGUF filename {:?} must end in -00001-of-NNNNN.gguf",
            path.display()
        ))
    })?;
    let (prefix, filename_no) = prefix_and_no.rsplit_once('-').ok_or_else(|| {
        invalid_gguf_shards(format!(
            "sharded GGUF filename {:?} must end in -00001-of-NNNNN.gguf",
            path.display()
        ))
    })?;
    let valid_digits =
        |value: &str| value.len() == 5 && value.bytes().all(|byte| byte.is_ascii_digit());
    if prefix.is_empty() || !valid_digits(filename_no) || !valid_digits(filename_count) {
        return Err(invalid_gguf_shards(format!(
            "sharded GGUF filename {:?} must end in -00001-of-NNNNN.gguf",
            path.display()
        )));
    }
    let filename_no = filename_no
        .parse::<usize>()
        .map_err(|error| invalid_gguf_shards(format!("invalid GGUF shard number: {error}")))?;
    let filename_count = filename_count
        .parse::<usize>()
        .map_err(|error| invalid_gguf_shards(format!("invalid GGUF shard count: {error}")))?;
    if filename_no != 1 {
        return Err(invalid_gguf_shards(format!(
            "sharded GGUF must be loaded from shard 00001, got {filename_no:05}"
        )));
    }
    if filename_count != split_count {
        return Err(invalid_gguf_shards(format!(
            "GGUF filename declares {filename_count} shards, but {GGUF_SPLIT_COUNT}={split_count}"
        )));
    }
    if split_count > 99_999 {
        return Err(invalid_gguf_shards(format!(
            "GGUF shard count {split_count} cannot be represented by the canonical five-digit filename"
        )));
    }

    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    Ok((1..=split_count)
        .map(|index| {
            parent.join(format!(
                "{prefix}-{index:05}-of-{split_count:05}.{extension}"
            ))
        })
        .collect())
}

fn invalid_gguf_shards(message: impl Into<String>) -> IoError {
    IoError::Exception(Exception::custom(format!(
        "invalid sharded GGUF: {}",
        message.into()
    )))
}

#[cfg(test)]
mod tests {
    use crate::{ops::GgufMetadataValue, Array};
    use std::path::Path;

    fn gguf_test_stream() -> crate::Stream {
        crate::Stream::new_with_device(&crate::Device::new(crate::DeviceType::Cpu, 0))
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
        let tmp_dir = tempfile::tempdir().unwrap();
        let path = tmp_dir.path().join("test.safetensors");

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
        let tmp_dir = tempfile::tempdir().unwrap();
        let path = tmp_dir.path().join("test.npy");

        let a = Array::ones::<i32>(&[2, 4], stream).unwrap();
        a.save_numpy(&path).unwrap();

        let b = Array::load_numpy(&path, stream).unwrap();
        assert!(a
            .all_close(&b, None, None, None, stream)
            .unwrap()
            .item::<bool>(&stream));
    }

    #[test]
    fn test_load_gguf_with_metadata() {
        let stream = crate::Stream::new_with_device(&crate::Device::new(crate::DeviceType::Cpu, 0));
        let tmp_dir = tempfile::tempdir().unwrap();
        let path = tmp_dir.path().join("test.gguf");
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

        let (arrays, metadata) = Array::load_gguf_with_metadata(&path, &stream).unwrap();
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
        let stream = gguf_test_stream();
        let tmp_dir = tempfile::tempdir().unwrap();
        let path = tmp_dir.path().join("quantized.gguf");
        let formats = [
            ("q4_0.weight", safemlx_gguf::GgmlType::Q4_0),
            ("q4_1.weight", safemlx_gguf::GgmlType::Q4_1),
            ("q8_0.weight", safemlx_gguf::GgmlType::Q8_0),
            ("q2_k.weight", safemlx_gguf::GgmlType::Q2K),
            ("q3_k.weight", safemlx_gguf::GgmlType::Q3K),
            ("q4_k.weight", safemlx_gguf::GgmlType::Q4K),
            ("q5_k.weight", safemlx_gguf::GgmlType::Q5K),
            ("q6_k.weight", safemlx_gguf::GgmlType::Q6K),
            ("q5_0_legacy", safemlx_gguf::GgmlType::Q5_0),
            ("q5_1_legacy", safemlx_gguf::GgmlType::Q5_1),
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
        let arrays = Array::load_gguf(&path, &stream).unwrap();
        for (name, _) in &formats[..8] {
            let prefix = name.strip_suffix(".weight").unwrap();
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
        assert_eq!(arrays["q5_0_legacy"].dtype(), crate::Dtype::Float16);
        assert_eq!(arrays["q5_1_legacy"].dtype(), crate::Dtype::Float16);
        assert_eq!(arrays.len(), 26);
    }

    #[test]
    fn test_gguf_shard_paths() {
        let paths = super::gguf_shard_paths(Path::new("dir/model-00001-of-00003.gguf"), 3).unwrap();
        assert_eq!(
            paths,
            [
                "dir/model-00001-of-00003.gguf",
                "dir/model-00002-of-00003.gguf",
                "dir/model-00003-of-00003.gguf",
            ]
            .map(std::path::PathBuf::from)
        );

        let error = super::gguf_shard_paths(Path::new("model.gguf"), 2)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("must end in -00001-of-NNNNN.gguf"),
            "{error}"
        );
    }

    #[test]
    fn test_load_sharded_gguf() {
        let stream = gguf_test_stream();
        let tmp_dir = tempfile::tempdir().unwrap();
        let first = tmp_dir.path().join("model-00001-of-00002.gguf");
        let second = tmp_dir.path().join("model-00002-of-00002.gguf");
        save_gguf_shard(&first, "first.weight", 1.0, 0, 2, 2, "first", &stream);
        save_gguf_shard(&second, "second.weight", 2.0, 1, 2, 2, "second", &stream);

        let arrays = Array::load_gguf(&first, &stream).unwrap();
        assert_eq!(arrays.len(), 2);
        assert_eq!(arrays["first.weight"].clone().item::<f32>(&stream), 1.0);
        assert_eq!(arrays["second.weight"].clone().item::<f32>(&stream), 2.0);

        let (arrays, metadata) = Array::load_gguf_with_metadata(&first, &stream).unwrap();
        assert_eq!(arrays.len(), 2);
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

        let error = Array::load_gguf(&first, &stream).unwrap_err().to_string();
        assert!(error.contains("missing GGUF shard"), "{error}");
        assert!(error.contains("model-00002-of-00002.gguf"), "{error}");
    }

    #[test]
    fn test_sharded_gguf_rejects_non_first_shard() {
        let stream = gguf_test_stream();
        let tmp_dir = tempfile::tempdir().unwrap();
        let second = tmp_dir.path().join("model-00002-of-00002.gguf");
        save_gguf_shard(&second, "weight", 1.0, 1, 2, 2, "second", &stream);

        let error = Array::load_gguf(&second, &stream).unwrap_err().to_string();
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

        let error = Array::load_gguf(&first, &stream).unwrap_err().to_string();
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

        let error = Array::load_gguf(&first, &stream).unwrap_err().to_string();
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

        let error = Array::load_gguf(&first, &stream).unwrap_err().to_string();
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

        let error = Array::load_gguf(&first, &stream).unwrap_err().to_string();
        assert!(
            error.contains("declares 3 tensors in split.tensors.count, but 2 were loaded"),
            "{error}"
        );
    }
}
