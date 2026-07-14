use crate::error::{Exception, IoError};
use crate::utils::guard::Guarded;
use crate::utils::io::{Gguf, SafeTensors};
use crate::utils::SUCCESS;
use crate::{Array, Stream};
use std::collections::HashMap;
use std::ffi::CString;
use std::path::{Path, PathBuf};

const GGUF_SPLIT_NO: &str = "split.no";
const GGUF_SPLIT_COUNT: &str = "split.count";
const GGUF_SPLIT_TENSORS_COUNT: &str = "split.tensors.count";

/// A typed metadata value loaded from a GGUF file.
#[derive(Debug, Clone)]
pub enum GgufMetadataValue {
    /// Numeric or boolean scalar/one-dimensional array metadata.
    Array(Array),
    /// UTF-8 string metadata.
    String(String),
    /// UTF-8 string-array metadata.
    Strings(Vec<String>),
}

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
    #[allow(clippy::type_complexity)]
    pub fn load_gguf_with_metadata(
        path: impl AsRef<Path>,
        stream: impl AsRef<Stream>,
    ) -> Result<(HashMap<String, Array>, HashMap<String, GgufMetadataValue>), IoError> {
        let (data, metadata) = load_gguf_shards(path.as_ref(), stream.as_ref(), true)?;
        Ok((data, metadata.expect("metadata was requested")))
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

type GgufData = (
    HashMap<String, Array>,
    Option<HashMap<String, GgufMetadataValue>>,
);

fn load_gguf_shards(
    path: &Path,
    stream: &Stream,
    with_metadata: bool,
) -> Result<GgufData, IoError> {
    let first = Gguf::load_device(path, stream)?;
    let split_count = gguf_split_value(&first, GGUF_SPLIT_COUNT, stream)?.unwrap_or(0);
    if split_count <= 1 {
        let data = first.data()?;
        let metadata = with_metadata.then(|| first.metadata()).transpose()?;
        return Ok((data, metadata));
    }

    let split_no = required_gguf_split_value(&first, GGUF_SPLIT_NO, path, stream)?;
    if split_no != 0 {
        return Err(invalid_gguf_shards(format!(
            "sharded GGUF must be loaded from its first shard, but {:?} has {GGUF_SPLIT_NO}={split_no}",
            path.display()
        )));
    }
    let expected_tensors =
        required_gguf_split_value(&first, GGUF_SPLIT_TENSORS_COUNT, path, stream)?;
    let shard_paths = gguf_shard_paths(path, split_count)?;

    let mut data = first.data()?;
    let metadata = with_metadata.then(|| first.metadata()).transpose()?;
    for (split_no, shard_path) in shard_paths.into_iter().enumerate().skip(1) {
        if !shard_path.is_file() {
            return Err(invalid_gguf_shards(format!(
                "missing GGUF shard {:?}",
                shard_path.display()
            )));
        }
        let shard = Gguf::load_device(&shard_path, stream)?;
        let actual_split_no =
            required_gguf_split_value(&shard, GGUF_SPLIT_NO, &shard_path, stream)?;
        if actual_split_no != split_no {
            return Err(invalid_gguf_shards(format!(
                "GGUF shard {:?} has {GGUF_SPLIT_NO}={actual_split_no}, expected {split_no}",
                shard_path.display()
            )));
        }
        if let Some(actual_count) = gguf_split_value(&shard, GGUF_SPLIT_COUNT, stream)? {
            if actual_count != split_count {
                return Err(invalid_gguf_shards(format!(
                    "GGUF shard {:?} has {GGUF_SPLIT_COUNT}={actual_count}, expected {split_count}",
                    shard_path.display()
                )));
            }
        }
        for (name, value) in shard.data()? {
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

    Ok((data, metadata))
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

fn gguf_split_value(gguf: &Gguf, key: &str, stream: &Stream) -> Result<Option<usize>, IoError> {
    match gguf.metadata_value(key)? {
        Some(GgufMetadataValue::Array(value)) if value.size() == 1 => {
            let value = value.try_item::<i64>(stream)?;
            usize::try_from(value).map(Some).map_err(|_| {
                invalid_gguf_shards(format!(
                    "GGUF metadata key {key:?} must be a non-negative integer"
                ))
            })
        }
        Some(GgufMetadataValue::Array(_)) => Err(invalid_gguf_shards(format!(
            "GGUF metadata key {key:?} must be scalar"
        ))),
        Some(_) => Err(invalid_gguf_shards(format!(
            "GGUF metadata key {key:?} has the wrong type"
        ))),
        None => Ok(None),
    }
}

fn required_gguf_split_value(
    gguf: &Gguf,
    key: &str,
    path: &Path,
    stream: &Stream,
) -> Result<usize, IoError> {
    gguf_split_value(gguf, key, stream)?.ok_or_else(|| {
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
    use crate::{ops::GgufMetadataValue, transforms::eval, Array};
    use std::ffi::CString;
    use std::path::Path;

    fn gguf_test_stream() -> crate::Stream {
        crate::Stream::new_with_device(&crate::Device::new(crate::DeviceType::Cpu, 0))
    }

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
        let scalar_i32 = |value| {
            Array::arange::<_, i32>(Some(value), value + 1, None, stream)
                .unwrap()
                .squeeze(stream)
                .unwrap()
        };
        let split_no_value = scalar_i32(i32::from(split_no));
        let split_count_value = scalar_i32(i32::from(split_count));
        let total_tensors_value = scalar_i32(total_tensors);
        stream.synchronize().unwrap();

        unsafe {
            let gguf = safemlx_sys::mlx_io_gguf_new();
            for (key, value) in [
                ("split.no", &split_no_value),
                ("split.count", &split_count_value),
                ("split.tensors.count", &total_tensors_value),
            ] {
                let key = CString::new(key).unwrap();
                assert_eq!(
                    safemlx_sys::mlx_io_gguf_set_metadata_array(gguf, key.as_ptr(), value.as_ptr(),),
                    0
                );
            }
            let tensor_name = CString::new(tensor_name).unwrap();
            assert_eq!(
                safemlx_sys::mlx_io_gguf_set_array(gguf, tensor_name.as_ptr(), tensor.as_ptr(),),
                0
            );
            let name_key = CString::new("general.name").unwrap();
            let name = CString::new(name).unwrap();
            assert_eq!(
                safemlx_sys::mlx_io_gguf_set_metadata_string(
                    gguf,
                    name_key.as_ptr(),
                    name.as_ptr(),
                ),
                0
            );
            let path = CString::new(path.to_str().unwrap()).unwrap();
            let status = safemlx_sys::mlx_save_gguf(path.as_ptr(), gguf);
            assert_eq!(
                status,
                0,
                "{}",
                crate::error::get_and_clear_last_mlx_error()
                    .map(|error| error.what)
                    .unwrap_or_else(|| "GGUF save failed without an MLX error".into())
            );
            assert_eq!(safemlx_sys::mlx_io_gguf_free(gguf), 0);
        }
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
        let answer = Array::from(42i32).copy(&stream).unwrap();
        eval([&tensor, &answer]).unwrap();

        unsafe {
            let gguf = safemlx_sys::mlx_io_gguf_new();
            let tensor_key = CString::new("tensor").unwrap();
            assert_eq!(
                safemlx_sys::mlx_io_gguf_set_array(gguf, tensor_key.as_ptr(), tensor.as_ptr()),
                0
            );
            let answer_key = CString::new("answer").unwrap();
            assert_eq!(
                safemlx_sys::mlx_io_gguf_set_metadata_array(
                    gguf,
                    answer_key.as_ptr(),
                    answer.as_ptr(),
                ),
                0
            );
            let name_key = CString::new("general.name").unwrap();
            let name = CString::new("tiny model").unwrap();
            assert_eq!(
                safemlx_sys::mlx_io_gguf_set_metadata_string(
                    gguf,
                    name_key.as_ptr(),
                    name.as_ptr(),
                ),
                0
            );
            let tags_key = CString::new("general.tags").unwrap();
            let tags = [CString::new("one").unwrap(), CString::new("two").unwrap()];
            let mut tag_ptrs = tags.iter().map(|tag| tag.as_ptr()).collect::<Vec<_>>();
            let tag_vector =
                safemlx_sys::mlx_vector_string_new_data(tag_ptrs.as_mut_ptr(), tag_ptrs.len());
            assert_eq!(
                safemlx_sys::mlx_io_gguf_set_metadata_vector_string(
                    gguf,
                    tags_key.as_ptr(),
                    tag_vector,
                ),
                0
            );
            assert_eq!(safemlx_sys::mlx_vector_string_free(tag_vector), 0);

            let path = CString::new(path.to_str().unwrap()).unwrap();
            assert_eq!(safemlx_sys::mlx_save_gguf(path.as_ptr(), gguf), 0);
            assert_eq!(safemlx_sys::mlx_io_gguf_free(gguf), 0);
        }

        let (arrays, metadata) = Array::load_gguf_with_metadata(&path, &stream).unwrap();
        assert_eq!(arrays["tensor"].shape(), &[2, 2]);
        assert!(arrays["tensor"].clone().try_item::<f32>(&stream).is_err());
        match &metadata["answer"] {
            GgufMetadataValue::Array(value) => {
                assert_eq!(value.clone().item::<i32>(&stream), 42)
            }
            value => panic!("unexpected answer metadata: {value:?}"),
        }
        match &metadata["general.name"] {
            GgufMetadataValue::String(value) => assert_eq!(value, "tiny model"),
            value => panic!("unexpected name metadata: {value:?}"),
        }
        match &metadata["general.tags"] {
            GgufMetadataValue::Strings(value) => assert_eq!(value, &["one", "two"]),
            value => panic!("unexpected tags metadata: {value:?}"),
        }
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
        assert!(error.contains("tensor \"weight\" is duplicated"), "{error}");
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
