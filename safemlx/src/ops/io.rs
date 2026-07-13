use crate::error::IoError;
use crate::utils::guard::Guarded;
use crate::utils::io::{Gguf, SafeTensors};
use crate::utils::SUCCESS;
use crate::{Array, Stream};
use std::collections::HashMap;
use std::ffi::CString;
use std::path::Path;

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

    /// Loads all tensors from a GGUF file.
    ///
    /// MLX preserves Q2_K, Q3_K, Q4_0, Q4_1, Q4_K, Q5_K, Q6_K, and Q8_0 tensors in its packed
    /// affine representation. Some other GGUF quantization formats are
    /// converted to floating point by MLX while loading; formats unsupported
    /// by MLX's bundled GGUF converter return an error.
    pub fn load_gguf(
        path: impl AsRef<Path>,
        stream: impl AsRef<Stream>,
    ) -> Result<HashMap<String, Array>, IoError> {
        let gguf = Gguf::load_device(path.as_ref(), stream)?;
        gguf.data().map_err(Into::into)
    }

    /// Loads all tensors and typed metadata from a GGUF file.
    #[allow(clippy::type_complexity)]
    pub fn load_gguf_with_metadata(
        path: impl AsRef<Path>,
        stream: impl AsRef<Stream>,
    ) -> Result<(HashMap<String, Array>, HashMap<String, GgufMetadataValue>), IoError> {
        let gguf = Gguf::load_device(path.as_ref(), stream)?;
        let data = gguf.data()?;
        let metadata = gguf.metadata()?;
        Ok((data, metadata))
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

#[cfg(test)]
mod tests {
    use crate::{ops::GgufMetadataValue, transforms::eval, Array};
    use std::ffi::CString;

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
}
