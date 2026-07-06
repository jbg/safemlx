use crate::error::IoError;
use crate::utils::guard::Guarded;
#[cfg(not(feature = "safetensors"))]
use crate::utils::io::SafeTensors;
use crate::utils::SUCCESS;
use crate::{Array, Stream};
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
    ///
    #[cfg(feature = "safetensors")]
    pub fn load_safetensors(path: impl AsRef<Path>) -> Result<HashMap<String, Array>, IoError> {
        let (data, _) = load_safetensors_with_rust_parser(path.as_ref())?;
        Ok(data)
    }

    /// Load dictionary of ``MLXArray`` from a `safetensors` file.
    ///
    /// # Params
    ///
    /// - path: path of file to load
    /// - stream: stream or device to load on
    #[cfg(not(feature = "safetensors"))]
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
    #[allow(clippy::type_complexity)]
    #[cfg(feature = "safetensors")]
    pub fn load_safetensors_with_metadata(
        path: impl AsRef<Path>,
    ) -> Result<(HashMap<String, Array>, HashMap<String, String>), IoError> {
        load_safetensors_with_rust_parser(path.as_ref())
    }

    /// Load dictionary of ``MLXArray`` and metadata `[String:String]` from a `safetensors` file.
    ///
    /// # Params
    ///
    /// - path: path of file to load
    /// - stream: stream or device to load on
    #[allow(clippy::type_complexity)]
    #[cfg(not(feature = "safetensors"))]
    pub fn load_safetensors_with_metadata(
        path: impl AsRef<Path>,
        stream: impl AsRef<Stream>,
    ) -> Result<(HashMap<String, Array>, HashMap<String, String>), IoError> {
        let safetensors = SafeTensors::load_device(path.as_ref(), stream)?;
        let data = safetensors.data()?;
        let metadata = safetensors.metadata()?;

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

#[cfg(feature = "safetensors")]
fn load_safetensors_with_rust_parser(
    path: &Path,
) -> Result<(HashMap<String, Array>, HashMap<String, String>), IoError> {
    if !path.is_file() {
        return Err(IoError::NotFile);
    }
    check_file_extension(path, "safetensors")?;

    let file = std::fs::File::open(path).map_err(|err| {
        IoError::Exception(crate::error::Exception::custom(format!(
            "failed to open safetensors file {}: {err}",
            path.display()
        )))
    })?;
    let bytes = unsafe {
        memmap2::MmapOptions::new().map(&file).map_err(|err| {
            IoError::Exception(crate::error::Exception::custom(format!(
                "failed to map safetensors file {}: {err}",
                path.display()
            )))
        })?
    };
    let metadata = safetensors::SafeTensors::read_metadata(&bytes)
        .map_err(|err| {
            IoError::Exception(crate::error::Exception::custom(format!(
                "failed to parse safetensors metadata from {}: {err}",
                path.display()
            )))
        })?
        .1
        .metadata()
        .clone()
        .unwrap_or_default();

    let safetensors = safetensors::SafeTensors::deserialize(&bytes).map_err(|err| {
        IoError::Exception(crate::error::Exception::custom(format!(
            "failed to parse safetensors file {}: {err}",
            path.display()
        )))
    })?;

    let mut data = HashMap::new();
    for name in safetensors.names() {
        let tensor = safetensors.tensor(name).map_err(|err| {
            IoError::Exception(crate::error::Exception::custom(format!(
                "failed to read tensor {name} from {}: {err}",
                path.display()
            )))
        })?;
        let array = Array::try_from(tensor).map_err(|err| {
            IoError::Exception(crate::error::Exception::custom(format!(
                "failed to convert tensor {name} from {}: {err}",
                path.display()
            )))
        })?;
        data.insert(name.to_string(), array);
    }

    Ok((data, metadata))
}

#[cfg(test)]
mod tests {
    use crate::Array;

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

        #[cfg(feature = "safetensors")]
        let loaded_arrays = Array::load_safetensors(&path).unwrap();
        #[cfg(not(feature = "safetensors"))]
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
}
