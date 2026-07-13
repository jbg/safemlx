use crate::error::{Exception, IoError};
use crate::ops::GgufMetadataValue;
use crate::utils::SUCCESS;
use crate::{Array, DeviceType, Stream};
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::path::Path;
use std::ptr::null_mut;

use super::Guarded;

pub(crate) struct Gguf {
    pub(crate) inner: safemlx_sys::mlx_io_gguf,
}

impl Drop for Gguf {
    fn drop(&mut self) {
        unsafe {
            safemlx_sys::mlx_io_gguf_free(self.inner);
        }
    }
}

impl Gguf {
    pub(crate) fn load_device(path: &Path, stream: impl AsRef<Stream>) -> Result<Self, IoError> {
        if !path.is_file() {
            return Err(IoError::NotFile);
        }

        let extension = path
            .extension()
            .and_then(|ext| ext.to_str())
            .ok_or(IoError::UnsupportedFormat)?;
        if !extension.eq_ignore_ascii_case("gguf") {
            return Err(IoError::UnsupportedFormat);
        }

        let path_str = path.to_str().ok_or(IoError::InvalidUtf8)?;
        let filepath = CString::new(path_str)?;
        let stream = stream.as_ref();
        let device_type = stream.get_device()?.get_type()?;
        if device_type != DeviceType::Cpu {
            return Err(IoError::Exception(Exception::custom(
                "native MLX GGUF loading requires a CPU stream",
            )));
        }

        Gguf::try_from_op(|res| unsafe {
            safemlx_sys::mlx_load_gguf(res, filepath.as_ptr(), stream.as_ptr())
        })
        .map_err(Into::into)
    }

    pub(crate) fn data(&self) -> Result<HashMap<String, Array>, Exception> {
        let keys = self.keys(false)?;
        let mut data = HashMap::with_capacity(keys.len());
        for key in keys {
            let c_key =
                CString::new(key.as_str()).map_err(|error| Exception::custom(error.to_string()))?;
            let value = Array::try_from_op(|res| unsafe {
                safemlx_sys::mlx_io_gguf_get_array(res, self.inner, c_key.as_ptr())
            })?;
            data.insert(key, value);
        }
        Ok(data)
    }

    pub(crate) fn metadata(&self) -> Result<HashMap<String, GgufMetadataValue>, Exception> {
        let keys = self.keys(true)?;
        let mut metadata = HashMap::with_capacity(keys.len());
        for key in keys {
            let c_key =
                CString::new(key.as_str()).map_err(|error| Exception::custom(error.to_string()))?;
            let mut is_array = false;
            check_status(unsafe {
                safemlx_sys::mlx_io_gguf_has_metadata_array(
                    &mut is_array,
                    self.inner,
                    c_key.as_ptr(),
                )
            })?;
            if is_array {
                let value = Array::try_from_op(|res| unsafe {
                    safemlx_sys::mlx_io_gguf_get_metadata_array(res, self.inner, c_key.as_ptr())
                })?;
                metadata.insert(key, GgufMetadataValue::Array(value));
                continue;
            }

            let mut is_string = false;
            check_status(unsafe {
                safemlx_sys::mlx_io_gguf_has_metadata_string(
                    &mut is_string,
                    self.inner,
                    c_key.as_ptr(),
                )
            })?;
            if is_string {
                let value = unsafe {
                    let mut value = safemlx_sys::mlx_string_new();
                    let status = safemlx_sys::mlx_io_gguf_get_metadata_string(
                        &mut value,
                        self.inner,
                        c_key.as_ptr(),
                    );
                    if let Err(error) = check_status(status) {
                        safemlx_sys::mlx_string_free(value);
                        return Err(error);
                    }
                    let value_str = CStr::from_ptr(safemlx_sys::mlx_string_data(value))
                        .to_string_lossy()
                        .into_owned();
                    safemlx_sys::mlx_string_free(value);
                    value_str
                };
                metadata.insert(key, GgufMetadataValue::String(value));
                continue;
            }

            let mut is_vector_string = false;
            check_status(unsafe {
                safemlx_sys::mlx_io_gguf_has_metadata_vector_string(
                    &mut is_vector_string,
                    self.inner,
                    c_key.as_ptr(),
                )
            })?;
            if is_vector_string {
                let values = unsafe {
                    let mut values = safemlx_sys::mlx_vector_string_new();
                    let status = safemlx_sys::mlx_io_gguf_get_metadata_vector_string(
                        &mut values,
                        self.inner,
                        c_key.as_ptr(),
                    );
                    if let Err(error) = check_status(status) {
                        safemlx_sys::mlx_vector_string_free(values);
                        return Err(error);
                    }
                    vector_strings(values)?
                };
                metadata.insert(key, GgufMetadataValue::Strings(values));
                continue;
            }

            return Err(Exception::custom(format!(
                "GGUF metadata key {key:?} has an unsupported value type"
            )));
        }
        Ok(metadata)
    }

    fn keys(&self, metadata: bool) -> Result<Vec<String>, Exception> {
        unsafe {
            let mut keys = safemlx_sys::mlx_vector_string_new();
            let status = if metadata {
                safemlx_sys::mlx_io_gguf_get_metadata_keys(&mut keys, self.inner)
            } else {
                safemlx_sys::mlx_io_gguf_get_keys(&mut keys, self.inner)
            };
            if let Err(error) = check_status(status) {
                safemlx_sys::mlx_vector_string_free(keys);
                return Err(error);
            }
            vector_strings(keys)
        }
    }
}

unsafe fn vector_strings(values: safemlx_sys::mlx_vector_string) -> Result<Vec<String>, Exception> {
    let size = unsafe { safemlx_sys::mlx_vector_string_size(values) };
    let mut strings = Vec::with_capacity(size);
    for index in 0..size {
        let mut value = null_mut();
        let status = unsafe { safemlx_sys::mlx_vector_string_get(&mut value, values, index) };
        if let Err(error) = check_status(status) {
            unsafe {
                safemlx_sys::mlx_vector_string_free(values);
            }
            return Err(error);
        }
        strings.push(
            unsafe { CStr::from_ptr(value) }
                .to_string_lossy()
                .into_owned(),
        );
    }
    unsafe {
        safemlx_sys::mlx_vector_string_free(values);
    }
    Ok(strings)
}

fn check_status(status: i32) -> Result<(), Exception> {
    if status == SUCCESS {
        Ok(())
    } else {
        let what = crate::error::get_and_clear_last_mlx_error()
            .map(|error| error.what)
            .unwrap_or_else(|| format!("MLX GGUF operation failed with status {status}"));
        Err(Exception::custom(what))
    }
}

pub(crate) struct SafeTensors {
    pub(crate) c_data: safemlx_sys::mlx_map_string_to_array,
    pub(crate) c_metadata: safemlx_sys::mlx_map_string_to_string,
}

impl Drop for SafeTensors {
    fn drop(&mut self) {
        unsafe {
            safemlx_sys::mlx_map_string_to_string_free(self.c_metadata);
            safemlx_sys::mlx_map_string_to_array_free(self.c_data);
        }
    }
}

impl SafeTensors {
    pub(crate) fn load_device(path: &Path, stream: impl AsRef<Stream>) -> Result<Self, IoError> {
        if !path.is_file() {
            return Err(IoError::NotFile);
        }

        let extension = path
            .extension()
            .and_then(|ext| ext.to_str())
            .ok_or(IoError::UnsupportedFormat)?;

        if extension != "safetensors" {
            return Err(IoError::UnsupportedFormat);
        }

        let path_str = path.to_str().ok_or(IoError::InvalidUtf8)?;
        let filepath = CString::new(path_str)?;
        let stream = stream.as_ref();
        let device_type = stream.get_device()?.get_type()?;
        if device_type != DeviceType::Cpu {
            return Err(IoError::Exception(Exception::custom(
                "native MLX safetensors loading requires a CPU stream",
            )));
        }

        SafeTensors::try_from_op(|(res_0, res_1)| unsafe {
            safemlx_sys::mlx_load_safetensors(res_0, res_1, filepath.as_ptr(), stream.as_ptr())
        })
        .map_err(Into::into)
    }

    pub(crate) fn data(&self) -> Result<HashMap<String, Array>, Exception> {
        crate::error::ensure_mlx_error_handler();
        let mut map = HashMap::new();
        unsafe {
            let iterator = safemlx_sys::mlx_map_string_to_array_iterator_new(self.c_data);

            loop {
                let mut key_ptr: *const ::std::os::raw::c_char = null_mut();
                let mut value = safemlx_sys::mlx_array_new();
                let status = safemlx_sys::mlx_map_string_to_array_iterator_next(
                    &mut key_ptr as *mut *const _,
                    &mut value,
                    iterator,
                );

                match status {
                    SUCCESS => {
                        let key = CStr::from_ptr(key_ptr).to_string_lossy().into_owned();
                        let array = Array::from_ptr(value);
                        map.insert(key, array);
                    }
                    1 => {
                        safemlx_sys::mlx_array_free(value);
                        return Err(crate::error::get_and_clear_last_mlx_error()
                            .expect("A non-success status was returned, but no error was set.")
                            .into());
                    }
                    2 => {
                        safemlx_sys::mlx_array_free(value);
                        break;
                    }
                    _ => unreachable!(),
                }
            }

            safemlx_sys::mlx_map_string_to_array_iterator_free(iterator);
        }

        Ok(map)
    }

    pub(crate) fn metadata(&self) -> Result<HashMap<String, String>, Exception> {
        crate::error::ensure_mlx_error_handler();

        let mut map = HashMap::new();
        unsafe {
            let iterator = safemlx_sys::mlx_map_string_to_string_iterator_new(self.c_metadata);

            let mut key: *const ::std::os::raw::c_char = null_mut();
            let mut value: *const ::std::os::raw::c_char = null_mut();
            loop {
                let status = safemlx_sys::mlx_map_string_to_string_iterator_next(
                    &mut key as *mut *const _,
                    &mut value as *mut *const _,
                    iterator,
                );

                match status {
                    SUCCESS => {
                        let key = CStr::from_ptr(key).to_string_lossy().into_owned();
                        let value = CStr::from_ptr(value).to_string_lossy().into_owned();
                        map.insert(key, value);
                    }
                    1 => {
                        return Err(crate::error::get_and_clear_last_mlx_error()
                            .expect("A non-success status was returned, but no error was set.")
                            .into());
                    }
                    2 => break,
                    _ => unreachable!(),
                }
            }
        }

        Ok(map)
    }
}
