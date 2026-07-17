use std::ffi::CStr;

use crate::{
    device::Device,
    error::Result,
    utils::{guard::Guarded, runtime_lock, SUCCESS},
};

/// Explicit execution context for MLX operations.
///
/// A context owns the stream used to schedule work. Construct it from an
/// explicit device instead of relying on MLX's process- or thread-local default
/// device/stream state.
#[derive(Debug)]
pub struct ExecutionContext {
    device: Device,
    stream: Stream,
}

impl ExecutionContext {
    /// Create a context with a new stream on `device`.
    pub fn new(device: Device) -> Self {
        let stream = Stream::new_with_device(&device);
        Self { device, stream }
    }

    /// The device associated with this context.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// The stream associated with this context.
    pub fn stream(&self) -> &Stream {
        &self.stream
    }
}

impl AsRef<Stream> for ExecutionContext {
    fn as_ref(&self) -> &Stream {
        &self.stream
    }
}

/// A stream of evaluation attached to a particular device.
///
/// Typically, this is used via the `stream:` parameter on MLX operations.
pub struct Stream {
    pub(crate) c_stream: safemlx_sys::mlx_stream,
}

impl AsRef<Stream> for Stream {
    fn as_ref(&self) -> &Stream {
        self
    }
}

impl Clone for Stream {
    fn clone(&self) -> Self {
        let _guard = runtime_lock::enter();
        Stream::try_from_op(|res| unsafe { safemlx_sys::mlx_stream_set(res, self.c_stream) })
            .expect("Failed to clone stream")
    }
}

impl Stream {
    /// Create a new stream on the given device
    pub fn new_with_device(device: &Device) -> Stream {
        let _guard = runtime_lock::enter();
        unsafe {
            let c_stream = safemlx_sys::mlx_stream_new_device(device.c_device);
            Stream { c_stream }
        }
    }

    /// Get the underlying C pointer.
    pub fn as_ptr(&self) -> safemlx_sys::mlx_stream {
        self.c_stream
    }

    /// Get the index of the stream.
    pub fn get_index(&self) -> Result<i32> {
        i32::try_from_op(|res| unsafe { safemlx_sys::mlx_stream_get_index(res, self.c_stream) })
    }

    /// Synchronize with the stream.
    pub fn synchronize(&self) -> Result<()> {
        let _guard = runtime_lock::enter();
        <() as Guarded>::try_from_op(|_| unsafe { safemlx_sys::mlx_synchronize(self.c_stream) })
    }

    /// Get the device associated with the stream.
    pub fn get_device(&self) -> Result<Device> {
        Device::try_from_op(|res| unsafe { safemlx_sys::mlx_stream_get_device(res, self.c_stream) })
    }

    fn describe(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let _guard = runtime_lock::enter();
        unsafe {
            let mut mlx_str = safemlx_sys::mlx_string_new();
            let result =
                match safemlx_sys::mlx_stream_tostring(&mut mlx_str as *mut _, self.c_stream) {
                    SUCCESS => {
                        let ptr = safemlx_sys::mlx_string_data(mlx_str);
                        let c_str = CStr::from_ptr(ptr);
                        write!(f, "{}", c_str.to_string_lossy())
                    }
                    _ => Err(std::fmt::Error),
                };
            safemlx_sys::mlx_string_free(mlx_str);
            result
        }
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        let _guard = runtime_lock::enter();
        unsafe { safemlx_sys::mlx_stream_free(self.c_stream) };
    }
}

impl std::fmt::Debug for Stream {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.describe(f)
    }
}

impl std::fmt::Display for Stream {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.describe(f)
    }
}

impl PartialEq for Stream {
    fn eq(&self, other: &Self) -> bool {
        unsafe { safemlx_sys::mlx_stream_equal(self.c_stream, other.c_stream) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_clone() {
        let stream = Stream::new_with_device(&crate::Device::new(crate::DeviceType::Gpu, 0));
        let cloned_stream = stream.clone();
        assert_eq!(stream, cloned_stream);
    }

    #[test]
    fn test_cpu_gpu_stream_not_equal() {
        let cpu_stream = Stream::new_with_device(&crate::Device::new(crate::DeviceType::Cpu, 0));
        let gpu_stream = Stream::new_with_device(&crate::Device::new(crate::DeviceType::Gpu, 0));

        // Assert that CPU and GPU streams are not equal
        assert_ne!(cpu_stream, gpu_stream);
    }

    #[test]
    fn execution_context_can_be_used_as_stream() {
        let ctx = ExecutionContext::new(crate::Device::new(crate::DeviceType::Cpu, 0));
        let array = crate::Array::zeros::<f32>(&[2], &ctx).unwrap();
        assert_eq!(array.shape(), &[2]);
    }

    #[test]
    fn cpu_stream_creation_is_concurrent_safe() {
        std::thread::scope(|scope| {
            for _ in 0..crate::test_concurrency() {
                scope.spawn(|| {
                    for _ in 0..64 {
                        let stream =
                            Stream::new_with_device(&crate::Device::new(crate::DeviceType::Cpu, 0));
                        let x = crate::Array::zeros::<f32>(&[1], &stream).unwrap();
                        x.evaluated().unwrap();
                    }
                });
            }
        });
    }
}
