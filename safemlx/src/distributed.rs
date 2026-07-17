//! Distributed communication groups and operations.
//!
//! MLX caches initialized backends process-wide. [`init`] preserves that
//! behavior. With `strict == false`, failure to establish the requested
//! backend returns a usable size-one group; collectives on that group return
//! their input unchanged. Point-to-point operations still reject a singleton.
//!
//! Backend setup follows MLX 0.32:
//!
//! - Ring uses `MLX_RANK` and a JSON file named by `MLX_HOSTFILE`.
//! - MPI dynamically loads Open MPI; `MLX_MPI_LIBNAME` can override its library.
//! - JACCL uses `MLX_RANK`, `MLX_IBV_DEVICES`, and
//!   `MLX_JACCL_COORDINATOR` (or their `JACCL_*` aliases) on supported macOS
//!   systems.
//! - NCCL uses `NCCL_HOST_IP`, `NCCL_PORT`, `MLX_RANK`, and
//!   `MLX_WORLD_SIZE` in NCCL-enabled builds.
//!
//! Operations are lazy, just like other MLX array operations. Evaluate their
//! returned arrays at the synchronization points required by the application.

use std::{ffi::c_char, marker::PhantomData, rc::Rc, str::FromStr};

use crate::{
    error::{Exception, Result},
    utils::{guard::Guarded, runtime_lock, SUCCESS},
    Array, Device, DeviceType, Dtype, Stream,
};

/// A distributed communication backend supported by MLX.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Backend {
    /// Let MLX choose an available backend.
    Any,
    /// MLX's TCP ring backend.
    Ring,
    /// Open MPI, loaded dynamically by MLX.
    Mpi,
    /// JACCL over RDMA on supported Apple systems.
    Jaccl,
    /// NCCL in CUDA/NCCL-enabled builds.
    Nccl,
}

impl Backend {
    /// Return the backend name accepted by MLX.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Any => "any",
            Self::Ring => "ring",
            Self::Mpi => "mpi",
            Self::Jaccl => "jaccl",
            Self::Nccl => "nccl",
        }
    }

    fn as_c_ptr(self) -> *const c_char {
        match self {
            // A null backend selects MLX's default/"any" behavior.
            Self::Any => std::ptr::null(),
            Self::Ring => c"ring".as_ptr(),
            Self::Mpi => c"mpi".as_ptr(),
            Self::Jaccl => c"jaccl".as_ptr(),
            Self::Nccl => c"nccl".as_ptr(),
        }
    }
}

impl FromStr for Backend {
    type Err = Exception;

    fn from_str(value: &str) -> Result<Self> {
        if value.as_bytes().contains(&0) {
            return Err(Exception::custom(
                "distributed backend name contains an interior NUL byte",
            ));
        }

        match value {
            "any" => Ok(Self::Any),
            "ring" => Ok(Self::Ring),
            "mpi" => Ok(Self::Mpi),
            "jaccl" => Ok(Self::Jaccl),
            "nccl" => Ok(Self::Nccl),
            _ => Err(Exception::custom(format!(
                "unknown distributed backend {value:?}; expected any, ring, mpi, jaccl, or nccl"
            ))),
        }
    }
}

impl TryFrom<&str> for Backend {
    type Error = Exception;

    fn try_from(value: &str) -> Result<Self> {
        value.parse()
    }
}

impl TryFrom<String> for Backend {
    type Error = Exception;

    fn try_from(value: String) -> Result<Self> {
        value.parse()
    }
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// An owned MLX distributed group.
///
/// The group frees its native handle on drop. It is intentionally neither
/// `Clone`, `Send`, nor `Sync`: the C API exposes no group-retain operation and
/// not every communication backend documents cross-thread group access.
pub struct Group {
    pub(crate) c_group: safemlx_sys::mlx_distributed_group,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl Group {
    pub(crate) fn from_owned_ptr(c_group: safemlx_sys::mlx_distributed_group) -> Self {
        Self {
            c_group,
            _not_send_or_sync: PhantomData,
        }
    }

    /// Initialize and own the process's group for `backend`.
    ///
    /// If `strict` is `false` and MLX cannot initialize the backend, the result
    /// is a size-one group. If `strict` is `true`, initialization instead
    /// returns the MLX error.
    pub fn init(strict: bool, backend: Backend) -> Result<Self> {
        init(strict, backend)
    }

    /// Return this process's zero-based rank within the group.
    pub fn rank(&self) -> usize {
        let _guard = runtime_lock::enter();
        // SAFETY: `self` owns a successfully initialized, non-empty group.
        let rank = unsafe { safemlx_sys::mlx_distributed_group_rank(self.c_group) };
        usize::try_from(rank).expect("MLX returned a negative distributed rank")
    }

    /// Return the number of processes in the group.
    pub fn size(&self) -> usize {
        let _guard = runtime_lock::enter();
        // SAFETY: `self` owns a successfully initialized, non-empty group.
        let size = unsafe { safemlx_sys::mlx_distributed_group_size(self.c_group) };
        usize::try_from(size).expect("MLX returned a negative distributed group size")
    }

    /// Split the group by `color`, optionally ordering new ranks by `key`.
    ///
    /// A missing or negative key asks MLX to use the current group rank. Backend
    /// support varies: MLX 0.32 supports splitting with MPI and NCCL, while its
    /// singleton, Ring, and JACCL groups return an error.
    pub fn split(&self, color: i32, key: Option<i32>) -> Result<Self> {
        let _guard = runtime_lock::enter();
        Self::try_from_op(|res| {
            // SAFETY: `res` is an initialized output guard and `self.c_group`
            // remains alive for the duration of this call.
            unsafe {
                safemlx_sys::mlx_distributed_group_split(
                    res,
                    self.c_group,
                    color,
                    key.unwrap_or(-1),
                )
            }
        })
    }
}

impl Drop for Group {
    fn drop(&mut self) {
        let _guard = runtime_lock::enter();
        // SAFETY: this is the sole Rust owner of the native group handle.
        let status = unsafe { safemlx_sys::mlx_distributed_group_free(self.c_group) };
        debug_assert_eq!(status, SUCCESS);
    }
}

impl std::fmt::Debug for Group {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Group")
            .field("rank", &self.rank())
            .field("size", &self.size())
            .finish()
    }
}

/// Check whether MLX was built with support for `backend`.
///
/// Availability does not imply that the backend's required environment or
/// communication peers are configured; use [`init`] to establish a group.
pub fn is_available(backend: Backend) -> bool {
    let _guard = runtime_lock::enter();
    // SAFETY: the pointer is null or a static NUL-terminated backend name.
    unsafe { safemlx_sys::mlx_distributed_is_available(backend.as_c_ptr()) }
}

/// Initialize and own the process's group for `backend`.
///
/// MLX caches backend initialization process-wide. With `strict == false`, a
/// backend that cannot be established produces a usable size-one group.
pub fn init(strict: bool, backend: Backend) -> Result<Group> {
    let _guard = runtime_lock::enter();
    Group::try_from_op(|res| {
        // SAFETY: `res` is owned by the output guard and the backend pointer is
        // null or a static NUL-terminated string.
        unsafe { safemlx_sys::mlx_distributed_init(res, strict, backend.as_c_ptr()) }
    })
}

fn collective(
    input: &Array,
    group: &Group,
    stream: &Stream,
    op: unsafe extern "C" fn(
        *mut safemlx_sys::mlx_array,
        safemlx_sys::mlx_array,
        safemlx_sys::mlx_distributed_group,
        safemlx_sys::mlx_stream,
    ) -> i32,
) -> Result<Array> {
    let _guard = runtime_lock::enter();
    Array::try_from_op(|res| {
        // SAFETY: all borrowed handles remain alive for the call and `res` is
        // an owned array output guard.
        unsafe { op(res, input.as_ptr(), group.c_group, stream.as_ptr()) }
    })
}

/// Sum `input` element-wise across `group` on `stream`.
pub fn all_sum(input: &Array, group: &Group, stream: impl AsRef<Stream>) -> Result<Array> {
    collective(
        input,
        group,
        stream.as_ref(),
        safemlx_sys::mlx_distributed_all_sum,
    )
}

/// Take the element-wise maximum of `input` across `group` on `stream`.
pub fn all_max(input: &Array, group: &Group, stream: impl AsRef<Stream>) -> Result<Array> {
    collective(
        input,
        group,
        stream.as_ref(),
        safemlx_sys::mlx_distributed_all_max,
    )
}

/// Take the element-wise minimum of `input` across `group` on `stream`.
pub fn all_min(input: &Array, group: &Group, stream: impl AsRef<Stream>) -> Result<Array> {
    collective(
        input,
        group,
        stream.as_ref(),
        safemlx_sys::mlx_distributed_all_min,
    )
}

/// Gather `input` from every rank, concatenating along axis zero.
///
/// Scalar inputs become a one-dimensional result for non-singleton groups.
pub fn all_gather(input: &Array, group: &Group, stream: impl AsRef<Stream>) -> Result<Array> {
    if group.size() > 1 {
        if let Some(&first_dim) = input.shape().first() {
            let group_size = i32::try_from(group.size())
                .map_err(|_| Exception::custom("distributed group size does not fit in i32"))?;
            first_dim.checked_mul(group_size).ok_or_else(|| {
                Exception::custom("all-gather output's first dimension exceeds i32")
            })?;
        }
    }
    collective(
        input,
        group,
        stream.as_ref(),
        safemlx_sys::mlx_distributed_all_gather,
    )
}

/// Sum across `group` and scatter equal axis-zero chunks to each rank.
pub fn sum_scatter(input: &Array, group: &Group, stream: impl AsRef<Stream>) -> Result<Array> {
    let group_size = group.size();
    if group_size > 1 {
        let first_dim = input
            .shape()
            .first()
            .copied()
            .ok_or_else(|| Exception::custom("sum-scatter requires a non-scalar input"))?;
        let first_dim = usize::try_from(first_dim)
            .map_err(|_| Exception::custom("sum-scatter input shape contains a negative size"))?;
        if first_dim % group_size != 0 {
            return Err(Exception::custom(format!(
                "sum-scatter input's first dimension ({first_dim}) is not divisible by group size ({group_size})"
            )));
        }
    }
    collective(
        input,
        group,
        stream.as_ref(),
        safemlx_sys::mlx_distributed_sum_scatter,
    )
}

fn checked_peer(peer: usize, group: &Group, role: &str) -> Result<i32> {
    let size = group.size();
    if size == 1 {
        return Err(Exception::custom(format!(
            "cannot use a {role} rank with a singleton distributed group"
        )));
    }
    if peer >= size {
        return Err(Exception::custom(format!(
            "invalid {role} rank {peer} for distributed group of size {size}"
        )));
    }
    i32::try_from(peer).map_err(|_| Exception::custom(format!("{role} rank does not fit in i32")))
}

/// Lazily send `input` to rank `destination` on `stream`.
///
/// Ring only supports direct neighbors; other backend restrictions are
/// reported by MLX.
pub fn send(
    input: &Array,
    destination: usize,
    group: &Group,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let destination = checked_peer(destination, group, "destination")?;
    let stream = stream.as_ref();
    let _guard = runtime_lock::enter();
    Array::try_from_op(|res| {
        // SAFETY: all input handles remain alive and `res` is an owned output
        // guard. `destination` was range-checked above.
        unsafe {
            safemlx_sys::mlx_distributed_send(
                res,
                input.as_ptr(),
                destination,
                group.c_group,
                stream.as_ptr(),
            )
        }
    })
}

/// Lazily receive an array of `shape` and `dtype` from rank `source`.
pub fn recv(
    shape: &[i32],
    dtype: Dtype,
    source: usize,
    group: &Group,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    if shape.iter().any(|dimension| *dimension < 0) {
        return Err(Exception::custom(
            "receive shape dimensions must be non-negative",
        ));
    }
    let source = checked_peer(source, group, "source")?;
    let stream = stream.as_ref();
    let _guard = runtime_lock::enter();
    Array::try_from_op(|res| {
        // SAFETY: `shape` and all borrowed handles remain alive for this call;
        // `source` and every shape dimension were validated above.
        unsafe {
            safemlx_sys::mlx_distributed_recv(
                res,
                shape.as_ptr(),
                shape.len(),
                dtype.into(),
                source,
                group.c_group,
                stream.as_ptr(),
            )
        }
    })
}

/// Lazily receive from rank `source`, using `like` for shape and dtype.
pub fn recv_like(
    like: &Array,
    source: usize,
    group: &Group,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let source = checked_peer(source, group, "source")?;
    let stream = stream.as_ref();
    let _guard = runtime_lock::enter();
    Array::try_from_op(|res| {
        // SAFETY: all borrowed handles remain alive and `source` was checked.
        unsafe {
            safemlx_sys::mlx_distributed_recv_like(
                res,
                like.as_ptr(),
                source,
                group.c_group,
                stream.as_ptr(),
            )
        }
    })
}

/// Select a process-local device by explicit local index.
///
/// Do not pass [`Group::rank`] blindly: global ranks span machines and need not
/// match local device indices. Launchers should pass a local rank explicitly.
/// In the common one-process-per-visible-GPU setup, `CUDA_VISIBLE_DEVICES`
/// exposes one GPU to each process, so `local_index` is usually zero even when
/// the global distributed rank is nonzero.
pub fn device_for_local_rank(device_type: DeviceType, local_index: usize) -> Result<Device> {
    let local_index = i32::try_from(local_index)
        .map_err(|_| Exception::custom("local device index does not fit in i32"))?;
    Ok(Device::new(device_type, local_index))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn singleton() -> Group {
        init(false, Backend::Ring).unwrap()
    }

    #[test]
    fn backend_names_are_checked_before_ffi() {
        assert_eq!(Backend::try_from("ring").unwrap(), Backend::Ring);
        assert!(Backend::try_from("ring\0other").is_err());
        assert!(Backend::try_from("future-backend").is_err());
    }

    #[test]
    fn non_strict_group_is_usable() {
        let group = singleton();
        assert_eq!(group.rank(), 0);
        assert_eq!(group.size(), 1);
        assert!(format!("{group:?}").contains("rank: 0"));
        assert!(format!("{group:?}").contains("size: 1"));
    }

    #[test]
    fn singleton_collectives_preserve_values() {
        let group = singleton();
        let stream = crate::test_stream();
        let input = Array::arange::<_, f32>(Some(1), 3, None::<i32>, stream).unwrap();

        for output in [
            all_sum(&input, &group, stream).unwrap(),
            all_max(&input, &group, stream).unwrap(),
            all_min(&input, &group, stream).unwrap(),
            all_gather(&input, &group, stream).unwrap(),
            sum_scatter(&input, &group, stream).unwrap(),
        ] {
            assert_eq!(output.shape(), &[2]);
            assert_eq!(output.dtype(), Dtype::Float32);
            let output = output.evaluated().unwrap();
            assert_eq!(output.as_slice::<f32>(), &[1.0, 2.0]);
        }
    }

    #[test]
    fn singleton_split_reports_backend_support() {
        let group = singleton();
        match group.split(0, None) {
            Ok(subgroup) => {
                assert_eq!(subgroup.rank(), 0);
                assert_eq!(subgroup.size(), 1);
            }
            Err(error) => assert!(error.what().contains("split")),
        }
    }

    #[test]
    fn validates_point_to_point_inputs() {
        let group = singleton();
        let stream = crate::test_stream();
        let input = Array::arange::<_, i32>(Some(0), 1, None::<i32>, stream).unwrap();
        assert!(send(&input, 0, &group, stream).is_err());
        assert!(recv(&[-1], Dtype::Int32, 0, &group, stream).is_err());
        assert!(recv_like(&input, 0, &group, stream).is_err());
    }

    #[test]
    fn local_device_index_is_explicit() {
        let device = device_for_local_rank(DeviceType::Cpu, 0).unwrap();
        assert_eq!(device.get_index().unwrap(), 0);
        assert!(device_for_local_rank(DeviceType::Cpu, usize::MAX).is_err());
    }
}
