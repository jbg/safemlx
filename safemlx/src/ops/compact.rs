use safemlx_internal_macros::{default_device, generate_macro};

use crate::{
    error::{Exception, Result},
    ops::{concatenate_axis_device, indexing::scatter_max_single_device, r#where_device},
    Array, Dtype, Stream,
};

/// Padded compact indices plus the device-side number of valid entries.
///
/// The valid entries are always the first `count` elements or rows of `indices`.
/// Remaining entries are padded with `-1`.
#[derive(Clone, Debug)]
pub struct CompactIndices {
    /// Padded compacted indices.
    pub indices: Array,

    /// Scalar `i32` array containing the number of valid entries.
    pub count: Array,
}

fn bool_condition(mask: &Array, stream: impl AsRef<Stream>) -> Result<Array> {
    if mask.dtype() == Dtype::Bool {
        Ok(mask.clone())
    } else {
        mask.ne_device(Array::from_int(0), stream)
    }
}

fn checked_size_i32(a: &Array) -> Result<i32> {
    i32::try_from(a.size()).map_err(|_| Exception::from("array is too large to compact as i32"))
}

impl Array {
    /// Count non-zero entries and keep the result on device.
    ///
    /// Returns a scalar `i32` array.
    #[default_device]
    pub fn count_nonzero_device(&self, stream: impl AsRef<Stream>) -> Result<Array> {
        let stream = stream.as_ref();
        let condition = bool_condition(self, stream)?;
        condition
            .as_type_device::<i32>(stream)?
            .sum_device(None, stream)
    }

    /// Compact flattened indices where `self` is non-zero.
    ///
    /// Returns a fixed-capacity `[self.size()]` `i32` index buffer and a scalar `i32`
    /// count. The first `count` entries are valid, and the rest are padded with `-1`.
    #[default_device]
    pub fn compact_indices_device(&self, stream: impl AsRef<Stream>) -> Result<CompactIndices> {
        let stream = stream.as_ref();
        let condition = bool_condition(self, stream)?;
        let flat = condition.reshape_device(&[-1], stream)?;
        let n = checked_size_i32(&flat)?;
        let flags = flat.as_type_device::<i32>(stream)?;
        let count = flags.sum_device(None, stream)?;
        let padded = Array::full_device::<i32>(&[n], Array::from_int(-1), stream)?;

        if n == 0 {
            return Ok(CompactIndices {
                indices: padded,
                count,
            });
        }

        let prefix = flags.cumsum_device(0, None, None, stream)?;
        let positions = prefix.subtract_device(Array::from_int(1), stream)?;
        let zero_positions = Array::zeros_device::<i32>(&[n], stream)?;
        let scatter_positions = r#where_device(&flat, positions, zero_positions, stream)?;

        let flat_indices = Array::arange_device::<_, i32>(None, n, None, stream)?;
        let masked_indices = r#where_device(&flat, flat_indices, padded.clone(), stream)?
            .reshape_device(&[n, 1], stream)?;
        let indices =
            scatter_max_single_device(padded, scatter_positions, masked_indices, 0, stream)?;

        Ok(CompactIndices { indices, count })
    }

    /// Alias for [`Array::compact_indices`].
    ///
    /// This is a fixed-capacity device-side form of `nonzero`; inspect only the
    /// first `count` entries of the returned `indices`.
    #[default_device]
    pub fn nonzero_device(&self, stream: impl AsRef<Stream>) -> Result<CompactIndices> {
        self.compact_indices_device(stream)
    }

    /// Return padded coordinate rows where `self` is non-zero.
    ///
    /// Returns a fixed-capacity `[self.size(), self.ndim()]` `i32` coordinate
    /// buffer and a scalar `i32` count. The first `count` rows are valid, and
    /// remaining rows are padded with `-1`.
    #[default_device]
    pub fn argwhere_device(&self, stream: impl AsRef<Stream>) -> Result<CompactIndices> {
        let stream = stream.as_ref();
        let flat = self.compact_indices_device(stream)?;
        let n = checked_size_i32(self)?;
        let rank = i32::try_from(self.ndim())
            .map_err(|_| Exception::from("array rank is too large to compact as i32"))?;

        if rank == 0 {
            let indices = Array::full_device::<i32>(&[n, 0], Array::from_int(-1), stream)?;
            return Ok(CompactIndices {
                indices,
                count: flat.count,
            });
        }

        let valid = flat.indices.ge_device(Array::from_int(0), stream)?;
        let safe_flat = r#where_device(
            &valid,
            flat.indices.clone(),
            Array::zeros_device::<i32>(&[n], stream)?,
            stream,
        )?;
        let padding = Array::full_device::<i32>(&[n], Array::from_int(-1), stream)?;

        let mut stride = 1;
        let mut coords = Vec::with_capacity(self.ndim());
        for &dim in self.shape().iter().rev() {
            let coord = safe_flat
                .floor_divide_device(Array::from_int(stride), stream)?
                .remainder_device(Array::from_int(dim), stream)?;
            coords.push(r#where_device(&valid, coord, padding.clone(), stream)?);
            stride = stride
                .checked_mul(dim)
                .ok_or_else(|| Exception::from("array shape is too large to compact as i32"))?;
        }
        coords.reverse();

        let coord_cols = coords
            .iter()
            .map(|coord| coord.expand_dims_device(1, stream))
            .collect::<Result<Vec<_>>>()?;
        let indices = concatenate_axis_device(&coord_cols, 1, stream)?;

        Ok(CompactIndices {
            indices,
            count: flat.count,
        })
    }
}

/// See [`Array::count_nonzero`].
#[generate_macro]
#[default_device]
pub fn count_nonzero_device(
    a: impl AsRef<Array>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    a.as_ref().count_nonzero_device(stream)
}

/// See [`Array::compact_indices`].
#[generate_macro]
#[default_device]
pub fn compact_indices_device(
    a: impl AsRef<Array>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<CompactIndices> {
    a.as_ref().compact_indices_device(stream)
}

/// See [`Array::nonzero`].
#[generate_macro]
#[default_device]
pub fn nonzero_device(
    a: impl AsRef<Array>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<CompactIndices> {
    a.as_ref().nonzero_device(stream)
}

/// See [`Array::argwhere`].
#[generate_macro]
#[default_device]
pub fn argwhere_device(
    a: impl AsRef<Array>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<CompactIndices> {
    a.as_ref().argwhere_device(stream)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array;

    #[test]
    fn test_count_nonzero_bool() {
        let mask = array!([true, false, true, true]);
        let count = count_nonzero(&mask).unwrap();
        assert_eq!(count.item::<i32>(), 3);
    }

    #[test]
    fn test_count_nonzero_numeric() {
        let mask = array!([0, 4, -2, 0]);
        let count = count_nonzero(&mask).unwrap();
        assert_eq!(count.item::<i32>(), 2);
    }

    #[test]
    fn test_compact_indices() {
        let mask = array!([false, true, false, true, true, false]);
        let out = compact_indices(&mask).unwrap();
        assert_eq!(out.count.item::<i32>(), 3);
        assert_eq!(out.indices.shape(), &[6]);
        assert_eq!(out.indices.as_slice::<i32>(), &[1, 3, 4, -1, -1, -1]);
    }

    #[test]
    fn test_compact_indices_empty_mask() {
        let mask = Array::from_slice::<bool>(&[], &[0]);
        let out = compact_indices(&mask).unwrap();
        assert_eq!(out.count.item::<i32>(), 0);
        assert_eq!(out.indices.shape(), &[0]);
        assert_eq!(out.indices.size(), 0);
    }

    #[test]
    fn test_compact_indices_all_false() {
        let mask = array!([false, false, false]);
        let out = compact_indices(&mask).unwrap();
        assert_eq!(out.count.item::<i32>(), 0);
        assert_eq!(out.indices.as_slice::<i32>(), &[-1, -1, -1]);
    }

    #[test]
    fn test_argwhere_1d() {
        let mask = array!([false, true, false, true]);
        let out = argwhere(&mask).unwrap();
        assert_eq!(out.count.item::<i32>(), 2);
        assert_eq!(out.indices.shape(), &[4, 1]);
        assert_eq!(out.indices.as_slice::<i32>(), &[1, 3, -1, -1]);
    }

    #[test]
    fn test_argwhere_2d() {
        let mask = array!([[true, false, true], [false, true, false]]);
        let out = argwhere(&mask).unwrap();
        assert_eq!(out.count.item::<i32>(), 3);
        assert_eq!(out.indices.shape(), &[6, 2]);
        assert_eq!(
            out.indices.as_slice::<i32>(),
            &[0, 0, 0, 2, 1, 1, -1, -1, -1, -1, -1, -1]
        );
    }
}
