//! Safe tensor-parallel linear layers.
//!
//! Column-parallel layers accept complete replicated inputs and retain a
//! sharded output. Row-parallel layers accept final-dimension input shards and
//! sum their full-width partial outputs across an explicit communication
//! group. Layers never retain a borrowed or process-global group.

use crate::{
    distributed::{self, Group},
    error::Exception,
    module::{Module, Param},
    nn::{Linear, QuantizedLinear},
    ops::{quantized_matmul_with_mode, quantized_packed_dimension},
    Array, Dtype, Stream,
};

fn checked_local_dimension(name: &str, dimension: i32, parts: usize) -> Result<i32, Exception> {
    if dimension <= 0 {
        return Err(Exception::custom(format!(
            "{name} must be positive, got {dimension}"
        )));
    }
    let parts_i32 = i32::try_from(parts)
        .map_err(|_| Exception::custom("tensor-parallel size does not fit in i32"))?;
    if parts == 0 || dimension % parts_i32 != 0 {
        return Err(Exception::custom(format!(
            "{name} {dimension} is not divisible by tensor-parallel size {parts}"
        )));
    }
    Ok(dimension / parts_i32)
}

fn validate_shard(rank: usize, size: usize) -> Result<(), Exception> {
    if size == 0 {
        return Err(Exception::custom("tensor-parallel size must be nonzero"));
    }
    if rank >= size {
        return Err(Exception::custom(format!(
            "tensor-parallel rank {rank} is outside size {size}"
        )));
    }
    Ok(())
}

fn validate_group(rank: usize, size: usize, group: &Group) -> Result<(), Exception> {
    if group.size() != size || group.rank() != rank {
        return Err(Exception::custom(format!(
            "layer shard rank/size {rank}/{size} does not match communication group {}/{}",
            group.rank(),
            group.size()
        )));
    }
    Ok(())
}

fn validate_last_dimension(input: &Array, expected: i32, role: &str) -> Result<(), Exception> {
    let actual = input
        .shape()
        .last()
        .copied()
        .ok_or_else(|| Exception::custom(format!("{role} requires a non-scalar input")))?;
    if actual != expected {
        return Err(Exception::custom(format!(
            "{role} expected final input dimension {expected}, got {actual} for shape {:?}",
            input.shape()
        )));
    }
    Ok(())
}

fn shard_axis(
    value: &Array,
    axis: i32,
    rank: usize,
    size: usize,
    stream: &Stream,
) -> Result<Array, Exception> {
    validate_shard(rank, size)?;
    let parts = i32::try_from(size)
        .map_err(|_| Exception::custom("tensor-parallel size does not fit in i32"))?;
    value
        .split(parts, Some(axis), stream)?
        .into_iter()
        .nth(rank)
        .ok_or_else(|| Exception::custom("validated tensor shard was absent"))
}

/// Dense column-parallel linear layer (complete input to sharded output).
#[derive(Debug, Clone)]
pub struct AllToShardedLinear {
    /// Rank-local dense linear parameters.
    pub local: Linear,
    global_input_dims: i32,
    global_output_dims: i32,
    shard_rank: usize,
    shard_size: usize,
}

impl AllToShardedLinear {
    /// Creates rank-local unloaded parameters without constructing a complete layer.
    #[allow(clippy::too_many_arguments)]
    pub fn unloaded(
        input_dims: i32,
        output_dims: i32,
        bias: bool,
        dtype: Dtype,
        shard_rank: usize,
        shard_size: usize,
        stream: impl AsRef<Stream>,
    ) -> Result<Self, Exception> {
        validate_shard(shard_rank, shard_size)?;
        let local_output = checked_local_dimension("output dimension", output_dims, shard_size)?;
        Ok(Self {
            local: Linear::unloaded(input_dims, local_output, bias, dtype, stream)?,
            global_input_dims: input_dims,
            global_output_dims: output_dims,
            shard_rank,
            shard_size,
        })
    }

    /// Shards a complete in-memory layer along its output dimension.
    pub fn from_linear(
        linear: &Linear,
        shard_rank: usize,
        shard_size: usize,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let (output_dims, input_dims) = linear.shape();
        checked_local_dimension("output dimension", output_dims, shard_size)?;
        let weight = shard_axis(&linear.weight.value, 0, shard_rank, shard_size, stream)?;
        let bias = linear
            .bias
            .value
            .as_ref()
            .map(|bias| shard_axis(bias, 0, shard_rank, shard_size, stream))
            .transpose()?;
        Ok(Self {
            local: Linear {
                weight: Param::new(weight),
                bias: Param::new(bias),
            },
            global_input_dims: input_dims,
            global_output_dims: output_dims,
            shard_rank,
            shard_size,
        })
    }

    /// Applies the rank-local projection. No collective is performed.
    pub fn forward(
        &mut self,
        input: &Array,
        group: &Group,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        validate_group(self.shard_rank, self.shard_size, group)?;
        validate_last_dimension(input, self.global_input_dims, "all-to-sharded linear")?;
        self.local.forward(input, stream)
    }

    /// Returns `(global output, global input)` dimensions.
    pub const fn global_shape(&self) -> (i32, i32) {
        (self.global_output_dims, self.global_input_dims)
    }

    /// Returns `(local output, local input)` dimensions.
    pub fn local_shape(&self) -> (i32, i32) {
        self.local.shape()
    }
}

/// Dense row-parallel linear layer (sharded input to complete output).
#[derive(Debug, Clone)]
pub struct ShardedToAllLinear {
    /// Rank-local dense weight. Its ordinary bias is held separately so it is
    /// applied only once after reduction.
    pub local: Linear,
    global_input_dims: i32,
    global_output_dims: i32,
    shard_rank: usize,
    shard_size: usize,
}

impl ShardedToAllLinear {
    /// Creates rank-local unloaded parameters without constructing a complete layer.
    #[allow(clippy::too_many_arguments)]
    pub fn unloaded(
        input_dims: i32,
        output_dims: i32,
        bias: bool,
        dtype: Dtype,
        shard_rank: usize,
        shard_size: usize,
        stream: impl AsRef<Stream>,
    ) -> Result<Self, Exception> {
        validate_shard(shard_rank, shard_size)?;
        let local_input = checked_local_dimension("input dimension", input_dims, shard_size)?;
        Ok(Self {
            local: Linear::unloaded(local_input, output_dims, bias, dtype, stream)?,
            global_input_dims: input_dims,
            global_output_dims: output_dims,
            shard_rank,
            shard_size,
        })
    }

    /// Shards a complete in-memory layer along its input dimension.
    pub fn from_linear(
        linear: &Linear,
        shard_rank: usize,
        shard_size: usize,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let (output_dims, input_dims) = linear.shape();
        checked_local_dimension("input dimension", input_dims, shard_size)?;
        Ok(Self {
            local: Linear {
                weight: Param::new(shard_axis(
                    &linear.weight.value,
                    1,
                    shard_rank,
                    shard_size,
                    stream,
                )?),
                bias: Param::new(linear.bias.value.clone()),
            },
            global_input_dims: input_dims,
            global_output_dims: output_dims,
            shard_rank,
            shard_size,
        })
    }

    /// Computes a partial projection, all-sums it, then adds ordinary bias once.
    pub fn forward(
        &mut self,
        input: &Array,
        group: &Group,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        validate_group(self.shard_rank, self.shard_size, group)?;
        let local_input = self.local.weight.value.shape()[1];
        validate_last_dimension(input, local_input, "sharded-to-all linear")?;
        let partial =
            crate::ops::matmul(input, self.local.weight.value.transpose(stream)?, stream)?;
        let mut output = distributed::all_sum(&partial, group, stream)?;
        if let Some(bias) = self.local.bias.value.as_ref() {
            output = output.add(bias, stream)?;
        }
        Ok(output)
    }

    /// Returns `(global output, global input)` dimensions.
    pub const fn global_shape(&self) -> (i32, i32) {
        (self.global_output_dims, self.global_input_dims)
    }

    /// Returns `(local output, local input)` dimensions.
    pub fn local_shape(&self) -> (i32, i32) {
        self.local.shape()
    }
}

fn validate_quantized_dimensions(
    input_dims: i32,
    group_size: i32,
    bits: i32,
) -> Result<(), Exception> {
    if group_size <= 0 || input_dims <= 0 || input_dims % group_size != 0 {
        return Err(Exception::custom(format!(
            "logical input dimension {input_dims} is not aligned to quantization group size {group_size}"
        )));
    }
    if quantized_packed_dimension(input_dims, bits) <= 0
        || input_dims.checked_mul(bits).is_none()
        || input_dims * bits % 32 != 0
    {
        return Err(Exception::custom(format!(
            "logical input dimension {input_dims} cannot be packed at {bits} bits"
        )));
    }
    Ok(())
}

/// Quantized column-parallel linear layer.
#[derive(Debug, Clone)]
pub struct QuantizedAllToShardedLinear {
    /// Rank-local packed quantized parameters.
    pub local: QuantizedLinear,
    global_input_dims: i32,
    global_output_dims: i32,
    shard_rank: usize,
    shard_size: usize,
}

impl QuantizedAllToShardedLinear {
    /// Creates rank-local unloaded packed parameters.
    #[allow(clippy::too_many_arguments)]
    pub fn unloaded(
        input_dims: i32,
        output_dims: i32,
        group_size: i32,
        bits: i32,
        mode: crate::ops::QuantizationMode,
        bias: bool,
        shard_rank: usize,
        shard_size: usize,
        stream: impl AsRef<Stream>,
    ) -> Result<Self, Exception> {
        validate_shard(shard_rank, shard_size)?;
        validate_quantized_dimensions(input_dims, group_size, bits)?;
        let local_output = checked_local_dimension("output dimension", output_dims, shard_size)?;
        Ok(Self {
            local: QuantizedLinear::unloaded_with_mode(
                input_dims,
                local_output,
                group_size,
                bits,
                mode,
                bias,
                stream,
            )?,
            global_input_dims: input_dims,
            global_output_dims: output_dims,
            shard_rank,
            shard_size,
        })
    }

    /// Shards a complete quantized layer along logical output rows.
    pub fn from_quantized(
        linear: &QuantizedLinear,
        shard_rank: usize,
        shard_size: usize,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let output_dims = linear.inner.weight.value.shape()[0];
        let input_dims = linear.scales.value.shape()[1] * linear.group_size;
        checked_local_dimension("output dimension", output_dims, shard_size)?;
        let local = QuantizedLinear {
            group_size: linear.group_size,
            bits: linear.bits,
            mode: linear.mode,
            native: None,
            native_format: None,
            native_endian: safemlx_gguf::Endian::Little,
            native_columns: 0,
            scales: Param::new(shard_axis(
                &linear.scales.value,
                0,
                shard_rank,
                shard_size,
                stream,
            )?),
            biases: Param::new(
                linear
                    .biases
                    .value
                    .as_ref()
                    .map(|value| shard_axis(value, 0, shard_rank, shard_size, stream))
                    .transpose()?,
            ),
            inner: Linear {
                weight: Param::new(shard_axis(
                    &linear.inner.weight.value,
                    0,
                    shard_rank,
                    shard_size,
                    stream,
                )?),
                bias: Param::new(
                    linear
                        .inner
                        .bias
                        .value
                        .as_ref()
                        .map(|value| shard_axis(value, 0, shard_rank, shard_size, stream))
                        .transpose()?,
                ),
            },
        };
        Ok(Self {
            local,
            global_input_dims: input_dims,
            global_output_dims: output_dims,
            shard_rank,
            shard_size,
        })
    }

    /// Applies the rank-local packed projection without a collective.
    pub fn forward(
        &mut self,
        input: &Array,
        group: &Group,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        validate_group(self.shard_rank, self.shard_size, group)?;
        validate_last_dimension(
            input,
            self.global_input_dims,
            "quantized all-to-sharded linear",
        )?;
        self.local.forward(input, stream)
    }

    /// Returns `(global output, global input)` dimensions.
    pub const fn global_shape(&self) -> (i32, i32) {
        (self.global_output_dims, self.global_input_dims)
    }

    /// Returns `(local output, local logical input)` dimensions.
    pub fn local_shape(&self) -> (i32, i32) {
        (
            self.local.inner.weight.value.shape()[0],
            self.global_input_dims,
        )
    }

    /// Returns the rank-local packed weight shape.
    pub fn local_packed_shape(&self) -> &[i32] {
        self.local.inner.weight.value.shape()
    }
}

/// Quantized row-parallel linear layer.
#[derive(Debug, Clone)]
pub struct QuantizedShardedToAllLinear {
    /// Rank-local packed quantized parameters.
    pub local: QuantizedLinear,
    global_input_dims: i32,
    global_output_dims: i32,
    shard_rank: usize,
    shard_size: usize,
}

impl QuantizedShardedToAllLinear {
    /// Creates rank-local unloaded packed parameters.
    #[allow(clippy::too_many_arguments)]
    pub fn unloaded(
        input_dims: i32,
        output_dims: i32,
        group_size: i32,
        bits: i32,
        mode: crate::ops::QuantizationMode,
        bias: bool,
        shard_rank: usize,
        shard_size: usize,
        stream: impl AsRef<Stream>,
    ) -> Result<Self, Exception> {
        validate_shard(shard_rank, shard_size)?;
        let local_input = checked_local_dimension("input dimension", input_dims, shard_size)?;
        validate_quantized_dimensions(local_input, group_size, bits).map_err(|_| {
            Exception::custom(format!(
                "input dimension {input_dims} with quantization group size {group_size} cannot be row-sharded across TP size {shard_size}"
            ))
        })?;
        Ok(Self {
            local: QuantizedLinear::unloaded_with_mode(
                local_input,
                output_dims,
                group_size,
                bits,
                mode,
                bias,
                stream,
            )?,
            global_input_dims: input_dims,
            global_output_dims: output_dims,
            shard_rank,
            shard_size,
        })
    }

    /// Shards a complete packed layer along aligned logical input groups.
    pub fn from_quantized(
        linear: &QuantizedLinear,
        shard_rank: usize,
        shard_size: usize,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let output_dims = linear.inner.weight.value.shape()[0];
        let input_dims = linear.scales.value.shape()[1] * linear.group_size;
        let local_input = checked_local_dimension("input dimension", input_dims, shard_size)?;
        validate_quantized_dimensions(local_input, linear.group_size, linear.bits).map_err(|_| {
            Exception::custom(format!(
                "input dimension {input_dims} with quantization group size {} cannot be row-sharded across TP size {shard_size}",
                linear.group_size
            ))
        })?;
        let local = QuantizedLinear {
            group_size: linear.group_size,
            bits: linear.bits,
            mode: linear.mode,
            native: None,
            native_format: None,
            native_endian: safemlx_gguf::Endian::Little,
            native_columns: 0,
            scales: Param::new(shard_axis(
                &linear.scales.value,
                1,
                shard_rank,
                shard_size,
                stream,
            )?),
            biases: Param::new(
                linear
                    .biases
                    .value
                    .as_ref()
                    .map(|value| shard_axis(value, 1, shard_rank, shard_size, stream))
                    .transpose()?,
            ),
            inner: Linear {
                weight: Param::new(shard_axis(
                    &linear.inner.weight.value,
                    1,
                    shard_rank,
                    shard_size,
                    stream,
                )?),
                bias: Param::new(linear.inner.bias.value.clone()),
            },
        };
        Ok(Self {
            local,
            global_input_dims: input_dims,
            global_output_dims: output_dims,
            shard_rank,
            shard_size,
        })
    }

    /// Computes a packed partial projection, all-sums it, and adds bias once.
    pub fn forward(
        &mut self,
        input: &Array,
        group: &Group,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        validate_group(self.shard_rank, self.shard_size, group)?;
        let local_input = self.local.scales.value.shape()[1] * self.local.group_size;
        validate_last_dimension(input, local_input, "quantized sharded-to-all linear")?;
        let partial = quantized_matmul_with_mode(
            input,
            &self.local.inner.weight,
            &self.local.scales,
            self.local.biases.value.as_ref(),
            true,
            self.local.group_size,
            self.local.bits,
            self.local.mode,
            stream,
        )?;
        let mut output = distributed::all_sum(&partial, group, stream)?;
        if let Some(bias) = self.local.inner.bias.value.as_ref() {
            output = output.add(bias, stream)?;
        }
        Ok(output)
    }

    /// Returns `(global output, global input)` dimensions.
    pub const fn global_shape(&self) -> (i32, i32) {
        (self.global_output_dims, self.global_input_dims)
    }

    /// Returns `(local output, local logical input)` dimensions.
    pub fn local_shape(&self) -> (i32, i32) {
        (
            self.global_output_dims,
            self.local.scales.value.shape()[1] * self.local.group_size,
        )
    }

    /// Returns the rank-local packed weight shape.
    pub fn local_packed_shape(&self) -> &[i32] {
        self.local.inner.weight.value.shape()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::QuantizationMode;

    #[test]
    fn unloaded_shapes_and_alignment_are_validated() {
        let stream = crate::test_stream();
        let column =
            AllToShardedLinear::unloaded(8, 12, true, Dtype::Float32, 1, 2, stream).unwrap();
        assert_eq!(column.global_shape(), (12, 8));
        assert_eq!(column.local_shape(), (6, 8));
        let row = ShardedToAllLinear::unloaded(8, 12, true, Dtype::Float32, 1, 2, stream).unwrap();
        assert_eq!(row.local_shape(), (12, 4));
        assert!(AllToShardedLinear::unloaded(8, 11, false, Dtype::Float32, 0, 2, stream).is_err());
    }

    #[test]
    fn dense_conversion_preserves_local_shapes_and_bias_placement() {
        let stream = crate::test_stream();
        let linear = Linear::unloaded(8, 12, true, Dtype::Float32, stream).unwrap();
        let column = AllToShardedLinear::from_linear(&linear, 1, 2, stream).unwrap();
        assert_eq!(column.local.weight.value.shape(), &[6, 8]);
        assert_eq!(column.local.bias.value.as_ref().unwrap().shape(), &[6]);
        let row = ShardedToAllLinear::from_linear(&linear, 1, 2, stream).unwrap();
        assert_eq!(row.local.weight.value.shape(), &[12, 4]);
        assert_eq!(row.local.bias.value.as_ref().unwrap().shape(), &[12]);
    }

    #[test]
    fn quantized_conversion_places_packed_companions() {
        let stream = crate::test_stream();
        let complete = QuantizedLinear::unloaded_with_mode(
            64,
            16,
            32,
            4,
            QuantizationMode::Affine,
            false,
            stream,
        )
        .unwrap();
        let column = QuantizedAllToShardedLinear::from_quantized(&complete, 0, 2, stream).unwrap();
        assert_eq!(column.local.inner.weight.value.shape(), &[8, 8]);
        assert_eq!(column.local.scales.value.shape(), &[8, 2]);
        assert_eq!(column.local.biases.value.as_ref().unwrap().shape(), &[8, 2]);

        let row = QuantizedShardedToAllLinear::from_quantized(&complete, 1, 2, stream).unwrap();
        assert_eq!(row.local.inner.weight.value.shape(), &[16, 4]);
        assert_eq!(row.local.scales.value.shape(), &[16, 1]);
        assert_eq!(row.local.biases.value.as_ref().unwrap().shape(), &[16, 1]);
    }

    #[test]
    fn quantized_row_shards_reject_split_groups() {
        let stream = crate::test_stream();
        let error = QuantizedShardedToAllLinear::unloaded(
            96,
            16,
            32,
            4,
            QuantizationMode::Affine,
            false,
            0,
            2,
            stream,
        )
        .unwrap_err();
        assert!(error.what().contains("cannot be row-sharded"));
    }
}
