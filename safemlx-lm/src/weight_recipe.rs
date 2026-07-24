//! Deterministic checkpoint-derived weight recipes.
//!
//! Recipes describe the runtime representation of a parameter without tying it
//! to a single checkpoint key. They are validated from checkpoint metadata and
//! materialized on the residency source stream before device promotion.

use std::collections::BTreeSet;

use safemlx::{
    ops::{concatenate_axis, stack_axis},
    transforms::eval,
    Array, Dtype, Stream,
};

use crate::weight_store::{
    PendingWeightMaterialization, StoredDtype, TensorSelection, WeightStore, WeightStoreError,
};

/// Scalar encoding produced by a derived-weight recipe.
#[derive(Debug, Clone, Eq, PartialEq)]
#[non_exhaustive]
pub enum RecipeDtype {
    /// Boolean values.
    Bool,
    /// Unsigned 8-bit integers.
    U8,
    /// Signed 8-bit integers.
    I8,
    /// Signed 16-bit integers.
    I16,
    /// Unsigned 16-bit integers.
    U16,
    /// IEEE half-precision floating point.
    F16,
    /// Brain floating point.
    BF16,
    /// Signed 32-bit integers.
    I32,
    /// Unsigned 32-bit integers.
    U32,
    /// IEEE single-precision floating point.
    F32,
    /// IEEE double-precision floating point.
    F64,
    /// Signed 64-bit integers.
    I64,
    /// Unsigned 64-bit integers.
    U64,
    /// Complex values with two 32-bit floating-point components.
    C64,
    /// Encoded FP8 E4M3 bytes.
    F8E4M3,
    /// Encoded FP8 E5M2 bytes.
    F8E5M2,
    /// A checkpoint encoding unknown to this runtime.
    Other(String),
}

impl RecipeDtype {
    fn byte_width(&self) -> Result<u64, WeightRecipeError> {
        match self {
            Self::Bool | Self::U8 | Self::I8 | Self::F8E4M3 | Self::F8E5M2 => Ok(1),
            Self::I16 | Self::U16 | Self::F16 | Self::BF16 => Ok(2),
            Self::I32 | Self::U32 | Self::F32 => Ok(4),
            Self::F64 | Self::I64 | Self::U64 | Self::C64 => Ok(8),
            Self::Other(dtype) => Err(WeightRecipeError::UnsupportedDtype {
                dtype: dtype.clone(),
            }),
        }
    }
}

impl From<StoredDtype> for RecipeDtype {
    fn from(value: StoredDtype) -> Self {
        match value {
            StoredDtype::Bool => Self::Bool,
            StoredDtype::U8 => Self::U8,
            StoredDtype::I8 => Self::I8,
            StoredDtype::I16 => Self::I16,
            StoredDtype::U16 => Self::U16,
            StoredDtype::F16 => Self::F16,
            StoredDtype::BF16 => Self::BF16,
            StoredDtype::I32 => Self::I32,
            StoredDtype::U32 => Self::U32,
            StoredDtype::F32 => Self::F32,
            StoredDtype::F64 => Self::F64,
            StoredDtype::I64 => Self::I64,
            StoredDtype::U64 => Self::U64,
            StoredDtype::C64 => Self::C64,
            StoredDtype::F8E4M3 => Self::F8E4M3,
            StoredDtype::F8E5M2 => Self::F8E5M2,
            StoredDtype::Other(dtype) => Self::Other(dtype),
        }
    }
}

impl From<Dtype> for RecipeDtype {
    fn from(value: Dtype) -> Self {
        match value {
            Dtype::Bool => Self::Bool,
            Dtype::Uint8 => Self::U8,
            Dtype::Uint16 => Self::U16,
            Dtype::Uint32 => Self::U32,
            Dtype::Uint64 => Self::U64,
            Dtype::Int8 => Self::I8,
            Dtype::Int16 => Self::I16,
            Dtype::Int32 => Self::I32,
            Dtype::Int64 => Self::I64,
            Dtype::Float16 => Self::F16,
            Dtype::Float32 => Self::F32,
            Dtype::Float64 => Self::F64,
            Dtype::Bfloat16 => Self::BF16,
            Dtype::Complex64 => Self::C64,
        }
    }
}

/// Shape, encoding, and materialized size inferred for a recipe.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct WeightRecipeMetadata {
    shape: Vec<usize>,
    dtype: RecipeDtype,
    byte_len: u64,
}

impl WeightRecipeMetadata {
    /// Returns the inferred output shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the inferred output scalar encoding.
    pub const fn dtype(&self) -> &RecipeDtype {
        &self.dtype
    }

    /// Returns the checked materialized output size.
    pub const fn byte_len(&self) -> u64 {
        self.byte_len
    }
}

/// A composable operation that derives one runtime weight from checkpoint tensors.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum DerivedWeightRecipe {
    /// Selects a complete tensor, contiguous range, or ordered indices.
    Source {
        /// Checkpoint tensor key. This also represents checkpoint-key renaming.
        key: String,
        /// Selection applied before any derived operations.
        selection: TensorSelection,
    },
    /// Applies another selection to a derived array.
    Select {
        /// Child recipe.
        input: Box<Self>,
        /// Selection applied to the child output.
        selection: TensorSelection,
    },
    /// Concatenates child outputs along an existing axis.
    Concatenate {
        /// Concatenation axis.
        axis: usize,
        /// Ordered child recipes.
        inputs: Vec<Self>,
    },
    /// Stacks child outputs along a new axis.
    Stack {
        /// Inserted axis.
        axis: usize,
        /// Ordered child recipes.
        inputs: Vec<Self>,
    },
    /// Reshapes a child while preserving its element count.
    Reshape {
        /// Child recipe.
        input: Box<Self>,
        /// Exact output shape.
        shape: Vec<usize>,
    },
    /// Reorders axes. Axis movement is represented by its resulting permutation.
    Transpose {
        /// Child recipe.
        input: Box<Self>,
        /// Output-axis to input-axis permutation.
        axes: Vec<usize>,
    },
    /// Casts a child to an MLX execution dtype.
    Cast {
        /// Child recipe.
        input: Box<Self>,
        /// Output execution dtype.
        dtype: Dtype,
    },
    /// Applies `log(-x)` elementwise.
    NegLog {
        /// Child recipe.
        input: Box<Self>,
    },
    /// Subtracts one elementwise for GGUF offset-normalization weights.
    SubtractOne {
        /// Child recipe.
        input: Box<Self>,
    },
}

impl DerivedWeightRecipe {
    /// Creates a direct checkpoint source recipe.
    pub fn source(key: impl Into<String>, selection: TensorSelection) -> Self {
        Self::Source {
            key: key.into(),
            selection,
        }
    }

    /// Returns every source checkpoint key in deterministic order.
    pub fn source_keys(&self) -> Vec<&str> {
        let mut keys = BTreeSet::new();
        self.collect_source_keys(&mut keys);
        keys.into_iter().collect()
    }

    fn collect_source_keys<'a>(&'a self, keys: &mut BTreeSet<&'a str>) {
        match self {
            Self::Source { key, .. } => {
                keys.insert(key);
            }
            Self::Concatenate { inputs, .. } | Self::Stack { inputs, .. } => {
                for input in inputs {
                    input.collect_source_keys(keys);
                }
            }
            Self::Select { input, .. }
            | Self::Reshape { input, .. }
            | Self::Transpose { input, .. }
            | Self::Cast { input, .. }
            | Self::NegLog { input }
            | Self::SubtractOne { input } => input.collect_source_keys(keys),
        }
    }

    /// Validates the complete recipe and infers its output metadata.
    pub fn infer(
        &self,
        store: &dyn WeightStore,
    ) -> Result<WeightRecipeMetadata, WeightRecipeError> {
        match self {
            Self::Source { key, selection } => infer_source(store, key, selection),
            Self::Select { input, selection } => {
                let metadata = input.infer(store)?;
                let shape = selected_shape(metadata.shape, selection)?;
                metadata_for(shape, metadata.dtype)
            }
            Self::Concatenate { axis, inputs } => infer_join(store, *axis, inputs, false),
            Self::Stack { axis, inputs } => infer_join(store, *axis, inputs, true),
            Self::Reshape { input, shape } => {
                let metadata = input.infer(store)?;
                let old_count = element_count(&metadata.shape, "reshape input")?;
                let new_count = element_count(shape, "reshape output")?;
                if old_count != new_count {
                    return Err(WeightRecipeError::ElementCountMismatch {
                        input: old_count,
                        output: new_count,
                    });
                }
                metadata_for(shape.clone(), metadata.dtype)
            }
            Self::Transpose { input, axes } => {
                let metadata = input.infer(store)?;
                if axes.len() != metadata.shape.len() {
                    return Err(WeightRecipeError::InvalidPermutation {
                        axes: axes.clone(),
                        rank: metadata.shape.len(),
                    });
                }
                let unique = axes.iter().copied().collect::<BTreeSet<_>>();
                if unique.len() != axes.len() || axes.iter().any(|axis| *axis >= axes.len()) {
                    return Err(WeightRecipeError::InvalidPermutation {
                        axes: axes.clone(),
                        rank: metadata.shape.len(),
                    });
                }
                let shape = axes.iter().map(|axis| metadata.shape[*axis]).collect();
                metadata_for(shape, metadata.dtype)
            }
            Self::Cast { input, dtype } => {
                let metadata = input.infer(store)?;
                metadata_for(metadata.shape, (*dtype).into())
            }
            Self::NegLog { input } => input.infer(store),
            Self::SubtractOne { input } => input.infer(store),
        }
    }

    /// Materializes this recipe on the host source stream.
    ///
    /// Source leases remain live until their dependent output has been
    /// evaluated. If a multi-input join reaches the mapping bound, completed
    /// children are detached before retrying so cross-shard recipes can honor
    /// a one-mapping limit without serializing the normal batched path.
    pub fn materialize(
        &self,
        store: &dyn WeightStore,
        source_stream: &Stream,
    ) -> Result<Array, WeightRecipeError> {
        self.prepare_materialization(store, source_stream)?.finish()
    }

    /// Schedules a recipe while retaining all mmap-backed source selections.
    pub(crate) fn prepare_materialization(
        &self,
        store: &dyn WeightStore,
        source_stream: &Stream,
    ) -> Result<PendingWeightRecipe, WeightRecipeError> {
        self.infer(store)?;
        let mut sources = Vec::new();
        let output = self.materialize_inner(store, source_stream, &mut sources)?;
        Ok(PendingWeightRecipe { output, sources })
    }

    fn materialize_inner(
        &self,
        store: &dyn WeightStore,
        stream: &Stream,
        sources: &mut Vec<PendingWeightMaterialization>,
    ) -> Result<Array, WeightRecipeError> {
        match self {
            Self::Source { key, selection } => {
                let lease = store.acquire(key, selection.clone())?;
                let pending = lease.prepare_materialization(stream, stream)?;
                let array = pending.output().clone();
                sources.push(pending);
                Ok(array)
            }
            Self::Select { input, selection } => {
                let array = input.materialize_inner(store, stream, sources)?;
                match selection {
                    TensorSelection::Full => Ok(array),
                    TensorSelection::Range { axis, start, end } => {
                        let indices = (*start..*end)
                            .map(|index| usize_to_i32(index, "selection index"))
                            .collect::<Result<Vec<_>, _>>()?;
                        Ok(array.take_axis(
                            Array::from_slice(&indices, &[indices.len() as i32]),
                            usize_to_i32(*axis, "selection axis")?,
                            stream,
                        )?)
                    }
                    TensorSelection::Indices { axis, indices } => {
                        let indices = indices
                            .iter()
                            .map(|index| usize_to_i32(*index, "selection index"))
                            .collect::<Result<Vec<_>, _>>()?;
                        Ok(array.take_axis(
                            Array::from_slice(&indices, &[indices.len() as i32]),
                            usize_to_i32(*axis, "selection axis")?,
                            stream,
                        )?)
                    }
                }
            }
            Self::Concatenate { axis, inputs } => {
                let arrays = materialize_inputs(inputs, store, stream, sources)?;
                let references = arrays.iter().collect::<Vec<_>>();
                Ok(concatenate_axis(
                    &references,
                    usize_to_i32(*axis, "concatenate axis")?,
                    stream,
                )?)
            }
            Self::Stack { axis, inputs } => {
                let arrays = materialize_inputs(inputs, store, stream, sources)?;
                let references = arrays.iter().collect::<Vec<_>>();
                Ok(stack_axis(
                    &references,
                    usize_to_i32(*axis, "stack axis")?,
                    stream,
                )?)
            }
            Self::Reshape { input, shape } => {
                let array = input.materialize_inner(store, stream, sources)?;
                let shape = shape
                    .iter()
                    .map(|dimension| usize_to_i32(*dimension, "reshape dimension"))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(array.reshape(&shape, stream)?)
            }
            Self::Transpose { input, axes } => {
                let array = input.materialize_inner(store, stream, sources)?;
                let axes = axes
                    .iter()
                    .map(|axis| usize_to_i32(*axis, "transpose axis"))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(array.transpose_axes(&axes, stream)?)
            }
            Self::Cast { input, dtype } => {
                let array = input.materialize_inner(store, stream, sources)?;
                Ok(array.as_dtype(*dtype, stream)?)
            }
            Self::NegLog { input } => {
                let array = input.materialize_inner(store, stream, sources)?;
                Ok(array.multiply(Array::from_f32(-1.0), stream)?.log(stream)?)
            }
            Self::SubtractOne { input } => {
                let array = input.materialize_inner(store, stream, sources)?;
                Ok(array.subtract(Array::from_f32(1.0), stream)?)
            }
        }
    }
}

/// Lazy recipe output and the source mappings required to evaluate it safely.
pub(crate) struct PendingWeightRecipe {
    output: Array,
    sources: Vec<PendingWeightMaterialization>,
}

impl PendingWeightRecipe {
    pub(crate) fn into_parts(self) -> (Array, Vec<PendingWeightMaterialization>) {
        (self.output, self.sources)
    }

    fn finish(self) -> Result<Array, WeightRecipeError> {
        eval([&self.output])?;
        for source in self.sources {
            source.complete();
        }
        Ok(self.output)
    }
}

fn materialize_inputs(
    inputs: &[DerivedWeightRecipe],
    store: &dyn WeightStore,
    stream: &Stream,
    sources: &mut Vec<PendingWeightMaterialization>,
) -> Result<Vec<Array>, WeightRecipeError> {
    let mut pending =
        Vec::<(Array, Vec<PendingWeightMaterialization>)>::with_capacity(inputs.len());
    let mut detach_remaining = false;
    for input in inputs {
        loop {
            let mut input_sources = Vec::new();
            match input.materialize_inner(store, stream, &mut input_sources) {
                Ok(array) => {
                    if detach_remaining && !input_sources.is_empty() {
                        eval([&array])?;
                        for source in input_sources.drain(..) {
                            source.complete();
                        }
                    }
                    pending.push((array, input_sources));
                    break;
                }
                Err(error)
                    if !detach_remaining
                        && !pending.is_empty()
                        && matches!(
                            &error,
                            WeightRecipeError::WeightStore(
                                WeightStoreError::CapacityExhausted { .. }
                            )
                        ) =>
                {
                    // The current child could not acquire another shard while
                    // earlier children pinned the mapping cache. Their arrays
                    // are sufficient evaluation roots, so detach them and retry.
                    drop(input_sources);
                    for (array, child_sources) in &mut pending {
                        if child_sources.is_empty() {
                            continue;
                        }
                        eval([&*array])?;
                        for source in child_sources.drain(..) {
                            source.complete();
                        }
                    }
                    detach_remaining = true;
                }
                Err(error) => return Err(error),
            }
        }
    }
    let mut arrays = Vec::with_capacity(pending.len());
    for (array, input_sources) in pending {
        arrays.push(array);
        sources.extend(input_sources);
    }
    Ok(arrays)
}

fn infer_source(
    store: &dyn WeightStore,
    key: &str,
    selection: &TensorSelection,
) -> Result<WeightRecipeMetadata, WeightRecipeError> {
    if key.trim().is_empty() {
        return Err(WeightRecipeError::EmptySourceKey);
    }
    let metadata = store.metadata(key)?;
    let shape = selected_shape(metadata.shape, selection)?;
    metadata_for(shape, metadata.stored_dtype.into())
}

fn selected_shape(
    mut shape: Vec<usize>,
    selection: &TensorSelection,
) -> Result<Vec<usize>, WeightRecipeError> {
    match selection {
        TensorSelection::Full => {}
        TensorSelection::Range { axis, start, end } => {
            let rank = shape.len();
            let dimension = shape
                .get_mut(*axis)
                .ok_or(WeightRecipeError::InvalidSelectionAxis { axis: *axis, rank })?;
            if start >= end || *end > *dimension {
                return Err(WeightRecipeError::InvalidRange {
                    axis: *axis,
                    start: *start,
                    end: *end,
                    dimension: *dimension,
                });
            }
            *dimension = end - start;
        }
        TensorSelection::Indices { axis, indices } => {
            let rank = shape.len();
            let dimension = shape
                .get_mut(*axis)
                .ok_or(WeightRecipeError::InvalidSelectionAxis { axis: *axis, rank })?;
            if indices.is_empty() || indices.iter().any(|index| *index >= *dimension) {
                return Err(WeightRecipeError::InvalidIndices {
                    axis: *axis,
                    dimension: *dimension,
                });
            }
            *dimension = indices.len();
        }
    }
    Ok(shape)
}

fn infer_join(
    store: &dyn WeightStore,
    axis: usize,
    inputs: &[DerivedWeightRecipe],
    stack: bool,
) -> Result<WeightRecipeMetadata, WeightRecipeError> {
    if inputs.is_empty() {
        return Err(WeightRecipeError::EmptyInputs);
    }
    let metadata = inputs
        .iter()
        .map(|input| input.infer(store))
        .collect::<Result<Vec<_>, _>>()?;
    let first = &metadata[0];
    if metadata.iter().any(|item| item.dtype != first.dtype) {
        return Err(WeightRecipeError::DtypeMismatch);
    }
    let rank = first.shape.len();
    if axis > rank || (!stack && axis == rank) {
        return Err(WeightRecipeError::InvalidJoinAxis { axis, rank, stack });
    }
    if stack {
        if metadata.iter().any(|item| item.shape != first.shape) {
            return Err(WeightRecipeError::ShapeMismatch);
        }
        let mut shape = first.shape.clone();
        shape.insert(axis, metadata.len());
        metadata_for(shape, first.dtype.clone())
    } else {
        let mut shape = first.shape.clone();
        shape[axis] = 0;
        for item in &metadata {
            if item.shape.len() != rank
                || item
                    .shape
                    .iter()
                    .enumerate()
                    .any(|(index, dimension)| index != axis && *dimension != first.shape[index])
            {
                return Err(WeightRecipeError::ShapeMismatch);
            }
            shape[axis] = shape[axis].checked_add(item.shape[axis]).ok_or(
                WeightRecipeError::ArithmeticOverflow("concatenate dimension"),
            )?;
        }
        metadata_for(shape, first.dtype.clone())
    }
}

fn metadata_for(
    shape: Vec<usize>,
    dtype: RecipeDtype,
) -> Result<WeightRecipeMetadata, WeightRecipeError> {
    let elements = element_count(&shape, "recipe output")?;
    let byte_len = elements
        .checked_mul(dtype.byte_width()?)
        .ok_or(WeightRecipeError::ArithmeticOverflow("recipe output bytes"))?;
    if byte_len == 0 {
        return Err(WeightRecipeError::ZeroSizedOutput);
    }
    Ok(WeightRecipeMetadata {
        shape,
        dtype,
        byte_len,
    })
}

fn element_count(shape: &[usize], context: &'static str) -> Result<u64, WeightRecipeError> {
    shape.iter().try_fold(1u64, |count, dimension| {
        let dimension = u64::try_from(*dimension)
            .map_err(|_| WeightRecipeError::ArithmeticOverflow(context))?;
        count
            .checked_mul(dimension)
            .ok_or(WeightRecipeError::ArithmeticOverflow(context))
    })
}

fn usize_to_i32(value: usize, context: &'static str) -> Result<i32, WeightRecipeError> {
    i32::try_from(value).map_err(|_| WeightRecipeError::ArithmeticOverflow(context))
}

/// Structured validation and materialization failures for derived weights.
#[derive(Debug, thiserror::Error)]
pub enum WeightRecipeError {
    /// A source key was empty.
    #[error("derived-weight source key must not be empty")]
    EmptySourceKey,
    /// A source selection axis was outside the tensor rank.
    #[error("selection axis {axis} is outside rank {rank}")]
    InvalidSelectionAxis {
        /// Requested axis.
        axis: usize,
        /// Source rank.
        rank: usize,
    },
    /// A source range was invalid.
    #[error("range {start}..{end} is invalid for axis {axis} dimension {dimension}")]
    InvalidRange {
        /// Requested axis.
        axis: usize,
        /// Inclusive start.
        start: usize,
        /// Exclusive end.
        end: usize,
        /// Source dimension.
        dimension: usize,
    },
    /// An ordered-index selection was empty or out of bounds.
    #[error("ordered indices are empty or outside axis {axis} dimension {dimension}")]
    InvalidIndices {
        /// Requested axis.
        axis: usize,
        /// Source dimension.
        dimension: usize,
    },
    /// Concatenate or stack had no children.
    #[error("concatenate and stack recipes require at least one input")]
    EmptyInputs,
    /// Child dtypes did not agree.
    #[error("derived-weight inputs have different dtypes")]
    DtypeMismatch,
    /// Child shapes were incompatible.
    #[error("derived-weight inputs have incompatible shapes")]
    ShapeMismatch,
    /// A concatenate or stack axis was outside the accepted range.
    #[error("axis {axis} is invalid for rank {rank} (stack={stack})")]
    InvalidJoinAxis {
        /// Requested axis.
        axis: usize,
        /// Child rank.
        rank: usize,
        /// Whether the operation was stack instead of concatenate.
        stack: bool,
    },
    /// A reshape changed the element count.
    #[error("reshape changes element count from {input} to {output}")]
    ElementCountMismatch {
        /// Input element count.
        input: u64,
        /// Requested output element count.
        output: u64,
    },
    /// A transpose was not a rank-sized permutation.
    #[error("axes {axes:?} are not a permutation of rank {rank}")]
    InvalidPermutation {
        /// Requested axis order.
        axes: Vec<usize>,
        /// Child rank.
        rank: usize,
    },
    /// The inferred output contains no bytes.
    #[error("derived-weight output must contain at least one byte")]
    ZeroSizedOutput,
    /// A stored encoding has no known runtime byte width.
    #[error("derived-weight dtype {dtype} is unsupported")]
    UnsupportedDtype {
        /// Debug name of the unsupported encoding.
        dtype: String,
    },
    /// Checked shape or byte arithmetic overflowed.
    #[error("derived-weight arithmetic overflow: {0}")]
    ArithmeticOverflow(&'static str),
    /// Checkpoint storage failed.
    #[error(transparent)]
    WeightStore(#[from] crate::weight_store::WeightStoreError),
    /// MLX transformation or synchronization failed.
    #[error(transparent)]
    Mlx(#[from] safemlx::error::Exception),
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use safemlx::{Device, DeviceType};
    use safetensors::tensor::{serialize_to_file, Dtype as SafeDtype, TensorView};

    use super::*;
    use crate::weight_store::SafetensorsWeightStore;

    fn fixture() -> (tempfile::TempDir, Arc<SafetensorsWeightStore>) {
        let dir = tempfile::tempdir().unwrap();
        let left = [1i32, 2, 3, 4]
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>();
        let right = [5i32, 6, 7, 8]
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>();
        serialize_to_file(
            [
                (
                    "left",
                    TensorView::new(SafeDtype::I32, vec![2, 2], &left).unwrap(),
                ),
                (
                    "right",
                    TensorView::new(SafeDtype::I32, vec![2, 2], &right).unwrap(),
                ),
            ],
            None,
            &dir.path().join("model.safetensors"),
        )
        .unwrap();
        let store = Arc::new(SafetensorsWeightStore::open(dir.path()).unwrap());
        (dir, store)
    }

    fn one_mapping_cross_shard_fixture() -> (tempfile::TempDir, Arc<SafetensorsWeightStore>) {
        let dir = tempfile::tempdir().unwrap();
        let left = [1i32, 2, 3, 4]
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>();
        let right = [5i32, 6, 7, 8]
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>();
        serialize_to_file(
            [(
                "left",
                TensorView::new(SafeDtype::I32, vec![2, 2], &left).unwrap(),
            )],
            None,
            &dir.path().join("model-00001-of-00002.safetensors"),
        )
        .unwrap();
        serialize_to_file(
            [(
                "right",
                TensorView::new(SafeDtype::I32, vec![2, 2], &right).unwrap(),
            )],
            None,
            &dir.path().join("model-00002-of-00002.safetensors"),
        )
        .unwrap();
        std::fs::write(
            dir.path().join("model.safetensors.index.json"),
            serde_json::to_vec(&serde_json::json!({
                "weight_map": {
                    "left": "model-00001-of-00002.safetensors",
                    "right": "model-00002-of-00002.safetensors"
                }
            }))
            .unwrap(),
        )
        .unwrap();
        let store =
            Arc::new(SafetensorsWeightStore::open_with_max_mapped_shards(dir.path(), 1).unwrap());
        (dir, store)
    }

    fn source(key: &str) -> DerivedWeightRecipe {
        DerivedWeightRecipe::source(key, TensorSelection::Full)
    }

    #[test]
    fn infers_nested_stack_concatenate_slice_and_cast() {
        let (_dir, store) = fixture();
        let recipe = DerivedWeightRecipe::Cast {
            input: Box::new(DerivedWeightRecipe::Stack {
                axis: 0,
                inputs: vec![
                    DerivedWeightRecipe::Concatenate {
                        axis: 0,
                        inputs: vec![source("left"), source("right")],
                    },
                    DerivedWeightRecipe::Concatenate {
                        axis: 0,
                        inputs: vec![
                            DerivedWeightRecipe::source(
                                "left",
                                TensorSelection::Indices {
                                    axis: 0,
                                    indices: vec![1, 0],
                                },
                            ),
                            source("right"),
                        ],
                    },
                ],
            }),
            dtype: Dtype::Float32,
        };
        let metadata = recipe.infer(store.as_ref()).unwrap();
        assert_eq!(metadata.shape(), &[2, 4, 2]);
        assert_eq!(metadata.dtype(), &RecipeDtype::F32);
        assert_eq!(metadata.byte_len(), 64);
        assert_eq!(recipe.source_keys(), vec!["left", "right"]);
    }

    #[test]
    fn materializes_ordered_expert_stack_on_cpu() {
        let (_dir, store) = fixture();
        let recipe = DerivedWeightRecipe::Stack {
            axis: 0,
            inputs: vec![source("right"), source("left")],
        };
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let output = recipe.materialize(store.as_ref(), &stream).unwrap();
        assert_eq!(output.shape(), &[2, 2, 2]);
        assert_eq!(output.nbytes(), 32);
        assert_eq!(
            output.evaluated().unwrap().as_slice::<i32>(),
            &[5, 6, 7, 8, 1, 2, 3, 4]
        );
    }

    #[test]
    fn materializes_cross_shard_join_with_one_mapping() {
        let (_dir, store) = one_mapping_cross_shard_fixture();
        let recipe = DerivedWeightRecipe::Stack {
            axis: 0,
            inputs: vec![DerivedWeightRecipe::Concatenate {
                axis: 0,
                inputs: vec![source("left"), source("right")],
            }],
        };
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let output = recipe.materialize(store.as_ref(), &stream).unwrap();
        assert_eq!(output.shape(), &[1, 4, 2]);
        assert_eq!(
            output.evaluated().unwrap().as_slice::<i32>(),
            &[1, 2, 3, 4, 5, 6, 7, 8]
        );
        let diagnostics = store.diagnostics().unwrap();
        assert_eq!(diagnostics.currently_mapped_shards, 1);
        assert_eq!(diagnostics.touched_shard_paths.len(), 2);
        assert!(diagnostics.evictions >= 1);
    }

    #[test]
    fn selects_an_axis_from_a_derived_source_selection() {
        let (_dir, store) = fixture();
        let recipe = DerivedWeightRecipe::Select {
            input: Box::new(DerivedWeightRecipe::source(
                "left",
                TensorSelection::Range {
                    axis: 0,
                    start: 0,
                    end: 1,
                },
            )),
            selection: TensorSelection::Indices {
                axis: 1,
                indices: vec![1],
            },
        };
        let metadata = recipe.infer(store.as_ref()).unwrap();
        assert_eq!(metadata.shape(), &[1, 1]);
        assert_eq!(recipe.source_keys(), vec!["left"]);
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let output = recipe.materialize(store.as_ref(), &stream).unwrap();
        assert_eq!(output.evaluated().unwrap().as_slice::<i32>(), &[2]);
    }

    #[test]
    fn rejects_shape_and_permutation_errors_before_materialization() {
        let (_dir, store) = fixture();
        let reshape = DerivedWeightRecipe::Reshape {
            input: Box::new(source("left")),
            shape: vec![3],
        };
        assert!(matches!(
            reshape.infer(store.as_ref()),
            Err(WeightRecipeError::ElementCountMismatch { .. })
        ));
        let transpose = DerivedWeightRecipe::Transpose {
            input: Box::new(source("left")),
            axes: vec![0, 0],
        };
        assert!(matches!(
            transpose.infer(store.as_ref()),
            Err(WeightRecipeError::InvalidPermutation { .. })
        ));
    }
}
