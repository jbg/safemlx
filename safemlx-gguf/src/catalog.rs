use crate::convert::{affine_shapes, conversion_kind, iquant_packed_shape, ConversionKind};
use crate::{
    ConvertedTensor, DenseDtype, Endian, Error, Limits, MetadataValue, OuterSelection, Reader,
    Result, TensorDescriptor,
};
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

const SPLIT_NO: &str = "split.no";
const SPLIT_COUNT: &str = "split.count";
const SPLIT_TENSORS_COUNT: &str = "split.tensors.count";

/// Scalar encoding of one logical tensor produced by GGUF conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogicalDtype {
    F32,
    F16,
    Bf16,
    I8,
    I16,
    U8,
    U32,
    I32,
    I64,
    F64,
}

impl From<DenseDtype> for LogicalDtype {
    fn from(value: DenseDtype) -> Self {
        match value {
            DenseDtype::F32 => Self::F32,
            DenseDtype::F16 => Self::F16,
            DenseDtype::Bf16 => Self::Bf16,
            DenseDtype::I8 => Self::I8,
            DenseDtype::I16 => Self::I16,
            DenseDtype::I32 => Self::I32,
            DenseDtype::I64 => Self::I64,
            DenseDtype::F64 => Self::F64,
        }
    }
}

/// Name, shape, and dtype of one converted tensor without its payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogicalTensorLayout {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: LogicalDtype,
}

/// One physical GGUF tensor and the logical tensors its conversion produces.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CatalogTensor {
    descriptor: TensorDescriptor,
    outputs: Vec<LogicalTensorLayout>,
    affine: Option<(u8, u32)>,
}

impl CatalogTensor {
    pub fn descriptor(&self) -> &TensorDescriptor {
        &self.descriptor
    }

    pub fn outputs(&self) -> &[LogicalTensorLayout] {
        &self.outputs
    }

    /// Packed affine `(bits, group_size)`, or `None` for dense and IQ outputs.
    pub fn affine(&self) -> Option<(u8, u32)> {
        self.affine
    }
}

/// Header-only description of one GGUF payload shard.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CatalogShard {
    path: PathBuf,
    split_no: usize,
    version: u32,
    endian: Endian,
    alignment: u64,
    tensors: Vec<CatalogTensor>,
}

impl CatalogShard {
    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn split_no(&self) -> usize {
        self.split_no
    }

    pub fn version(&self) -> u32 {
        self.version
    }

    pub fn endian(&self) -> Endian {
        self.endian
    }

    pub fn alignment(&self) -> u64 {
        self.alignment
    }

    pub fn tensors(&self) -> &[CatalogTensor] {
        &self.tensors
    }
}

/// A validated, streaming handle for a single-file or sharded GGUF checkpoint.
///
/// Opening a checkpoint parses every shard header and tensor descriptor but does
/// not read or convert tensor payloads. Payloads are materialized one physical
/// tensor at a time through [`Self::converted_tensors`] or
/// [`Self::for_each_converted_tensor`].
#[derive(Debug, Clone)]
pub struct Checkpoint {
    metadata: BTreeMap<String, MetadataValue>,
    shards: Vec<CatalogShard>,
    physical_tensor_count: usize,
    limits: Limits,
}

/// One materialized physical GGUF tensor and its converted logical output group.
#[derive(Debug, Clone, PartialEq)]
pub struct ConvertedCheckpointTensor {
    shard_index: usize,
    tensor_index: usize,
    descriptor: TensorDescriptor,
    converted: ConvertedTensor,
}

/// One physical GGUF tensor retained in its checkpoint-native byte encoding.
///
/// This is the ownership seam used by device backends which can consume GGUF
/// blocks directly. The byte vector is intentionally not interpreted or
/// repacked by this crate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawCheckpointTensor {
    shard_index: usize,
    tensor_index: usize,
    endian: Endian,
    descriptor: TensorDescriptor,
    data: Vec<u8>,
}

impl RawCheckpointTensor {
    /// Zero-based index of the shard containing this tensor.
    pub fn shard_index(&self) -> usize {
        self.shard_index
    }

    /// Zero-based tensor index within the containing shard.
    pub fn tensor_index(&self) -> usize {
        self.tensor_index
    }

    /// Endianness declared by the containing GGUF shard.
    pub fn endian(&self) -> Endian {
        self.endian
    }

    /// Physical GGUF tensor descriptor.
    pub fn descriptor(&self) -> &TensorDescriptor {
        &self.descriptor
    }

    /// Checkpoint-native tensor payload.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Consume this tensor and return its checkpoint-native payload.
    pub fn into_data(self) -> Vec<u8> {
        self.data
    }
}

impl ConvertedCheckpointTensor {
    /// Zero-based index of the shard containing this tensor.
    pub fn shard_index(&self) -> usize {
        self.shard_index
    }

    /// Zero-based tensor index within the containing shard.
    pub fn tensor_index(&self) -> usize {
        self.tensor_index
    }

    /// Physical GGUF tensor descriptor.
    pub fn descriptor(&self) -> &TensorDescriptor {
        &self.descriptor
    }

    /// Converted dense tensor or atomic affine tensor group.
    pub fn converted(&self) -> &ConvertedTensor {
        &self.converted
    }

    /// Consume the item and return its converted tensor group.
    pub fn into_converted(self) -> ConvertedTensor {
        self.converted
    }
}

/// Fallible iterator that materializes one physical tensor at a time.
pub struct ConvertedTensorIter<'a> {
    checkpoint: &'a Checkpoint,
    shard_index: usize,
    tensor_index: usize,
    reader: Option<Reader<BufReader<File>>>,
    finished: bool,
}

#[derive(Debug, Clone, Copy)]
struct TensorLocation {
    shard_index: usize,
    tensor_index: usize,
}

/// Indexed materializer that reuses the currently open GGUF shard reader.
///
/// Name lookup is constant-time after construction. Consecutive requests from
/// the same shard reuse one parsed reader; switching shards closes the previous
/// reader before opening the next, so file-descriptor use remains bounded.
pub struct TensorMaterializer {
    checkpoint: Checkpoint,
    locations: HashMap<String, TensorLocation>,
    reader: Option<(usize, Reader<BufReader<File>>)>,
}

impl std::fmt::Debug for TensorMaterializer {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TensorMaterializer")
            .field("tensor_count", &self.locations.len())
            .field(
                "open_shard_index",
                &self.reader.as_ref().map(|(index, _)| index),
            )
            .finish_non_exhaustive()
    }
}

impl std::fmt::Debug for ConvertedTensorIter<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ConvertedTensorIter")
            .field("shard_index", &self.shard_index)
            .field("tensor_index", &self.tensor_index)
            .field("finished", &self.finished)
            .finish_non_exhaustive()
    }
}

/// A logical layout paired with the pre-translation name that produced it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranslatedTensorLayout {
    pub physical_name: String,
    pub original_name: String,
    pub layout: LogicalTensorLayout,
}

impl Checkpoint {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with_limits(path, Limits::default())
    }

    pub fn open_with_limits(path: impl AsRef<Path>, limits: Limits) -> Result<Self> {
        let path = path.as_ref();
        validate_extension(path)?;
        let first = open_reader(path, limits.clone())?;
        let metadata = first.metadata().clone();
        let split_count = split_value(&metadata, SPLIT_COUNT)?.unwrap_or(0);
        if split_count <= 1 {
            let tensors = catalog_tensors(first.tensors())?;
            if let Some(split_no) = split_value(&metadata, SPLIT_NO)? {
                if split_no != 0 {
                    return Err(shard_error(format!(
                        "single-shard GGUF {:?} has {SPLIT_NO}={split_no}, expected 0",
                        path.display()
                    )));
                }
            }
            if let Some(expected_tensors) = split_value(&metadata, SPLIT_TENSORS_COUNT)? {
                if expected_tensors != tensors.len() {
                    return Err(shard_error(format!(
                        "single-shard GGUF declares {expected_tensors} tensors in {SPLIT_TENSORS_COUNT}, but {} were cataloged",
                        tensors.len()
                    )));
                }
            }
            validate_logical_names(tensors.iter())?;
            return Ok(Self {
                physical_tensor_count: tensors.len(),
                metadata,
                shards: vec![CatalogShard {
                    path: path.to_path_buf(),
                    split_no: 0,
                    version: first.version(),
                    endian: first.endian(),
                    alignment: first.alignment(),
                    tensors,
                }],
                limits,
            });
        }

        let first_split_no = required_split_value(&metadata, SPLIT_NO, path)?;
        if first_split_no != 0 {
            return Err(shard_error(format!(
                "sharded GGUF must be loaded from its first shard, but {:?} has {SPLIT_NO}={first_split_no}",
                path.display()
            )));
        }
        let expected_tensors = required_split_value(&metadata, SPLIT_TENSORS_COUNT, path)?;
        let paths = shard_paths(path, split_count)?;
        let mut shards = Vec::with_capacity(split_count);
        let mut names = HashMap::<String, PathBuf>::new();
        let first_tensors = catalog_tensors(first.tensors())?;
        for tensor in &first_tensors {
            names.insert(tensor.descriptor.name.clone(), path.to_path_buf());
        }
        let mut physical_tensor_count = first_tensors.len();
        shards.push(CatalogShard {
            path: path.to_path_buf(),
            split_no: 0,
            version: first.version(),
            endian: first.endian(),
            alignment: first.alignment(),
            tensors: first_tensors,
        });

        for (split_no, shard_path) in paths.into_iter().enumerate().skip(1) {
            if !shard_path.is_file() {
                return Err(shard_error(format!(
                    "missing GGUF shard {:?}",
                    shard_path.display()
                )));
            }
            let reader = open_reader(&shard_path, limits.clone())?;
            let shard_metadata = reader.metadata();
            let actual_split_no = required_split_value(shard_metadata, SPLIT_NO, &shard_path)?;
            if actual_split_no != split_no {
                return Err(shard_error(format!(
                    "GGUF shard {:?} has {SPLIT_NO}={actual_split_no}, expected {split_no}",
                    shard_path.display()
                )));
            }
            let actual_count = required_split_value(shard_metadata, SPLIT_COUNT, &shard_path)?;
            if actual_count != split_count {
                return Err(shard_error(format!(
                    "GGUF shard {:?} has {SPLIT_COUNT}={actual_count}, expected {split_count}",
                    shard_path.display()
                )));
            }
            let actual_tensors =
                required_split_value(shard_metadata, SPLIT_TENSORS_COUNT, &shard_path)?;
            if actual_tensors != expected_tensors {
                return Err(shard_error(format!(
                    "GGUF shard {:?} has {SPLIT_TENSORS_COUNT}={actual_tensors}, expected {expected_tensors}",
                    shard_path.display()
                )));
            }

            let tensors = catalog_tensors(reader.tensors())?;
            for tensor in &tensors {
                let source = tensor.descriptor.name.clone();
                if let Some(previous) = names.insert(source.clone(), shard_path.clone()) {
                    return Err(shard_error(format!(
                        "tensor {source:?} is duplicated across GGUF shards {:?} and {:?}",
                        previous.display(),
                        shard_path.display()
                    )));
                }
            }
            physical_tensor_count = physical_tensor_count
                .checked_add(tensors.len())
                .ok_or(Error::Overflow("sharded tensor count"))?;
            shards.push(CatalogShard {
                path: shard_path,
                split_no,
                version: reader.version(),
                endian: reader.endian(),
                alignment: reader.alignment(),
                tensors,
            });
        }

        if physical_tensor_count != expected_tensors {
            return Err(shard_error(format!(
                "sharded GGUF declares {expected_tensors} tensors in {SPLIT_TENSORS_COUNT}, but {physical_tensor_count} were cataloged"
            )));
        }
        validate_logical_names(shards.iter().flat_map(|shard| shard.tensors.iter()))?;
        Ok(Self {
            metadata,
            shards,
            physical_tensor_count,
            limits,
        })
    }

    pub fn metadata(&self) -> &BTreeMap<String, MetadataValue> {
        &self.metadata
    }

    pub fn shards(&self) -> &[CatalogShard] {
        &self.shards
    }

    pub fn physical_tensor_count(&self) -> usize {
        self.physical_tensor_count
    }

    pub fn tensors(&self) -> impl Iterator<Item = &CatalogTensor> {
        self.shards.iter().flat_map(|shard| shard.tensors.iter())
    }

    pub fn logical_outputs(&self) -> impl Iterator<Item = &LogicalTensorLayout> {
        self.tensors().flat_map(|tensor| tensor.outputs.iter())
    }

    /// Iterate over converted tensor groups without retaining earlier payloads.
    pub fn converted_tensors(&self) -> ConvertedTensorIter<'_> {
        ConvertedTensorIter {
            checkpoint: self,
            shard_index: 0,
            tensor_index: 0,
            reader: None,
            finished: false,
        }
    }

    /// Create an indexed named-tensor materializer with bounded reader reuse.
    pub fn materializer(&self) -> TensorMaterializer {
        let locations = self
            .shards
            .iter()
            .enumerate()
            .flat_map(|(shard_index, shard)| {
                shard
                    .tensors
                    .iter()
                    .enumerate()
                    .map(move |(tensor_index, tensor)| {
                        (
                            tensor.descriptor.name.clone(),
                            TensorLocation {
                                shard_index,
                                tensor_index,
                            },
                        )
                    })
            })
            .collect();
        TensorMaterializer {
            checkpoint: self.clone(),
            locations,
            reader: None,
        }
    }

    /// Materialize and visit one physical GGUF tensor at a time.
    ///
    /// Dense tensors are delivered as one dense output. Packed affine tensors
    /// are delivered as one atomic weight/scales/biases group.
    pub fn for_each_converted_tensor<F>(&self, mut visitor: F) -> Result<()>
    where
        F: FnMut(ConvertedCheckpointTensor) -> Result<()>,
    {
        for tensor in self.converted_tensors() {
            visitor(tensor?)?;
        }
        Ok(())
    }

    /// Translate every logical tensor name and reject collisions before payload reads.
    pub fn translated_outputs<F>(&self, mut translate: F) -> Result<Vec<TranslatedTensorLayout>>
    where
        F: FnMut(&str) -> String,
    {
        let mut owners = HashMap::<String, String>::new();
        let mut translated = Vec::new();
        for tensor in self.tensors() {
            for output in tensor.outputs() {
                let name = translate(&output.name);
                if let Some(first_source) = owners.insert(name.clone(), output.name.clone()) {
                    return Err(Error::TranslatedTensorCollision {
                        name,
                        first_source,
                        second_source: output.name.clone(),
                    });
                }
                translated.push(TranslatedTensorLayout {
                    physical_name: tensor.descriptor.name.clone(),
                    original_name: output.name.clone(),
                    layout: LogicalTensorLayout {
                        name,
                        shape: output.shape.clone(),
                        dtype: output.dtype,
                    },
                });
            }
        }
        Ok(translated)
    }
}

impl TensorMaterializer {
    /// Path of the shard containing `name`, without opening its payload reader.
    pub fn shard_path_for_tensor(&self, name: &str) -> Result<&Path> {
        let location = self
            .locations
            .get(name)
            .copied()
            .ok_or_else(|| Error::InvalidTensor {
                tensor: name.to_string(),
                reason: "tensor is not present in the checkpoint".into(),
            })?;
        Ok(&self.checkpoint.shards[location.shard_index].path)
    }

    /// Path of the currently cached shard reader, if any.
    pub fn open_shard_path(&self) -> Option<&Path> {
        self.reader
            .as_ref()
            .map(|(index, _)| self.checkpoint.shards[*index].path.as_path())
    }

    /// Close the currently cached shard reader.
    pub fn close_reader(&mut self) -> Option<PathBuf> {
        let (index, _) = self.reader.take()?;
        Some(self.checkpoint.shards[index].path.clone())
    }

    fn location_and_reader(
        &mut self,
        name: &str,
    ) -> Result<(TensorLocation, TensorDescriptor, Endian)> {
        let location = self
            .locations
            .get(name)
            .copied()
            .ok_or_else(|| Error::InvalidTensor {
                tensor: name.to_string(),
                reason: "tensor is not present in the checkpoint".into(),
            })?;
        let shard = &self.checkpoint.shards[location.shard_index];
        if self
            .reader
            .as_ref()
            .is_none_or(|(shard_index, _)| *shard_index != location.shard_index)
        {
            let reader = validate_reopened_shard(
                open_reader(&shard.path, self.checkpoint.limits.clone())?,
                shard,
            )?;
            self.reader = Some((location.shard_index, reader));
        }
        Ok((
            location,
            shard.tensors[location.tensor_index].descriptor.clone(),
            shard.endian,
        ))
    }

    /// Materialize one physical tensor by its GGUF name.
    pub fn converted_tensor(&mut self, name: &str) -> Result<ConvertedCheckpointTensor> {
        let (location, descriptor, _) = self.location_and_reader(name)?;
        let converted = self
            .reader
            .as_mut()
            .expect("requested shard reader opened above")
            .1
            .read_tensor(&descriptor)
            .map_err(|source| Error::Shard {
                path: self.checkpoint.shards[location.shard_index].path.clone(),
                source: Box::new(source),
            })?;
        Ok(ConvertedCheckpointTensor {
            shard_index: location.shard_index,
            tensor_index: location.tensor_index,
            descriptor,
            converted,
        })
    }

    /// Materialize selected slabs along the outermost MLX tensor axis.
    pub fn converted_tensor_outer(
        &mut self,
        name: &str,
        selection: &OuterSelection,
    ) -> Result<ConvertedCheckpointTensor> {
        let (location, mut descriptor, _) = self.location_and_reader(name)?;
        let converted = self
            .reader
            .as_mut()
            .expect("requested shard reader opened above")
            .1
            .read_tensor_outer(&descriptor, selection)
            .map_err(|source| Error::Shard {
                path: self.checkpoint.shards[location.shard_index].path.clone(),
                source: Box::new(source),
            })?;
        let selected_outer = match selection {
            OuterSelection::Range { start, end } => end - start,
            OuterSelection::Indices(indices) => indices.len(),
        };
        let original_outer = *descriptor
            .dimensions
            .last()
            .expect("reader rejects scalar outer selections");
        let selected_outer_u64 = u64::try_from(selected_outer)
            .map_err(|_| Error::Overflow("selected outer dimension"))?;
        descriptor.byte_len = descriptor
            .byte_len
            .checked_div(original_outer)
            .and_then(|bytes| bytes.checked_mul(selected_outer_u64))
            .ok_or(Error::Overflow("selected GGUF payload bytes"))?;
        *descriptor
            .dimensions
            .last_mut()
            .expect("reader rejects scalar outer selections") = selected_outer_u64;
        Ok(ConvertedCheckpointTensor {
            shard_index: location.shard_index,
            tensor_index: location.tensor_index,
            descriptor,
            converted,
        })
    }

    /// Materialize one physical tensor without converting its native encoding.
    pub fn raw_tensor(&mut self, name: &str) -> Result<RawCheckpointTensor> {
        let (location, descriptor, endian) = self.location_and_reader(name)?;
        let data = self
            .reader
            .as_mut()
            .expect("requested shard reader opened above")
            .1
            .read_raw(&descriptor)
            .map_err(|source| Error::Shard {
                path: self.checkpoint.shards[location.shard_index].path.clone(),
                source: Box::new(source),
            })?;
        Ok(RawCheckpointTensor {
            shard_index: location.shard_index,
            tensor_index: location.tensor_index,
            endian,
            descriptor,
            data,
        })
    }
}

impl Iterator for ConvertedTensorIter<'_> {
    type Item = Result<ConvertedCheckpointTensor>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        loop {
            let Some(shard) = self.checkpoint.shards.get(self.shard_index) else {
                self.finished = true;
                return None;
            };
            if self.tensor_index >= shard.tensors.len() {
                self.shard_index += 1;
                self.tensor_index = 0;
                self.reader = None;
                continue;
            }
            if self.reader.is_none() {
                match open_reader(&shard.path, self.checkpoint.limits.clone())
                    .and_then(|reader| validate_reopened_shard(reader, shard))
                {
                    Ok(reader) => self.reader = Some(reader),
                    Err(error) => {
                        self.finished = true;
                        return Some(Err(error));
                    }
                }
            }

            let tensor_index = self.tensor_index;
            let descriptor = shard.tensors[tensor_index].descriptor.clone();
            self.tensor_index += 1;
            let converted = self
                .reader
                .as_mut()
                .expect("reader opened above")
                .read_tensor(&descriptor)
                .map_err(|source| Error::Shard {
                    path: shard.path.clone(),
                    source: Box::new(source),
                });
            return Some(converted.map(|converted| ConvertedCheckpointTensor {
                shard_index: self.shard_index,
                tensor_index,
                descriptor,
                converted,
            }));
        }
    }
}

fn validate_reopened_shard(
    reader: Reader<BufReader<File>>,
    shard: &CatalogShard,
) -> Result<Reader<BufReader<File>>> {
    let unchanged = reader.version() == shard.version
        && reader.endian() == shard.endian
        && reader.alignment() == shard.alignment
        && reader.tensors().len() == shard.tensors.len()
        && reader
            .tensors()
            .iter()
            .zip(&shard.tensors)
            .all(|(actual, cataloged)| actual == &cataloged.descriptor);
    if unchanged {
        Ok(reader)
    } else {
        Err(shard_error(format!(
            "GGUF shard {:?} changed after the checkpoint was opened",
            shard.path.display()
        )))
    }
}

fn catalog_tensors(descriptors: &[TensorDescriptor]) -> Result<Vec<CatalogTensor>> {
    descriptors.iter().map(catalog_tensor).collect()
}

fn catalog_tensor(descriptor: &TensorDescriptor) -> Result<CatalogTensor> {
    let kind = conversion_kind(descriptor.ggml_type)?;
    if let ConversionKind::IQuant = kind {
        return Ok(CatalogTensor {
            descriptor: descriptor.clone(),
            outputs: vec![LogicalTensorLayout {
                name: descriptor.name.clone(),
                shape: iquant_packed_shape(&descriptor.mlx_shape(), descriptor.ggml_type)?,
                dtype: LogicalDtype::U8,
            }],
            affine: None,
        });
    }
    if let ConversionKind::Dense(dtype) = kind {
        return Ok(CatalogTensor {
            descriptor: descriptor.clone(),
            outputs: vec![LogicalTensorLayout {
                name: descriptor.name.clone(),
                shape: descriptor.mlx_shape(),
                dtype: dtype.into(),
            }],
            affine: None,
        });
    }

    let ConversionKind::Affine { bits, group_size } = kind else {
        unreachable!("dense and IQ conversions returned above");
    };
    let prefix = descriptor.name.strip_suffix(".weight").ok_or_else(|| {
        Error::tensor(
            &descriptor.name,
            "quantized tensor name must end in .weight",
        )
    })?;
    let (weight_shape, scale_shape) = affine_shapes(descriptor, bits, group_size)?;
    let outputs = vec![
        LogicalTensorLayout {
            name: descriptor.name.clone(),
            shape: weight_shape,
            dtype: LogicalDtype::U32,
        },
        LogicalTensorLayout {
            name: format!("{prefix}.scales"),
            shape: scale_shape.clone(),
            dtype: LogicalDtype::F16,
        },
        LogicalTensorLayout {
            name: format!("{prefix}.biases"),
            shape: scale_shape,
            dtype: LogicalDtype::F16,
        },
    ];
    Ok(CatalogTensor {
        descriptor: descriptor.clone(),
        outputs,
        affine: Some((bits, group_size)),
    })
}

fn validate_logical_names<'a>(tensors: impl Iterator<Item = &'a CatalogTensor>) -> Result<()> {
    let mut owners = HashMap::<String, String>::new();
    for tensor in tensors {
        for output in &tensor.outputs {
            if let Some(first_source) =
                owners.insert(output.name.clone(), tensor.descriptor.name.clone())
            {
                return Err(Error::DuplicateLogicalTensor {
                    name: output.name.clone(),
                    first_source,
                    second_source: tensor.descriptor.name.clone(),
                });
            }
        }
    }
    Ok(())
}

fn split_value(metadata: &BTreeMap<String, MetadataValue>, key: &str) -> Result<Option<usize>> {
    metadata
        .get(key)
        .map(|value| {
            value
                .as_i64()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| {
                    shard_error(format!(
                        "GGUF metadata key {key:?} must be a non-negative integer scalar"
                    ))
                })
        })
        .transpose()
}

fn required_split_value(
    metadata: &BTreeMap<String, MetadataValue>,
    key: &str,
    path: &Path,
) -> Result<usize> {
    split_value(metadata, key)?.ok_or_else(|| {
        shard_error(format!(
            "GGUF shard {:?} is missing required metadata key {key:?}",
            path.display()
        ))
    })
}

fn shard_paths(path: &Path, split_count: usize) -> Result<Vec<PathBuf>> {
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .ok_or_else(|| {
            shard_error(format!(
                "GGUF path {:?} has a non-UTF-8 extension",
                path.display()
            ))
        })?;
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .ok_or_else(|| {
            shard_error(format!(
                "GGUF path {:?} has a non-UTF-8 filename",
                path.display()
            ))
        })?;
    let invalid_name = || {
        shard_error(format!(
            "sharded GGUF filename {:?} must end in -00001-of-NNNNN.gguf",
            path.display()
        ))
    };
    let (prefix_and_no, filename_count) = stem.rsplit_once("-of-").ok_or_else(&invalid_name)?;
    let (prefix, filename_no) = prefix_and_no.rsplit_once('-').ok_or_else(&invalid_name)?;
    let valid_digits =
        |value: &str| value.len() == 5 && value.bytes().all(|byte| byte.is_ascii_digit());
    if prefix.is_empty() || !valid_digits(filename_no) || !valid_digits(filename_count) {
        return Err(invalid_name());
    }
    let filename_no = filename_no
        .parse::<usize>()
        .map_err(|error| shard_error(format!("invalid GGUF shard number: {error}")))?;
    let filename_count = filename_count
        .parse::<usize>()
        .map_err(|error| shard_error(format!("invalid GGUF shard count: {error}")))?;
    if filename_no != 1 {
        return Err(shard_error(format!(
            "sharded GGUF must be loaded from shard 00001, got {filename_no:05}"
        )));
    }
    if filename_count != split_count {
        return Err(shard_error(format!(
            "GGUF filename declares {filename_count} shards, but {SPLIT_COUNT}={split_count}"
        )));
    }
    if split_count > 99_999 {
        return Err(shard_error(format!(
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

fn validate_extension(path: &Path) -> Result<()> {
    if path
        .extension()
        .and_then(|value| value.to_str())
        .is_some_and(|value| value.eq_ignore_ascii_case("gguf"))
    {
        Ok(())
    } else {
        Err(shard_error(format!(
            "checkpoint path {:?} must have a .gguf extension",
            path.display()
        )))
    }
}

fn open_reader(path: &Path, limits: Limits) -> Result<Reader<std::io::BufReader<std::fs::File>>> {
    Reader::open_with_limits(path, limits).map_err(|source| Error::Shard {
        path: path.to_path_buf(),
        source: Box::new(source),
    })
}

fn shard_error(message: impl Into<String>) -> Error {
    Error::InvalidShardSet(message.into())
}
