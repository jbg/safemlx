use crate::error::IoError;
use crate::{Array, Dtype};
use std::collections::{BTreeMap, HashMap};
use std::io::{Cursor, Read};
use std::ops::{Deref, DerefMut};
use std::path::{Path, PathBuf};

pub use safemlx_gguf::{
    Endian as GgufEndian, GgmlType as GgufType, LogicalDtype as GgufLogicalDtype,
    MetadataArray as GgufMetadataArray, MetadataValue as GgufMetadataValue,
    OuterSelection as GgufOuterSelection,
};

/// A validated GGUF checkpoint that materializes one physical tensor at a time.
#[derive(Debug, Clone)]
pub struct GgufCheckpoint {
    inner: safemlx_gguf::Checkpoint,
}

/// One named MLX array produced from a GGUF tensor.
#[derive(Debug)]
pub struct GgufArray {
    name: String,
    array: Array,
}

impl GgufArray {
    /// Logical checkpoint name of the array.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Materialized MLX array.
    pub fn array(&self) -> &Array {
        &self.array
    }

    /// Consume the value into its logical name and MLX array.
    pub fn into_parts(self) -> (String, Array) {
        (self.name, self.array)
    }
}

/// Atomic MLX representation of one packed affine GGUF tensor.
#[derive(Debug)]
pub struct GgufAffineTensor {
    physical_name: String,
    bits: u8,
    group_size: u32,
    weight: GgufArray,
    scales: GgufArray,
    biases: GgufArray,
}

impl GgufAffineTensor {
    /// Name of the physical tensor in the GGUF file.
    pub fn physical_name(&self) -> &str {
        &self.physical_name
    }

    /// Number of quantized bits per weight.
    pub fn bits(&self) -> u8 {
        self.bits
    }

    /// Quantization group size.
    pub fn group_size(&self) -> u32 {
        self.group_size
    }

    /// Packed weight array.
    pub fn weight(&self) -> &GgufArray {
        &self.weight
    }

    /// Per-group scale array.
    pub fn scales(&self) -> &GgufArray {
        &self.scales
    }

    /// Per-group bias array.
    pub fn biases(&self) -> &GgufArray {
        &self.biases
    }

    /// Consume the group into weight, scales, and biases arrays.
    pub fn into_arrays(self) -> [GgufArray; 3] {
        [self.weight, self.scales, self.biases]
    }
}

/// One converted physical GGUF tensor.
#[derive(Debug)]
pub enum GgufTensor {
    /// A physical tensor represented by one dense MLX array.
    Dense(GgufArray),
    /// A packed tensor represented by one atomic affine triple.
    Affine(GgufAffineTensor),
}

impl GgufTensor {
    /// Name of the physical tensor in the GGUF file.
    pub fn physical_name(&self) -> &str {
        match self {
            Self::Dense(tensor) => tensor.name(),
            Self::Affine(tensor) => tensor.physical_name(),
        }
    }

    /// Consume the tensor group into its logical named arrays.
    pub fn into_arrays(self) -> Vec<(String, Array)> {
        match self {
            Self::Dense(tensor) => vec![tensor.into_parts()],
            Self::Affine(tensor) => tensor
                .into_arrays()
                .into_iter()
                .map(GgufArray::into_parts)
                .collect(),
        }
    }
}

/// Fallible iterator over materialized MLX tensor groups.
pub struct GgufTensorIter<'a> {
    inner: safemlx_gguf::ConvertedTensorIter<'a>,
}

/// Indexed named-tensor materializer that reuses the current shard reader.
pub struct GgufMaterializer {
    inner: safemlx_gguf::TensorMaterializer,
}

/// One physical GGUF tensor retained in its checkpoint-native byte encoding.
#[derive(Debug)]
pub struct GgufRawTensor {
    inner: safemlx_gguf::RawCheckpointTensor,
}

impl GgufRawTensor {
    /// Endianness declared by the containing GGUF shard.
    pub fn endian(&self) -> safemlx_gguf::Endian {
        self.inner.endian()
    }

    /// Physical tensor descriptor.
    pub fn descriptor(&self) -> &safemlx_gguf::TensorDescriptor {
        self.inner.descriptor()
    }

    /// Checkpoint-native payload bytes.
    pub fn data(&self) -> &[u8] {
        self.inner.data()
    }

    /// Consume this tensor and return its checkpoint-native payload.
    pub fn into_data(self) -> Vec<u8> {
        self.inner.into_data()
    }
}

impl std::fmt::Debug for GgufTensorIter<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GgufTensorIter")
            .finish_non_exhaustive()
    }
}

impl std::fmt::Debug for GgufMaterializer {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GgufMaterializer")
            .finish_non_exhaustive()
    }
}

impl Iterator for GgufTensorIter<'_> {
    type Item = Result<GgufTensor, IoError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner
            .next()
            .map(|result| result.map_err(IoError::from).and_then(convert_tensor))
    }
}

impl GgufCheckpoint {
    /// Open and validate a single-file or canonically sharded GGUF checkpoint.
    ///
    /// This parses all shard headers and descriptors, but reads no tensor payloads.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, IoError> {
        let path = path.as_ref();
        if !path.is_file() {
            return Err(IoError::NotFile);
        }
        if !path
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
        {
            return Err(IoError::UnsupportedFormat);
        }
        Ok(Self {
            inner: safemlx_gguf::Checkpoint::open(path)?,
        })
    }

    /// Typed metadata from the first checkpoint shard.
    pub fn metadata(&self) -> &BTreeMap<String, GgufMetadataValue> {
        self.inner.metadata()
    }

    /// Validated header-only checkpoint description.
    pub fn catalog(&self) -> &safemlx_gguf::Checkpoint {
        &self.inner
    }

    /// Iterate over converted tensors without retaining earlier payloads.
    pub fn converted_tensors(&self) -> GgufTensorIter<'_> {
        GgufTensorIter {
            inner: self.inner.converted_tensors(),
        }
    }

    /// Create an indexed named-tensor materializer with bounded reader reuse.
    pub fn materializer(&self) -> GgufMaterializer {
        GgufMaterializer {
            inner: self.inner.materializer(),
        }
    }

    /// Materialize and visit one physical tensor at a time.
    pub fn for_each_converted_tensor<F>(&self, mut visitor: F) -> Result<(), IoError>
    where
        F: FnMut(GgufTensor) -> Result<(), IoError>,
    {
        for tensor in self.converted_tensors() {
            visitor(tensor?)?;
        }
        Ok(())
    }
}

impl GgufMaterializer {
    /// Path of the shard containing `name`, without opening its payload reader.
    pub fn shard_path_for_tensor(&self, name: &str) -> Result<&Path, IoError> {
        self.inner
            .shard_path_for_tensor(name)
            .map_err(IoError::from)
    }

    /// Path of the currently cached shard reader, if any.
    pub fn open_shard_path(&self) -> Option<&Path> {
        self.inner.open_shard_path()
    }

    /// Close the currently cached shard reader.
    pub fn close_reader(&mut self) -> Option<PathBuf> {
        self.inner.close_reader()
    }

    /// Materialize one physical tensor by its GGUF name.
    pub fn converted_tensor(&mut self, name: &str) -> Result<GgufTensor, IoError> {
        convert_tensor(self.inner.converted_tensor(name)?)
    }

    /// Materialize selected slabs along the outermost MLX tensor axis.
    pub fn converted_tensor_outer(
        &mut self,
        name: &str,
        selection: &GgufOuterSelection,
    ) -> Result<GgufTensor, IoError> {
        convert_tensor(self.inner.converted_tensor_outer(name, selection)?)
    }

    /// Materialize one physical tensor without converting its GGUF blocks.
    pub fn raw_tensor(&mut self, name: &str) -> Result<GgufRawTensor, IoError> {
        Ok(GgufRawTensor {
            inner: self.inner.raw_tensor(name)?,
        })
    }
}

fn convert_tensor(tensor: safemlx_gguf::ConvertedCheckpointTensor) -> Result<GgufTensor, IoError> {
    let descriptor = tensor.descriptor().clone();
    match tensor.into_converted() {
        safemlx_gguf::ConvertedTensor::Dense(dense) => {
            let shape = mlx_shape_i32(&descriptor.name, &dense.shape)?;
            let dtype = match dense.dtype {
                safemlx_gguf::DenseDtype::F32 => Dtype::Float32,
                safemlx_gguf::DenseDtype::F16 => Dtype::Float16,
                safemlx_gguf::DenseDtype::Bf16 => Dtype::Bfloat16,
                safemlx_gguf::DenseDtype::I8 => Dtype::Int8,
                safemlx_gguf::DenseDtype::I16 => Dtype::Int16,
                safemlx_gguf::DenseDtype::I32 => Dtype::Int32,
                safemlx_gguf::DenseDtype::I64 => Dtype::Int64,
                safemlx_gguf::DenseDtype::F64 => Dtype::Float64,
            };
            let array = unsafe { Array::from_raw_data(dense.data.as_ptr().cast(), &shape, dtype) };
            Ok(GgufTensor::Dense(GgufArray {
                name: descriptor.name,
                array,
            }))
        }
        safemlx_gguf::ConvertedTensor::Affine(affine) => {
            let weight_shape = mlx_shape_i32(&descriptor.name, &affine.weight_shape)?;
            let scale_shape = mlx_shape_i32(&descriptor.name, &affine.scale_shape)?;
            let weight = unsafe {
                Array::from_raw_data(affine.weights.as_ptr().cast(), &weight_shape, Dtype::Uint32)
            };
            let scales = unsafe {
                Array::from_raw_data(affine.scales.as_ptr().cast(), &scale_shape, Dtype::Float16)
            };
            let biases = unsafe {
                Array::from_raw_data(affine.biases.as_ptr().cast(), &scale_shape, Dtype::Float16)
            };
            let prefix = descriptor
                .name
                .strip_suffix(".weight")
                .ok_or_else(|| {
                    IoError::InvalidGguf(format!(
                        "quantized tensor {:?} must end in .weight",
                        descriptor.name
                    ))
                })?
                .to_owned();
            Ok(GgufTensor::Affine(GgufAffineTensor {
                physical_name: descriptor.name.clone(),
                bits: affine.bits,
                group_size: affine.group_size,
                weight: GgufArray {
                    name: descriptor.name,
                    array: weight,
                },
                scales: GgufArray {
                    name: format!("{prefix}.scales"),
                    array: scales,
                },
                biases: GgufArray {
                    name: format!("{prefix}.biases"),
                    array: biases,
                },
            }))
        }
    }
}

fn mlx_shape_i32(name: &str, shape: &[u64]) -> Result<Vec<i32>, IoError> {
    shape
        .iter()
        .map(|&value| {
            i32::try_from(value).map_err(|_| {
                IoError::InvalidGguf(format!(
                    "tensor {name:?} dimension {value} exceeds MLX i32 shape limits"
                ))
            })
        })
        .collect()
}

/// GGUF key/value metadata parsed by the pure-Rust backend without an MLX device.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GgufMetadata(HashMap<String, GgufMetadataValue>);

impl GgufMetadata {
    /// Parse only the header and metadata section of a GGUF file.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, IoError> {
        let path = path.as_ref();
        if !path.is_file() {
            return Err(IoError::NotFile);
        }
        let reader = safemlx_gguf::Reader::open(path)?;
        Ok(Self(
            reader
                .metadata()
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        ))
    }
    /// Parse metadata from a non-seekable source. File-backed callers should
    /// prefer [`Self::from_file`], which never buffers tensor payloads.
    pub fn from_reader(mut reader: impl Read) -> Result<Self, IoError> {
        const MAX_READER_BYTES: u64 = (2u64 << 30) + 1;
        let mut bytes = Vec::new();
        reader
            .by_ref()
            .take(MAX_READER_BYTES)
            .read_to_end(&mut bytes)
            .map_err(|_| IoError::UnableToOpenFile)?;
        if bytes.len() as u64 == MAX_READER_BYTES {
            return Err(IoError::InvalidGguf(
                "reader exceeds the 2 GiB compatibility limit".into(),
            ));
        }
        let parsed = safemlx_gguf::Reader::new(Cursor::new(bytes))?;
        Ok(Self(
            parsed
                .metadata()
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        ))
    }
    /// Consume the wrapper and return the metadata map.
    pub fn into_inner(self) -> HashMap<String, GgufMetadataValue> {
        self.0
    }
}
impl Deref for GgufMetadata {
    type Target = HashMap<String, GgufMetadataValue>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for GgufMetadata {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl FromIterator<(String, GgufMetadataValue)> for GgufMetadata {
    fn from_iter<T: IntoIterator<Item = (String, GgufMetadataValue)>>(iter: T) -> Self {
        Self(iter.into_iter().collect())
    }
}
