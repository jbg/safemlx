use crate::{Error, Result};

pub const DEFAULT_ALIGNMENT: u64 = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Endian {
    Little,
    Big,
}

impl Endian {
    pub(crate) fn u16(self, b: [u8; 2]) -> u16 {
        match self {
            Self::Little => u16::from_le_bytes(b),
            Self::Big => u16::from_be_bytes(b),
        }
    }
    pub(crate) fn u32(self, b: [u8; 4]) -> u32 {
        match self {
            Self::Little => u32::from_le_bytes(b),
            Self::Big => u32::from_be_bytes(b),
        }
    }
    pub(crate) fn u64(self, b: [u8; 8]) -> u64 {
        match self {
            Self::Little => u64::from_le_bytes(b),
            Self::Big => u64::from_be_bytes(b),
        }
    }
    pub(crate) fn put_u16(self, value: u16) -> [u8; 2] {
        match self {
            Self::Little => value.to_le_bytes(),
            Self::Big => value.to_be_bytes(),
        }
    }
    pub(crate) fn put_u32(self, value: u32) -> [u8; 4] {
        match self {
            Self::Little => value.to_le_bytes(),
            Self::Big => value.to_be_bytes(),
        }
    }
    pub(crate) fn put_u64(self, value: u64) -> [u8; 8] {
        match self {
            Self::Little => value.to_le_bytes(),
            Self::Big => value.to_be_bytes(),
        }
    }
}

/// Every GGUF metadata value type, including recursively nested arrays.
#[derive(Debug, Clone, PartialEq)]
pub enum MetadataValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(MetadataArray),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

#[derive(Debug, Clone, PartialEq)]
pub enum MetadataArray {
    Uint8(Vec<u8>),
    Int8(Vec<i8>),
    Uint16(Vec<u16>),
    Int16(Vec<i16>),
    Uint32(Vec<u32>),
    Int32(Vec<i32>),
    Float32(Vec<f32>),
    Bool(Vec<bool>),
    String(Vec<String>),
    Array(Vec<MetadataArray>),
    Uint64(Vec<u64>),
    Int64(Vec<i64>),
    Float64(Vec<f64>),
}

impl MetadataArray {
    pub fn len(&self) -> usize {
        match self {
            Self::Uint8(v) => v.len(),
            Self::Int8(v) => v.len(),
            Self::Uint16(v) => v.len(),
            Self::Int16(v) => v.len(),
            Self::Uint32(v) => v.len(),
            Self::Int32(v) => v.len(),
            Self::Float32(v) => v.len(),
            Self::Bool(v) => v.len(),
            Self::String(v) => v.len(),
            Self::Array(v) => v.len(),
            Self::Uint64(v) => v.len(),
            Self::Int64(v) => v.len(),
            Self::Float64(v) => v.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn as_strings(&self) -> Option<&[String]> {
        if let Self::String(v) = self {
            Some(v)
        } else {
            None
        }
    }
    pub fn to_i64_vec(&self) -> Option<Vec<i64>> {
        match self {
            Self::Uint8(v) => Some(v.iter().map(|&x| x.into()).collect()),
            Self::Int8(v) => Some(v.iter().map(|&x| x.into()).collect()),
            Self::Uint16(v) => Some(v.iter().map(|&x| x.into()).collect()),
            Self::Int16(v) => Some(v.iter().map(|&x| x.into()).collect()),
            Self::Uint32(v) => Some(v.iter().map(|&x| x.into()).collect()),
            Self::Int32(v) => Some(v.iter().map(|&x| x.into()).collect()),
            Self::Uint64(v) => v.iter().map(|&x| x.try_into().ok()).collect(),
            Self::Int64(v) => Some(v.clone()),
            _ => None,
        }
    }
    pub fn to_f32_vec(&self) -> Option<Vec<f32>> {
        match self {
            Self::Float32(v) => Some(v.clone()),
            Self::Float64(v) => Some(v.iter().map(|&x| x as f32).collect()),
            _ => None,
        }
    }
    pub(crate) fn type_code(&self) -> u32 {
        match self {
            Self::Uint8(_) => 0,
            Self::Int8(_) => 1,
            Self::Uint16(_) => 2,
            Self::Int16(_) => 3,
            Self::Uint32(_) => 4,
            Self::Int32(_) => 5,
            Self::Float32(_) => 6,
            Self::Bool(_) => 7,
            Self::String(_) => 8,
            Self::Array(_) => 9,
            Self::Uint64(_) => 10,
            Self::Int64(_) => 11,
            Self::Float64(_) => 12,
        }
    }
}

impl MetadataValue {
    pub fn as_str(&self) -> Option<&str> {
        if let Self::String(v) = self {
            Some(v)
        } else {
            None
        }
    }
    pub fn as_array(&self) -> Option<&MetadataArray> {
        if let Self::Array(v) = self {
            Some(v)
        } else {
            None
        }
    }
    pub fn as_strings(&self) -> Option<&[String]> {
        self.as_array()?.as_strings()
    }
    pub fn as_bool(&self) -> Option<bool> {
        if let Self::Bool(v) = self {
            Some(*v)
        } else {
            None
        }
    }
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Uint8(v) => Some((*v).into()),
            Self::Int8(v) => Some((*v).into()),
            Self::Uint16(v) => Some((*v).into()),
            Self::Int16(v) => Some((*v).into()),
            Self::Uint32(v) => Some((*v).into()),
            Self::Int32(v) => Some((*v).into()),
            Self::Uint64(v) => (*v).try_into().ok(),
            Self::Int64(v) => Some(*v),
            _ => None,
        }
    }
    pub fn to_i64_vec(&self) -> Option<Vec<i64>> {
        match self {
            Self::Array(v) => v.to_i64_vec(),
            v => v.as_i64().map(|x| vec![x]),
        }
    }
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::Float32(v) => Some(*v),
            Self::Float64(v) => Some(*v as f32),
            _ => None,
        }
    }
    pub(crate) fn type_code(&self) -> u32 {
        match self {
            Self::Uint8(_) => 0,
            Self::Int8(_) => 1,
            Self::Uint16(_) => 2,
            Self::Int16(_) => 3,
            Self::Uint32(_) => 4,
            Self::Int32(_) => 5,
            Self::Float32(_) => 6,
            Self::Bool(_) => 7,
            Self::String(_) => 8,
            Self::Array(_) => 9,
            Self::Uint64(_) => 10,
            Self::Int64(_) => 11,
            Self::Float64(_) => 12,
        }
    }
}

/// GGML tensor encodings relevant to the safemlx public path.
///
/// Numeric codes and block geometry are pinned to llama.cpp commit
/// `c0bc8591e8815c63cb01dd3f051a8b0df02501c9` (`ggml/include/ggml.h`,
/// `ggml/src/ggml-common.h`, and `ggml/src/ggml.c`). The three
/// `IQ4_NL_*_*` entries are retained only so their historical numeric codes do
/// not become [`Unknown`](Self::Unknown): that upstream revision explicitly
/// marks them as removed runtime-repacking layouts with zero block and type
/// sizes, not GGUF tensor encodings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    IQ2XXS,
    IQ2XS,
    IQ3XXS,
    IQ1S,
    IQ4NL,
    IQ3S,
    IQ2S,
    IQ4XS,
    I8,
    I16,
    I32,
    I64,
    F64,
    IQ1M,
    Bf16,
    RemovedIQ4NL4_4,
    RemovedIQ4NL4_8,
    RemovedIQ4NL8_8,
    Unknown(u32),
}

impl GgmlType {
    pub fn from_code(code: u32) -> Self {
        match code {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            16 => Self::IQ2XXS,
            17 => Self::IQ2XS,
            18 => Self::IQ3XXS,
            19 => Self::IQ1S,
            20 => Self::IQ4NL,
            21 => Self::IQ3S,
            22 => Self::IQ2S,
            23 => Self::IQ4XS,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            29 => Self::IQ1M,
            30 => Self::Bf16,
            36 => Self::RemovedIQ4NL4_4,
            37 => Self::RemovedIQ4NL4_8,
            38 => Self::RemovedIQ4NL8_8,
            other => Self::Unknown(other),
        }
    }
    pub fn code(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q4_1 => 3,
            Self::Q5_0 => 6,
            Self::Q5_1 => 7,
            Self::Q8_0 => 8,
            Self::Q2K => 10,
            Self::Q3K => 11,
            Self::Q4K => 12,
            Self::Q5K => 13,
            Self::Q6K => 14,
            Self::IQ2XXS => 16,
            Self::IQ2XS => 17,
            Self::IQ3XXS => 18,
            Self::IQ1S => 19,
            Self::IQ4NL => 20,
            Self::IQ3S => 21,
            Self::IQ2S => 22,
            Self::IQ4XS => 23,
            Self::I8 => 24,
            Self::I16 => 25,
            Self::I32 => 26,
            Self::I64 => 27,
            Self::F64 => 28,
            Self::IQ1M => 29,
            Self::Bf16 => 30,
            Self::RemovedIQ4NL4_4 => 36,
            Self::RemovedIQ4NL4_8 => 37,
            Self::RemovedIQ4NL8_8 => 38,
            Self::Unknown(v) => v,
        }
    }
    pub fn block_and_bytes(self) -> Result<(u64, u64)> {
        match self {
            Self::F32 => Ok((1, 4)),
            Self::F16 | Self::Bf16 | Self::I16 => Ok((1, 2)),
            Self::I8 => Ok((1, 1)),
            Self::I32 => Ok((1, 4)),
            Self::I64 | Self::F64 => Ok((1, 8)),
            Self::Q4_0 => Ok((32, 18)),
            Self::Q4_1 => Ok((32, 20)),
            Self::Q5_0 => Ok((32, 22)),
            Self::Q5_1 => Ok((32, 24)),
            Self::Q8_0 => Ok((32, 34)),
            Self::Q2K => Ok((256, 84)),
            Self::Q3K => Ok((256, 110)),
            Self::Q4K => Ok((256, 144)),
            Self::Q5K => Ok((256, 176)),
            Self::Q6K => Ok((256, 210)),
            Self::IQ2XXS => Ok((256, 66)),
            Self::IQ2XS => Ok((256, 74)),
            Self::IQ3XXS => Ok((256, 98)),
            Self::IQ1S => Ok((256, 50)),
            Self::IQ4NL => Ok((32, 18)),
            Self::IQ3S => Ok((256, 110)),
            Self::IQ2S => Ok((256, 82)),
            Self::IQ4XS => Ok((256, 136)),
            Self::IQ1M => Ok((256, 56)),
            Self::RemovedIQ4NL4_4 | Self::RemovedIQ4NL4_8 | Self::RemovedIQ4NL8_8 => {
                Err(Error::UnsupportedTensorType(self.code()))
            }
            Self::Unknown(v) => Err(Error::UnsupportedTensorType(v)),
        }
    }

    /// Whether this is one of the nine canonical, on-disk GGML IQ encodings.
    pub fn is_iq(self) -> bool {
        matches!(
            self,
            Self::IQ2XXS
                | Self::IQ2XS
                | Self::IQ3XXS
                | Self::IQ1S
                | Self::IQ4NL
                | Self::IQ3S
                | Self::IQ2S
                | Self::IQ4XS
                | Self::IQ1M
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorDescriptor {
    pub name: String,
    /// Dimensions in GGML order (fastest-moving dimension first).
    pub dimensions: Vec<u64>,
    pub ggml_type: GgmlType,
    pub relative_offset: u64,
    pub data_offset: u64,
    pub byte_len: u64,
}

impl TensorDescriptor {
    pub fn element_count(&self) -> Result<u64> {
        self.dimensions.iter().try_fold(1u64, |a, &b| {
            a.checked_mul(b)
                .ok_or(Error::Overflow("tensor element count"))
        })
    }
    /// Shape in MLX/row-major order.
    pub fn mlx_shape(&self) -> Vec<u64> {
        self.dimensions.iter().rev().copied().collect()
    }
}

pub(crate) fn align_up(value: u64, alignment: u64) -> Result<u64> {
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(Error::InvalidHeader(format!(
            "alignment {alignment} is not a non-zero power of two"
        )));
    }
    value
        .checked_add(alignment - 1)
        .map(|v| v & !(alignment - 1))
        .ok_or(Error::Overflow("alignment"))
}
