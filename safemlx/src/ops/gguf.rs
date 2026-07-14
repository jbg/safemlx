use crate::error::IoError;
use std::collections::HashMap;
use std::io::{BufReader, Read};
use std::ops::{Deref, DerefMut};
use std::path::Path;

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const MAX_METADATA_ENTRIES: u64 = 10_000_000;
const MAX_ARRAY_ELEMENTS: u64 = 100_000_000;
const MAX_STRING_BYTES: u64 = 1 << 30;

/// A homogeneous GGUF metadata array stored entirely in host memory.
#[derive(Debug, Clone, PartialEq)]
pub enum GgufMetadataArray {
    /// Unsigned 8-bit values.
    Uint8(Vec<u8>),
    /// Signed 8-bit values.
    Int8(Vec<i8>),
    /// Unsigned 16-bit values.
    Uint16(Vec<u16>),
    /// Signed 16-bit values.
    Int16(Vec<i16>),
    /// Unsigned 32-bit values.
    Uint32(Vec<u32>),
    /// Signed 32-bit values.
    Int32(Vec<i32>),
    /// 32-bit floating-point values.
    Float32(Vec<f32>),
    /// Boolean values.
    Bool(Vec<bool>),
    /// UTF-8 string values.
    String(Vec<String>),
    /// Nested homogeneous arrays.
    Array(Vec<GgufMetadataArray>),
    /// Unsigned 64-bit values.
    Uint64(Vec<u64>),
    /// Signed 64-bit values.
    Int64(Vec<i64>),
    /// 64-bit floating-point values.
    Float64(Vec<f64>),
}

impl GgufMetadataArray {
    /// Number of values in the array.
    pub fn len(&self) -> usize {
        match self {
            Self::Uint8(values) => values.len(),
            Self::Int8(values) => values.len(),
            Self::Uint16(values) => values.len(),
            Self::Int16(values) => values.len(),
            Self::Uint32(values) => values.len(),
            Self::Int32(values) => values.len(),
            Self::Float32(values) => values.len(),
            Self::Bool(values) => values.len(),
            Self::String(values) => values.len(),
            Self::Array(values) => values.len(),
            Self::Uint64(values) => values.len(),
            Self::Int64(values) => values.len(),
            Self::Float64(values) => values.len(),
        }
    }

    /// Whether the array contains no values.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return string values when this is a string array.
    pub fn as_strings(&self) -> Option<&[String]> {
        match self {
            Self::String(values) => Some(values),
            _ => None,
        }
    }

    /// Convert any integer array to signed 64-bit values.
    pub fn to_i64_vec(&self) -> Option<Vec<i64>> {
        match self {
            Self::Uint8(values) => Some(values.iter().map(|&value| i64::from(value)).collect()),
            Self::Int8(values) => Some(values.iter().map(|&value| i64::from(value)).collect()),
            Self::Uint16(values) => Some(values.iter().map(|&value| i64::from(value)).collect()),
            Self::Int16(values) => Some(values.iter().map(|&value| i64::from(value)).collect()),
            Self::Uint32(values) => Some(values.iter().map(|&value| i64::from(value)).collect()),
            Self::Int32(values) => Some(values.iter().map(|&value| i64::from(value)).collect()),
            Self::Uint64(values) => values
                .iter()
                .map(|&value| i64::try_from(value).ok())
                .collect(),
            Self::Int64(values) => Some(values.clone()),
            _ => None,
        }
    }

    /// Convert float metadata to 32-bit values.
    pub fn to_f32_vec(&self) -> Option<Vec<f32>> {
        match self {
            Self::Float32(values) => Some(values.clone()),
            Self::Float64(values) => Some(values.iter().map(|&value| value as f32).collect()),
            _ => None,
        }
    }
}

/// A scalar or array GGUF metadata value.
#[derive(Debug, Clone, PartialEq)]
pub enum GgufMetadataValue {
    /// Unsigned 8-bit value.
    Uint8(u8),
    /// Signed 8-bit value.
    Int8(i8),
    /// Unsigned 16-bit value.
    Uint16(u16),
    /// Signed 16-bit value.
    Int16(i16),
    /// Unsigned 32-bit value.
    Uint32(u32),
    /// Signed 32-bit value.
    Int32(i32),
    /// 32-bit floating-point value.
    Float32(f32),
    /// Boolean value.
    Bool(bool),
    /// UTF-8 string value.
    String(String),
    /// Homogeneous array value.
    Array(GgufMetadataArray),
    /// Unsigned 64-bit value.
    Uint64(u64),
    /// Signed 64-bit value.
    Int64(i64),
    /// 64-bit floating-point value.
    Float64(f64),
}

impl GgufMetadataValue {
    /// Borrow this value as a string.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value),
            _ => None,
        }
    }

    /// Borrow this value as an array.
    pub fn as_array(&self) -> Option<&GgufMetadataArray> {
        match self {
            Self::Array(value) => Some(value),
            _ => None,
        }
    }

    /// Borrow this value as an array of strings.
    pub fn as_strings(&self) -> Option<&[String]> {
        self.as_array()?.as_strings()
    }

    /// Read this value as a boolean.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(value) => Some(*value),
            _ => None,
        }
    }

    /// Convert any integer scalar to a signed 64-bit value.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Uint8(value) => Some(i64::from(*value)),
            Self::Int8(value) => Some(i64::from(*value)),
            Self::Uint16(value) => Some(i64::from(*value)),
            Self::Int16(value) => Some(i64::from(*value)),
            Self::Uint32(value) => Some(i64::from(*value)),
            Self::Int32(value) => Some(i64::from(*value)),
            Self::Uint64(value) => i64::try_from(*value).ok(),
            Self::Int64(value) => Some(*value),
            _ => None,
        }
    }

    /// Convert an integer scalar or integer array to signed 64-bit values.
    pub fn to_i64_vec(&self) -> Option<Vec<i64>> {
        match self {
            Self::Array(values) => values.to_i64_vec(),
            value => value.as_i64().map(|value| vec![value]),
        }
    }

    /// Convert either floating-point scalar representation to `f32`.
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::Float32(value) => Some(*value),
            Self::Float64(value) => Some(*value as f32),
            _ => None,
        }
    }
}

/// GGUF key/value metadata parsed without initializing an MLX device or stream.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GgufMetadata(HashMap<String, GgufMetadataValue>);

impl GgufMetadata {
    /// Parse only the header and metadata section of a GGUF file.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, IoError> {
        let path = path.as_ref();
        if !path.is_file() {
            return Err(IoError::NotFile);
        }
        let file = std::fs::File::open(path).map_err(|_| IoError::UnableToOpenFile)?;
        Self::from_reader(BufReader::new(file))
    }

    /// Parse GGUF metadata from a reader positioned at the beginning of a file.
    pub fn from_reader(reader: impl Read) -> Result<Self, IoError> {
        Parser::new(reader).parse()
    }

    /// Consume the metadata wrapper and return its key/value map.
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

#[derive(Clone, Copy)]
enum Endian {
    Little,
    Big,
}

struct Parser<R> {
    reader: R,
    endian: Endian,
    version: u32,
}

impl<R: Read> Parser<R> {
    fn new(reader: R) -> Self {
        Self {
            reader,
            endian: Endian::Little,
            version: 0,
        }
    }

    fn parse(mut self) -> Result<GgufMetadata, IoError> {
        let mut magic = [0; 4];
        self.read_exact(&mut magic)?;
        if &magic != GGUF_MAGIC {
            return Err(self.invalid("invalid GGUF magic"));
        }

        let mut version_bytes = [0; 4];
        self.read_exact(&mut version_bytes)?;
        let little = u32::from_le_bytes(version_bytes);
        let big = u32::from_be_bytes(version_bytes);
        let (version, endian) = if matches!(little, 1..=3) {
            (little, Endian::Little)
        } else if matches!(big, 1..=3) {
            (big, Endian::Big)
        } else {
            return Err(self.invalid(format!("unsupported GGUF version {little}")));
        };
        self.version = version;
        self.endian = endian;

        let _tensor_count = self.read_count()?;
        let metadata_count = self.read_count()?;
        if metadata_count > MAX_METADATA_ENTRIES {
            return Err(self.invalid("GGUF metadata entry count is unreasonably large"));
        }
        let capacity = usize::try_from(metadata_count)
            .map_err(|_| self.invalid("GGUF metadata entry count exceeds this platform"))?;
        let mut values = HashMap::with_capacity(capacity);
        for _ in 0..metadata_count {
            let key = self.read_string()?;
            if key.len() > u16::MAX as usize || !key.is_ascii() {
                return Err(self.invalid("GGUF metadata key is not valid ASCII"));
            }
            let value_type = self.read_u32()?;
            let value = self.read_value(value_type)?;
            if values.insert(key.clone(), value).is_some() {
                return Err(self.invalid(format!("duplicate GGUF metadata key {key:?}")));
            }
        }
        Ok(GgufMetadata(values))
    }

    fn read_value(&mut self, value_type: u32) -> Result<GgufMetadataValue, IoError> {
        Ok(match value_type {
            0 => GgufMetadataValue::Uint8(self.read_u8()?),
            1 => GgufMetadataValue::Int8(self.read_u8()? as i8),
            2 => GgufMetadataValue::Uint16(self.read_u16()?),
            3 => GgufMetadataValue::Int16(self.read_u16()? as i16),
            4 => GgufMetadataValue::Uint32(self.read_u32()?),
            5 => GgufMetadataValue::Int32(self.read_u32()? as i32),
            6 => GgufMetadataValue::Float32(f32::from_bits(self.read_u32()?)),
            7 => match self.read_u8()? {
                0 => GgufMetadataValue::Bool(false),
                1 => GgufMetadataValue::Bool(true),
                value => return Err(self.invalid(format!("invalid GGUF boolean value {value}"))),
            },
            8 => GgufMetadataValue::String(self.read_string()?),
            9 => GgufMetadataValue::Array(self.read_array()?),
            10 => GgufMetadataValue::Uint64(self.read_u64()?),
            11 => GgufMetadataValue::Int64(self.read_u64()? as i64),
            12 => GgufMetadataValue::Float64(f64::from_bits(self.read_u64()?)),
            other => return Err(self.invalid(format!("unknown GGUF metadata type {other}"))),
        })
    }

    fn read_array(&mut self) -> Result<GgufMetadataArray, IoError> {
        let element_type = self.read_u32()?;
        let len = self.read_count()?;
        if len > MAX_ARRAY_ELEMENTS {
            return Err(self.invalid("GGUF metadata array is unreasonably large"));
        }
        let len = usize::try_from(len)
            .map_err(|_| self.invalid("GGUF metadata array exceeds this platform"))?;

        macro_rules! read_values {
            ($variant:ident, $read:ident) => {{
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(self.$read()?);
                }
                GgufMetadataArray::$variant(values)
            }};
        }

        Ok(match element_type {
            0 => read_values!(Uint8, read_u8),
            1 => GgufMetadataArray::Int8({
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(self.read_u8()? as i8);
                }
                values
            }),
            2 => read_values!(Uint16, read_u16),
            3 => GgufMetadataArray::Int16({
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(self.read_u16()? as i16);
                }
                values
            }),
            4 => read_values!(Uint32, read_u32),
            5 => GgufMetadataArray::Int32({
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(self.read_u32()? as i32);
                }
                values
            }),
            6 => GgufMetadataArray::Float32({
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(f32::from_bits(self.read_u32()?));
                }
                values
            }),
            7 => GgufMetadataArray::Bool({
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(match self.read_u8()? {
                        0 => false,
                        1 => true,
                        value => {
                            return Err(
                                self.invalid(format!("invalid GGUF boolean array value {value}"))
                            )
                        }
                    });
                }
                values
            }),
            8 => GgufMetadataArray::String({
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(self.read_string()?);
                }
                values
            }),
            9 => GgufMetadataArray::Array({
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(self.read_array()?);
                }
                values
            }),
            10 => read_values!(Uint64, read_u64),
            11 => GgufMetadataArray::Int64({
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(self.read_u64()? as i64);
                }
                values
            }),
            12 => GgufMetadataArray::Float64({
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(f64::from_bits(self.read_u64()?));
                }
                values
            }),
            other => {
                return Err(
                    self.invalid(format!("unknown GGUF metadata array element type {other}"))
                )
            }
        })
    }

    fn read_string(&mut self) -> Result<String, IoError> {
        let len = self.read_count()?;
        if len > MAX_STRING_BYTES {
            return Err(self.invalid("GGUF string is unreasonably large"));
        }
        let len = usize::try_from(len)
            .map_err(|_| self.invalid("GGUF string length exceeds this platform"))?;
        let mut bytes = vec![0; len];
        self.read_exact(&mut bytes)?;
        String::from_utf8(bytes).map_err(|error| self.invalid(error.to_string()))
    }

    fn read_count(&mut self) -> Result<u64, IoError> {
        if self.version == 1 {
            self.read_u32().map(u64::from)
        } else {
            self.read_u64()
        }
    }

    fn read_u8(&mut self) -> Result<u8, IoError> {
        let mut bytes = [0; 1];
        self.read_exact(&mut bytes)?;
        Ok(bytes[0])
    }

    fn read_u16(&mut self) -> Result<u16, IoError> {
        let mut bytes = [0; 2];
        self.read_exact(&mut bytes)?;
        Ok(match self.endian {
            Endian::Little => u16::from_le_bytes(bytes),
            Endian::Big => u16::from_be_bytes(bytes),
        })
    }

    fn read_u32(&mut self) -> Result<u32, IoError> {
        let mut bytes = [0; 4];
        self.read_exact(&mut bytes)?;
        Ok(match self.endian {
            Endian::Little => u32::from_le_bytes(bytes),
            Endian::Big => u32::from_be_bytes(bytes),
        })
    }

    fn read_u64(&mut self) -> Result<u64, IoError> {
        let mut bytes = [0; 8];
        self.read_exact(&mut bytes)?;
        Ok(match self.endian {
            Endian::Little => u64::from_le_bytes(bytes),
            Endian::Big => u64::from_be_bytes(bytes),
        })
    }

    fn read_exact(&mut self, bytes: &mut [u8]) -> Result<(), IoError> {
        self.reader
            .read_exact(bytes)
            .map_err(|error| self.invalid(error.to_string()))
    }

    fn invalid(&self, message: impl Into<String>) -> IoError {
        IoError::InvalidGguf(message.into())
    }
}

#[cfg(test)]
mod tests {
    use super::{GgufMetadata, GgufMetadataArray, GgufMetadataValue};
    use std::io::Cursor;

    fn string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }

    #[test]
    fn parses_metadata_without_tensor_data() {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&123u64.to_le_bytes());
        bytes.extend_from_slice(&3u64.to_le_bytes());

        string(&mut bytes, "general.architecture");
        bytes.extend_from_slice(&8u32.to_le_bytes());
        string(&mut bytes, "llama");

        string(&mut bytes, "tokenizer.ggml.tokens");
        bytes.extend_from_slice(&9u32.to_le_bytes());
        bytes.extend_from_slice(&8u32.to_le_bytes());
        bytes.extend_from_slice(&2u64.to_le_bytes());
        string(&mut bytes, "<unk>");
        string(&mut bytes, "hello");

        string(&mut bytes, "tokenizer.ggml.eos_token_id");
        bytes.extend_from_slice(&4u32.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());

        let metadata = GgufMetadata::from_reader(Cursor::new(bytes)).unwrap();
        assert_eq!(metadata["general.architecture"].as_str(), Some("llama"));
        assert_eq!(metadata["tokenizer.ggml.eos_token_id"].as_i64(), Some(1));
        assert_eq!(
            metadata["tokenizer.ggml.tokens"],
            GgufMetadataValue::Array(GgufMetadataArray::String(vec![
                "<unk>".into(),
                "hello".into()
            ]))
        );
    }
}
