use crate::error::IoError;
use std::collections::HashMap;
use std::io::{Cursor, Read};
use std::ops::{Deref, DerefMut};
use std::path::Path;

pub use safemlx_gguf::{MetadataArray as GgufMetadataArray, MetadataValue as GgufMetadataValue};

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
