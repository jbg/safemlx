use crate::format::{align_up, Endian, GgmlType, MetadataArray, MetadataValue, DEFAULT_ALIGNMENT};
use crate::{Error, Result};
use std::collections::{BTreeMap, HashSet};
use std::io::{Seek, SeekFrom, Write};

#[derive(Debug, Clone, Copy)]
pub struct WriterOptions {
    pub version: u32,
    pub endian: Endian,
    pub alignment: u64,
}
impl Default for WriterOptions {
    fn default() -> Self {
        Self {
            version: 3,
            endian: Endian::Little,
            alignment: DEFAULT_ALIGNMENT,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TensorInput<'a> {
    pub name: &'a str,
    pub dimensions: &'a [u64],
    pub ggml_type: GgmlType,
    pub data: &'a [u8],
}

pub struct Writer {
    options: WriterOptions,
}
impl Default for Writer {
    fn default() -> Self {
        Self::new(WriterOptions::default()).expect("default GGUF options are valid")
    }
}

impl Writer {
    pub fn new(options: WriterOptions) -> Result<Self> {
        if !(1..=3).contains(&options.version) {
            return Err(Error::UnsupportedVersion(options.version));
        }
        if options.alignment == 0 || !options.alignment.is_power_of_two() {
            return Err(Error::InvalidHeader(format!(
                "invalid alignment {}",
                options.alignment
            )));
        }
        Ok(Self { options })
    }

    /// Write a deterministic GGUF file without buffering tensor payloads.
    pub fn write<W: Write + Seek>(
        &self,
        mut out: W,
        metadata: &BTreeMap<String, MetadataValue>,
        tensors: &[TensorInput<'_>],
    ) -> Result<()> {
        let o = self.options;
        let mut metadata = metadata.clone();
        match metadata.get("general.alignment") {
            None => {
                if o.alignment != DEFAULT_ALIGNMENT {
                    let value = u32::try_from(o.alignment)
                        .map(MetadataValue::Uint32)
                        .unwrap_or(MetadataValue::Uint64(o.alignment));
                    metadata.insert("general.alignment".into(), value);
                }
            }
            Some(MetadataValue::Uint32(v)) if u64::from(*v) == o.alignment => {}
            Some(MetadataValue::Uint64(v)) if *v == o.alignment => {}
            Some(_) => {
                return Err(Error::InvalidMetadata {
                    key: "general.alignment".into(),
                    reason: "does not match writer alignment".into(),
                })
            }
        }
        let mut names = HashSet::new();
        let mut offsets = Vec::with_capacity(tensors.len());
        let mut next = 0u64;
        for t in tensors {
            if t.name.is_empty() {
                return Err(Error::tensor(t.name, "empty name"));
            }
            if !names.insert(t.name) {
                return Err(Error::DuplicateTensor(t.name.into()));
            }
            let elements = t.dimensions.iter().try_fold(1u64, |a, &b| {
                a.checked_mul(b).ok_or(Error::Overflow("tensor elements"))
            })?;
            let (block, bytes) = t.ggml_type.block_and_bytes()?;
            if elements != 0
                && (t.dimensions.first().copied().unwrap_or(1) % block != 0
                    || elements % block != 0)
            {
                return Err(Error::tensor(
                    t.name,
                    format!("shape is not divisible by block size {block}"),
                ));
            }
            let expected = (elements / block)
                .checked_mul(bytes)
                .ok_or(Error::Overflow("tensor bytes"))?;
            if expected != t.data.len() as u64 {
                return Err(Error::tensor(
                    t.name,
                    format!("payload has {} bytes, expected {expected}", t.data.len()),
                ));
            }
            next = align_up(next, o.alignment)?;
            offsets.push(next);
            next = next
                .checked_add(expected)
                .ok_or(Error::Overflow("tensor offsets"))?;
        }
        out.write_all(match o.endian {
            Endian::Little => b"GGUF",
            Endian::Big => b"FUGG",
        })
        .map_err(|source| Error::Io { offset: 0, source })?;
        let mut enc = Encoder {
            out: &mut out,
            endian: o.endian,
            version: o.version,
        };
        enc.u32(o.version)?;
        enc.count(tensors.len() as u64)?;
        enc.count(metadata.len() as u64)?;
        for (k, v) in &metadata {
            if k.len() > u16::MAX as usize || !k.is_ascii() {
                return Err(Error::InvalidMetadata {
                    key: k.clone(),
                    reason: "key must be ASCII and at most 65535 bytes".into(),
                });
            }
            enc.string(k)?;
            enc.u32(v.type_code())?;
            enc.value(v)?;
        }
        for (t, &offset) in tensors.iter().zip(&offsets) {
            enc.string(t.name)?;
            enc.u32(
                t.dimensions
                    .len()
                    .try_into()
                    .map_err(|_| Error::Overflow("rank"))?,
            )?;
            for &d in t.dimensions {
                enc.dimension(d)?;
            }
            enc.u32(t.ggml_type.code())?;
            enc.u64(offset)?;
        }
        let pos = enc
            .out
            .stream_position()
            .map_err(|source| Error::Io { offset: 0, source })?;
        let data_start = align_up(pos, o.alignment)?;
        extend_with_zeros(enc.out, data_start - pos)?;
        let mut relative = 0u64;
        for (t, &offset) in tensors.iter().zip(&offsets) {
            extend_with_zeros(enc.out, offset - relative)?;
            enc.out.write_all(t.data).map_err(|source| Error::Io {
                offset: data_start + offset,
                source,
            })?;
            relative = offset + t.data.len() as u64;
        }
        Ok(())
    }
}

fn extend_with_zeros<W: Write + Seek>(out: &mut W, mut n: u64) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    while n > i64::MAX as u64 {
        out.seek(SeekFrom::Current(i64::MAX))
            .map_err(|source| Error::Io { offset: 0, source })?;
        n -= i64::MAX as u64;
    }
    if n > 1 {
        out.seek(SeekFrom::Current((n - 1) as i64))
            .map_err(|source| Error::Io { offset: 0, source })?;
    }
    out.write_all(&[0])
        .map_err(|source| Error::Io { offset: 0, source })?;
    Ok(())
}

struct Encoder<'a, W> {
    out: &'a mut W,
    endian: Endian,
    version: u32,
}
impl<W: Write> Encoder<'_, W> {
    fn u8(&mut self, v: u8) -> Result<()> {
        self.bytes(&[v])
    }
    fn u16(&mut self, v: u16) -> Result<()> {
        self.bytes(&self.endian.put_u16(v))
    }
    fn u32(&mut self, v: u32) -> Result<()> {
        self.bytes(&self.endian.put_u32(v))
    }
    fn u64(&mut self, v: u64) -> Result<()> {
        self.bytes(&self.endian.put_u64(v))
    }
    fn bytes(&mut self, v: &[u8]) -> Result<()> {
        self.out
            .write_all(v)
            .map_err(|source| Error::Io { offset: 0, source })
    }
    fn count(&mut self, v: u64) -> Result<()> {
        if self.version == 1 {
            self.u32(v.try_into().map_err(|_| Error::Overflow("v1 count"))?)
        } else {
            self.u64(v)
        }
    }
    fn dimension(&mut self, v: u64) -> Result<()> {
        self.count(v)
    }
    fn string(&mut self, v: &str) -> Result<()> {
        self.count(v.len() as u64)?;
        self.bytes(v.as_bytes())
    }
    fn value(&mut self, v: &MetadataValue) -> Result<()> {
        match v {
            MetadataValue::Uint8(x) => self.u8(*x),
            MetadataValue::Int8(x) => self.u8(*x as u8),
            MetadataValue::Uint16(x) => self.u16(*x),
            MetadataValue::Int16(x) => self.u16(*x as u16),
            MetadataValue::Uint32(x) => self.u32(*x),
            MetadataValue::Int32(x) => self.u32(*x as u32),
            MetadataValue::Float32(x) => self.u32(x.to_bits()),
            MetadataValue::Bool(x) => self.u8(*x as u8),
            MetadataValue::String(x) => self.string(x),
            MetadataValue::Array(x) => self.array(x),
            MetadataValue::Uint64(x) => self.u64(*x),
            MetadataValue::Int64(x) => self.u64(*x as u64),
            MetadataValue::Float64(x) => self.u64(x.to_bits()),
        }
    }
    fn array(&mut self, a: &MetadataArray) -> Result<()> {
        self.u32(a.type_code())?;
        self.count(a.len() as u64)?;
        match a {
            MetadataArray::Uint8(v) => self.bytes(v),
            MetadataArray::Int8(v) => {
                self.bytes(unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), v.len()) })
            }
            MetadataArray::Uint16(v) => {
                for x in v {
                    self.u16(*x)?;
                }
                Ok(())
            }
            MetadataArray::Int16(v) => {
                for x in v {
                    self.u16(*x as u16)?;
                }
                Ok(())
            }
            MetadataArray::Uint32(v) => {
                for x in v {
                    self.u32(*x)?;
                }
                Ok(())
            }
            MetadataArray::Int32(v) => {
                for x in v {
                    self.u32(*x as u32)?;
                }
                Ok(())
            }
            MetadataArray::Float32(v) => {
                for x in v {
                    self.u32(x.to_bits())?;
                }
                Ok(())
            }
            MetadataArray::Bool(v) => {
                for x in v {
                    self.u8(*x as u8)?;
                }
                Ok(())
            }
            MetadataArray::String(v) => {
                for x in v {
                    self.string(x)?;
                }
                Ok(())
            }
            MetadataArray::Array(v) => {
                for x in v {
                    self.array(x)?;
                }
                Ok(())
            }
            MetadataArray::Uint64(v) => {
                for x in v {
                    self.u64(*x)?;
                }
                Ok(())
            }
            MetadataArray::Int64(v) => {
                for x in v {
                    self.u64(*x as u64)?;
                }
                Ok(())
            }
            MetadataArray::Float64(v) => {
                for x in v {
                    self.u64(x.to_bits())?;
                }
                Ok(())
            }
        }
    }
}
