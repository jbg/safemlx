// Conversion layout translated from MLX v0.32.0 `mlx/io/gguf_quants.cpp`
// (Apple Inc., MIT license) and safemlx's former MLX patch set. The resulting
// buffers intentionally match MLX affine quantization byte-for-byte.
use crate::{Endian, Error, GgmlType, Result, TensorDescriptor};
use half::f16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenseDtype {
    F32,
    F16,
    Bf16,
    I8,
    I16,
    I32,
    I64,
    F64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseTensor {
    pub shape: Vec<u64>,
    pub dtype: DenseDtype,
    pub data: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AffineTensor {
    pub weight_shape: Vec<u64>,
    pub scale_shape: Vec<u64>,
    pub bits: u8,
    pub group_size: u32,
    pub weights: Vec<u32>,
    /// IEEE f16 bit patterns.
    pub scales: Vec<u16>,
    /// IEEE f16 bit patterns.
    pub biases: Vec<u16>,
}

impl AffineTensor {
    /// Dequantize the affine representation using f16-rounded scales/biases.
    pub fn dequantize(&self) -> Vec<f32> {
        let count = self.scales.len() * self.group_size as usize;
        let mut out = Vec::with_capacity(count);
        let mask = (1u32 << self.bits) - 1;
        for index in 0..count {
            let bit = index * self.bits as usize;
            let word = bit / 32;
            let shift = bit % 32;
            let mut code = self.weights[word] >> shift;
            if shift + self.bits as usize > 32 {
                code |= self.weights[word + 1] << (32 - shift);
            }
            let group = index / self.group_size as usize;
            let scale = f16::from_bits(self.scales[group]).to_f32();
            let bias = f16::from_bits(self.biases[group]).to_f32();
            out.push(scale * (code & mask) as f32 + bias);
        }
        out
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConvertedTensor {
    Dense(DenseTensor),
    Affine(AffineTensor),
}

pub(crate) fn convert(
    desc: &TensorDescriptor,
    raw: &[u8],
    endian: Endian,
) -> Result<ConvertedTensor> {
    if raw.len() as u64 != desc.byte_len {
        return Err(Error::tensor(
            &desc.name,
            "payload length does not match descriptor",
        ));
    }
    if let Some(dtype) = dense_dtype(desc.ggml_type) {
        return Ok(ConvertedTensor::Dense(DenseTensor {
            shape: desc.mlx_shape(),
            dtype,
            data: normalize_dense(raw, dtype, endian),
        }));
    }
    if matches!(desc.ggml_type, GgmlType::Q5_0 | GgmlType::Q5_1) {
        return legacy_q5(desc, raw, endian).map(ConvertedTensor::Dense);
    }
    affine(desc, raw, endian).map(ConvertedTensor::Affine)
}

fn dense_dtype(ty: GgmlType) -> Option<DenseDtype> {
    Some(match ty {
        GgmlType::F32 => DenseDtype::F32,
        GgmlType::F16 => DenseDtype::F16,
        GgmlType::Bf16 => DenseDtype::Bf16,
        GgmlType::I8 => DenseDtype::I8,
        GgmlType::I16 => DenseDtype::I16,
        GgmlType::I32 => DenseDtype::I32,
        GgmlType::I64 => DenseDtype::I64,
        GgmlType::F64 => DenseDtype::F64,
        _ => return None,
    })
}

fn normalize_dense(raw: &[u8], dtype: DenseDtype, endian: Endian) -> Vec<u8> {
    let width = match dtype {
        DenseDtype::I8 => 1,
        DenseDtype::F16 | DenseDtype::Bf16 | DenseDtype::I16 => 2,
        DenseDtype::F32 | DenseDtype::I32 => 4,
        DenseDtype::I64 | DenseDtype::F64 => 8,
    };
    if width == 1
        || (cfg!(target_endian = "little") && endian == Endian::Little)
        || (cfg!(target_endian = "big") && endian == Endian::Big)
    {
        return raw.to_vec();
    }
    let mut out = raw.to_vec();
    for chunk in out.chunks_exact_mut(width) {
        chunk.reverse();
    }
    out
}

fn affine(desc: &TensorDescriptor, raw: &[u8], endian: Endian) -> Result<AffineTensor> {
    let (bits, group_size) = match desc.ggml_type {
        GgmlType::Q2K => (2, 16),
        GgmlType::Q3K => (3, 16),
        GgmlType::Q4_0 | GgmlType::Q4_1 | GgmlType::Q4K => (4, 32),
        GgmlType::Q5K => (5, 32),
        GgmlType::Q6K => (6, 16),
        GgmlType::Q8_0 => (8, 32),
        other => return Err(Error::UnsupportedTensorType(other.code())),
    };
    let mut weight_shape = desc.mlx_shape();
    let last = weight_shape
        .last_mut()
        .ok_or_else(|| Error::tensor(&desc.name, "quantized scalar is invalid"))?;
    if *last % group_size != 0 {
        return Err(Error::tensor(
            &desc.name,
            format!("last dimension is not divisible by group size {group_size}"),
        ));
    }
    *last = *last * (bits as u64) / 32;
    let mut scale_shape = desc.mlx_shape();
    *scale_shape.last_mut().unwrap() /= group_size;
    let groups = scale_shape.iter().try_fold(1u64, |a, &b| {
        a.checked_mul(b)
            .ok_or(Error::Overflow("affine group count"))
    })?;
    let words = weight_shape.iter().try_fold(1u64, |a, &b| {
        a.checked_mul(b).ok_or(Error::Overflow("affine word count"))
    })?;
    let mut out = AffineTensor {
        weight_shape,
        scale_shape,
        bits,
        group_size: group_size as u32,
        weights: Vec::with_capacity(words as usize),
        scales: Vec::with_capacity(groups as usize),
        biases: Vec::with_capacity(groups as usize),
    };
    match desc.ggml_type {
        GgmlType::Q4_0 => {
            for b in raw.chunks_exact(18) {
                let d = half(b, endian);
                out.scales.push(d);
                out.biases.push(hbits(-8.0 * f16::from_bits(d).to_f32()));
                let mut codes = [0; 32];
                for i in 0..16 {
                    codes[i] = b[2 + i] & 15;
                    codes[16 + i] = b[2 + i] >> 4;
                }
                pack(&codes, 4, &mut out.weights);
            }
        }
        GgmlType::Q4_1 => {
            for b in raw.chunks_exact(20) {
                out.scales.push(half(b, endian));
                out.biases.push(half(&b[2..], endian));
                let mut codes = [0; 32];
                for i in 0..16 {
                    codes[i] = b[4 + i] & 15;
                    codes[16 + i] = b[4 + i] >> 4;
                }
                pack(&codes, 4, &mut out.weights);
            }
        }
        GgmlType::Q8_0 => {
            for b in raw.chunks_exact(34) {
                let d = half(b, endian);
                out.scales.push(d);
                out.biases.push(hbits(-128.0 * f16::from_bits(d).to_f32()));
                let codes: Vec<_> = b[2..].iter().map(|x| x ^ 0x80).collect();
                pack(&codes, 8, &mut out.weights);
            }
        }
        GgmlType::Q4K | GgmlType::Q5K => {
            q45k(raw, endian, desc.ggml_type == GgmlType::Q5K, &mut out)
        }
        GgmlType::Q6K => q6k(raw, endian, &mut out),
        GgmlType::Q2K => q2k(raw, endian, &mut out),
        GgmlType::Q3K => q3k(raw, endian, &mut out),
        _ => unreachable!(),
    }
    if out.weights.len() as u64 != words
        || out.scales.len() as u64 != groups
        || out.biases.len() != out.scales.len()
    {
        return Err(Error::tensor(
            &desc.name,
            "conversion produced an inconsistent affine shape",
        ));
    }
    Ok(out)
}

fn half(b: &[u8], e: Endian) -> u16 {
    e.u16([b[0], b[1]])
}
fn hbits(v: f32) -> u16 {
    f16::from_f32(v).to_bits()
}

fn pack(codes: &[u8], bits: u8, out: &mut Vec<u32>) {
    let words = (codes.len() * bits as usize).div_ceil(32);
    let start = out.len();
    out.resize(start + words, 0);
    let mask = (1u32 << bits) - 1;
    for (i, &c) in codes.iter().enumerate() {
        let off = i * bits as usize;
        let w = off / 32;
        let s = off % 32;
        out[start + w] |= ((c as u32) & mask) << s;
        if s + bits as usize > 32 {
            out[start + w + 1] |= (c as u32) >> (32 - s);
        }
    }
}

fn scale_min(s: &[u8], i: usize) -> (u8, u8) {
    if i < 4 {
        (s[i] & 63, s[i + 4] & 63)
    } else {
        (
            (s[i + 4] & 15) | ((s[i - 4] >> 6) << 4),
            (s[i + 4] >> 4) | ((s[i] >> 6) << 4),
        )
    }
}
fn q45k(raw: &[u8], e: Endian, is_q5: bool, out: &mut AffineTensor) {
    let size = if is_q5 { 176 } else { 144 };
    for b in raw.chunks_exact(size) {
        let d = f16::from_bits(half(b, e)).to_f32();
        let dm = f16::from_bits(half(&b[2..], e)).to_f32();
        let s = &b[4..16];
        let qh = if is_q5 { Some(&b[16..48]) } else { None };
        let qs = if is_q5 { &b[48..] } else { &b[16..] };
        for g in 0..8 {
            let (sc, m) = scale_min(s, g);
            out.scales.push(hbits(d * sc as f32));
            out.biases.push(hbits(-dm * m as f32));
            let mut c = [0; 32];
            for i in 0..32 {
                let p = qs[(g / 2) * 32 + i];
                let lo = if g % 2 == 0 { p & 15 } else { p >> 4 };
                let hi = if qh.is_some_and(|h| h[i] & (1 << g) != 0) {
                    16
                } else {
                    0
                };
                c[i] = lo | hi;
            }
            pack(&c, if is_q5 { 5 } else { 4 }, &mut out.weights);
        }
    }
}

fn q6k(raw: &[u8], e: Endian, out: &mut AffineTensor) {
    for b in raw.chunks_exact(210) {
        let d = f16::from_bits(half(&b[208..], e)).to_f32();
        for section in 0..2 {
            let ql = &b[section * 64..];
            let qh = &b[128 + section * 32..];
            let scales = &b[192 + section * 8..];
            let mut vals = [0; 128];
            for i in 0..32 {
                vals[i] = (ql[i] & 15) | ((qh[i] & 3) << 4);
                vals[i + 32] = (ql[i + 32] & 15) | (((qh[i] >> 2) & 3) << 4);
                vals[i + 64] = (ql[i] >> 4) | (((qh[i] >> 4) & 3) << 4);
                vals[i + 96] = (ql[i + 32] >> 4) | (((qh[i] >> 6) & 3) << 4);
            }
            for g in 0..8 {
                let sc = d * (scales[g] as i8) as f32;
                out.scales.push(hbits(sc));
                out.biases.push(hbits(-32.0 * sc));
                pack(&vals[g * 16..g * 16 + 16], 6, &mut out.weights);
            }
        }
    }
}

fn q2k(raw: &[u8], e: Endian, out: &mut AffineTensor) {
    for b in raw.chunks_exact(84) {
        let d = f16::from_bits(half(&b[80..], e)).to_f32();
        let dm = f16::from_bits(half(&b[82..], e)).to_f32();
        let mut all = [0; 256];
        for g in 0..16 {
            let s = b[g];
            out.scales.push(hbits(d * (s & 15) as f32));
            out.biases.push(hbits(-dm * (s >> 4) as f32));
            let qo = (g / 8) * 32 + (g % 2) * 16;
            let shift = ((g % 8) / 2) * 2;
            for i in 0..16 {
                all[g * 16 + i] = (b[16 + qo + i] >> shift) & 3;
            }
        }
        pack(&all, 2, &mut out.weights);
    }
}

fn q3k(raw: &[u8], e: Endian, out: &mut AffineTensor) {
    for b in raw.chunks_exact(110) {
        let hm = &b[..32];
        let qs = &b[32..96];
        let src = &b[96..108];
        let mut enc = [0u8; 16];
        for i in 0..4 {
            enc[i] = (src[i] & 15) | ((src[8 + i] & 3) << 4);
            enc[4 + i] = (src[4 + i] & 15) | (((src[8 + i] >> 2) & 3) << 4);
            enc[8 + i] = (src[i] >> 4) | (((src[8 + i] >> 4) & 3) << 4);
            enc[12 + i] = (src[4 + i] >> 4) | (((src[8 + i] >> 6) & 3) << 4);
        }
        let d = f16::from_bits(half(&b[108..], e)).to_f32();
        let mut all = [0; 256];
        for g in 0..16 {
            let sc = d * (enc[g] as i32 - 32) as f32;
            out.scales.push(hbits(sc));
            out.biases.push(hbits(-4.0 * sc));
            let qo = (g / 8) * 32 + (g % 2) * 16;
            let shift = ((g % 8) / 2) * 2;
            let mask = 1 << (g / 2);
            for i in 0..16 {
                all[g * 16 + i] = ((qs[qo + i] >> shift) & 3)
                    | if hm[(g % 2) * 16 + i] & mask != 0 {
                        4
                    } else {
                        0
                    };
            }
        }
        pack(&all, 3, &mut out.weights);
    }
}

fn legacy_q5(desc: &TensorDescriptor, raw: &[u8], e: Endian) -> Result<DenseTensor> {
    let mut data = Vec::with_capacity(desc.element_count()? as usize * 2);
    let size = if desc.ggml_type == GgmlType::Q5_0 {
        22
    } else {
        24
    };
    for b in raw.chunks_exact(size) {
        let d = f16::from_bits(half(b, e)).to_f32();
        let m = if size == 24 {
            f16::from_bits(half(&b[2..], e)).to_f32()
        } else {
            0.0
        };
        let qh_off = if size == 22 { 2 } else { 4 };
        let qs_off = if size == 22 { 6 } else { 8 };
        let qh = e.u32(b[qh_off..qh_off + 4].try_into().unwrap());
        let mut values = [0i32; 32];
        for j in 0..16 {
            values[j] = ((b[qs_off + j] & 15) | ((((qh >> j) << 4) & 16) as u8)) as i32
                - (if size == 22 { 16 } else { 0 });
            values[j + 16] = ((b[qs_off + j] >> 4) | (((qh >> (j + 12)) & 16) as u8)) as i32
                - (if size == 22 { 16 } else { 0 });
        }
        for q in values {
            let value = if size == 22 {
                q as f32 * d
            } else {
                q as f32 * d + m
            };
            data.extend_from_slice(&hbits(value).to_ne_bytes());
        }
    }
    Ok(DenseTensor {
        shape: desc.mlx_shape(),
        dtype: DenseDtype::F16,
        data,
    })
}
