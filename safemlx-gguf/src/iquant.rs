// Safe Rust translations of the scalar IQ dequantizers in llama.cpp
// `ggml/src/ggml-quants.c`, pinned at commit
// c0bc8591e8815c63cb01dd3f051a8b0df02501c9.
//
// IQ codebooks are nonlinear and are never mislabeled as MLX affine
// quantization. Model loading retains the blocks; this scalar decoder is used
// by differential tests and generic execution fallbacks.

use crate::iquant_tables::{
    IQ1S_GRID, IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID, KMASK_IQ2XS,
    KSIGNS_IQ2XS, KVALUES_IQ4NL,
};
use crate::{Endian, Error, GgmlType, Result};
use half::f16;

pub(crate) fn decode_f32(ty: GgmlType, raw: &[u8], endian: Endian) -> Result<Vec<f32>> {
    let (block_values, block_bytes) = ty.block_and_bytes()?;
    let blocks = raw.len() / usize::try_from(block_bytes).unwrap();
    let expected = blocks * usize::try_from(block_values).unwrap();
    let mut data = Vec::with_capacity(expected);
    match ty {
        GgmlType::IQ2XXS => iq2_xxs(raw, endian, &mut data),
        GgmlType::IQ2XS => iq2_xs(raw, endian, &mut data),
        GgmlType::IQ3XXS => iq3_xxs(raw, endian, &mut data),
        GgmlType::IQ1S => iq1_s(raw, endian, &mut data),
        GgmlType::IQ4NL => iq4_nl(raw, endian, &mut data),
        GgmlType::IQ3S => iq3_s(raw, endian, &mut data),
        GgmlType::IQ2S => iq2_s(raw, endian, &mut data),
        GgmlType::IQ4XS => iq4_xs(raw, endian, &mut data),
        GgmlType::IQ1M => iq1_m(raw, endian, &mut data),
        other => return Err(Error::UnsupportedTensorType(other.code())),
    }
    if data.len() != expected {
        return Err(Error::tensor(
            "<unnamed>",
            "IQ dequantization produced an inconsistent element count",
        ));
    }
    Ok(data)
}

fn emit(out: &mut Vec<f32>, value: f32) {
    out.push(value);
}

fn half(bytes: &[u8], endian: Endian) -> f32 {
    f16::from_bits(endian.u16([bytes[0], bytes[1]])).to_f32()
}

fn word(bytes: &[u8], endian: Endian) -> u16 {
    endian.u16([bytes[0], bytes[1]])
}

fn dword(bytes: &[u8], endian: Endian) -> u32 {
    match endian {
        Endian::Little => u32::from_le_bytes(bytes[..4].try_into().unwrap()),
        Endian::Big => u32::from_be_bytes(bytes[..4].try_into().unwrap()),
    }
}

fn grid8(table: &[u64], index: usize) -> [u8; 8] {
    table[index].to_le_bytes()
}

fn grid4(table: &[u32], index: usize) -> [u8; 4] {
    table[index].to_le_bytes()
}

fn signed_grid8(index: usize) -> [i8; 8] {
    IQ1S_GRID[index].to_le_bytes().map(|value| value as i8)
}

fn signed(value: f32, signs: u8, index: usize) -> f32 {
    if signs & KMASK_IQ2XS[index] != 0 {
        -value
    } else {
        value
    }
}

fn iq2_xxs(raw: &[u8], endian: Endian, out: &mut Vec<f32>) {
    for block in raw.chunks_exact(66) {
        let d = half(block, endian);
        let qs = &block[2..66];
        for ib32 in 0..8 {
            let words = std::array::from_fn::<_, 4, _>(|index| {
                let offset = 2 * (4 * ib32 + index);
                word(&qs[offset..], endian)
            });
            let aux = u32::from(words[2]) | (u32::from(words[3]) << 16);
            let db = d * (0.5 + (aux >> 28) as f32) * 0.25;
            for l in 0..4 {
                let index = if l & 1 == 0 {
                    words[l / 2] & 0xff
                } else {
                    words[l / 2] >> 8
                };
                let grid = grid8(&IQ2XXS_GRID, usize::from(index));
                let signs = KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize];
                for (j, quant) in grid.into_iter().enumerate() {
                    emit(out, signed(db * f32::from(quant), signs, j));
                }
            }
        }
    }
}

fn iq2_xs(raw: &[u8], endian: Endian, out: &mut Vec<f32>) {
    for block in raw.chunks_exact(74) {
        let d = half(block, endian);
        let qs = &block[2..66];
        let scales = &block[66..74];
        for (ib32, &scale) in scales.iter().enumerate() {
            let db = [
                d * (0.5 + f32::from(scale & 0xf)) * 0.25,
                d * (0.5 + f32::from(scale >> 4)) * 0.25,
            ];
            for l in 0..4 {
                let offset = 2 * (4 * ib32 + l);
                let quant = word(&qs[offset..], endian);
                let grid = grid8(&IQ2XS_GRID, usize::from(quant & 511));
                let signs = KSIGNS_IQ2XS[usize::from(quant >> 9)];
                for (j, value) in grid.into_iter().enumerate() {
                    emit(out, signed(db[l / 2] * f32::from(value), signs, j));
                }
            }
        }
    }
}

fn iq2_s(raw: &[u8], endian: Endian, out: &mut Vec<f32>) {
    for block in raw.chunks_exact(82) {
        let d = half(block, endian);
        let qs = &block[2..66];
        let qh = &block[66..74];
        let scales = &block[74..82];
        for ib32 in 0..8 {
            let scale = scales[ib32];
            let db = [
                d * (0.5 + f32::from(scale & 0xf)) * 0.25,
                d * (0.5 + f32::from(scale >> 4)) * 0.25,
            ];
            for l in 0..4 {
                let index = usize::from(qs[4 * ib32 + l])
                    | ((usize::from(qh[ib32]) << (8 - 2 * l)) & 0x300);
                let grid = grid8(&IQ2S_GRID, index);
                let signs = qs[32 + 4 * ib32 + l];
                for (j, value) in grid.into_iter().enumerate() {
                    emit(out, signed(db[l / 2] * f32::from(value), signs, j));
                }
            }
        }
    }
}

fn iq3_xxs(raw: &[u8], endian: Endian, out: &mut Vec<f32>) {
    for block in raw.chunks_exact(98) {
        let d = half(block, endian);
        let qs = &block[2..66];
        let scales_and_signs = &block[66..98];
        for ib32 in 0..8 {
            let aux = dword(&scales_and_signs[4 * ib32..], endian);
            let db = d * (0.5 + (aux >> 28) as f32) * 0.5;
            for l in 0..4 {
                let signs = KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize];
                let first = grid4(&IQ3XXS_GRID, usize::from(qs[8 * ib32 + 2 * l]));
                let second = grid4(&IQ3XXS_GRID, usize::from(qs[8 * ib32 + 2 * l + 1]));
                for (j, value) in first.into_iter().enumerate() {
                    emit(out, signed(db * f32::from(value), signs, j));
                }
                for (j, value) in second.into_iter().enumerate() {
                    emit(out, signed(db * f32::from(value), signs, j + 4));
                }
            }
        }
    }
}

fn iq3_s(raw: &[u8], endian: Endian, out: &mut Vec<f32>) {
    for block in raw.chunks_exact(110) {
        let d = half(block, endian);
        let qs = &block[2..66];
        let qh = &block[66..74];
        let signs = &block[74..106];
        let scales = &block[106..110];
        for pair in 0..4 {
            let scale = scales[pair];
            let db = [
                d * f32::from(1 + 2 * (scale & 0xf)),
                d * f32::from(1 + 2 * (scale >> 4)),
            ];
            for side in 0..2 {
                let q = &qs[16 * pair + 8 * side..];
                let sign = &signs[8 * pair + 4 * side..];
                let high = usize::from(qh[2 * pair + side]);
                for l in 0..4 {
                    let first_index = usize::from(q[2 * l]) | ((high << (8 - 2 * l)) & 256);
                    let second_index = usize::from(q[2 * l + 1]) | ((high << (7 - 2 * l)) & 256);
                    let first = grid4(&IQ3S_GRID, first_index);
                    let second = grid4(&IQ3S_GRID, second_index);
                    for (j, value) in first.into_iter().enumerate() {
                        emit(out, signed(db[side] * f32::from(value), sign[l], j));
                    }
                    for (j, value) in second.into_iter().enumerate() {
                        emit(out, signed(db[side] * f32::from(value), sign[l], j + 4));
                    }
                }
            }
        }
    }
}

fn iq1_s(raw: &[u8], endian: Endian, out: &mut Vec<f32>) {
    for block in raw.chunks_exact(50) {
        let d = half(block, endian);
        let qs = &block[2..34];
        let qh = &block[34..50];
        for ib in 0..8 {
            let high = word(&qh[2 * ib..], endian);
            let dl = d * f32::from(2 * ((high >> 12) & 7) + 1);
            let delta = if high & 0x8000 != 0 { -0.125 } else { 0.125 };
            for l in 0..4 {
                let index = usize::from(qs[4 * ib + l]) | (usize::from((high >> (3 * l)) & 7) << 8);
                for quant in signed_grid8(index) {
                    emit(out, dl * (f32::from(quant) + delta));
                }
            }
        }
    }
}

fn iq1_m(raw: &[u8], endian: Endian, out: &mut Vec<f32>) {
    for block in raw.chunks_exact(56) {
        let qs = &block[..32];
        let qh = &block[32..48];
        let scales = &block[48..56];
        let sc = std::array::from_fn::<_, 4, _>(|index| word(&scales[2 * index..], endian));
        let scale_bits =
            (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
        let d = f16::from_bits(scale_bits).to_f32();
        for ib in 0..8 {
            let scale_word = sc[ib / 2];
            let shift = 6 * (ib % 2);
            let dl = [
                d * f32::from(2 * ((scale_word >> shift) & 7) + 1),
                d * f32::from(2 * ((scale_word >> (shift + 3)) & 7) + 1),
            ];
            let q = &qs[4 * ib..];
            let high = &qh[2 * ib..];
            let indices = [
                usize::from(q[0]) | ((usize::from(high[0]) << 8) & 0x700),
                usize::from(q[1]) | ((usize::from(high[0]) << 4) & 0x700),
                usize::from(q[2]) | ((usize::from(high[1]) << 8) & 0x700),
                usize::from(q[3]) | ((usize::from(high[1]) << 4) & 0x700),
            ];
            let deltas = [
                if high[0] & 0x08 != 0 { -0.125 } else { 0.125 },
                if high[0] & 0x80 != 0 { -0.125 } else { 0.125 },
                if high[1] & 0x08 != 0 { -0.125 } else { 0.125 },
                if high[1] & 0x80 != 0 { -0.125 } else { 0.125 },
            ];
            for l in 0..4 {
                for quant in signed_grid8(indices[l]) {
                    emit(out, dl[l / 2] * (f32::from(quant) + deltas[l]));
                }
            }
        }
    }
}

fn iq4_nl(raw: &[u8], endian: Endian, out: &mut Vec<f32>) {
    for block in raw.chunks_exact(18) {
        let d = half(block, endian);
        for &quant in &block[2..18] {
            emit(out, d * f32::from(KVALUES_IQ4NL[usize::from(quant & 0xf)]));
        }
        for &quant in &block[2..18] {
            emit(out, d * f32::from(KVALUES_IQ4NL[usize::from(quant >> 4)]));
        }
    }
}

fn iq4_xs(raw: &[u8], endian: Endian, out: &mut Vec<f32>) {
    for block in raw.chunks_exact(136) {
        let d = half(block, endian);
        let scales_high = word(&block[2..], endian);
        let scales_low = &block[4..8];
        let qs = &block[8..136];
        for ib in 0..8 {
            let low = (scales_low[ib / 2] >> (4 * (ib % 2))) & 0xf;
            let high = ((scales_high >> (2 * ib)) & 3) as u8;
            let dl = d * f32::from(i16::from(low | (high << 4)) - 32);
            for &quant in &qs[16 * ib..16 * ib + 16] {
                emit(out, dl * f32::from(KVALUES_IQ4NL[usize::from(quant & 0xf)]));
            }
            for &quant in &qs[16 * ib..16 * ib + 16] {
                emit(out, dl * f32::from(KVALUES_IQ4NL[usize::from(quant >> 4)]));
            }
        }
    }
}
