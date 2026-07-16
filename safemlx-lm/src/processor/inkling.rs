//! Thinking Machines Lab Inkling image and dMel preprocessing.

use std::path::Path;

use safemlx::Array;
use serde::Deserialize;

use crate::{error::Error, models::input::Modality};

use super::{
    prepared_model_input, push_text_token_ids, MediaInput, MediaPayload, OwnedInputMetadata,
    PreparedInputPart, PreparedModelInput, ProcessorInput,
};

#[derive(Debug, Clone)]
pub(super) struct InklingProcessor {
    #[cfg(feature = "image-processing")]
    image_bos_token_id: u32,
    #[cfg(feature = "audio-processing")]
    audio_bos_token_id: u32,
    #[cfg(feature = "audio-processing")]
    dmel_bins: usize,
    #[cfg(feature = "audio-processing")]
    dmel_min: f32,
    #[cfg(feature = "audio-processing")]
    dmel_max: f32,
}

impl InklingProcessor {
    pub(super) fn load(model_dir: &Path) -> Result<Option<Self>, Error> {
        #[derive(Deserialize)]
        struct Config {
            model_type: String,
            #[cfg(feature = "image-processing")]
            #[serde(default = "default_image_bos")]
            image_bos_token_id: u32,
            #[cfg(feature = "audio-processing")]
            #[serde(default = "default_audio_bos")]
            audio_bos_token_id: u32,
            #[cfg(feature = "audio-processing")]
            #[serde(default)]
            audio_config: Option<AudioConfig>,
        }
        #[cfg(feature = "audio-processing")]
        #[derive(Deserialize)]
        struct AudioConfig {
            #[serde(default = "default_dmel_bins")]
            mel_vocab_size: usize,
            #[serde(default = "default_dmel_min")]
            dmel_min_value: f32,
            #[serde(default = "default_dmel_max")]
            dmel_max_value: f32,
        }
        let config: Config =
            serde_json::from_slice(&std::fs::read(model_dir.join("config.json"))?)?;
        if config.model_type != "inkling_mm_model" {
            return Ok(None);
        }
        #[cfg(feature = "audio-processing")]
        let audio = config.audio_config.unwrap_or(AudioConfig {
            mel_vocab_size: default_dmel_bins(),
            dmel_min_value: default_dmel_min(),
            dmel_max_value: default_dmel_max(),
        });
        Ok(Some(Self {
            #[cfg(feature = "image-processing")]
            image_bos_token_id: config.image_bos_token_id,
            #[cfg(feature = "audio-processing")]
            audio_bos_token_id: config.audio_bos_token_id,
            #[cfg(feature = "audio-processing")]
            dmel_bins: audio.mel_vocab_size,
            #[cfg(feature = "audio-processing")]
            dmel_min: audio.dmel_min_value,
            #[cfg(feature = "audio-processing")]
            dmel_max: audio.dmel_max_value,
        }))
    }

    pub(super) fn prepare_input(
        &self,
        input: &[ProcessorInput<'_>],
        encode_text: &mut dyn FnMut(&str) -> Result<Vec<u32>, Error>,
    ) -> Result<PreparedModelInput, Error> {
        let mut parts = Vec::new();
        for item in input {
            match *item {
                ProcessorInput::Text(text) => {
                    push_text_token_ids(&mut parts, &encode_text(text)?);
                }
                ProcessorInput::TokenIds(ids) => push_text_token_ids(&mut parts, ids),
                ProcessorInput::Media(media) => self.push_media(&mut parts, media)?,
            }
        }
        prepared_model_input(parts)
    }

    fn push_media(
        &self,
        parts: &mut Vec<PreparedInputPart>,
        media: MediaInput<'_>,
    ) -> Result<(), Error> {
        match (media.modality, media.payload) {
            #[cfg(feature = "image-processing")]
            (Modality::Image, MediaPayload::Rgb8(image)) => {
                push_text_token_ids(parts, &[self.image_bos_token_id]);
                parts.push(process_image(image)?);
                Ok(())
            }
            #[cfg(feature = "audio-processing")]
            (Modality::Audio, MediaPayload::AudioF32(waveform)) => {
                push_text_token_ids(parts, &[self.audio_bos_token_id]);
                parts.push(self.process_audio(waveform)?);
                Ok(())
            }
            _ => Err(Error::Processor(format!(
                "Inkling processor does not support {} media with the enabled features",
                media.modality.as_str()
            ))),
        }
    }

    #[cfg(feature = "audio-processing")]
    fn process_audio(
        &self,
        waveform: super::audio::AudioWaveform<'_>,
    ) -> Result<PreparedInputPart, Error> {
        let features = inkling_log_mel(waveform)?;
        let span = (self.dmel_max - self.dmel_min) as f64;
        if self.dmel_bins < 2 || span <= 0.0 {
            return Err(Error::Processor(
                "invalid Inkling dMel bin configuration".into(),
            ));
        }
        let centers = (0..self.dmel_bins)
            .map(|index| self.dmel_min as f64 + span * index as f64 / (self.dmel_bins - 1) as f64)
            .collect::<Vec<_>>();
        let ids = features
            .iter()
            .map(|value| {
                let value = (*value as f64).clamp(self.dmel_min as f64, self.dmel_max as f64);
                centers
                    .iter()
                    .enumerate()
                    .min_by(|(_, left), (_, right)| {
                        (value - **left).abs().total_cmp(&(value - **right).abs())
                    })
                    .map(|(index, _)| index as i32)
                    .unwrap_or(0)
            })
            .collect::<Vec<_>>();
        let frames = ids.len() / 80;
        let tensor = Array::from_slice(&ids, &[1, frames as i32, 80]);
        let mask = Array::from_slice(&vec![true; frames], &[1, frames as i32]);
        Ok(PreparedInputPart::media_tensor(
            Modality::Audio,
            tensor,
            OwnedInputMetadata::AudioMask(mask),
        ))
    }
}

#[cfg(feature = "image-processing")]
fn default_image_bos() -> u32 {
    200_005
}

#[cfg(feature = "audio-processing")]
fn default_audio_bos() -> u32 {
    200_020
}

#[cfg(feature = "audio-processing")]
fn default_dmel_bins() -> usize {
    16
}

#[cfg(feature = "audio-processing")]
fn default_dmel_min() -> f32 {
    -7.0
}

#[cfg(feature = "audio-processing")]
fn default_dmel_max() -> f32 {
    2.0
}

#[cfg(feature = "image-processing")]
fn process_image(image: super::image::RgbImageView<'_>) -> Result<PreparedInputPart, Error> {
    const PATCH: usize = 40;
    const MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
    const STD: [f32; 3] = [0.268_629_54, 0.261_302_6, 0.275_777_1];
    let width = image.width() as usize;
    let height = image.height() as usize;
    let pixels = image.packed_pixels();
    let (rows, cols) = image_patch_grid(height, width);
    let mut output = Vec::with_capacity(rows * cols * 2 * PATCH * PATCH * 3);
    for row in 0..rows {
        for col in 0..cols {
            let mut patch = vec![0.0f32; 2 * PATCH * PATCH * 3];
            for time in 0..2 {
                for y in 0..PATCH {
                    for x in 0..PATCH {
                        let source_y = row * PATCH + y;
                        let source_x = col * PATCH + x;
                        for channel in 0..3 {
                            let raw = if source_y < height && source_x < width {
                                pixels[(source_y * width + source_x) * 3 + channel] as f32
                            } else {
                                -1.0
                            };
                            let normalized = (raw / 255.0 - MEAN[channel]) / STD[channel];
                            patch[((time * PATCH + y) * PATCH + x) * 3 + channel] = normalized;
                        }
                    }
                }
            }
            output.extend(patch);
        }
    }
    Ok(PreparedInputPart::media_tensor(
        Modality::Image,
        Array::from_slice(&output, &[(rows * cols) as i32, 2, 40, 40, 3]),
        OwnedInputMetadata::None,
    ))
}

#[cfg(feature = "image-processing")]
fn image_patch_grid(height: usize, width: usize) -> (usize, usize) {
    // The reference deliberately includes a final partial/empty column when
    // width is an exact multiple of the patch size.
    (height.div_ceil(40), width / 40 + 1)
}

#[cfg(feature = "audio-processing")]
fn inkling_log_mel(waveform: super::audio::AudioWaveform<'_>) -> Result<Vec<f32>, Error> {
    use rustfft::{num_complex::Complex32, FftPlanner};

    const SAMPLE_RATE: u32 = 16_000;
    const FFT: usize = 1_600;
    const HOP: usize = 800;
    const MELS: usize = 80;
    if waveform.sample_rate() != SAMPLE_RATE {
        return Err(Error::Processor(format!(
            "Inkling audio requires {SAMPLE_RATE} Hz PCM, got {} Hz",
            waveform.sample_rate()
        )));
    }
    let samples = waveform.samples();
    let frames = samples.len().div_ceil(HOP);
    let mut padded = vec![0.0f32; HOP + frames * HOP];
    padded[HOP..HOP + samples.len()].copy_from_slice(samples);
    let filters = slaney_mel_filters(FFT, SAMPLE_RATE as usize, MELS);
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(FFT);
    let mut spectrum = vec![Complex32::default(); FFT];
    let mut output = vec![0.0f32; frames * MELS];
    for frame in 0..frames {
        spectrum.fill(Complex32::default());
        let start = frame * HOP;
        for index in 0..FFT {
            let window = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * index as f32 / FFT as f32).cos();
            spectrum[index].re = padded[start + index] * window;
        }
        fft.process(&mut spectrum);
        for mel in 0..MELS {
            let mut energy = 0.0f32;
            for frequency in 0..=FFT / 2 {
                energy += spectrum[frequency].norm() * filters[mel * (FFT / 2 + 1) + frequency];
            }
            output[frame * MELS + mel] = energy.max(1e-10).log10();
        }
    }
    Ok(output)
}

#[cfg(feature = "audio-processing")]
fn slaney_mel_filters(fft: usize, sample_rate: usize, mel_bins: usize) -> Vec<f32> {
    let hz_to_mel = |hz: f64| {
        if hz < 1_000.0 {
            hz / (200.0 / 3.0)
        } else {
            15.0 + (hz / 1_000.0).ln() / (6.4f64.ln() / 27.0)
        }
    };
    let mel_to_hz = |mel: f64| {
        if mel < 15.0 {
            mel * (200.0 / 3.0)
        } else {
            1_000.0 * ((mel - 15.0) * (6.4f64.ln() / 27.0)).exp()
        }
    };
    let mel_max = hz_to_mel(sample_rate as f64 / 2.0);
    let edges = (0..mel_bins + 2)
        .map(|index| mel_to_hz(mel_max * index as f64 / (mel_bins + 1) as f64))
        .collect::<Vec<_>>();
    let frequency_bins = fft / 2 + 1;
    let mut filters = vec![0.0f32; mel_bins * frequency_bins];
    for mel in 0..mel_bins {
        let normalization = 2.0 / (edges[mel + 2] - edges[mel]);
        for frequency in 0..frequency_bins {
            let hz = sample_rate as f64 * frequency as f64 / fft as f64;
            let lower = (hz - edges[mel]) / (edges[mel + 1] - edges[mel]);
            let upper = (edges[mel + 2] - hz) / (edges[mel + 2] - edges[mel + 1]);
            filters[mel * frequency_bins + frequency] =
                (lower.min(upper).max(0.0) * normalization) as f32;
        }
    }
    filters
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "image-processing")]
    #[test]
    fn exact_patch_width_keeps_reference_extra_column() {
        assert_eq!(super::image_patch_grid(40, 40), (1, 2));
        assert_eq!(super::image_patch_grid(41, 39), (2, 1));
    }

    #[cfg(feature = "audio-processing")]
    #[test]
    fn dmel_frontend_uses_fifty_millisecond_frames() {
        let samples = vec![0.0f32; 801];
        let waveform = crate::processor::AudioWaveform::new(&samples, 16_000).unwrap();
        let features = super::inkling_log_mel(waveform).unwrap();
        assert_eq!(features.len(), 2 * 80);
        assert!(features.iter().all(|value| (*value + 10.0).abs() < 1e-6));
    }
}
