//! Shared PCM validation and log-mel feature extraction.

use rustfft::{num_complex::Complex32, FftPlanner};

use crate::error::Error;

/// Borrowed mono floating-point PCM waveform.
#[derive(Debug, Clone, Copy)]
pub struct AudioWaveform<'a> {
    samples: &'a [f32],
    sample_rate: u32,
}

impl<'a> AudioWaveform<'a> {
    /// Validates and creates a mono PCM waveform.
    pub fn new(samples: &'a [f32], sample_rate: u32) -> Result<Self, Error> {
        if samples.is_empty() {
            return Err(Error::Processor("audio waveform must not be empty".into()));
        }
        if sample_rate == 0 {
            return Err(Error::Processor(
                "audio sample rate must be positive".into(),
            ));
        }
        if samples.iter().any(|sample| !sample.is_finite()) {
            return Err(Error::Processor(
                "audio waveform samples must all be finite".into(),
            ));
        }
        Ok(Self {
            samples,
            sample_rate,
        })
    }

    /// Returns the PCM samples.
    pub fn samples(self) -> &'a [f32] {
        self.samples
    }

    /// Returns the sampling rate in hertz.
    pub fn sample_rate(self) -> u32 {
        self.sample_rate
    }
}

/// Model-independent log-mel extraction parameters.
#[derive(Debug, Clone)]
pub struct LogMelConfig {
    /// Required input sampling rate.
    pub sample_rate: u32,
    /// Analysis frame length in samples.
    pub frame_length: usize,
    /// Frame step in samples.
    pub hop_length: usize,
    /// FFT length, at least `frame_length`.
    pub fft_length: usize,
    /// Number of HTK mel filters.
    pub mel_bins: usize,
    /// Lowest filter frequency.
    pub min_frequency: f32,
    /// Highest filter frequency.
    pub max_frequency: f32,
    /// Additive floor before taking the natural logarithm.
    pub mel_floor: f32,
    /// Maximum waveform length before truncation.
    pub max_samples: usize,
    /// Waveform padding multiple.
    pub pad_to_multiple: usize,
}

/// Owned model-ready features and their valid-frame mask.
#[derive(Debug, Clone)]
pub struct LogMelFeatures {
    /// Row-major `[frames, mel_bins]` feature values.
    pub values: Vec<f32>,
    /// True for frames whose analysis endpoint is real audio.
    pub mask: Vec<bool>,
    /// Number of feature frames.
    pub frames: usize,
    /// Number of mel bins.
    pub mel_bins: usize,
}

/// Extracts semicausal periodic-Hann HTK log-mel features.
pub fn extract_log_mel(
    waveform: AudioWaveform<'_>,
    config: &LogMelConfig,
) -> Result<LogMelFeatures, Error> {
    if waveform.sample_rate != config.sample_rate {
        return Err(Error::Processor(format!(
            "audio processor requires {} Hz PCM, got {} Hz",
            config.sample_rate, waveform.sample_rate
        )));
    }
    if config.frame_length == 0
        || config.hop_length == 0
        || config.fft_length < config.frame_length
        || config.mel_bins == 0
        || config.pad_to_multiple == 0
    {
        return Err(Error::Processor(
            "invalid log-mel processor configuration".into(),
        ));
    }

    let real_samples = waveform.samples.len().min(config.max_samples);
    let padded_samples = real_samples.div_ceil(config.pad_to_multiple) * config.pad_to_multiple;
    let left_padding = config.frame_length / 2;
    let frame_span = config.frame_length + 1;
    let total = left_padding + padded_samples;
    let frames = total.saturating_sub(frame_span) / config.hop_length + 1;
    let mut padded = vec![0.0f32; total];
    padded[left_padding..left_padding + real_samples]
        .copy_from_slice(&waveform.samples[..real_samples]);

    let window = (0..config.frame_length)
        .map(|index| {
            0.5 - 0.5
                * (2.0 * std::f32::consts::PI * index as f32 / config.frame_length as f32).cos()
        })
        .collect::<Vec<_>>();
    let mel_filters = htk_mel_filters(config);
    let frequency_bins = config.fft_length / 2 + 1;
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(config.fft_length);
    let mut spectrum = vec![Complex32::default(); config.fft_length];
    let mut values = vec![0.0f32; frames * config.mel_bins];
    let mut mask = vec![false; frames];

    for frame in 0..frames {
        spectrum.fill(Complex32::default());
        let start = frame * config.hop_length;
        for index in 0..config.frame_length {
            spectrum[index].re = padded[start + index] * window[index];
        }
        fft.process(&mut spectrum);
        for mel in 0..config.mel_bins {
            let mut magnitude = 0.0f32;
            for frequency in 0..frequency_bins {
                magnitude +=
                    spectrum[frequency].norm() * mel_filters[frequency * config.mel_bins + mel];
            }
            values[frame * config.mel_bins + mel] = (magnitude + config.mel_floor).ln();
        }
        let endpoint = start + frame_span - 1;
        mask[frame] = endpoint >= left_padding && endpoint < left_padding + real_samples;
        if !mask[frame] {
            values[frame * config.mel_bins..(frame + 1) * config.mel_bins].fill(0.0);
        }
    }

    Ok(LogMelFeatures {
        values,
        mask,
        frames,
        mel_bins: config.mel_bins,
    })
}

fn htk_mel_filters(config: &LogMelConfig) -> Vec<f32> {
    let frequency_bins = config.fft_length / 2 + 1;
    let hertz_to_mel = |frequency: f32| 2595.0 * (1.0 + frequency / 700.0).log10();
    let mel_to_hertz = |mel: f32| 700.0 * (10.0f32.powf(mel / 2595.0) - 1.0);
    let mel_min = hertz_to_mel(config.min_frequency);
    let mel_max = hertz_to_mel(config.max_frequency);
    let centers = (0..config.mel_bins + 2)
        .map(|index| {
            let mel = mel_min + (mel_max - mel_min) * index as f32 / (config.mel_bins + 1) as f32;
            mel_to_hertz(mel)
        })
        .collect::<Vec<_>>();
    let mut filters = vec![0.0f32; frequency_bins * config.mel_bins];
    for frequency in 0..frequency_bins {
        let hertz =
            config.sample_rate as f32 * 0.5 * frequency as f32 / (frequency_bins - 1) as f32;
        for mel in 0..config.mel_bins {
            let down = (hertz - centers[mel]) / (centers[mel + 1] - centers[mel]);
            let up = (centers[mel + 2] - hertz) / (centers[mel + 2] - centers[mel + 1]);
            filters[frequency * config.mel_bins + mel] = down.min(up).max(0.0);
        }
    }
    filters
}

#[cfg(test)]
mod tests {
    use super::{extract_log_mel, AudioWaveform, LogMelConfig};

    #[test]
    fn validates_waveform_and_extracts_masked_features() {
        let samples = vec![0.0f32; 16_000];
        let waveform = AudioWaveform::new(&samples, 16_000).unwrap();
        let features = extract_log_mel(
            waveform,
            &LogMelConfig {
                sample_rate: 16_000,
                frame_length: 320,
                hop_length: 160,
                fft_length: 512,
                mel_bins: 128,
                min_frequency: 0.0,
                max_frequency: 8_000.0,
                mel_floor: 1e-3,
                max_samples: 480_000,
                pad_to_multiple: 128,
            },
        )
        .unwrap();
        assert_eq!(features.frames, 99);
        assert_eq!(features.values.len(), 99 * 128);
        assert!(features.mask.iter().all(|valid| *valid));
        assert!(features.values.iter().all(|value| value.is_finite()));
    }
}
