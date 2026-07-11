//! Shared decoded-video validation, sampling, and timing operations.

use crate::{error::Error, processor::RgbImageView};

/// Validates that a decoded frame sequence is non-empty and has stable dimensions.
pub fn validate_rgb_frames(frames: &[RgbImageView<'_>]) -> Result<(u32, u32), Error> {
    let first = frames
        .first()
        .ok_or_else(|| Error::Processor("video must contain at least one frame".to_string()))?;
    let dimensions = (first.width(), first.height());
    if let Some((index, frame)) = frames
        .iter()
        .enumerate()
        .find(|(_, frame)| (frame.width(), frame.height()) != dimensions)
    {
        return Err(Error::Processor(format!(
            "video frame {index} is {}x{}, expected {}x{}",
            frame.width(),
            frame.height(),
            dimensions.0,
            dimensions.1
        )));
    }
    Ok(dimensions)
}

/// Returns evenly spaced source-frame indices, including both endpoints.
pub fn uniform_sample_indices(
    total_frames: usize,
    sample_count: usize,
) -> Result<Vec<usize>, Error> {
    if total_frames == 0 || sample_count == 0 {
        return Err(Error::Processor(format!(
            "video sampling requires positive frame counts, got {total_frames} source and {sample_count} requested"
        )));
    }
    let sample_count = sample_count.min(total_frames);
    if sample_count == 1 {
        return Ok(vec![0]);
    }
    let last = (total_frames - 1) as f64;
    let denominator = (sample_count - 1) as f64;
    Ok((0..sample_count)
        .map(|index| (index as f64 * last / denominator).round_ties_even() as usize)
        .collect())
}

/// Computes a uniformly sampled frame count from source and target rates.
pub fn sampled_frame_count(
    total_frames: usize,
    source_fps: f64,
    target_fps: f64,
    min_frames: usize,
    max_frames: usize,
) -> Result<usize, Error> {
    if !source_fps.is_finite() || source_fps <= 0.0 {
        return Err(Error::Processor(format!(
            "video source FPS must be finite and positive, got {source_fps}"
        )));
    }
    if !target_fps.is_finite() || target_fps <= 0.0 {
        return Err(Error::Processor(format!(
            "video target FPS must be finite and positive, got {target_fps}"
        )));
    }
    if min_frames == 0 || max_frames < min_frames {
        return Err(Error::Processor(format!(
            "video frame limits must be positive and ordered, got {min_frames}..{max_frames}"
        )));
    }
    let requested = (total_frames as f64 / source_fps * target_fps) as usize;
    Ok(requested
        .max(min_frames)
        .min(max_frames)
        .min(total_frames)
        .max(1))
}

/// Converts selected source-frame indices to timestamps in seconds.
pub fn frame_timestamps(indices: &[usize], source_fps: f64) -> Result<Vec<f64>, Error> {
    if !source_fps.is_finite() || source_fps <= 0.0 {
        return Err(Error::Processor(format!(
            "video source FPS must be finite and positive, got {source_fps}"
        )));
    }
    Ok(indices
        .iter()
        .map(|index| *index as f64 / source_fps)
        .collect())
}

/// Formats a timestamp using Gemma-style zero-padded `mm:ss` text.
pub fn format_mm_ss(timestamp: f64) -> Result<String, Error> {
    if !timestamp.is_finite() || timestamp < 0.0 {
        return Err(Error::Processor(format!(
            "video timestamp must be finite and non-negative, got {timestamp}"
        )));
    }
    let seconds = timestamp.floor() as u64;
    Ok(format!("{:02}:{:02}", seconds / 60, seconds % 60))
}

/// Pads frame indices by repeating the last frame to a temporal multiple.
pub fn pad_frame_indices(indices: &mut Vec<usize>, temporal_factor: usize) -> Result<(), Error> {
    if indices.is_empty() || temporal_factor == 0 {
        return Err(Error::Processor(
            "temporal frame padding requires frames and a positive factor".to_string(),
        ));
    }
    let remainder = indices.len() % temporal_factor;
    if remainder != 0 {
        let last = *indices.last().expect("indices are non-empty");
        indices.resize(indices.len() + temporal_factor - remainder, last);
    }
    Ok(())
}

/// Computes one average timestamp per temporal frame group.
pub fn temporal_group_timestamps(
    indices: &[usize],
    source_fps: f64,
    temporal_factor: usize,
) -> Result<Vec<f64>, Error> {
    if !source_fps.is_finite() || source_fps <= 0.0 {
        return Err(Error::Processor(format!(
            "video source FPS must be finite and positive, got {source_fps}"
        )));
    }
    if temporal_factor == 0 || indices.is_empty() || indices.len() % temporal_factor != 0 {
        return Err(Error::Processor(format!(
            "{} frame indices cannot be grouped by temporal factor {temporal_factor}",
            indices.len()
        )));
    }
    Ok(indices
        .chunks_exact(temporal_factor)
        .map(|chunk| {
            let first = chunk[0] as f64 / source_fps;
            let last = chunk[temporal_factor - 1] as f64 / source_fps;
            (first + last) / 2.0
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::{
        format_mm_ss, frame_timestamps, pad_frame_indices, sampled_frame_count,
        temporal_group_timestamps, uniform_sample_indices,
    };

    #[test]
    fn uniform_sampling_includes_endpoints() {
        assert_eq!(uniform_sample_indices(10, 4).unwrap(), vec![0, 3, 6, 9]);
    }

    #[test]
    fn temporal_padding_and_timestamps_repeat_last_frame() {
        let mut indices = vec![0, 2, 4];
        pad_frame_indices(&mut indices, 2).unwrap();
        assert_eq!(indices, vec![0, 2, 4, 4]);
        assert_eq!(
            temporal_group_timestamps(&indices, 2.0, 2).unwrap(),
            vec![0.5, 2.0]
        );
    }

    #[test]
    fn shared_sampling_and_timestamp_helpers_cover_architecture_formats() {
        assert_eq!(sampled_frame_count(120, 24.0, 2.0, 1, 32).unwrap(), 10);
        assert_eq!(
            frame_timestamps(&[0, 24, 1_464], 24.0).unwrap(),
            vec![0.0, 1.0, 61.0]
        );
        assert_eq!(format_mm_ss(61.9).unwrap(), "01:01");
    }
}
