use std::{fs, path::Path};

#[cfg(any(feature = "image-processing", feature = "audio-processing"))]
use safemlx::Array;
use serde::Deserialize;

#[cfg(feature = "audio-processing")]
use super::audio::{extract_log_mel, LogMelConfig};
#[cfg(feature = "image-processing")]
use super::image::{rescale_and_normalize_rgb8, resize_rgb8_bicubic, NormalizedImage};
#[cfg(feature = "image-processing")]
use super::video::{
    format_mm_ss, frame_timestamps, sampled_frame_count, uniform_sample_indices,
    validate_rgb_frames,
};
use super::{
    prepared_model_input, push_text_token_ids, MediaInput, PreparedInputPart, PreparedModelInput,
    ProcessorInput,
};
#[cfg(any(feature = "image-processing", feature = "audio-processing"))]
use super::{MediaPayload, OwnedInputMetadata};
#[cfg(feature = "image-processing")]
use super::{VideoFrames, VideoSampling};
use crate::error::Error;
#[cfg(any(feature = "image-processing", feature = "audio-processing"))]
use crate::models::input::Modality;

#[derive(Debug, Clone, Deserialize)]
struct Gemma4ModelConfig {
    #[cfg(feature = "image-processing")]
    boi_token_id: Option<u32>,
    #[cfg(feature = "image-processing")]
    eoi_token_id: Option<u32>,
    #[cfg(feature = "audio-processing")]
    boa_token_id: Option<u32>,
    #[cfg(feature = "audio-processing")]
    eoa_token_id: Option<u32>,
    #[cfg(feature = "image-processing")]
    #[serde(default = "default_soft_tokens")]
    vision_soft_tokens_per_image: usize,
    #[cfg(feature = "image-processing")]
    vision_config: Option<Gemma4VisionProcessorConfig>,
    #[cfg(feature = "audio-processing")]
    audio_config: Option<serde_json::Value>,
}

#[cfg(feature = "image-processing")]
#[derive(Debug, Clone, Deserialize)]
struct Gemma4VisionProcessorConfig {
    #[serde(default = "default_patch_size")]
    patch_size: usize,
    #[serde(default = "default_pooling_kernel_size")]
    pooling_kernel_size: usize,
}

#[cfg(feature = "image-processing")]
#[derive(Debug, Clone, Default, Deserialize)]
struct Gemma4PreprocessorConfig {
    #[serde(default)]
    patch_size: Option<usize>,
    #[serde(default)]
    pooling_kernel_size: Option<usize>,
    #[serde(default)]
    max_soft_tokens: Option<usize>,
}

#[cfg(feature = "image-processing")]
#[derive(Debug, Clone, Deserialize)]
struct Gemma4VideoPreprocessorConfig {
    #[serde(default = "default_patch_size")]
    patch_size: usize,
    #[serde(default = "default_pooling_kernel_size")]
    pooling_kernel_size: usize,
    #[serde(default = "default_video_soft_tokens")]
    max_soft_tokens: usize,
    #[serde(default = "default_video_frames")]
    num_frames: usize,
}

#[cfg(feature = "image-processing")]
impl Default for Gemma4VideoPreprocessorConfig {
    fn default() -> Self {
        Self {
            patch_size: default_patch_size(),
            pooling_kernel_size: default_pooling_kernel_size(),
            max_soft_tokens: default_video_soft_tokens(),
            num_frames: default_video_frames(),
        }
    }
}

#[cfg(feature = "image-processing")]
fn default_patch_size() -> usize {
    16
}

#[cfg(feature = "image-processing")]
fn default_pooling_kernel_size() -> usize {
    3
}

#[cfg(feature = "image-processing")]
fn default_soft_tokens() -> usize {
    280
}

#[cfg(feature = "image-processing")]
fn default_video_soft_tokens() -> usize {
    70
}

#[cfg(feature = "image-processing")]
fn default_video_frames() -> usize {
    32
}

#[derive(Debug, Clone)]
pub(super) struct Gemma4Processor {
    #[cfg(feature = "image-processing")]
    patch_size: usize,
    #[cfg(feature = "image-processing")]
    pooling_kernel_size: usize,
    #[cfg(feature = "image-processing")]
    max_soft_tokens: usize,
    #[cfg(feature = "image-processing")]
    boi_token_id: Option<u32>,
    #[cfg(feature = "image-processing")]
    eoi_token_id: Option<u32>,
    #[cfg(feature = "image-processing")]
    video_patch_size: usize,
    #[cfg(feature = "image-processing")]
    video_pooling_kernel_size: usize,
    #[cfg(feature = "image-processing")]
    video_max_soft_tokens: usize,
    #[cfg(feature = "image-processing")]
    video_num_frames: usize,
    #[cfg(feature = "audio-processing")]
    boa_token_id: Option<u32>,
    #[cfg(feature = "audio-processing")]
    eoa_token_id: Option<u32>,
}

impl Gemma4Processor {
    pub(super) fn load(model_dir: &Path) -> Result<Option<Self>, Error> {
        let config: Gemma4ModelConfig =
            serde_json::from_slice(&fs::read(model_dir.join("config.json"))?)?;
        #[cfg(not(any(feature = "image-processing", feature = "audio-processing")))]
        let _ = &config;
        #[cfg(feature = "image-processing")]
        let has_image_processor = config.vision_config.is_some();
        #[cfg(not(feature = "image-processing"))]
        let has_image_processor = false;
        #[cfg(feature = "audio-processing")]
        let has_audio_processor = config.audio_config.is_some();
        #[cfg(not(feature = "audio-processing"))]
        let has_audio_processor = false;
        let has_supported_processor = has_image_processor || has_audio_processor;
        if !has_supported_processor {
            return Ok(None);
        }
        #[cfg(feature = "image-processing")]
        let processor_path = model_dir.join("preprocessor_config.json");
        #[cfg(feature = "image-processing")]
        let processor = if processor_path.exists() {
            serde_json::from_slice(&fs::read(processor_path)?)?
        } else {
            Gemma4PreprocessorConfig::default()
        };
        #[cfg(feature = "image-processing")]
        let video_processor_path = model_dir.join("video_preprocessor_config.json");
        #[cfg(feature = "image-processing")]
        let video_processor = if video_processor_path.exists() {
            serde_json::from_slice(&fs::read(video_processor_path)?)?
        } else {
            Gemma4VideoPreprocessorConfig::default()
        };
        #[cfg(feature = "image-processing")]
        let max_soft_tokens = processor
            .max_soft_tokens
            .unwrap_or(config.vision_soft_tokens_per_image);
        #[cfg(feature = "image-processing")]
        if config.vision_config.is_some() && !matches!(max_soft_tokens, 70 | 140 | 280 | 560 | 1120)
        {
            return Err(Error::Processor(format!(
                "Gemma 4 max_soft_tokens must be one of 70, 140, 280, 560, or 1120, got {max_soft_tokens}"
            )));
        }
        #[cfg(feature = "image-processing")]
        if !matches!(video_processor.max_soft_tokens, 70 | 140 | 280 | 560 | 1120)
            || video_processor.num_frames == 0
        {
            return Err(Error::Processor(format!(
                "Gemma 4 video processor requires a supported soft-token budget and positive frame count, got {} tokens and {} frames",
                video_processor.max_soft_tokens, video_processor.num_frames
            )));
        }
        Ok(Some(Self {
            #[cfg(feature = "image-processing")]
            patch_size: processor.patch_size.unwrap_or_else(|| {
                config
                    .vision_config
                    .as_ref()
                    .map_or(default_patch_size(), |vision| vision.patch_size)
            }),
            #[cfg(feature = "image-processing")]
            pooling_kernel_size: processor.pooling_kernel_size.unwrap_or_else(|| {
                config
                    .vision_config
                    .as_ref()
                    .map_or(default_pooling_kernel_size(), |vision| {
                        vision.pooling_kernel_size
                    })
            }),
            #[cfg(feature = "image-processing")]
            max_soft_tokens,
            #[cfg(feature = "image-processing")]
            boi_token_id: config.boi_token_id,
            #[cfg(feature = "image-processing")]
            eoi_token_id: config.eoi_token_id,
            #[cfg(feature = "image-processing")]
            video_patch_size: video_processor.patch_size,
            #[cfg(feature = "image-processing")]
            video_pooling_kernel_size: video_processor.pooling_kernel_size,
            #[cfg(feature = "image-processing")]
            video_max_soft_tokens: video_processor.max_soft_tokens,
            #[cfg(feature = "image-processing")]
            video_num_frames: video_processor.num_frames,
            #[cfg(feature = "audio-processing")]
            boa_token_id: config.boa_token_id,
            #[cfg(feature = "audio-processing")]
            eoa_token_id: config.eoa_token_id,
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
                ProcessorInput::TokenIds(token_ids) => {
                    push_text_token_ids(&mut parts, token_ids);
                }
                ProcessorInput::Media(media) => {
                    self.push_media_parts(&mut parts, media, encode_text)?;
                }
            }
        }
        prepared_model_input(parts)
    }

    fn push_media_parts(
        &self,
        parts: &mut Vec<PreparedInputPart>,
        item: MediaInput<'_>,
        encode_text: &mut dyn FnMut(&str) -> Result<Vec<u32>, Error>,
    ) -> Result<(), Error> {
        #[cfg(not(feature = "image-processing"))]
        let _ = &encode_text;
        #[cfg(not(any(feature = "image-processing", feature = "audio-processing")))]
        let _ = &parts;
        match (item.modality, item.payload) {
            #[cfg(feature = "image-processing")]
            (Modality::Image, MediaPayload::Rgb8(image)) => {
                push_text_token_ids(
                    parts,
                    &[self.boi_token_id.ok_or_else(|| {
                        Error::Processor("Gemma 4 image processor requires boi_token_id".into())
                    })?],
                );
                parts.push(self.process_image(image)?);
                push_text_token_ids(
                    parts,
                    &[self.eoi_token_id.ok_or_else(|| {
                        Error::Processor("Gemma 4 image processor requires eoi_token_id".into())
                    })?],
                );
                Ok(())
            }
            #[cfg(feature = "image-processing")]
            (Modality::Video, MediaPayload::VideoFrames(video)) => {
                parts.extend(self.process_video(video, encode_text)?);
                Ok(())
            }
            #[cfg(feature = "audio-processing")]
            (Modality::Audio, MediaPayload::AudioF32(waveform)) => {
                push_text_token_ids(
                    parts,
                    &[self.boa_token_id.ok_or_else(|| {
                        Error::Processor("Gemma 4 audio processor requires boa_token_id".into())
                    })?],
                );
                parts.push(self.process_audio(waveform)?);
                push_text_token_ids(
                    parts,
                    &[self.eoa_token_id.ok_or_else(|| {
                        Error::Processor("Gemma 4 audio processor requires eoa_token_id".into())
                    })?],
                );
                Ok(())
            }
            _ => Err(Error::Processor(format!(
                "Gemma 4 processor does not support {} media with the enabled features",
                item.modality.as_str()
            ))),
        }
    }

    #[cfg(feature = "image-processing")]
    fn process_image(
        &self,
        image: super::image::RgbImageView<'_>,
    ) -> Result<PreparedInputPart, Error> {
        let max_patches = self
            .max_soft_tokens
            .checked_mul(self.pooling_kernel_size * self.pooling_kernel_size)
            .ok_or_else(|| Error::Processor("Gemma 4 patch budget overflow".into()))?;
        let (height, width) = aspect_ratio_preserving_size(
            image.height() as usize,
            image.width() as usize,
            self.patch_size,
            max_patches,
            self.pooling_kernel_size,
        )?;
        let resized = resize_rgb8_bicubic(image, width as u32, height as u32)?;
        let normalized =
            rescale_and_normalize_rgb8(resized.as_view(), 1.0 / 255.0, [0.0; 3], [1.0; 3])?;
        let (patches, positions) = pack_patches(&normalized, self.patch_size, max_patches)?;
        Ok(PreparedInputPart::media_tensor(
            Modality::Image,
            patches,
            OwnedInputMetadata::PatchPositionIds(positions),
        ))
    }

    #[cfg(feature = "image-processing")]
    fn process_video(
        &self,
        video: VideoFrames<'_>,
        encode_text: &mut dyn FnMut(&str) -> Result<Vec<u32>, Error>,
    ) -> Result<Vec<PreparedInputPart>, Error> {
        let boi_token_id = self.boi_token_id.ok_or_else(|| {
            Error::Processor("Gemma 4 video processor requires boi_token_id".into())
        })?;
        let eoi_token_id = self.eoi_token_id.ok_or_else(|| {
            Error::Processor("Gemma 4 video processor requires eoi_token_id".into())
        })?;
        let (width, height) = validate_rgb_frames(video.frames)?;
        let source_fps = video.source_fps.unwrap_or(24.0);
        let total_frames = video.frames.len();
        let sample_count = match video.sampling {
            VideoSampling::ProcessorDefault => self.video_num_frames.min(total_frames),
            VideoSampling::All => total_frames,
            VideoSampling::FrameCount(count) => count.clamp(1, total_frames),
            VideoSampling::Fps(target_fps) => sampled_frame_count(
                total_frames,
                source_fps,
                target_fps,
                1,
                self.video_num_frames,
            )?,
        };
        let indices = uniform_sample_indices(total_frames, sample_count)?;
        let timestamps = frame_timestamps(&indices, source_fps)?;
        let max_patches = self
            .video_max_soft_tokens
            .checked_mul(self.video_pooling_kernel_size * self.video_pooling_kernel_size)
            .ok_or_else(|| Error::Processor("Gemma 4 video patch budget overflow".into()))?;
        let (resized_height, resized_width) = aspect_ratio_preserving_size(
            height as usize,
            width as usize,
            self.video_patch_size,
            max_patches,
            self.video_pooling_kernel_size,
        )?;
        let mut replacement = Vec::with_capacity(indices.len() * 3);
        for (frame_index, (source_index, timestamp)) in
            indices.into_iter().zip(timestamps).enumerate()
        {
            let timestamp = format_mm_ss(timestamp)?;
            let timestamp_text = if frame_index == 0 {
                format!("{timestamp} ")
            } else {
                format!(" {timestamp} ")
            };
            let mut prefix = encode_text(&timestamp_text)?;
            prefix.push(boi_token_id);
            replacement.push(PreparedInputPart::text_token_ids(&prefix));

            let resized = resize_rgb8_bicubic(
                video.frames[source_index],
                resized_width as u32,
                resized_height as u32,
            )?;
            let normalized =
                rescale_and_normalize_rgb8(resized.as_view(), 1.0 / 255.0, [0.0; 3], [1.0; 3])?;
            let (patches, positions) =
                pack_patches(&normalized, self.video_patch_size, max_patches)?;
            replacement.push(PreparedInputPart::media_tensor(
                Modality::Video,
                patches,
                OwnedInputMetadata::PatchPositionIds(positions),
            ));
            replacement.push(PreparedInputPart::text_token_ids(&[eoi_token_id]));
        }
        Ok(replacement)
    }

    #[cfg(feature = "audio-processing")]
    fn process_audio(
        &self,
        waveform: super::audio::AudioWaveform<'_>,
    ) -> Result<PreparedInputPart, Error> {
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
        )?;
        let tensor = Array::from_slice(
            &features.values,
            &[1, features.frames as i32, features.mel_bins as i32],
        );
        let mask = Array::from_slice(&features.mask, &[1, features.frames as i32]);
        Ok(PreparedInputPart::media_tensor(
            Modality::Audio,
            tensor,
            OwnedInputMetadata::AudioMask(mask),
        ))
    }
}

#[cfg(feature = "image-processing")]
fn aspect_ratio_preserving_size(
    height: usize,
    width: usize,
    patch_size: usize,
    max_patches: usize,
    pooling_kernel_size: usize,
) -> Result<(usize, usize), Error> {
    if patch_size == 0 || pooling_kernel_size == 0 || max_patches == 0 {
        return Err(Error::Processor(
            "Gemma 4 image processor dimensions must be positive".into(),
        ));
    }
    let target_pixels = max_patches as f64 * (patch_size * patch_size) as f64;
    let factor = (target_pixels / (height * width) as f64).sqrt();
    let side_multiple = patch_size * pooling_kernel_size;
    let mut target_height =
        ((factor * height as f64).floor() as usize / side_multiple) * side_multiple;
    let mut target_width =
        ((factor * width as f64).floor() as usize / side_multiple) * side_multiple;
    let max_side = (max_patches / (pooling_kernel_size * pooling_kernel_size)) * side_multiple;
    if target_height == 0 && target_width == 0 {
        return Err(Error::Processor(format!(
            "Gemma 4 image is too small for resize multiple {side_multiple}"
        )));
    }
    if target_height == 0 {
        target_height = side_multiple;
        target_width = (width / height).saturating_mul(side_multiple).min(max_side);
    } else if target_width == 0 {
        target_width = side_multiple;
        target_height = (height / width).saturating_mul(side_multiple).min(max_side);
    }
    if target_height * target_width > max_patches * patch_size * patch_size {
        return Err(Error::Processor(format!(
            "Gemma 4 resize {target_height}x{target_width} exceeds the {max_patches}-patch budget"
        )));
    }
    Ok((target_height, target_width))
}

#[cfg(feature = "image-processing")]
fn pack_patches(
    image: &NormalizedImage,
    patch_size: usize,
    max_patches: usize,
) -> Result<(Array, Array), Error> {
    if image.height() % patch_size != 0 || image.width() % patch_size != 0 {
        return Err(Error::Processor(format!(
            "Gemma 4 image dimensions {}x{} are not divisible by patch size {patch_size}",
            image.height(),
            image.width()
        )));
    }
    let patch_height = image.height() / patch_size;
    let patch_width = image.width() / patch_size;
    let patch_count = patch_height * patch_width;
    if patch_count > max_patches {
        return Err(Error::Processor(format!(
            "Gemma 4 image produced {patch_count} patches, exceeding {max_patches}"
        )));
    }
    let patch_dims = image.channels() * patch_size * patch_size;
    let mut patches = vec![0.0f32; max_patches * patch_dims];
    let mut positions = vec![-1i32; max_patches * 2];
    for patch_y in 0..patch_height {
        for patch_x in 0..patch_width {
            let patch_index = patch_y * patch_width + patch_x;
            positions[patch_index * 2] = patch_x as i32;
            positions[patch_index * 2 + 1] = patch_y as i32;
            let mut output = patch_index * patch_dims;
            for inner_y in 0..patch_size {
                for inner_x in 0..patch_size {
                    for channel in 0..image.channels() {
                        patches[output] = image.get(
                            channel,
                            patch_y * patch_size + inner_y,
                            patch_x * patch_size + inner_x,
                        );
                        output += 1;
                    }
                }
            }
        }
    }
    Ok((
        Array::from_slice(&patches, &[1, max_patches as i32, patch_dims as i32]),
        Array::from_slice(&positions, &[1, max_patches as i32, 2]),
    ))
}

#[cfg(all(test, feature = "image-processing"))]
mod tests {
    use super::{aspect_ratio_preserving_size, Gemma4Processor};
    use crate::{
        models::input::{InputPayload, Modality},
        processor::{MediaInput, ProcessorInput, RgbImageView},
    };

    #[test]
    fn resize_preserves_budget_and_pooling_multiple() {
        let (height, width) = aspect_ratio_preserving_size(320, 480, 16, 2520, 3).unwrap();
        assert_eq!((height, width), (624, 960));
        assert_eq!(height % 48, 0);
        assert_eq!(width % 48, 0);
        assert!(height * width <= 2520 * 16 * 16);
    }

    #[test]
    fn processor_wraps_ordered_image_with_boundary_tokens() {
        let processor = Gemma4Processor {
            patch_size: 2,
            pooling_kernel_size: 1,
            max_soft_tokens: 70,
            boi_token_id: Some(43),
            eoi_token_id: Some(44),
            video_patch_size: 2,
            video_pooling_kernel_size: 1,
            video_max_soft_tokens: 70,
            video_num_frames: 32,
            #[cfg(feature = "audio-processing")]
            boa_token_id: None,
            #[cfg(feature = "audio-processing")]
            eoa_token_id: None,
        };
        let pixels = vec![128u8; 4 * 4 * 3];
        let image = RgbImageView::packed(&pixels, 4, 4).unwrap();
        let prepared = processor
            .prepare_input(
                &[
                    ProcessorInput::TokenIds(&[7]),
                    ProcessorInput::Media(MediaInput::image_rgb8(image)),
                    ProcessorInput::TokenIds(&[8]),
                ],
                &mut |_| Ok(Vec::new()),
            )
            .unwrap();
        let parts = prepared.input_parts();
        assert_eq!(parts.len(), 5);
        assert_eq!(parts[2].modality, Modality::Image);
        assert!(matches!(parts[2].payload, InputPayload::Tensor(_)));
        assert!(parts[2].metadata.patch_position_ids.is_some());
    }

    #[test]
    fn processor_interleaves_timestamped_video_frames() {
        let processor = Gemma4Processor {
            patch_size: 2,
            pooling_kernel_size: 1,
            max_soft_tokens: 70,
            boi_token_id: Some(43),
            eoi_token_id: Some(44),
            video_patch_size: 2,
            video_pooling_kernel_size: 1,
            video_max_soft_tokens: 70,
            video_num_frames: 32,
            #[cfg(feature = "audio-processing")]
            boa_token_id: None,
            #[cfg(feature = "audio-processing")]
            eoa_token_id: None,
        };
        let pixels = vec![128u8; 4 * 4 * 3];
        let frames = [
            RgbImageView::packed(&pixels, 4, 4).unwrap(),
            RgbImageView::packed(&pixels, 4, 4).unwrap(),
        ];
        let mut encoded = Vec::new();
        let prepared = processor
            .prepare_input(
                &[
                    ProcessorInput::TokenIds(&[7]),
                    ProcessorInput::Media(MediaInput::video_rgb8(&frames, Some(1.0))),
                    ProcessorInput::TokenIds(&[8]),
                ],
                &mut |text| {
                    encoded.push(text.to_string());
                    Ok(vec![90])
                },
            )
            .unwrap();
        let parts = prepared.input_parts();
        let video_parts = parts
            .iter()
            .filter(|part| part.modality == Modality::Video)
            .collect::<Vec<_>>();
        assert_eq!(video_parts.len(), 2);
        assert!(video_parts
            .iter()
            .all(|part| part.metadata.patch_position_ids.is_some()));
        assert_eq!(encoded, vec!["00:00 ", " 00:01 "]);
    }
}

#[cfg(all(test, feature = "audio-processing"))]
mod audio_tests {
    use super::Gemma4Processor;
    use crate::{
        models::input::{InputPayload, Modality},
        processor::{MediaInput, ProcessorInput},
    };

    #[test]
    fn processor_wraps_ordered_audio_with_boundary_tokens() {
        let processor = Gemma4Processor {
            #[cfg(feature = "image-processing")]
            patch_size: 16,
            #[cfg(feature = "image-processing")]
            pooling_kernel_size: 3,
            #[cfg(feature = "image-processing")]
            max_soft_tokens: 280,
            #[cfg(feature = "image-processing")]
            boi_token_id: None,
            #[cfg(feature = "image-processing")]
            eoi_token_id: None,
            #[cfg(feature = "image-processing")]
            video_patch_size: 16,
            #[cfg(feature = "image-processing")]
            video_pooling_kernel_size: 3,
            #[cfg(feature = "image-processing")]
            video_max_soft_tokens: 70,
            #[cfg(feature = "image-processing")]
            video_num_frames: 32,
            boa_token_id: Some(43),
            eoa_token_id: Some(44),
        };
        let samples = vec![0.0f32; 16_000];
        let audio = MediaInput::audio_f32(&samples, 16_000).unwrap();
        let prepared = processor
            .prepare_input(
                &[
                    ProcessorInput::TokenIds(&[7]),
                    ProcessorInput::Media(audio),
                    ProcessorInput::TokenIds(&[8]),
                ],
                &mut |_| Ok(Vec::new()),
            )
            .unwrap();
        let parts = prepared.input_parts();
        assert_eq!(parts.len(), 5);
        assert_eq!(parts[2].modality, Modality::Audio);
        assert!(matches!(parts[2].payload, InputPayload::Tensor(_)));
        assert!(parts[2].metadata.audio_mask.is_some());
    }
}
