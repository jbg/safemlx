use std::{fs, path::Path};

use safemlx::Array;
use serde::Deserialize;

use super::video::{
    pad_frame_indices, sampled_frame_count, temporal_group_timestamps, uniform_sample_indices,
    validate_rgb_frames,
};
use super::{
    image::{rescale_and_normalize_rgb8, resize_rgb8_bicubic, NormalizedImage, RgbImageView},
    prepared_model_input, push_text_token_ids, MediaInput, MediaPayload, OwnedInputMetadata,
    PreparedInputPart, PreparedModelInput, ProcessorInput, VideoFrames, VideoSampling,
};
use crate::{error::Error, models::input::Modality};

#[derive(Debug, Clone, Deserialize)]
struct QwenProcessorSize {
    shortest_edge: u64,
    longest_edge: u64,
}

#[derive(Debug, Clone, Deserialize)]
struct QwenVisualProcessorConfig {
    size: QwenProcessorSize,
    patch_size: usize,
    temporal_patch_size: usize,
    merge_size: usize,
    #[serde(default = "default_true")]
    do_resize: bool,
    #[serde(default = "default_true")]
    do_rescale: bool,
    #[serde(default = "default_rescale_factor")]
    rescale_factor: f32,
    #[serde(default = "default_true")]
    do_normalize: bool,
    #[serde(default = "default_bicubic_resample")]
    resample: u8,
    image_mean: [f32; 3],
    image_std: [f32; 3],
    #[serde(default = "default_video_fps")]
    fps: f64,
    #[serde(default = "default_min_frames")]
    min_frames: usize,
    #[serde(default = "default_max_frames")]
    max_frames: usize,
    #[serde(default = "default_true")]
    do_sample_frames: bool,
}

fn default_true() -> bool {
    true
}

fn default_rescale_factor() -> f32 {
    1.0 / 255.0
}

fn default_bicubic_resample() -> u8 {
    3
}

fn default_video_fps() -> f64 {
    2.0
}

fn default_min_frames() -> usize {
    4
}

fn default_max_frames() -> usize {
    768
}

#[derive(Debug, Deserialize)]
struct QwenModelConfig {
    vision_start_token_id: Option<u32>,
    vision_end_token_id: Option<u32>,
    #[serde(default)]
    text_config: Option<QwenTextConfig>,
}

#[derive(Debug, Deserialize)]
struct QwenTextConfig {
    vision_start_token_id: Option<u32>,
    vision_end_token_id: Option<u32>,
}

#[derive(Debug, Clone)]
pub(super) struct QwenProcessor {
    image_config: Option<QwenVisualProcessorConfig>,
    video_config: Option<QwenVisualProcessorConfig>,
    vision_start_token_id: Option<u32>,
    vision_end_token_id: Option<u32>,
}

impl QwenProcessor {
    pub(super) fn load(model_dir: &Path) -> Result<Option<Self>, Error> {
        let image_config = load_visual_config(&model_dir.join("preprocessor_config.json"))?;
        let video_config = load_visual_config(&model_dir.join("video_preprocessor_config.json"))?;
        if image_config.is_none() && video_config.is_none() {
            return Ok(None);
        }
        let model_config: QwenModelConfig =
            serde_json::from_slice(&fs::read(model_dir.join("config.json"))?)?;
        let text_config = model_config.text_config.as_ref();
        Ok(Some(Self {
            image_config,
            video_config,
            vision_start_token_id: model_config
                .vision_start_token_id
                .or_else(|| text_config.and_then(|text| text.vision_start_token_id)),
            vision_end_token_id: model_config
                .vision_end_token_id
                .or_else(|| text_config.and_then(|text| text.vision_end_token_id)),
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
        match (item.modality, item.payload) {
            (Modality::Image, MediaPayload::Rgb8(image)) => {
                push_text_token_ids(parts, &[self.vision_start_token_id()?]);
                parts.push(self.process_image(image)?);
                push_text_token_ids(parts, &[self.vision_end_token_id()?]);
            }
            (Modality::Video, MediaPayload::VideoFrames(video)) => {
                parts.extend(self.process_video(video, encode_text)?);
            }
            (modality, _) => {
                return Err(Error::Processor(format!(
                    "Qwen processor does not support {} media yet",
                    modality.as_str()
                )));
            }
        }
        Ok(())
    }

    fn vision_start_token_id(&self) -> Result<u32, Error> {
        self.vision_start_token_id.ok_or_else(|| {
            Error::Processor("Qwen processor requires vision_start_token_id in config.json".into())
        })
    }

    fn vision_end_token_id(&self) -> Result<u32, Error> {
        self.vision_end_token_id.ok_or_else(|| {
            Error::Processor("Qwen processor requires vision_end_token_id in config.json".into())
        })
    }

    fn process_image(&self, image: RgbImageView<'_>) -> Result<PreparedInputPart, Error> {
        let config = self.image_config.as_ref().ok_or_else(|| {
            Error::Processor("Qwen model directory has no image processor config".into())
        })?;
        let factor = config
            .patch_size
            .checked_mul(config.merge_size)
            .ok_or_else(|| Error::Processor("Qwen image resize factor overflow".into()))?;
        let (height, width) = if config.do_resize {
            smart_resize(
                image.height() as usize,
                image.width() as usize,
                factor,
                config.size.shortest_edge as usize,
                config.size.longest_edge as usize,
            )?
        } else {
            (image.height() as usize, image.width() as usize)
        };
        let resized = resize_rgb8_bicubic(image, width as u32, height as u32)?;
        let rescale_factor = if config.do_rescale {
            config.rescale_factor
        } else {
            1.0
        };
        let (mean, std) = if config.do_normalize {
            (config.image_mean, config.image_std)
        } else {
            ([0.0; 3], [1.0; 3])
        };
        let normalized = rescale_and_normalize_rgb8(resized.as_view(), rescale_factor, mean, std)?;
        let (patches, grid_thw) = pack_image_patches(&normalized, config)?;
        Ok(PreparedInputPart::media_tensor(
            Modality::Image,
            patches,
            OwnedInputMetadata::GridThw(grid_thw),
        ))
    }

    fn process_video(
        &self,
        video: VideoFrames<'_>,
        encode_text: &mut dyn FnMut(&str) -> Result<Vec<u32>, Error>,
    ) -> Result<Vec<PreparedInputPart>, Error> {
        let config = self.video_config.as_ref().ok_or_else(|| {
            Error::Processor("Qwen model directory has no video processor config".into())
        })?;
        let (width, height) = validate_rgb_frames(video.frames)?;
        let source_fps = video.source_fps.unwrap_or(24.0);
        if !source_fps.is_finite() || source_fps <= 0.0 {
            return Err(Error::Processor(format!(
                "video source FPS must be finite and positive, got {source_fps}"
            )));
        }
        let total_frames = video.frames.len();
        let sample_count = match video.sampling {
            VideoSampling::ProcessorDefault if config.do_sample_frames => sampled_frame_count(
                total_frames,
                source_fps,
                config.fps,
                config.min_frames,
                config.max_frames,
            )?,
            VideoSampling::ProcessorDefault | VideoSampling::All => total_frames,
            VideoSampling::Fps(target_fps) => sampled_frame_count(
                total_frames,
                source_fps,
                target_fps,
                config.min_frames,
                config.max_frames,
            )?,
            VideoSampling::FrameCount(count) => count.clamp(1, total_frames),
        };
        let mut indices = uniform_sample_indices(total_frames, sample_count)?;
        let factor = config
            .patch_size
            .checked_mul(config.merge_size)
            .ok_or_else(|| Error::Processor("Qwen video resize factor overflow".into()))?;
        let (resized_height, resized_width) = if config.do_resize {
            smart_resize_video(
                indices.len(),
                height as usize,
                width as usize,
                config.temporal_patch_size,
                factor,
                config.size.shortest_edge as usize,
                config.size.longest_edge as usize,
            )?
        } else {
            (height as usize, width as usize)
        };
        pad_frame_indices(&mut indices, config.temporal_patch_size)?;
        let timestamps =
            temporal_group_timestamps(&indices, source_fps, config.temporal_patch_size)?;
        let rescale_factor = if config.do_rescale {
            config.rescale_factor
        } else {
            1.0
        };
        let (mean, std) = if config.do_normalize {
            (config.image_mean, config.image_std)
        } else {
            ([0.0; 3], [1.0; 3])
        };
        let mut frames = Vec::with_capacity(indices.len());
        for index in indices {
            let resized = resize_rgb8_bicubic(
                video.frames[index],
                resized_width as u32,
                resized_height as u32,
            )?;
            frames.push(rescale_and_normalize_rgb8(
                resized.as_view(),
                rescale_factor,
                mean,
                std,
            )?);
        }
        let mut parts = Vec::with_capacity(timestamps.len() * 3);
        for (timestamp, chunk) in timestamps
            .iter()
            .zip(frames.chunks(config.temporal_patch_size))
        {
            let mut prefix = encode_text(&format!("<{timestamp:.1} seconds>"))?;
            prefix.push(self.vision_start_token_id()?);
            push_text_token_ids(&mut parts, &prefix);
            let (patches, grid_thw) = pack_video_patches(chunk, config)?;
            parts.push(PreparedInputPart::media_tensor(
                Modality::Video,
                patches,
                OwnedInputMetadata::GridThw(grid_thw),
            ));
            push_text_token_ids(&mut parts, &[self.vision_end_token_id()?]);
        }
        Ok(parts)
    }
}

fn load_visual_config(path: &Path) -> Result<Option<QwenVisualProcessorConfig>, Error> {
    if !path.exists() {
        return Ok(None);
    }
    let config: QwenVisualProcessorConfig = serde_json::from_slice(&fs::read(path)?)?;
    validate_config(&config)?;
    Ok(Some(config))
}

fn validate_config(config: &QwenVisualProcessorConfig) -> Result<(), Error> {
    if config.patch_size == 0 || config.temporal_patch_size == 0 || config.merge_size == 0 {
        return Err(Error::Processor(
            "Qwen patch_size, temporal_patch_size, and merge_size must be positive".into(),
        ));
    }
    if config.size.shortest_edge == 0
        || config.size.longest_edge == 0
        || config.size.shortest_edge > config.size.longest_edge
    {
        return Err(Error::Processor(format!(
            "invalid Qwen image size constraints: {}..{} pixels",
            config.size.shortest_edge, config.size.longest_edge
        )));
    }
    if config.resample != default_bicubic_resample() {
        return Err(Error::Processor(format!(
            "Qwen visual resample mode {} is unsupported; expected bicubic mode 3",
            config.resample
        )));
    }
    if !config.fps.is_finite()
        || config.fps <= 0.0
        || config.min_frames == 0
        || config.max_frames < config.min_frames
    {
        return Err(Error::Processor(format!(
            "invalid Qwen video sampling defaults: fps {}, frames {}..{}",
            config.fps, config.min_frames, config.max_frames
        )));
    }
    Ok(())
}

fn smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(usize, usize), Error> {
    if height == 0 || width == 0 || factor == 0 {
        return Err(Error::Processor(format!(
            "smart resize requires positive dimensions and factor, got {width}x{height}, factor {factor}"
        )));
    }
    let ratio = height.max(width) as f64 / height.min(width) as f64;
    if ratio > 200.0 {
        return Err(Error::Processor(format!(
            "absolute image aspect ratio must be at most 200, got {ratio}"
        )));
    }
    let round_to_factor =
        |value: usize| ((value as f64 / factor as f64).round_ties_even() as usize) * factor;
    let mut resized_height = round_to_factor(height).max(factor);
    let mut resized_width = round_to_factor(width).max(factor);
    let area = resized_height.saturating_mul(resized_width);
    if area > max_pixels {
        let beta = ((height * width) as f64 / max_pixels as f64).sqrt();
        resized_height =
            ((height as f64 / beta / factor as f64).floor() as usize * factor).max(factor);
        resized_width =
            ((width as f64 / beta / factor as f64).floor() as usize * factor).max(factor);
    } else if area < min_pixels {
        let beta = (min_pixels as f64 / (height * width) as f64).sqrt();
        resized_height = (height as f64 * beta / factor as f64).ceil() as usize * factor;
        resized_width = (width as f64 * beta / factor as f64).ceil() as usize * factor;
    }
    Ok((resized_height, resized_width))
}

fn smart_resize_video(
    num_frames: usize,
    height: usize,
    width: usize,
    temporal_factor: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(usize, usize), Error> {
    if num_frames == 0 || temporal_factor == 0 || factor == 0 {
        return Err(Error::Processor(
            "video smart resize requires frames and positive factors".to_string(),
        ));
    }
    if height < factor || width < factor {
        return Err(Error::Processor(format!(
            "video dimensions {width}x{height} must be at least resize factor {factor}"
        )));
    }
    let ratio = height.max(width) as f64 / height.min(width) as f64;
    if ratio > 200.0 {
        return Err(Error::Processor(format!(
            "absolute video aspect ratio must be at most 200, got {ratio}"
        )));
    }
    let round_to_factor =
        |value: usize| ((value as f64 / factor as f64).round_ties_even() as usize) * factor;
    let mut resized_height = round_to_factor(height).max(factor);
    let mut resized_width = round_to_factor(width).max(factor);
    let padded_frames = num_frames.div_ceil(temporal_factor) * temporal_factor;
    let volume = padded_frames
        .saturating_mul(resized_height)
        .saturating_mul(resized_width);
    if volume > max_pixels {
        let beta = ((num_frames * height * width) as f64 / max_pixels as f64).sqrt();
        resized_height =
            ((height as f64 / beta / factor as f64).floor() as usize * factor).max(factor);
        resized_width =
            ((width as f64 / beta / factor as f64).floor() as usize * factor).max(factor);
    } else if volume < min_pixels {
        let beta = (min_pixels as f64 / (num_frames * height * width) as f64).sqrt();
        resized_height = (height as f64 * beta / factor as f64).ceil() as usize * factor;
        resized_width = (width as f64 * beta / factor as f64).ceil() as usize * factor;
    }
    Ok((resized_height, resized_width))
}

fn pack_image_patches(
    image: &NormalizedImage,
    config: &QwenVisualProcessorConfig,
) -> Result<(Array, Array), Error> {
    let grid_h = image.height() / config.patch_size;
    let grid_w = image.width() / config.patch_size;
    if image.height() % config.patch_size != 0 || image.width() % config.patch_size != 0 {
        return Err(Error::Processor(format!(
            "processed image {}x{} is not divisible by patch size {}",
            image.width(),
            image.height(),
            config.patch_size
        )));
    }
    if grid_h % config.merge_size != 0 || grid_w % config.merge_size != 0 {
        return Err(Error::Processor(format!(
            "image patch grid {grid_h}x{grid_w} is not divisible by merge size {}",
            config.merge_size
        )));
    }

    let patch_count = grid_h * grid_w;
    let patch_width =
        image.channels() * config.temporal_patch_size * config.patch_size * config.patch_size;
    let mut patches = Vec::with_capacity(patch_count * patch_width);
    for block_y in 0..grid_h / config.merge_size {
        for block_x in 0..grid_w / config.merge_size {
            for merge_y in 0..config.merge_size {
                for merge_x in 0..config.merge_size {
                    let patch_y = (block_y * config.merge_size + merge_y) * config.patch_size;
                    let patch_x = (block_x * config.merge_size + merge_x) * config.patch_size;
                    for channel in 0..image.channels() {
                        for _time in 0..config.temporal_patch_size {
                            for y in 0..config.patch_size {
                                for x in 0..config.patch_size {
                                    patches.push(image.get(channel, patch_y + y, patch_x + x));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    let patches = Array::from_slice(&patches, &[patch_count as i32, patch_width as i32]);
    let grid_thw = Array::from_slice(&[1i32, grid_h as i32, grid_w as i32], &[1, 3]);
    Ok((patches, grid_thw))
}

fn pack_video_patches(
    frames: &[NormalizedImage],
    config: &QwenVisualProcessorConfig,
) -> Result<(Array, Array), Error> {
    let first = frames
        .first()
        .ok_or_else(|| Error::Processor("video must contain processed frames".to_string()))?;
    if frames.len() % config.temporal_patch_size != 0 {
        return Err(Error::Processor(format!(
            "{} processed video frames are not divisible by temporal patch size {}",
            frames.len(),
            config.temporal_patch_size
        )));
    }
    if frames.iter().any(|frame| {
        frame.width() != first.width()
            || frame.height() != first.height()
            || frame.channels() != first.channels()
    }) {
        return Err(Error::Processor(
            "processed video frames must have identical dimensions".to_string(),
        ));
    }
    if first.height() % config.patch_size != 0 || first.width() % config.patch_size != 0 {
        return Err(Error::Processor(format!(
            "processed video frame {}x{} is not divisible by patch size {}",
            first.width(),
            first.height(),
            config.patch_size
        )));
    }
    let grid_t = frames.len() / config.temporal_patch_size;
    let grid_h = first.height() / config.patch_size;
    let grid_w = first.width() / config.patch_size;
    if grid_h % config.merge_size != 0 || grid_w % config.merge_size != 0 {
        return Err(Error::Processor(format!(
            "video patch grid {grid_h}x{grid_w} is not divisible by merge size {}",
            config.merge_size
        )));
    }

    let patch_count = grid_t * grid_h * grid_w;
    let patch_width =
        first.channels() * config.temporal_patch_size * config.patch_size * config.patch_size;
    let mut patches = Vec::with_capacity(patch_count * patch_width);
    for temporal_group in 0..grid_t {
        for block_y in 0..grid_h / config.merge_size {
            for block_x in 0..grid_w / config.merge_size {
                for merge_y in 0..config.merge_size {
                    for merge_x in 0..config.merge_size {
                        let patch_y = (block_y * config.merge_size + merge_y) * config.patch_size;
                        let patch_x = (block_x * config.merge_size + merge_x) * config.patch_size;
                        for channel in 0..first.channels() {
                            for time in 0..config.temporal_patch_size {
                                let frame =
                                    &frames[temporal_group * config.temporal_patch_size + time];
                                for y in 0..config.patch_size {
                                    for x in 0..config.patch_size {
                                        patches.push(frame.get(channel, patch_y + y, patch_x + x));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    let patches = Array::from_slice(&patches, &[patch_count as i32, patch_width as i32]);
    let grid_thw = Array::from_slice(&[grid_t as i32, grid_h as i32, grid_w as i32], &[1, 3]);
    Ok((patches, grid_thw))
}

#[cfg(test)]
mod tests {
    use super::{
        pack_image_patches, smart_resize, QwenProcessor, QwenProcessorSize,
        QwenVisualProcessorConfig,
    };
    use crate::{
        models::input::{InputPayload, Modality},
        processor::{
            image::{rescale_and_normalize_rgb8, RgbImageView},
            MediaInput, ProcessorInput, VideoSampling,
        },
    };

    fn tiny_config() -> QwenVisualProcessorConfig {
        QwenVisualProcessorConfig {
            size: QwenProcessorSize {
                shortest_edge: 16,
                longest_edge: 16,
            },
            patch_size: 2,
            temporal_patch_size: 2,
            merge_size: 2,
            do_resize: true,
            do_rescale: true,
            rescale_factor: 1.0 / 255.0,
            do_normalize: true,
            resample: 3,
            image_mean: [0.0; 3],
            image_std: [1.0; 3],
            fps: 2.0,
            min_frames: 1,
            max_frames: 8,
            do_sample_frames: true,
        }
    }

    #[test]
    fn smart_resize_matches_qwen_constraints() {
        assert_eq!(
            smart_resize(100, 200, 32, 65_536, 16_777_216).unwrap(),
            (192, 384)
        );
        assert_eq!(
            smart_resize(1024, 1024, 32, 65_536, 262_144).unwrap(),
            (512, 512)
        );
    }

    #[test]
    fn patch_packing_groups_merge_cells_and_duplicates_time() {
        let mut pixels = Vec::new();
        for value in 0u8..16 {
            pixels.extend_from_slice(&[value, 100 + value, 200 + value]);
        }
        let image = RgbImageView::packed(&pixels, 4, 4).unwrap();
        let normalized =
            rescale_and_normalize_rgb8(image, 1.0 / 255.0, [0.0; 3], [1.0; 3]).unwrap();
        let (patches, grid) = pack_image_patches(&normalized, &tiny_config()).unwrap();
        assert_eq!(patches.shape(), &[4, 24]);
        assert_eq!(grid.evaluated().unwrap().as_slice::<i32>(), &[1, 2, 2]);
        let evaluated = patches.evaluated().unwrap();
        let values = evaluated.as_slice::<f32>();
        let first_channel = [0.0, 1.0, 4.0, 5.0].map(|value| value / 255.0);
        assert_eq!(&values[..4], &first_channel);
        assert_eq!(&values[4..8], &first_channel);
    }

    #[test]
    fn processor_wraps_ordered_image_with_vision_boundaries() {
        let processor = QwenProcessor {
            image_config: Some(tiny_config()),
            video_config: Some(tiny_config()),
            vision_start_token_id: Some(44),
            vision_end_token_id: Some(45),
        };
        let pixels = vec![128u8; 4 * 4 * 3];
        let image = RgbImageView::packed(&pixels, 4, 4).unwrap();
        let prepared = processor
            .prepare_input(
                &[
                    ProcessorInput::TokenIds(&[10]),
                    ProcessorInput::Media(MediaInput::image_rgb8(image)),
                    ProcessorInput::TokenIds(&[11]),
                ],
                &mut |_text| Ok(Vec::new()),
            )
            .unwrap();
        let parts = prepared.input_parts();
        assert_eq!(parts.len(), 5);
        assert_eq!(parts[0].modality, Modality::Text);
        assert_eq!(parts[2].modality, Modality::Image);
        assert_eq!(parts[4].modality, Modality::Text);
        let InputPayload::TokenIds(start) = parts[1].payload else {
            panic!("expected vision-start token");
        };
        assert_eq!(start.evaluated().unwrap().as_slice::<u32>(), &[44]);
        let InputPayload::TokenIds(end) = parts[3].payload else {
            panic!("expected vision-end token");
        };
        assert_eq!(end.evaluated().unwrap().as_slice::<u32>(), &[45]);
        let InputPayload::Tensor(patches) = parts[2].payload else {
            panic!("expected processed image tensor");
        };
        assert_eq!(patches.shape(), &[4, 24]);
        assert_eq!(
            parts[2]
                .metadata
                .qwen_grid_thw
                .unwrap()
                .evaluated()
                .unwrap()
                .as_slice::<i32>(),
            &[1, 2, 2]
        );
    }

    #[test]
    fn processor_expands_video_timestamps_and_packs_temporal_frames() {
        let processor = QwenProcessor {
            image_config: Some(tiny_config()),
            video_config: Some(tiny_config()),
            vision_start_token_id: Some(44),
            vision_end_token_id: Some(45),
        };
        let frame_pixels = (0..4)
            .map(|frame| vec![frame as u8 * 32; 4 * 4 * 3])
            .collect::<Vec<_>>();
        let frames = frame_pixels
            .iter()
            .map(|pixels| RgbImageView::packed(pixels, 4, 4).unwrap())
            .collect::<Vec<_>>();
        let mut timestamp_text = Vec::new();
        let prepared = processor
            .prepare_input(
                &[
                    ProcessorInput::TokenIds(&[10]),
                    ProcessorInput::Media(MediaInput::video_rgb8_with_sampling(
                        &frames,
                        Some(2.0),
                        VideoSampling::All,
                    )),
                    ProcessorInput::TokenIds(&[11]),
                ],
                &mut |text| {
                    timestamp_text.push(text.to_string());
                    Ok(vec![90 + timestamp_text.len() as u32])
                },
            )
            .unwrap();
        let parts = prepared.input_parts();
        assert_eq!(timestamp_text, vec!["<0.2 seconds>", "<1.2 seconds>"]);
        assert_eq!(parts.len(), 8);
        assert_eq!(parts[2].modality, Modality::Video);
        assert_eq!(parts[5].modality, Modality::Video);
        let InputPayload::TokenIds(replacement) = parts[1].payload else {
            panic!("expected timestamp replacement tokens");
        };
        assert_eq!(
            replacement.evaluated().unwrap().as_slice::<u32>(),
            &[91, 44]
        );
        let InputPayload::Tensor(first_patches) = parts[2].payload else {
            panic!("expected first processed video tensor");
        };
        assert_eq!(first_patches.shape(), &[4, 24]);
        assert_eq!(
            parts[2]
                .metadata
                .qwen_grid_thw
                .unwrap()
                .evaluated()
                .unwrap()
                .as_slice::<i32>(),
            &[1, 2, 2]
        );
        let InputPayload::TokenIds(first_end) = parts[3].payload else {
            panic!("expected first vision-end token");
        };
        assert_eq!(first_end.evaluated().unwrap().as_slice::<u32>(), &[45]);
        let InputPayload::TokenIds(second_prefix) = parts[4].payload else {
            panic!("expected second timestamp prefix tokens");
        };
        assert_eq!(
            second_prefix.evaluated().unwrap().as_slice::<u32>(),
            &[92, 44]
        );
        let InputPayload::Tensor(second_patches) = parts[5].payload else {
            panic!("expected second processed video tensor");
        };
        assert_eq!(second_patches.shape(), &[4, 24]);
    }
}
