//! Shared decoded-image validation and transforms.

use image::{imageops::FilterType, ImageBuffer, Rgb};

use crate::error::Error;

/// Borrowed RGB8 image pixels.
#[derive(Debug, Clone, Copy)]
pub struct RgbImageView<'a> {
    pixels: &'a [u8],
    width: u32,
    height: u32,
    row_stride: usize,
}

impl<'a> RgbImageView<'a> {
    /// Creates an RGB8 image view with tightly packed rows.
    pub fn packed(pixels: &'a [u8], width: u32, height: u32) -> Result<Self, Error> {
        let row_stride = width as usize * 3;
        Self::with_row_stride(pixels, width, height, row_stride)
    }

    /// Creates an RGB8 image view with an explicit byte stride between rows.
    pub fn with_row_stride(
        pixels: &'a [u8],
        width: u32,
        height: u32,
        row_stride: usize,
    ) -> Result<Self, Error> {
        if width == 0 || height == 0 {
            return Err(Error::Processor(format!(
                "image dimensions must be positive, got {width}x{height}"
            )));
        }
        let packed_stride = width as usize * 3;
        if row_stride < packed_stride {
            return Err(Error::Processor(format!(
                "RGB8 row stride {row_stride} is smaller than packed row size {packed_stride}"
            )));
        }
        let required = row_stride
            .checked_mul(height.saturating_sub(1) as usize)
            .and_then(|prefix| prefix.checked_add(packed_stride))
            .ok_or_else(|| Error::Processor("image buffer dimensions overflow".to_string()))?;
        if pixels.len() < required {
            return Err(Error::Processor(format!(
                "RGB8 image requires at least {required} bytes, got {}",
                pixels.len()
            )));
        }
        Ok(Self {
            pixels,
            width,
            height,
            row_stride,
        })
    }

    /// Image width in pixels.
    pub fn width(self) -> u32 {
        self.width
    }

    /// Image height in pixels.
    pub fn height(self) -> u32 {
        self.height
    }

    fn packed_pixels(self) -> Vec<u8> {
        let packed_stride = self.width as usize * 3;
        if self.row_stride == packed_stride {
            return self.pixels[..packed_stride * self.height as usize].to_vec();
        }
        let mut packed = Vec::with_capacity(packed_stride * self.height as usize);
        for row in 0..self.height as usize {
            let start = row * self.row_stride;
            packed.extend_from_slice(&self.pixels[start..start + packed_stride]);
        }
        packed
    }
}

/// Owned, tightly packed RGB8 image.
#[derive(Debug, Clone)]
pub struct RgbImage {
    pixels: Vec<u8>,
    width: u32,
    height: u32,
}

impl RgbImage {
    /// Borrows this image as an RGB8 view.
    pub fn as_view(&self) -> RgbImageView<'_> {
        RgbImageView {
            pixels: &self.pixels,
            width: self.width,
            height: self.height,
            row_stride: self.width as usize * 3,
        }
    }

    /// Returns tightly packed RGB8 pixels.
    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }

    /// Image width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Image height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }
}

/// Owned normalized image in channel-first `[channels, height, width]` order.
#[derive(Debug, Clone)]
pub struct NormalizedImage {
    data: Vec<f32>,
    channels: usize,
    width: usize,
    height: usize,
}

impl NormalizedImage {
    /// Returns normalized channel-first pixels.
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Number of channels.
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Image width in pixels.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Image height in pixels.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Returns one channel-first pixel.
    pub fn get(&self, channel: usize, y: usize, x: usize) -> f32 {
        self.data[(channel * self.height + y) * self.width + x]
    }
}

/// Resizes an RGB8 image using bicubic interpolation.
pub fn resize_rgb8_bicubic(
    image: RgbImageView<'_>,
    width: u32,
    height: u32,
) -> Result<RgbImage, Error> {
    if width == 0 || height == 0 {
        return Err(Error::Processor(format!(
            "resize dimensions must be positive, got {width}x{height}"
        )));
    }
    if width == image.width && height == image.height {
        return Ok(RgbImage {
            pixels: image.packed_pixels(),
            width,
            height,
        });
    }
    let source =
        ImageBuffer::<Rgb<u8>, Vec<u8>>::from_raw(image.width, image.height, image.packed_pixels())
            .ok_or_else(|| Error::Processor("failed to construct RGB8 image buffer".to_string()))?;
    let resized = image::imageops::resize(&source, width, height, FilterType::CatmullRom);
    Ok(RgbImage {
        pixels: resized.into_raw(),
        width,
        height,
    })
}

/// Rescales and normalizes RGB8 pixels, returning channel-first data.
pub fn rescale_and_normalize_rgb8(
    image: RgbImageView<'_>,
    rescale_factor: f32,
    mean: [f32; 3],
    std: [f32; 3],
) -> Result<NormalizedImage, Error> {
    if !rescale_factor.is_finite() {
        return Err(Error::Processor(format!(
            "image rescale factor must be finite, got {rescale_factor}"
        )));
    }
    if std.iter().any(|value| *value == 0.0 || !value.is_finite()) {
        return Err(Error::Processor(format!(
            "image normalization standard deviations must be finite and nonzero, got {std:?}"
        )));
    }
    let width = image.width as usize;
    let height = image.height as usize;
    let mut data = vec![0.0f32; 3 * width * height];
    for y in 0..height {
        let row = &image.pixels[y * image.row_stride..][..width * 3];
        for x in 0..width {
            for channel in 0..3 {
                let value = row[x * 3 + channel] as f32 * rescale_factor;
                data[(channel * height + y) * width + x] = (value - mean[channel]) / std[channel];
            }
        }
    }
    Ok(NormalizedImage {
        data,
        channels: 3,
        width,
        height,
    })
}

#[cfg(test)]
mod tests {
    use super::{rescale_and_normalize_rgb8, resize_rgb8_bicubic, RgbImageView};

    #[test]
    fn image_view_honors_row_stride() {
        let pixels = [255, 0, 0, 9, 9, 9, 0, 255, 0];
        let view = RgbImageView::with_row_stride(&pixels, 1, 2, 6).unwrap();
        let normalized = rescale_and_normalize_rgb8(view, 1.0 / 255.0, [0.5; 3], [0.5; 3]).unwrap();
        assert_eq!(normalized.data(), &[1.0, -1.0, -1.0, 1.0, -1.0, -1.0]);
    }

    #[test]
    fn no_op_resize_tightly_packs_rows() {
        let pixels = [1, 2, 3, 0, 4, 5, 6];
        let view = RgbImageView::with_row_stride(&pixels, 1, 2, 4).unwrap();
        let resized = resize_rgb8_bicubic(view, 1, 2).unwrap();
        assert_eq!(resized.pixels(), &[1, 2, 3, 4, 5, 6]);
    }
}
