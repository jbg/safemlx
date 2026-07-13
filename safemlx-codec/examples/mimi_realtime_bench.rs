use std::{path::PathBuf, time::Instant};

use safemlx::{
    ops::{concatenate_axis, indexing::TryIndexOp},
    transforms::eval,
    Array, Device, DeviceType, ExecutionContext,
};
use safemlx_codec::mimi::Mimi;

const SAMPLE_RATE: f64 = 24_000.0;
const FRAME_RATE: f64 = 12.5;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let checkpoint = args
        .first()
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("SAFEMLX_MIMI_PATH").map(PathBuf::from))
        .expect("usage: mimi_realtime_bench <tokenizer.safetensors> [frames] [codebooks]");
    let frames = args
        .get(1)
        .and_then(|value| value.parse::<i32>().ok())
        .unwrap_or(64);
    let codebooks = args
        .get(2)
        .and_then(|value| value.parse::<i32>().ok())
        .unwrap_or(8);
    let frame_samples = (SAMPLE_RATE / FRAME_RATE) as i32;
    let audio_s = frames as f64 / FRAME_RATE;

    println!("checkpoint={}", checkpoint.display());
    println!("frames={frames}");
    println!("codebooks={codebooks}");
    println!("frame_samples={frame_samples}");
    println!("audio_s={audio_s:.3}");

    let ctx = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
    let stream = ctx.stream();

    let load_start = Instant::now();
    let mut mimi = Mimi::load(&checkpoint, Some(codebooks), stream)?;
    stream.synchronize()?;
    println!("load_s={:.3}", load_start.elapsed().as_secs_f64());

    let pcm = Array::zeros::<f32>(&[1, 1, frames * frame_samples], stream)?;
    let pcm_frame = Array::zeros::<f32>(&[1, 1, frame_samples], stream)?;
    let codes = Array::zeros::<i32>(&[1, codebooks, frames], stream)?;
    let code_frames = (0..frames)
        .map(|frame| codes.try_index_device((.., .., frame), stream))
        .collect::<Result<Vec<_>, _>>()?;

    // Warm up kernels and allocation paths with the same shapes used below.
    let warm_tokens = mimi.encode(&pcm_frame, stream)?;
    let warm_pcm = mimi.decode(&warm_tokens, stream)?;
    eval([&warm_pcm])?;
    let warm_encoded = mimi.encode(&pcm, stream)?;
    let warm_decoded = mimi.decode(&warm_encoded, stream)?;
    eval([&warm_decoded])?;
    mimi.reset_decode_state();
    for frame in code_frames.iter().take(4) {
        let chunk = mimi.decode_step(frame, stream)?;
        eval([&chunk])?;
    }
    mimi.reset_decode_state();
    stream.synchronize()?;

    let encoded = time("offline_encode", audio_s, || {
        let encoded = mimi.encode(&pcm, stream)?;
        eval([&encoded])?;
        stream.synchronize()?;
        Ok(encoded)
    })?;

    time("offline_decode", audio_s, || {
        let decoded = mimi.decode(&encoded, stream)?;
        eval([&decoded])?;
        stream.synchronize()?;
        Ok(())
    })?;

    mimi.reset_decode_state();
    time("streaming_decode", audio_s, || {
        let mut chunks = Vec::with_capacity(frames as usize);
        for frame in &code_frames {
            let chunk = mimi.decode_step(frame, stream)?;
            eval([&chunk])?;
            chunks.push(chunk);
        }
        let decoded = concatenate_axis(&chunks, 2, stream)?;
        eval([&decoded])?;
        stream.synchronize()?;
        Ok(())
    })?;

    mimi.reset_decode_state();
    time("frame_encode_then_stream_decode", audio_s, || {
        let mut chunks = Vec::with_capacity(frames as usize);
        for _ in 0..frames {
            let tokens = mimi.encode(&pcm_frame, stream)?;
            let chunk = mimi.decode_step(&tokens, stream)?;
            eval([&chunk])?;
            chunks.push(chunk);
        }
        let decoded = concatenate_axis(&chunks, 2, stream)?;
        eval([&decoded])?;
        stream.synchronize()?;
        Ok(())
    })?;

    Ok(())
}

fn time<T>(
    name: &str,
    audio_s: f64,
    f: impl FnOnce() -> Result<T, Box<dyn std::error::Error>>,
) -> Result<T, Box<dyn std::error::Error>> {
    let start = Instant::now();
    let value = f()?;
    let elapsed_s = start.elapsed().as_secs_f64();
    let realtime_factor = elapsed_s / audio_s;
    let realtime_multiple = audio_s / elapsed_s;
    println!(
        "{name}_s={elapsed_s:.6} {name}_rtf={realtime_factor:.4} {name}_x_realtime={realtime_multiple:.2}"
    );
    Ok(value)
}
