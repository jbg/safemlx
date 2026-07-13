use std::{path::PathBuf, time::Instant};

use safemlx::{transforms::eval, Array, Device, DeviceType, Dtype, ExecutionContext, Stream};
use safemlx_lm::{
    load_realtime_model,
    models::personaplex,
    quantization::AffineQuantization,
    realtime::{RealtimeSampling, RealtimeSpeechModel, RealtimeStepInput},
    sampler::DefaultSampler,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let quantize_on_load = args.iter().any(|value| value == "--quantize-on-load");
    let positional = args
        .iter()
        .filter(|value| value.as_str() != "--quantize-on-load")
        .collect::<Vec<_>>();
    let model_dir = positional
        .first()
        .map(|value| PathBuf::from(value.as_str()))
        .or_else(|| std::env::var_os("SAFEMLX_PERSONAPLEX_DIR").map(PathBuf::from))
        .expect("usage: personaplex_step_bench <model-dir> [frames] [--quantize-on-load]");
    let frames = positional
        .get(1)
        .and_then(|value| value.parse::<i32>().ok())
        .unwrap_or(64);

    println!("model_dir={}", model_dir.display());
    println!("frames={frames}");
    println!("quantize_on_load={quantize_on_load}");

    let ctx = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
    let stream = ctx.stream();
    let weights_ctx = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let weights_stream = weights_ctx.stream();

    if quantize_on_load {
        safemlx::memory::reset_peak_memory()?;
        let load_start = Instant::now();
        let mut model = personaplex::load_model_quantized(
            &model_dir,
            AffineQuantization::default(),
            stream,
            weights_stream,
        )?;
        stream.synchronize()?;
        println!("load_s={:.3}", load_start.elapsed().as_secs_f64());
        report_memory()?;
        benchmark_model(&mut model, frames, stream)?;
    } else {
        safemlx::memory::reset_peak_memory()?;
        let load_start = Instant::now();
        let mut model = load_realtime_model(&model_dir, stream, weights_stream)?;
        stream.synchronize()?;
        println!("load_s={:.3}", load_start.elapsed().as_secs_f64());
        report_memory()?;
        benchmark_model(&mut model, frames, stream)?;
    }
    Ok(())
}

fn report_memory() -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "mlx_active_memory_bytes={}",
        safemlx::memory::active_memory()?
    );
    println!("mlx_peak_memory_bytes={}", safemlx::memory::peak_memory()?);
    println!(
        "mlx_cache_memory_bytes={}",
        safemlx::memory::cache_memory()?
    );
    Ok(())
}

fn benchmark_model<M: RealtimeSpeechModel>(
    model: &mut M,
    frames: i32,
    stream: &Stream,
) -> Result<(), Box<dyn std::error::Error>> {
    let input = personaplex::sine_frame(1, stream)?;
    let _ = run_steps(model, &input, 4, stream)?;
    let result = run_steps(model, &input, frames, stream)?;
    report(
        "personaplex_step",
        &result.latencies_ms,
        result.emitted_frames,
        result.token_hash,
    );
    Ok(())
}

struct StepBenchResult {
    latencies_ms: Vec<f64>,
    emitted_frames: i32,
    token_hash: u64,
}

fn run_steps<M: RealtimeSpeechModel>(
    model: &mut M,
    input: &safemlx::Array,
    frames: i32,
    stream: &Stream,
) -> Result<StepBenchResult, Box<dyn std::error::Error>> {
    let mut state = model.new_realtime_state();
    let config = model.realtime_config();
    let mut text_sampler = DefaultSampler;
    let mut audio_samplers = (0..config.depth_audio_codebooks)
        .map(|_| DefaultSampler)
        .collect::<Vec<_>>();
    let mut latencies_ms = Vec::with_capacity(frames as usize);
    let mut emitted_frames = 0;
    let mut token_hash = 0xcbf29ce484222325u64;

    for _ in 0..frames {
        let start = Instant::now();
        let output = model.step_realtime(
            &mut state,
            RealtimeStepInput::encoded_audio(input),
            RealtimeSampling::new(&mut text_sampler, &mut audio_samplers, 0.0, 0.0, None),
            stream,
        )?;
        if let Some(tokens) = &output.output_audio_tokens {
            eval([&output.text_token, &output.sampled_audio_tokens, tokens])?;
            emitted_frames += 1;
        } else {
            eval([&output.text_token, &output.sampled_audio_tokens])?;
        }
        stream.synchronize()?;
        latencies_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
        hash_token_array(&mut token_hash, &output.text_token)?;
        hash_token_array(&mut token_hash, &output.sampled_audio_tokens)?;
        if let Some(tokens) = &output.output_audio_tokens {
            hash_token_array(&mut token_hash, tokens)?;
        } else {
            hash_value(&mut token_hash, u64::MAX);
        }
    }

    Ok(StepBenchResult {
        latencies_ms,
        emitted_frames,
        token_hash,
    })
}

fn hash_value(hash: &mut u64, value: u64) {
    for byte in value.to_le_bytes() {
        *hash ^= u64::from(byte);
        *hash = hash.wrapping_mul(0x100000001b3);
    }
}

fn hash_token_array(hash: &mut u64, array: &Array) -> Result<(), Box<dyn std::error::Error>> {
    let evaluated = array.evaluated()?;
    match array.dtype() {
        Dtype::Int32 => {
            for &value in evaluated.as_slice::<i32>() {
                hash_value(hash, value as i64 as u64);
            }
        }
        Dtype::Uint32 => {
            for &value in evaluated.as_slice::<u32>() {
                hash_value(hash, u64::from(value));
            }
        }
        Dtype::Int64 => {
            for &value in evaluated.as_slice::<i64>() {
                hash_value(hash, value as u64);
            }
        }
        Dtype::Uint64 => {
            for &value in evaluated.as_slice::<u64>() {
                hash_value(hash, value);
            }
        }
        dtype => return Err(format!("unexpected token dtype: {dtype:?}").into()),
    }
    Ok(())
}

fn report(name: &str, latencies_ms: &[f64], emitted_frames: i32, token_hash: u64) {
    let mut sorted = latencies_ms.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let total_ms = latencies_ms.iter().sum::<f64>();
    let mean_ms = total_ms / latencies_ms.len() as f64;
    println!("emitted_frames={emitted_frames}");
    println!("token_hash={token_hash:016x}");
    println!(
        "{name}_mean_ms={mean_ms:.3} {name}_p50_ms={:.3} {name}_p95_ms={:.3} {name}_p99_ms={:.3}",
        percentile(&sorted, 0.50),
        percentile(&sorted, 0.95),
        percentile(&sorted, 0.99)
    );
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let index = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[index]
}
