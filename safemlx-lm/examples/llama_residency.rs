//! Compare Llama residency policies with an opt-in local checkpoint benchmark.

use std::{path::PathBuf, time::Instant};

use clap::Parser;
use safemlx::{Array, Device, DeviceType, ExecutionContext};
use safemlx_lm::{
    layerwise::LayerwiseLoadOptions,
    llama::{load_llama_model, LlamaLoadOptions},
    models::llama,
    offload::{MemoryTier, OffloadConfig, TransferDirection},
};

#[derive(Debug, Parser)]
#[command(about = "Measure synchronous Llama decoder-layer host transfers")]
struct Args {
    /// Directory containing config.json and Llama-compatible safetensors.
    model_dir: PathBuf,
    /// Comma-separated prompt token ids.
    #[arg(long, default_value = "1,2,3,4")]
    prompt_tokens: String,
    /// Number of greedy decode tokens.
    #[arg(long, default_value_t = 8)]
    decode_tokens: usize,
    /// Maximum decoder layers resident on the execution device.
    #[arg(long, default_value_t = 1)]
    device_layer_window: usize,
    /// Optional device parameter budget in bytes.
    #[arg(long)]
    device_budget_bytes: Option<u64>,
    /// Optional host parameter budget in bytes.
    #[arg(long)]
    host_budget_bytes: Option<u64>,
    /// Maximum simultaneously mapped checkpoint shards.
    #[arg(long, default_value_t = 4)]
    mapped_shards: usize,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let prompt = args
        .prompt_tokens
        .split(',')
        .map(|value| value.trim().parse::<u32>())
        .collect::<Result<Vec<_>, _>>()?;
    anyhow::ensure!(!prompt.is_empty(), "prompt token list must not be empty");

    let execution = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
    let weights = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let config = OffloadConfig::new(
        args.device_budget_bytes,
        args.host_budget_bytes,
        args.device_layer_window,
    )?;
    let layerwise = LayerwiseLoadOptions {
        offload: config,
        max_mapped_shards: args.mapped_shards,
        sample_mlx_memory: true,
        sample_process_memory: true,
        ..LayerwiseLoadOptions::default()
    };
    let mut model = load_llama_model(
        &args.model_dir,
        LlamaLoadOptions::layerwise_host(layerwise),
        stream,
        weights.stream(),
    )?;
    let metadata = model
        .layerwise_metadata()
        .expect("layerwise residency was selected")
        .clone();
    let mut cache = model.new_cache();

    stream.synchronize()?;
    let prompt_array = Array::from_slice(&prompt, &[1, prompt.len() as i32]);
    let prefill_started = Instant::now();
    let mut logits = model.prefill(&prompt_array, &mut cache, stream)?;
    stream.synchronize()?;
    let prefill = prefill_started.elapsed();
    let time_to_first_token = prefill;

    let decode_started = Instant::now();
    for _ in 0..args.decode_tokens {
        let token = llama::sample(&logits, 0.0, None, stream)?;
        stream.synchronize()?;
        let token = token.reshape(&[1, 1], stream)?;
        logits = model.decode(&token, &mut cache, stream)?;
        stream.synchronize()?;
    }
    let decode = decode_started.elapsed();
    let report = model
        .residency_report()?
        .expect("layerwise residency was selected");
    let offload = report.offload();
    let observed_window = report
        .units()
        .iter()
        .filter(|unit| unit.id().as_str().starts_with("llama.layer.") && unit.device_resident())
        .count();

    println!("model type: {}", metadata.model_type());
    println!("quantization: {:?}", metadata.quantization());
    println!(
        "static device-weight bytes: {}",
        metadata.static_device_bytes()
    );
    println!(
        "host-resident decoder bytes: {}",
        metadata.host_layer_bytes()
    );
    println!(
        "configured/observed device-layer window: {}/{}",
        metadata.device_layer_window(),
        observed_window
    );
    println!(
        "current logical host/device parameter bytes: {}/{}",
        offload.resident_bytes().get(MemoryTier::Host),
        offload.resident_bytes().get(MemoryTier::Device)
    );
    println!(
        "peak logical host/device parameter bytes: {}/{}",
        offload.peak_resident_bytes().get(MemoryTier::Host),
        offload.peak_resident_bytes().get(MemoryTier::Device)
    );
    for direction in TransferDirection::ALL {
        let transfer = offload.transfer(direction);
        if transfer.count() > 0 {
            println!(
                "synchronous {direction:?}: {} transfers, {} bytes, {:?}",
                transfer.count(),
                transfer.bytes(),
                transfer.duration()
            );
        }
    }
    println!("prefill latency: {prefill:?}");
    println!(
        "prefill throughput: {:.2} tokens/s",
        prompt.len() as f64 / prefill.as_secs_f64()
    );
    println!("decode latency: {decode:?}");
    if args.decode_tokens > 0 {
        println!(
            "decode throughput: {:.2} tokens/s",
            args.decode_tokens as f64 / decode.as_secs_f64()
        );
    }
    println!("time to first token: {time_to_first_token:?}");
    println!("MLX memory: {:?}", offload.mlx_memory());
    println!("weight-store diagnostics: {:?}", report.weight_store());
    println!("KV cache and activations are not included in logical parameter residency totals.");
    #[cfg(target_os = "macos")]
    println!(
        "Metal uses unified physical memory; this run validates scheduling and logical residency, not expanded model capacity."
    );
    Ok(())
}
