//! Save, drop, lazily reopen, and continue a deterministic text prompt cache.

use std::path::PathBuf;

use clap::Parser;
use safemlx::{Array, Device, DeviceType, ExecutionContext};
use safemlx_lm::{
    models::{
        input::{InputPart, ModelInput},
        LoadedModel, ModelCache,
    },
    CacheResidencyPolicy, PagedCacheOptions, PromptCacheDescriptor, PromptCacheOptions,
    PromptCacheTopology,
};

#[derive(Debug, Parser)]
#[command(about = "Verify reusable paged prompt-cache parity")]
struct Args {
    /// Directory containing a supported text model.
    model_dir: PathBuf,
    /// Persistent prompt-cache destination.
    cache_dir: PathBuf,
    /// Deterministic text prefix.
    #[arg(long)]
    prompt: String,
    /// One token appended after the restored prefix.
    #[arg(long)]
    suffix_token: u32,
    /// Total decoder layer count from the normalized model configuration.
    #[arg(long)]
    layer_count: usize,
    /// Stable content-based checkpoint identity.
    #[arg(long)]
    checkpoint_fingerprint: String,
    /// Stable identity covering RoPE and cache-relevant architecture settings.
    #[arg(long)]
    architecture_fingerprint: String,
    /// Sliding attention window, when the model uses one.
    #[arg(long)]
    sliding_window: Option<i32>,
    /// Token count per immutable cache block.
    #[arg(long, default_value_t = 128)]
    block_tokens: i32,
    /// Finite logical execution-device cache bytes.
    #[arg(long, default_value_t = 512 << 20)]
    device_cache_bytes: u64,
    /// Finite logical host cache bytes.
    #[arg(long, default_value_t = 2 << 30)]
    host_cache_bytes: u64,
    /// Recent device blocks protected per layer.
    #[arg(long, default_value_t = 1)]
    recent_device_blocks: usize,
    /// Optional explicit live-cache backing directory.
    #[arg(long)]
    live_disk_dir: Option<PathBuf>,
    /// Finite logical live-cache disk bytes.
    #[arg(long, default_value_t = 8 << 30)]
    live_disk_bytes: u64,
    /// Replace an existing persistent cache directory atomically.
    #[arg(long)]
    replace: bool,
    /// Use the ordinary device-resident cache and skip persistence.
    #[arg(long)]
    device_cache: bool,
}

fn input_for_tokens(
    tokens: &Array,
    cache: &mut ModelCache,
    model: &mut LoadedModel,
    stream: &safemlx::Stream,
) -> Result<Array, safemlx::error::Exception> {
    let parts = [InputPart::text_token_ids(tokens)];
    model.prefill_input_with_cache(ModelInput::new(&parts), cache, stream)
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let execution = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
    let weights = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let mut model = LoadedModel::load(&args.model_dir, stream, weights.stream())?;
    let prefix_ids = model.encode(&args.prompt, false)?;
    anyhow::ensure!(
        !prefix_ids.is_empty(),
        "prompt must encode to at least one token"
    );
    let prefix = Array::from_slice(&prefix_ids, &[1, prefix_ids.len() as i32]);
    let suffix = Array::from_slice(&[args.suffix_token], &[1, 1]);

    if args.device_cache {
        let mut cache = model.new_cache();
        let _ = input_for_tokens(&prefix, &mut cache, &mut model, stream)?;
        let logits = input_for_tokens(&suffix, &mut cache, &mut model, stream)?;
        stream.synchronize()?;
        println!(
            "ordinary device cache suffix logits shape: {:?}",
            logits.shape()
        );
        return Ok(());
    }

    let mut paged = PagedCacheOptions::new(
        args.block_tokens,
        args.device_cache_bytes,
        args.host_cache_bytes,
        args.recent_device_blocks,
    )?
    .with_full_attention(args.sliding_window.is_none())
    .with_persistence_retention(true)
    .with_process_sampling(true);
    if let Some(directory) = &args.live_disk_dir {
        paged = paged.with_live_disk(directory, args.live_disk_bytes, 2)?;
    }

    let mut uninterrupted =
        model.new_cache_with_options(CacheResidencyPolicy::Paged(paged.clone()))?;
    let _ = input_for_tokens(&prefix, &mut uninterrupted, &mut model, stream)?;
    let uninterrupted_logits = input_for_tokens(&suffix, &mut uninterrupted, &mut model, stream)?;
    stream.synchronize()?;
    println!(
        "uninterrupted report: {:#?}",
        uninterrupted.residency_report()?
    );

    let mut persisted = model.new_cache_with_options(CacheResidencyPolicy::Paged(paged.clone()))?;
    let _ = input_for_tokens(&prefix, &mut persisted, &mut model, stream)?;
    let model_type = model.model_type().to_owned();
    let model_family = if model_type.contains("deepseek") {
        "deepseek_v3"
    } else {
        "llama"
    };
    let descriptor = PromptCacheDescriptor {
        model_family: model_family.into(),
        effective_model_type: model_type,
        checkpoint_fingerprint: args.checkpoint_fingerprint,
        architecture_fingerprint: args.architecture_fingerprint,
        layer_count: args.layer_count,
        global_layer_start: 0,
        global_layer_end: args.layer_count,
        batch_size: 1,
        sliding_window: args.sliding_window,
        sink_tokens: 0,
        topology: PromptCacheTopology::default(),
    };
    let manifest = persisted.save_prompt_cache(
        &args.cache_dir,
        descriptor.clone(),
        &prefix_ids,
        &PromptCacheOptions {
            application_namespace: Some("paged-prompt-cache-example".into()),
            replace_existing: args.replace,
        },
    )?;
    println!("saved blocks: {}", manifest.blocks.len());
    println!("save report: {:#?}", persisted.residency_report()?);
    drop(persisted);

    let (mut restored, inspected) =
        model.load_prompt_cache(&args.cache_dir, &descriptor, &prefix_ids, paged)?;
    println!("cataloged blocks: {}", inspected.blocks.len());
    println!("load report: {:#?}", restored.residency_report()?);
    let restored_logits = input_for_tokens(&suffix, &mut restored, &mut model, stream)?;
    stream.synchronize()?;
    let equal = restored_logits
        .all_close(&uninterrupted_logits, 1e-4, 1e-4, None, stream)?
        .item::<bool>(stream);
    println!("restored suffix logits match uninterrupted execution: {equal}");
    println!("continued report: {:#?}", restored.residency_report()?);
    anyhow::ensure!(
        equal,
        "restored suffix logits differ from uninterrupted execution"
    );
    Ok(())
}
