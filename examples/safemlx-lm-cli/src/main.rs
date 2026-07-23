use std::{
    io::{self, IsTerminal, Read, Write},
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{bail, Context, Result};
use clap::{Parser, ValueEnum};
use hf_hub::{cache::CachedRevisionInfo, HFClientSync};
use safemlx::{
    error::Exception,
    ops::indexing::TryIndexOp,
    random::RandomState,
    transforms::{async_eval, eval},
    Array, Device, DeviceType, ExecutionContext, Stream,
};
use safemlx_lm::{
    expert_cache::{ExpertCacheLoadOptions, ExpertPassStatistics, ExpertTierStatistics},
    layerwise::{LayerwiseLoadOptions, WeightResidency},
    models::{
        input::{InputPart, ModelInput},
        LoadedModel, ModelLoadOptions, TextDecoder,
    },
    mtp::{LoadedDrafter, MtpConfig, MtpStats},
    offload::{CacheEvictionPolicy, MemoryTier, OffloadConfig, TransferDirection},
    quantization::AffineQuantization,
    sampler::{DefaultSampler, GenerationSampler, Sampler, SpeculativeSampler},
};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ExpertCacheEviction {
    Lru,
    Lfu,
}

impl From<ExpertCacheEviction> for CacheEvictionPolicy {
    fn from(value: ExpertCacheEviction) -> Self {
        match value {
            ExpertCacheEviction::Lru => Self::LeastRecentlyUsed,
            ExpertCacheEviction::Lfu => Self::LeastFrequentlyUsed,
        }
    }
}

enum CliSampler {
    Greedy(DefaultSampler),
    Configured(GenerationSampler),
}

impl Sampler for CliSampler {
    fn sample(
        &mut self,
        logits: &Array,
        temp: f32,
        prng_state: Option<&mut RandomState>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        match self {
            Self::Greedy(sampler) => sampler.sample(logits, temp, prng_state, stream),
            Self::Configured(sampler) => sampler.sample(logits, temp, prng_state, stream),
        }
    }
}

impl SpeculativeSampler for CliSampler {
    fn process_logits(
        &self,
        logits: &Array,
        temperature: f32,
        history: &[u32],
        stream: &Stream,
    ) -> Result<Array, Exception> {
        match self {
            Self::Greedy(sampler) => sampler.process_logits(logits, temperature, history, stream),
            Self::Configured(sampler) => {
                sampler.process_logits(logits, temperature, history, stream)
            }
        }
    }
}

/// Generate text with a model supported by safemlx-lm.
#[derive(Debug, Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    /// Model directory, GGUF file, or cached Hugging Face model identifier.
    /// Append `:QUANT` to select a cached GGUF quantization.
    #[arg(short, long, value_name = "PATH_OR_ID")]
    model: String,

    /// Explicit Gemma assistant directory, GGUF file, or Hugging Face identifier.
    #[arg(long, value_name = "PATH_OR_ID")]
    draft_model: Option<String>,

    /// Maximum speculative tokens proposed before each target verification.
    #[arg(long, default_value_t = 3, value_name = "TOKENS")]
    mtp_draft_tokens: usize,

    /// Prompt text. Reads the prompt from stdin when omitted and stdin is piped.
    #[arg(value_name = "PROMPT")]
    prompt: Option<String>,

    /// Cached Hugging Face revision (a ref such as `main` or a commit hash).
    #[arg(long, value_name = "REVISION")]
    revision: Option<String>,

    /// Maximum number of tokens to generate. Generation stops earlier at EOS.
    #[arg(short = 'n', long, default_value_t = 256, value_name = "TOKENS")]
    max_tokens: usize,

    /// Sampling temperature. Zero selects greedy decoding.
    #[arg(short = 't', long, default_value_t = 0.0, value_name = "FLOAT")]
    temperature: f32,

    /// Keep only the K most likely tokens. Zero disables top-k filtering.
    #[arg(long, default_value_t = 40, value_name = "K")]
    top_k: i32,

    /// Keep the smallest token set with at least this cumulative probability.
    #[arg(long, default_value_t = 0.95, value_name = "FLOAT")]
    top_p: f32,

    /// Remove tokens below this fraction of the most likely token's probability.
    #[arg(long, default_value_t = 0.05, value_name = "FLOAT")]
    min_p: f32,

    /// Penalty for repeating a token. One disables the penalty.
    #[arg(long, default_value_t = 1.0, value_name = "FLOAT")]
    repeat_penalty: f32,

    /// Number of generated tokens considered for repetition penalties; -1 means all.
    #[arg(long, default_value_t = 64, value_name = "TOKENS")]
    repeat_last_n: i32,

    /// Penalty proportional to the number of times a token was generated.
    #[arg(long, default_value_t = 0.0, value_name = "FLOAT")]
    frequency_penalty: f32,

    /// Penalty applied once when a token has already been generated.
    #[arg(long, default_value_t = 0.0, value_name = "FLOAT")]
    presence_penalty: f32,

    /// Random seed used when temperature is non-zero.
    #[arg(long, default_value_t = 0)]
    seed: u64,

    /// Quantize eligible dense weights to this bit width while loading.
    #[arg(long, value_name = "BITS")]
    quantize: Option<i32>,

    /// Number of adjacent weights sharing quantization parameters.
    #[arg(long, default_value_t = 64, value_name = "WEIGHTS")]
    quantization_group_size: i32,

    /// Keep repeated model layers on the host and use a bounded device window.
    #[arg(long)]
    layerwise_host: bool,

    /// Cache routed experts independently for any supported safetensors MoE model.
    #[arg(long)]
    expert_cache: bool,

    /// Maximum repeated layers resident on the execution device.
    #[arg(long, default_value_t = 1, value_name = "LAYERS")]
    device_layer_window: usize,

    /// Optional logical device parameter budget in bytes.
    #[arg(long, value_name = "BYTES")]
    device_budget_bytes: Option<u64>,

    /// Optional logical host parameter budget in bytes.
    #[arg(long, value_name = "BYTES")]
    host_budget_bytes: Option<u64>,

    /// Optional logical device budget for cached routed experts.
    #[arg(long, value_name = "BYTES")]
    expert_cache_device_budget_bytes: Option<u64>,

    /// Optional logical host budget for cached routed experts; zero uses disk fallback.
    #[arg(long, value_name = "BYTES")]
    expert_cache_host_budget_bytes: Option<u64>,

    /// Maximum temporary compact expert-bank allocation per routed block.
    #[arg(long, default_value_t = 1_073_741_824, value_name = "BYTES")]
    expert_cache_scratch_bytes: u64,

    /// Deterministic expert cache eviction ordering.
    #[arg(long, value_enum, default_value_t = ExpertCacheEviction::Lru)]
    expert_cache_eviction: ExpertCacheEviction,

    /// Measure cold prefill, repeated prefill, and one cached decode separately.
    #[arg(long)]
    expert_cache_benchmark: bool,

    /// Maximum simultaneously mapped safetensors payload shards.
    #[arg(long, default_value_t = 4, value_name = "SHARDS")]
    mapped_shards: usize,

    /// Pass the prompt directly instead of applying the model's chat template.
    #[arg(long)]
    raw: bool,

    /// Print model resolution and generation statistics to stderr.
    #[arg(short, long)]
    verbose: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum StopReason {
    Eos,
    MaxTokens,
    GeneratorExhausted,
}

impl StopReason {
    const fn label(self) -> &'static str {
        match self {
            Self::Eos => "eos",
            Self::MaxTokens => "max_tokens",
            Self::GeneratorExhausted => "generator_exhausted",
        }
    }
}

fn stop_reason(output_ids: &[u32], eos_token_ids: &[u32], max_tokens: usize) -> StopReason {
    if output_ids
        .last()
        .is_some_and(|token| eos_token_ids.contains(token))
    {
        StopReason::Eos
    } else if output_ids.len() >= max_tokens {
        StopReason::MaxTokens
    } else {
        StopReason::GeneratorExhausted
    }
}

fn should_report_stop_reason(stop_reason: StopReason, verbose: bool) -> bool {
    verbose || stop_reason == StopReason::MaxTokens
}

fn write_streamed_token(
    decoder: &mut TextDecoder,
    stdout: &mut impl Write,
    streamed_text: &mut String,
    token_id: u32,
) -> Result<()> {
    if let Some(text) = decoder.step(token_id)? {
        stdout.write_all(text.as_bytes())?;
        stdout.flush()?;
        streamed_text.push_str(&text);
    }
    Ok(())
}

fn main() -> Result<()> {
    let total_started = Instant::now();
    let args = Cli::parse();
    validate_args(&args)?;
    let prompt = read_prompt(args.prompt.as_deref())?;
    let model_path = resolve_model(&args.model, args.revision.as_deref())?;
    let draft_model_path = args
        .draft_model
        .as_deref()
        .map(|source| resolve_model(source, args.revision.as_deref()))
        .transpose()?;

    if args.verbose {
        eprintln!("--- safemlx diagnostics (stderr) ---");
        eprintln!("model: {}", model_path.display());
        if let Some(path) = &draft_model_path {
            eprintln!("draft_model: {}", path.display());
        }
    }

    let execution = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
    let weights = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    if args.verbose {
        // Capture the complete model-load and generation high-water mark.
        safemlx::memory::reset_peak_memory()?;
    }
    let load_started = Instant::now();
    let mut load_options = match args.quantize {
        Some(bits) => ModelLoadOptions::with_quantization(AffineQuantization::new(
            args.quantization_group_size,
            bits,
        )?),
        None => ModelLoadOptions::default(),
    };
    if args.expert_cache {
        let non_expert = LayerwiseLoadOptions {
            offload: OffloadConfig::new(
                args.device_budget_bytes,
                args.host_budget_bytes,
                args.device_layer_window,
            )?,
            max_mapped_shards: args.mapped_shards,
            sample_mlx_memory: args.verbose,
            sample_process_memory: args.verbose,
            ..LayerwiseLoadOptions::default()
        };
        let experts = OffloadConfig::new(
            args.expert_cache_device_budget_bytes,
            args.expert_cache_host_budget_bytes,
            1,
        )?
        .with_eviction_policy(args.expert_cache_eviction.into());
        load_options = load_options.with_weight_residency(WeightResidency::SparseExpertCache(
            ExpertCacheLoadOptions::new(non_expert, experts, args.expert_cache_scratch_bytes)?,
        ));
    } else if args.layerwise_host {
        let offload = OffloadConfig::new(
            args.device_budget_bytes,
            args.host_budget_bytes,
            args.device_layer_window,
        )?;
        load_options = load_options.with_weight_residency(WeightResidency::LayerwiseHost(
            LayerwiseLoadOptions {
                offload,
                max_mapped_shards: args.mapped_shards,
                sample_mlx_memory: args.verbose,
                sample_process_memory: args.verbose,
                ..LayerwiseLoadOptions::default()
            },
        ));
    }
    let mut model =
        LoadedModel::load_with_options(&model_path, load_options, stream, weights.stream())
            .with_context(|| format!("failed to load model from {}", model_path.display()))?;
    if draft_model_path.is_some()
        && !matches!(
            model.mtp_capability(),
            safemlx_lm::mtp::MtpCapability::Ready {
                checkpoint: safemlx_lm::mtp::MtpCheckpointKind::Separate
            }
        )
    {
        bail!(
            "--draft-model cannot be used with target capability {:?}",
            model.mtp_capability()
        );
    }
    let mut drafter = draft_model_path
        .as_ref()
        .map(|path| {
            let options = match args.quantize {
                Some(bits) => ModelLoadOptions::with_quantization(AffineQuantization::new(
                    args.quantization_group_size,
                    bits,
                )?),
                None => ModelLoadOptions::default(),
            };
            LoadedDrafter::load_with_options(path, options, stream, weights.stream())
                .with_context(|| format!("failed to load draft model from {}", path.display()))
        })
        .transpose()?;
    stream.synchronize()?;
    let load_elapsed = load_started.elapsed();

    let (rendered_prompt, add_special_tokens) = if args.raw {
        (prompt, true)
    } else {
        match model.apply_chat_template_json(
            vec![vec![serde_json::json!({
                "role": "user",
                "content": prompt,
            })]],
            None,
            true,
        )? {
            Some(rendered) => (rendered, false),
            None => (prompt, true),
        }
    };

    let tokens = model.encode_to_array(&rendered_prompt, add_special_tokens, stream)?;
    if tokens.shape()[1] == 0 {
        bail!("the prompt produced no input tokens");
    }

    if args.expert_cache_benchmark {
        run_expert_cache_benchmark(&mut model, &tokens, stream)?;
    }

    let eos_token_ids = model.eos_token_ids().to_vec();
    let mut cache = model.new_cache();
    let configured_sampler = GenerationSampler::new()
        .top_k(args.top_k)
        .top_p(args.top_p)
        .min_p(args.min_p)
        .penalties(
            args.repeat_penalty,
            args.repeat_last_n,
            args.frequency_penalty,
            args.presence_penalty,
        );
    // Probability filters cannot change a greedy argmax. Avoid their full-vocabulary
    // sorting and softmax work, as well as GenerationSampler's token-history readback,
    // when no repetition/frequency/presence penalty is active.
    let penalties_active =
        args.repeat_penalty != 1.0 || args.frequency_penalty != 0.0 || args.presence_penalty != 0.0;
    let sampler = if args.temperature == 0.0 && !penalties_active {
        CliSampler::Greedy(DefaultSampler)
    } else {
        CliSampler::Configured(configured_sampler)
    };
    let prng_key = (args.temperature != 0.0)
        .then(|| safemlx::random::key(args.seed))
        .transpose()?;
    let mut output_ids = Vec::with_capacity(args.max_tokens);
    let generation_started = Instant::now();
    let mut time_to_first_token = None;
    let mut mtp_stats: Option<MtpStats> = None;
    let mut decoder = model.text_decoder(true);
    let mut streamed_text = String::new();
    let stdout = io::stdout();
    let mut stdout = stdout.lock();

    if args.verbose {
        eprintln!("--- generated content (stdout) ---");
    }

    let embedded_mtp = args.mtp_draft_tokens > 0
        && matches!(
            model.mtp_capability(),
            safemlx_lm::mtp::MtpCapability::Ready {
                checkpoint: safemlx_lm::mtp::MtpCheckpointKind::Embedded
            }
        );
    if let Some(drafter) = drafter.as_mut() {
        let parts = [InputPart::text_token_ids(&tokens)];
        let input = ModelInput::new(&parts);
        let config = MtpConfig {
            max_tokens: args.max_tokens,
            max_draft_tokens: args.mtp_draft_tokens,
            temperature: args.temperature,
            eos_token_ids: eos_token_ids.clone(),
        };
        let (tokens, stats) = model.generate_mtp_input_with_sampler_callback(
            drafter,
            &mut cache,
            input,
            &config,
            prng_key,
            &sampler,
            stream,
            |token_id| {
                if time_to_first_token.is_none() {
                    time_to_first_token = Some(generation_started.elapsed());
                }
                if eos_token_ids.contains(&token_id) {
                    Ok(())
                } else {
                    write_streamed_token(&mut decoder, &mut stdout, &mut streamed_text, token_id)
                        .map_err(|error| Exception::custom(error.to_string()))
                }
            },
        )?;
        output_ids = tokens;
        mtp_stats = Some(stats);
    } else if embedded_mtp {
        let parts = [InputPart::text_token_ids(&tokens)];
        let input = ModelInput::new(&parts);
        let config = MtpConfig {
            max_tokens: args.max_tokens,
            max_draft_tokens: args.mtp_draft_tokens,
            temperature: args.temperature,
            eos_token_ids: eos_token_ids.clone(),
        };
        let (tokens, stats) = model.generate_embedded_mtp_input_with_sampler_callback(
            &mut cache,
            input,
            &config,
            prng_key,
            &sampler,
            stream,
            |token_id| {
                if time_to_first_token.is_none() {
                    time_to_first_token = Some(generation_started.elapsed());
                }
                if eos_token_ids.contains(&token_id) {
                    Ok(())
                } else {
                    write_streamed_token(&mut decoder, &mut stdout, &mut streamed_text, token_id)
                        .map_err(|error| Exception::custom(error.to_string()))
                }
            },
        )?;
        output_ids = tokens;
        mtp_stats = Some(stats);
    } else {
        let parts = [InputPart::text_token_ids(&tokens)];
        let input = ModelInput::new(&parts);
        let mut generator = model.generate_input_with_cache_sampler(
            &mut cache,
            args.temperature,
            input,
            prng_key,
            stream,
            sampler,
        );

        let mut current = generator.next().transpose()?;
        for index in 0..args.max_tokens {
            let Some(token) = current.take() else {
                break;
            };
            // Start the following decode before reading the current token back to
            // the CPU. This mirrors mlx-lm's one-token async evaluation pipeline.
            let next = if index + 1 < args.max_tokens {
                let next = generator.next();
                if let Some(Ok(next_token)) = next.as_ref() {
                    async_eval([next_token])?;
                }
                next
            } else {
                None
            };
            let token_id = token.item::<u32>(stream);
            if time_to_first_token.is_none() {
                time_to_first_token = Some(generation_started.elapsed());
            }
            output_ids.push(token_id);
            if eos_token_ids.contains(&token_id) {
                break;
            }
            write_streamed_token(&mut decoder, &mut stdout, &mut streamed_text, token_id)?;
            current = next.transpose()?;
        }
    }
    let generation_elapsed = generation_started.elapsed();
    let stop_reason = stop_reason(&output_ids, &eos_token_ids, args.max_tokens);
    if stop_reason == StopReason::Eos {
        output_ids.pop();
    }

    let output = model.decode(&output_ids, true)?;
    let remaining = output
        .strip_prefix(&streamed_text)
        .with_context(|| "incremental tokenizer output did not match the final decoded response")?;
    stdout.write_all(remaining.as_bytes())?;
    if !output.ends_with('\n') {
        writeln!(stdout)?;
    }
    stdout.flush()?;

    if args.verbose {
        eprintln!("--- safemlx diagnostics (stderr) ---");
    }
    if should_report_stop_reason(stop_reason, args.verbose) {
        eprintln!("stop_reason: {}", stop_reason.label());
    }

    if args.verbose {
        stream.synchronize()?;
        let peak_memory = safemlx::memory::peak_memory()?;
        let active_memory = safemlx::memory::active_memory()?;
        let cache_memory = safemlx::memory::cache_memory()?;
        let total_elapsed = total_started.elapsed();
        let token_rate = if generation_elapsed.is_zero() {
            0.0
        } else {
            output_ids.len() as f64 / generation_elapsed.as_secs_f64()
        };
        eprintln!(
            "model_type: {}, prompt_tokens: {}, generated_tokens: {}",
            model.model_type(),
            tokens.shape()[1],
            output_ids.len(),
        );
        eprintln!("load_time: {:.3} s", load_elapsed.as_secs_f64());
        eprintln!("generation_time: {:.3} s", generation_elapsed.as_secs_f64());
        if let Some(stats) = &mtp_stats {
            eprintln!(
                "mtp_rounds: {}, mtp_draft_tokens: {}, mtp_accepted_tokens: {}, mtp_accept_rate: {:.3}",
                stats.rounds,
                stats.draft_tokens,
                stats.accepted_tokens,
                stats.accept_rate(),
            );
            eprintln!("mtp_accept_lens: {:?}", stats.accept_lens);
        }
        match time_to_first_token {
            Some(elapsed) => {
                eprintln!("time_to_first_token: {:.3} s", elapsed.as_secs_f64());
                let decode_tokens = output_ids.len().saturating_sub(1);
                let decode_elapsed = generation_elapsed.saturating_sub(elapsed);
                let decode_token_rate = if decode_elapsed.is_zero() {
                    0.0
                } else {
                    decode_tokens as f64 / decode_elapsed.as_secs_f64()
                };
                eprintln!("decode_token_rate: {decode_token_rate:.2} tokens/s");
            }
            None => eprintln!("time_to_first_token: n/a"),
        }
        eprintln!("token_rate: {token_rate:.2} tokens/s");
        eprintln!("total_execution_time: {:.3} s", total_elapsed.as_secs_f64());
        eprintln!("mlx_peak_memory: {}", format_bytes(peak_memory));
        eprintln!("mlx_active_memory: {}", format_bytes(active_memory));
        eprintln!("mlx_cache_memory: {}", format_bytes(cache_memory));
        if let Some(report) = model.residency_report()? {
            let offload = report.offload();
            eprintln!(
                "residency_current_host_device: {} / {} bytes",
                offload.resident_bytes().get(MemoryTier::Host),
                offload.resident_bytes().get(MemoryTier::Device)
            );
            eprintln!(
                "residency_peak_host_device: {} / {} bytes",
                offload.peak_resident_bytes().get(MemoryTier::Host),
                offload.peak_resident_bytes().get(MemoryTier::Device)
            );
            for direction in TransferDirection::ALL {
                let transfer = offload.transfer(direction);
                if transfer.count() > 0 {
                    eprintln!(
                        "residency_{direction:?}: {} transfers, {} bytes",
                        transfer.count(),
                        transfer.bytes()
                    );
                }
            }
            eprintln!("weight_store: {:?}", report.weight_store());
        }
        if let Some(report) = model.expert_cache_report()? {
            eprintln!(
                "expert_cache_owned: {} experts, {} bytes",
                report.owned_experts, report.owned_bytes
            );
            eprintln!(
                "expert_cache_current_host_device: {} / {} experts, {} / {} bytes",
                report.host_resident_experts,
                report.device_resident_experts,
                report.host_resident_bytes,
                report.device_resident_bytes
            );
            eprintln!("expert_cache_prefill: {:?}", report.prefill);
            eprintln!("expert_cache_decode: {:?}", report.decode);
        }
        if eos_token_ids.is_empty() {
            eprintln!("warning: the model config contains no EOS token id");
        } else {
            match stop_reason {
                StopReason::MaxTokens => {
                    eprintln!("warning: generation reached --max-tokens before EOS");
                }
                StopReason::GeneratorExhausted => {
                    eprintln!("warning: the token generator ended before EOS");
                }
                StopReason::Eos => {}
            }
        }
    }

    Ok(())
}

fn format_bytes(bytes: usize) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = KIB * 1024.0;
    const GIB: f64 = MIB * 1024.0;

    let bytes_float = bytes as f64;
    let (value, unit) = if bytes_float >= GIB {
        (bytes_float / GIB, "GiB")
    } else if bytes_float >= MIB {
        (bytes_float / MIB, "MiB")
    } else if bytes_float >= KIB {
        (bytes_float / KIB, "KiB")
    } else {
        (bytes_float, "B")
    };
    format!("{value:.2} {unit} ({bytes} bytes)")
}

#[derive(Clone, Copy)]
struct ExpertBenchmarkSnapshot {
    prefill: ExpertPassStatistics,
    decode: ExpertPassStatistics,
    host_experts: usize,
    device_experts: usize,
    host_bytes: u64,
    device_bytes: u64,
}

fn expert_benchmark_snapshot(model: &LoadedModel) -> Result<ExpertBenchmarkSnapshot> {
    let report = model
        .expert_cache_report()?
        .context("sparse expert cache benchmark requires an expert-cache model")?;
    Ok(ExpertBenchmarkSnapshot {
        prefill: report.prefill,
        decode: report.decode,
        host_experts: report.host_resident_experts,
        device_experts: report.device_resident_experts,
        host_bytes: report.host_resident_bytes,
        device_bytes: report.device_resident_bytes,
    })
}

fn tier_delta(after: ExpertTierStatistics, before: ExpertTierStatistics) -> ExpertTierStatistics {
    ExpertTierStatistics {
        requests: after.requests.saturating_sub(before.requests),
        hits: after.hits.saturating_sub(before.hits),
        misses: after.misses.saturating_sub(before.misses),
        evictions: after.evictions.saturating_sub(before.evictions),
        eviction_bytes: after.eviction_bytes.saturating_sub(before.eviction_bytes),
    }
}

fn print_expert_benchmark_result(
    label: &str,
    elapsed: std::time::Duration,
    before: ExpertPassStatistics,
    after: ExpertPassStatistics,
    occupancy: ExpertBenchmarkSnapshot,
) {
    let host = tier_delta(after.host, before.host);
    let device = tier_delta(after.device, before.device);
    eprintln!(
        "expert_cache_benchmark_{label}: latency={:.3}s routes={} distinct={} coalesced={} compact_banks={} compact_bytes={} host_hits={} host_misses={} host_evictions={} device_hits={} device_misses={} device_evictions={} host_resident={}({} bytes) device_resident={}({} bytes)",
        elapsed.as_secs_f64(),
        after.requested_routes.saturating_sub(before.requested_routes),
        after.distinct_experts.saturating_sub(before.distinct_experts),
        after
            .coalesced_duplicates
            .saturating_sub(before.coalesced_duplicates),
        after.compact_banks.saturating_sub(before.compact_banks),
        after
            .compact_bank_bytes
            .saturating_sub(before.compact_bank_bytes),
        host.hits,
        host.misses,
        host.evictions,
        device.hits,
        device.misses,
        device.evictions,
        occupancy.host_experts,
        occupancy.host_bytes,
        occupancy.device_experts,
        occupancy.device_bytes,
    );
}

fn run_expert_cache_benchmark(
    model: &mut LoadedModel,
    tokens: &Array,
    stream: &Stream,
) -> Result<()> {
    let parts = [InputPart::text_token_ids(tokens)];
    let input = ModelInput::new(&parts);

    let before_cold = expert_benchmark_snapshot(model)?;
    let mut cold_cache = model.new_cache();
    let started = Instant::now();
    let logits = model.prefill_input_with_cache(input, &mut cold_cache, stream)?;
    eval([&logits])?;
    stream.synchronize()?;
    let cold_elapsed = started.elapsed();
    let after_cold = expert_benchmark_snapshot(model)?;
    print_expert_benchmark_result(
        "cold_prefill",
        cold_elapsed,
        before_cold.prefill,
        after_cold.prefill,
        after_cold,
    );

    let before_repeated = after_cold;
    let mut repeated_cache = model.new_cache();
    let started = Instant::now();
    let logits = model.prefill_input_with_cache(input, &mut repeated_cache, stream)?;
    eval([&logits])?;
    stream.synchronize()?;
    let repeated_elapsed = started.elapsed();
    let after_repeated = expert_benchmark_snapshot(model)?;
    print_expert_benchmark_result(
        "repeated_prefill",
        repeated_elapsed,
        before_repeated.prefill,
        after_repeated.prefill,
        after_repeated,
    );

    let last = tokens.try_index_device((.., tokens.dim(1) - 1..), stream)?;
    let decode_parts = [InputPart::text_token_ids(&last)];
    let decode_input = ModelInput::new(&decode_parts);
    let before_decode = after_repeated;
    let started = Instant::now();
    let logits = model.prefill_input_with_cache(decode_input, &mut repeated_cache, stream)?;
    eval([&logits])?;
    stream.synchronize()?;
    let decode_elapsed = started.elapsed();
    let after_decode = expert_benchmark_snapshot(model)?;
    print_expert_benchmark_result(
        "cached_decode",
        decode_elapsed,
        before_decode.decode,
        after_decode.decode,
        after_decode,
    );
    Ok(())
}

fn validate_args(args: &Cli) -> Result<()> {
    if args.max_tokens == 0 {
        bail!("--max-tokens must be greater than zero");
    }
    if args.draft_model.is_some() && args.mtp_draft_tokens == 0 {
        bail!("--mtp-draft-tokens must be greater than zero when --draft-model is used");
    }
    if !args.temperature.is_finite() || args.temperature < 0.0 {
        bail!("--temperature must be a finite, non-negative number");
    }
    if args.top_k < 0 {
        bail!("--top-k must be non-negative");
    }
    if !args.top_p.is_finite() || !(0.0..=1.0).contains(&args.top_p) {
        bail!("--top-p must be between zero and one");
    }
    if !args.min_p.is_finite() || !(0.0..=1.0).contains(&args.min_p) {
        bail!("--min-p must be between zero and one");
    }
    if !args.repeat_penalty.is_finite() || args.repeat_penalty <= 0.0 {
        bail!("--repeat-penalty must be a finite number greater than zero");
    }
    if args.repeat_last_n < -1 {
        bail!("--repeat-last-n must be -1 or greater");
    }
    if !args.frequency_penalty.is_finite() || !args.presence_penalty.is_finite() {
        bail!("frequency and presence penalties must be finite numbers");
    }
    if let Some(bits) = args.quantize {
        AffineQuantization::new(args.quantization_group_size, bits)?;
    }
    if args.layerwise_host && args.quantize.is_some() {
        bail!("--quantize is not supported with --layerwise-host; use matching checkpoint-native quantization");
    }
    if args.expert_cache && args.quantize.is_some() {
        bail!("--quantize is not supported with --expert-cache; use checkpoint-native weights");
    }
    if args.expert_cache && args.layerwise_host {
        bail!("--expert-cache conflicts with --layerwise-host");
    }
    if args.expert_cache_benchmark && !args.expert_cache {
        bail!("--expert-cache-benchmark requires --expert-cache");
    }
    if args.expert_cache_scratch_bytes == 0 {
        bail!("--expert-cache-scratch-bytes must be greater than zero");
    }
    if args.device_layer_window == 0 {
        bail!("--device-layer-window must be greater than zero");
    }
    if args.mapped_shards == 0 {
        bail!("--mapped-shards must be greater than zero");
    }
    if args.revision.is_some() && Path::new(&args.model).exists() {
        bail!("--revision can only be used with a Hugging Face model identifier");
    }
    Ok(())
}

fn read_prompt(argument: Option<&str>) -> Result<String> {
    if let Some(prompt) = argument {
        return Ok(prompt.to_owned());
    }
    if io::stdin().is_terminal() {
        bail!("provide PROMPT as an argument or pipe it on stdin");
    }

    let mut prompt = String::new();
    io::stdin()
        .read_to_string(&mut prompt)
        .context("failed to read the prompt from stdin")?;
    if prompt.is_empty() {
        bail!("stdin contained no prompt");
    }
    Ok(prompt)
}

fn resolve_model(spec: &str, requested_revision: Option<&str>) -> Result<PathBuf> {
    let path = Path::new(spec);
    if path.exists() {
        return path
            .canonicalize()
            .with_context(|| format!("failed to resolve model path {}", path.display()));
    }

    let (repo_id, quantization) = split_hf_model_spec(spec)?;

    let client = HFClientSync::new().context("failed to initialize the Hugging Face cache")?;
    let cache = client
        .scan_cache()
        .send()
        .context("failed to scan the Hugging Face cache")?;
    let repo = cache
        .repos
        .iter()
        .find(|repo| repo.repo_type == "model" && repo.repo_id == repo_id)
        .with_context(|| {
            format!(
                "{repo_id:?} is not an existing path or a model in the local Hugging Face cache at {}",
                cache.cache_dir.display()
            )
        })?;
    let revision = select_revision(&repo.revisions, requested_revision).with_context(|| {
        format!("could not select a cached revision for Hugging Face model {repo_id:?}")
    })?;
    match quantization {
        Some(quantization) => select_cached_gguf(revision, quantization).with_context(|| {
            format!(
                "could not select GGUF quantization {quantization:?} for Hugging Face model {repo_id:?}"
            )
        }),
        None => Ok(revision.snapshot_path.clone()),
    }
}

fn split_hf_model_spec(spec: &str) -> Result<(&str, Option<&str>)> {
    let Some((repo_id, quantization)) = spec.rsplit_once(':') else {
        return Ok((spec, None));
    };
    if repo_id.is_empty() {
        bail!("Hugging Face model identifier before ':' must not be empty");
    }
    if quantization.is_empty() {
        bail!("GGUF quantization selector after ':' must not be empty");
    }
    Ok((repo_id, Some(quantization)))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum QuantizationMatch {
    Exact,
    UnslothAlias,
}

fn select_cached_gguf(revision: &CachedRevisionInfo, quantization: &str) -> Result<PathBuf> {
    let files = revision
        .files
        .iter()
        .map(|file| file.file_path.as_path())
        .collect::<Vec<_>>();
    select_cached_gguf_path(&files, quantization)
}

fn select_cached_gguf_path(files: &[&Path], quantization: &str) -> Result<PathBuf> {
    let selector = quantization.to_ascii_uppercase();
    let unsloth_alias = (!selector.starts_with("UD-")).then(|| format!("UD-{selector}"));
    let mut gguf_files = Vec::new();
    let mut candidates = Vec::new();

    for &path in files {
        if !path
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
        {
            continue;
        }
        gguf_files.push(path);

        let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) else {
            continue;
        };
        let (stem, first_shard) = strip_gguf_shard_suffix(stem);
        if !first_shard {
            continue;
        }
        let stem = stem.to_ascii_uppercase();
        let matched = if quantization_suffix_matches(&stem, &selector) {
            if unsloth_alias
                .as_deref()
                .is_some_and(|alias| quantization_suffix_matches(&stem, alias))
            {
                QuantizationMatch::UnslothAlias
            } else {
                QuantizationMatch::Exact
            }
        } else {
            continue;
        };
        candidates.push((path, matched));
    }

    if candidates
        .iter()
        .any(|(_, matched)| *matched == QuantizationMatch::Exact)
    {
        candidates.retain(|(_, matched)| *matched == QuantizationMatch::Exact);
    }
    candidates.sort_unstable_by_key(|(path, _)| *path);

    match candidates.as_slice() {
        [(path, _)] => Ok((*path).to_owned()),
        [] => {
            let available = format_cached_paths(&gguf_files);
            if available.is_empty() {
                bail!("the selected cached revision contains no GGUF files");
            }
            bail!(
                "no cached GGUF filename matches quantization {quantization:?}; available GGUF files: {available}"
            )
        }
        _ => {
            let paths = candidates.iter().map(|(path, _)| *path).collect::<Vec<_>>();
            bail!(
                "quantization {quantization:?} matches multiple cached GGUF files: {}",
                format_cached_paths(&paths)
            )
        }
    }
}

fn quantization_suffix_matches(stem: &str, selector: &str) -> bool {
    let Some(prefix) = stem.strip_suffix(selector) else {
        return false;
    };
    prefix.is_empty()
        || prefix
            .chars()
            .next_back()
            .is_some_and(|separator| matches!(separator, '-' | '.' | '_'))
}

fn strip_gguf_shard_suffix(stem: &str) -> (&str, bool) {
    let Some((prefix, count)) = stem.rsplit_once("-of-") else {
        return (stem, true);
    };
    let Some((base, number)) = prefix.rsplit_once('-') else {
        return (stem, true);
    };
    let canonical_number = number.len() == 5 && number.bytes().all(|byte| byte.is_ascii_digit());
    let canonical_count = count.len() == 5 && count.bytes().all(|byte| byte.is_ascii_digit());
    if canonical_number && canonical_count {
        (base, number == "00001")
    } else {
        (stem, true)
    }
}

fn format_cached_paths(paths: &[&Path]) -> String {
    let mut paths = paths
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>();
    paths.sort_unstable();
    paths.join(", ")
}

fn select_revision<'a>(
    revisions: &'a [CachedRevisionInfo],
    requested: Option<&str>,
) -> Result<&'a CachedRevisionInfo> {
    if let Some(requested) = requested {
        return revisions
            .iter()
            .find(|revision| {
                revision.commit_hash == requested
                    || revision.refs.iter().any(|name| name == requested)
            })
            .with_context(|| format!("revision {requested:?} is not present in the cache"));
    }

    revisions
        .iter()
        .find(|revision| revision.refs.iter().any(|name| name == "main"))
        .or_else(|| {
            revisions
                .iter()
                .max_by_key(|revision| revision.last_modified)
        })
        .context("the cached repository contains no snapshots")
}

#[cfg(test)]
mod tests {
    use std::{
        path::Path,
        time::{Duration, SystemTime},
    };

    use clap::{CommandFactory, Parser};
    use hf_hub::cache::CachedRevisionInfo;

    use super::{
        format_bytes, select_cached_gguf_path, select_revision, should_report_stop_reason,
        split_hf_model_spec, stop_reason, validate_args, Cli, StopReason,
    };

    fn revision(hash: &str, refs: &[&str], modified: u64) -> CachedRevisionInfo {
        CachedRevisionInfo {
            commit_hash: hash.to_owned(),
            snapshot_path: hash.into(),
            files: Vec::new(),
            size_on_disk: 0,
            refs: refs.iter().map(|value| (*value).to_owned()).collect(),
            last_modified: SystemTime::UNIX_EPOCH + Duration::from_secs(modified),
        }
    }

    #[test]
    fn command_definition_is_valid() {
        Cli::command().debug_assert();
    }

    #[test]
    fn classifies_generation_stop_reason() {
        assert_eq!(stop_reason(&[4, 2], &[2], 10), StopReason::Eos);
        assert_eq!(stop_reason(&[4, 5], &[2], 2), StopReason::MaxTokens);
        assert_eq!(stop_reason(&[4], &[2], 2), StopReason::GeneratorExhausted);
    }

    #[test]
    fn reports_only_max_tokens_without_verbose_output() {
        assert!(should_report_stop_reason(StopReason::MaxTokens, false));
        assert!(!should_report_stop_reason(StopReason::Eos, false));
        assert!(!should_report_stop_reason(
            StopReason::GeneratorExhausted,
            false
        ));
        assert!(should_report_stop_reason(StopReason::Eos, true));
    }

    #[test]
    fn validates_load_time_quantization_arguments() {
        let valid = Cli::try_parse_from([
            "safemlx-lm",
            "--model",
            "model-id",
            "--quantize",
            "4",
            "prompt",
        ])
        .unwrap();
        validate_args(&valid).unwrap();

        let invalid = Cli::try_parse_from([
            "safemlx-lm",
            "--model",
            "model-id",
            "--quantize",
            "7",
            "prompt",
        ])
        .unwrap();
        assert!(validate_args(&invalid)
            .unwrap_err()
            .to_string()
            .contains("bits must be one of"));
    }

    #[test]
    fn selects_main_by_default() {
        let revisions = [revision("older", &["main"], 1), revision("newer", &[], 2)];
        assert_eq!(
            select_revision(&revisions, None).unwrap().commit_hash,
            "older"
        );
    }

    #[test]
    fn selects_requested_ref_or_hash() {
        let revisions = [
            revision("first", &["main"], 1),
            revision("second", &["experiment"], 2),
        ];
        assert_eq!(
            select_revision(&revisions, Some("experiment"))
                .unwrap()
                .commit_hash,
            "second"
        );
        assert_eq!(
            select_revision(&revisions, Some("first"))
                .unwrap()
                .commit_hash,
            "first"
        );
    }

    #[test]
    fn selects_newest_snapshot_without_main() {
        let revisions = [revision("older", &[], 1), revision("newer", &[], 2)];
        assert_eq!(
            select_revision(&revisions, None).unwrap().commit_hash,
            "newer"
        );
    }

    #[test]
    fn parses_hugging_face_quantization_selector() {
        assert_eq!(
            split_hf_model_spec("unsloth/model-GGUF:UD-Q4_K_M").unwrap(),
            ("unsloth/model-GGUF", Some("UD-Q4_K_M"))
        );
        assert_eq!(
            split_hf_model_spec("unsloth/model-GGUF").unwrap(),
            ("unsloth/model-GGUF", None)
        );
        assert!(split_hf_model_spec("unsloth/model-GGUF:").is_err());
    }

    #[test]
    fn selects_exact_and_unsloth_aliased_quantizations() {
        let q4 = Path::new("snapshot/model-Q4_K_M.gguf");
        let ud_q4 = Path::new("snapshot/model-UD-Q4_K_M.gguf");
        let files = [q4, ud_q4];

        assert_eq!(select_cached_gguf_path(&files, "UD-Q4_K_M").unwrap(), ud_q4);
        assert_eq!(select_cached_gguf_path(&files, "q4_k_m").unwrap(), q4);
        assert_eq!(select_cached_gguf_path(&[ud_q4], "Q4_K_M").unwrap(), ud_q4);
    }

    #[test]
    fn selects_first_shard_for_quantization() {
        let first = Path::new("snapshot/model-Q4_K_M-00001-of-00002.gguf");
        let second = Path::new("snapshot/model-Q4_K_M-00002-of-00002.gguf");
        assert_eq!(
            select_cached_gguf_path(&[second, first], "Q4_K_M").unwrap(),
            first
        );
    }

    #[test]
    fn rejects_ambiguous_or_missing_quantizations() {
        let first = Path::new("snapshot/first-Q4_K_M.gguf");
        let second = Path::new("snapshot/second-Q4_K_M.gguf");
        let error = select_cached_gguf_path(&[first, second], "Q4_K_M")
            .unwrap_err()
            .to_string();
        assert!(error.contains("matches multiple cached GGUF files"));

        let error = select_cached_gguf_path(&[first], "Q8_0")
            .unwrap_err()
            .to_string();
        assert!(error.contains("available GGUF files"));
    }

    #[test]
    fn formats_memory_with_exact_bytes() {
        assert_eq!(format_bytes(512), "512.00 B (512 bytes)");
        assert_eq!(format_bytes(1536), "1.50 KiB (1536 bytes)");
        assert_eq!(
            format_bytes(3 * 1024 * 1024 * 1024),
            "3.00 GiB (3221225472 bytes)"
        );
    }
}
