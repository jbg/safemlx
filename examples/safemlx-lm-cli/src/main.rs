use std::{
    io::{self, IsTerminal, Read, Write},
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{bail, Context, Result};
use clap::Parser;
use hf_hub::{cache::CachedRevisionInfo, HFClientSync};
use safemlx::{
    error::Exception, random::RandomState, transforms::async_eval, Array, Device, DeviceType,
    ExecutionContext, Stream,
};
use safemlx_lm::{
    models::{
        input::{InputPart, ModelInput},
        LoadedModel, ModelLoadOptions,
    },
    quantization::AffineQuantization,
    sampler::{DefaultSampler, GenerationSampler, Sampler},
};

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

/// Generate text with a model supported by safemlx-lm.
#[derive(Debug, Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    /// Model directory, GGUF file, or Hugging Face model identifier.
    #[arg(short, long, value_name = "PATH_OR_ID")]
    model: String,

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

    /// Pass the prompt directly instead of applying the model's chat template.
    #[arg(long)]
    raw: bool,

    /// Print model resolution and generation statistics to stderr.
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> Result<()> {
    let total_started = Instant::now();
    let args = Cli::parse();
    validate_args(&args)?;
    let prompt = read_prompt(args.prompt.as_deref())?;
    let model_path = resolve_model(&args.model, args.revision.as_deref())?;

    if args.verbose {
        eprintln!("model: {}", model_path.display());
    }

    let execution = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
    let weights = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    if args.verbose {
        // Capture the complete model-load and generation high-water mark.
        safemlx::memory::reset_peak_memory()?;
    }
    let load_started = Instant::now();
    let load_options = match args.quantize {
        Some(bits) => ModelLoadOptions::with_quantization(AffineQuantization::new(
            args.quantization_group_size,
            bits,
        )?),
        None => ModelLoadOptions::default(),
    };
    let mut model =
        LoadedModel::load_with_options(&model_path, load_options, stream, weights.stream())
            .with_context(|| format!("failed to load model from {}", model_path.display()))?;
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

    {
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
            current = next.transpose()?;
        }
    }
    let generation_elapsed = generation_started.elapsed();

    let output = model.decode(&output_ids, true)?;
    let stdout = io::stdout();
    let mut stdout = stdout.lock();
    write!(stdout, "{output}")?;
    if !output.ends_with('\n') {
        writeln!(stdout)?;
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
        if eos_token_ids.is_empty() {
            eprintln!("warning: the model config contains no EOS token id");
        } else if !output_ids
            .last()
            .is_some_and(|token| eos_token_ids.contains(token))
        {
            eprintln!("warning: generation reached --max-tokens before EOS");
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

fn validate_args(args: &Cli) -> Result<()> {
    if args.max_tokens == 0 {
        bail!("--max-tokens must be greater than zero");
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

    let client = HFClientSync::new().context("failed to initialize the Hugging Face cache")?;
    let cache = client
        .scan_cache()
        .send()
        .context("failed to scan the Hugging Face cache")?;
    let repo = cache
        .repos
        .iter()
        .find(|repo| repo.repo_type == "model" && repo.repo_id == spec)
        .with_context(|| {
            format!(
                "{spec:?} is not an existing path or a model in the local Hugging Face cache at {}",
                cache.cache_dir.display()
            )
        })?;
    let revision = select_revision(&repo.revisions, requested_revision).with_context(|| {
        format!("could not select a cached revision for Hugging Face model {spec:?}")
    })?;
    Ok(revision.snapshot_path.clone())
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
    use std::time::{Duration, SystemTime};

    use clap::{CommandFactory, Parser};
    use hf_hub::cache::CachedRevisionInfo;

    use super::{format_bytes, select_revision, validate_args, Cli};

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
    fn formats_memory_with_exact_bytes() {
        assert_eq!(format_bytes(512), "512.00 B (512 bytes)");
        assert_eq!(format_bytes(1536), "1.50 KiB (1536 bytes)");
        assert_eq!(
            format_bytes(3 * 1024 * 1024 * 1024),
            "3.00 GiB (3221225472 bytes)"
        );
    }
}
