use std::{path::PathBuf, time::Instant};

use safemlx::{
    ops::indexing::{NewAxis, TryIndexOp},
    transforms::eval,
    Array, ExecutionContext, Stream,
};
use safemlx_lm::models::{
    input::{InputPart, ModelInput},
    qwen3_5_moe, LoadedModel, ModelLoadOptions,
};
use safemlx_lm::quantization::AffineQuantization;

const DEFAULT_DECODE_TOKENS: usize = 128;
const CASES: &[(&str, usize)] = &[
    ("short", 16),
    ("prefill_128", 128),
    ("prefill_512", 512),
    ("prefill_2048", 2048),
];

fn main() -> anyhow::Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let quantize_on_load = args.iter().any(|value| value == "--quantize-on-load");
    let args = args
        .into_iter()
        .filter(|value| value != "--quantize-on-load")
        .collect::<Vec<_>>();
    let model_dir = args
        .first()
        .map(PathBuf::from)
        .or_else(default_qwen35_moe_snapshot)
        .expect(
            "usage: qwen35_moe_bench [model-dir] [decode-tokens] [case-name] [--quantize-on-load]",
        );
    let decode_tokens = args
        .get(1)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_DECODE_TOKENS);
    let case_filter = args.get(2).map(String::as_str);
    let profile_components = args
        .get(3)
        .is_some_and(|value| matches!(value.as_str(), "profile" | "true" | "1"))
        || std::env::var_os("QWEN35_MOE_BENCH_PROFILE").is_some();

    println!("model_dir={}", model_dir.display());
    println!("decode_tokens={decode_tokens}");
    println!("profile_components={profile_components}");
    println!("quantize_on_load={quantize_on_load}");

    let ctx = ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    let stream = ctx.stream();
    let weights_ctx = ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let weights_stream = weights_ctx.stream();
    safemlx::memory::reset_peak_memory()?;
    let load_start = Instant::now();
    let options = if quantize_on_load {
        ModelLoadOptions::with_quantization(AffineQuantization::default())
    } else {
        ModelLoadOptions::default()
    };
    let mut model = LoadedModel::load_with_options(&model_dir, options, stream, weights_stream)?;
    stream.synchronize()?;
    let load_elapsed = load_start.elapsed();
    println!("load_s={:.3}", load_elapsed.as_secs_f64());
    println!(
        "mlx_active_memory_bytes={}",
        safemlx::memory::active_memory()?
    );
    println!("mlx_peak_memory_bytes={}", safemlx::memory::peak_memory()?);
    println!(
        "mlx_cache_memory_bytes={}",
        safemlx::memory::cache_memory()?
    );

    let warmup_start = Instant::now();
    let warmup_prompt = prompt_near_token_count(&mut model, 16)?;
    let _ = run_case(&mut model, &warmup_prompt, 2, false, stream)?;
    println!("warmup_s={:.3}", warmup_start.elapsed().as_secs_f64());

    println!(
        "case,prompt_tokens,prefill_s,generated_tokens,decode_s,decode_tok_s,first_id,last_id"
    );
    if profile_components {
        println!(
            "profile,case,phase,total_s,component_total_s,other_s,embed_s,full_attention_s,linear_attention_s,moe_router_s,moe_shared_s,moe_routed_s,moe_combine_s,final_norm_s,lm_head_s,prefill_state_dependency_s"
        );
    }

    for (name, target_tokens) in CASES {
        if case_filter.is_some_and(|filter| filter != *name) {
            continue;
        }
        let prompt = prompt_near_token_count(&mut model, *target_tokens)?;
        let result = run_case(
            &mut model,
            &prompt,
            decode_tokens,
            profile_components,
            stream,
        )?;
        println!(
            "{},{},{:.6},{},{:.6},{:.3},{},{}",
            name,
            result.prompt_tokens,
            result.prefill_s,
            result.generated_tokens,
            result.decode_s,
            result.decode_tok_s,
            result.first_id,
            result.last_id
        );
        if let Some(stats) = &result.prefill_profile {
            print_profile(name, "prefill", result.prefill_s, stats);
        }
        if let Some(stats) = &result.decode_profile {
            print_profile(name, "decode", result.decode_s, stats);
        }
    }
    qwen3_5_moe::set_perf_profiling(false);

    Ok(())
}

struct BenchResult {
    prompt_tokens: usize,
    prefill_s: f64,
    generated_tokens: usize,
    decode_s: f64,
    decode_tok_s: f64,
    first_id: u32,
    last_id: u32,
    prefill_profile: Option<qwen3_5_moe::PerfStats>,
    decode_profile: Option<qwen3_5_moe::PerfStats>,
}

fn run_case(
    model: &mut LoadedModel,
    prompt: &str,
    decode_tokens: usize,
    profile_components: bool,
    stream: &Stream,
) -> anyhow::Result<BenchResult> {
    let prompt_ids = model.encode(prompt, false)?;
    let prompt_tokens = Array::from(prompt_ids.as_slice()).try_index_device(NewAxis, stream)?;
    let input_parts = [InputPart::text_token_ids(&prompt_tokens)];
    let input = ModelInput::new(&input_parts);
    let mut cache = model.new_cache();
    let mut generator = model.generate_input_with_cache(&mut cache, 0.0, input, None, stream);
    let mut ids = Vec::with_capacity(decode_tokens);

    qwen3_5_moe::set_perf_profiling(profile_components);
    qwen3_5_moe::reset_perf_stats();
    let prefill_start = Instant::now();
    let Some(first) = generator.next() else {
        anyhow::bail!("generator produced no tokens");
    };
    let first = first?;
    eval([&first])?;
    let prefill_s = prefill_start.elapsed().as_secs_f64();
    let prefill_profile = profile_components.then(qwen3_5_moe::perf_stats).flatten();
    ids.push(first.item::<u32>(stream));

    qwen3_5_moe::reset_perf_stats();
    let decode_start = Instant::now();
    for _ in 1..decode_tokens {
        let Some(token) = generator.next() else {
            break;
        };
        let token = token?;
        eval([&token])?;
        ids.push(token.item::<u32>(stream));
    }
    let decode_s = decode_start.elapsed().as_secs_f64();
    let decode_profile = profile_components.then(qwen3_5_moe::perf_stats).flatten();
    qwen3_5_moe::set_perf_profiling(false);

    let decode_count = ids.len().saturating_sub(1);
    let decode_tok_s = if decode_count == 0 {
        0.0
    } else {
        decode_count as f64 / decode_s
    };

    Ok(BenchResult {
        prompt_tokens: prompt_ids.len(),
        prefill_s,
        generated_tokens: ids.len(),
        decode_s,
        decode_tok_s,
        first_id: ids[0],
        last_id: *ids.last().expect("first token was pushed"),
        prefill_profile,
        decode_profile,
    })
}

fn print_profile(case: &str, phase: &str, total_s: f64, stats: &qwen3_5_moe::PerfStats) {
    let component_total_s = stats.component_total_s();
    println!(
        "profile,{case},{phase},{total_s:.6},{component_total_s:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
        total_s - component_total_s,
        stats.embed_s,
        stats.full_attention_s,
        stats.linear_attention_s,
        stats.moe_router_s,
        stats.moe_shared_s,
        stats.moe_routed_s,
        stats.moe_combine_s,
        stats.final_norm_s,
        stats.lm_head_s,
        stats.prefill_state_dependency_s,
    );
}

fn prompt_near_token_count(
    model: &mut LoadedModel,
    target_tokens: usize,
) -> anyhow::Result<String> {
    let base = "Discuss hybrid linear attention, sparse mixture-of-experts routing, recurrent cache updates, grouped convolution, and vocabulary projection in a text generation runtime. ";
    let mut prompt = "Summarize linear attention performance.".to_string();
    while model.encode(&prompt, false)?.len() < target_tokens {
        prompt.push_str(base);
    }
    Ok(prompt)
}

fn default_qwen35_moe_snapshot() -> Option<PathBuf> {
    let snapshots = PathBuf::from(std::env::var_os("HOME")?)
        .join(".cache/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots");
    snapshots
        .read_dir()
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .find(|path| path.join("config.json").exists())
}
