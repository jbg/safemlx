use std::{path::PathBuf, time::Instant};

use safemlx::{transforms::eval, ExecutionContext, Stream};
use safemlx_lm::{
    models::{
        input::{InputPart, ModelInput},
        LoadedModel,
    },
    mtp::{LoadedDrafter, MtpConfig},
};

fn main() -> anyhow::Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let target_dir = args
        .first()
        .map(PathBuf::from)
        .or_else(default_target_snapshot)
        .expect("target model dir required");
    let assistant_dir = args
        .get(1)
        .map(PathBuf::from)
        .or_else(default_assistant_snapshot)
        .expect("assistant model dir required");
    let prompt = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "Why is the sky blue?".to_string());
    let max_tokens = args
        .get(3)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(96);

    println!("target: {}", target_dir.display());
    println!("assistant: {}", assistant_dir.display());
    println!("prompt: {prompt:?}");

    let ctx = ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    let stream = ctx.stream();
    let weights_ctx = ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let weights_stream = weights_ctx.stream();
    let rendered = render_prompt(&target_dir, &prompt, stream, weights_stream)?;
    println!("\n=== rendered prompt ===\n{rendered}\n");

    let greedy = run_greedy(&target_dir, &rendered, max_tokens, stream, weights_stream)?;
    println!("\n=== greedy ===");
    println!(
        "tokens: {} elapsed: {:.2?}",
        greedy.token_ids.len(),
        greedy.elapsed
    );
    println!("{}", greedy.text);

    let mtp = run_mtp(
        &target_dir,
        &assistant_dir,
        &rendered,
        max_tokens,
        stream,
        weights_stream,
    )?;
    println!("\n=== mtp ===");
    println!(
        "tokens: {} elapsed: {:.2?}",
        mtp.token_ids.len(),
        mtp.elapsed
    );
    println!("accepted per round: {:?}", mtp.accept_lens);
    println!("{}", mtp.text);

    Ok(())
}

struct ProbeResult {
    token_ids: Vec<u32>,
    text: String,
    elapsed: std::time::Duration,
    accept_lens: Vec<usize>,
}

fn render_prompt(
    target_dir: &PathBuf,
    prompt: &str,
    stream: &Stream,
    weights_stream: &Stream,
) -> anyhow::Result<String> {
    let mut loaded = LoadedModel::load(target_dir, stream, weights_stream)?;
    Ok(loaded
        .apply_chat_template_json(
            vec![vec![serde_json::json!({
                "role": "user",
                "content": [{"type": "text", "text": prompt, "content": prompt}],
            })]],
            None,
            true,
        )?
        .unwrap_or_else(|| prompt.to_string()))
}

fn run_greedy(
    target_dir: &PathBuf,
    prompt: &str,
    max_tokens: usize,
    stream: &Stream,
    weights_stream: &Stream,
) -> anyhow::Result<ProbeResult> {
    let mut loaded = LoadedModel::load(target_dir, stream, weights_stream)?;
    let prompt_tokens = loaded.encode_to_array(prompt, false, stream)?;
    let eos = loaded.eos_token_ids().to_vec();
    let mut cache = loaded.new_cache();
    let mut ids = Vec::new();
    let start = Instant::now();
    {
        let input_parts = [InputPart::text_token_ids(&prompt_tokens)];
        let input = ModelInput::new(&input_parts);
        let generator = loaded
            .generate_input_with_cache(&mut cache, 0.0, input, None, stream)
            .take(max_tokens);
        for token in generator {
            let token = token?;
            eval([&token])?;
            let id = token.item::<u32>(stream);
            if eos.contains(&id) {
                break;
            }
            ids.push(id);
        }
    }
    let elapsed = start.elapsed();
    let text = loaded.decode(&ids, true)?;
    Ok(ProbeResult {
        token_ids: ids,
        text,
        elapsed,
        accept_lens: Vec::new(),
    })
}

fn run_mtp(
    target_dir: &PathBuf,
    assistant_dir: &PathBuf,
    prompt: &str,
    max_tokens: usize,
    stream: &Stream,
    weights_stream: &Stream,
) -> anyhow::Result<ProbeResult> {
    let mut target = LoadedModel::load(target_dir, stream, weights_stream)?;
    let mut assistant = LoadedDrafter::load(assistant_dir, stream, weights_stream)?;
    let prompt_tokens = target.encode_to_array(prompt, false, stream)?;
    let mut cache = target.new_cache();
    let parts = [InputPart::text_token_ids(&prompt_tokens)];
    let input = ModelInput::new(&parts);
    let config = MtpConfig {
        max_tokens,
        max_draft_tokens: 3,
        temperature: 0.0,
        eos_token_ids: target.eos_token_ids().to_vec(),
    };
    let (mut generated, stats) =
        target.generate_mtp_input(&mut assistant, &mut cache, input, &config, None, stream)?;
    if generated
        .last()
        .is_some_and(|token| config.eos_token_ids.contains(token))
    {
        generated.pop();
    }
    let text = target.decode(&generated, true)?;
    Ok(ProbeResult {
        token_ids: generated,
        text,
        elapsed: stats.elapsed,
        accept_lens: stats.accept_lens,
    })
}

fn default_target_snapshot() -> Option<PathBuf> {
    default_snapshot("models--mlx-community--gemma-4-e4b-it-4bit")
}

fn default_assistant_snapshot() -> Option<PathBuf> {
    default_snapshot("models--mlx-community--gemma-4-e4b-it-assistant-bf16")
}

fn default_snapshot(repo_dir: &str) -> Option<PathBuf> {
    let snapshots = PathBuf::from(std::env::var_os("HOME")?)
        .join(".cache/huggingface/hub")
        .join(repo_dir)
        .join("snapshots");
    snapshots
        .read_dir()
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .find(|path| path.join("config.json").exists())
}
