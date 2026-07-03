use std::{path::PathBuf, time::Instant};

use safemlx::{ops::indexing::IndexOp, transforms::eval, Array};
use safemlx_lm::{
    gemma4_mtp::generate_gemma4_mtp,
    models::{
        gemma4::load_gemma4_model, gemma4_assistant::load_gemma4_assistant_model, LoadedModel,
    },
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

    let rendered = render_prompt(&target_dir, &prompt)?;
    println!("\n=== rendered prompt ===\n{rendered}\n");

    let greedy = run_greedy(&target_dir, &rendered, max_tokens)?;
    println!("\n=== greedy ===");
    println!(
        "tokens: {} elapsed: {:.2?}",
        greedy.token_ids.len(),
        greedy.elapsed
    );
    println!("{}", greedy.text);

    let mtp = run_mtp(&target_dir, &assistant_dir, &rendered, max_tokens)?;
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

fn render_prompt(target_dir: &PathBuf, prompt: &str) -> anyhow::Result<String> {
    let mut loaded = LoadedModel::load(target_dir)?;
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
) -> anyhow::Result<ProbeResult> {
    let mut loaded = LoadedModel::load(target_dir)?;
    let prompt_tokens = loaded.encode_to_array(prompt, false)?;
    let eos = loaded.eos_token_ids().to_vec();
    let mut cache = loaded.new_cache();
    let mut ids = Vec::new();
    let start = Instant::now();
    {
        let generator = loaded
            .generate_with_cache(&mut cache, 0.0, &prompt_tokens)
            .take(max_tokens);
        for token in generator {
            let token = token?;
            eval([&token])?;
            let id = token.item::<u32>();
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
) -> anyhow::Result<ProbeResult> {
    let tokenizer_holder = LoadedModel::load(target_dir)?;
    let prompt_ids = tokenizer_holder.encode(prompt, false)?;
    let prompt_tokens = Array::from(prompt_ids.as_slice()).index(safemlx::ops::indexing::NewAxis);

    let mut target = load_gemma4_model(target_dir)?;
    let mut assistant = load_gemma4_assistant_model(assistant_dir)?;
    let (generated, stats) = generate_gemma4_mtp(
        &mut target,
        &mut assistant,
        &prompt_tokens,
        tokenizer_holder.eos_token_ids(),
        max_tokens,
        0.0,
    )?;

    let text = tokenizer_holder.decode(&generated, true)?;
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
