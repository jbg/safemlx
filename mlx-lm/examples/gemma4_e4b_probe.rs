use std::{collections::BTreeMap, path::PathBuf};

use mlx_lm::{cache::ConcatKeyValueCache, error::Error, models::LoadedModel};
use mlx_rs::ops::indexing::{IndexOp, NewAxis};
use serde_json::Value;

fn main() -> anyhow::Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let model_dir = args
        .first()
        .map(PathBuf::from)
        .or_else(default_e4b_snapshot)
        .expect("usage: cargo run -p mlx-lm --example gemma4_e4b_probe -- <model-dir> [prompt]");
    let prompt = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "what is goose?".to_string());

    print_config_summary(&model_dir)?;

    let mut model = match LoadedModel::load(&model_dir) {
        Ok(model) => model,
        Err(Error::StrictLoadValidation { missing, unused }) => {
            print_strict_report(&missing, &unused);
            anyhow::bail!(
                "strict load failed; implement the missing architecture or key mapping above"
            );
        }
        Err(error) => return Err(error.into()),
    };

    let rendered = model
        .apply_chat_template_json(
            vec![vec![serde_json::json!({"role":"user","content":prompt})]],
            None,
            true,
        )?
        .unwrap_or_else(|| {
            args.get(1)
                .cloned()
                .unwrap_or_else(|| "what is goose?".to_string())
        });
    println!("\n=== prompt ===\n{rendered}\n");

    let ids = model.encode(&rendered, false)?;
    let tokens = mlx_rs::Array::from(ids.as_slice()).index(NewAxis);
    let eos = model.eos_token_ids().to_vec();
    let mut cache: Vec<Option<ConcatKeyValueCache>> = Vec::new();
    let mut output_ids = Vec::new();

    {
        let mut generator = model.generate(&mut cache, 0.0, &tokens);
        for _ in 0..120 {
            let token = match generator.next() {
                Some(token) => token?,
                None => break,
            };
            let id = token.item::<u32>();
            output_ids.push(id);
            if eos.contains(&id) {
                break;
            }
        }
    }

    println!("=== output ids ===\n{output_ids:?}\n");
    println!("=== output ===\n{}", model.decode(&output_ids, false)?);
    Ok(())
}

fn default_e4b_snapshot() -> Option<PathBuf> {
    let home = std::env::var_os("HOME").map(PathBuf::from)?;
    let snapshots = home
        .join(".cache/huggingface/hub")
        .join("models--mlx-community--gemma-4-e4b-it-4bit")
        .join("snapshots");
    snapshots
        .read_dir()
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .find(|path| path.join("config.json").exists())
}

fn print_config_summary(model_dir: &PathBuf) -> anyhow::Result<()> {
    let config_path = model_dir.join("config.json");
    let config: Value = serde_json::from_str(&std::fs::read_to_string(config_path)?)?;
    let text = config.get("text_config").unwrap_or(&config);
    println!("=== Gemma 4 E4B probe ===");
    println!("model_dir: {}", model_dir.display());
    for key in [
        "model_type",
        "hidden_size",
        "num_hidden_layers",
        "hidden_size_per_layer_input",
        "num_kv_shared_layers",
        "attention_k_eq_v",
        "enable_moe_block",
    ] {
        println!("{key}: {}", text.get(key).unwrap_or(&Value::Null));
    }
    if let Some(quantization) = config.get("quantization") {
        println!("quantization: {quantization}");
    }
    Ok(())
}

fn print_strict_report(missing: &[String], unused: &[String]) {
    println!("\n=== strict load failed ===");
    println!("missing parameters: {}", missing.len());
    print_groups("missing", missing);
    print_examples("missing examples", missing);
    println!("\nunused weights: {}", unused.len());
    print_groups("unused", unused);
    print_examples("unused examples", unused);
}

fn print_groups(label: &str, keys: &[String]) {
    let mut groups = BTreeMap::<String, usize>::new();
    for key in keys {
        *groups.entry(group_key(key)).or_default() += 1;
    }
    println!("\n{label} groups:");
    for (group, count) in groups.iter().take(80) {
        println!("  {count:4}  {group}");
    }
    if groups.len() > 80 {
        println!("  ... and {} more groups", groups.len() - 80);
    }
}

fn print_examples(label: &str, keys: &[String]) {
    println!("\n{label}:");
    for key in keys.iter().take(80) {
        println!("  {key}");
    }
    if keys.len() > 80 {
        println!("  ... and {} more", keys.len() - 80);
    }
}

fn group_key(key: &str) -> String {
    let key = key
        .strip_prefix("language_model.model.")
        .or_else(|| key.strip_prefix("model.language_model."))
        .unwrap_or(key);
    let mut parts = key.split('.').collect::<Vec<_>>();
    for part in &mut parts {
        if part.chars().all(|ch| ch.is_ascii_digit()) {
            *part = "#";
        }
    }
    parts.into_iter().take(4).collect::<Vec<_>>().join(".")
}
