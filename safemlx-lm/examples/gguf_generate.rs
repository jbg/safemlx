use std::path::PathBuf;

use safemlx::{Device, DeviceType, ExecutionContext};
use safemlx_lm::models::{
    input::{InputPart, ModelInput},
    LoadedModel,
};

fn main() -> anyhow::Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let gguf_file = args.first().map(PathBuf::from).ok_or_else(|| {
        anyhow::anyhow!(
            "usage: cargo run -p safemlx-lm --example gguf_generate -- <model.gguf> [prompt] [max-tokens] [temperature]"
        )
    })?;
    let prompt = args
        .get(1)
        .map(String::as_str)
        .unwrap_or("Briefly explain what MLX is.");
    let max_tokens = args
        .get(2)
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(16);
    let temperature = args
        .get(3)
        .map(|value| value.parse::<f32>())
        .transpose()?
        .unwrap_or(0.0);

    let ctx = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
    let weights_ctx = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = ctx.stream();
    let mut model = LoadedModel::load(&gguf_file, stream, weights_ctx.stream())?;

    println!("model type: {}", model.model_type());
    println!("chat template: {}", model.has_chat_template());

    let rendered = model
        .apply_chat_template_json(
            vec![vec![serde_json::json!({
                "role": "user",
                "content": prompt,
            })]],
            None,
            true,
        )?
        .unwrap_or_else(|| prompt.to_owned());
    let tokens = model.encode_to_array(&rendered, false, stream)?;
    let eos_token_ids = model.eos_token_ids().to_vec();
    let mut cache = model.new_cache();
    let mut output_ids = Vec::new();
    let prng_key = if temperature == 0.0 {
        None
    } else {
        Some(safemlx::random::key(0)?)
    };

    {
        let input_parts = [InputPart::text_token_ids(&tokens)];
        let input = ModelInput::new(&input_parts);
        let mut generator =
            model.generate_input_with_cache(&mut cache, temperature, input, prng_key, stream);
        for _ in 0..max_tokens {
            let Some(token) = generator.next() else {
                break;
            };
            let token_id = token?.item::<u32>(stream);
            output_ids.push(token_id);
            if eos_token_ids.contains(&token_id) {
                break;
            }
        }
    }

    println!("prompt tokens: {}", tokens.shape()[1]);
    println!("output ids: {output_ids:?}");
    println!("output: {}", model.decode(&output_ids, false)?);
    Ok(())
}
