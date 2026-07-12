use std::{collections::HashMap, path::PathBuf};

use safemlx::{
    module::ModuleParameters, ops::indexing::TryIndexOp, Array, Device, DeviceType,
    ExecutionContext,
};
use safemlx_lm::models::moshi;

fn main() -> anyhow::Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let model_dir = args
        .first()
        .map(PathBuf::from)
        .expect("usage: moshi_token_parity <model-dir> <fixture.safetensors> [rtol] [atol]");
    let fixture_path = args
        .get(1)
        .map(PathBuf::from)
        .expect("missing moshi_mlx fixture path");
    let rtol = args
        .get(2)
        .map(|value| value.parse::<f64>())
        .transpose()?
        .unwrap_or(2e-2);
    let atol = args
        .get(3)
        .map(|value| value.parse::<f64>())
        .transpose()?
        .unwrap_or(2e-2);

    let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
    let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = gpu.stream();
    let mut model = moshi::load_model(&model_dir, stream, cpu.stream())?;
    let fixture = Array::load_safetensors(&fixture_path, cpu.stream())?;
    let checkpoint = Array::load_safetensors(model_dir.join("model.safetensors"), cpu.stream())?;
    verify_loaded_weights(&model, &checkpoint, stream)?;
    let text = required(&fixture, "input.text")?;
    let audio = required(&fixture, "input.audio")?;
    let depth = required(&fixture, "input.depth")?;
    validate_inputs(text, audio, depth, &model)?;

    let mut cache = model.new_cache();
    let mut comparisons = 0usize;
    let mut worst = (0.0_f32, String::new());
    for step in 0..text.dim(0) {
        let text_step = text.try_index_device((step, .., ..), stream)?;
        let audio_step = audio.try_index_device((step, .., ..), stream)?;
        let depth_step = depth.try_index_device((step, .., ..), stream)?;
        let output = model.token_step(&text_step, &audio_step, &depth_step, &mut cache, stream)?;

        track(
            &mut worst,
            &output.temporal_input,
            required(&fixture, &format!("expected.{step}.temporal_input"))?,
            rtol,
            atol,
            stream,
            &format!("step {step} temporal input"),
        )?;
        comparisons += 1;
        for (layer, trace) in output.temporal_layer_traces.iter().enumerate() {
            for (name, actual) in [
                ("norm1", &trace.norm1),
                ("attention_in_proj", &trace.attention_in_proj),
                ("attention_queries", &trace.attention_queries),
                ("attention_keys", &trace.attention_keys),
                ("attention_values", &trace.attention_values),
                ("attention_attended", &trace.attention_attended),
                ("attention", &trace.attention),
                ("post_attention", &trace.post_attention),
                ("norm2", &trace.norm2),
                ("mlp", &trace.mlp),
            ] {
                track(
                    &mut worst,
                    actual,
                    required(
                        &fixture,
                        &format!("expected.{step}.temporal_layer.{layer}.{name}"),
                    )?,
                    rtol,
                    atol,
                    stream,
                    &format!("step {step} temporal layer {layer} {name}"),
                )?;
                comparisons += 1;
            }
            track(
                &mut worst,
                &trace.output,
                required(&fixture, &format!("expected.{step}.temporal_layer.{layer}"))?,
                rtol,
                atol,
                stream,
                &format!("step {step} temporal layer {layer} output"),
            )?;
            comparisons += 1;
        }
        track(
            &mut worst,
            &output.temporal_output,
            required(&fixture, &format!("expected.{step}.temporal"))?,
            rtol,
            atol,
            stream,
            &format!("step {step} temporal"),
        )?;
        comparisons += 1;
        track(
            &mut worst,
            &output.text_logits,
            required(&fixture, &format!("expected.{step}.text_logits"))?,
            rtol,
            atol,
            stream,
            &format!("step {step} text logits"),
        )?;
        comparisons += 1;
        for (slice, actual) in output.audio_logits.iter().enumerate() {
            track(
                &mut worst,
                actual,
                required(&fixture, &format!("expected.{step}.audio_logits.{slice}"))?,
                rtol,
                atol,
                stream,
                &format!("step {step} depth slice {slice}"),
            )?;
            comparisons += 1;
        }
    }

    println!(
        "Moshi token parity passed: {} frames, {} tensors, rtol={}, atol={}, worst max_abs={} ({})",
        text.dim(0),
        comparisons,
        rtol,
        atol,
        worst.0,
        worst.1
    );
    Ok(())
}

fn verify_loaded_weights(
    model: &moshi::Model,
    weights: &HashMap<String, Array>,
    stream: &safemlx::Stream,
) -> anyhow::Result<()> {
    let params = model.parameters().flatten();
    anyhow::ensure!(
        params.len() == weights.len(),
        "parameter count differs: Rust {}, checkpoint {}",
        params.len(),
        weights.len()
    );
    for (key, expected) in weights {
        let actual = params
            .get(key.as_str())
            .ok_or_else(|| anyhow::anyhow!("Rust model is missing checkpoint parameter {key}"))?;
        compare(actual, expected, 0.0, 0.0, stream, &format!("weight {key}"))?;
    }
    println!("verified {} loaded checkpoint tensors", weights.len());
    Ok(())
}

fn required<'a>(fixture: &'a HashMap<String, Array>, key: &str) -> anyhow::Result<&'a Array> {
    fixture
        .get(key)
        .ok_or_else(|| anyhow::anyhow!("fixture is missing tensor {key}"))
}

fn validate_inputs(
    text: &Array,
    audio: &Array,
    depth: &Array,
    model: &moshi::Model,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        text.shape().len() == 3 && text.dim(2) == 1,
        "input.text must be [steps, batch, 1], got {:?}",
        text.shape()
    );
    anyhow::ensure!(
        audio.shape().len() == 3
            && audio.dim(0) == text.dim(0)
            && audio.dim(1) == text.dim(1)
            && audio.dim(2) == model.args.n_q,
        "input.audio must be [steps, batch, n_q], got {:?}",
        audio.shape()
    );
    anyhow::ensure!(
        depth.shape().len() == 3
            && depth.dim(0) == text.dim(0)
            && depth.dim(1) == text.dim(1)
            && depth.dim(2) == model.args.dep_q,
        "input.depth must be [steps, batch, dep_q], got {:?}",
        depth.shape()
    );
    Ok(())
}

fn compare(
    actual: &Array,
    expected: &Array,
    rtol: f64,
    atol: f64,
    stream: &safemlx::Stream,
    label: &str,
) -> anyhow::Result<f32> {
    anyhow::ensure!(
        actual.shape() == expected.shape(),
        "{label}: shape mismatch: Rust {:?}, moshi_mlx {:?}",
        actual.shape(),
        expected.shape()
    );
    let expected = expected.copy(stream)?;
    let close = actual.all_close(&expected, rtol, atol, false, stream)?;
    let max_abs = actual
        .subtract(&expected, stream)?
        .abs(stream)?
        .max(None, stream)?
        .item::<f32>(stream);
    let actual_first = actual
        .reshape(&[-1], stream)?
        .try_index_device(0, stream)?
        .item::<f32>(stream);
    let expected_first = expected
        .reshape(&[-1], stream)?
        .try_index_device(0, stream)?
        .item::<f32>(stream);
    anyhow::ensure!(
        close.item::<bool>(stream),
        "{label}: max_abs={max_abs}, first Rust={actual_first}, first reference={expected_first}; exceeds rtol={rtol}, atol={atol}"
    );
    Ok(max_abs)
}

#[allow(clippy::too_many_arguments)]
fn track(
    worst: &mut (f32, String),
    actual: &Array,
    expected: &Array,
    rtol: f64,
    atol: f64,
    stream: &safemlx::Stream,
    label: &str,
) -> anyhow::Result<()> {
    let max_abs = compare(actual, expected, rtol, atol, stream, label)?;
    if worst.1.is_empty() || max_abs > worst.0 {
        *worst = (max_abs, label.to_owned());
    }
    Ok(())
}
