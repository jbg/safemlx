use std::{collections::HashMap, path::PathBuf};

use safemlx::{
    ops::{indexing::TryIndexOp, stack_axis},
    Array, Device, DeviceType, ExecutionContext, Stream,
};
use safemlx_lm::{
    models::{moshi, personaplex},
    realtime::{RealtimeSampling, RealtimeSpeechModel, RealtimeStepInput},
    sampler::DefaultSampler,
};

fn main() -> anyhow::Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let model_dir = args
        .first()
        .map(PathBuf::from)
        .expect("usage: personaplex_torch_parity <tiny-model-dir> <fixture.safetensors>");
    let fixture_path = args
        .get(1)
        .map(PathBuf::from)
        .expect("missing PersonaPlex PyTorch fixture path");

    let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
    let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = gpu.stream();
    let config = std::fs::File::open(model_dir.join("config.json"))?;
    let args: moshi::ModelArgs = serde_json::from_reader(config)?;
    let mut model = moshi::load_pytorch_safetensors_model(
        args,
        model_dir.join(personaplex::MODEL_SAFETENSORS),
        stream,
        cpu.stream(),
    )?;
    let fixture = Array::load_safetensors(&fixture_path, cpu.stream())?;
    let input_audio = required(&fixture, "input.audio")?;
    let expected_sampled = required(&fixture, "expected.sampled")?;
    let expected_output_audio = required(&fixture, "expected.output_audio")?;

    let mut state = model.new_realtime_state();
    let config = model.realtime_config();
    let depth_audio_codebooks = config.depth_audio_codebooks;
    let generated_audio_codebooks = config.generated_audio_codebooks;
    let mut text_sampler = DefaultSampler;
    let mut audio_samplers = (0..depth_audio_codebooks)
        .map(|_| DefaultSampler)
        .collect::<Vec<_>>();
    let mut sampled = Vec::new();
    let mut emitted = Vec::new();
    for step in 0..input_audio.dim(2) {
        let input = input_audio.try_index_device((.., .., step), stream)?;
        let output = model.step_realtime(
            &mut state,
            RealtimeStepInput::encoded_audio(&input),
            RealtimeSampling::new(&mut text_sampler, &mut audio_samplers, 0.0, 0.0, None),
            stream,
        )?;
        if step > 0 {
            let text = output.text_token.squeeze_axes(&[-1], stream)?;
            let text = text.expand_dims(1, stream)?;
            let frame =
                safemlx::ops::concatenate_axis(&[text, output.sampled_audio_tokens], 1, stream)?;
            sampled.push(frame);
        }
        if let Some(tokens) = output.output_audio_tokens {
            emitted.push(tokens);
        }
    }

    let actual_sampled = if sampled.is_empty() {
        Array::zeros::<i32>(&[input_audio.dim(0), 17, 0], stream)?
    } else {
        stack_axis(&sampled, 2, stream)?
    };
    let actual_output_audio = if emitted.is_empty() {
        Array::zeros::<i32>(&[input_audio.dim(0), generated_audio_codebooks, 0], stream)?
    } else {
        stack_axis(&emitted, 2, stream)?
    };
    compare_tokens(
        &actual_sampled,
        expected_sampled,
        stream,
        "sampled model tokens",
    )?;
    compare_tokens(
        &actual_output_audio,
        expected_output_audio,
        stream,
        "delay-aligned output audio",
    )?;

    println!(
        "PersonaPlex PyTorch parity passed: {} input frames, {} sampled frames, {} emitted frames",
        input_audio.dim(2),
        actual_sampled.dim(2),
        actual_output_audio.dim(2)
    );
    Ok(())
}

fn required<'a>(fixture: &'a HashMap<String, Array>, key: &str) -> anyhow::Result<&'a Array> {
    fixture
        .get(key)
        .ok_or_else(|| anyhow::anyhow!("fixture is missing tensor {key}"))
}

fn compare_tokens(
    actual: &Array,
    expected: &Array,
    stream: &Stream,
    label: &str,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        actual.shape() == expected.shape(),
        "{label}: shape mismatch: Rust {:?}, upstream {:?}",
        actual.shape(),
        expected.shape()
    );
    let expected = expected.copy(stream)?;
    let equal = actual.eq(&expected, stream)?.all(None, stream)?;
    anyhow::ensure!(equal.item::<bool>(stream), "{label}: token mismatch");
    Ok(())
}
