use std::{
    error::Error,
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
    time::Instant,
};

use safemlx::{random::RandomState, Array, Device, DeviceType, Dtype, ExecutionContext, Stream};
use safemlx_codec::mimi::Mimi;
use safemlx_lm::{
    models::{moshi, personaplex},
    realtime::RealtimeSpeechModel,
    sampler::{DefaultSampler, GenerationSampler},
};
use sentencepiece_rs::SentencePieceProcessor;
use serde::Serialize;
use serde_json::json;

const SAMPLE_RATE: u32 = 24_000;
const FRAME_RATE: f64 = 12.5;
const FRAME_SAMPLES: usize = 1_920;
const DEADLINE_MS: f64 = 1_000.0 / FRAME_RATE;
const TAIL_ACTIVITY_FRAMES: usize = 3;
const ACTIVE_AUDIO_DBFS: f64 = -40.0;
const DEFAULT_SAMPLING_SEED: u64 = 20260713;
const TEXT_TEMPERATURE: f32 = 0.7;
const AUDIO_TEMPERATURE: f32 = 0.8;
const TEXT_TOP_K: i32 = 25;
const AUDIO_TOP_K: i32 = 250;
const PROMPT_SILENCE_FRAMES: usize = 6;
const DEFAULT_TEXT_PROMPT: &str = "You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way.";

type EvalResult<T> = Result<T, Box<dyn Error>>;

fn main() -> EvalResult<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() < 7 {
        return Err(invalid(
            "usage: personaplex_quantization_eval <dense-model-dir> <quantized-model-dir> <mimi.safetensors> <text-tokenizer.model> <voice-prompt-mono-24khz-f32le> <input-mono-24khz-f32le> <output-dir> [frames] [text-prompt] [sampling-seed]",
        ));
    }
    let dense_dir = PathBuf::from(&args[0]);
    let quantized_dir = PathBuf::from(&args[1]);
    let mimi_path = PathBuf::from(&args[2]);
    let text_tokenizer_path = PathBuf::from(&args[3]);
    let voice_prompt_path = PathBuf::from(&args[4]);
    let input_path = PathBuf::from(&args[5]);
    let output_dir = PathBuf::from(&args[6]);
    let requested_frames = args
        .get(7)
        .map(|value| value.parse::<usize>())
        .transpose()?;
    let text_prompt = args
        .get(8)
        .map(String::as_str)
        .unwrap_or(DEFAULT_TEXT_PROMPT);
    let sampling_seed = args
        .get(9)
        .map(|value| value.parse::<u64>())
        .transpose()?
        .unwrap_or(DEFAULT_SAMPLING_SEED);
    if output_dir.exists() {
        return Err(invalid(format!(
            "output directory already exists: {}",
            output_dir.display()
        )));
    }

    let voice_prompt_pcm = read_f32le(&voice_prompt_path)?;
    if voice_prompt_pcm.len() < FRAME_SAMPLES {
        return Err(invalid(
            "voice prompt must contain at least one complete 80 ms frame",
        ));
    }
    let pcm = read_f32le(&input_path)?;
    let available_frames = pcm.len() / FRAME_SAMPLES;
    let frames = requested_frames
        .unwrap_or(available_frames)
        .min(available_frames);
    if frames < 4 {
        return Err(invalid(format!(
            "input must contain at least 4 complete 80 ms frames; found {available_frames}"
        )));
    }
    let pcm = &pcm[..frames * FRAME_SAMPLES];
    let input_tail_max_rms_dbfs = tail_max_rms_dbfs(pcm);
    let input_likely_truncated = input_tail_max_rms_dbfs > ACTIVE_AUDIO_DBFS;
    let input_warning = input_likely_truncated.then_some(
        "The final 240 ms contains active audio; the frame limit may truncate the user utterance.",
    );
    if let Some(warning) = input_warning {
        eprintln!("warning: {warning} tail_max_rms_dbfs={input_tail_max_rms_dbfs:.1}");
    }

    println!("dense_model={}", dense_dir.display());
    println!("quantized_model={}", quantized_dir.display());
    println!("mimi={}", mimi_path.display());
    println!("text_tokenizer={}", text_tokenizer_path.display());
    println!("voice_prompt={}", voice_prompt_path.display());
    println!("input={}", input_path.display());
    println!("frames={frames} audio_s={:.3}", frames as f64 / FRAME_RATE);
    println!("sampling_seed={sampling_seed}");

    let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
    let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = gpu.stream();
    let weights_stream = cpu.stream();

    let codec_start = Instant::now();
    let mut mimi = Mimi::load(
        &mimi_path,
        Some(personaplex::AUDIO_TOKENS_PER_STREAM),
        stream,
    )?;
    let voice_prompt_tokens = encode_pcm(&mut mimi, &voice_prompt_pcm, stream)?;
    let input_tokens = encode_pcm(&mut mimi, pcm, stream)?;
    stream.synchronize()?;
    let encode_s = codec_start.elapsed().as_secs_f64();
    if input_tokens.len() != frames {
        return Err(invalid(format!(
            "Mimi produced {} token frames for {frames} PCM frames",
            input_tokens.len()
        )));
    }
    let text_tokenizer = SentencePieceProcessor::open(&text_tokenizer_path)?;
    let wrapped_text_prompt = personaplex::wrap_system_prompt(text_prompt);
    let text_prompt_tokens = text_tokenizer
        .encode_to_ids(&wrapped_text_prompt)?
        .into_iter()
        .map(i32::try_from)
        .collect::<Result<Vec<_>, _>>()?;
    if text_prompt_tokens.is_empty() {
        return Err(invalid("text prompt tokenized to an empty sequence"));
    }
    let prompt = PromptConditioning {
        voice_frames: voice_prompt_tokens,
        text_tokens: text_prompt_tokens,
    };
    println!(
        "voice_prompt_frames={} text_prompt_tokens={}",
        prompt.voice_frames.len(),
        prompt.text_tokens.len()
    );
    let offline_pcm_array = Array::from_slice(pcm, &[1, 1, pcm.len() as i32]);
    let offline_codes = mimi.encode(&offline_pcm_array, stream)?;
    safemlx::transforms::eval([&offline_codes])?;
    stream.synchronize()?;
    let offline_input_tokens = code_array_to_frames(&offline_codes)?;
    let offline_roundtrip = mimi.decode(&offline_codes, stream)?;
    safemlx::transforms::eval([&offline_roundtrip])?;
    stream.synchronize()?;
    let offline_codec_roundtrip_pcm =
        fit_pcm_length(array_f32(&offline_roundtrip, stream)?, pcm.len());
    let codec_roundtrip_pcm =
        fit_pcm_length(decode_tokens(&mut mimi, &input_tokens, stream)?, pcm.len());
    let streaming_offline_token_agreement =
        token_frame_agreement(&input_tokens, &offline_input_tokens);
    println!("mimi_streaming_offline_token_agreement={streaming_offline_token_agreement:.4}");

    let dense_load_start = Instant::now();
    let mut dense = personaplex::load_model(&dense_dir, stream, weights_stream)?;
    stream.synchronize()?;
    let dense_load_s = dense_load_start.elapsed().as_secs_f64();
    warmup(&mut dense, &input_tokens[0], stream)?;
    let dense_reference = run_dense(&mut dense, &prompt, &input_tokens, stream)?;
    let dense_run = run_free(&mut dense, &prompt, &input_tokens, sampling_seed, stream)?;
    drop(dense);
    safemlx::transforms::compile::clear_cache()?;

    let quantized_load_start = Instant::now();
    let mut quantized = personaplex::load_model(&quantized_dir, stream, weights_stream)?;
    stream.synchronize()?;
    let quantized_load_s = quantized_load_start.elapsed().as_secs_f64();
    warmup(&mut quantized, &input_tokens[0], stream)?;
    let quality = run_teacher_forced_quality(
        &mut quantized,
        &prompt,
        &input_tokens,
        &dense_reference.reference,
        stream,
    )?;
    let quantized_run = run_free(
        &mut quantized,
        &prompt,
        &input_tokens,
        sampling_seed,
        stream,
    )?;
    drop(quantized);
    safemlx::transforms::compile::clear_cache()?;

    let decode_start = Instant::now();
    let dense_pcm = fit_pcm_length(
        decode_tokens(&mut mimi, &dense_run.emitted_audio, stream)?,
        pcm.len(),
    );
    let quantized_pcm = fit_pcm_length(
        decode_tokens(&mut mimi, &quantized_run.emitted_audio, stream)?,
        pcm.len(),
    );
    let decode_s = decode_start.elapsed().as_secs_f64();

    fs::create_dir(&output_dir)?;
    write_wav_pcm16(&output_dir.join("input.wav"), pcm, SAMPLE_RATE)?;
    write_wav_pcm16(
        &output_dir.join("input_codec_roundtrip.wav"),
        &codec_roundtrip_pcm,
        SAMPLE_RATE,
    )?;
    write_wav_pcm16(
        &output_dir.join("input_codec_roundtrip_offline.wav"),
        &offline_codec_roundtrip_pcm,
        SAMPLE_RATE,
    )?;
    let swap = blind_swap_key(&input_path, &output_dir, frames);
    let dense_tail_max_rms_dbfs = tail_max_rms_dbfs(&dense_pcm);
    let quantized_tail_max_rms_dbfs = tail_max_rms_dbfs(&quantized_pcm);
    let (a_pcm, b_pcm, a_label, b_label, a_tail_max_rms_dbfs, b_tail_max_rms_dbfs) = if swap {
        (
            &quantized_pcm,
            &dense_pcm,
            "quantized",
            "dense",
            quantized_tail_max_rms_dbfs,
            dense_tail_max_rms_dbfs,
        )
    } else {
        (
            &dense_pcm,
            &quantized_pcm,
            "dense",
            "quantized",
            dense_tail_max_rms_dbfs,
            quantized_tail_max_rms_dbfs,
        )
    };
    let a_likely_truncated = a_tail_max_rms_dbfs > ACTIVE_AUDIO_DBFS;
    let b_likely_truncated = b_tail_max_rms_dbfs > ACTIVE_AUDIO_DBFS;
    if a_likely_truncated || b_likely_truncated {
        eprintln!(
            "warning: generated speech is active at the output boundary; sample_a_tail_dbfs={a_tail_max_rms_dbfs:.1} sample_b_tail_dbfs={b_tail_max_rms_dbfs:.1}"
        );
    }
    write_wav_pcm16(&output_dir.join("sample_a.wav"), a_pcm, SAMPLE_RATE)?;
    write_wav_pcm16(&output_dir.join("sample_b.wav"), b_pcm, SAMPLE_RATE)?;

    let dense_performance = performance_summary(&dense_run.latencies_ms);
    let quantized_performance = performance_summary(&quantized_run.latencies_ms);
    let free_agreement = free_run_agreement(&dense_run.reference, &quantized_run.reference, 8);
    let metrics = json!({
        "format_version": 1,
        "input": {
            "path": input_path,
            "sample_rate": SAMPLE_RATE,
            "frame_rate": FRAME_RATE,
            "frames": frames,
            "audio_seconds": frames as f64 / FRAME_RATE,
            "tail_max_rms_dbfs": input_tail_max_rms_dbfs,
            "likely_truncated": input_likely_truncated,
            "warning": input_warning,
        },
        "conditioning": {
            "voice_prompt_path": voice_prompt_path,
            "voice_prompt_frames": prompt.voice_frames.len(),
            "text_tokenizer_path": text_tokenizer_path,
            "text_prompt": text_prompt,
            "wrapped_text_prompt": wrapped_text_prompt,
            "text_prompt_tokens": prompt.text_tokens.len(),
            "sequence": ["voice_prompt", "silence_frames", "text_prompt", "silence_frames"],
            "silence_frames_after_voice": PROMPT_SILENCE_FRAMES,
            "silence_frames_after_text": PROMPT_SILENCE_FRAMES,
        },
        "codec_diagnostic": {
            "streaming_roundtrip": "input_codec_roundtrip.wav",
            "offline_roundtrip": "input_codec_roundtrip_offline.wav",
            "streaming_offline_token_agreement": streaming_offline_token_agreement,
        },
        "performance": {
            "frame_deadline_ms": DEADLINE_MS,
            "codec_encode_seconds": encode_s,
            "codec_decode_both_outputs_seconds": decode_s,
            "dense": {
                "load_seconds": dense_load_s,
                "model": dense_performance,
            },
            "quantized": {
                "load_seconds": quantized_load_s,
                "model": quantized_performance,
            },
        },
        "teacher_forced_quality": quality,
        "free_run_divergence_diagnostic": free_agreement,
        "listening_test": {
            "input": "input.wav",
            "sample_a": "sample_a.wav",
            "sample_b": "sample_b.wav",
            "sampling": {
                "seed": sampling_seed,
                "text_temperature": TEXT_TEMPERATURE,
                "audio_temperature": AUDIO_TEMPERATURE,
                "text_top_k": TEXT_TOP_K,
                "audio_top_k": AUDIO_TOP_K,
            },
            "sample_a_tail_max_rms_dbfs": a_tail_max_rms_dbfs,
            "sample_b_tail_max_rms_dbfs": b_tail_max_rms_dbfs,
            "sample_a_likely_truncated": a_likely_truncated,
            "sample_b_likely_truncated": b_likely_truncated,
            "input_warning": input_warning,
            "instructions": "Listen blind; rate naturalness, intelligibility, voice/persona consistency, semantic quality, and turn timing before opening answer_key.json. Free-running responses need not contain identical words. The PersonaPlex 7B v1 checkpoint is intended for English input.",
        },
    });
    fs::write(
        output_dir.join("metrics.json"),
        serde_json::to_vec_pretty(&metrics)?,
    )?;
    fs::write(
        output_dir.join("answer_key.json"),
        serde_json::to_vec_pretty(&json!({ "sample_a": a_label, "sample_b": b_label }))?,
    )?;
    fs::write(
        output_dir.join("listening_manifest.json"),
        serde_json::to_vec_pretty(&json!({
            "format_version": 1,
            "trials": [{
                "id": "personaplex_quantization_001",
                "input": "input.wav",
                "sample_a": "sample_a.wav",
                "sample_b": "sample_b.wav",
                "input_warning": input_warning,
                "sample_a_likely_truncated": a_likely_truncated,
                "sample_b_likely_truncated": b_likely_truncated,
                "rate_each_1_to_5": [
                    "naturalness",
                    "intelligibility",
                    "voice_persona_consistency",
                    "semantic_response_quality",
                    "turn_timing"
                ],
                "forced_choice": ["a_better", "same", "b_better"]
            }],
            "instructions": "Complete ratings before opening answer_key.json. Different wording is not itself a defect. Confirm that both samples contain intelligible speech before scoring quantization quality."
        }))?,
    )?;
    fs::write(
        output_dir.join("token_diagnostics.json"),
        serde_json::to_vec_pretty(&json!({
            "input": input_tokens,
            "input_offline": offline_input_tokens,
            "conditioning": {
                "voice_prompt": prompt.voice_frames,
                "text_prompt": prompt.text_tokens,
                "silence_frames_after_voice": PROMPT_SILENCE_FRAMES,
                "silence_frames_after_text": PROMPT_SILENCE_FRAMES,
            },
            "sampling": {
                "seed": sampling_seed,
                "text_temperature": TEXT_TEMPERATURE,
                "audio_temperature": AUDIO_TEMPERATURE,
                "text_top_k": TEXT_TOP_K,
                "audio_top_k": AUDIO_TOP_K,
            },
            "dense_emitted": dense_run.emitted_audio,
            "dense_sampled_frames": reference_tokens(&dense_run.reference),
            "dense_greedy_emitted": dense_reference.emitted_audio,
            "dense_greedy_frames": reference_tokens(&dense_reference.reference),
            "quantized_emitted": quantized_run.emitted_audio,
        }))?,
    )?;

    print_report(
        dense_load_s,
        quantized_load_s,
        &dense_performance,
        &quantized_performance,
        &quality,
        &free_agreement,
        &output_dir,
    );
    Ok(())
}

fn invalid(message: impl Into<String>) -> Box<dyn Error> {
    Box::new(io::Error::new(io::ErrorKind::InvalidInput, message.into()))
}

fn read_f32le(path: &Path) -> EvalResult<Vec<f32>> {
    let bytes = fs::read(path)?;
    if bytes.len() % 4 != 0 {
        return Err(invalid(format!(
            "raw f32le input length must be divisible by 4, got {} bytes",
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect())
}

fn rms_dbfs(samples: &[f32]) -> f64 {
    let mean_square = samples
        .iter()
        .map(|sample| (*sample as f64) * (*sample as f64))
        .sum::<f64>()
        / samples.len().max(1) as f64;
    20.0 * mean_square.sqrt().max(1e-12).log10()
}

fn tail_max_rms_dbfs(samples: &[f32]) -> f64 {
    samples
        .chunks_exact(FRAME_SAMPLES)
        .rev()
        .take(TAIL_ACTIVITY_FRAMES)
        .map(rms_dbfs)
        .fold(f64::NEG_INFINITY, f64::max)
}

fn seeded_random_state(seed: u64, stream: &Stream) -> EvalResult<RandomState> {
    let key = Array::from_slice(&[(seed >> 32) as u32, seed as u32], &[2]).copy(stream)?;
    Ok(RandomState::from_key(key))
}

fn encode_pcm(mimi: &mut Mimi, pcm: &[f32], stream: &Stream) -> EvalResult<Vec<Vec<i32>>> {
    mimi.reset_encode_state();
    let mut output = Vec::with_capacity(pcm.len() / FRAME_SAMPLES);
    for frame in pcm.chunks_exact(FRAME_SAMPLES) {
        let frame = Array::from_slice(frame, &[1, 1, FRAME_SAMPLES as i32]);
        if let Some(tokens) = mimi.encode_step(&frame, stream)? {
            stream.synchronize()?;
            output.push(array_i32(&tokens)?);
        }
    }
    Ok(output)
}

fn warmup(model: &mut moshi::Model, input: &[i32], stream: &Stream) -> EvalResult<()> {
    let mut state = model.new_realtime_state();
    let mut text_sampler = DefaultSampler;
    let mut audio_samplers = (0..model.args.dep_q)
        .map(|_| DefaultSampler)
        .collect::<Vec<_>>();
    let input = Array::from_slice(input, &[1, model.args.input_audio_codebooks()]);
    for _ in 0..4 {
        let output = model.generate_step(
            &mut state,
            &input,
            &mut text_sampler,
            &mut audio_samplers,
            0.0,
            0.0,
            None,
            stream,
        )?;
        safemlx::transforms::eval([&output.text_token, &output.sampled_audio_tokens])?;
        stream.synchronize()?;
    }
    Ok(())
}

struct PromptConditioning {
    voice_frames: Vec<Vec<i32>>,
    text_tokens: Vec<i32>,
}

fn prompted_state(
    model: &mut moshi::Model,
    prompt: &PromptConditioning,
    stream: &Stream,
) -> EvalResult<moshi::GenerationState> {
    let mut state = model.new_realtime_state();
    let mut text_sampler = DefaultSampler;
    let mut audio_samplers = (0..model.args.dep_q)
        .map(|_| DefaultSampler)
        .collect::<Vec<_>>();
    let silence = personaplex::silence_frame(1, stream)?;
    let sine = personaplex::sine_frame(1, stream)?;
    let text_padding = personaplex::text_padding_frame(1, stream)?;

    for frame in &prompt.voice_frames {
        let agent = Array::from_slice(frame, &[1, personaplex::AUDIO_TOKENS_PER_STREAM]);
        force_prompt_frame(
            model,
            &mut state,
            &agent,
            &sine,
            &text_padding,
            &mut text_sampler,
            &mut audio_samplers,
            stream,
        )?;
    }
    for _ in 0..PROMPT_SILENCE_FRAMES {
        force_prompt_frame(
            model,
            &mut state,
            &silence,
            &sine,
            &text_padding,
            &mut text_sampler,
            &mut audio_samplers,
            stream,
        )?;
    }
    for &token in &prompt.text_tokens {
        let text = Array::from_slice(&[token], &[1, 1]);
        force_prompt_frame(
            model,
            &mut state,
            &silence,
            &sine,
            &text,
            &mut text_sampler,
            &mut audio_samplers,
            stream,
        )?;
    }
    for _ in 0..PROMPT_SILENCE_FRAMES {
        force_prompt_frame(
            model,
            &mut state,
            &silence,
            &sine,
            &text_padding,
            &mut text_sampler,
            &mut audio_samplers,
            stream,
        )?;
    }
    Ok(state)
}

#[allow(clippy::too_many_arguments)]
fn force_prompt_frame(
    model: &mut moshi::Model,
    state: &mut moshi::GenerationState,
    agent_audio: &Array,
    user_audio: &Array,
    text: &Array,
    text_sampler: &mut DefaultSampler,
    audio_samplers: &mut [DefaultSampler],
    stream: &Stream,
) -> EvalResult<()> {
    let step = model.generate_step_forced_with_logits(
        state,
        user_audio,
        Some(agent_audio),
        Some(text),
        text_sampler,
        audio_samplers,
        0.0,
        0.0,
        None,
        stream,
    )?;
    let mut arrays = vec![&step.output.text_token, &step.output.sampled_audio_tokens];
    if let Some(logits) = &step.text_logits {
        arrays.push(logits);
    }
    arrays.extend(step.audio_logits.iter());
    safemlx::transforms::eval(arrays)?;
    stream.synchronize()?;
    Ok(())
}

struct ReferenceFrame {
    text_token: i32,
    sampled_audio: Vec<i32>,
    text_logits: Option<Vec<f32>>,
    audio_logits: Vec<Vec<f32>>,
}

struct ModelRun {
    reference: Vec<ReferenceFrame>,
    emitted_audio: Vec<Vec<i32>>,
    latencies_ms: Vec<f64>,
}

fn reference_tokens(reference: &[ReferenceFrame]) -> Vec<serde_json::Value> {
    reference
        .iter()
        .map(|frame| {
            json!({
                "text": frame.text_token,
                "sampled_audio": frame.sampled_audio,
            })
        })
        .collect()
}

fn run_dense(
    model: &mut moshi::Model,
    prompt: &PromptConditioning,
    input_tokens: &[Vec<i32>],
    stream: &Stream,
) -> EvalResult<ModelRun> {
    let mut state = prompted_state(model, prompt, stream)?;
    let mut text_sampler = DefaultSampler;
    let mut audio_samplers = (0..model.args.dep_q)
        .map(|_| DefaultSampler)
        .collect::<Vec<_>>();
    let mut reference = Vec::with_capacity(input_tokens.len());
    let mut emitted_audio = Vec::new();
    let mut latencies_ms = Vec::with_capacity(input_tokens.len());

    for input in input_tokens {
        let input = Array::from_slice(input, &[1, model.args.input_audio_codebooks()]);
        let start = Instant::now();
        let step = model.generate_step_forced_with_logits(
            &mut state,
            &input,
            None,
            None,
            &mut text_sampler,
            &mut audio_samplers,
            0.0,
            0.0,
            None,
            stream,
        )?;
        stream.synchronize()?;
        latencies_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
        let text_token = array_i32(&step.output.text_token)?[0];
        let sampled_audio = array_i32(&step.output.sampled_audio_tokens)?;
        let text_logits = step
            .text_logits
            .as_ref()
            .map(|logits| array_f32(logits, stream))
            .transpose()?;
        let audio_logits = step
            .audio_logits
            .iter()
            .map(|logits| array_f32(logits, stream))
            .collect::<EvalResult<Vec<_>>>()?;
        if let Some(tokens) = &step.output.output_audio_tokens {
            emitted_audio.push(array_i32(tokens)?);
        }
        reference.push(ReferenceFrame {
            text_token,
            sampled_audio,
            text_logits,
            audio_logits,
        });
    }
    Ok(ModelRun {
        reference,
        emitted_audio,
        latencies_ms,
    })
}

#[derive(Debug, Clone, Default)]
struct DistributionAccumulator {
    count: usize,
    target_count: usize,
    kl_nats_sum: f64,
    dense_entropy_nats_sum: f64,
    target_nll_delta_nats_sum: f64,
    centered_logit_rmse_sum: f64,
    top1_matches: usize,
    top5_overlap_sum: f64,
}

impl DistributionAccumulator {
    fn update(&mut self, dense: &[f32], candidate: &[f32], target: usize) -> EvalResult<()> {
        if dense.len() != candidate.len() || dense.is_empty() {
            return Err(invalid("invalid distribution shapes"));
        }
        let dense_lse = logsumexp(dense);
        let candidate_lse = logsumexp(candidate);
        let mut kl = 0.0;
        let mut entropy = 0.0;
        for (&dense_logit, &candidate_logit) in dense.iter().zip(candidate) {
            let log_p = dense_logit as f64 - dense_lse;
            let log_q = candidate_logit as f64 - candidate_lse;
            let p = log_p.exp();
            kl += p * (log_p - log_q);
            entropy -= p * log_p;
        }
        let dense_mean = dense.iter().map(|value| *value as f64).sum::<f64>() / dense.len() as f64;
        let candidate_mean =
            candidate.iter().map(|value| *value as f64).sum::<f64>() / candidate.len() as f64;
        let rmse = dense
            .iter()
            .zip(candidate)
            .map(|(&dense, &candidate)| {
                let difference = (dense as f64 - dense_mean) - (candidate as f64 - candidate_mean);
                difference * difference
            })
            .sum::<f64>()
            / dense.len() as f64;
        let rmse = rmse.sqrt();
        let dense_top = top_indices(dense, 5);
        let candidate_top = top_indices(candidate, 5);
        let overlap = dense_top
            .iter()
            .filter(|index| candidate_top.contains(index))
            .count() as f64
            / dense_top.len() as f64;
        self.count += 1;
        self.kl_nats_sum += kl.max(0.0);
        self.dense_entropy_nats_sum += entropy;
        if target < dense.len() {
            self.target_count += 1;
            self.target_nll_delta_nats_sum +=
                (candidate_lse - candidate[target] as f64) - (dense_lse - dense[target] as f64);
        }
        self.centered_logit_rmse_sum += rmse;
        self.top1_matches += usize::from(dense_top[0] == candidate_top[0]);
        self.top5_overlap_sum += overlap;
        Ok(())
    }

    fn merge(&mut self, other: &Self) {
        self.count += other.count;
        self.target_count += other.target_count;
        self.kl_nats_sum += other.kl_nats_sum;
        self.dense_entropy_nats_sum += other.dense_entropy_nats_sum;
        self.target_nll_delta_nats_sum += other.target_nll_delta_nats_sum;
        self.centered_logit_rmse_sum += other.centered_logit_rmse_sum;
        self.top1_matches += other.top1_matches;
        self.top5_overlap_sum += other.top5_overlap_sum;
    }

    fn summary(&self) -> MetricSummary {
        let count = self.count.max(1) as f64;
        MetricSummary {
            distributions: self.count,
            target_distributions: self.target_count,
            mean_kl_nats: self.kl_nats_sum / count,
            mean_dense_entropy_nats: self.dense_entropy_nats_sum / count,
            mean_target_nll_delta_nats: self.target_nll_delta_nats_sum
                / self.target_count.max(1) as f64,
            mean_centered_logit_rmse: self.centered_logit_rmse_sum / count,
            top1_agreement: self.top1_matches as f64 / count,
            mean_top5_overlap: self.top5_overlap_sum / count,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct MetricSummary {
    distributions: usize,
    target_distributions: usize,
    mean_kl_nats: f64,
    mean_dense_entropy_nats: f64,
    mean_target_nll_delta_nats: f64,
    mean_centered_logit_rmse: f64,
    top1_agreement: f64,
    mean_top5_overlap: f64,
}

#[derive(Debug, Clone, Serialize)]
struct QualitySummary {
    methodology: &'static str,
    text: MetricSummary,
    audio_generated: MetricSummary,
    audio_input_conditioned: MetricSummary,
    audio_overall: MetricSummary,
    audio_by_codebook: Vec<MetricSummary>,
}

fn run_teacher_forced_quality(
    model: &mut moshi::Model,
    prompt: &PromptConditioning,
    input_tokens: &[Vec<i32>],
    reference: &[ReferenceFrame],
    stream: &Stream,
) -> EvalResult<QualitySummary> {
    let mut state = prompted_state(model, prompt, stream)?;
    let mut text_sampler = DefaultSampler;
    let mut audio_samplers = (0..model.args.dep_q)
        .map(|_| DefaultSampler)
        .collect::<Vec<_>>();
    let mut text = DistributionAccumulator::default();
    let mut audio = (0..model.args.dep_q)
        .map(|_| DistributionAccumulator::default())
        .collect::<Vec<_>>();

    for (input, reference) in input_tokens.iter().zip(reference) {
        let input = Array::from_slice(input, &[1, model.args.input_audio_codebooks()]);
        let forced_text = Array::from_slice(&[reference.text_token], &[1, 1]);
        let generated = model.args.generated_audio_codebooks() as usize;
        let forced_audio = Array::from_slice(
            &reference.sampled_audio[..generated],
            &[1, generated as i32],
        );
        let step = model.generate_step_forced_with_logits(
            &mut state,
            &input,
            Some(&forced_audio),
            Some(&forced_text),
            &mut text_sampler,
            &mut audio_samplers,
            0.0,
            0.0,
            None,
            stream,
        )?;
        stream.synchronize()?;
        if let (Some(dense_logits), Some(candidate_logits)) =
            (&reference.text_logits, step.text_logits.as_ref())
        {
            let candidate_logits = array_f32(candidate_logits, stream)?;
            text.update(
                dense_logits,
                &candidate_logits,
                reference.text_token as usize,
            )?;
        }
        if reference.audio_logits.len() != step.audio_logits.len() {
            return Err(invalid("dense and quantized audio head counts differ"));
        }
        for (codebook, (dense_logits, candidate_logits)) in reference
            .audio_logits
            .iter()
            .zip(&step.audio_logits)
            .enumerate()
        {
            let candidate_logits = array_f32(candidate_logits, stream)?;
            audio[codebook].update(
                dense_logits,
                &candidate_logits,
                reference.sampled_audio[codebook] as usize,
            )?;
        }
    }
    let mut overall = DistributionAccumulator::default();
    for accumulator in &audio {
        overall.merge(accumulator);
    }
    let generated_codebooks = model.args.generated_audio_codebooks() as usize;
    let mut generated = DistributionAccumulator::default();
    for accumulator in audio.iter().take(generated_codebooks) {
        generated.merge(accumulator);
    }
    let mut input_conditioned = DistributionAccumulator::default();
    for accumulator in audio.iter().skip(generated_codebooks) {
        input_conditioned.merge(accumulator);
    }
    Ok(QualitySummary {
        methodology: "Both models receive the same voice and wrapped text prompt sequence. The quantized model is then teacher-forced onto the dense model's exact text and generated-audio token history; KL uses the dense distribution as reference.",
        text: text.summary(),
        audio_generated: generated.summary(),
        audio_input_conditioned: input_conditioned.summary(),
        audio_overall: overall.summary(),
        audio_by_codebook: audio.iter().map(DistributionAccumulator::summary).collect(),
    })
}

fn run_free(
    model: &mut moshi::Model,
    prompt: &PromptConditioning,
    input_tokens: &[Vec<i32>],
    sampling_seed: u64,
    stream: &Stream,
) -> EvalResult<ModelRun> {
    let mut state = prompted_state(model, prompt, stream)?;
    let mut text_sampler = GenerationSampler::new()
        .top_k(TEXT_TOP_K)
        .top_p(1.0)
        .min_p(0.0);
    let mut audio_samplers = (0..model.args.dep_q)
        .map(|_| {
            GenerationSampler::new()
                .top_k(AUDIO_TOP_K)
                .top_p(1.0)
                .min_p(0.0)
        })
        .collect::<Vec<_>>();
    let mut prng_state = seeded_random_state(sampling_seed, stream)?;
    let mut reference = Vec::with_capacity(input_tokens.len());
    let mut emitted_audio = Vec::new();
    let mut latencies_ms = Vec::with_capacity(input_tokens.len());

    for input in input_tokens {
        let input = Array::from_slice(input, &[1, model.args.input_audio_codebooks()]);
        let start = Instant::now();
        let output = model.generate_step(
            &mut state,
            &input,
            &mut text_sampler,
            &mut audio_samplers,
            TEXT_TEMPERATURE,
            AUDIO_TEMPERATURE,
            Some(&mut prng_state),
            stream,
        )?;
        if let Some(tokens) = &output.output_audio_tokens {
            safemlx::transforms::eval([&output.text_token, &output.sampled_audio_tokens, tokens])?;
        } else {
            safemlx::transforms::eval([&output.text_token, &output.sampled_audio_tokens])?;
        }
        stream.synchronize()?;
        latencies_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
        let text_token = array_i32(&output.text_token)?[0];
        let sampled_audio = array_i32(&output.sampled_audio_tokens)?;
        if let Some(tokens) = &output.output_audio_tokens {
            emitted_audio.push(array_i32(tokens)?);
        }
        reference.push(ReferenceFrame {
            text_token,
            sampled_audio,
            text_logits: None,
            audio_logits: Vec::new(),
        });
    }
    Ok(ModelRun {
        reference,
        emitted_audio,
        latencies_ms,
    })
}

fn logsumexp(values: &[f32]) -> f64 {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    max + values
        .iter()
        .map(|value| (*value as f64 - max).exp())
        .sum::<f64>()
        .ln()
}

fn top_indices(values: &[f32], count: usize) -> Vec<usize> {
    let mut indices = (0..values.len()).collect::<Vec<_>>();
    indices.sort_unstable_by(|left, right| values[*right].total_cmp(&values[*left]));
    indices.truncate(count.min(indices.len()));
    indices
}

fn array_i32(array: &Array) -> EvalResult<Vec<i32>> {
    let evaluated = array.evaluated()?;
    match array.dtype() {
        Dtype::Int32 => Ok(evaluated.as_slice::<i32>().to_vec()),
        Dtype::Uint32 => evaluated
            .as_slice::<u32>()
            .iter()
            .map(|value| i32::try_from(*value).map_err(Into::into))
            .collect(),
        Dtype::Int64 => evaluated
            .as_slice::<i64>()
            .iter()
            .map(|value| i32::try_from(*value).map_err(Into::into))
            .collect(),
        Dtype::Uint64 => evaluated
            .as_slice::<u64>()
            .iter()
            .map(|value| i32::try_from(*value).map_err(Into::into))
            .collect(),
        dtype => Err(invalid(format!(
            "expected integer token array, got {dtype:?}"
        ))),
    }
}

fn code_array_to_frames(array: &Array) -> EvalResult<Vec<Vec<i32>>> {
    if array.shape().len() != 3 || array.dim(0) != 1 {
        return Err(invalid(format!(
            "expected codec tokens shaped [1, codebooks, frames], got {:?}",
            array.shape()
        )));
    }
    let codebooks = array.dim(1) as usize;
    let frames = array.dim(2) as usize;
    let flattened = array_i32(array)?;
    Ok((0..frames)
        .map(|frame| {
            (0..codebooks)
                .map(|codebook| flattened[codebook * frames + frame])
                .collect()
        })
        .collect())
}

fn token_frame_agreement(left: &[Vec<i32>], right: &[Vec<i32>]) -> f64 {
    let mut matches = 0usize;
    let mut total = 0usize;
    for (left, right) in left.iter().zip(right) {
        for (left, right) in left.iter().zip(right) {
            matches += usize::from(left == right);
            total += 1;
        }
    }
    matches as f64 / total.max(1) as f64
}

fn array_f32(array: &Array, stream: &Stream) -> EvalResult<Vec<f32>> {
    let array = if array.dtype() == Dtype::Float32 {
        array.clone()
    } else {
        array.as_dtype(Dtype::Float32, stream)?
    };
    Ok(array.evaluated()?.as_slice::<f32>().to_vec())
}

fn decode_tokens(
    mimi: &mut Mimi,
    token_frames: &[Vec<i32>],
    stream: &Stream,
) -> EvalResult<Vec<f32>> {
    mimi.reset_decode_state();
    let mut pcm = Vec::with_capacity(token_frames.len() * FRAME_SAMPLES);
    for frame in token_frames {
        let tokens = Array::from_slice(frame, &[1, frame.len() as i32]);
        let decoded = mimi.decode_step(&tokens, stream)?;
        stream.synchronize()?;
        pcm.extend(array_f32(&decoded, stream)?);
    }
    Ok(pcm)
}

fn fit_pcm_length(mut pcm: Vec<f32>, target_samples: usize) -> Vec<f32> {
    pcm.truncate(target_samples);
    pcm.resize(target_samples, 0.0);
    pcm
}

#[derive(Debug, Clone, Serialize)]
struct PerformanceSummary {
    frames: usize,
    mean_ms: f64,
    p50_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
    max_ms: f64,
    deadline_ms: f64,
    deadline_misses: usize,
    deadline_miss_rate: f64,
    realtime_factor: f64,
    realtime_multiple: f64,
}

fn performance_summary(latencies_ms: &[f64]) -> PerformanceSummary {
    let mut sorted = latencies_ms.to_vec();
    sorted.sort_by(f64::total_cmp);
    let total = latencies_ms.iter().sum::<f64>();
    let mean = total / latencies_ms.len().max(1) as f64;
    let misses = latencies_ms
        .iter()
        .filter(|latency| **latency > DEADLINE_MS)
        .count();
    PerformanceSummary {
        frames: latencies_ms.len(),
        mean_ms: mean,
        p50_ms: percentile(&sorted, 0.50),
        p95_ms: percentile(&sorted, 0.95),
        p99_ms: percentile(&sorted, 0.99),
        max_ms: sorted.last().copied().unwrap_or(0.0),
        deadline_ms: DEADLINE_MS,
        deadline_misses: misses,
        deadline_miss_rate: misses as f64 / latencies_ms.len().max(1) as f64,
        realtime_factor: mean / DEADLINE_MS,
        realtime_multiple: DEADLINE_MS / mean,
    }
}

fn percentile(sorted: &[f64], quantile: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let index = ((sorted.len() - 1) as f64 * quantile).round() as usize;
    sorted[index]
}

#[derive(Debug, Clone, Serialize)]
struct FreeRunAgreement {
    warning: &'static str,
    frames: usize,
    text_token_agreement: f64,
    generated_audio_token_agreement: f64,
    first_text_divergence_frame: Option<usize>,
    first_generated_audio_divergence_frame: Option<usize>,
}

fn free_run_agreement(
    dense: &[ReferenceFrame],
    quantized: &[ReferenceFrame],
    generated_codebooks: usize,
) -> FreeRunAgreement {
    let frames = dense.len().min(quantized.len());
    let mut text_matches = 0;
    let mut audio_matches = 0;
    let mut audio_total = 0;
    let mut first_text = None;
    let mut first_audio = None;
    for (frame, (dense, quantized)) in dense.iter().zip(quantized).enumerate() {
        if dense.text_token == quantized.text_token {
            text_matches += 1;
        } else if first_text.is_none() {
            first_text = Some(frame);
        }
        for (dense, quantized) in dense
            .sampled_audio
            .iter()
            .take(generated_codebooks)
            .zip(quantized.sampled_audio.iter().take(generated_codebooks))
        {
            audio_total += 1;
            if dense == quantized {
                audio_matches += 1;
            } else if first_audio.is_none() {
                first_audio = Some(frame);
            }
        }
    }
    FreeRunAgreement {
        warning: "Autoregressive divergence is not a quality score; different tokens can produce equally valid conversations.",
        frames,
        text_token_agreement: text_matches as f64 / frames.max(1) as f64,
        generated_audio_token_agreement: audio_matches as f64 / audio_total.max(1) as f64,
        first_text_divergence_frame: first_text,
        first_generated_audio_divergence_frame: first_audio,
    }
}

fn write_wav_pcm16(path: &Path, samples: &[f32], sample_rate: u32) -> EvalResult<()> {
    let data_bytes = samples
        .len()
        .checked_mul(2)
        .ok_or_else(|| invalid("WAV data size overflow"))?;
    let data_bytes = u32::try_from(data_bytes)?;
    let riff_bytes = 36u32
        .checked_add(data_bytes)
        .ok_or_else(|| invalid("WAV RIFF size overflow"))?;
    let mut file = fs::File::create(path)?;
    file.write_all(b"RIFF")?;
    file.write_all(&riff_bytes.to_le_bytes())?;
    file.write_all(b"WAVEfmt ")?;
    file.write_all(&16u32.to_le_bytes())?;
    file.write_all(&1u16.to_le_bytes())?;
    file.write_all(&1u16.to_le_bytes())?;
    file.write_all(&sample_rate.to_le_bytes())?;
    file.write_all(&(sample_rate * 2).to_le_bytes())?;
    file.write_all(&2u16.to_le_bytes())?;
    file.write_all(&16u16.to_le_bytes())?;
    file.write_all(b"data")?;
    file.write_all(&data_bytes.to_le_bytes())?;
    for sample in samples {
        let sample = (sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16;
        file.write_all(&sample.to_le_bytes())?;
    }
    Ok(())
}

fn blind_swap_key(input_path: &Path, output_dir: &Path, frames: usize) -> bool {
    input_path
        .as_os_str()
        .as_encoded_bytes()
        .iter()
        .chain(output_dir.as_os_str().as_encoded_bytes())
        .fold(frames as u64, |hash, byte| {
            hash.wrapping_mul(1_099_511_628_211)
                .wrapping_add(*byte as u64)
        })
        & 1
        == 1
}

fn print_report(
    dense_load_s: f64,
    quantized_load_s: f64,
    dense: &PerformanceSummary,
    quantized: &PerformanceSummary,
    quality: &QualitySummary,
    agreement: &FreeRunAgreement,
    output_dir: &Path,
) {
    println!(
        "dense_load_s={dense_load_s:.3} quantized_load_s={quantized_load_s:.3} load_reduction_pct={:.1}",
        100.0 * (dense_load_s - quantized_load_s) / dense_load_s
    );
    println!(
        "dense_mean_ms={:.3} dense_p95_ms={:.3} dense_p99_ms={:.3} dense_deadline_misses={}",
        dense.mean_ms, dense.p95_ms, dense.p99_ms, dense.deadline_misses
    );
    println!(
        "quantized_mean_ms={:.3} quantized_p95_ms={:.3} quantized_p99_ms={:.3} quantized_deadline_misses={}",
        quantized.mean_ms,
        quantized.p95_ms,
        quantized.p99_ms,
        quantized.deadline_misses
    );
    println!(
        "frame_time_reduction_pct={:.1} realtime_capacity_ratio={:.2}",
        100.0 * (dense.mean_ms - quantized.mean_ms) / dense.mean_ms,
        quantized.realtime_multiple / dense.realtime_multiple
    );
    println!(
        "text_kl_nats={:.6} text_target_nll_delta_nats={:.6} text_top1_agreement={:.4} text_top5_overlap={:.4}",
        quality.text.mean_kl_nats,
        quality.text.mean_target_nll_delta_nats,
        quality.text.top1_agreement,
        quality.text.mean_top5_overlap
    );
    println!(
        "generated_audio_kl_nats={:.6} generated_audio_target_nll_delta_nats={:.6} generated_audio_top1_agreement={:.4} generated_audio_top5_overlap={:.4}",
        quality.audio_generated.mean_kl_nats,
        quality.audio_generated.mean_target_nll_delta_nats,
        quality.audio_generated.top1_agreement,
        quality.audio_generated.mean_top5_overlap
    );
    println!(
        "free_text_token_agreement={:.4} free_generated_audio_token_agreement={:.4} first_text_divergence={:?} first_audio_divergence={:?}",
        agreement.text_token_agreement,
        agreement.generated_audio_token_agreement,
        agreement.first_text_divergence_frame,
        agreement.first_generated_audio_divergence_frame
    );
    println!("artifacts={}", output_dir.display());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_dbfs_detects_active_and_silent_audio() {
        assert!((rms_dbfs(&[1.0, -1.0]) - 0.0).abs() < 1e-12);
        assert!(rms_dbfs(&[0.0; 16]) < -200.0);
    }

    #[test]
    fn output_directory_changes_blind_assignment() {
        assert_ne!(
            blind_swap_key(Path::new("input"), Path::new("a"), 1),
            blind_swap_key(Path::new("input"), Path::new("b"), 1),
        );
    }

    #[test]
    fn identical_distributions_have_zero_drift() {
        let logits = [0.0, 1.0, -1.0, 0.5, 0.25];
        let mut metric = DistributionAccumulator::default();
        metric.update(&logits, &logits, 1).unwrap();
        let summary = metric.summary();
        assert!(summary.mean_kl_nats.abs() < 1e-12);
        assert!(summary.mean_target_nll_delta_nats.abs() < 1e-12);
        assert!(summary.mean_centered_logit_rmse.abs() < 1e-12);
        assert_eq!(summary.top1_agreement, 1.0);
        assert_eq!(summary.mean_top5_overlap, 1.0);
    }

    #[test]
    fn shifted_logits_do_not_change_distribution_metrics() {
        let dense = [0.0, 1.0, -1.0, 0.5, 0.25];
        let shifted = [10.0, 11.0, 9.0, 10.5, 10.25];
        let mut metric = DistributionAccumulator::default();
        metric.update(&dense, &shifted, 1).unwrap();
        let summary = metric.summary();
        assert!(summary.mean_kl_nats < 1e-12);
        assert!(summary.mean_centered_logit_rmse < 1e-12);
    }

    #[test]
    fn wav_writer_emits_valid_mono_pcm_header() {
        let path = std::env::temp_dir().join(format!(
            "safemlx-eval-wav-{}-{}.wav",
            std::process::id(),
            FRAME_SAMPLES
        ));
        write_wav_pcm16(&path, &[0.0, 0.5, -0.5], SAMPLE_RATE).unwrap();
        let bytes = fs::read(&path).unwrap();
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        assert_eq!(&bytes[36..40], b"data");
        assert_eq!(u32::from_le_bytes(bytes[40..44].try_into().unwrap()), 6);
        fs::remove_file(path).unwrap();
    }
}
