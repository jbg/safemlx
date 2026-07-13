#!/usr/bin/env python3
"""Generate an upstream PyTorch PersonaPlex reference from safemlx token fixtures.

The input JSON is produced by the safemlx-codec
`personaplex_quantization_eval` example. Voice, text, and user-audio codec
tokens are consumed directly so the backend comparison does not depend on
separate tokenizer or Mimi encoder numerics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np
import torch


SAMPLE_RATE = 24_000
FRAME_RATE = 12.5
FRAME_SAMPLES = 1_920
ACTIVE_AUDIO_DBFS = -40.0
TAIL_ACTIVITY_FRAMES = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare dense safemlx PersonaPlex with upstream PyTorch."
    )
    parser.add_argument("--moshi-source", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--mimi", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--safemlx-eval-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--reuse-pytorch-wav",
        type=Path,
        help="Reuse a completed upstream sampled WAV and run only the greedy parity prefix.",
    )
    parser.add_argument(
        "--greedy-frames",
        type=int,
        default=64,
        help="Input frames used for deterministic parity (default: 64); sampled listening always uses all frames.",
    )
    return parser.parse_args()


def synchronize(device: str) -> None:
    if device.startswith("mps"):
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


def seed_backend(seed: int, device: str) -> None:
    torch.manual_seed(seed)
    if device.startswith("mps"):
        torch.mps.manual_seed(seed)
    elif device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)


def validate_frames(name: str, frames: list[list[int]], width: int) -> None:
    if not frames:
        raise ValueError(f"{name} contains no frames")
    for index, frame in enumerate(frames):
        if len(frame) != width:
            raise ValueError(
                f"{name}[{index}] has {len(frame)} tokens; expected {width}"
            )


def prompt_lm(lm_gen: Any, fixture: dict[str, Any], device: str) -> None:
    conditioning = fixture["conditioning"]
    voice_frames = conditioning["voice_prompt"]
    text_tokens = conditioning["text_prompt"]
    silence_after_voice = conditioning["silence_frames_after_voice"]
    silence_after_text = conditioning["silence_frames_after_text"]
    if silence_after_voice != lm_gen.audio_silence_frame_cnt:
        raise ValueError("voice-prompt silence count differs from upstream LMGen")
    if silence_after_text != lm_gen.audio_silence_frame_cnt:
        raise ValueError("text-prompt silence count differs from upstream LMGen")

    for frame in voice_frames:
        tokens = torch.tensor(frame, dtype=torch.long, device=device).view(1, 8, 1)
        lm_gen._step_voice_prompt_frame(tokens)
    lm_gen._step_audio_silence()
    lm_gen.text_prompt_tokens = text_tokens
    lm_gen._step_text_prompt()
    lm_gen._step_audio_silence()


def run_lm(
    lm: Any,
    lm_gen_type: Any,
    fixture: dict[str, Any],
    device: str,
    sampled: bool,
    max_frames: int | None = None,
) -> dict[str, Any]:
    sampling = fixture["sampling"]
    lm_gen = lm_gen_type(
        lm,
        device=device,
        use_sampling=sampled,
        temp=float(sampling["audio_temperature"]),
        temp_text=float(sampling["text_temperature"]),
        top_k=int(sampling["audio_top_k"]),
        top_k_text=int(sampling["text_top_k"]),
        audio_silence_frame_cnt=int(
            fixture["conditioning"]["silence_frames_after_voice"]
        ),
        sample_rate=SAMPLE_RATE,
        frame_rate=FRAME_RATE,
    )
    lm_gen.streaming_forever(1)
    prompt_lm(lm_gen, fixture, device)
    seed_backend(int(sampling["seed"]), device)

    output_frames: list[list[int]] = []
    latencies_ms: list[float] = []
    input_frames = fixture["input"]
    if max_frames is not None:
        input_frames = input_frames[:max_frames]
    for index, frame in enumerate(input_frames, start=1):
        input_tokens = torch.tensor(frame, dtype=torch.long, device=device).view(1, 8, 1)
        synchronize(device)
        start = time.perf_counter()
        output = lm_gen.step(input_tokens)
        synchronize(device)
        latencies_ms.append((time.perf_counter() - start) * 1_000.0)
        if output is not None:
            output_frames.append(output[0, :, 0].detach().cpu().tolist())
        if index % 50 == 0 or index == len(input_frames):
            print(f"  frames {index}/{len(input_frames)}", flush=True)

    lm_gen._stop_streaming()
    emitted_audio = [frame[1:9] for frame in output_frames]
    text_tokens = [frame[0] for frame in output_frames]
    ordered = sorted(latencies_ms)
    return {
        "frames": output_frames,
        "emitted_audio": emitted_audio,
        "text_tokens": text_tokens,
        "latency": {
            "count": len(latencies_ms),
            "mean_ms": float(np.mean(latencies_ms)),
            "p50_ms": float(np.percentile(latencies_ms, 50)),
            "p95_ms": float(np.percentile(latencies_ms, 95)),
            "p99_ms": float(np.percentile(latencies_ms, 99)),
            "max_ms": ordered[-1],
            "deadline_misses": sum(value > 80.0 for value in latencies_ms),
        },
    }


def decode_audio(loaders: Any, mimi_path: Path, frames: list[list[int]], device: str) -> np.ndarray:
    mimi = loaders.get_mimi(mimi_path, device=device)
    mimi.streaming_forever(1)
    chunks = []
    for frame in frames:
        codes = torch.tensor(frame, dtype=torch.long, device=device).view(1, 8, 1)
        pcm = mimi.decode(codes)
        chunks.append(pcm[0, 0].detach().cpu().numpy())
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32, copy=False)


def fit_length(samples: np.ndarray, length: int) -> np.ndarray:
    if samples.size >= length:
        return samples[:length]
    return np.pad(samples, (0, length - samples.size))


def write_wav(path: Path, samples: np.ndarray) -> None:
    pcm16 = np.round(np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(pcm16.tobytes())


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as source:
        if source.getnchannels() != 1 or source.getsampwidth() != 2:
            raise ValueError("reused PyTorch WAV must be mono PCM16")
        if source.getframerate() != SAMPLE_RATE:
            raise ValueError(f"reused PyTorch WAV must be {SAMPLE_RATE} Hz")
        return (
            np.frombuffer(source.readframes(source.getnframes()), dtype="<i2")
            .astype(np.float32)
            / 32767.0
        )


def rms_dbfs(samples: np.ndarray) -> float:
    if samples.size == 0:
        return -240.0
    mean_square = float(np.mean(samples.astype(np.float64) ** 2))
    return 20.0 * np.log10(max(mean_square**0.5, 1e-12))


def tail_max_rms_dbfs(samples: np.ndarray) -> float:
    frames = samples.size // FRAME_SAMPLES
    if frames == 0:
        return -240.0
    return max(
        rms_dbfs(samples[index * FRAME_SAMPLES : (index + 1) * FRAME_SAMPLES])
        for index in range(max(0, frames - TAIL_ACTIVITY_FRAMES), frames)
    )


def token_agreement(left: list[list[int]], right: list[list[int]]) -> dict[str, Any]:
    frames = min(len(left), len(right))
    width = min(len(left[0]), len(right[0])) if frames else 0
    total = frames * width
    matches = sum(
        left[frame][token] == right[frame][token]
        for frame in range(frames)
        for token in range(width)
    )
    exact = sum(left[index][:width] == right[index][:width] for index in range(frames))
    per_token = [
        sum(left[frame][token] == right[frame][token] for frame in range(frames))
        / max(frames, 1)
        for token in range(width)
    ]
    return {
        "compared_frames": frames,
        "tokens_per_frame": width,
        "token_agreement": matches / max(total, 1),
        "exact_frame_agreement": exact / max(frames, 1),
        "per_token_agreement": per_token,
        "first_divergence_frame": next(
            (
                frame
                for frame in range(frames)
                if left[frame][:width] != right[frame][:width]
            ),
            None,
        ),
        "left_frames": len(left),
        "right_frames": len(right),
    }


def best_sequence_alignment(left: list[int], right: list[int]) -> dict[str, Any]:
    candidates = []
    for right_offset in range(-2, 3):
        left_start = max(0, -right_offset)
        right_start = max(0, right_offset)
        count = min(len(left) - left_start, len(right) - right_start)
        if count <= 0:
            continue
        matches = sum(
            left[left_start + index] == right[right_start + index]
            for index in range(count)
        )
        candidates.append((matches / count, count, right_offset))
    agreement, count, right_offset = max(candidates)
    return {
        "agreement": agreement,
        "compared_tokens": count,
        "pytorch_frame_offset_relative_to_safemlx": right_offset,
    }


def dense_wav(eval_dir: Path) -> Path:
    key = json.loads((eval_dir / "answer_key.json").read_text())
    for sample in ("sample_a", "sample_b"):
        if key[sample] == "dense":
            return eval_dir / f"{sample}.wav"
    raise ValueError("safemlx answer key does not contain a dense sample")


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    sys.path.insert(0, str(args.moshi_source))
    from moshi.models import LMGen, loaders  # pylint: disable=import-outside-toplevel

    eval_dir = args.safemlx_eval_dir
    fixture = json.loads((eval_dir / "token_diagnostics.json").read_text())
    validate_frames("input", fixture["input"], 8)
    validate_frames("voice_prompt", fixture["conditioning"]["voice_prompt"], 8)

    print(f"loading upstream model on {args.device}", flush=True)
    load_start = time.perf_counter()
    lm = loaders.get_moshi_lm(args.model, device=args.device)
    lm.eval()
    synchronize(args.device)
    load_seconds = time.perf_counter() - load_start

    with torch.no_grad():
        print("running upstream greedy parity trace", flush=True)
        greedy = run_lm(
            lm,
            LMGen,
            fixture,
            args.device,
            sampled=False,
            max_frames=args.greedy_frames,
        )
        if args.reuse_pytorch_wav is None:
            print("running upstream production sampling trace", flush=True)
            sampled = run_lm(lm, LMGen, fixture, args.device, sampled=True)
        else:
            sampled = {
                "frames": [],
                "emitted_audio": [],
                "text_tokens": [],
                "latency": None,
                "reused_wav": str(args.reuse_pytorch_wav),
            }

    del lm
    if args.device.startswith("mps"):
        torch.mps.empty_cache()
    target_samples = len(fixture["input"]) * FRAME_SAMPLES
    if args.reuse_pytorch_wav is None:
        print("decoding upstream output", flush=True)
        with torch.no_grad():
            pytorch_pcm = decode_audio(
                loaders, args.mimi, sampled["emitted_audio"], args.device
            )
        pytorch_pcm = fit_length(pytorch_pcm, target_samples)
    else:
        print(f"reusing upstream output {args.reuse_pytorch_wav}", flush=True)
        pytorch_pcm = fit_length(read_wav(args.reuse_pytorch_wav), target_samples)

    args.output_dir.mkdir()
    pytorch_wav = args.output_dir / "pytorch_unblinded.tmp.wav"
    write_wav(pytorch_wav, pytorch_pcm)
    safemlx_wav = dense_wav(eval_dir)
    assignment_hash = hashlib.sha256(str(args.output_dir).encode()).digest()[0]
    if assignment_hash & 1:
        assignment = {"sample_a": "pytorch", "sample_b": "safemlx"}
        source_a, source_b = pytorch_wav, safemlx_wav
    else:
        assignment = {"sample_a": "safemlx", "sample_b": "pytorch"}
        source_a, source_b = safemlx_wav, pytorch_wav
    shutil.copyfile(source_a, args.output_dir / "sample_a.wav")
    shutil.copyfile(source_b, args.output_dir / "sample_b.wav")
    pytorch_wav.unlink()
    shutil.copyfile(eval_dir / "input.wav", args.output_dir / "input.wav")
    shutil.copyfile(
        eval_dir / "input_codec_roundtrip.wav",
        args.output_dir / "input_codec_roundtrip.wav",
    )

    greedy_parity = token_agreement(
        fixture["dense_greedy_emitted"], greedy["emitted_audio"]
    )
    greedy_text_parity = best_sequence_alignment(
        [frame["text"] for frame in fixture["dense_greedy_frames"]],
        greedy["text_tokens"],
    )
    pytorch_tail_max_rms_dbfs = float(tail_max_rms_dbfs(pytorch_pcm))
    sample_tail = {
        "pytorch_tail_max_rms_dbfs": pytorch_tail_max_rms_dbfs,
        "pytorch_likely_truncated": bool(
            pytorch_tail_max_rms_dbfs > ACTIVE_AUDIO_DBFS
        ),
    }
    metrics = {
        "format_version": 1,
        "comparison": "dense safemlx versus upstream PyTorch PersonaPlex",
        "conditioning": {
            "source": str(eval_dir / "token_diagnostics.json"),
            "voice_frames": len(fixture["conditioning"]["voice_prompt"]),
            "text_tokens": len(fixture["conditioning"]["text_prompt"]),
            "input_frames": len(fixture["input"]),
            "shared_codec_tokens": True,
        },
        "sampling": {
            **fixture["sampling"],
            "rng_note": "The integer seed and sampling policy match; PyTorch and MLX use backend-native RNG algorithms, so sampled draws are not expected to be identical.",
        },
        "pytorch": {
            "device": args.device,
            "load_seconds": load_seconds,
            "sampled_latency": sampled["latency"],
            **sample_tail,
        },
        "greedy_token_parity": greedy_parity,
        "greedy_text_parity": greedy_text_parity,
        "codec_note": "The safemlx sample uses safemlx Mimi and the PyTorch sample uses upstream Mimi. Independent codec parity was previously measured at 0.999994 waveform correlation on the same token stream.",
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (args.output_dir / "answer_key.json").write_text(json.dumps(assignment, indent=2))
    (args.output_dir / "pytorch_tokens.json").write_text(
        json.dumps({"greedy": greedy, "sampled": sampled})
    )
    (args.output_dir / "listening_manifest.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "input": "input.wav",
                "sample_a": "sample_a.wav",
                "sample_b": "sample_b.wav",
                "scoring_scale": "1 = very poor, 5 = excellent",
                "rate_each_1_to_5": {
                    "semantic_response_quality": "5 = relevant, coherent, and useful",
                    "naturalness": "5 = convincingly human-like voice",
                    "pacing_and_pauses": "5 = natural conversational timing",
                    "freedom_from_robotic_affect": "5 = expressive rather than flat or robotic",
                    "freedom_from_silence_boundary_artifacts": "5 = clean transitions into and out of pauses",
                },
                "instructions": "Listen through both files before opening answer_key.json.",
            },
            indent=2,
        )
    )
    print(f"greedy_audio_token_agreement={greedy_parity['token_agreement']:.6f}")
    print(f"artifacts={args.output_dir}")


if __name__ == "__main__":
    main()
