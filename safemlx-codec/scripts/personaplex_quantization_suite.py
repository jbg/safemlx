#!/usr/bin/env python3
"""Run and summarize a blinded multi-case PersonaPlex dense/Q4 evaluation."""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


FRAME_BYTES = 1_920 * 4
CRITERIA = (
    "naturalness",
    "intelligibility",
    "voice_persona_consistency",
    "semantic_response_quality",
    "turn_timing",
    "freedom_from_silence_boundary_artifacts",
)
QUALITY_SECTIONS = ("text", "audio_generated", "audio_input_conditioned")
QUALITY_FIELDS = (
    "mean_kl_nats",
    "mean_target_nll_delta_nats",
    "mean_centered_logit_rmse",
    "top1_agreement",
    "mean_top5_overlap",
)
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
DEFAULT_TEXT_PROMPT = "You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run every manifest trial")
    run.add_argument("manifest", type=Path)
    run.add_argument("output_dir", type=Path)
    run.add_argument("--binary", type=Path)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--dry-run", action="store_true")

    summarize = subparsers.add_parser(
        "summarize", help="unblind and summarize a completed ratings file"
    )
    summarize.add_argument("suite_dir", type=Path)
    summarize.add_argument("ratings", type=Path)
    summarize.add_argument("--output", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def resolve_path(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def input_stats(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    samples = array.array("f")
    samples.frombytes(data)
    if sys.byteorder != "little":
        samples.byteswap()
    mean_square = sum(float(sample) ** 2 for sample in samples) / max(len(samples), 1)
    return {
        "sha256": hashlib.sha256(data).hexdigest(),
        "rms_dbfs": 20.0 * math.log10(max(mean_square**0.5, 1e-12)),
        "peak": max((abs(sample) for sample in samples), default=0.0),
    }


def expanded_trials(manifest: dict[str, Any], base: Path) -> list[dict[str, Any]]:
    default_seeds = manifest.get("sampling_seeds", [20260713])
    if not default_seeds:
        raise ValueError("sampling_seeds must not be empty")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("manifest cases must be a non-empty array")
    trials = []
    seen = set()
    seen_inputs = {}
    for case in cases:
        case_id = case.get("id")
        if not isinstance(case_id, str) or not SAFE_ID.fullmatch(case_id):
            raise ValueError(f"unsafe or missing case id: {case_id!r}")
        input_path = resolve_path(base, case["input"])
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        stats = input_stats(input_path)
        if stats["rms_dbfs"] < -60.0 and not case.get("allow_silent_input", False):
            raise ValueError(
                f"case {case_id} input appears silent ({stats['rms_dbfs']:.1f} dBFS)"
            )
        duplicate = seen_inputs.get(stats["sha256"])
        if duplicate is not None and not case.get("allow_duplicate_input", False):
            raise ValueError(
                f"cases {duplicate} and {case_id} have byte-identical inputs"
            )
        seen_inputs[stats["sha256"]] = case_id
        seeds = case.get("sampling_seeds", default_seeds)
        if not seeds:
            raise ValueError(f"case {case_id} has no sampling seeds")
        frames = case.get("frames", input_path.stat().st_size // FRAME_BYTES)
        if not isinstance(frames, int) or frames < 4:
            raise ValueError(f"case {case_id} must contain at least four frames")
        if frames * FRAME_BYTES > input_path.stat().st_size:
            raise ValueError(f"case {case_id} requests more frames than its input contains")
        for seed in seeds:
            seed = int(seed)
            trial_id = case_id if len(seeds) == 1 else f"{case_id}__seed_{seed}"
            if trial_id in seen:
                raise ValueError(f"duplicate trial id: {trial_id}")
            seen.add(trial_id)
            trials.append(
                {
                    "id": trial_id,
                    "case_id": case_id,
                    "category": case.get("category", "unspecified"),
                    "notes": case.get("notes"),
                    "input": input_path,
                    "frames": frames,
                    "text_prompt": case.get(
                        "text_prompt", manifest.get("text_prompt", DEFAULT_TEXT_PROMPT)
                    ),
                    "sampling_seed": seed,
                    "expected_active_tail": bool(case.get("expected_active_tail", False)),
                    "input_rms_dbfs": stats["rms_dbfs"],
                    "input_sha256": stats["sha256"],
                }
            )
    return trials


def common_paths(manifest: dict[str, Any], base: Path) -> dict[str, Path]:
    names = ("dense_model", "quantized_model", "mimi", "text_tokenizer", "voice_prompt")
    paths = {name: resolve_path(base, manifest[name]) for name in names}
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    return paths


def eval_command(
    binary: Path,
    common: dict[str, Path],
    trial: dict[str, Any],
    output: Path,
) -> list[str]:
    return [
        str(binary),
        str(common["dense_model"]),
        str(common["quantized_model"]),
        str(common["mimi"]),
        str(common["text_tokenizer"]),
        str(common["voice_prompt"]),
        str(trial["input"]),
        str(output),
        str(trial["frames"]),
        trial["text_prompt"],
        str(trial["sampling_seed"]),
    ]


def weighted_mean(records: list[tuple[float, int]]) -> float:
    weight = sum(item[1] for item in records)
    return sum(value * count for value, count in records) / max(weight, 1)


def aggregate_suite(
    suite_dir: Path, trials: list[dict[str, Any]]
) -> dict[str, Any]:
    case_metrics = []
    total_frames = 0
    dense_means = []
    quantized_means = []
    dense_p95 = []
    quantized_p95 = []
    dense_deadline_misses = 0
    quantized_deadline_misses = 0
    dense_loads = []
    quantized_loads = []
    unexpected_input_truncated = 0
    expected_active_tail_trials = 0
    output_truncated = 0
    quality: dict[str, dict[str, list[tuple[float, int]]]] = {
        section: {field: [] for field in QUALITY_FIELDS}
        for section in QUALITY_SECTIONS
    }

    for trial in trials:
        metrics_path = suite_dir / "cases" / trial["id"] / "metrics.json"
        metrics = load_json(metrics_path)
        performance = metrics["performance"]
        dense = performance["dense"]["model"]
        quantized = performance["quantized"]["model"]
        frames = int(dense["frames"])
        total_frames += frames
        dense_means.append((float(dense["mean_ms"]), frames))
        quantized_means.append((float(quantized["mean_ms"]), frames))
        dense_p95.append(float(dense["p95_ms"]))
        quantized_p95.append(float(quantized["p95_ms"]))
        dense_deadline_misses += int(dense["deadline_misses"])
        quantized_deadline_misses += int(quantized["deadline_misses"])
        dense_loads.append(float(performance["dense"]["load_seconds"]))
        quantized_loads.append(float(performance["quantized"]["load_seconds"]))
        input_active_tail = bool(metrics["input"]["likely_truncated"])
        expected_active_tail_trials += int(
            input_active_tail and trial["expected_active_tail"]
        )
        unexpected_input_truncated += int(
            input_active_tail and not trial["expected_active_tail"]
        )
        listening = metrics["listening_test"]
        output_truncated += int(bool(listening["sample_a_likely_truncated"]))
        output_truncated += int(bool(listening["sample_b_likely_truncated"]))
        for section in QUALITY_SECTIONS:
            values = metrics["teacher_forced_quality"][section]
            distributions = int(values["distributions"])
            for field in QUALITY_FIELDS:
                quality[section][field].append((float(values[field]), distributions))
        case_metrics.append(
            {
                "id": trial["id"],
                "case_id": trial["case_id"],
                "category": trial["category"],
                "sampling_seed": trial["sampling_seed"],
                "frames": frames,
                "dense_mean_ms": dense["mean_ms"],
                "quantized_mean_ms": quantized["mean_ms"],
                "speedup_pct": 100.0
                * (float(dense["mean_ms"]) - float(quantized["mean_ms"]))
                / float(dense["mean_ms"]),
                "input_likely_truncated": metrics["input"]["likely_truncated"],
                "expected_active_tail": trial["expected_active_tail"],
                "sample_a_likely_truncated": listening["sample_a_likely_truncated"],
                "sample_b_likely_truncated": listening["sample_b_likely_truncated"],
            }
        )

    dense_mean = weighted_mean(dense_means)
    quantized_mean = weighted_mean(quantized_means)
    quality_summary = {
        section: {
            field: weighted_mean(records) for field, records in fields.items()
        }
        for section, fields in quality.items()
    }
    return {
        "format_version": 1,
        "trial_count": len(trials),
        "case_count": len({trial["case_id"] for trial in trials}),
        "total_frames": total_frames,
        "total_audio_seconds": total_frames / 12.5,
        "performance": {
            "dense_weighted_mean_ms": dense_mean,
            "quantized_weighted_mean_ms": quantized_mean,
            "mean_step_reduction_pct": 100.0
            * (dense_mean - quantized_mean)
            / dense_mean,
            "dense_realtime_capacity": 80.0 / dense_mean,
            "quantized_realtime_capacity": 80.0 / quantized_mean,
            "realtime_capacity_ratio": dense_mean / quantized_mean,
            "dense_case_median_p95_ms": statistics.median(dense_p95),
            "quantized_case_median_p95_ms": statistics.median(quantized_p95),
            "dense_worst_case_p95_ms": max(dense_p95),
            "quantized_worst_case_p95_ms": max(quantized_p95),
            "dense_deadline_misses": dense_deadline_misses,
            "quantized_deadline_misses": quantized_deadline_misses,
            "dense_deadline_miss_rate": dense_deadline_misses / total_frames,
            "quantized_deadline_miss_rate": quantized_deadline_misses / total_frames,
            "dense_median_load_seconds": statistics.median(dense_loads),
            "quantized_median_load_seconds": statistics.median(quantized_loads),
        },
        "teacher_forced_quality": quality_summary,
        "truncation": {
            "unexpected_input_trials": unexpected_input_truncated,
            "expected_active_tail_trials": expected_active_tail_trials,
            "output_samples": output_truncated,
        },
        "cases": case_metrics,
    }


def blind_manifest(trials: list[dict[str, Any]], suite_dir: Path) -> dict[str, Any]:
    entries = []
    for trial in trials:
        metrics = load_json(suite_dir / "cases" / trial["id"] / "metrics.json")
        listening = metrics["listening_test"]
        root = f"cases/{trial['id']}"
        entries.append(
            {
                "id": trial["id"],
                "category": trial["category"],
                "notes": trial["notes"],
                "input": f"{root}/input.wav",
                "codec_roundtrip": f"{root}/input_codec_roundtrip.wav",
                "sample_a": f"{root}/sample_a.wav",
                "sample_b": f"{root}/sample_b.wav",
                "input_warning": listening["input_warning"],
                "expected_active_tail": trial["expected_active_tail"],
                "sample_a_likely_truncated": listening["sample_a_likely_truncated"],
                "sample_b_likely_truncated": listening["sample_b_likely_truncated"],
            }
        )
    return {
        "format_version": 1,
        "scoring_scale": "1 = very poor, 5 = excellent",
        "criteria": list(CRITERIA),
        "forced_choice": ["a_better", "same", "b_better"],
        "instructions": "Rate every trial before inspecting any case answer_key.json. Different valid wording is not itself a defect.",
        "trials": entries,
    }


def ratings_template(trials: list[dict[str, Any]]) -> dict[str, Any]:
    empty_scores = lambda: {criterion: None for criterion in CRITERIA}
    return {
        "format_version": 1,
        "instructions": "Fill every score with an integer from 1 to 5 and forced_choice with a_better, same, or b_better.",
        "ratings": [
            {
                "id": trial["id"],
                "scores": {"a": empty_scores(), "b": empty_scores()},
                "forced_choice": None,
                "notes": "",
            }
            for trial in trials
        ],
    }


def listening_markdown(trials: list[dict[str, Any]]) -> str:
    lines = [
        "# PersonaPlex dense versus Q4 blind listening",
        "",
        "Listen to the input and codec roundtrip first, then A and B. Do not open any `answer_key.json` until all ratings are complete.",
        "",
        "Score naturalness, intelligibility, voice/persona consistency, semantic response quality, turn timing, and silence-boundary cleanliness from 1 (very poor) to 5 (excellent). Record scores in `human_ratings.json`.",
        "",
    ]
    for index, trial in enumerate(trials, start=1):
        root = f"cases/{trial['id']}"
        lines.extend(
            [
                f"## {index}. {trial['id']}",
                "",
                f"Category: {trial['category']}",
                "",
                f"[Input]({root}/input.wav) · [Codec roundtrip]({root}/input_codec_roundtrip.wav) · [Sample A]({root}/sample_a.wav) · [Sample B]({root}/sample_b.wav)",
                "",
            ]
        )
        if trial["notes"]:
            lines.extend([f"Note: {trial['notes']}", ""])
    return "\n".join(lines)


def run_suite(args: argparse.Namespace) -> None:
    manifest_path = args.manifest.resolve()
    manifest = load_json(manifest_path)
    trials = expanded_trials(manifest, manifest_path.parent)
    common = common_paths(manifest, manifest_path.parent)
    repo_root = Path(__file__).resolve().parents[2]
    binary = (args.binary or repo_root / "target/release/examples/personaplex_quantization_eval").resolve()
    if not binary.is_file() and not args.dry_run:
        subprocess.run(
            [
                "cargo",
                "build",
                "--release",
                "-p",
                "safemlx-codec",
                "--example",
                "personaplex_quantization_eval",
            ],
            cwd=repo_root,
            check=True,
        )
    commands = [
        eval_command(binary, common, trial, args.output_dir / "cases" / trial["id"])
        for trial in trials
    ]
    if args.dry_run:
        for command in commands:
            print(json.dumps(command))
        return

    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    (args.output_dir / "cases").mkdir(parents=True, exist_ok=args.resume)
    write_json(args.output_dir / "suite_manifest.json", manifest)
    for index, (trial, command) in enumerate(zip(trials, commands), start=1):
        output = args.output_dir / "cases" / trial["id"]
        if args.resume and (output / "metrics.json").is_file():
            print(f"[{index}/{len(trials)}] reuse {trial['id']}", flush=True)
            continue
        print(f"[{index}/{len(trials)}] run {trial['id']}", flush=True)
        subprocess.run(command, cwd=repo_root, check=True)

    write_json(args.output_dir / "summary.json", aggregate_suite(args.output_dir, trials))
    write_json(
        args.output_dir / "listening_manifest.json",
        blind_manifest(trials, args.output_dir),
    )
    write_json(args.output_dir / "human_ratings.json", ratings_template(trials))
    (args.output_dir / "LISTENING.md").write_text(listening_markdown(trials))
    print(f"suite={args.output_dir}")


def summarize_ratings(args: argparse.Namespace) -> None:
    suite_dir = args.suite_dir.resolve()
    ratings = load_json(args.ratings)["ratings"]
    by_criterion = {
        criterion: {"dense": [], "quantized": []} for criterion in CRITERIA
    }
    preferences = {"dense_better": 0, "same": 0, "quantized_better": 0}
    for rating in ratings:
        trial_id = rating["id"]
        key = load_json(suite_dir / "cases" / trial_id / "answer_key.json")
        labels = {"a": key["sample_a"], "b": key["sample_b"]}
        for sample in ("a", "b"):
            for criterion in CRITERIA:
                score = rating["scores"][sample][criterion]
                if not isinstance(score, int) or not 1 <= score <= 5:
                    raise ValueError(f"{trial_id} {sample} {criterion} must be 1..5")
                by_criterion[criterion][labels[sample]].append(score)
        choice = rating["forced_choice"]
        if choice == "same":
            preferences["same"] += 1
        elif choice in ("a_better", "b_better"):
            sample = choice[0]
            preferences[f"{labels[sample]}_better"] += 1
        else:
            raise ValueError(f"{trial_id} has invalid forced_choice: {choice!r}")

    criteria = {}
    for criterion, values in by_criterion.items():
        dense = statistics.mean(values["dense"])
        quantized = statistics.mean(values["quantized"])
        criteria[criterion] = {
            "dense_mean": dense,
            "quantized_mean": quantized,
            "quantized_minus_dense": quantized - dense,
            "ratings_per_model": len(values["dense"]),
        }
    output = {
        "format_version": 1,
        "trial_count": len(ratings),
        "preference_counts": preferences,
        "criteria": criteria,
        "automated_summary": "summary.json",
    }
    destination = args.output or suite_dir / "human_summary.json"
    write_json(destination, output)
    print(f"human_summary={destination}")


def main() -> None:
    args = parse_args()
    if args.command == "run":
        run_suite(args)
    else:
        summarize_ratings(args)


if __name__ == "__main__":
    try:
        main()
    except (KeyError, OSError, TypeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
