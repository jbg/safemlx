import argparse
import json
import struct
import tempfile
import unittest
from pathlib import Path

import personaplex_quantization_suite as suite


def write_json(path, value):
    path.write_text(json.dumps(value))


def quality(distributions, value):
    return {
        "distributions": distributions,
        "mean_kl_nats": value,
        "mean_target_nll_delta_nats": value,
        "mean_centered_logit_rmse": value,
        "top1_agreement": value,
        "mean_top5_overlap": value,
    }


class QuantizationSuiteTests(unittest.TestCase):
    def test_rejects_silent_and_duplicate_case_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            silent = root / "silent.f32le"
            silent.write_bytes(bytes(suite.FRAME_BYTES * 4))
            with self.assertRaisesRegex(ValueError, "appears silent"):
                suite.expanded_trials(
                    {"cases": [{"id": "silent", "input": silent.name}]}, root
                )

            active = struct.pack("<f", 0.1) * (1_920 * 4)
            (root / "one.f32le").write_bytes(active)
            (root / "two.f32le").write_bytes(active)
            with self.assertRaisesRegex(ValueError, "byte-identical"):
                suite.expanded_trials(
                    {
                        "cases": [
                            {"id": "one", "input": "one.f32le"},
                            {"id": "two", "input": "two.f32le"},
                        ]
                    },
                    root,
                )

    def test_expands_case_seeds_and_aggregates_by_frame_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "input.f32le"
            input_path.write_bytes(bytes(suite.FRAME_BYTES * 8))
            manifest = {
                "sampling_seeds": [10, 20],
                "cases": [
                    {
                        "id": "question",
                        "input": "input.f32le",
                        "allow_silent_input": True,
                    }
                ],
            }
            trials = suite.expanded_trials(manifest, root)
            self.assertEqual(
                [trial["id"] for trial in trials],
                ["question__seed_10", "question__seed_20"],
            )

            for index, trial in enumerate(trials, start=1):
                case_dir = root / "cases" / trial["id"]
                case_dir.mkdir(parents=True)
                frames = index * 10
                metrics = {
                    "input": {"likely_truncated": False},
                    "performance": {
                        "dense": {
                            "load_seconds": 2.0,
                            "model": {
                                "frames": frames,
                                "mean_ms": 40.0,
                                "p95_ms": 42.0,
                                "deadline_misses": 0,
                            },
                        },
                        "quantized": {
                            "load_seconds": 0.6,
                            "model": {
                                "frames": frames,
                                "mean_ms": 16.0,
                                "p95_ms": 18.0,
                                "deadline_misses": 0,
                            },
                        },
                    },
                    "teacher_forced_quality": {
                        section: quality(frames, float(index))
                        for section in suite.QUALITY_SECTIONS
                    },
                    "listening_test": {
                        "input_warning": None,
                        "sample_a_likely_truncated": False,
                        "sample_b_likely_truncated": False,
                    },
                }
                write_json(case_dir / "metrics.json", metrics)

            result = suite.aggregate_suite(root, trials)
            self.assertEqual(result["trial_count"], 2)
            self.assertEqual(result["total_frames"], 30)
            self.assertAlmostEqual(
                result["performance"]["mean_step_reduction_pct"], 60.0
            )
            self.assertAlmostEqual(
                result["teacher_forced_quality"]["text"]["mean_kl_nats"],
                5.0 / 3.0,
            )

    def test_human_summary_maps_blind_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case_dir = root / "cases" / "trial"
            case_dir.mkdir(parents=True)
            write_json(
                case_dir / "answer_key.json",
                {"sample_a": "quantized", "sample_b": "dense"},
            )
            scores = {criterion: 4 for criterion in suite.CRITERIA}
            ratings_path = root / "ratings.json"
            write_json(
                ratings_path,
                {
                    "ratings": [
                        {
                            "id": "trial",
                            "scores": {"a": scores, "b": scores},
                            "forced_choice": "a_better",
                        }
                    ]
                },
            )
            output = root / "human.json"
            suite.summarize_ratings(
                argparse.Namespace(
                    suite_dir=root, ratings=ratings_path, output=output
                )
            )
            summary = json.loads(output.read_text())
            self.assertEqual(summary["preference_counts"]["quantized_better"], 1)
            self.assertEqual(summary["criteria"]["naturalness"]["dense_mean"], 4)


if __name__ == "__main__":
    unittest.main()
