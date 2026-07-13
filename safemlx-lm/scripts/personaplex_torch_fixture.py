#!/usr/bin/env python3
"""Export deterministic PersonaPlex parity tensors from NVIDIA's PyTorch code.

The default mode creates a tiny PersonaPlex-shaped checkpoint.  It keeps the
released 8+8 audio stream layout and PyTorch safetensors key structure, but uses
small hidden sizes so the Rust parity example can run quickly.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file


PERSONAPLEX_DELAYS = [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=314_159)
    parser.add_argument(
        "--personaplex-src",
        type=Path,
        default=Path("/private/tmp"),
        help="Directory containing personaplex_lm.py, personaplex_transformer.py, and personaplex_streaming.py",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="personaplex-ref-") as package_root:
        package_root = Path(package_root)
        build_reference_package(package_root, args.personaplex_src)
        sys.path.insert(0, str(package_root))
        from moshi.models.lm import LMGen, LMModel  # type: ignore

        torch.manual_seed(args.seed)
        model = tiny_model(LMModel)
        model.eval()
        args.model_dir.mkdir(parents=True, exist_ok=True)
        write_tiny_config(args.model_dir / "config.json")
        save_file(
            {key: value.detach().contiguous() for key, value in model.state_dict().items()},
            args.model_dir / "model.safetensors",
        )

        rng = np.random.default_rng(args.seed + 1)
        input_audio = torch.tensor(
            rng.integers(
                0,
                TINY_CONFIG["card"],
                size=(args.batch_size, 8, args.steps),
                dtype=np.int64,
            ),
            dtype=torch.long,
        )

        gen = LMGen(
            model,
            device=torch.device("cpu"),
            use_sampling=False,
            temp=0.0,
            temp_text=0.0,
            check=True,
            report_loss=False,
            return_logits=False,
        )
        sampled = []
        emitted = []
        emitted_steps = []
        with gen.streaming(args.batch_size):
            for step in range(args.steps):
                old_offset = gen._streaming_state.offset
                out = gen.step(input_tokens=input_audio[:, :, step : step + 1])
                if old_offset > 0:
                    target_position = old_offset % gen._streaming_state.cache.shape[2]
                    sampled.append(
                        gen._streaming_state.cache[:, :17, target_position].to(torch.int32)
                    )
                if out is not None:
                    emitted.append(out[:, 1:9, 0].to(torch.int32))
                    emitted_steps.append(step)

        tensors = {
            "input.audio": input_audio.to(torch.int32),
            "expected.sampled": torch.stack(sampled, dim=2),
            "expected.emitted_steps": torch.tensor(emitted_steps, dtype=torch.int32),
        }
        if emitted:
            tensors["expected.output_audio"] = torch.stack(emitted, dim=2)
        else:
            tensors["expected.output_audio"] = torch.zeros(
                (args.batch_size, 8, 0), dtype=torch.int32
            )

        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_file(tensors, args.output)
        print(
            f"wrote PersonaPlex tiny checkpoint to {args.model_dir} and {len(tensors)} fixture tensors to {args.output}"
        )


TINY_CONFIG = {
    "model_type": "personaplex",
    "version": "tiny-parity",
    "moshi_name": "model.safetensors",
    "dim": 32,
    "text_card": 64,
    "existing_text_padding_id": 3,
    "n_q": 16,
    "dep_q": 16,
    "generated_audio_codebooks": 8,
    "card": 32,
    "num_heads": 4,
    "num_layers": 2,
    "dim_feedforward": 132,
    "causal": True,
    "context": 16,
    "max_period": 10_000,
    "positional_embedding": "rope",
    "depformer_dim": 16,
    "depformer_dim_feedforward": 66,
    "depformer_num_heads": 4,
    "depformer_num_layers": 2,
    "depformer_context": 16,
    "depformer_max_period": 10_000,
    "depformer_pos_emb": "none",
    "delays": PERSONAPLEX_DELAYS,
    "conditioners": {},
    "cross_attention": False,
}


def write_tiny_config(path: Path) -> None:
    with path.open("w", encoding="utf-8") as config_file:
        json.dump(TINY_CONFIG, config_file, indent=2)
        config_file.write("\n")


def tiny_model(lm_model_cls):
    return lm_model_cls(
        delays=PERSONAPLEX_DELAYS,
        n_q=TINY_CONFIG["n_q"],
        dep_q=TINY_CONFIG["dep_q"],
        card=TINY_CONFIG["card"],
        text_card=TINY_CONFIG["text_card"],
        dim=TINY_CONFIG["dim"],
        num_heads=TINY_CONFIG["num_heads"],
        hidden_scale=TINY_CONFIG["dim_feedforward"] / TINY_CONFIG["dim"],
        norm="rms_norm_f32",
        norm_emb=False,
        bias_proj=False,
        depformer_dim=TINY_CONFIG["depformer_dim"],
        depformer_dim_feedforward=TINY_CONFIG["depformer_dim_feedforward"],
        depformer_multi_linear=True,
        depformer_weights_per_step=True,
        depformer_pos_emb=TINY_CONFIG["depformer_pos_emb"],
        existing_text_padding_id=TINY_CONFIG["existing_text_padding_id"],
        context=TINY_CONFIG["context"],
        num_layers=TINY_CONFIG["num_layers"],
        causal=TINY_CONFIG["causal"],
        positional_embedding=TINY_CONFIG["positional_embedding"],
        max_period=TINY_CONFIG["max_period"],
        gating="silu",
        depformer_num_heads=TINY_CONFIG["depformer_num_heads"],
        depformer_num_layers=TINY_CONFIG["depformer_num_layers"],
        depformer_max_period=TINY_CONFIG["depformer_max_period"],
        depformer_gating="silu",
    )


def build_reference_package(package_root: Path, src: Path) -> None:
    moshi = package_root / "moshi"
    models = moshi / "models"
    modules = moshi / "modules"
    utils = moshi / "utils"
    tqdm_pkg = package_root / "tqdm"
    for path in [models, modules, utils, tqdm_pkg]:
        path.mkdir(parents=True, exist_ok=True)
        (path / "__init__.py").write_text("", encoding="utf-8")
    (moshi / "__init__.py").write_text("", encoding="utf-8")

    shutil.copyfile(src / "personaplex_lm.py", models / "lm.py")
    shutil.copyfile(src / "personaplex_transformer.py", modules / "transformer.py")
    shutil.copyfile(src / "personaplex_streaming.py", modules / "streaming.py")
    shutil.copyfile(
        Path("/private/tmp/kyutai-moshi/moshi/moshi/modules/gating.py"),
        modules / "gating.py",
    )
    write_rope_module(modules / "rope.py")
    write_compile_module(utils / "compile.py")
    write_sampling_module(utils / "sampling.py")
    (package_root / "sphn.py").write_text(
        "def read(*args, **kwargs): raise RuntimeError('sphn.read is unavailable in parity fixture generation')\n"
        "def resample(*args, **kwargs): raise RuntimeError('sphn.resample is unavailable in parity fixture generation')\n",
        encoding="utf-8",
    )
    (tqdm_pkg / "auto.py").write_text(
        "def tqdm(iterable=None, *args, **kwargs):\n"
        "    return iterable if iterable is not None else []\n",
        encoding="utf-8",
    )


def write_compile_module(path: Path) -> None:
    path.write_text(
        "from contextlib import contextmanager\n\n"
        "def torch_compile_lazy(fn):\n"
        "    return fn\n\n"
        "@contextmanager\n"
        "def no_compile():\n"
        "    yield\n\n"
        "class CUDAGraphed:\n"
        "    def __init__(self, fn, disable=True):\n"
        "        self.fn = fn\n"
        "    def __call__(self, *args, **kwargs):\n"
        "        return self.fn(*args, **kwargs)\n",
        encoding="utf-8",
    )


def write_sampling_module(path: Path) -> None:
    path.write_text(
        "def sample_token(logits, use_sampling, temp, top_k):\n"
        "    return logits.argmax(dim=-1)\n",
        encoding="utf-8",
    )


def write_rope_module(path: Path) -> None:
    source = Path("/private/tmp/kyutai-moshi/moshi/moshi/modules/rope.py").read_text(
        encoding="utf-8"
    )
    source = source.replace(
        "def __init__(self, interleave: bool, max_period: float = 10000.0):",
        "def __init__(self, interleave: bool = True, max_period: float = 10000.0):",
    )
    path.write_text(source, encoding="utf-8")


if __name__ == "__main__":
    main()
