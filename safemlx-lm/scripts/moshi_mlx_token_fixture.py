#!/usr/bin/env python3
"""Export deterministic Moshi token-only parity tensors from moshi_mlx."""

import argparse
import importlib.metadata
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from moshi_mlx import models, utils


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=299_792_458)
    parser.add_argument(
        "--require-mlx-version",
        help="Fail unless the Python MLX package has this exact version",
    )
    parser.add_argument(
        "--create-tiny",
        action="store_true",
        help="Create a deterministic miniature BF16 Moshi checkpoint in model_dir",
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        help="Optional safetensors file containing input.text, input.audio, and input.depth",
    )
    args = parser.parse_args()

    mlx_version = importlib.metadata.version("mlx")
    if (
        args.require_mlx_version is not None
        and mlx_version != args.require_mlx_version
    ):
        parser.error(
            f"MLX {args.require_mlx_version} is required, but {mlx_version} is installed"
        )

    if args.create_tiny:
        create_tiny_checkpoint(args.model_dir, args.seed)

    config_path = args.model_dir / "config.json"
    if config_path.exists():
        with config_path.open() as config_file:
            raw_config = json.load(config_file)
        config = models.LmConfig.from_config_dict(raw_config)
        weights_name = raw_config.get("moshi_name", "model.safetensors")
    else:
        config = models.config_v0_1()
        weights_name = "model.safetensors"
    model = models.Lm(config)
    model.set_dtype(mx.bfloat16)

    if ".q4." in weights_name or ".q8." in weights_name:
        bits = 4 if ".q4." in weights_name else 8
        group_size = 32 if bits == 4 else 64
        nn.quantize(model, bits=bits, group_size=group_size)
    model.load_weights(str(args.model_dir / weights_name), strict=True)

    if args.inputs is not None:
        inputs = mx.load(str(args.inputs))
        tensors = {
            key: inputs[key] for key in ("input.text", "input.audio", "input.depth")
        }
    else:
        rng = np.random.default_rng(args.seed)
        text = rng.integers(
            0,
            config.text_out_vocab_size,
            size=(args.steps, args.batch_size, 1),
            dtype=np.int32,
        )
        text[0, :, :] = config.text_out_vocab_size
        audio = rng.integers(
            0,
            config.audio_vocab_size - 1,
            size=(args.steps, args.batch_size, config.audio_codebooks),
            dtype=np.int32,
        )
        audio[0, :, :] = config.audio_padding_token
        depth = np.empty(
            (args.steps, args.batch_size, config.generated_codebooks), dtype=np.int32
        )
        depth[:, :, 0] = rng.integers(
            0,
            config.text_out_vocab_size,
            size=(args.steps, args.batch_size),
            dtype=np.int32,
        )
        if config.generated_codebooks > 1:
            depth[:, :, 1:] = rng.integers(
                0,
                config.audio_vocab_size - 1,
                size=(args.steps, args.batch_size, config.generated_codebooks - 1),
                dtype=np.int32,
            )
        tensors = {
            "input.text": mx.array(text),
            "input.audio": mx.array(audio),
            "input.depth": mx.array(depth),
        }

    for step in range(tensors["input.text"].shape[0]):
        text_step = tensors["input.text"][step]
        audio_step = tensors["input.audio"][step]
        x = model.text_emb(text_step)
        for codebook, embedding in enumerate(model.audio_embs):
            x = x + embedding(audio_step[:, codebook : codebook + 1])
        tensors[f"expected.{step}.temporal_input"] = x
        temporal = x
        for layer_index, (layer, cache) in enumerate(
            zip(model.transformer.layers, model.transformer_cache)
        ):
            norm1 = layer.norm1(temporal)
            batch, sequence, dim = norm1.shape
            attention_in_proj = layer.self_attn.in_proj(norm1)
            qkv = attention_in_proj.reshape(
                batch,
                sequence,
                3,
                layer.self_attn.cfg.num_heads,
                layer.self_attn.cfg.head_dim,
            )
            q = qkv[:, :, 0].transpose(0, 2, 1, 3)
            k = qkv[:, :, 1].transpose(0, 2, 1, 3)
            v = qkv[:, :, 2].transpose(0, 2, 1, 3)
            if layer.self_attn.rope is not None:
                q = layer.self_attn.rope(q, offset=cache.self_attn.offset)
                k = layer.self_attn.rope(k, offset=cache.self_attn.offset)
            tensors[
                f"expected.{step}.temporal_layer.{layer_index}.attention_queries"
            ] = q
            tensors[
                f"expected.{step}.temporal_layer.{layer_index}.attention_keys"
            ] = k
            k, v = cache.self_attn.update_and_fetch(k, v)
            key_length = k.shape[2]
            target_length = sequence + min(
                layer.self_attn.cfg.context, key_length - sequence
            )
            if target_length < key_length:
                k = k[:, :, key_length - target_length :]
                v = v[:, :, key_length - target_length :]
            attention_attended = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=layer.self_attn.scale
            )
            attention_attended = attention_attended.transpose(0, 2, 1, 3).reshape(
                batch, sequence, dim
            )
            attention = layer.self_attn.out_proj(attention_attended)
            post_attention = temporal + attention
            norm2 = layer.norm2(post_attention)
            mlp = layer.gating(norm2)
            temporal = post_attention + mlp
            tensors[f"expected.{step}.temporal_layer.{layer_index}.norm1"] = norm1
            tensors[
                f"expected.{step}.temporal_layer.{layer_index}.attention_in_proj"
            ] = attention_in_proj
            tensors[
                f"expected.{step}.temporal_layer.{layer_index}.attention_values"
            ] = v
            tensors[
                f"expected.{step}.temporal_layer.{layer_index}.attention_attended"
            ] = attention_attended
            tensors[f"expected.{step}.temporal_layer.{layer_index}.attention"] = (
                attention
            )
            tensors[
                f"expected.{step}.temporal_layer.{layer_index}.post_attention"
            ] = post_attention
            tensors[f"expected.{step}.temporal_layer.{layer_index}.norm2"] = norm2
            tensors[f"expected.{step}.temporal_layer.{layer_index}.mlp"] = mlp
            tensors[f"expected.{step}.temporal_layer.{layer_index}"] = temporal
        temporal = model.out_norm(temporal)
        text_logits = model.text_linear(temporal)
        tensors[f"expected.{step}.temporal"] = temporal
        tensors[f"expected.{step}.text_logits"] = text_logits

        for cache in model.depformer_cache:
            cache.reset()
        for slice_index, dep_slice in enumerate(model.depformer.slices):
            token = tensors["input.depth"][step, :, slice_index : slice_index + 1]
            dep_x = dep_slice.linear_in(temporal) + dep_slice.emb(token)
            dep_x = dep_slice.transformer(dep_x, cache=model.depformer_cache)
            tensors[f"expected.{step}.audio_logits.{slice_index}"] = (
                dep_slice.linear_out(dep_x)
            )

    input_codebooks = config.audio_codebooks - config.generated_codebooks
    rng = np.random.default_rng(args.seed + 1)
    generation_input = rng.integers(
        0,
        config.audio_vocab_size - 1,
        size=(args.batch_size, input_codebooks, args.steps),
        dtype=np.int32,
    )
    tensors["generation.input_audio"] = mx.array(generation_input)

    generation_model = models.Lm(config)
    generation_model.set_dtype(mx.bfloat16)
    if ".q4." in weights_name or ".q8." in weights_name:
        bits = 4 if ".q4." in weights_name else 8
        group_size = 32 if bits == 4 else 64
        nn.quantize(generation_model, bits=bits, group_size=group_size)
    generation_model.load_weights(str(args.model_dir / weights_name), strict=True)
    generator = models.LmGen(
        model=generation_model,
        max_steps=args.steps,
        text_sampler=utils.Sampler(temp=0),
        audio_sampler=utils.Sampler(temp=0),
        check=False,
    )
    generated_text = []
    generated_audio = []
    for step in range(args.steps):
        step_result = generator.step(tensors["generation.input_audio"][:, :, step])
        text_token = step_result[0] if isinstance(step_result, tuple) else step_result
        generated_text.append(text_token.squeeze(-1))
        audio_tokens = generator.last_audio_tokens()
        if audio_tokens is not None:
            generated_audio.append(audio_tokens)
    tensors["generation.expected_text"] = mx.stack(generated_text, axis=1)
    if generated_audio:
        tensors["generation.expected_audio"] = mx.stack(generated_audio, axis=2)
    else:
        tensors["generation.expected_audio"] = mx.zeros(
            (args.batch_size, config.generated_codebooks, 0), dtype=mx.int32
        )

    mx.eval(*tensors.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(args.output), tensors)
    print(f"wrote {len(tensors)} tensors to {args.output} with MLX {mlx_version}")


def create_tiny_checkpoint(model_dir: Path, seed: int) -> None:
    raw_config = {
        "model_type": "moshi",
        "moshi_name": "model.safetensors",
        "dim": 32,
        "text_card": 64,
        "n_q": 4,
        "dep_q": 2,
        "card": 16,
        "num_heads": 4,
        "num_layers": 2,
        "causal": True,
        "layer_scale": None,
        "context": 3,
        "max_period": 10_000,
        "positional_embedding": "rope",
        "depformer_dim": 16,
        "depformer_dim_feedforward": 64,
        "depformer_num_heads": 4,
        "depformer_num_layers": 2,
        "depformer_context": 2,
        "depformer_max_period": 10_000,
        "depformer_pos_emb": "none",
        "delays": [0, 0, 1, 0, 1],
        "conditioners": {},
        "cross_attention": False,
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    with (model_dir / "config.json").open("w") as config_file:
        json.dump(raw_config, config_file, indent=2)
        config_file.write("\n")

    mx.random.seed(seed)
    config = models.LmConfig.from_config_dict(raw_config)
    model = models.Lm(config)
    model.set_dtype(mx.bfloat16)
    model.save_weights(str(model_dir / raw_config["moshi_name"]))
    print(f"wrote tiny checkpoint to {model_dir}")


if __name__ == "__main__":
    main()
