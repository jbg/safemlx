#!/usr/bin/env python3
"""Generate a deterministic tiny fixture with DeepSeek's released HF code.

Usage:
  python deepseek_v3_transformers_fixture.py \
      --reference-source /path/to/DeepSeek-R1-0528 \
      --output /tmp/deepseek-v3-tiny-reference

`reference-source` must contain the official `configuration_deepseek.py` and
`modeling_deepseek.py`. The script intentionally imports those files directly
instead of comparing safemlx-lm against another reimplementation.
"""

import argparse
import importlib.util
import json
import pathlib
import sys

import torch


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-source", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    configuration = load_module(
        "configuration_deepseek",
        args.reference_source / "configuration_deepseek.py",
    )
    modeling = load_module(
        "modeling_deepseek", args.reference_source / "modeling_deepseek.py"
    )
    config = configuration.DeepseekV3Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        moe_intermediate_size=4,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
        q_lora_rank=4,
        kv_lora_rank=4,
        qk_nope_head_dim=2,
        qk_rope_head_dim=2,
        v_head_dim=2,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=2,
        topk_group=1,
        topk_method="noaux_tc",
        scoring_func="sigmoid",
        norm_topk_prob=True,
        routed_scaling_factor=1.5,
        num_nextn_predict_layers=0,
        attention_dropout=0.0,
        attention_bias=False,
        tie_word_embeddings=False,
        torch_dtype=torch.float32,
        use_cache=False,
    )
    model = modeling.DeepseekV3ForCausalLM(config).eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("layernorm.weight") or name == "model.norm.weight":
                parameter.fill_(1.0)
            else:
                values = torch.arange(parameter.numel(), dtype=torch.float32)
                values = ((values % 17) - 8) * 0.001
                parameter.copy_(values.reshape_as(parameter))

        input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        logits = model(input_ids=input_ids, use_cache=False).logits

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output, safe_serialization=True)
    (args.output / "reference.json").write_text(
        json.dumps(
            {
                "input_ids": input_ids.tolist(),
                "logits": logits.tolist(),
                "source": str(args.reference_source),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
