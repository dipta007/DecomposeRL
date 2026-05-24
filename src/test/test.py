"""Evaluate a LoRA checkpoint on a test dataset using vLLM.

Usage:
    PYTHONPATH=. uv run python src/test/test.py -d pubmedclaim -c outputs/2way_7b/checkpoint-100
"""

import argparse
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import jsonlines
import numpy as np
import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from src.baselines.utils import compute_classification_metrics, load_test_data
from src.train.decomposerl.prompts import USER_PROMPT_2WAY_TEMPLATE
from src.train.decomposerl.format_reward import get_format_reward
from src.train.decomposerl.rewards_llm import (
    extract_qa_pairs,
    verification_reward,
)


def extract_verification_label(generation: str) -> Optional[str]:
    try:
        label = generation.split("<verification>")[1].split("</verification>")[0].strip()
        if label.lower() in ("supported", "refuted"):
            return label.lower()
    except Exception:
        pass
    return None


def extract_verification_confidence(
    generation: str, logprobs_list: Optional[List] = None
) -> Optional[float]:
    if logprobs_list is None or "<verification>" not in generation:
        return None
    try:
        tokens = []
        for token_logprobs in logprobs_list:
            if token_logprobs is not None:
                for logprob_obj in token_logprobs.values():
                    tokens.append((logprob_obj.decoded_token, logprob_obj.logprob))

        text_so_far = ""
        tag_end = None
        for i, (tok, _) in enumerate(tokens):
            text_so_far += tok
            if "<verification>" in text_so_far:
                tag_end = i + 1
                break

        if tag_end is None:
            return None

        for i in range(tag_end, min(tag_end + 5, len(tokens))):
            tok, logprob = tokens[i]
            if tok.strip():
                return math.exp(logprob)
    except Exception:
        pass
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", "-c", required=True, help="Path to a single LoRA checkpoint directory")
    parser.add_argument("--dataset", "-d", required=True, help="Dataset name (e.g. pubmedclaim)")
    parser.add_argument("--max_tokens", type=int, default=12000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_model_len", type=int, default=32768)
    parser.add_argument("--force", "-f", action="store_true")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    dataset_name = args.dataset

    adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
    if not os.path.exists(adapter_config_path):
        raise FileNotFoundError(f"adapter_config.json not found in {checkpoint_path}")

    with open(adapter_config_path) as f:
        adapter_config = json.load(f)
    base_model = adapter_config["base_model_name_or_path"]
    lora_rank = adapter_config.get("r", 64)

    metrics_path = os.path.join(checkpoint_path, f"{dataset_name}_test_metrics.json")
    if not args.force and os.path.exists(metrics_path):
        print(f"[SKIP] Already evaluated: {metrics_path}")
        return

    test_data = load_test_data(dataset_name)

    print(f"Base model: {base_model}, LoRA rank: {lora_rank}")
    model = LLM(
        model=base_model,
        max_model_len=args.max_model_len,
        enable_lora=True,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.92,
        tensor_parallel_size=torch.cuda.device_count(),
        dtype="bfloat16",
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        max_num_seqs=256,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=0.95 if args.temperature > 0 else 1.0,
        max_tokens=args.max_tokens,
        seed=42,
        logprobs=1,
    )

    prompts = [
        [{"role": "user", "content": USER_PROMPT_2WAY_TEMPLATE.format(
            claim=s["claim"], evidence_doc=s["evidence"]
        )}]
        for s in test_data
    ]
    print(f"Running inference on {len(prompts)} samples...")
    outputs = model.chat(
        prompts,
        sampling_params=sampling_params,
        lora_request=LoRARequest("test_lora", 1, checkpoint_path),
    )

    results = []
    for i, (sample, output) in enumerate(zip(test_data, outputs)):
        gen = output.outputs[0].text
        logprobs = output.outputs[0].logprobs
        qa_pairs = extract_qa_pairs(gen)

        results.append({
            "id": sample.get("id"),
            "dataset": sample.get("src"),
            "claim": sample.get("claim"),
            "gt_label": sample.get("label"),
            "generation": gen,
            "pred_label": extract_verification_label(gen),
            "pred_confidence": extract_verification_confidence(gen, logprobs),
            "num_of_questions": len(qa_pairs),
            "questions": [p["question"] for p in qa_pairs],
            "answers": [p["answer"] for p in qa_pairs],
            "format_reward": get_format_reward(gen),
            "verification_reward": verification_reward(gen, sample["label"])[0],
        })

    gt = [r["gt_label"] for r in results]
    pred = [r["pred_label"] for r in results]
    metrics = compute_classification_metrics(gt, pred)
    metrics.update({
        "mean_num_of_questions": float(np.mean([r["num_of_questions"] for r in results])),
        "mean_format_reward": float(np.mean([r["format_reward"] for r in results])),
        "mean_verification_reward": float(np.mean([r["verification_reward"] for r in results])),
        "total_samples": len(results),
    })
    confidences = [r["pred_confidence"] for r in results if r["pred_confidence"] is not None]
    if confidences:
        metrics["mean_pred_confidence"] = float(np.mean(confidences))

    os.makedirs(checkpoint_path, exist_ok=True)
    jsonl_path = os.path.join(checkpoint_path, f"{dataset_name}_test_results.jsonl")
    with jsonlines.open(jsonl_path, "w") as w:
        for r in results:
            w.write(r)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Results: {jsonl_path}")
    print(f"Metrics: {metrics_path}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
