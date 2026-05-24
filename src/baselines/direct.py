"""Simple and Chain-of-Thought baselines for claim verification.

Usage:
    PYTHONPATH=. uv run python src/baselines/direct.py -d pubmedclaim --mode simple -o outputs/baseline_7b
    PYTHONPATH=. uv run python src/baselines/direct.py -d pubmedclaim --mode cot -o outputs/baseline_7b
"""

import argparse
import json
import os
from typing import Optional

import jsonlines

from src.baselines.utils import compute_classification_metrics, load_test_data


SIMPLE_TEMPLATE = """\
You are a fact-checking assistant. Given the evidence and claim below, determine if the claim is Supported or Refuted by the evidence.

<evidence>
{evidence}
</evidence>

<claim>
{claim}
</claim>

Based on the evidence provided, is the claim Supported or Refuted?

Answer with exactly one word: Supported or Refuted"""

COT_TEMPLATE = """\
You are a fact-checking assistant. Given the evidence and claim below, determine if the claim is Supported or Refuted by the evidence.

<evidence>
{evidence}
</evidence>

<claim>
{claim}
</claim>

First, analyze the evidence step by step and reason about whether it supports or refutes the claim. Then provide your final verdict.

Format your response as:
<reasoning>
[Your step-by-step analysis here]
</reasoning>

<verdict>
[Supported or Refuted]
</verdict>"""


def extract_label(generation: str, mode: str) -> Optional[str]:
    if mode == "cot":
        try:
            label = generation.split("<verdict>")[1].split("</verdict>")[0].strip()
            if label.lower() in ("supported", "refuted"):
                return label.lower()
        except Exception:
            pass

    g = generation.strip().lower()
    if g in ("supported", "refuted"):
        return g
    if "supported" in g and "refuted" not in g:
        return "supported"
    if "refuted" in g and "supported" not in g:
        return "refuted"

    words = g.split()
    if words:
        if words[0] in ("supported", "refuted"):
            return words[0]
        if words[-1] in ("supported", "refuted"):
            return words[-1]
    return None


def main():
    parser = argparse.ArgumentParser(description="Simple/CoT baseline evaluation")
    parser.add_argument("--backend", choices=["vllm", "api"], default="vllm")
    parser.add_argument("--model", "-m", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max_model_len", type=int, default=32768)
    parser.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    parser.add_argument("--api_model", default=None)
    parser.add_argument("--api_base_url", default=None)
    parser.add_argument("--api_key_env", default=None)
    parser.add_argument("--max_concurrency", type=int, default=64)
    parser.add_argument("--dataset", "-d", type=str, required=True)
    parser.add_argument("--output_dir", "-o", required=True)
    parser.add_argument("--mode", choices=["simple", "cot"], default="simple")
    parser.add_argument("--max_tokens", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--force", "-f", action="store_true")
    args = parser.parse_args()

    dataset_name = args.dataset
    metrics_path = os.path.join(
        args.output_dir, f"{dataset_name}_baseline_metrics_{args.mode}.json"
    )

    if not args.force and os.path.exists(metrics_path):
        print(f"[SKIP] Already evaluated: {metrics_path}")
        return

    # Load data
    test_data = load_test_data(args.dataset)

    # Build prompts
    template = COT_TEMPLATE if args.mode == "cot" else SIMPLE_TEMPLATE
    prompts = [
        template.format(evidence=s["evidence"], claim=s["claim"])
        for s in test_data
    ]

    # Run inference
    if args.backend == "vllm":
        import torch
        from vllm import LLM, SamplingParams

        model = LLM(
            model=args.model,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=0.9,
            tensor_parallel_size=torch.cuda.device_count(),
        )
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=0.95 if args.temperature > 0 else 1.0,
            max_tokens=args.max_tokens,
            seed=42,
        )
        chat_prompts = [[{"role": "user", "content": p}] for p in prompts]
        outputs = model.chat(chat_prompts, sampling_params=sampling_params)
        generated = [o.outputs[0].text for o in outputs]
    else:
        from src.baselines.api import build_config, run_api_inference

        cfg = build_config(
            provider=args.provider,
            model=args.api_model,
            base_url=args.api_base_url,
            api_key_env=args.api_key_env,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_concurrency=args.max_concurrency,
        )
        generated = run_api_inference(cfg, prompts)

    # Process results
    results = []
    for sample, gen in zip(test_data, generated):
        results.append({
            "id": sample.get("id"),
            "claim": sample.get("claim"),
            "gt_label": sample.get("label"),
            "dataset": sample.get("src"),
            "generation": gen,
            "pred_label": extract_label(gen, args.mode),
        })

    # Compute metrics
    gt = [r["gt_label"] for r in results]
    pred = [r["pred_label"] for r in results]
    metrics = compute_classification_metrics(gt, pred)
    metrics["total_samples"] = len(results)
    metrics["mode"] = args.mode
    metrics["backend"] = args.backend
    metrics["model"] = args.model if args.backend == "vllm" else (args.api_model or args.provider)

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    jsonl_path = os.path.join(
        args.output_dir, f"{dataset_name}_baseline_results_{args.mode}.jsonl"
    )
    with jsonlines.open(jsonl_path, "w") as writer:
        for r in results:
            writer.write(r)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Results: {jsonl_path}")
    print(f"Metrics: {metrics_path}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
