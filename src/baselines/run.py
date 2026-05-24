"""
Unified runner for prompted baselines on a single eval JSONL.

Supports 6 methods (see baselines/prompts.py):
  self_ask, decomposed_prompting, hiss, folk, programfc, chen_complex

Two inference backends:
  --backend vllm   : local Qwen via vLLM (default; same as decomposer/eval/baseline.py)
  --backend api    : OpenAI-compatible API (gpt-4o-mini default, claude-haiku-4-5 via --provider anthropic)

Output format mirrors decomposer/eval/baseline.py so existing analysis scripts
(decomposer/analysis, etc.) keep working: per-dataset JSONL of results + a
metrics JSON alongside it.

Usage examples:
    # vLLM, self-ask, Qwen2.5-7B, one dataset
    PYTHONPATH=. python decomposer/baselines/run.py \\
        --method self_ask \\
        --backend vllm \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --test_data data/combined/step_9/test_pubmedclaim.jsonl \\
        --output_dir outputs/baseline_self_ask_qwen7b

    # OpenAI API, hiss, gpt-4o-mini
    PYTHONPATH=. python decomposer/baselines/run.py \\
        --method hiss \\
        --backend api --provider openai \\
        --test_data data/combined/step_9/test_pubmedclaim.jsonl \\
        --output_dir outputs/baseline_hiss_gpt4o_mini

    # Anthropic via OpenAI SDK, folk, claude-haiku-4-5
    PYTHONPATH=. python decomposer/baselines/run.py \\
        --method folk \\
        --backend api --provider anthropic \\
        --test_data data/combined/step_9/test_pubmedclaim.jsonl \\
        --output_dir outputs/baseline_folk_haiku
"""

import argparse
import json
import os
from typing import Dict, List

import jsonlines

from decomposer.baselines.prompts import (
    PROMPTED_TEMPLATES,
    build_prompted_prompt,
    extract_verdict_tag,
)
from decomposer.eval.utils import compute_classification_metrics


def get_dataset_name(test_data_path: str) -> str:
    filename = os.path.basename(test_data_path)
    if filename.startswith("test_") and filename.endswith(".jsonl"):
        return filename[len("test_") : -len(".jsonl")]
    raise ValueError(f"Unexpected test data filename format: {filename}")


def load_test_data(test_data_path: str) -> List[Dict]:
    data = []
    with jsonlines.open(test_data_path) as reader:
        for line in reader:
            data.append(line)
    print(f"Loaded {len(data)} samples from {test_data_path}")
    return data


def build_prompts(method: str, test_data: List[Dict]) -> List[str]:
    return [
        build_prompted_prompt(method, sample["claim"], sample["evidence"])
        for sample in test_data
    ]


def run_vllm(model_id: str, prompts: List[str], max_tokens: int, max_model_len: int) -> List[str]:
    import torch
    from vllm import LLM, SamplingParams

    print(f"Initializing vLLM with model: {model_id}")
    model = LLM(
        model=model_id,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=torch.cuda.device_count(),
    )
    sampling_params = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=max_tokens, seed=42
    )
    chat_prompts = [[{"role": "user", "content": p}] for p in prompts]
    outputs = model.chat(chat_prompts, sampling_params=sampling_params)
    return [o.outputs[0].text for o in outputs]


def run_api(args, prompts: List[str]) -> List[str]:
    from decomposer.baselines.api import build_config, run_api_inference

    cfg = build_config(
        provider=args.provider,
        model=args.api_model,
        base_url=args.api_base_url,
        api_key_env=args.api_key_env,
        temperature=0.0,
        max_tokens=args.max_tokens,
        max_concurrency=args.max_concurrency,
    )
    print(f"Calling API: provider={cfg.provider} model={cfg.model} base_url={cfg.base_url}")
    return run_api_inference(cfg, prompts)


def process_results(
    method: str, test_data: List[Dict], generated_texts: List[str]
) -> List[Dict]:
    results = []
    for sample, gen in zip(test_data, generated_texts):
        pred = extract_verdict_tag(gen)
        results.append(
            {
                "id": sample.get("id"),
                "claim": sample.get("claim"),
                "gt_label": sample.get("label"),
                "dataset": sample.get("src"),
                "method": method,
                "generation": gen,
                "pred_label": pred,
            }
        )
    return results


def compute_metrics(results: List[Dict]) -> Dict:
    gt = [r["gt_label"] for r in results]
    pred = [r["pred_label"] for r in results]
    metrics = compute_classification_metrics(gt, pred)
    metrics["total_samples"] = len(results)
    metrics["unparsed"] = sum(1 for r in results if r["pred_label"] is None)
    return metrics


def save(
    results: List[Dict],
    metrics: Dict,
    output_dir: str,
    dataset_name: str,
    method: str,
):
    """Match the filename convention of decomposer/eval/baseline.py:
        {dataset}_baseline_results_{method}.jsonl
        {dataset}_baseline_metrics_{method}.json
    Backend / model variant is encoded in the output_dir, not the filename.
    """
    os.makedirs(output_dir, exist_ok=True)
    suffix = f"_{method}"
    jsonl_path = os.path.join(output_dir, f"{dataset_name}_baseline_results{suffix}.jsonl")
    with jsonlines.open(jsonl_path, "w") as w:
        for r in results:
            w.write(r)
    json_path = os.path.join(output_dir, f"{dataset_name}_baseline_metrics{suffix}.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved {jsonl_path}")
    print(f"Saved {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Prompted-baseline runner")
    parser.add_argument(
        "--method",
        required=True,
        choices=list(PROMPTED_TEMPLATES.keys()),
        help="Which prompted baseline to run",
    )
    parser.add_argument(
        "--backend",
        choices=["vllm", "api"],
        default="vllm",
        help="Inference backend",
    )
    # vLLM-only
    parser.add_argument("--model", "-m", default="Qwen/Qwen2.5-7B-Instruct", help="vLLM model id")
    parser.add_argument("--max_model_len", type=int, default=32768)
    # API-only
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "custom", "anthropic_native", "umbc"],
        default="openai",
    )
    parser.add_argument("--api_model", default=None, help="Override default model for the provider")
    parser.add_argument("--api_base_url", default=None, help="Override base URL (custom provider)")
    parser.add_argument(
        "--api_key_env",
        default=None,
        help="Env var name holding the API key (custom provider)",
    )
    parser.add_argument("--max_concurrency", type=int, default=64)
    # Shared
    parser.add_argument(
        "--test_data", "-d", required=True, help="Path to test JSONL"
    )
    parser.add_argument("--output_dir", "-o", required=True)
    parser.add_argument("--max_tokens", type=int, default=16768)
    parser.add_argument("--force", "-f", action="store_true")
    args = parser.parse_args()

    # Idempotency check before loading anything heavy.
    dataset_name = get_dataset_name(args.test_data)
    metrics_path = os.path.join(
        args.output_dir, f"{dataset_name}_baseline_metrics_{args.method}.json"
    )
    if not args.force and os.path.exists(metrics_path):
        print(f"[SKIP] {metrics_path} already exists. Pass --force to recompute.")
        return

    test_data = load_test_data(args.test_data)
    prompts = build_prompts(args.method, test_data)

    if args.backend == "vllm":
        generations = run_vllm(args.model, prompts, args.max_tokens, args.max_model_len)
    else:
        generations = run_api(args, prompts)

    results = process_results(args.method, test_data, generations)
    metrics = compute_metrics(results)
    # Match decomposer/eval/baseline.py metadata keys.
    metrics["prompt_mode"] = args.method
    metrics["backend"] = args.backend
    if args.backend == "vllm":
        metrics["model"] = args.model
    else:
        metrics["provider"] = args.provider
        metrics["model"] = args.api_model or "default"

    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print("=" * 60)

    save(results, metrics, args.output_dir, dataset_name, args.method)


if __name__ == "__main__":
    main()
