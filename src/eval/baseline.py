"""
Baseline evaluation script for raw instruct models (without LoRA fine-tuning).

Usage:
    # Iterative mode (same prompt as fine-tuned model)
    PYTHONPATH=. python decomposer/unsloth/baseline.py -m Qwen/Qwen2.5-7B-Instruct -o outputs/baseline_7b --prompt_mode iterative

    # Simple mode (direct verification)
    PYTHONPATH=. python decomposer/unsloth/baseline.py -m Qwen/Qwen2.5-7B-Instruct -o outputs/baseline_7b_simple --prompt_mode simple

    # CoT mode (chain-of-thought reasoning then verdict)
    PYTHONPATH=. python decomposer/unsloth/baseline.py -m Qwen/Qwen2.5-7B-Instruct -o outputs/baseline_7b_cot --prompt_mode cot

    # With LLM rewards (iterative only)
    PYTHONPATH=. python decomposer/unsloth/baseline.py -m Qwen/Qwen2.5-7B-Instruct -o outputs/baseline_7b --prompt_mode iterative --with_llm_rewards
"""

import argparse
import asyncio
import json
import os
from typing import Callable, Dict, List, Optional

import jsonlines
import numpy as np

from decomposer.eval.utils import (
    compute_classification_metrics,
    compute_llm_rewards,
)
from decomposer.prompts import USER_PROMPT_2WAY_TEMPLATE
from decomposer.unsloth.format_reward import format_reward, get_format_reward
from decomposer.unsloth.rewards import (
    extract_qa_pairs,
    verification_reward,
)


# Simple prompt template for direct verification
SIMPLE_PROMPT_TEMPLATE = """\
You are a fact-checking assistant. Given the evidence and claim below, determine if the claim is Supported or Refuted by the evidence.

<evidence>
{evidence}
</evidence>

<claim>
{claim}
</claim>

Based on the evidence provided, is the claim Supported or Refuted?

Answer with exactly one word: Supported or Refuted"""

# Chain-of-thought prompt template
COT_PROMPT_TEMPLATE = """\
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


def get_dataset_name(test_data_path: str) -> str:
    """Extract dataset name from the test data path."""
    filename = os.path.basename(test_data_path)
    if filename.startswith("test_") and filename.endswith(".jsonl"):
        return filename[len("test_") : -len(".jsonl")]

    raise ValueError(f"Unexpected test data filename format: {filename}")


def load_test_data(test_data_path: str) -> List[Dict]:
    """Load test data from JSONL file."""
    data = []
    with jsonlines.open(test_data_path) as reader:
        for line in reader:
            data.append(line)
    print(f"Loaded {len(data)} samples from {test_data_path}")
    return data


def create_simple_prompts(test_data: List[Dict]) -> List[str]:
    """Create simple direct verification prompts."""
    return [
        SIMPLE_PROMPT_TEMPLATE.format(evidence=s["evidence"], claim=s["claim"])
        for s in test_data
    ]


def create_cot_prompts(test_data: List[Dict]) -> List[str]:
    """Create chain-of-thought prompts."""
    return [
        COT_PROMPT_TEMPLATE.format(evidence=s["evidence"], claim=s["claim"])
        for s in test_data
    ]


def extract_simple_label(generation: str) -> Optional[str]:
    """Extract verification label from simple prompt output."""
    generation_lower = generation.strip().lower()

    # Check for exact match first
    if generation_lower in ["supported", "refuted"]:
        return generation_lower

    # Check if the response contains the label
    if "supported" in generation_lower and "refuted" not in generation_lower:
        return "supported"
    if "refuted" in generation_lower and "supported" not in generation_lower:
        return "refuted"

    # Try to find at the beginning or end
    words = generation_lower.split()
    if words:
        if words[0] in ["supported", "refuted"]:
            return words[0]
        if words[-1] in ["supported", "refuted"]:
            return words[-1]

    return None


def extract_cot_label(generation: str) -> Optional[str]:
    """Extract verification label from chain-of-thought output."""
    # Try to extract from <verdict> tags first
    try:
        label = generation.split("<verdict>")[1].split("</verdict>")[0].strip()
        if label.lower() in ["supported", "refuted"]:
            return label.lower()
    except Exception:
        pass

    # Fallback to simple extraction if tags not found
    return extract_simple_label(generation)


def extract_cot_reasoning(generation: str) -> Optional[str]:
    """Extract reasoning from chain-of-thought output."""
    try:
        reasoning = generation.split("<reasoning>")[1].split("</reasoning>")[0].strip()
        return reasoning
    except Exception:
        return None


def extract_iterative_label(generation: str) -> Optional[str]:
    """Extract verification label from iterative prompt output."""
    try:
        label = (
            generation.split("<verification>")[1].split("</verification>")[0].strip()
        )
        if label.lower() in ["supported", "refuted", "mixed"]:
            return label.lower()
        return None
    except Exception:
        return None


def make_vllm_inference(
    model_id: str, max_tokens: int, max_model_len: int, temperature: float
) -> tuple[Callable[[List[str]], List[str]], str]:
    """Lazy-init vLLM and return (infer, resolved_model_id)."""
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
        temperature=temperature,
        top_p=0.95 if temperature > 0 else 1.0,
        max_tokens=max_tokens,
        seed=42,
    )

    def infer(prompts: List[str]) -> List[str]:
        if not prompts:
            return []
        chat_prompts = [[{"role": "user", "content": p}] for p in prompts]
        outputs = model.chat(chat_prompts, sampling_params=sampling_params)
        return [o.outputs[0].text for o in outputs]

    return infer, model_id


def make_api_inference(args) -> tuple[Callable[[List[str]], List[str]], str]:
    """Build (infer, resolved_model_id) for an API-backed run.

    The returned model_id is the one `build_config` actually resolved (e.g.
    the provider default when `--api_model` was omitted), so it can be
    recorded in metrics instead of the literal string "default".
    """
    from decomposer.baselines.api import build_config, run_api_inference

    cfg = build_config(
        provider=args.provider,
        model=args.api_model,
        base_url=args.api_base_url,
        api_key_env=args.api_key_env,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_concurrency=args.max_concurrency,
    )
    print(
        f"Calling API: provider={cfg.provider} model={cfg.model} base_url={cfg.base_url}"
    )

    def infer(prompts: List[str]) -> List[str]:
        if not prompts:
            return []
        return run_api_inference(cfg, prompts)

    return infer, cfg.model


def process_simple_results(
    test_data: List[Dict],
    generated_texts: List[str],
) -> List[Dict]:
    """Process results for simple prompt mode."""
    print("Processing simple mode results...")
    results = []

    for i, (sample, generation) in enumerate(zip(test_data, generated_texts)):
        pred_label = extract_simple_label(generation)

        result = {
            "id": sample.get("id"),
            "claim": sample.get("claim"),
            "gt_label": sample.get("label"),
            "dataset": sample.get("src"),
            "generation": generation,
            "pred_label": pred_label,
        }
        results.append(result)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(test_data)} samples")

    return results


def process_cot_results(
    test_data: List[Dict],
    generated_texts: List[str],
) -> List[Dict]:
    """Process results for chain-of-thought prompt mode."""
    print("Processing CoT mode results...")
    results = []

    for i, (sample, generation) in enumerate(zip(test_data, generated_texts)):
        pred_label = extract_cot_label(generation)
        reasoning = extract_cot_reasoning(generation)

        result = {
            "id": sample.get("id"),
            "claim": sample.get("claim"),
            "gt_label": sample.get("label"),
            "dataset": sample.get("src"),
            "generation": generation,
            "reasoning": reasoning,
            "pred_label": pred_label,
        }
        results.append(result)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(test_data)} samples")

    return results


def process_iterative_results(
    test_data: List[Dict],
    generated_texts: List[str],
    with_llm_rewards: bool = False,
) -> List[Dict]:
    """Process results for iterative prompt mode (same as test.py)."""
    print("Processing iterative mode results...")
    results = []
    all_questions = []

    for i, (sample, generation) in enumerate(zip(test_data, generated_texts)):
        # Extract QA pairs
        qa_pairs = extract_qa_pairs(generation)
        questions = [pair["question"] for pair in qa_pairs]
        answers = [pair["answer"] for pair in qa_pairs]
        all_questions.append(questions)

        # Extract verification label
        pred_label = extract_iterative_label(generation)

        # Compute local rewards
        fmt_reward = get_format_reward(generation)
        fmt_reward_details = format_reward(generation, verbose=False)[1]
        verif_reward, _ = verification_reward(generation, sample["label"])

        result = {
            "id": sample.get("id"),
            "claim": sample.get("claim"),
            "gt_label": sample.get("label"),
            "dataset": sample.get("src"),
            "generation": generation,
            "pred_label": pred_label,
            "num_of_questions": len(questions),
            "questions": questions,
            "answers": answers,
            "format_reward": fmt_reward,
            "format_reward_details": fmt_reward_details,
            "verification_reward": verif_reward,
        }
        results.append(result)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(test_data)} samples")

    # Compute LLM-based rewards in parallel if enabled
    if with_llm_rewards:
        print("Computing LLM-based rewards in parallel...")

        async def _compute_all_llm_rewards():
            tasks = []
            for i in range(len(test_data)):
                document = test_data[i]["evidence"]
                tasks.append(
                    compute_llm_rewards(
                        claim=test_data[i]["claim"],
                        generation=generated_texts[i],
                        questions=all_questions[i],
                        answers=results[i]["answers"],
                        document=document,
                        gt_label=test_data[i]["label"],
                        gt_num_of_questions=test_data[i].get("num_of_questions"),
                    )
                )
            return await asyncio.gather(*tasks)

        llm_rewards_list = asyncio.run(_compute_all_llm_rewards())

        for i, llm_rewards in enumerate(llm_rewards_list):
            results[i].update(llm_rewards)

        print(f"  LLM rewards computed for {len(results)} samples")

    return results


def compute_simple_metrics(results: List[Dict]) -> Dict:
    """Compute metrics for simple mode (classification only)."""
    gt_labels = [r["gt_label"] for r in results]
    pred_labels = [r["pred_label"] for r in results]

    metrics = compute_classification_metrics(gt_labels, pred_labels)
    metrics["total_samples"] = len(results)

    return metrics


def compute_iterative_metrics(results: List[Dict], with_llm_rewards: bool) -> Dict:
    """Compute metrics for iterative mode (all rewards)."""
    gt_labels = [r["gt_label"] for r in results]
    pred_labels = [r["pred_label"] for r in results]

    # Classification metrics
    metrics = compute_classification_metrics(gt_labels, pred_labels)

    # Question statistics
    num_of_questions = [r["num_of_questions"] for r in results]
    metrics.update(
        {
            "mean_num_of_questions": float(np.mean(num_of_questions)),
            "max_num_of_questions": int(np.max(num_of_questions)),
            "min_num_of_questions": int(np.min(num_of_questions)),
            "std_num_of_questions": float(np.std(num_of_questions)),
        }
    )

    # Reward statistics
    format_rewards = [r["format_reward"] for r in results]
    verification_rewards = [r["verification_reward"] for r in results]

    metrics.update(
        {
            "mean_format_reward": float(np.mean(format_rewards)),
            "mean_verification_reward": float(np.mean(verification_rewards)),
        }
    )

    if with_llm_rewards:
        llm_reward_keys = [
            "saliency_reward",
            "atomicity_reward",
            "atomicity_checklist_reward",
            "mmr_reward",
            "vendi_diversity_reward",
            "question_answerable_reward",
            "answer_correctness_reward",
            "coverage_reward",
            "good_number_of_questions_reward",
            "necessity_saliency_reward",
            "joint_quality_reward",
        ]
        for key in llm_reward_keys:
            values = [r[key] for r in results]
            metrics[f"mean_{key}"] = float(np.mean(values))

    metrics["total_samples"] = len(results)

    return metrics


def save_results(
    results: List[Dict],
    metrics: Dict,
    output_dir: str,
    dataset_name: str,
    prompt_mode: str,
    with_llm_rewards: bool,
):
    """Save results and metrics to files."""
    os.makedirs(output_dir, exist_ok=True)

    suffix = f"_{prompt_mode}"
    if with_llm_rewards and prompt_mode == "iterative":
        suffix += "_with_llm_rewards"

    # Save per-sample results as JSONL
    jsonl_path = os.path.join(
        output_dir, f"{dataset_name}_baseline_results{suffix}.jsonl"
    )
    with jsonlines.open(jsonl_path, "w") as writer:
        for result in results:
            writer.write(result)
    print(f"Saved per-sample results to {jsonl_path}")

    # Save aggregated metrics as JSON
    json_path = os.path.join(
        output_dir, f"{dataset_name}_baseline_metrics{suffix}.json"
    )
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved aggregated metrics to {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Baseline evaluation for raw instruct models"
    )
    parser.add_argument(
        "--backend",
        choices=["vllm", "api"],
        default="vllm",
        help="Inference backend: local vLLM or remote API (default: vllm)",
    )
    # vLLM-only
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="vLLM model id (ignored when --backend api)",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=32768,
        help="vLLM max model context length (default: 32768)",
    )
    # API-only
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "custom", "anthropic_native", "umbc"],
        default="openai",
        help="API provider (only used when --backend api)",
    )
    parser.add_argument("--api_model", default=None,
        help="API model id; falls back to provider default")
    parser.add_argument("--api_base_url", default=None)
    parser.add_argument("--api_key_env", default=None)
    parser.add_argument("--max_concurrency", type=int, default=64)
    parser.add_argument(
        "--test_data",
        "-d",
        type=str,
        default="data/combined/step_9/test_pubmedclaim.jsonl",
        help="Path to test data JSONL file",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "--prompt_mode",
        type=str,
        choices=["iterative", "simple", "cot"],
        default="iterative",
        help="Prompt mode: 'iterative' (same as fine-tuned), 'simple' (direct verification), or 'cot' (chain-of-thought)",
    )
    parser.add_argument(
        "--with_llm_rewards",
        action="store_true",
        help="Enable LLM-based rewards (only for iterative mode)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=12000,
        help="Maximum tokens to generate (default: 6000 for iterative, 1000 for cot, 50 for simple)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force re-evaluation even if results already exist",
    )

    args = parser.parse_args()

    # Warn if LLM rewards requested for non-iterative mode
    if args.with_llm_rewards and args.prompt_mode != "iterative":
        print(
            f"Warning: --with_llm_rewards is ignored for {args.prompt_mode} prompt mode"
        )
        args.with_llm_rewards = False

    print("=" * 60)
    print("Baseline Configuration:")
    print("=" * 60)
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("=" * 60)

    # Check if already evaluated before loading the model
    dataset_name = get_dataset_name(args.test_data)
    print(f"Dataset name: {dataset_name}")

    suffix = f"_{args.prompt_mode}"
    if args.with_llm_rewards and args.prompt_mode == "iterative":
        suffix += "_with_llm_rewards"
    metrics_path = os.path.join(
        args.output_dir, f"{dataset_name}_baseline_metrics{suffix}.json"
    )

    if not args.force and os.path.exists(metrics_path):
        print(f"\n[SKIP] Already evaluated: {metrics_path}")
        print("Use --force to re-evaluate.")
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        print(f"\n{'=' * 60}")
        print("Existing Metrics:")
        print(f"{'=' * 60}")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        print(f"{'=' * 60}")
        print("\nDone!")
        return

    # Load test data
    test_data = load_test_data(args.test_data)

    # Create prompts based on mode (all return List[str])
    if args.prompt_mode == "simple":
        prompts = create_simple_prompts(test_data)
    elif args.prompt_mode == "cot":
        prompts = create_cot_prompts(test_data)
    else:  # iterative
        prompts = [
            USER_PROMPT_2WAY_TEMPLATE.format(
                claim=s["claim"], evidence_doc=s["evidence"]
            )
            for s in test_data
        ]

    # Dispatch to the chosen backend
    if args.backend == "vllm":
        infer, resolved_model = make_vllm_inference(
            args.model, args.max_tokens, args.max_model_len, args.temperature
        )
    else:
        infer, resolved_model = make_api_inference(args)

    print(f"Running inference on {len(prompts)} samples (backend={args.backend})...")
    generated_texts = infer(prompts)
    print(f"Generated {len(generated_texts)} outputs")

    # Process results based on mode
    if args.prompt_mode == "simple":
        results = process_simple_results(test_data, generated_texts)
        metrics = compute_simple_metrics(results)
    elif args.prompt_mode == "cot":
        results = process_cot_results(test_data, generated_texts)
        metrics = compute_simple_metrics(results)  # CoT uses same metrics as simple
    else:  # iterative
        results = process_iterative_results(
            test_data, generated_texts, args.with_llm_rewards
        )
        metrics = compute_iterative_metrics(results, args.with_llm_rewards)

    # Add metadata
    metrics["prompt_mode"] = args.prompt_mode
    metrics["backend"] = args.backend
    metrics["model"] = resolved_model
    if args.backend == "api":
        metrics["provider"] = args.provider

    print(f"\n{'=' * 60}")
    print("Aggregated Metrics:")
    print(f"{'=' * 60}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"{'=' * 60}")

    # Save results
    save_results(
        results,
        metrics,
        args.output_dir,
        dataset_name,
        args.prompt_mode,
        args.with_llm_rewards,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
