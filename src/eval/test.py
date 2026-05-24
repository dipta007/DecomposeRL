"""
Test script for evaluating LoRA checkpoints on test data using vLLM.

Usage:
    # Run on all checkpoints in a directory
    PYTHONPATH=. python decomposer/unsloth/test.py -d data/coverbench/coverbench_2way.jsonl -c outputs/2way_3b

    # Run on a single checkpoint (auto-detected)
    PYTHONPATH=. python decomposer/unsloth/test.py -c outputs/2way_3b/checkpoint-100

With LLM-based rewards (requires running judge server):
    PYTHONPATH=. python decomposer/unsloth/test.py -c outputs/2way_3b --with_llm_rewards
"""

import argparse
import asyncio
import json
import os
import math
from typing import Dict, List, Optional, Tuple

import jsonlines
import numpy as np
import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

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


def get_base_model_from_adapter_config(checkpoint_path: str) -> str:
    """Read base model name from adapter_config.json in the checkpoint directory."""
    adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
    if not os.path.exists(adapter_config_path):
        raise FileNotFoundError(
            f"adapter_config.json not found in {checkpoint_path}. Make sure the path points to a valid LoRA checkpoint directory."
        )

    with open(adapter_config_path, "r") as f:
        config = json.load(f)

    base_model = config.get("base_model_name_or_path")
    if not base_model:
        raise ValueError(f"base_model_name_or_path not found in {adapter_config_path}")

    return base_model


def get_lora_rank_from_adapter_config(checkpoint_path: str) -> int:
    """Read LoRA rank from adapter_config.json in the checkpoint directory."""
    adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
    with open(adapter_config_path, "r") as f:
        config = json.load(f)
    return config.get("r", 64)


def find_checkpoints(checkpoint_dir: str) -> List[Tuple[str, int]]:
    """
    Find all checkpoint directories in the given path.

    Returns a list of tuples (checkpoint_path, checkpoint_index) sorted by index.
    """
    checkpoints = []

    # Check if the directory itself is a checkpoint (has adapter_config.json)
    if os.path.exists(os.path.join(checkpoint_dir, "adapter_config.json")):
        # first check if hte ckpt dir name is checkpoint-*
        checkpoint_index = 1
        if os.path.basename(checkpoint_dir).startswith("checkpoint-"):
            try:
                checkpoint_index = int(os.path.basename(checkpoint_dir).split("-")[-1])
            except ValueError:
                pass
        # It's a single checkpoint directory
        return [(checkpoint_dir, checkpoint_index)]

    # Otherwise, look for checkpoint-* subdirectories
    if not os.path.isdir(checkpoint_dir):
        raise NotADirectoryError(f"{checkpoint_dir} is not a directory")

    for item in os.listdir(checkpoint_dir):
        item_path = os.path.join(checkpoint_dir, item)
        if os.path.isdir(item_path) and item.startswith("checkpoint-"):
            # Check if it has adapter_config.json
            if os.path.exists(os.path.join(item_path, "adapter_config.json")):
                try:
                    checkpoint_index = int(item.split("-")[-1])
                    checkpoints.append((item_path, checkpoint_index))
                except ValueError:
                    print(f"Warning: Could not parse checkpoint index from {item}")

    if not checkpoints:
        raise FileNotFoundError(
            f"No valid checkpoints found in {checkpoint_dir}. "
            "Expected either adapter_config.json in the directory or "
            "checkpoint-* subdirectories with adapter_config.json."
        )

    # Sort by checkpoint index
    checkpoints.sort(key=lambda x: x[1])
    return checkpoints


def load_test_data(test_data_path: str) -> List[Dict]:
    """Load test data from JSONL file."""
    data = []
    with jsonlines.open(test_data_path) as reader:
        for line in reader:
            data.append(line)
    print(f"Loaded {len(data)} samples from {test_data_path}")
    return data


def extract_verification_label(generation: str) -> Optional[str]:
    """Extract the verification label from the generated output."""
    try:
        label = (
            generation.split("<verification>")[1].split("</verification>")[0].strip()
        )
        if label.lower() in ["supported", "refuted", "mixed"]:
            return label.lower()
        return None
    except Exception:
        return None


def extract_verification_confidence(
    generation: str, logprobs_list: Optional[List] = None
) -> Optional[float]:
    """
    Extract confidence for the verification label from token logprobs.

    Handles multi-token labels (e.g. "Refuted" -> ["Ref", "uted"] in Qwen)
    by locating the <verification> tag in the token stream, then using the
    first label token's logprob as the confidence. Only the first token
    matters because that's where the model decides between labels; subsequent
    sub-tokens (e.g. "uted" after "Ref") are near-deterministic continuations.

    Args:
        generation: The generated text
        logprobs_list: List of logprob dicts from vLLM output (one per token position,
                       each mapping token_id -> Logprob with .decoded_token and .logprob)

    Returns:
        Confidence (probability) for the verification label, or None if not found
    """
    LABELS = ["supported", "refuted", "mixed"]

    if logprobs_list is None or "<verification>" not in generation:
        return None

    try:
        # Step 1: Flatten logprobs into (decoded_token, logprob) pairs
        tokens = []
        for token_logprobs in logprobs_list:
            if token_logprobs is not None:
                for logprob_obj in token_logprobs.values():
                    tokens.append((logprob_obj.decoded_token, logprob_obj.logprob))

        # Step 2: Find the token index where <verification> tag ends
        # e.g. "<" + "verification" + ">\n" -> tag ends after ">\n"
        text_so_far = ""
        tag_end = None
        for i, (tok, _) in enumerate(tokens):
            text_so_far += tok
            if "<verification>" in text_so_far:
                tag_end = i + 1
                break

        if tag_end is None:
            return None

        # Step 3: Read the first non-whitespace token after the tag — that's
        # the label decision token (e.g. "Supported", "Ref", or "Mixed")
        for i in range(tag_end, min(tag_end + 5, len(tokens))):
            tok, logprob = tokens[i]
            if tok.strip():  # skip whitespace/newline tokens
                return math.exp(logprob)

        return None
    except Exception:
        return None


def run_inference(
    model: LLM,
    sampling_params: SamplingParams,
    test_data: List[Dict],
    checkpoint_path: str,
    lora_id: int = 1,
) -> Tuple[List[str], List[Optional[List]]]:
    """Run inference on test data using vLLM with LoRA.

    Returns:
        Tuple of (generated_texts, logprobs_list) where logprobs_list contains
        the token logprobs for each generation (or None if logprobs not enabled).
    """
    # Build prompts from claim + evidence using the standard template
    prompts = []
    for sample in test_data:
        messages = [
            {
                "role": "user",
                "content": USER_PROMPT_2WAY_TEMPLATE.format(
                    claim=sample["claim"],
                    evidence_doc=sample["evidence"],
                ),
            }
        ]
        prompts.append(messages)

    print(f"Running inference on {len(prompts)} samples...")

    # Generate outputs
    outputs = model.chat(
        prompts,
        sampling_params=sampling_params,
        lora_request=LoRARequest("test_lora", lora_id, checkpoint_path),
    )

    # Extract generated text and logprobs
    generated_texts = [output.outputs[0].text for output in outputs]
    logprobs_list = [output.outputs[0].logprobs for output in outputs]
    print(f"Generated {len(generated_texts)} outputs")

    return generated_texts, logprobs_list


def process_results(
    test_data: List[Dict],
    generated_texts: List[str],
    logprobs_list: Optional[List[Optional[List]]] = None,
    with_llm_rewards: bool = False,
) -> List[Dict]:
    """Process results and compute metrics for each sample."""

    # First pass: compute all local rewards synchronously (fast operations)
    print("Computing local rewards...")
    results = []
    all_questions = []

    for i, (sample, generation) in enumerate(zip(test_data, generated_texts)):
        # Extract QA pairs
        qa_pairs = extract_qa_pairs(generation)
        questions = [pair["question"] for pair in qa_pairs]
        answers = [pair["answer"] for pair in qa_pairs]
        all_questions.append(questions)

        # Extract verification label and confidence
        pred_label = extract_verification_label(generation)
        logprobs = logprobs_list[i] if logprobs_list else None
        pred_confidence = extract_verification_confidence(generation, logprobs)

        # Compute local rewards
        fmt_reward = get_format_reward(generation)
        fmt_reward_details = format_reward(generation, verbose=False)[1]
        verif_reward, _ = verification_reward(generation, sample["label"])

        # Build result dict
        result = {
            # Original fields
            "id": sample.get("id"),
            "dataset": sample.get("src"),
            "claim": sample.get("claim"),
            "gt_evidence": sample.get("evidence"),
            "gt_label": sample.get("label"),
            "metadata": sample.get("metadata", {}),
            # Generated output
            "generation": generation,
            "pred_label": pred_label,
            "pred_confidence": pred_confidence,
            # Extracted QA
            "num_of_questions": len(questions),
            "questions": questions,
            "answers": answers,
            # Local rewards
            "format_reward": fmt_reward,
            "format_reward_details": fmt_reward_details,
            "verification_reward": verif_reward,
        }
        results.append(result)

        if (i + 1) % 100 == 0:
            print(f"  Local rewards: {i + 1}/{len(test_data)} samples")

    # Second pass: compute LLM-based rewards in parallel (if enabled)
    if with_llm_rewards:
        # set WANDB_NAME to test
        os.environ["WANDB_NAME"] = "test"
        print("Computing LLM-based rewards in parallel...")

        from joblib import Parallel, delayed
        from tqdm import tqdm

        def _compute_single_llm_reward(i):
            document = test_data[i]["evidence"]
            return asyncio.run(
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

        llm_rewards_list = Parallel(n_jobs=64)(
            delayed(_compute_single_llm_reward)(i)
            for i in tqdm(range(len(test_data)), desc="Computing LLM rewards")
        )

        # Merge LLM rewards into results
        for i, llm_rewards in enumerate(llm_rewards_list):
            results[i].update(llm_rewards)

        print(f"  LLM rewards computed for {len(results)} samples")

    return results


def compute_aggregated_metrics(results: List[Dict], with_llm_rewards: bool) -> Dict:
    """Compute aggregated metrics from results."""
    gt_labels = [r["gt_label"] for r in results]
    pred_labels = [r["pred_label"] for r in results]

    # Classification metrics
    classification_metrics = compute_classification_metrics(gt_labels, pred_labels)

    # Question statistics
    num_of_questions = [r["num_of_questions"] for r in results]
    question_stats = {
        "mean_num_of_questions": float(np.mean(num_of_questions)),
        "max_num_of_questions": int(np.max(num_of_questions)),
        "min_num_of_questions": int(np.min(num_of_questions)),
        "std_num_of_questions": float(np.std(num_of_questions)),
    }

    # Reward statistics
    format_rewards = [r["format_reward"] for r in results]
    verification_rewards = [r["verification_reward"] for r in results]

    reward_stats = {
        "mean_format_reward": float(np.mean(format_rewards)),
        "mean_verification_reward": float(np.mean(verification_rewards)),
    }

    # Confidence statistics
    pred_confidences = [
        r["pred_confidence"] for r in results if r["pred_confidence"] is not None
    ]
    confidence_stats = {}
    if pred_confidences:
        confidence_stats = {
            "mean_pred_confidence": float(np.mean(pred_confidences)),
            "std_pred_confidence": float(np.std(pred_confidences)),
            "min_pred_confidence": float(np.min(pred_confidences)),
            "max_pred_confidence": float(np.max(pred_confidences)),
            "num_samples_with_confidence": len(pred_confidences),
        }

    if with_llm_rewards:
        llm_reward_keys = [
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
            reward_stats[f"mean_{key}"] = float(np.mean(values))

    return {
        **classification_metrics,
        **question_stats,
        **reward_stats,
        **confidence_stats,
        "total_samples": len(results),
    }


def get_dataset_name(test_data_path: str) -> str:
    """Extract dataset name from the test data path."""
    filename = os.path.basename(test_data_path)
    if filename.startswith("test_") and filename.endswith(".jsonl"):
        return filename[len("test_") : -len(".jsonl")]
    raise ValueError(
        f"Could not extract dataset name from {test_data_path}. "
        "Expected filename format: test_{dataset_name}.jsonl"
    )


def get_results_paths(
    output_dir: str, dataset_name: str, with_llm_rewards: bool
) -> tuple:
    """Get the paths for results files."""
    suffix = "_with_llm_rewards" if with_llm_rewards else ""
    jsonl_path = os.path.join(
        output_dir, f"{dataset_name}_test_results{suffix}_v2.jsonl"
    )
    json_path = os.path.join(output_dir, f"{dataset_name}_test_metrics{suffix}_v2.json")
    return jsonl_path, json_path


def is_checkpoint_evaluated(
    checkpoint_path: str, dataset_name: str, with_llm_rewards: bool
) -> bool:
    """Check if a checkpoint has already been evaluated."""
    jsonl_path, json_path = get_results_paths(
        checkpoint_path, dataset_name, with_llm_rewards
    )
    if not os.path.exists(json_path):
        return False
    try:
        with open(json_path, "r") as f:
            json.load(f)
        return True
    except (json.JSONDecodeError, Exception):
        print(f"[WARN] Corrupted metrics file, will re-evaluate: {json_path}")
        os.remove(json_path)
        if os.path.exists(jsonl_path):
            os.remove(jsonl_path)
        return False


def load_existing_metrics(
    checkpoint_path: str, dataset_name: str, with_llm_rewards: bool
) -> Optional[Dict]:
    """Load existing metrics from a previously evaluated checkpoint."""
    try:
        _, json_path = get_results_paths(
            checkpoint_path, dataset_name, with_llm_rewards
        )
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load existing metrics from {json_path}: {e}")
    return None


def save_results(
    results: List[Dict],
    metrics: Dict,
    output_dir: str,
    dataset_name: str,
    with_llm_rewards: bool,
):
    """Save results and metrics to files."""
    os.makedirs(output_dir, exist_ok=True)

    jsonl_path, json_path = get_results_paths(
        output_dir, dataset_name, with_llm_rewards
    )

    # Atomic write: temp file in same dir, then os.replace.
    # Prevents concurrent eval processes from reading a truncated file
    # and triggering the corruption-cleanup in is_checkpoint_evaluated.
    jsonl_tmp = f"{jsonl_path}.tmp"
    with jsonlines.open(jsonl_tmp, "w") as writer:
        for result in results:
            writer.write(result)
    os.replace(jsonl_tmp, jsonl_path)
    print(f"Saved per-sample results to {jsonl_path}")

    json_tmp = f"{json_path}.tmp"
    with open(json_tmp, "w") as f:
        json.dump(metrics, f, indent=2)
    os.replace(json_tmp, json_path)
    print(f"Saved aggregated metrics to {json_path}")


def evaluate_checkpoint(
    model: LLM,
    sampling_params: SamplingParams,
    test_data: List[Dict],
    checkpoint_path: str,
    checkpoint_index: int,
    dataset_name: str,
    with_llm_rewards: bool,
) -> Dict:
    """Evaluate a single checkpoint and return metrics."""
    print(f"\n{'=' * 60}")
    print(f"Evaluating checkpoint: {checkpoint_path}")
    print(f"Checkpoint index: {checkpoint_index}")
    print(f"{'=' * 60}")

    # Run inference
    generated_texts, logprobs_list = run_inference(
        model=model,
        sampling_params=sampling_params,
        test_data=test_data,
        checkpoint_path=checkpoint_path,
        lora_id=checkpoint_index,
    )

    # Process results and compute metrics
    print("Processing results and computing metrics...")
    results = process_results(
        test_data=test_data,
        generated_texts=generated_texts,
        logprobs_list=logprobs_list,
        with_llm_rewards=with_llm_rewards,
    )

    # Compute aggregated metrics
    metrics = compute_aggregated_metrics(results, with_llm_rewards)
    metrics["checkpoint_index"] = checkpoint_index
    metrics["checkpoint_path"] = checkpoint_path

    print(f"\n{'=' * 60}")
    print(f"Metrics for checkpoint {checkpoint_index}:")
    print(f"{'=' * 60}")
    for k, v in metrics.items():
        if k not in ["checkpoint_path"]:
            print(f"  {k}: {v}")
    print(f"{'=' * 60}")

    # Save results to the checkpoint directory
    save_results(results, metrics, checkpoint_path, dataset_name, with_llm_rewards)

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Test LoRA checkpoint(s) on test data using vLLM"
    )
    parser.add_argument(
        "--checkpoint_dir",
        "-c",
        type=str,
        required=True,
        help="Path to checkpoint directory (contains checkpoint-* subdirs) or single checkpoint",
    )
    parser.add_argument(
        "--test_data",
        "-d",
        type=str,
        default="data/combined/step_9/test_pubmedclaim.jsonl",
        help="Path to test data JSONL file",
    )
    parser.add_argument(
        "--with_llm_rewards",
        action="store_true",
        help="Enable LLM-based rewards (requires running judge server)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=12000,
        help="Maximum tokens to generate (default: 6000)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=32768,
        help="Maximum model context length (default: 8192)",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force re-evaluation of already-evaluated checkpoints",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Test Configuration:")
    print("=" * 60)
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("=" * 60)

    # Find all checkpoints (auto-detects single checkpoint or directory with multiple)
    checkpoints = find_checkpoints(args.checkpoint_dir)

    print(f"\nFound {len(checkpoints)} checkpoint(s):")
    for cp_path, cp_idx in checkpoints:
        print(f"  [{cp_idx}] {cp_path}")

    # Get dataset name early so we can check for existing evaluations
    dataset_name = get_dataset_name(args.test_data)
    print(f"Dataset name: {dataset_name}")

    # Filter out already-evaluated checkpoints before loading the model
    all_metrics = []
    remaining_checkpoints = []
    skipped_count = 0
    for checkpoint_path, checkpoint_index in checkpoints:
        if checkpoint_index % 50 != 0:
            print(
                f"\n[SKIP] Checkpoint {checkpoint_index} does not meet evaluation frequency (every 50 steps): {checkpoint_path}"
            )
            continue
        if not args.force and is_checkpoint_evaluated(
            checkpoint_path, dataset_name, args.with_llm_rewards
        ):
            print(
                f"\n[SKIP] Checkpoint {checkpoint_index} already evaluated: {checkpoint_path}"
            )
            existing_metrics = load_existing_metrics(
                checkpoint_path, dataset_name, args.with_llm_rewards
            )
            if existing_metrics:
                all_metrics.append(existing_metrics)
            skipped_count += 1
        else:
            remaining_checkpoints.append((checkpoint_path, checkpoint_index))

    if skipped_count > 0:
        print(
            f"\nSkipped {skipped_count} already-evaluated checkpoint(s) (use --force to re-evaluate)"
        )

    if not remaining_checkpoints:
        print("\nAll checkpoints already evaluated. Nothing to do.")
    else:
        print(f"\n{len(remaining_checkpoints)} checkpoint(s) to evaluate:")
        for cp_path, cp_idx in remaining_checkpoints:
            print(f"  [{cp_idx}] {cp_path}")

        # Get base model from the first checkpoint's adapter_config.json
        first_checkpoint_path = checkpoints[0][0]
        base_model = get_base_model_from_adapter_config(first_checkpoint_path)
        max_lora_rank = get_lora_rank_from_adapter_config(first_checkpoint_path)

        print(f"\nBase model (from adapter_config.json): {base_model}")
        print(f"LoRA rank (from adapter_config.json): {max_lora_rank}")

        # Load test data
        test_data = load_test_data(args.test_data)

        # Initialize vLLM with LoRA support
        print(f"\nInitializing vLLM with base model: {base_model}")
        model = LLM(
            model=base_model,
            max_model_len=args.max_model_len,
            enable_lora=True,
            max_lora_rank=max_lora_rank,
            gpu_memory_utilization=0.92,
            tensor_parallel_size=torch.cuda.device_count(),
            dtype="bfloat16",
            enable_prefix_caching=True,
            enable_chunked_prefill=True,
            max_num_seqs=256,
        )

        # Set up sampling parameters (enable logprobs for confidence extraction)
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=0.95 if args.temperature > 0 else 1.0,
            max_tokens=args.max_tokens,
            seed=42,
            logprobs=1,  # Enable logprobs for verification label confidence
        )

        # Evaluate remaining checkpoints
        for checkpoint_path, checkpoint_index in remaining_checkpoints:
            metrics = evaluate_checkpoint(
                model=model,
                sampling_params=sampling_params,
                test_data=test_data,
                checkpoint_path=checkpoint_path,
                checkpoint_index=checkpoint_index,
                dataset_name=dataset_name,
                with_llm_rewards=args.with_llm_rewards,
            )
            all_metrics.append(metrics)

    # Save summary of all checkpoints
    if all_metrics:
        summary_path = os.path.join(
            args.checkpoint_dir, f"{dataset_name}_test_summary.json"
        )
        with open(summary_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\nSaved summary of all checkpoints to {summary_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()


"""
PYTHONPATH=. uv run decomposer/unsloth/test.py -c outputs/2way_7b/
PYTHONPATH=. uv run decomposer/unsloth/test.py -c outputs/2way_3b/
"""
