"""
QACheck baseline (Pan et al., EMNLP 2023 demo paper).

Paper: https://aclanthology.org/2023.emnlp-demo.23/
Repo:  https://github.com/XinyuanLu00/QACheck

Pipeline (per claim):
  state = []  # list of (question, answer) pairs
  for turn in range(max_turns):
      if SUFFICIENCY_CHECK(claim, state) == "yes":
          break
      q = NEXT_QUESTION(claim, evidence, state)
      a = ANSWER(evidence, q)
      state.append((q, a))
  verdict = FINAL_VERDICT(claim, evidence, state)

We keep the 4 prompts separate (faithful to the paper) but batch all claims at
each step so a sweep over 1500 claims is N_turns * 3 batched LLM calls (vLLM)
or N_turns * 3 rounds of async API calls instead of 1500 * N_turns * 3
sequential calls. Claims that exit the loop early sit out subsequent rounds.

Adaptations:
  - We answer from provided evidence only (closed-evidence) instead of using
    a search engine.
  - We run on the same Qwen / frontier-API LLMs as the other prompted baselines.
"""

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import jsonlines

from src.baselines.prompts import extract_verdict_tag
from src.baselines.utils import compute_classification_metrics, load_test_data


METHOD_NAME = "qacheck"
DEFAULT_MAX_TURNS = 5


# QACheck-specific prompts. Kept in this file (not prompts.py) because they
# form a state machine of 4 cooperating prompts, not a single-shot template.
SUFFICIENCY_PROMPT = """\
You are verifying the claim below against the evidence below. Given the QA
history collected so far, do we have enough information to determine whether
the claim is Supported or Refuted? Answer with exactly "Yes" or "No" inside the
tag below.

<claim>{claim}</claim>
<evidence>{evidence}</evidence>
<qa_history>
{qa_history}
</qa_history>

Output exactly one of:
<enough>Yes</enough>
<enough>No</enough>"""


NEXT_QUESTION_PROMPT = """\
You are verifying the claim below against the evidence below. Given the QA
history so far, generate the SINGLE most useful next question to make progress
toward a verdict. The question must be answerable from the evidence and must
not duplicate any question already in the history.

<claim>{claim}</claim>
<evidence>{evidence}</evidence>
<qa_history>
{qa_history}
</qa_history>

Output your question inside <question>...</question>."""


ANSWER_PROMPT = """\
Answer the question using ONLY the evidence below. If the evidence does not
answer the question, reply with "I don't know".

<evidence>{evidence}</evidence>
<question>{question}</question>

Output your answer inside <answer>...</answer>."""


FINAL_VERDICT_PROMPT = """\
You are verifying the claim below against the evidence below. Based on the QA
history collected so far, output the final verdict.

<claim>{claim}</claim>
<evidence>{evidence}</evidence>
<qa_history>
{qa_history}
</qa_history>

Output exactly one of:
<verdict>Supported</verdict>
<verdict>Refuted</verdict>"""


# --------------------------------------------------------------------------
# State machine
# --------------------------------------------------------------------------
@dataclass
class QAState:
    history: List[tuple] = field(default_factory=list)  # list[(question, answer)]
    done: bool = False  # set True once sufficiency says "yes"
    turns_used: int = 0  # # turns actually run


def format_history(history) -> str:
    if not history:
        return "(no questions asked yet)"
    return "\n".join(
        f"Q{i + 1}: {q}\nA{i + 1}: {a}" for i, (q, a) in enumerate(history)
    )


def _extract_tag(text: str, tag: str) -> Optional[str]:
    """Pull <tag>...</tag>. Returns None if not present."""
    try:
        return text.split(f"<{tag}>", 1)[1].split(f"</{tag}>", 1)[0].strip()
    except Exception:
        return None


def parse_sufficiency(text: str) -> bool:
    """True if model says it has enough information."""
    val = _extract_tag(text, "enough")
    if val is not None:
        return val.lower().startswith("y")
    # Fallback: look for a bare yes/no near the start.
    t = text.lower().strip()
    if t.startswith("yes"):
        return True
    if t.startswith("no"):
        return False
    return False  # default to "keep asking"


def parse_question(text: str) -> str:
    q = _extract_tag(text, "question")
    if q:
        return q
    # Fallback: take the first non-empty line ending with '?'.
    for line in text.splitlines():
        line = line.strip()
        if line.endswith("?"):
            return line
    return text.strip().split("\n", 1)[0].strip() or "Is the claim true?"


def parse_answer(text: str) -> str:
    a = _extract_tag(text, "answer")
    if a:
        return a
    return text.strip()


# --------------------------------------------------------------------------
# Backends: same vLLM / API plumbing as the other baselines, exposed as a
# callable that takes a list of prompts and returns a list of generations.
# --------------------------------------------------------------------------
def make_vllm_inference(model_id: str, max_tokens: int, max_model_len: int) -> Callable[[List[str]], List[str]]:
    """Load vLLM ONCE; return a function that runs batched chat completions."""
    import torch
    from vllm import LLM, SamplingParams

    print(f"Initializing vLLM with model: {model_id}")
    model = LLM(
        model=model_id,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=torch.cuda.device_count(),
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
    )
    sampling_params = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=max_tokens, seed=42
    )
    tokenizer = model.get_tokenizer()
    # Leave room for the generated output; vLLM rejects prompts whose
    # input + max_tokens exceeds max_model_len.
    max_input_len = max_model_len - max_tokens

    def infer(prompts: List[str]) -> List[str]:
        if not prompts:
            return []
        chat_prompts = [[{"role": "user", "content": p}] for p in prompts]
        safe_indices: List[int] = []
        safe_chats: List[list] = []
        skipped = 0
        for i, chat in enumerate(chat_prompts):
            n_tokens = len(
                tokenizer.apply_chat_template(
                    chat, add_generation_prompt=True, tokenize=True
                )
            )
            if n_tokens <= max_input_len:
                safe_indices.append(i)
                safe_chats.append(chat)
            else:
                skipped += 1
        if skipped:
            print(
                f"[qacheck] skipped {skipped}/{len(prompts)} prompts exceeding "
                f"token budget ({max_input_len}); returning empty generations"
            )
        outputs = model.chat(safe_chats, sampling_params=sampling_params) if safe_chats else []
        results = [""] * len(prompts)
        for idx, out in zip(safe_indices, outputs):
            results[idx] = out.outputs[0].text
        return results

    return infer


def make_api_inference(args) -> Callable[[List[str]], List[str]]:
    from src.baselines.api import build_config, run_api_inference

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

    def infer(prompts: List[str]) -> List[str]:
        if not prompts:
            return []
        return run_api_inference(cfg, prompts)

    return infer


# --------------------------------------------------------------------------
# Multi-turn loop (batched across claims)
# --------------------------------------------------------------------------
def run_qacheck_loop(
    test_data: List[Dict],
    infer: Callable[[List[str]], List[str]],
    max_turns: int,
) -> tuple:
    """Run QACheck across all test claims in lock-step turn-by-turn batches.

    Returns (states, verdict_generations) parallel to test_data.
    """
    n = len(test_data)
    states: List[QAState] = [QAState() for _ in range(n)]

    for turn in range(max_turns):
        # ----- Step A: sufficiency check for not-yet-done claims -----
        active = [i for i, s in enumerate(states) if not s.done]
        if not active:
            print(f"[qacheck] turn {turn}: all claims done early; exiting loop")
            break

        sufficiency_prompts = [
            SUFFICIENCY_PROMPT.format(
                claim=test_data[i]["claim"],
                evidence=test_data[i]["evidence"],
                qa_history=format_history(states[i].history),
            )
            for i in active
        ]
        print(
            f"[qacheck] turn {turn}: sufficiency check on {len(active)} active claims"
        )
        sufficiency_gens = infer(sufficiency_prompts)
        for i, gen in zip(active, sufficiency_gens):
            if parse_sufficiency(gen):
                states[i].done = True

        # ----- Step B: next-question generation for still-active claims -----
        still_active = [i for i in active if not states[i].done]
        if not still_active:
            print(f"[qacheck] turn {turn}: all became sufficient; exiting loop")
            break

        nq_prompts = [
            NEXT_QUESTION_PROMPT.format(
                claim=test_data[i]["claim"],
                evidence=test_data[i]["evidence"],
                qa_history=format_history(states[i].history),
            )
            for i in still_active
        ]
        print(f"[qacheck] turn {turn}: next-question on {len(still_active)} claims")
        nq_gens = infer(nq_prompts)
        questions = [parse_question(g) for g in nq_gens]

        # ----- Step C: answer the new question from evidence -----
        a_prompts = [
            ANSWER_PROMPT.format(
                evidence=test_data[i]["evidence"], question=q
            )
            for i, q in zip(still_active, questions)
        ]
        print(f"[qacheck] turn {turn}: answer on {len(still_active)} claims")
        a_gens = infer(a_prompts)
        answers = [parse_answer(g) for g in a_gens]

        # ----- Commit (q, a) to per-claim history -----
        for i, q, a in zip(still_active, questions, answers):
            states[i].history.append((q, a))
            states[i].turns_used = turn + 1

    # ----- Final verdict for every claim (regardless of how the loop ended) -----
    verdict_prompts = [
        FINAL_VERDICT_PROMPT.format(
            claim=test_data[i]["claim"],
            evidence=test_data[i]["evidence"],
            qa_history=format_history(states[i].history),
        )
        for i in range(n)
    ]
    print(f"[qacheck] final verdict on all {n} claims")
    verdict_gens = infer(verdict_prompts)
    return states, verdict_gens


def process_results(
    test_data: List[Dict], states: List[QAState], verdict_gens: List[str]
) -> List[Dict]:
    out = []
    for sample, state, gen in zip(test_data, states, verdict_gens):
        out.append(
            {
                "id": sample.get("id"),
                "claim": sample.get("claim"),
                "gt_label": sample.get("label"),
                "dataset": sample.get("src"),
                "method": METHOD_NAME,
                "qa_history": state.history,
                "num_turns": state.turns_used,
                "exited_early": state.done,  # True = sufficiency said yes; False = hit max_turns
                "generation": gen,
                "pred_label": extract_verdict_tag(gen),
            }
        )
    return out


def compute_metrics(results: List[Dict]) -> Dict:
    gt = [r["gt_label"] for r in results]
    pred = [r["pred_label"] for r in results]
    metrics = compute_classification_metrics(gt, pred)
    metrics["total_samples"] = len(results)
    metrics["unparsed"] = sum(1 for r in results if r["pred_label"] is None)
    metrics["mean_num_turns"] = (
        sum(r["num_turns"] for r in results) / max(len(results), 1)
    )
    metrics["frac_exited_early"] = (
        sum(1 for r in results if r["exited_early"]) / max(len(results), 1)
    )
    return metrics


def save(results, metrics, output_dir, dataset_name):
    os.makedirs(output_dir, exist_ok=True)
    suffix = f"_{METHOD_NAME}"
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
    parser = argparse.ArgumentParser(description="QACheck baseline runner")
    parser.add_argument(
        "--max_turns",
        type=int,
        default=DEFAULT_MAX_TURNS,
        help=f"Max sub-Q/A turns per claim (default {DEFAULT_MAX_TURNS})",
    )
    parser.add_argument(
        "--backend", choices=["vllm", "api"], default="vllm"
    )
    # vLLM-only
    parser.add_argument("--model", "-m", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max_model_len", type=int, default=32768)
    # API-only
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic"],
        default="openai",
    )
    parser.add_argument("--api_model", default=None)
    parser.add_argument("--api_base_url", default=None)
    parser.add_argument("--api_key_env", default=None)
    parser.add_argument("--max_concurrency", type=int, default=64)
    # Shared
    parser.add_argument("--dataset", "-d", type=str, required=True)
    parser.add_argument("--output_dir", "-o", required=True)
    parser.add_argument("--max_tokens", type=int, default=16768)
    parser.add_argument("--force", "-f", action="store_true")
    args = parser.parse_args()

    dataset_name = args.dataset
    metrics_path = os.path.join(
        args.output_dir, f"{dataset_name}_baseline_metrics_{METHOD_NAME}.json"
    )
    if not args.force and os.path.exists(metrics_path):
        print(f"[SKIP] {metrics_path} already exists. Pass --force to recompute.")
        return

    test_data = load_test_data(args.dataset)

    if args.backend == "vllm":
        infer = make_vllm_inference(args.model, args.max_tokens, args.max_model_len)
    else:
        infer = make_api_inference(args)

    states, verdict_gens = run_qacheck_loop(test_data, infer, args.max_turns)
    results = process_results(test_data, states, verdict_gens)
    metrics = compute_metrics(results)
    metrics["prompt_mode"] = METHOD_NAME
    metrics["backend"] = args.backend
    metrics["max_turns"] = args.max_turns
    if args.backend == "vllm":
        metrics["model"] = args.model
    else:
        metrics["provider"] = args.provider
        metrics["model"] = args.api_model or "default"

    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print("=" * 60)

    save(results, metrics, args.output_dir, dataset_name)


if __name__ == "__main__":
    main()
