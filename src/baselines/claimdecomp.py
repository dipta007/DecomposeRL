"""
ClaimDecomp baseline (Chen et al., NAACL 2022).

Paper:  https://arxiv.org/abs/2205.06938
Repo:   https://github.com/jifan-chen/fact-checking-via-decomposition
Decomposer checkpoint: Factiverse/T5-3B-ClaimDecomp (HF Hub).

Pipeline:
  Stage 1 (decomposition).  Run Chen 2022's supervised T5 decomposer over each
    test claim to obtain a list of yes/no sub-questions. We use the released
    checkpoint `Factiverse/T5-3B-ClaimDecomp` (3B T5; ~6 GB fp16). The model
    card has no input-format spec, so we feed the raw claim text as input
    (matching the most common usage of this artefact) and parse the output as
    newline / numbered sub-questions, taking up to --max_subq questions.

  Stage 2 (answer + aggregate).  Build an aggregator prompt (see
    decomposer/baselines/prompts.py::CLAIMDECOMP_AGGREGATOR_TEMPLATE) that
    presents (claim, evidence, sub-questions) and asks the LLM to answer each
    sub-question from the evidence and emit a Supported/Refuted verdict in a
    <verdict>…</verdict> tag. Reuses the same vLLM / API backends as
    decomposer/baselines/run.py for an apples-to-apples comparison.

Output layout matches the existing prompted-vLLM convention so the loaders in
decomposer/analysis/utils.py find it without change:
    outputs/baseline_{size}_prompted/{dataset}_baseline_metrics_claimdecomp.json
    outputs/baseline_api_{provider}_prompted/{dataset}_baseline_metrics_claimdecomp.json

Adaptation notes (also in ADAPTATIONS.md):
  - Original ClaimDecomp used a supervised RoBERTa veracity classifier on top of
    the decomposed sub-questions and their answers; we replace that with LLM
    aggregation. This is the same closed-evidence simplification we apply to
    Chen-2024 (Complex Claim Verification).
  - We do NOT use ClaimDecomp's gold sub-questions even on the `claimdecomp`
    test split — every dataset runs through the same T5 decomposer for
    consistency. (Using gold sub-questions for one dataset only would create an
    asymmetric comparison.)
"""

import argparse
import json
import os
import re
from typing import Dict, List

import jsonlines

from decomposer.baselines.prompts import build_claimdecomp_aggregator_prompt
from decomposer.eval.utils import compute_classification_metrics


METHOD_NAME = "claimdecomp"
DEFAULT_DECOMPOSER = "Factiverse/T5-3B-ClaimDecomp"


# --------------------------------------------------------------------------
# Shared I/O (mirrors decomposer/baselines/run.py)
# --------------------------------------------------------------------------
def get_dataset_name(test_data_path: str) -> str:
    filename = os.path.basename(test_data_path)
    if filename.startswith("test_") and filename.endswith(".jsonl"):
        return filename[len("test_") : -len(".jsonl")]
    raise ValueError(f"Unexpected test data filename format: {filename}")


def load_test_data(path: str) -> List[Dict]:
    data = []
    with jsonlines.open(path) as reader:
        for line in reader:
            data.append(line)
    print(f"Loaded {len(data)} samples from {path}")
    return data


# --------------------------------------------------------------------------
# Stage 1: decompose every claim with the T5 decomposer.
# --------------------------------------------------------------------------
_QUESTION_LINE = re.compile(
    r"^\s*(?:Q\d*[.:)]?\s*|\d+[.):]\s*|[-*]\s*)?(.+?)\s*$", re.IGNORECASE
)


def _parse_subquestions(text: str, max_subq: int, claim: str = "") -> List[str]:
    """Pull yes/no-ish sub-questions out of a T5 decomposer generation.

    The Factiverse/T5-3B-ClaimDecomp checkpoint emits sub-questions in a noisy
    single-line format: it concatenates 1-N "Did X? Did Y? Did Z?" fragments
    on a single line with no newline separators, sometimes repeats the same
    sub-question, and sometimes echoes the original claim as a final fragment.
    Splitting on '?' is therefore the primary separator, not newlines.

    Heuristics, in order:
      1. Flatten newlines into spaces.
      2. Split on '?' boundaries; reattach '?' to each fragment.
      3. Strip bullet / numbered / "Q:" prefixes and surrounding quotes.
      4. Drop fragments that match the input claim (verbatim echo).
      5. Dedupe (case-insensitive) while preserving order.
      6. Cap at max_subq.
    """
    def _norm(s: str) -> str:
        """Strip quotes/punctuation/whitespace for fuzzy comparison."""
        return re.sub(r"[\s\"'`.,;:!?]+", " ", s.lower()).strip()

    flat = " ".join(text.splitlines()).strip()
    fragments = [p.strip() for p in flat.split("?") if p.strip()]
    seen: set = set()
    out: List[str] = []
    claim_norm = _norm(claim) if claim else ""
    for frag in fragments:
        q = frag + "?"
        m = _QUESTION_LINE.match(q)
        if m:
            q = m.group(1).strip()
        q = q.strip('"').strip("'").strip()
        if not q.endswith("?"):
            q = q + "?"
        key = _norm(q)
        if not key:
            continue
        if claim_norm and key == claim_norm:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(q)
        if len(out) >= max_subq:
            break
    return out


def decompose_all(
    claims: List[str],
    decomposer_id: str,
    max_subq: int,
    batch_size: int,
) -> List[List[str]]:
    """Run the T5 decomposer over every claim; return per-claim sub-question lists."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    print(f"Loading decomposer: {decomposer_id}")
    tokenizer = AutoTokenizer.from_pretrained(decomposer_id)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSeq2SeqLM.from_pretrained(decomposer_id, torch_dtype=dtype).to(device)
    model.eval()

    decompositions: List[List[str]] = []
    debug_n = int(os.environ.get("CLAIMDECOMP_DEBUG_N", "3"))
    print(f"Decomposing {len(claims)} claims (batch_size={batch_size})...")
    for start in range(0, len(claims), batch_size):
        batch = claims[start : start + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=256,
                num_beams=4,
                do_sample=False,
                early_stopping=True,
            )
        texts = tokenizer.batch_decode(out, skip_special_tokens=True)
        for c, t in zip(batch, texts):
            parsed = _parse_subquestions(t, max_subq, claim=c)
            decompositions.append(parsed)
            idx = len(decompositions) - 1
            if idx < debug_n:
                print(f"  [decomp dbg {idx}] claim: {c[:120]}")
                print(f"  [decomp dbg {idx}] raw T5: {t!r}")
                print(f"  [decomp dbg {idx}] parsed ({len(parsed)}): {parsed}")
        if (start // batch_size) % 10 == 0:
            print(f"  decomposed {start + len(batch)}/{len(claims)}")
    n_empty = sum(1 for d in decompositions if not d)
    if n_empty:
        print(f"  WARNING: {n_empty}/{len(decompositions)} claims produced 0 parsed sub-questions")
    avg = sum(len(d) for d in decompositions) / max(len(decompositions), 1)
    print(f"  decomposer summary: mean #sub-questions = {avg:.2f}")

    # Free GPU memory before stage 2 (vLLM may want all of it).
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return decompositions


# --------------------------------------------------------------------------
# Stage 2: answer + aggregate via vLLM or API.
# --------------------------------------------------------------------------
def build_aggregator_prompts(
    test_data: List[Dict], decompositions: List[List[str]]
) -> List[str]:
    return [
        build_claimdecomp_aggregator_prompt(
            claim=s["claim"], evidence=s["evidence"], subquestions=qs
        )
        for s, qs in zip(test_data, decompositions)
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


# --------------------------------------------------------------------------
# Results / metrics (mirrors run.py)
# --------------------------------------------------------------------------
def process_results(
    test_data: List[Dict], generations: List[str], decompositions: List[List[str]]
) -> List[Dict]:
    """Apply Chen 2022's rule-based aggregation per claim.

    The LLM only emits per-sub-Q yes/no/unknown answers; the verdict is the
    Python-side aggregator (decomposer/baselines/prompts.py::aggregate_claimdecomp).
    This is the actual "question aggregation" baseline reported in Chen 2022
    Section 5.3 / Table 6.
    """
    from decomposer.baselines.prompts import aggregate_claimdecomp

    out = []
    for sample, gen, qs in zip(test_data, generations, decompositions):
        verdict, parsed_answers = aggregate_claimdecomp(gen)
        frac_yes = (
            sum(1 for a in parsed_answers if a == "yes")
            / max(sum(1 for a in parsed_answers if a in ("yes", "no")), 1)
            if parsed_answers
            else None
        )
        out.append(
            {
                "id": sample.get("id"),
                "claim": sample.get("claim"),
                "gt_label": sample.get("label"),
                "dataset": sample.get("src"),
                "method": METHOD_NAME,
                "subquestions": qs,
                "num_subquestions": len(qs),
                "parsed_answers": parsed_answers,  # ["yes","no","unknown",...]
                "frac_yes_decisive": frac_yes,
                "generation": gen,
                "pred_label": verdict,
            }
        )
    return out


def compute_metrics(results: List[Dict]) -> Dict:
    gt = [r["gt_label"] for r in results]
    pred = [r["pred_label"] for r in results]
    metrics = compute_classification_metrics(gt, pred)
    metrics["total_samples"] = len(results)
    metrics["unparsed"] = sum(1 for r in results if r["pred_label"] is None)
    metrics["mean_num_subquestions"] = (
        sum(r["num_subquestions"] for r in results) / max(len(results), 1)
    )
    decisive_fracs = [
        r["frac_yes_decisive"] for r in results if r.get("frac_yes_decisive") is not None
    ]
    metrics["mean_frac_yes_decisive"] = (
        sum(decisive_fracs) / len(decisive_fracs) if decisive_fracs else None
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


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ClaimDecomp baseline runner")
    parser.add_argument(
        "--decomposer", default=DEFAULT_DECOMPOSER, help="HF id of the T5 decomposer"
    )
    parser.add_argument(
        "--max_subq",
        type=int,
        default=6,
        help="Max sub-questions kept per claim (Chen 2022 reports ~6 per claim)",
    )
    parser.add_argument("--decomposer_batch_size", type=int, default=8)
    parser.add_argument(
        "--backend",
        choices=["vllm", "api"],
        default="vllm",
        help="Aggregator backend",
    )
    # vLLM-only
    parser.add_argument("--model", "-m", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max_model_len", type=int, default=32768)
    # API-only
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "custom", "anthropic_native", "umbc"],
        default="openai",
    )
    parser.add_argument("--api_model", default=None)
    parser.add_argument("--api_base_url", default=None)
    parser.add_argument("--api_key_env", default=None)
    parser.add_argument("--max_concurrency", type=int, default=64)
    # Shared
    parser.add_argument("--test_data", "-d", required=True)
    parser.add_argument("--output_dir", "-o", required=True)
    parser.add_argument("--max_tokens", type=int, default=16768)
    parser.add_argument("--force", "-f", action="store_true")
    args = parser.parse_args()

    dataset_name = get_dataset_name(args.test_data)
    metrics_path = os.path.join(
        args.output_dir, f"{dataset_name}_baseline_metrics_{METHOD_NAME}.json"
    )
    if not args.force and os.path.exists(metrics_path):
        print(f"[SKIP] {metrics_path} already exists. Pass --force to recompute.")
        return

    test_data = load_test_data(args.test_data)
    claims = [s["claim"] for s in test_data]

    # Stage 1: decompose
    decompositions = decompose_all(
        claims=claims,
        decomposer_id=args.decomposer,
        max_subq=args.max_subq,
        batch_size=args.decomposer_batch_size,
    )

    # Stage 2: aggregate
    prompts = build_aggregator_prompts(test_data, decompositions)
    if args.backend == "vllm":
        generations = run_vllm(args.model, prompts, args.max_tokens, args.max_model_len)
    else:
        generations = run_api(args, prompts)

    results = process_results(test_data, generations, decompositions)
    metrics = compute_metrics(results)
    metrics["prompt_mode"] = METHOD_NAME
    metrics["backend"] = args.backend
    metrics["decomposer"] = args.decomposer
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
