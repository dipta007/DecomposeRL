"""Decompose claims for augmented long-evidence samples (step 8 → step 9).

Reads   data/combined/step_8/train.jsonl
Writes  data/combined/step_9/train.jsonl   (all records with decompositions filled in)
        data/combined/step_9/test_*.jsonl   (copied from step_8)

Only processes records that are missing `num_of_questions` or `decomposed_questions`.
Records that already have these fields (from step_7) are passed through unchanged.
Uses the same GPT decomposition logic and shared cache as decompose_claims.py.

Run:  PYTHONPATH=. uv run decomposer/data_process/decompose_augmented.py
"""

import json
import os
import shutil
import sys
import logging
from pathlib import Path

from joblib import Parallel, delayed
from openai import OpenAI
from pydantic import BaseModel
from retry import retry
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from decomposer.prompts import (
    DECOMPOSE_QUESTIONS_PROMPT_TEMPLATE,
    NUMBER_OF_QUESTIONS_PROMPT_TEMPLATE,
)

MODEL = "gpt-5-mini"
N_JOBS = 64
CACHE_DIR = "data/combined/cache"
NUM_QUESTIONS_CACHE_PATH = os.path.join(CACHE_DIR, "num_questions_cache.json")
DECOMPOSED_QUESTIONS_CACHE_PATH = os.path.join(
    CACHE_DIR, "decomposed_questions_cache.json"
)

INPUT_DIR = Path("data/combined/step_8")
OUTPUT_DIR = Path("data/combined/step_9")

# gpt-5-mini pricing (USD per million tokens)
PRICE_INPUT_PER_M = 0.25
PRICE_OUTPUT_PER_M = 2.00

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


# --- Structured output schemas ---


class NumQuestionsResponse(BaseModel):
    num_of_questions: int


class DecomposeQuestionsResponse(BaseModel):
    questions: list[str]


def compute_cost(usage: dict) -> float:
    """Compute USD cost from token usage."""
    input_cost = usage["input_tokens"] * PRICE_INPUT_PER_M / 1_000_000
    output_cost = usage["output_tokens"] * PRICE_OUTPUT_PER_M / 1_000_000
    return input_cost + output_cost


@retry(tries=-1, delay=2, backoff=2, max_delay=60, logger=logger)
def call_gpt_structured(
    client: OpenAI, prompt: str, text_format, tag: str, record_id: str
):
    """Call gpt-5-mini via Responses API with structured output and low reasoning effort."""
    logger.debug("[%s] %s | Sending request...", record_id, tag)

    response = client.responses.parse(
        model=MODEL,
        input=[{"role": "user", "content": prompt}],
        reasoning={"effort": "low"},
        text_format=text_format,
    )

    parsed = response.output_parsed
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "total_tokens": response.usage.total_tokens,
    }
    cost = compute_cost(usage)

    logger.debug("[%s] %s | Tokens: %s | Cost: $%.6f", record_id, tag, usage, cost)

    return parsed, usage


def get_num_questions(record: dict) -> tuple[int | None, dict]:
    """Get num_of_questions for a single record."""
    client = OpenAI()
    record_id = record.get("id", "unknown")
    empty_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    prompt = NUMBER_OF_QUESTIONS_PROMPT_TEMPLATE.format(
        evidence_doc=record["evidence"], claim=record["claim"]
    )
    try:
        parsed, usage = call_gpt_structured(
            client, prompt, NumQuestionsResponse, "NUM_QUESTIONS", record_id
        )
        return parsed.num_of_questions, usage
    except Exception as e:
        logger.error("[%s] NUM_QUESTIONS failed after retries: %s", record_id, e)
        return None, empty_usage


def get_decomposed_questions(record: dict) -> tuple[list[str], dict]:
    """Get decomposed_questions for a single record."""
    client = OpenAI()
    record_id = record.get("id", "unknown")
    empty_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    prompt = DECOMPOSE_QUESTIONS_PROMPT_TEMPLATE.format(
        evidence_doc=record["evidence"], claim=record["claim"]
    )
    try:
        parsed, usage = call_gpt_structured(
            client, prompt, DecomposeQuestionsResponse, "DECOMPOSE", record_id
        )
        return parsed.questions, usage
    except Exception as e:
        logger.error("[%s] DECOMPOSE failed after retries: %s", record_id, e)
        return [], empty_usage


def load_cache(cache_path: str) -> dict:
    """Load a JSON cache file. Returns dict keyed by claim text."""
    if not os.path.exists(cache_path):
        return {}
    with open(cache_path, "r") as f:
        return json.load(f)


def save_cache(cache: dict, cache_path: str):
    """Save cache dict to a JSON file atomically to prevent corruption."""
    tmp_path = cache_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(cache, f)
    os.replace(tmp_path, cache_path)


def run_task(task_name, task_fn, todo, cache, cache_path):
    """Run a task via joblib Parallel, saving cache to disk after each completion."""
    task_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    if not todo:
        return task_usage

    results = Parallel(n_jobs=N_JOBS, prefer="threads", return_as="generator")(
        delayed(task_fn)(record) for record in todo
    )

    for record, (result, usage) in zip(
        todo, tqdm(results, total=len(todo), desc=task_name, unit="rec")
    ):
        cache[record["claim"]] = result
        for k in task_usage:
            task_usage[k] += usage.get(k, 0)
        save_cache(cache, cache_path)

    return task_usage


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load train data
    train_path = INPUT_DIR / "train.jsonl"
    with open(train_path, "r") as f:
        records = [json.loads(line) for line in f if line.strip()]
    logger.info("Loaded %d records from %s", len(records), train_path)

    # 2. Identify records missing decompositions
    needs_decomp = []
    already_done = 0
    seen_claims = set()
    for r in records:
        has_num = r.get("num_of_questions") is not None
        has_decomp = r.get("decomposed_questions") is not None
        if has_num and has_decomp:
            already_done += 1
        elif r["claim"] not in seen_claims:
            needs_decomp.append(r)
            seen_claims.add(r["claim"])

    logger.info(
        "Already decomposed: %d, need decomposition: %d unique claims",
        already_done,
        len(needs_decomp),
    )

    if not needs_decomp:
        logger.info("All records already have decompositions — copying as-is")
    else:
        # 3. Load shared caches
        num_cache = load_cache(NUM_QUESTIONS_CACHE_PATH)
        decompose_cache = load_cache(DECOMPOSED_QUESTIONS_CACHE_PATH)
        logger.info(
            "Caches loaded — num_questions: %d, decomposed_questions: %d",
            len(num_cache),
            len(decompose_cache),
        )

        grand_total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        # 4. Task 1: num_of_questions
        num_todo = [r for r in needs_decomp if r["claim"] not in num_cache]
        logger.info(
            "num_of_questions — cached: %d, todo: %d",
            len(needs_decomp) - len(num_todo),
            len(num_todo),
        )
        task_usage = run_task(
            "num_of_questions",
            get_num_questions,
            num_todo,
            num_cache,
            NUM_QUESTIONS_CACHE_PATH,
        )
        for k in grand_total:
            grand_total[k] += task_usage[k]

        # 5. Task 2: decomposed_questions
        decompose_todo = [r for r in needs_decomp if r["claim"] not in decompose_cache]
        logger.info(
            "decomposed_questions — cached: %d, todo: %d",
            len(needs_decomp) - len(decompose_todo),
            len(decompose_todo),
        )
        task_usage = run_task(
            "decomposed_questions",
            get_decomposed_questions,
            decompose_todo,
            decompose_cache,
            DECOMPOSED_QUESTIONS_CACHE_PATH,
        )
        for k in grand_total:
            grand_total[k] += task_usage[k]

        # 6. Merge results back into records
        for record in records:
            claim = record["claim"]
            if record.get("num_of_questions") is None and claim in num_cache:
                record["num_of_questions"] = num_cache[claim]
            if record.get("decomposed_questions") is None and claim in decompose_cache:
                record["decomposed_questions"] = decompose_cache[claim]
            if "decomposer_model" not in record:
                record["decomposer_model"] = MODEL

        grand_cost = compute_cost(grand_total)
        logger.info(
            "New tokens: %s | New cost: $%.4f",
            grand_total,
            grand_cost,
        )

    # 7. Write train output
    train_out = OUTPUT_DIR / "train.jsonl"
    with open(train_out, "w") as f:
        for record in tqdm(records, desc="Writing train"):
            f.write(json.dumps(record) + "\n")
    logger.info("Wrote %d records to %s", len(records), train_out)

    # 8. Copy test files from step_8
    test_files = sorted(
        f for f in INPUT_DIR.glob("*.jsonl") if f.name != "train.jsonl"
    )
    for test_path in tqdm(test_files, desc="Copying test files"):
        shutil.copy2(test_path, OUTPUT_DIR / test_path.name)
    logger.info("Copied %d test files to %s", len(test_files), OUTPUT_DIR)


if __name__ == "__main__":
    main()
