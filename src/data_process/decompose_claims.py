import json
import os
import sys
import logging
from pathlib import Path
from glob import glob

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
        # max_output_tokens=4096,
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


def process_file(
    input_path: str, output_path: str, num_cache: dict, decompose_cache: dict
):
    """Process all records in a single jsonl file with standalone cache files."""
    logger.info("Processing %s -> %s", input_path, output_path)

    with open(input_path, "r") as f:
        records = [json.loads(line) for line in f if line.strip()]

    logger.info("Loaded %d records from %s", len(records), input_path)

    file_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    # Deduplicate by claim text so each unique claim is only processed once
    seen_claims = set()
    unique_records = []
    for r in records:
        if r["claim"] not in seen_claims:
            seen_claims.add(r["claim"])
            unique_records.append(r)

    # --- Task 1: num_of_questions ---
    num_todo = [r for r in unique_records if r["claim"] not in num_cache]
    logger.info(
        "num_of_questions — cached: %d, todo: %d",
        len(unique_records) - len(num_todo),
        len(num_todo),
    )
    task_usage = run_task(
        "num_of_questions",
        get_num_questions,
        num_todo,
        num_cache,
        NUM_QUESTIONS_CACHE_PATH,
    )
    for k in file_usage:
        file_usage[k] += task_usage[k]

    # --- Task 2: decomposed_questions ---
    decompose_todo = [r for r in unique_records if r["claim"] not in decompose_cache]
    logger.info(
        "decomposed_questions — cached: %d, todo: %d",
        len(unique_records) - len(decompose_todo),
        len(decompose_todo),
    )
    task_usage = run_task(
        "decomposed_questions",
        get_decomposed_questions,
        decompose_todo,
        decompose_cache,
        DECOMPOSED_QUESTIONS_CACHE_PATH,
    )
    for k in file_usage:
        file_usage[k] += task_usage[k]

    # Write final output by merging input records with cached results (keyed by claim)
    with open(output_path, "w") as f:
        for record in records:
            result = dict(record)
            claim = record["claim"]
            if claim in num_cache:
                result["num_of_questions"] = num_cache[claim]
            if claim in decompose_cache:
                result["decomposed_questions"] = decompose_cache[claim]
            result["decomposer_model"] = MODEL
            f.write(json.dumps(result) + "\n")

    file_cost = compute_cost(file_usage)
    logger.info(
        "File %s | New tokens: %s | New cost: $%.4f",
        os.path.basename(input_path),
        file_usage,
        file_cost,
    )
    logger.info("Saved %d records to %s", len(records), output_path)
    return file_usage


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    input_dir = project_root / "data" / "combined" / "step_4"
    output_dir = project_root / "data" / "combined" / "step_5"
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(glob(str(input_dir / "*.jsonl")))
    if not input_files:
        logger.error("No .jsonl files found in %s", input_dir)
        return

    logger.info("Found %d files to process", len(input_files))

    # Load caches once, shared across all files (keyed by claim text)
    num_cache = load_cache(NUM_QUESTIONS_CACHE_PATH)
    decompose_cache = load_cache(DECOMPOSED_QUESTIONS_CACHE_PATH)
    logger.info(
        "Caches loaded — num_questions: %d, decomposed_questions: %d",
        len(num_cache),
        len(decompose_cache),
    )

    grand_total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    for input_path in input_files:
        filename = os.path.basename(input_path)
        output_path = str(output_dir / filename)
        file_usage = process_file(input_path, output_path, num_cache, decompose_cache)
        for k in grand_total:
            grand_total[k] += file_usage.get(k, 0)

    grand_cost = compute_cost(grand_total)
    logger.info("=== All files processed ===")
    logger.info("Grand total tokens: %s", grand_total)
    logger.info(
        "Grand total cost: $%.4f (input: $%.4f, output: $%.4f)",
        grand_cost,
        grand_total["input_tokens"] * PRICE_INPUT_PER_M / 1_000_000,
        grand_total["output_tokens"] * PRICE_OUTPUT_PER_M / 1_000_000,
    )


if __name__ == "__main__":
    main()
