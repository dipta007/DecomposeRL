import re
import os
import json
import random
import logging
import asyncio
import uuid
import threading
from functools import lru_cache
from hashlib import sha256
from datetime import datetime
from typing import List, Dict, Optional, Any

import numpy as np
import torch
from openai import AsyncOpenAI
from vendi_score import vendi
from tenacity import (
    retry,
    wait_exponential,
    before_sleep_log,
)

from decomposer.prompts import (
    ATOMICITY_CHECKLIST_PROMPT_TEMPLATE,
    QUESTION_CHECKER_PROMPT_TEMPLATE,
    ANSWER_CHECKER_PROMPT_TEMPLATE,
    COVERAGE_PROMPT_TEMPLATE,
)
from decomposer.unsloth.format_reward import get_format_reward

logger = logging.getLogger(__name__)


def _majority_vote(verdicts: List[str]) -> str:
    """Return the most common verdict. Ties broken alphabetically for determinism."""
    from collections import Counter

    counts = Counter(verdicts)
    max_count = max(counts.values())
    candidates = sorted(v for v, c in counts.items() if c == max_count)
    return candidates[0]


# Constants
JUDGE_MODEL_ID = "Qwen/Qwen3-32B"
JUDGE_EMBEDDING_MODEL_ID = "Qwen/Qwen3-Embedding-8B"
RETRY_MIN_WAIT = 1
RETRY_MAX_WAIT = 2

JUDGE_PORT = os.getenv("JUDGE_PORT", "8000")
async_client = AsyncOpenAI(
    api_key="EMPTY", base_url=f"http://localhost:{JUDGE_PORT}/v1", timeout=1200
)
async_emb_client = AsyncOpenAI(
    api_key="EMPTY", base_url="http://localhost:8004/v1", timeout=1200
)

logger.info(f"Judge Model ID: {JUDGE_MODEL_ID}")

# Lock for thread-safe file operations
_reward_save_lock = threading.Lock()

atomicity_checklist_cache_dir = ".cache/test/atomicity_checklist"
question_answerable_cache_dir = ".cache/test/question_answerable"
answer_correctness_cache_dir = ".cache/test/answer_correctness"
coverage_cache_dir = ".cache/test/coverage"
rewards_cache_dir = ".cache/test/rewards"

os.makedirs(atomicity_checklist_cache_dir, exist_ok=True)
os.makedirs(question_answerable_cache_dir, exist_ok=True)
os.makedirs(answer_correctness_cache_dir, exist_ok=True)
os.makedirs(coverage_cache_dir, exist_ok=True)
os.makedirs(rewards_cache_dir, exist_ok=True)


from decomposer.unsloth.llm_cache import LLMCache

_llm_cache = LLMCache.from_env()


@retry(
    wait=wait_exponential(multiplier=1, min=RETRY_MIN_WAIT, max=RETRY_MAX_WAIT),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
async def _call_embedding_model(input: list[str]) -> list:
    """Call embedding model with retry logic."""
    response = await async_emb_client.embeddings.create(
        input=input, encoding_format="float", model=JUDGE_EMBEDDING_MODEL_ID
    )
    return response.data


async def _call_judge_model(
    prompt: str,
    cache_dir: str,
    cache_data: Dict[str, Any],
    response_tag: str = "answer",
    max_tokens: Optional[int] = None,
) -> Optional[str]:
    """
    Common helper for calling the judge model with caching and error handling.

    Args:
        prompt: The prompt to send to the model
        cache_dir: Full path to cache directory for storing responses
        cache_data: Data to store in the cache file
        response_tag: The XML tag to extract from response ("answer" or "verdict")

    Returns:
        The extracted response content, or None if the call failed
    """

    # Build the full config — all inputs that determine the LLM output
    configs = {
        "model": JUDGE_MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "seed": 42,
    }
    if max_tokens:
        configs["max_tokens"] = max_tokens

    # Cache key includes configs + response_tag (same response parsed differently)
    cache_key = sha256(
        json.dumps(
            {"configs": configs, "response_tag": response_tag}, sort_keys=True
        ).encode()
    ).hexdigest()

    cached = _llm_cache.get(cache_key)
    if cached is not None:
        return cached

    _non_stop_attempts = 0

    @retry(
        wait=wait_exponential(multiplier=1, min=RETRY_MIN_WAIT, max=RETRY_MAX_WAIT),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _call_with_retry():
        nonlocal _non_stop_attempts
        response = await async_client.chat.completions.create(**configs)
        finish_reason = response.choices[0].finish_reason
        msg = response.choices[0].message

        # If finish_reason is not "stop", only retry twice then give up
        if finish_reason != "stop":
            _non_stop_attempts += 1
            if _non_stop_attempts > 0:
                logger.warning(
                    f"Judge model finished with reason={finish_reason} after 2 retries, giving up"
                )
                return None
            raise ValueError(
                f"Judge model finished with reason={finish_reason}, retrying "
                f"({_non_stop_attempts}/2)"
            )

        if msg.content is None:
            reasoning = getattr(msg, "reasoning", None) or getattr(
                msg, "reasoning_content", None
            )
            if reasoning:
                raise ValueError(
                    f"Judge model returned content=None but has reasoning "
                    f"({len(reasoning)} chars) — likely exhausted max_tokens on thinking. "
                    f"Retrying."
                )
            else:
                raise ValueError(
                    "Judge model returned content=None with no reasoning — unexpected empty response. "
                    "Retrying."
                )
        return response

    response = None
    try:
        response = await _call_with_retry()
        out = (
            response.choices[0]
            .message.content.split(f"<{response_tag}>")[1]
            .split(f"</{response_tag}>")[0]
            .strip()
        )

        # Store in SQLite cache
        _llm_cache.set(cache_key, out)

        # Save to disk for analysis or later training set
        cache_data["response"] = response.model_dump()
        cache_data["judge_model_id"] = JUDGE_MODEL_ID
        cache_data["prompt"] = prompt
        cache_data["extracted_response"] = out

        with _reward_save_lock:
            with open(
                f"{cache_dir}/unsloth_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}.json",
                "w",
            ) as f:
                json.dump(cache_data, f, indent=2)

        return out
    except Exception as e:
        raw_content = (
            (response.choices[0].message.content or "")[:500]
            if response and response.choices
            else "NO_RESPONSE"
        )
        logger.error(
            f"Error parsing {response_tag} from response: {e} | cache_dir={cache_dir} | raw_content={raw_content}"
        )
        return None


@lru_cache(maxsize=256)
def extract_qa_pairs(generation: str) -> List[Dict[str, str]]:
    """Extract QA pairs from generation. Results are cached for efficiency."""
    # Extract all questions and answers with positions
    question_pattern = r"<question>(.*?)</question>"
    answer_pattern = r"<answer>(.*?)</answer>"

    questions = [
        (m.group(1).strip(), m.start())
        for m in re.finditer(question_pattern, generation, re.DOTALL)
    ]
    answers = [
        (m.group(1).strip(), m.start())
        for m in re.finditer(answer_pattern, generation, re.DOTALL)
    ]

    # Pair questions with their immediately following answers
    qa_pairs = []

    for i, (q_text, q_pos) in enumerate(questions):
        # Find the answer that comes immediately after this question
        matching_answer = None
        for a_text, a_pos in answers:
            if a_pos > q_pos:
                matching_answer = a_text
                break

        qa_pairs.append(
            {
                "index": i + 1,
                "question": q_text,
                "answer": matching_answer if matching_answer else "[No answer found]",
            }
        )

    return qa_pairs


def verification_reward(generation, label) -> tuple[float, str]:
    max_possible_reward = 1.0
    min_possible_reward = 0.0

    pred = "UNKNOWN".lower()
    if "<verification>" not in generation or "</verification>" not in generation:
        reward = min_possible_reward
    else:
        gen_response = (
            generation.split("<verification>")[1].split("</verification>")[0].strip()
        )
        verified = gen_response.lower() == label.lower()
        pred = gen_response.lower()
        reward = max_possible_reward if verified else min_possible_reward

    return reward, pred


ATOMICITY_CRITERIA = [
    "is_question",
    "single_focus",
    "no_conjunctions",
    "verifiable",
    "grounded",
]


async def atomicity_checklist_reward(
    generation, claim, questions
) -> tuple[float, List[float]]:
    """Binary checklist atomicity: evaluates each question on 4 binary criteria."""

    async def _get_checklist_score(claim, question) -> float:
        result = await _call_judge_model(
            prompt=ATOMICITY_CHECKLIST_PROMPT_TEMPLATE.format(
                claim=claim, question=question
            ),
            cache_dir=atomicity_checklist_cache_dir,
            cache_data={"claim": claim, "question": question},
            max_tokens=4096,
        )
        if result is None:
            return 0.0

        # Parse "single_focus:YES\nno_conjunctions:NO\n..." format
        passed = 0
        found = [False] * len(ATOMICITY_CRITERIA)
        for criterion in ATOMICITY_CRITERIA:
            for line in result.strip().splitlines():
                line = line.strip().lower()
                if line.startswith(criterion):
                    if "yes" in line:
                        passed += 1
                    found[ATOMICITY_CRITERIA.index(criterion)] = True
                    break
        not_found = [c for i, c in enumerate(ATOMICITY_CRITERIA) if not found[i]]
        if not_found:
            logger.warning(
                f"Could not find all criteria in atomicity checklist response for question: {question}. Missing: {not_found}. Full response: {result}"
            )
        return passed / len(ATOMICITY_CRITERIA)

    tasks = [_get_checklist_score(claim, question) for question in questions]
    curr_scores = await asyncio.gather(*tasks)
    all_scores = list(curr_scores)
    avg_score = sum(curr_scores) / len(curr_scores) if curr_scores else 0.0
    return avg_score, all_scores


def good_number_of_questions_reward(
    generation, gen_num_of_questions, gt_num_of_questions
):
    gt_num_of_questions = max(gt_num_of_questions, 1)
    ratio = gen_num_of_questions / gt_num_of_questions

    if ratio <= 0:
        return 0.0
    elif ratio <= 1.0:
        # Under-generating: linear ramp [0 → 1]
        return ratio
    elif ratio <= 2.0:
        # Mild over-generating (up to 2x): linear decay [1 → 0]
        return 2.0 - ratio
    else:
        # More than 2x GT: hard zero
        return 0.0


async def maximal_margin_classifier_reward(generation, questions):
    if len(questions) <= 1:
        return -1.0

    def get_inst(query):
        return f"Instruct: Given a question, retrieve the similar questions that answer the same question.\nQuery:{query}"

    queries = [get_inst(question) for question in questions]
    documents = [question for question in questions]

    query_emb, document_emb = await asyncio.gather(
        _call_embedding_model(queries),
        _call_embedding_model(documents),
    )

    query_emb = torch.tensor([q.embedding for q in query_emb])
    document_emb = torch.tensor([d.embedding for d in document_emb])

    similarity = query_emb @ document_emb.T

    mmr = 0.0
    for i in range(len(questions)):
        curr_similarity = similarity[i, :i]
        max_prev = curr_similarity.max().item() if len(curr_similarity) > 0 else 0.0
        mmr += max_prev
    mmr /= len(questions)
    return -mmr


async def vendi_diversity_reward(generation, questions):
    """Diversity reward using Vendi Score (effective number of unique questions).

    Returns VS/n normalized to (0, 1], where 1.0 means all questions are
    maximally diverse and 1/n means all questions are identical.
    """
    if len(questions) <= 1:
        return 0.0  # single question is not diverse

    emb_data = await _call_embedding_model(questions)
    emb = torch.tensor([e.embedding for e in emb_data])
    emb = torch.nn.functional.normalize(emb, dim=1)

    # Cosine similarity matrix (n x n, with 1s on diagonal)
    K = (emb @ emb.T).clamp(-1.0, 1.0).cpu().numpy()

    n = len(questions)
    vs = vendi.score_K(K)
    return float(vs / n)  # normalized to (0, 1]


async def question_answerable_reward(
    generation, document, questions, answers
) -> tuple[float, List[float]]:
    """Reward based on whether questions can be answered from the document."""

    async def _get_question_answerable_score(document, question, answer) -> float:
        is_idk = answer.strip(" .").lower() == "i don't know" or answer.strip() == ""

        result = await _call_judge_model(
            prompt=QUESTION_CHECKER_PROMPT_TEMPLATE.format(
                document=document, question=question
            ),
            cache_dir=question_answerable_cache_dir,
            cache_data={"document": document, "question": question},
            max_tokens=4096,
        )
        if result is None:
            return 0.0
        try:
            question_is_answerable = int(result) == 1
        except Exception as e:
            logger.error(f"Error parsing question answerable score: {result}, {e}")
            return 0.0

        if is_idk:
            # IDK is correct when question CAN'T be answered from the doc
            return 0.0 if question_is_answerable else 1.0
        else:
            # Substantive answer is correct when question CAN be answered
            return 1.0 if question_is_answerable else 0.0

    tasks = [
        _get_question_answerable_score(document, question, answer)
        for question, answer in zip(questions, answers)
    ]
    curr_scores = await asyncio.gather(*tasks)
    all_scores = list(curr_scores)
    avg_score = sum(curr_scores) / len(curr_scores) if curr_scores else 0.0
    return avg_score, all_scores


async def answer_correctness_reward(
    generation, document, qa_pairs
) -> tuple[float, List[float]]:
    """Reward based on whether answers are correct according to the document."""

    async def _get_answer_correctness_score(document, question, answer) -> float:
        sentence = f"Q: {question}\nA: {answer}"
        result = await _call_judge_model(
            prompt=ANSWER_CHECKER_PROMPT_TEMPLATE.format(
                document=document, sentence=sentence
            ),
            cache_dir=answer_correctness_cache_dir,
            cache_data={"document": document, "question": question, "answer": answer},
            max_tokens=4096,
        )
        if result is None:
            return 0.0
        try:
            return 1.0 if int(result) == 1 else 0.0
        except Exception as e:
            logger.error(f"Error parsing answer correctness score: {result}, {e}")
            return 0.0

    tasks = [
        _get_answer_correctness_score(document, pair["question"], pair["answer"])
        for pair in qa_pairs
    ]
    curr_scores = await asyncio.gather(*tasks)
    all_scores = list(curr_scores)
    avg_score = sum(curr_scores) / len(curr_scores) if curr_scores else 0.0
    return avg_score, all_scores


async def necessity_saliency_reward(
    generation, claim, answers, gt_label=None, aggregation="mean"
) -> tuple[float, List[float]]:
    """Leave-one-out saliency: a question is salient if removing it hurts verification.

    When gt_label is provided, uses the original 4-way scoring against the ground truth.
    When gt_label is None (label-free / relative necessity), compares each LOO verdict
    against the full-set verdict instead.
    """
    if len(answers) == 0:
        return 0.0, []

    async def _run_coverage(answers_subset) -> Optional[str]:
        """Run coverage judge and return the predicted label string."""
        formatted_answers = "\n".join([f"- {answer}" for answer in answers_subset])
        result = await _call_judge_model(
            prompt=COVERAGE_PROMPT_TEMPLATE.format(
                claim=claim, answers=formatted_answers
            ),
            cache_dir=coverage_cache_dir,
            cache_data={
                "claim": claim,
                "answers": formatted_answers,
                "gt_label": gt_label,
                "type": "necessity_saliency",
            },
            response_tag="verdict",
            max_tokens=4096 * 2,
        )
        if result is None:
            return None
        verdict = result.lower()
        if "supported" in verdict:
            return "supported"
        elif "refuted" in verdict:
            return "refuted"
        else:
            return "not enough information"

    if gt_label is None:
        # --- Label-free / relative necessity mode ---
        if len(answers) == 1:
            # Single question is necessary by definition if full verdict exists
            return 1.0, [1.0]

        # Multiple questions: run full coverage + N leave-one-out coverages
        tasks = [_run_coverage(answers)]
        for i in range(len(answers)):
            loo_answers = answers[:i] + answers[i + 1 :]
            tasks.append(_run_coverage(loo_answers))

        results = await asyncio.gather(*tasks)
        full_label = results[0]

        if full_label is None:
            return 0.0, [0.0] * len(answers)

        per_question_scores = []
        for i in range(len(answers)):
            loo_label = results[i + 1]
            if loo_label is None:
                # Judge failed for this LOO call → can't determine necessity
                per_question_scores.append(0.0)
            elif loo_label != full_label:
                # Removing this Q changed the verdict → necessary
                per_question_scores.append(1.0)
            else:
                # Removing this Q didn't change the verdict → redundant
                per_question_scores.append(0.0)

        if aggregation == "min":
            score = min(per_question_scores) if per_question_scores else 0.0
        else:
            score = (
                sum(per_question_scores) / len(per_question_scores)
                if per_question_scores
                else 0.0
            )
        return score, per_question_scores

    # --- Original mode: gt_label is provided ---
    if len(answers) == 1:
        # Single question: if full coverage is correct, it's necessary by definition
        full_label = await _run_coverage(answers)
        full_correct = (full_label == gt_label.lower()) if full_label else False
        score = 1.0 if full_correct else 0.0
        return score, [score]

    # Run full coverage + N leave-one-out coverages concurrently
    tasks = [_run_coverage(answers)]
    for i in range(len(answers)):
        loo_answers = answers[:i] + answers[i + 1 :]
        tasks.append(_run_coverage(loo_answers))

    results = await asyncio.gather(*tasks)

    full_label = results[0]
    full_correct = (full_label == gt_label.lower()) if full_label else False

    per_question_scores = []
    for i in range(len(answers)):
        loo_label = results[i + 1]
        loo_correct = (loo_label == gt_label.lower()) if loo_label else False

        if full_correct and not loo_correct:
            # Removing this Q broke the verdict → necessary
            per_question_scores.append(1.0)
        elif full_correct and loo_correct:
            # Removing didn't matter → redundant but harmless
            per_question_scores.append(0.5)
        elif not full_correct and loo_correct:
            # Removing this Q FIXED the verdict → harmful
            per_question_scores.append(-1.0)
        else:
            # Full wrong, still wrong without this Q → neutral
            per_question_scores.append(0.0)

    if aggregation == "min":
        score = min(per_question_scores) if per_question_scores else 0.0
    else:
        score = (
            sum(per_question_scores) / len(per_question_scores)
            if per_question_scores
            else 0.0
        )
    return score, per_question_scores


async def coverage_predict(generation, claim, answers) -> Optional[str]:
    """Predict the verification verdict for a claim given QA answers. Returns label string or None."""
    formatted_answers = "\n".join([f"- {answer}" for answer in answers])

    result = await _call_judge_model(
        prompt=COVERAGE_PROMPT_TEMPLATE.format(claim=claim, answers=formatted_answers),
        cache_dir=coverage_cache_dir,
        cache_data={"claim": claim, "answers": formatted_answers},
        response_tag="verdict",
        max_tokens=4096 * 2,
    )
    if result is None:
        return None

    verdict = result.lower()
    if "supported" in verdict:
        return "supported"
    elif "refuted" in verdict:
        return "refuted"
    else:
        return "not enough information"


async def coverage_evaluate(generation, claim, answers, gt_label) -> float:
    """Reward based on whether the answers lead to the correct verdict for the claim."""
    predicted_label = await coverage_predict(generation, claim, answers)
    if predicted_label is None:
        return 0.0
    score = 1.0 if predicted_label == gt_label.lower() else 0.0
    return score


async def joint_quality_reward(
    generation, claim, document, qa_pairs, gt_label=None
) -> tuple[float, List[float]]:
    """Joint saliency-atomicity-coverage interaction reward.

    per_question_quality_i = is_answerable(q_i) * is_atomic(q_i) * answer_is_correct(q_i)
    decomposition_reward = mean(per_question_quality_i) * coverage(all_answers)  [if JOINT_COVERAGE=1]
    """
    questions = [pair["question"] for pair in qa_pairs]
    answers = [pair["answer"] for pair in qa_pairs]

    if len(qa_pairs) == 0:
        return 0.0, []

    use_joint_coverage = os.getenv("JOINT_COVERAGE", "0") == "1"

    if use_joint_coverage and gt_label is None:
        raise ValueError(
            "JOINT_COVERAGE=1 requires gt_label but got None. "
            "Set JOINT_COVERAGE=0 for unsupervised samples (SUPERVISION_RATE < 1.0)."
        )

    # Only compute coverage if we'll use it AND we have a label
    if use_joint_coverage and gt_label is not None:
        (
            (answerable_avg, answerable_scores),
            (atomicity_avg, atomicity_scores),
            (correctness_avg, correctness_scores),
            coverage_score,
        ) = await asyncio.gather(
            question_answerable_reward(generation, document, questions, answers),
            atomicity_checklist_reward(generation, claim, questions),
            answer_correctness_reward(generation, document, qa_pairs),
            coverage_evaluate(generation, claim, answers, gt_label),
        )
    else:
        # Skip coverage computation — either not needed or no gt_label
        (
            (answerable_avg, answerable_scores),
            (atomicity_avg, atomicity_scores),
            (correctness_avg, correctness_scores),
        ) = await asyncio.gather(
            question_answerable_reward(generation, document, questions, answers),
            atomicity_checklist_reward(generation, claim, questions),
            answer_correctness_reward(generation, document, qa_pairs),
        )
        coverage_score = 0.0

    # per_question_quality_i = answerable_i * atomic_i * correct_i
    # For IDK answers, skip correctness (no answer to check)
    per_q_quality = []
    for i in range(len(qa_pairs)):
        answer = qa_pairs[i]["answer"]
        is_idk = answer.strip(" .").lower() == "i don't know" or answer.strip() == ""
        if is_idk:
            q_quality = answerable_scores[i] * atomicity_scores[i]
        else:
            q_quality = (
                answerable_scores[i] * atomicity_scores[i] * correctness_scores[i]
            )
        per_q_quality.append(q_quality)

    mean_quality = sum(per_q_quality) / len(per_q_quality)
    if use_joint_coverage:
        reward = mean_quality * coverage_score
    else:
        reward = mean_quality

    return reward, per_q_quality


def save_reward_data(kwargs=None, **kw) -> None:
    """Save comprehensive reward data to cache for analysis.

    - Saves whatever data has been provided (partial saves allowed)
    - Deterministic file naming based on claim hash and global step
    - Merges with existing file data if present
    - Thread-safe using a lock
    """
    if kwargs is None:
        kwargs = {}
    if kw:
        kwargs = {**kwargs, **kw}

    if "gt_label" in kwargs and "label" not in kwargs:
        kwargs["label"] = kwargs["gt_label"]

    # Minimum required fields to determine file path
    if "claim" not in kwargs or "trainer_state" not in kwargs:
        return

    def _ensure_list(value):
        return value if isinstance(value, list) else [value]

    # Mapping from kwargs keys to reward_data structure
    reward_key_mapping = {
        "format_rewards": ("rewards", "format"),
        "verification_rewards": ("rewards", "verification"),
        "good_number_of_questions_rewards": ("rewards", "good_number_of_questions"),
        "maximal_margin_classifier_rewards": ("rewards", "maximal_margin_classifier"),
        "vendi_diversity_rewards": ("rewards", "vendi_diversity"),
        "coverage_rewards": ("rewards", "coverage"),
        "necessity_saliency_rewards": ("rewards", "necessity_saliency"),
        "necessity_saliency_all_scores": ("rewards", "necessity_saliency_all_scores"),
        "joint_quality_rewards": ("rewards", "joint_quality"),
        "joint_quality_all_scores": ("rewards", "joint_quality_all_scores"),
    }

    top_level_mapping = {
        "claim": "claim",
        "label": "gt_label",
        "questions": "questions",
        "answers": "answers",
        "generation": "generation",
        "predicted_label": "predicted_label",
        "total_reward": "total_reward",
        "num_of_questions": "num_of_questions",
        "pubid": "id",
        "document": "document",
        "trainer_state": "trainer_state_info",
    }

    try:
        claims = _ensure_list(kwargs["claim"])
        global_step = kwargs["trainer_state"].global_step

        for i in range(len(claims)):
            claim_hash = sha256(claims[i].encode()).hexdigest()[:16]
            step = str(global_step).zfill(6)
            file_name = f"{rewards_cache_dir}/{step}_{i}_{claim_hash}.json"

            with _reward_save_lock:
                # Load existing data if file exists
                existing_data = {}
                if os.path.exists(file_name):
                    try:
                        with open(file_name, "r") as f:
                            existing_data = json.load(f)
                    except (json.JSONDecodeError, IOError):
                        existing_data = {}

                # Initialize reward_data from existing or empty
                reward_data = existing_data.copy()
                if "rewards" not in reward_data:
                    reward_data["rewards"] = {}

                # Update top-level fields
                for kwarg_key, data_key in top_level_mapping.items():
                    if kwarg_key in kwargs:
                        value = kwargs[kwarg_key]
                        if kwarg_key == "trainer_state":
                            value = {
                                "epoch": value.epoch,
                                "global_step": value.global_step,
                                "max_steps": value.max_steps,
                                "num_input_tokens_seen": value.num_input_tokens_seen,
                            }
                        if isinstance(value, list) and i < len(value):
                            reward_data[data_key] = value[i]
                        elif not isinstance(value, list):
                            reward_data[data_key] = value

                # Update reward fields
                for kwarg_key, (section, data_key) in reward_key_mapping.items():
                    if kwarg_key in kwargs:
                        value = kwargs[kwarg_key]
                        if isinstance(value, list) and i < len(value):
                            reward_data[section][data_key] = value[i]
                        elif not isinstance(value, list):
                            reward_data[section][data_key] = value

                # Save merged data
                with open(file_name, "w") as f:
                    json.dump(reward_data, f, indent=2)

            if random.random() < 0.01:
                logger.debug(
                    f"Saved reward data for claim index {i}, step {global_step}"
                )
    except Exception as e:
        logger.error(f"Error saving reward data: {e}")


NECESSITY_AGGREGATION = os.getenv("NECESSITY_AGGREGATION", "mean")  # "mean" or "min"
DIVERSITY_REWARD = os.getenv("DIVERSITY_REWARD", "mmr")  # "mmr" or "vendi"
SUPERVISION_RATE = float(os.getenv("SUPERVISION_RATE", "1.0"))
assert 0.0 <= SUPERVISION_RATE <= 1.0, (
    f"SUPERVISION_RATE must be between 0.0 and 1.0, got {SUPERVISION_RATE}"
)


def is_supervised(claim: str) -> bool:
    """Deterministic per-claim supervision decision based on SUPERVISION_RATE.

    Returns True if this claim should use ground-truth labels.
    Hash-based: same claim always gets the same decision.
    """
    if SUPERVISION_RATE >= 1.0:
        return True
    if SUPERVISION_RATE <= 0.0:
        return False
    h = int(sha256(claim.encode()).hexdigest(), 16)
    return (h % 10000) / 10000 < SUPERVISION_RATE


import time


def _timed(name, fn):
    """Wrap a reward function with timing logs."""

    def wrapper(completions, **kwargs):
        start = time.time()
        result = fn(completions, **kwargs)
        elapsed = time.time() - start
        print(f"[TIMING] {name}: {elapsed:.2f}s ({len(completions)} completions)")
        return result

    wrapper.__name__ = fn.__name__
    return wrapper


# NOTE: `get_reward_funcs` and the cross-reward overlap caches
# (`_joint_quality_cache`, `_coverage_verdicts_cache`) used to live here. They
# moved to `decomposer.unsloth.reward_funcs` so a single dispatcher can switch
# between this LLM-judge backend and the tiny-judge backend via REWARD_BACKEND.
# Existing call sites should `from decomposer.unsloth.reward_funcs import
# get_reward_funcs` — default behavior is unchanged (`REWARD_BACKEND=llm-judge`).
