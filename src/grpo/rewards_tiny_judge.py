from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import List, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

# === Configurable: which trained models back each judge ==================
# Each value is a HuggingFace repo id of a ModernBERT-large SequenceClassification
# checkpoint trained by decomposer/tiny_judge/train_encoder_balanced.py. Override
# any entry to swap individual judges (e.g. point one at the LoRA-natural
# variant: `dipta007/{task}-judge-lora`).
JUDGE_MODELS: dict[str, str] = {
    "atomicity_is_question": "xxxx/atomicity-is-question-judge-balanced-v2",
    "atomicity_single_focus": "dipta007/atomicity-single-focus-judge-balanced-v2",
    "atomicity_no_conjunctions": "dipta007/atomicity-no-conjunctions-judge-balanced-v2",
    "atomicity_verifiable": "dipta007/atomicity-verifiable-judge-balanced-v2",
    "atomicity_grounded": "dipta007/atomicity-grounded-judge-balanced-v2",
    "question_answerable": "dipta007/question-judge-balanced-v2",
    "answer_correctness": "dipta007/answer-judge-balanced-v2",
    "coverage": "dipta007/coverage-judge-balanced-v2",
}

# Inference knobs
JUDGE_MAX_LENGTH = 8192  # matches training-time max_length
JUDGE_BATCH_CHUNK = 64  # max tokenized rows per forward pass; tune for VRAM

# Atomicity criteria in the same order as build_dataset.py / training.
ATOMICITY_CRITERIA = (
    "is_question",
    "single_focus",
    "no_conjunctions",
    "verifiable",
    "grounded",
)

COVERAGE_LABEL_NAMES = ("supported", "refuted", "not enough information")
_device = "cuda" if torch.cuda.is_available() else "cpu"
_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32


# === Lazy model registry ==================================================
class _JudgeRegistry:
    """Lazy-load classifier judges on first use; cache in memory thereafter.

    Thread-safe: `_classify_batch_sync` is dispatched via `asyncio.to_thread`
    and many of those threads can race to first-load the same task. A global
    lock + double-checked locking serializes loads (so we never end up with
    duplicate models on GPU) without paying lock cost on the steady-state hot
    path where the model is already cached.
    """

    def __init__(self):
        self._models: dict[str, tuple] = {}
        self._lock = threading.Lock()

    def get(self, task: str):
        cached = self._models.get(task)
        if cached is not None:
            return cached
        with self._lock:
            cached = self._models.get(task)
            if cached is not None:
                return cached
            repo = JUDGE_MODELS[task]
            logger.info(f"[tiny-judge] loading {task} from {repo}")
            tok = AutoTokenizer.from_pretrained(repo)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            mdl = (
                AutoModelForSequenceClassification.from_pretrained(
                    repo,
                    dtype=_dtype,
                    # ModernBERT + flash_attention_2 hits an `assert sin.shape ==
                    # cos.shape` in flash-attn's triton rotary kernel; SDPA is
                    # the safe path (same one `train_encoder_balanced.py`'s
                    # module docstring calls out).
                    attn_implementation="sdpa",
                    # ModernBERT defaults `reference_compile` to True on CUDA,
                    # which wraps `self.mlp` in `torch.compile`. That collides
                    # with the dynamo/FX state in the GRPO trainer process
                    # ("Detected that you are using FX to symbolically trace a
                    # dynamo-optimized function"). Plain MLP forward is fine
                    # at inference — we don't need the compile speedup.
                    reference_compile=False,
                )
                # Force uniform dtype across all params: Unsloth patches torch
                # globally on import (grpo_lora.py:49) and that can leave some
                # newly-built Linears at the process default dtype rather than
                # the requested one, producing a mid-forward
                # `BFloat16 != float` mismatch in ModernBERT's Wo projection.
                .to(device=_device, dtype=_dtype)
                .eval()
            )
            for p in mdl.parameters():
                p.requires_grad_(False)
            self._models[task] = (mdl, tok)
            return self._models[task]


_registry = _JudgeRegistry()


# === Inference helpers ====================================================
@torch.no_grad()
def _classify_batch_sync(task: str, texts: List[str]) -> torch.Tensor:
    """Argmax-over-logits over all `texts`. Returns CPU long tensor of length len(texts)."""
    if not texts:
        return torch.empty(0, dtype=torch.long)
    mdl, tok = _registry.get(task)
    preds_chunks = []
    for i in range(0, len(texts), JUDGE_BATCH_CHUNK):
        chunk = texts[i : i + JUDGE_BATCH_CHUNK]
        enc = tok(
            chunk,
            padding=True,
            truncation=True,
            max_length=JUDGE_MAX_LENGTH,
            return_tensors="pt",
        ).to(_device)
        # Disable any outer autocast (the GRPO trainer runs its training step
        # under torch.autocast(dtype=bf16); inheriting that into the judge
        # forward can promote intermediates and produce a `BFloat16 != float`
        # mismatch against the judge's bf16 weights). The judge model is
        # already a fixed-dtype eval-only forward, so amp gives us nothing.
        with torch.amp.autocast(device_type=_device, enabled=False):
            logits = mdl(**enc).logits  # (b, num_labels)
        preds_chunks.append(logits.argmax(dim=-1).cpu())
    return torch.cat(preds_chunks, dim=0)


async def _classify_batch(task: str, texts: List[str]) -> torch.Tensor:
    """Async wrapper: keeps call-site shape parity with the LLM-pathway primitives.

    On a single GPU these don't actually parallelize across calls (CUDA stream
    serializes), but the inner batched forward IS where the speedup lives.
    """
    return await asyncio.to_thread(_classify_batch_sync, task, texts)


# === Text formatting (MUST match build_dataset.py / training) ============
def _atomicity_text(claim: str, question: str) -> str:
    return f"Claim: {claim}\nQuestion: {question}"


def _question_answerable_text(document: str, question: str) -> str:
    return f"Document: {document}\nQuestion: {question}"


def _answer_correctness_text(document: str, question: str, answer: str) -> str:
    return f"Document: {document}\nQuestion: {question}\nAnswer: {answer}"


def _coverage_text(claim: str, answers_formatted: str) -> str:
    return f"Claim: {claim}\nAnswers:\n{answers_formatted}"


def _format_answers(answers: List[str]) -> str:
    """Multi-line bulleted answers, matching the training-time format."""
    return "\n".join(f"- {a}" for a in answers)


# === Primitive reward functions ==========================================
async def atomicity_checklist_reward(generation, claim, questions):
    """5 binary judges over the same N questions; per-question score = (sum yes) / 5."""
    if not questions:
        return 0.0, []
    texts = [_atomicity_text(claim, q) for q in questions]

    # asyncio.gather over the 5 criterion-specific judges. Each does one batched
    # forward of length N (across all questions).
    per_crit = await asyncio.gather(
        *[_classify_batch(f"atomicity_{c}", texts) for c in ATOMICITY_CRITERIA]
    )

    n_crit = len(ATOMICITY_CRITERIA)
    n_q = len(questions)
    per_q_scores: List[float] = []
    for q_idx in range(n_q):
        passed = sum(int(per_crit[c_idx][q_idx]) for c_idx in range(n_crit))
        per_q_scores.append(passed / n_crit)

    avg = sum(per_q_scores) / n_q
    return avg, per_q_scores


async def question_answerable_reward(generation, document, questions, answers):
    """Same IDK-aware scoring as rewards.py.question_answerable_reward."""
    if not questions:
        return 0.0, []
    texts = [_question_answerable_text(document, q) for q in questions]
    preds = await _classify_batch("question_answerable", texts)

    scores: List[float] = []
    for ans, pred in zip(answers, preds):
        is_idk = ans.strip(" .").lower() == "i don't know" or ans.strip() == ""
        is_answerable = int(pred) == 1
        if is_idk:
            scores.append(0.0 if is_answerable else 1.0)
        else:
            scores.append(1.0 if is_answerable else 0.0)
    avg = sum(scores) / len(scores)
    return avg, scores


async def answer_correctness_reward(generation, document, qa_pairs):
    if not qa_pairs:
        return 0.0, []
    texts = [
        _answer_correctness_text(document, p["question"], p["answer"]) for p in qa_pairs
    ]
    preds = await _classify_batch("answer_correctness", texts)
    scores = [1.0 if int(p) == 1 else 0.0 for p in preds]
    avg = sum(scores) / len(scores)
    return avg, scores


async def coverage_predict(generation, claim, answers) -> Optional[str]:
    """Single coverage prediction → 'supported' / 'refuted' / 'not enough information'.

    Always returns a label (never None) since the local classifier always produces
    an argmax; the Optional return type is for signature parity with rewards.py.
    """
    text = _coverage_text(claim, _format_answers(answers))
    preds = await _classify_batch("coverage", [text])
    return COVERAGE_LABEL_NAMES[int(preds[0])]


async def coverage_evaluate(generation, claim, answers, gt_label) -> float:
    pred = await coverage_predict(generation, claim, answers)
    if pred is None:
        return 0.0
    return 1.0 if pred == gt_label.lower() else 0.0


async def necessity_saliency_reward(
    generation, claim, answers, gt_label=None, aggregation="mean"
) -> tuple[float, List[float]]:
    """Leave-one-out coverage. Same scoring logic as rewards.py but N+1 forwards
    are batched through the coverage classifier in one tensor."""
    if len(answers) == 0:
        return 0.0, []

    # Single-question short-circuits (mirror rewards.py): the only Q is necessary
    # by definition if the full verdict exists / is correct; running LOO with
    # zero answers is meaningless.
    if len(answers) == 1:
        if gt_label is None:
            return 1.0, [1.0]
        full_label = await coverage_predict(generation, claim, answers)
        full_correct = (full_label == gt_label.lower()) if full_label else False
        score = 1.0 if full_correct else 0.0
        return score, [score]

    # Build the N+1 coverage inputs (full + each leave-one-out) and classify in one batch.
    full_text = _coverage_text(claim, _format_answers(answers))
    loo_texts = [
        _coverage_text(claim, _format_answers(answers[:i] + answers[i + 1 :]))
        for i in range(len(answers))
    ]
    preds = await _classify_batch("coverage", [full_text] + loo_texts)
    all_labels = [COVERAGE_LABEL_NAMES[int(p)] for p in preds]
    full_label = all_labels[0]
    loo_labels = all_labels[1:]

    if gt_label is None:
        # Label-free relative necessity: a question is necessary iff removing it flips the verdict.
        per_q = [1.0 if lbl != full_label else 0.0 for lbl in loo_labels]
    else:
        full_correct = full_label == gt_label.lower()
        per_q: List[float] = []
        for lbl in loo_labels:
            loo_correct = lbl == gt_label.lower()
            if full_correct and not loo_correct:
                per_q.append(1.0)  # necessary
            elif full_correct and loo_correct:
                per_q.append(0.5)  # redundant but harmless
            elif not full_correct and loo_correct:
                per_q.append(-1.0)  # harmful
            else:
                per_q.append(0.0)  # neutral (both wrong)

    if aggregation == "min":
        score = min(per_q) if per_q else 0.0
    else:
        score = sum(per_q) / len(per_q) if per_q else 0.0
    return score, per_q


async def joint_quality_reward(
    generation, claim, document, qa_pairs, gt_label=None
) -> tuple[float, List[float]]:
    """answerable * atomic * correct per question; optionally * coverage."""
    questions = [p["question"] for p in qa_pairs]
    answers = [p["answer"] for p in qa_pairs]
    if not qa_pairs:
        return 0.0, []

    use_joint_coverage = os.getenv("JOINT_COVERAGE", "0") == "1"
    if use_joint_coverage and gt_label is None:
        raise ValueError(
            "JOINT_COVERAGE=1 requires gt_label but got None. "
            "Set JOINT_COVERAGE=0 for unsupervised samples (SUPERVISION_RATE < 1.0)."
        )

    if use_joint_coverage and gt_label is not None:
        (
            (_answerable_avg, answerable_scores),
            (_atomicity_avg, atomicity_scores),
            (_correctness_avg, correctness_scores),
            coverage_score,
        ) = await asyncio.gather(
            question_answerable_reward(generation, document, questions, answers),
            atomicity_checklist_reward(generation, claim, questions),
            answer_correctness_reward(generation, document, qa_pairs),
            coverage_evaluate(generation, claim, answers, gt_label),
        )
    else:
        (
            (_answerable_avg, answerable_scores),
            (_atomicity_avg, atomicity_scores),
            (_correctness_avg, correctness_scores),
        ) = await asyncio.gather(
            question_answerable_reward(generation, document, questions, answers),
            atomicity_checklist_reward(generation, claim, questions),
            answer_correctness_reward(generation, document, qa_pairs),
        )
        coverage_score = 0.0

    per_q_quality: List[float] = []
    for i in range(len(qa_pairs)):
        ans = qa_pairs[i]["answer"]
        is_idk = ans.strip(" .").lower() == "i don't know" or ans.strip() == ""
        if is_idk:
            per_q_quality.append(answerable_scores[i] * atomicity_scores[i])
        else:
            per_q_quality.append(
                answerable_scores[i] * atomicity_scores[i] * correctness_scores[i]
            )

    mean_quality = sum(per_q_quality) / len(per_q_quality)
    reward = mean_quality * coverage_score if use_joint_coverage else mean_quality
    return reward, per_q_quality
