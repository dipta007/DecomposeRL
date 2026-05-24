"""Shared utility functions for evaluation scripts."""

import asyncio
from typing import Dict, List, Optional

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from decomposer.unsloth.rewards import (
    answer_correctness_reward,
    atomicity_checklist_reward,
    coverage_evaluate,
    good_number_of_questions_reward,
    joint_quality_reward,
    maximal_margin_classifier_reward,
    necessity_saliency_reward,
    question_answerable_reward,
    vendi_diversity_reward,
)


def compute_classification_metrics(
    gt_labels: List[str], pred_labels: List[str]
) -> Dict[str, float]:
    """Compute classification metrics."""
    gt_labels_lower = [l.lower() for l in gt_labels]
    pred_labels_lower = [l.lower() if l else "refuted" for l in pred_labels]

    accuracy = accuracy_score(gt_labels_lower, pred_labels_lower)
    balanced_accuracy = balanced_accuracy_score(gt_labels_lower, pred_labels_lower)
    precision = precision_score(
        gt_labels_lower, pred_labels_lower, pos_label="supported", zero_division=0
    )
    recall = recall_score(
        gt_labels_lower, pred_labels_lower, pos_label="supported", zero_division=0
    )
    f1 = f1_score(
        gt_labels_lower, pred_labels_lower, pos_label="supported", zero_division=0
    )

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def format_evidence_from_context(context) -> str:
    """Format evidence document from context dictionary."""
    if isinstance(context, dict):
        contexts = context.get("contexts", [])
        labels = context.get("labels", [])

        evidence_parts = []
        for ctx, label in zip(contexts, labels):
            evidence_parts.append(f"## {label}\n{ctx}")
        evidence = "\n\n".join(evidence_parts)
    else:
        evidence = context

    return evidence


async def compute_llm_rewards(
    claim: str,
    generation: str,
    questions: List[str],
    answers: List[str],
    document: str,
    gt_label: str,
    gt_num_of_questions: Optional[int] = None,
) -> Dict:
    """Compute LLM-based rewards (atomicity, diversity, question_answerable, answer_correctness, coverage)."""
    if not questions:
        return {
            "atomicity_checklist_reward": 0.0,
            "atomicity_checklist_scores": [],
            "mmr_reward": 0.0,
            "vendi_diversity_reward": 0.0,
            "question_answerable_reward": 0.0,
            "question_answerable_scores": [],
            "answer_correctness_reward": 0.0,
            "answer_correctness_scores": [],
            "coverage_reward": 0.0,
            "good_number_of_questions_reward": 0.0,
            "necessity_saliency_reward": 0.0,
            "necessity_saliency_scores": [],
            "joint_quality_reward": 0.0,
            "joint_quality_scores": [],
        }

    # Build qa_pairs for answer_correctness_reward and joint_quality_reward
    qa_pairs = [{"question": q, "answer": a} for q, a in zip(questions, answers)]

    # Gather all async rewards in parallel
    (
        atomicity_checklist_result,
        mmr,
        vendi_diversity,
        question_answerable_result,
        answer_correctness_result,
        coverage_score,
        necessity_saliency_result,
        joint_quality_result,
    ) = await asyncio.gather(
        atomicity_checklist_reward(generation, claim, questions),
        maximal_margin_classifier_reward(generation, questions),
        vendi_diversity_reward(generation, questions),
        question_answerable_reward(generation, document, questions, answers),
        answer_correctness_reward(generation, document, qa_pairs),
        coverage_evaluate(generation, claim, answers, gt_label),
        necessity_saliency_reward(generation, claim, answers, gt_label),
        joint_quality_reward(generation, claim, document, qa_pairs, gt_label),
    )

    atomicity_checklist_avg, atomicity_checklist_scores = atomicity_checklist_result
    question_answerable_avg, question_answerable_scores = question_answerable_result
    answer_correctness_avg, answer_correctness_scores = answer_correctness_result
    necessity_saliency_avg, necessity_saliency_scores = necessity_saliency_result
    joint_quality_avg, joint_quality_scores = joint_quality_result

    # Compute good_number_of_questions_reward (synchronous)
    good_num_reward = 0.0
    if gt_num_of_questions is not None:
        good_num_reward = good_number_of_questions_reward(
            generation, len(questions), gt_num_of_questions
        )

    return {
        "atomicity_checklist_reward": atomicity_checklist_avg,
        "atomicity_checklist_scores": atomicity_checklist_scores,
        "mmr_reward": mmr,
        "vendi_diversity_reward": vendi_diversity,
        "question_answerable_reward": question_answerable_avg,
        "question_answerable_scores": question_answerable_scores,
        "answer_correctness_reward": answer_correctness_avg,
        "answer_correctness_scores": answer_correctness_scores,
        "coverage_reward": coverage_score,
        "good_number_of_questions_reward": good_num_reward,
        "necessity_saliency_reward": necessity_saliency_avg,
        "necessity_saliency_scores": necessity_saliency_scores,
        "joint_quality_reward": joint_quality_avg,
        "joint_quality_scores": joint_quality_scores,
    }
