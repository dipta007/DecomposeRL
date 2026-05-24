"""Reward-function dispatcher: pick the LLM or tiny-judge backend by env var.

`REWARD_BACKEND` selects which set of primitive reward functions powers the
GRPO reward list:

  - `llm-judge` (default): `decomposer.unsloth.rewards` — Qwen3-32B judge over
    vLLM + Qwen3-Embedding-8B over vLLM. Preserves the current training flow.
  - `tiny-judge`: `decomposer.unsloth.rewards_tiny_judge` — locally-loaded
    ModernBERT-large judges (one per task) + same vLLM embedding server.

Both backends expose the same primitive function signatures, so the
orchestration `_get_*_rewards` wrappers below are backend-agnostic.

Shared module-level caches (`_joint_quality_cache`, `_coverage_verdicts_cache`)
let the necessity-saliency wrapper precompute joint-quality and coverage in a
single asyncio.gather per training step; downstream wrappers pop those results
instead of re-running the judges.
"""

from __future__ import annotations

import asyncio
import logging
import os

from decomposer.unsloth.format_reward import get_format_reward
from decomposer.unsloth.rewards import (
    DIVERSITY_REWARD,
    NECESSITY_AGGREGATION,
    SUPERVISION_RATE,
    _majority_vote,
    _timed,
    extract_qa_pairs,
    is_supervised,
    save_reward_data,
)

logger = logging.getLogger(__name__)

# Cross-reward overlap caches, keyed by trainer global_step. Populated by
# _get_necessity_saliency_rewards, popped by _get_joint_quality_rewards and
# _get_unified_coverage so all three rewards' judge calls share one gather.
_joint_quality_cache: dict = {}
_coverage_verdicts_cache: dict = {}


def _select_backend():
    """Resolve REWARD_BACKEND to a module exposing the reward primitives."""
    name = os.getenv("REWARD_BACKEND", "llm-judge").lower()
    if name in ("tiny", "tiny-judge"):
        from decomposer.unsloth import rewards_tiny_judge as mod
    elif name in ("llm", "llm-judge"):
        from decomposer.unsloth import rewards as mod
    else:
        raise ValueError(
            f"REWARD_BACKEND={name!r} unknown; expected 'llm-judge' or 'tiny-judge'."
        )
    return mod


def get_reward_funcs():
    backend = _select_backend()
    logger.info(f"reward backend: {backend.__name__}")

    # Pull primitives from the chosen backend so the closures below work against
    # either implementation. All names exist in both modules with matching shapes.
    verification_reward = backend.verification_reward
    atomicity_checklist_reward = backend.atomicity_checklist_reward
    good_number_of_questions_reward = backend.good_number_of_questions_reward
    maximal_margin_classifier_reward = backend.maximal_margin_classifier_reward
    vendi_diversity_reward = backend.vendi_diversity_reward
    question_answerable_reward = backend.question_answerable_reward
    answer_correctness_reward = backend.answer_correctness_reward
    necessity_saliency_reward = backend.necessity_saliency_reward
    coverage_predict = backend.coverage_predict
    coverage_evaluate = backend.coverage_evaluate  # noqa: F841 (used via joint_quality_reward)
    joint_quality_reward = backend.joint_quality_reward

    necessity_saliency_reward_enabled = (
        os.getenv("NECESSITY_SALIENCY_REWARD", "1") == "1"
    )
    joint_quality_reward_enabled = os.getenv("JOINT_QUALITY_REWARD", "1") == "1"
    diversity_reward_enabled = os.getenv("DIVERSITY_REWARD_ENABLED", "1") == "1"
    coverage_reward_enabled = os.getenv("COVERAGE_REWARD", "1") == "1"

    def _get_format_rewards(completions, **kwargs):
        rewards = []
        kwargs["generation"] = []
        for completion in completions:
            generation = completion[0]["content"]
            kwargs["generation"].append(generation)
            reward = get_format_reward(generation)
            rewards.append(reward)
        kwargs["format_rewards"] = rewards
        save_reward_data(kwargs=kwargs)
        return rewards

    def _get_verification_rewards(completions, **kwargs):
        gt_labels = kwargs.get("label")
        claims = kwargs.get("claim", [])
        results = []
        for i, completion in enumerate(completions):
            generation = completion[0]["content"]
            if gt_labels is not None and is_supervised(claims[i]):
                result = verification_reward(generation, gt_labels[i])
            else:
                result = (0.0, "unknown")
            results.append(result)

        rewards = [res[0] for res in results]
        kwargs["predicted_label"] = [res[1] for res in results]
        kwargs["verification_rewards"] = rewards
        save_reward_data(kwargs=kwargs)
        return rewards

    def _get_good_number_of_questions_rewards(completions, **kwargs):
        gt_num_of_questions_list = kwargs["decomposed_questions"]
        gt_num_of_questions_list = [len(qs) for qs in gt_num_of_questions_list]
        rewards = []
        for i, completion in enumerate(completions):
            generation = completion[0]["content"]
            qa_pairs = extract_qa_pairs(generation)
            generated_num_of_questions = len(qa_pairs)
            reward = good_number_of_questions_reward(
                generation, generated_num_of_questions, gt_num_of_questions_list[i]
            )
            rewards.append(reward)

        kwargs["good_number_of_questions_rewards"] = rewards
        save_reward_data(kwargs=kwargs)
        return rewards

    def _get_maximal_margin_classifier_rewards(completions, **kwargs):
        async def _inner():
            func_args = []
            for i in range(len(completions)):
                generation = completions[i][0]["content"]
                qa_pairs = extract_qa_pairs(generation)
                questions = [pair["question"] for pair in qa_pairs]
                func_args.append((generation, questions))

            funcs = [maximal_margin_classifier_reward(*args) for args in func_args]
            rewards = await asyncio.gather(*funcs)
            kwargs["maximal_margin_classifier_rewards"] = rewards
            save_reward_data(kwargs=kwargs)
            return rewards

        return asyncio.run(_inner())

    def _get_vendi_diversity_rewards(completions, **kwargs):
        async def _inner():
            func_args = []
            for i in range(len(completions)):
                generation = completions[i][0]["content"]
                qa_pairs = extract_qa_pairs(generation)
                questions = [pair["question"] for pair in qa_pairs]
                func_args.append((generation, questions))

            funcs = [vendi_diversity_reward(*args) for args in func_args]
            rewards = await asyncio.gather(*funcs)
            kwargs["vendi_diversity_rewards"] = rewards
            save_reward_data(kwargs=kwargs)
            return rewards

        return asyncio.run(_inner())

    def _get_unified_coverage(completions, **kwargs):
        """Coverage reward: gt-based for supervised claims, self-consistency for unsupervised."""
        step = kwargs["trainer_state"].global_step
        claims = kwargs["claim"]
        gt_labels = kwargs.get("label")

        # Verdicts are usually pre-computed by _get_necessity_saliency_rewards so
        # the judge calls overlap with necessity/joint. Fall back to computing
        # fresh here if the cache is empty (e.g., necessity disabled or reordered).
        verdicts = _coverage_verdicts_cache.pop(step, None)
        if verdicts is None:

            async def _inner():
                func_args = []
                for i in range(len(completions)):
                    generation = completions[i][0]["content"]
                    qa_pairs = extract_qa_pairs(generation)
                    answers = [pair["answer"] for pair in qa_pairs]
                    func_args.append((generation, claims[i], answers))
                return await asyncio.gather(
                    *[coverage_predict(*args) for args in func_args]
                )

            verdicts = asyncio.run(_inner())

        rewards_list = [0.0] * len(completions)
        has_unsupervised = SUPERVISION_RATE < 1.0

        if has_unsupervised:
            num_generations = int(os.getenv("NUM_GENERATIONS", "8"))
            assert len(completions) % num_generations == 0, (
                f"Expected completions ({len(completions)}) to be divisible by "
                f"num_generations ({num_generations})"
            )
            num_prompts = len(completions) // num_generations
        else:
            num_prompts = None

        # Process supervised completions (direct gt comparison)
        for i in range(len(completions)):
            if is_supervised(claims[i]):
                if verdicts[i] is None or gt_labels is None:
                    rewards_list[i] = 0.0
                else:
                    rewards_list[i] = (
                        1.0 if verdicts[i] == gt_labels[i].lower() else 0.0
                    )

        # Process unsupervised completions (majority vote within group).
        # INVARIANT: In GRPO, all num_generations completions for a prompt share
        # the same claim, so is_supervised is consistent within each block.
        if has_unsupervised:
            for p in range(num_prompts):
                start = p * num_generations
                end = start + num_generations

                if is_supervised(claims[start]):
                    continue

                group_verdicts = [v for v in verdicts[start:end] if v is not None]
                if not group_verdicts:
                    continue

                pseudo_label = _majority_vote(group_verdicts)

                for j in range(start, end):
                    if verdicts[j] is not None and verdicts[j] == pseudo_label:
                        rewards_list[j] = 1.0
                    else:
                        rewards_list[j] = 0.0

        kwargs["coverage_rewards"] = rewards_list
        save_reward_data(kwargs=kwargs)
        return rewards_list

    def _get_necessity_saliency_rewards(completions, **kwargs):
        # Runs necessity-saliency, joint-quality, and (when coverage is in the
        # reward_list) coverage judge calls in one asyncio.gather so the slow
        # async work overlaps. Joint-quality results go into _joint_quality_cache
        # and coverage verdicts go into _coverage_verdicts_cache for the
        # downstream reward funcs to pop. Skips precomputing whichever rewards
        # are disabled via env flags so ablation runs don't pay for judge calls
        # nobody will read.
        precompute_joint = joint_quality_reward_enabled
        precompute_coverage = (
            coverage_reward_enabled and os.getenv("JOINT_COVERAGE", "0") != "1"
        )

        async def _inner():
            claims = kwargs["claim"]
            documents = kwargs["document"]
            gt_labels = kwargs.get("label")
            necessity_args = []
            joint_args = []
            coverage_args = []
            for i in range(len(completions)):
                generation = completions[i][0]["content"]
                qa_pairs = extract_qa_pairs(generation)
                answers = [pair["answer"] for pair in qa_pairs]
                gl = gt_labels[i] if (gt_labels and is_supervised(claims[i])) else None
                necessity_args.append((generation, claims[i], answers, gl))
                joint_args.append((generation, claims[i], documents[i], qa_pairs, gl))
                coverage_args.append((generation, claims[i], answers))

            necessity_coros = [
                necessity_saliency_reward(*a, aggregation=NECESSITY_AGGREGATION)
                for a in necessity_args
            ]
            n = len(completions)

            coros = list(necessity_coros)
            if precompute_joint:
                coros += [joint_quality_reward(*a) for a in joint_args]
            if precompute_coverage:
                coros += [coverage_predict(*a) for a in coverage_args]

            all_results = await asyncio.gather(*coros)
            necessity_results = all_results[:n]
            offset = n
            joint_results = None
            if precompute_joint:
                joint_results = all_results[offset : offset + n]
                offset += n
            coverage_verdicts = None
            if precompute_coverage:
                coverage_verdicts = all_results[offset : offset + n]
            return necessity_results, joint_results, coverage_verdicts

        necessity_results, joint_results, coverage_verdicts = asyncio.run(_inner())
        step = kwargs["trainer_state"].global_step
        if joint_results is not None:
            _joint_quality_cache[step] = joint_results
        if coverage_verdicts is not None:
            _coverage_verdicts_cache[step] = coverage_verdicts

        rewards = [r[0] for r in necessity_results]
        all_scores = [r[1] for r in necessity_results]
        kwargs["necessity_saliency_rewards"] = rewards
        kwargs["necessity_saliency_all_scores"] = all_scores
        save_reward_data(kwargs=kwargs)
        return rewards

    def _get_joint_quality_rewards(completions, **kwargs):
        step = kwargs["trainer_state"].global_step
        results = _joint_quality_cache.pop(step, None)

        if results is None:
            # Defensive fallback: necessity-saliency didn't populate the cache.
            # Shouldn't happen given the reward_list order below, but compute fresh
            # so a reordering or disabled necessity reward doesn't silently break.
            async def _inner():
                claims = kwargs["claim"]
                documents = kwargs["document"]
                gt_labels = kwargs.get("label")
                func_args = []
                for i in range(len(completions)):
                    generation = completions[i][0]["content"]
                    qa_pairs = extract_qa_pairs(generation)
                    gl = (
                        gt_labels[i]
                        if (gt_labels and is_supervised(claims[i]))
                        else None
                    )
                    func_args.append(
                        (generation, claims[i], documents[i], qa_pairs, gl)
                    )
                return await asyncio.gather(
                    *[joint_quality_reward(*a) for a in func_args]
                )

            results = asyncio.run(_inner())

        rewards = [r[0] for r in results]
        all_scores = [r[1] for r in results]
        kwargs["joint_quality_rewards"] = rewards
        kwargs["joint_quality_all_scores"] = all_scores
        save_reward_data(kwargs=kwargs)
        return rewards

    # Refresh disk cache paths under .cache/<WANDB_NAME>/... so per-judge LLM
    # cache writes land in a run-specific directory. Mutating the LLM rewards
    # module's globals is harmless for the tiny-judge backend (which doesn't
    # call _call_judge_model) but keeps `rewards_cache_dir` (used by
    # save_reward_data) pointed at the right place for either backend.
    import decomposer.unsloth.rewards as _r

    wandb_name = os.getenv("WANDB_NAME", "default_run")
    _r.atomicity_checklist_cache_dir = f".cache/{wandb_name}/atomicity_checklist"
    _r.question_answerable_cache_dir = f".cache/{wandb_name}/question_answerable"
    _r.answer_correctness_cache_dir = f".cache/{wandb_name}/answer_correctness"
    _r.coverage_cache_dir = f".cache/{wandb_name}/coverage"
    _r.rewards_cache_dir = f".cache/{wandb_name}/rewards"
    for d in (
        _r.atomicity_checklist_cache_dir,
        _r.question_answerable_cache_dir,
        _r.answer_correctness_cache_dir,
        _r.coverage_cache_dir,
        _r.rewards_cache_dir,
    ):
        os.makedirs(d, exist_ok=True)

    diversity_fn = (
        _get_maximal_margin_classifier_rewards
        if DIVERSITY_REWARD == "mmr"
        else _get_vendi_diversity_rewards
    )

    joint_coverage = os.getenv("JOINT_COVERAGE", "0") == "1"
    good_num_q_reward = os.getenv("GOOD_NUM_Q_REWARD", "1") == "1"
    logger.info(f"Joint coverage: {joint_coverage}")
    logger.info(f"Good number of questions reward: {good_num_q_reward}")
    logger.info(f"Supervision rate: {SUPERVISION_RATE}")
    logger.info(f"Necessity-saliency reward: {necessity_saliency_reward_enabled}")
    logger.info(f"Joint-quality reward: {joint_quality_reward_enabled}")
    logger.info(
        f"Diversity reward: {DIVERSITY_REWARD} (enabled={diversity_reward_enabled})"
    )
    logger.info(f"Coverage reward: {coverage_reward_enabled}")

    reward_list = [_timed("format", _get_format_rewards)]
    if necessity_saliency_reward_enabled:
        reward_list.append(
            _timed("necessity_saliency", _get_necessity_saliency_rewards)
        )
    if joint_quality_reward_enabled:
        reward_list.append(_timed("joint_quality", _get_joint_quality_rewards))
    if diversity_reward_enabled:
        reward_list.append(_timed("diversity", diversity_fn))
    if SUPERVISION_RATE > 0.0:
        reward_list.append(_timed("verification", _get_verification_rewards))
    if good_num_q_reward:
        reward_list.append(_timed("good_num_q", _get_good_number_of_questions_rewards))
    if not joint_coverage and coverage_reward_enabled:
        reward_list.append(_timed("coverage", _get_unified_coverage))

    return reward_list
