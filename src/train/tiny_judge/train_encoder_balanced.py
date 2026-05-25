#!/usr/bin/env python3
"""LoRA fine-tune via Unsloth as a tiny-judge classifier/regressor.

Trains on the balanced `train_balanced` split, validates on natural
`validation`, evaluates on natural `test` and `test_balanced`. Best ckpt =
highest macro-F1. LoRA adapters are merged into the base before push so
consumers can use a plain `from_pretrained` without PEFT. Pushes to
anonymous/{task}-judge-balanced (private).

Backbone is `ENCODER_BACKBONE` from `_common.py` (currently Unsloth's
ModernBERT-large). Unsloth's FastModel + 8-bit Adam + "unsloth"-mode
gradient checkpointing gets us back the speed we lost with vanilla SDPA
without hitting the flash-attn rotary kernel crash.

Note: `train_balanced` size varies wildly per task — atomicity is small
(~1.3k rows) and converges in minutes, but the others are large
(question_answerable ~243k, answer_correctness ~1.1M, coverage ~3.3M),
where one "epoch" is hundreds of thousands of steps. Tune `--epochs` and
`--eval-steps` per task accordingly.

Usage:
    uv run -m src.tiny_judge.train_encoder_balanced --task atomicity_checklist --push
    uv run -m src.tiny_judge.train_encoder_balanced --task all --push
"""

from __future__ import annotations

# IMPORTANT: Unsloth must be imported before transformers/torch so it can patch
# kernels at import time.
from unsloth import FastModel, is_bfloat16_supported  # isort: skip

import argparse
import logging
from pathlib import Path

import torch
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.train.tiny_judge._common import (
    ALL_TASK_NAMES,
    ENCODER_BACKBONE,
    TASKS,
    TaskConfig,
    compute_classification_metrics,
    finish_wandb_run,
    load_task_dataset,
    metric_for_best,
    repo_name_for,
    setup_wandb,
)
from src.train.tiny_judge._encoder_common import (
    auto_eval_steps,
    prepare_for_trainer,
    sort_by_length,
    stratified_eval_subset,
    tokenize_dataset,
)

logger = logging.getLogger("tiny_judge.train_encoder_balanced")
RUN_KIND = "encoder_lora_balanced"
PUSH_SUFFIX = "balanced"

# "all-linear" lets PEFT auto-target every nn.Linear in the backbone (and skip
# the classification head, which is kept trainable via modules_to_save). Avoids
# hard-coding backbone-specific names like LLaMA's q_proj or ModernBERT's Wqkv.
LORA_TARGET_MODULES = "all-linear"


def _train_one_task(task: TaskConfig, args):
    logger.info(f"========== Task: {task.name} (Unsloth LoRA, balanced) ==========")
    setup_wandb(
        RUN_KIND, task, enabled=not args.no_wandb,
        train_split="balanced", backbone="modernbert-large",
    )

    train_raw, val_raw, test_raw, test_bal_raw = load_task_dataset(
        task, train_split="train_balanced", limit=args.limit, eval_limit=args.eval_limit
    )
    # Stratified subsample of val for in-training eval (5000 max, balanced via
    # waterfilling). test/test_balanced stay full for honest final reporting.
    val_raw = stratified_eval_subset(val_raw, target=args.eval_balanced_size, seed=42)
    logger.info(f"[{task.name}] val (stratified subset for training eval): {len(val_raw)} rows")

    # FastModel.from_pretrained loads model + tokenizer together and applies
    # Unsloth's kernel patches. `auto_model` selects the SeqClassification head.
    model, tokenizer = FastModel.from_pretrained(
        model_name=ENCODER_BACKBONE,
        auto_model=AutoModelForSequenceClassification,
        max_seq_length=args.max_length,
        dtype=None,  # auto: bf16 on Ampere+, fp16 on T4/V100
        num_labels=task.num_labels,
        load_in_8bit=False,
        load_in_4bit=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = prepare_for_trainer(
        tokenize_dataset(train_raw, tokenizer, args.max_length), task.problem_type
    )
    # Eval sets are length-sorted to minimize per-batch padding waste; HF's
    # `group_by_length=True` only handles the train sampler.
    val_ds = prepare_for_trainer(
        sort_by_length(tokenize_dataset(val_raw, tokenizer, args.max_length)),
        task.problem_type,
    )
    test_ds = prepare_for_trainer(
        sort_by_length(tokenize_dataset(test_raw, tokenizer, args.max_length)),
        task.problem_type,
    )
    test_bal_ds = prepare_for_trainer(
        sort_by_length(tokenize_dataset(test_bal_raw, tokenizer, args.max_length)),
        task.problem_type,
    )

    # Standard LoRA scaling rule: alpha = 2 * r → effective scale = 2.
    # Auto-derived when --lora-alpha is left unset.
    lora_alpha = args.lora_alpha if args.lora_alpha is not None else 2 * args.lora_rank

    # Wrap with LoRA. Unsloth's get_peft_model freezes the backbone and keeps
    # the classification head trainable for task_type="SEQ_CLS".
    model = FastModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=LORA_TARGET_MODULES,
        lora_alpha=lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        # "unsloth" mode = ~30% less VRAM and lets us fit ~2x batch.
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
        task_type="SEQ_CLS",
    )
    logger.info(f"[{task.name}] r={args.lora_rank} alpha={lora_alpha} (alpha/r={lora_alpha/args.lora_rank:.1f})")
    model.print_trainable_parameters()

    metric_name, greater_is_better = metric_for_best(task)
    output_dir = (
        Path(args.output_dir) / repo_name_for(task, PUSH_SUFFIX).split("/", 1)[1]
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-task epochs: atomicity (1.3k rows) wants many passes; the other
    # balanced splits (10k–3.3M rows) saturate in a few epochs and rely on early
    # stopping to cut off.
    epochs = args.epochs
    if epochs is None:
        epochs = 20 if task.name == "atomicity_checklist" else 5

    eval_steps = args.eval_steps or auto_eval_steps(
        len(train_raw), args.batch_size * args.grad_accum, epochs
    )
    if args.eval_steps is None:
        logger.info(f"[{task.name}] auto eval_steps={eval_steps} (~30 evals)")

    use_bf16 = is_bfloat16_supported()
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        run_name=f"{RUN_KIND}__{task.name}",  # explicit per-task wandb name
        num_train_epochs=epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.0,  # LoRA conventionally uses 0 weight decay
        warmup_ratio=0.1,  # longer warmup tames the random-init classification head
        lr_scheduler_type="linear",
        label_smoothing_factor=0.1
        if task.problem_type == "single_label_classification"
        else 0.0,
        bf16=use_bf16,
        fp16=not use_bf16,
        optim="adamw_torch",
        group_by_length=True,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,  # must align with eval_steps for load_best_model_at_end
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        greater_is_better=greater_is_better,
        logging_steps=1,
        report_to=["wandb"] if not args.no_wandb else [],
        seed=args.seed,
        # Workers >0 use multiprocessing shared-memory tempdirs that fail to clean up
        # on NFS-mounted home dirs (EBUSY during rmtree). Keep at 0 unless TMPDIR is local.
        dataloader_num_workers=0,
        remove_unused_columns=True,
        push_to_hub=False,
    )

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        return compute_classification_metrics(preds, labels, task)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer, padding="longest"),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    logger.info(f"[{task.name}] starting Unsloth LoRA training")
    trainer.train()
    logger.info(f"[{task.name}] training complete; best model loaded")

    test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
    logger.info(f"[{task.name}] test (natural): {test_metrics}")
    test_bal_metrics = trainer.evaluate(test_bal_ds, metric_key_prefix="test_balanced")
    logger.info(f"[{task.name}] test_balanced: {test_bal_metrics}")

    if args.push:
        repo_id = repo_name_for(task, PUSH_SUFFIX)
        logger.info(f"[{task.name}] merging LoRA into base → {repo_id} (private)")
        # Don't use Unsloth's `push_to_hub_merged(save_method="merged_16bit")`:
        # on a `ModernBertForSequenceClassification` it writes the wrong
        # `architectures: [ModernBertForMaskedLM]` to config.json and drops
        # `classifier.weight`/`classifier.bias` from the safetensors, which
        # leaves consumers loading random classifier heads (and on transformers
        # >= 4.5x crashing with `Cannot copy out of meta tensor`). Instead, use
        # PEFT's standard merge + HF's standard push: classifier is preserved
        # because it was in `modules_to_save`, and the correct
        # SeqClassification architecture lands in config.json.
        merged = model.merge_and_unload()
        merge_out = output_dir / "merged"
        merge_out.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(merge_out, safe_serialization=True)
        tokenizer.save_pretrained(merge_out)
        merged.push_to_hub(repo_id, private=True)
        tokenizer.push_to_hub(repo_id, private=True)
        logger.info(f"[{task.name}] push complete: https://huggingface.co/{repo_id}")

    return {"test": test_metrics, "test_balanced": test_bal_metrics}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--task",
        default="all",
        choices=("all",) + ALL_TASK_NAMES,
        help="Task to train (default: all 9)",
    )
    ap.add_argument(
        "--output-dir",
        default="outputs/tiny_judge/.ckpts",
        help="Local checkpoint dir",
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs. If unset, defaults to 20 for atomicity "
        "(tiny train_balanced) and 5 for the larger tasks (rely on early stopping).",
    )
    ap.add_argument(
        "--eval-steps",
        type=int,
        default=None,
        help="Eval + save every N train steps. If unset, auto-computed to give "
        "~30 evals per run based on (train_balanced size, batch, grad_accum, epochs).",
    )
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument(
        "--max-length",
        type=int,
        default=8192,
        help="Truncate at this many tokens. Unsloth handles RoPE scaling internally.",
    )
    ap.add_argument(
        "--grad-accum",
        type=int,
        default=1,
        help="Gradient accumulation steps. Effective batch = batch_size * grad_accum.",
    )
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument(
        "--eval-balanced-size",
        type=int,
        default=5000,
        help="Class-balanced (waterfilling) subset of natural validation to use "
        "for in-training eval. Test sets stay full for the end-of-training pass.",
    )
    ap.add_argument("--lora-rank", type=int, default=64)
    ap.add_argument(
        "--lora-alpha",
        type=int,
        default=None,
        help="LoRA scaling factor. If unset, defaults to 2 * --lora-rank (standard rule).",
    )
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early-stopping patience in eval intervals (= patience * eval_steps train steps).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None, help="Debug: cap train rows")
    ap.add_argument(
        "--eval-limit",
        type=int,
        default=None,
        help="Debug: cap val + test + test_balanced rows (for CPU smoke tests)",
    )
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    tasks = ALL_TASK_NAMES if args.task == "all" else (args.task,)
    summary = {}
    for t in tasks:
        summary[t] = _train_one_task(TASKS[t], args)
        finish_wandb_run()  # ensure each task gets its own wandb run + ID
        torch.cuda.empty_cache()

    logger.info("========== Summary ==========")
    for t, m in summary.items():
        logger.info(f"  {t}: {m}")


if __name__ == "__main__":
    main()
