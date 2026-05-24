import os
import logging

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

GPU_MEMORY_UTILIZATION = 0.5
BATCH_SIZE = 4
ENFORCE_EAGER = False

import unsloth
from src.baselines.utils import HF_DATASET, HF_SUBSET
import torch

logger = logging.getLogger(__name__)
from datasets import Dataset
from src.train.decomposerl.rewards import get_reward_funcs
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel
from vllm import SamplingParams
from src.train.decomposerl.prompts import USER_PROMPT_2WAY_TEMPLATE

NPROC_PER_NODE = torch.cuda.device_count()
DEVICE_IDS = list(range(NPROC_PER_NODE))

GLOBAL_BATCH_SIZE = 16
GRAD_ACC_STEPS = GLOBAL_BATCH_SIZE / (BATCH_SIZE * NPROC_PER_NODE)
if GLOBAL_BATCH_SIZE % (BATCH_SIZE * NPROC_PER_NODE) != 0:
    raise ValueError(
        f"Global batch size {GLOBAL_BATCH_SIZE} is not divisible by {BATCH_SIZE * NPROC_PER_NODE}"
    )
GRAD_ACC_STEPS = int(GRAD_ACC_STEPS)


def get_dataset() -> Dataset:
    from datasets import load_dataset as hf_load_dataset

    dataset_ds = hf_load_dataset(HF_DATASET, HF_SUBSET, split="train")

    def _process_row(row):
        claim = row["claim"]
        evidence_text = row["evidence"]
        conversation = [
            {
                "role": "user",
                "content": USER_PROMPT_2WAY_TEMPLATE.format(
                    claim=claim, evidence_doc=evidence_text
                ),
            }
        ]
        row["prompt"] = conversation
        row["document"] = evidence_text
        return row

    dataset_ds = dataset_ds.map(_process_row)
    logger.info("========================================")
    logger.info(f"Loaded {len(dataset_ds)} samples from {HF_DATASET}/{HF_SUBSET}")
    logger.info(f"dataset_ds: {dataset_ds}")
    logger.debug(dataset_ds["prompt"][0])
    logger.info("========================================")
    return dataset_ds


def log_configs(args):
    logger.info("========================================")
    logger.info("Configs:")
    logger.info("========================================")
    for k, v in vars(args).items():
        logger.info(f"{k}: {v}")
    logger.info("========================================")
    logger.info("Hyperparameters:")
    logger.info("========================================")
    logger.info(f"NPROC_PER_NODE: {NPROC_PER_NODE}")
    logger.info(f"DEVICE_IDS: {DEVICE_IDS}")
    logger.info(f"GLOBAL_BATCH_SIZE: {GLOBAL_BATCH_SIZE}")
    logger.info(f"BATCH_SIZE: {BATCH_SIZE}")
    logger.info(f"GRAD_ACC_STEPS: {GRAD_ACC_STEPS}")
    logger.info(f"SUPERVISION_RATE: {os.getenv('SUPERVISION_RATE', '1.0')}")
    logger.info("========================================")


def main(args):
    dataset_ds = get_dataset()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        load_in_8bit=False,
        fast_inference=True,
        max_lora_rank=args.lora_rank,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        enforce_eager=ENFORCE_EAGER,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0.0,
        use_gradient_checkpointing="unsloth",
        random_state=args.random_seed,
    )

    vllm_sampling_params = SamplingParams(
        seed=args.random_seed,
        stop=[tokenizer.eos_token],
        include_stop_str_in_output=True,
    )

    output_path = f"outputs/{RUN_NAME}"
    training_args = GRPOConfig(
        vllm_sampling_params=vllm_sampling_params,
        temperature=1.0,
        lr_scheduler_type="cosine_with_min_lr",
        learning_rate=5e-6,
        lr_scheduler_kwargs={"min_lr": 5e-7},
        weight_decay=0.001,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        optim="adamw_torch",
        logging_steps=1,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACC_STEPS,
        num_generations=args.num_of_generations,
        max_prompt_length=11500,
        max_completion_length=4500,
        num_train_epochs=args.num_train_epochs,
        save_steps=10,
        report_to="wandb",
        log_completions=True,
        num_completions_to_print=0,
        output_dir=output_path,
        epsilon=0.2,
        epsilon_high=0.28,
        loss_type=args.loss_type,
        beta=0.0,
        mask_truncated_completions=True,
        shuffle_dataset=True,
        seed=args.random_seed,
        bf16=True,
        fp16=False,
    )

    os.environ["NUM_GENERATIONS"] = str(args.num_of_generations)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=get_reward_funcs(),
        args=training_args,
        train_dataset=dataset_ds,
    )
    resume_from_checkpoint = False
    if (
        os.path.exists(output_path)
        and len([f for f in os.listdir(output_path) if f.startswith("checkpoint-")]) > 0
    ):
        resume_from_checkpoint = True
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="unsloth/Qwen2.5-7B-Instruct")
    parser.add_argument("--run_name", "-r", type=str, required=True)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--num_of_generations", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--max_seq_length", type=int, default=16016)
    parser.add_argument("--loss_type", type=str, default="bnpo")

    args = parser.parse_args()
    RUN_NAME = args.run_name
    os.environ["WANDB_RUN_ID"] = RUN_NAME
    os.environ["WANDB_RESUME"] = "auto"
    os.environ["WANDB_ENTITY"] = "gcnssdvae"
    os.environ["WANDB_PROJECT"] = "DecompseRL"
    os.environ["WANDB_NAME"] = RUN_NAME
    os.environ["WANDB_TAGS"] = f"lora,unsloth,{args.model_name},grpo,{RUN_NAME}"
    log_configs(args)
    main(args)
