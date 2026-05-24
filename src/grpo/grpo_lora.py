import os
import logging
import subprocess

# Avoid late-training OOMs from allocator fragmentation: variable-length
# completions + tiny-judge forwards produce scattered free segments that the
# default fixed-segment allocator can't coalesce. Must be set before
# torch/unsloth import.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Detect GPU memory BEFORE importing torch/unsloth (env vars must be set first)
try:
    _out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
        text=True,
    )
    _gpu_mem_gb = int(_out.strip().split("\n")[0]) / 1024
except Exception:
    _gpu_mem_gb = 0

if _gpu_mem_gb > 90:
    GPU_MEMORY_UTILIZATION = 0.5
    BATCH_SIZE = 4
    ENFORCE_EAGER = False
else:
    GPU_MEMORY_UTILIZATION = 0.5
    BATCH_SIZE = 4
    ENFORCE_EAGER = False
    # # Smaller GPUs (e.g. L40S 48GB) — standby mode causes wake_up() OOM
    # GPU_MEMORY_UTILIZATION = 0.95
    # BATCH_SIZE = 2
    # ENFORCE_EAGER = True
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ""
    # os.environ["PYTORCH_HIP_ALLOC_CONF"] = ""
    # os.environ["UNSLOTH_VLLM_STANDBY"] = "1"
    # os.environ["PYTORCH_ALLOC_CONF"] = ""

print("****************************************")
print("****************************************")
print(f"Detected GPU Memory: {_gpu_mem_gb:.1f} GB")
print(f"GPU Memory Utilization set to {GPU_MEMORY_UTILIZATION * 100:.0f}%")
print(f"Batch size set to {BATCH_SIZE}")
print("****************************************")
print("****************************************")


# extra configurations
# os.environ["HF_HUB_OFFLINE"] = "1"
# os.environ["UNSLOTH_STABLE_DOWNLOADS"] = "1"
# os.environ["UNSLOTH_DISABLE_AUTO_UPDATES"] = "1"
# os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
# os.environ["WANDB_MODE"] = "offline"


import unsloth
import jsonlines
import torch

logger = logging.getLogger(__name__)
from datasets import Dataset
from decomposer.unsloth.reward_funcs import get_reward_funcs
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel
from vllm import SamplingParams
from decomposer.prompts import USER_PROMPT_2WAY_TEMPLATE

NPROC_PER_NODE = torch.cuda.device_count()
DEVICE_IDS = list(range(NPROC_PER_NODE))

GLOBAL_BATCH_SIZE = 16
GRAD_ACC_STEPS = GLOBAL_BATCH_SIZE / (BATCH_SIZE * NPROC_PER_NODE)
if GLOBAL_BATCH_SIZE % (BATCH_SIZE * NPROC_PER_NODE) != 0:
    raise ValueError(
        f"Global batch size {GLOBAL_BATCH_SIZE} is not divisible by {BATCH_SIZE * NPROC_PER_NODE}"
    )
GRAD_ACC_STEPS = int(GRAD_ACC_STEPS)


def get_dataset(path: str) -> Dataset:
    dataset = []
    with jsonlines.open(path) as reader:
        for line in reader:
            dataset.append(line)

    # Normalize mixed-type columns before building Arrow table
    for row in dataset:
        if "orig_label" in row:
            row["orig_label"] = str(row["orig_label"])

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
        row["document"] = evidence_text  # reward funcs expect "document" key
        return row

    dataset_ds = Dataset.from_list(dataset)
    dataset_ds = dataset_ds.map(_process_row)
    logger.info("========================================")
    logger.info(f"Loaded {len(dataset_ds)} samples from {path}")
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
    dataset_ds = get_dataset(args.train_dataset_path)
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
        # min_p=0.1,
        # top_p=1.0,
        # top_k=-1,
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
        gradient_accumulation_steps=GRAD_ACC_STEPS,  # Increase to 4 for smoother training
        num_generations=args.num_of_generations,
        max_prompt_length=11500,
        max_completion_length=4500,
        num_train_epochs=args.num_train_epochs,  # Set to 1 for a full training run
        save_steps=10,
        report_to="wandb",
        log_completions=True,
        num_completions_to_print=0,
        output_dir=output_path,
        epsilon=0.2,
        epsilon_high=0.28,
        loss_type=args.loss_type,
        beta=0.0,  # 0.01-0.04
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
        # For optional training + evaluation
        # train_dataset = new_dataset["train"],
        # eval_dataset = new_dataset["test"],
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
    DATA_VERSION = os.getenv("DATA_VERSION", "5k")
    parser.add_argument(
        "--train_dataset_path",
        type=str,
        default=f"data/combined_{DATA_VERSION}/step_9/train.jsonl",
    )
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


"""
# for 4 L40s
uvx --with flashinfer-python vllm serve Qwen/Qwen3-32B \
    --port 8000 \
    --tensor-parallel-size 2 \
    --pipeline-parallel-size 2 \
    --max-model-len 16384 \
    --reasoning-parser qwen3 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.92 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max-num-seqs 512 \
    --max-num-batched-tokens 16384 \
    --attention-backend FLASHINFER

# for 1 h100
uvx --with flashinfer-python vllm serve Qwen/Qwen3-32B \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 16384 \
    --reasoning-parser qwen3 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.92 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max-num-seqs 256 \
    --max-num-batched-tokens 16384 \
    --attention-backend FLASHINFER

uvx vllm serve Qwen/Qwen3-32B \
    --port 8000 \
    --tensor-parallel-size 4 \
    --max-model-len 16384 \
    --reasoning-parser qwen3 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.92 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max-num-seqs 256

uvx vllm==0.19.1 serve Qwen/Qwen3-32B \
    --port 8000 \
    --tensor-parallel-size 4 \
    --max-model-len 16384 \
    --reasoning-parser qwen3 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.92 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max-num-seqs 256


uvx vllm==0.19.1 serve "Qwen/Qwen3-Embedding-8B" \
    --data-parallel-size 2 \
    --max-model-len 8192 \
    --trust-remote-code \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 256 \
    --port 8004


ssh -N -L 8000:g24-11:8000 sroydip1@chip.rs.umbc.edu
ssh -N -L 8004:g24-08:8004 sroydip1@chip.rs.umbc.edu

########################################
# 5k New data after 18 different dataset + filtering + diversity + human manual filtering of datasets (kept only human annotated)
########################################

# joint reward + mmr score + necessity salience + coverage reward
DIVERSITY_REWARD=mmr NECESSITY_AGGREGATION=mean CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. uv run decomposer/unsloth/grpo_lora.py --model_name unsloth/Qwen2.5-7B-instruct --run_name 2way_7b_v40

DIVERSITY_REWARD=mmr NECESSITY_AGGREGATION=min CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. uv run decomposer/unsloth/grpo_lora.py --model_name unsloth/Qwen2.5-7B-instruct --run_name 2way_7b_v41

# joint reward + vendi score + necessity salience + coverage reward
DIVERSITY_REWARD=vendi NECESSITY_AGGREGATION=mean CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. uv run decomposer/unsloth/grpo_lora.py --model_name unsloth/Qwen2.5-7B-instruct --run_name 2way_7b_v42

DIVERSITY_REWARD=vendi NECESSITY_AGGREGATION=min CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. uv run decomposer/unsloth/grpo_lora.py --model_name unsloth/Qwen2.5-7B-instruct --run_name 2way_7b_v43

# joint w/ coverage reward + mmr score + necessity salience
JOINT_COVERAGE=1 DIVERSITY_REWARD=mmr NECESSITY_AGGREGATION=mean CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. uv run decomposer/unsloth/grpo_lora.py --model_name unsloth/Qwen2.5-7B-instruct --run_name 2way_7b_v44

JOINT_COVERAGE=1 DIVERSITY_REWARD=mmr NECESSITY_AGGREGATION=min CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. uv run decomposer/unsloth/grpo_lora.py --model_name unsloth/Qwen2.5-7B-instruct --run_name 2way_7b_v45

# joint w/ coverage reward + vendi score + necessity salience
JOINT_COVERAGE=1 DIVERSITY_REWARD=vendi NECESSITY_AGGREGATION=mean CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. uv run decomposer/unsloth/grpo_lora.py --model_name unsloth/Qwen2.5-7B-instruct --run_name 2way_7b_v46

JOINT_COVERAGE=1 DIVERSITY_REWARD=vendi NECESSITY_AGGREGATION=min CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. uv run decomposer/unsloth/grpo_lora.py --model_name unsloth/Qwen2.5-7B-instruct --run_name 2way_7b_v47

GOOD_NUM_Q_REWARD=0 DIVERSITY_REWARD=mmr NECESSITY_AGGREGATION=min CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. uv run decomposer/unsloth/grpo_lora.py --model_name unsloth/Qwen2.5-7B-instruct --run_name 2way_7b_v48

JOINT_COVERAGE=1 DIVERSITY_REWARD=vendi NECESSITY_AGGREGATION=min CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. uv run decomposer/unsloth/grpo_lora.py --model_name unsloth/Qwen2.5-7B-instruct --run_name 2way_7b_v49

GOOD_NUM_Q_REWARD=0 JOINT_COVERAGE=0 DIVERSITY_REWARD=vendi NECESSITY_AGGREGATION=min CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. uv run decomposer/unsloth/grpo_lora.py --model_name unsloth/Qwen2.5-7B-instruct --run_name 2way_7b_v50

DIVERSITY_REWARD=mmr NECESSITY_AGGREGATION=min CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. uv run decomposer/unsloth/grpo_lora.py --model_name unsloth/Qwen2.5-3B-instruct --run_name 2way_3b_v51

SUPERVISION_RATE=0.0 DIVERSITY_REWARD=mmr NECESSITY_AGGREGATION=min CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. uv run decomposer/unsloth/grpo_lora.py --model_name unsloth/Qwen2.5-7B-instruct --run_name 2way_7b_v52

SUPERVISION_RATE=0.7 DIVERSITY_REWARD=mmr NECESSITY_AGGREGATION=min CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. uv run decomposer/unsloth/grpo_lora.py --model_name unsloth/Qwen2.5-7B-instruct --run_name 2way_7b_v53

SUPERVISION_RATE=0.5 DIVERSITY_REWARD=mmr NECESSITY_AGGREGATION=min CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. uv run decomposer/unsloth/grpo_lora.py --model_name unsloth/Qwen2.5-7B-instruct --run_name 2way_7b_v54

SUPERVISION_RATE=0.3 DIVERSITY_REWARD=mmr NECESSITY_AGGREGATION=min CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. uv run decomposer/unsloth/grpo_lora.py --model_name unsloth/Qwen2.5-7B-instruct --run_name 2way_7b_v55

SUPERVISION_RATE=0.0 DIVERSITY_REWARD=mmr NECESSITY_AGGREGATION=min CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. uv run decomposer/unsloth/grpo_lora.py --model_name unsloth/Qwen2.5-14B-instruct --run_name 2way_14b_v56

########################################
# update the prompts
- not question penalization
- better user prompt
########################################

✅✅✅ PROMPT_VERSION=v1 DIVERSITY_REWARD=mmr NECESSITY_AGGREGATION=min PYTHONPATH=. uv run decomposer/unsloth/grpo_lora.py --model_name unsloth/Qwen2.5-7B-instruct --run_name 2way_7b_v61

❌ PROMPT_VERSION=v2 DIVERSITY_REWARD=mmr NECESSITY_AGGREGATION=min PYTHONPATH=. uv run decomposer/unsloth/grpo_lora.py --model_name unsloth/Qwen2.5-7B-instruct --run_name 2way_7b_v62

✅✅✅ GOOD_NUM_Q_REWARD=0 DIVERSITY_REWARD=mmr NECESSITY_AGGREGATION=min PYTHONPATH=. python decomposer/unsloth/grpo_lora.py --model_name unsloth/Qwen2.5-7B-instruct --run_name 2way_7b_v63

✅✅✅ SUPERVISION_RATE=0.7 PROMPT_VERSION=v1 DIVERSITY_REWARD=mmr NECESSITY_AGGREGATION=min PYTHONPATH=. uv run decomposer/unsloth/grpo_lora.py --model_name unsloth/Qwen2.5-7B-instruct --run_name 2way_7b_v64

✋🏻 SUPERVISION_RATE=0.5 PROMPT_VERSION=v1 DIVERSITY_REWARD=mmr NECESSITY_AGGREGATION=min PYTHONPATH=. uv run decomposer/unsloth/grpo_lora.py --model_name unsloth/Qwen2.5-7B-instruct --run_name 2way_7b_v65

✅✅✅ SUPERVISION_RATE=0.3 PROMPT_VERSION=v1 DIVERSITY_REWARD=mmr NECESSITY_AGGREGATION=min PYTHONPATH=. uv run decomposer/unsloth/grpo_lora.py --model_name unsloth/Qwen2.5-7B-instruct --run_name 2way_7b_v66

SUPERVISION_RATE=0.1 PROMPT_VERSION=v1 DIVERSITY_REWARD=mmr NECESSITY_AGGREGATION=min PYTHONPATH=. uv run decomposer/unsloth/grpo_lora.py --model_name unsloth/Qwen2.5-7B-instruct --run_name 2way_7b_v67

✅✅✅ PROMPT_VERSION=v1 DIVERSITY_REWARD=mmr NECESSITY_AGGREGATION=min PYTHONPATH=. uv run decomposer/unsloth/grpo_lora.py --model_name unsloth/Qwen2.5-7B-instruct  --num_train_epochs 1 --run_name 2way_7b_v68

✅✅✅ DATA_VERSION=2k PROMPT_VERSION=v1 DIVERSITY_REWARD=mmr NECESSITY_AGGREGATION=min PYTHONPATH=. uv run decomposer/unsloth/grpo_lora.py --model_name unsloth/Qwen2.5-7B-instruct --run_name 2way_7b_v69

✅✅✅ DATA_VERSION=0.5k PROMPT_VERSION=v1 DIVERSITY_REWARD=mmr NECESSITY_AGGREGATION=min PYTHONPATH=. uv run decomposer/unsloth/grpo_lora.py --model_name unsloth/Qwen2.5-7B-instruct --run_name 2way_7b_v70

✅✅✅ DATA_VERSION=1k PROMPT_VERSION=v1 DIVERSITY_REWARD=mmr NECESSITY_AGGREGATION=min PYTHONPATH=. uv run decomposer/unsloth/grpo_lora.py --model_name unsloth/Qwen2.5-7B-instruct --run_name 2way_7b_v71

✋🏻 DATA_VERSION=3.5k PROMPT_VERSION=v1 DIVERSITY_REWARD=mmr NECESSITY_AGGREGATION=min PYTHONPATH=. uv run decomposer/unsloth/grpo_lora.py --model_name unsloth/Qwen2.5-7B-instruct --run_name 2way_7b_v72

DATA_VERSION=r1k PROMPT_VERSION=v1 DIVERSITY_REWARD=mmr NECESSITY_AGGREGATION=min PYTHONPATH=. uv run decomposer/unsloth/grpo_lora.py --model_name unsloth/Qwen2.5-7B-instruct --run_name 2way_7b_v79

📡 DATA_VERSION=r5k PROMPT_VERSION=v1 DIVERSITY_REWARD=mmr NECESSITY_AGGREGATION=min PYTHONPATH=. uv run decomposer/unsloth/grpo_lora.py --model_name unsloth/Qwen2.5-7B-instruct --run_name 2way_7b_v80

✋🏻✅✅ REWARD_BACKEND=tiny PROMPT_VERSION=v1 DIVERSITY_REWARD=mmr NECESSITY_AGGREGATION=min PYTHONPATH=. uv run decomposer/unsloth/grpo_lora.py --model_name unsloth/Qwen2.5-7B-instruct --run_name 2way_7b_v78


# v73 — necessity-saliency OFF
✅✅✅ NECESSITY_SALIENCY_REWARD=0 PROMPT_VERSION=v1 DIVERSITY_REWARD=mmr NECESSITY_AGGREGATION=min PYTHONPATH=. uv run decomposer/unsloth/grpo_lora.py --model_name unsloth/Qwen2.5-7B-instruct --num_train_epochs 1 --run_name 2way_7b_v73

# v74 — joint-quality OFF
⚠️✅✅ JOINT_QUALITY_REWARD=0 PROMPT_VERSION=v1 DIVERSITY_REWARD=mmr NECESSITY_AGGREGATION=min PYTHONPATH=. uv run decomposer/unsloth/grpo_lora.py --model_name unsloth/Qwen2.5-7B-instruct --num_train_epochs 1 --run_name 2way_7b_v74

# v75 — diversity OFF
DIVERSITY_REWARD_ENABLED=0 PROMPT_VERSION=v1 DIVERSITY_REWARD=mmr NECESSITY_AGGREGATION=min PYTHONPATH=. uv run decomposer/unsloth/grpo_lora.py --model_name unsloth/Qwen2.5-7B-instruct --num_train_epochs 1 --run_name 2way_7b_v75

# v76 — coverage OFF
✅✅✅ COVERAGE_REWARD=0 PROMPT_VERSION=v1 DIVERSITY_REWARD=mmr NECESSITY_AGGREGATION=min PYTHONPATH=. uv run decomposer/unsloth/grpo_lora.py --model_name unsloth/Qwen2.5-7B-instruct --num_train_epochs 1 --run_name 2way_7b_v76

# v77 — good-num-questions OFF
GOOD_NUM_Q_REWARD=0 PROMPT_VERSION=v1 DIVERSITY_REWARD=mmr NECESSITY_AGGREGATION=min PYTHONPATH=. uv run decomposer/unsloth/grpo_lora.py --model_name unsloth/Qwen2.5-7B-instruct --num_train_epochs 1 --run_name 2way_7b_v77


✅✅✅ PROMPT_VERSION=v1 DIVERSITY_REWARD=mmr NECESSITY_AGGREGATION=min PYTHONPATH=. uv run decomposer/unsloth/grpo_lora.py --model_name unsloth/Qwen2.5-3B-instruct --run_name 2way_3b_v81

❌ PROMPT_VERSION=v2 DIVERSITY_REWARD=mmr NECESSITY_AGGREGATION=min PYTHONPATH=. uv run decomposer/unsloth/grpo_lora.py --model_name unsloth/Qwen2.5-3B-instruct --run_name 2way_3b_v82

❌ REWARD_BACKEND=tiny PROMPT_VERSION=v1 DIVERSITY_REWARD=mmr NECESSITY_AGGREGATION=min PYTHONPATH=. uv run decomposer/unsloth/grpo_lora.py --model_name unsloth/Qwen2.5-3B-instruct --run_name 2way_3b_v83

✅✅✅ REWARD_BACKEND=tiny PROMPT_VERSION=v1 DIVERSITY_REWARD=mmr NECESSITY_AGGREGATION=min PYTHONPATH=. uv run decomposer/unsloth/grpo_lora.py --model_name unsloth/Qwen2.5-3B-instruct --run_name 2way_3b_v84

📡 PROMPT_VERSION=v1 DIVERSITY_REWARD=mmr NECESSITY_AGGREGATION=min PYTHONPATH=. uv run decomposer/unsloth/grpo_lora.py --model_name unsloth/Qwen2.5-14B-instruct --run_name 2way_7b_v91

📡 PROMPT_VERSION=v1 DIVERSITY_REWARD=mmr NECESSITY_AGGREGATION=min PYTHONPATH=. uv run decomposer/unsloth/grpo_lora.py --model_name unsloth/Qwen2.5-32B-instruct --run_name 2way_7b_v92

########################################
# update the prompts
- not question penalization
- better user prompt
# 1k data
########################################

DATA_VERSION=1k PROMPT_VERSION=v1 DIVERSITY_REWARD=mmr NECESSITY_AGGREGATION=min PYTHONPATH=. uv run decomposer/unsloth/grpo_lora.py --model_name unsloth/Qwen2.5-3B-instruct --run_name 2way_3b_v101

DATA_VERSION=1k GOOD_NUM_Q_REWARD=0 PROMPT_VERSION=v1 DIVERSITY_REWARD=mmr NECESSITY_AGGREGATION=min PYTHONPATH=. uv run decomposer/unsloth/grpo_lora.py --model_name unsloth/Qwen2.5-3B-instruct --run_name 2way_3b_v102

DATA_VERSION=1k SUPERVISION_RATE=0.7 PROMPT_VERSION=v1 DIVERSITY_REWARD=mmr NECESSITY_AGGREGATION=min PYTHONPATH=. uv run decomposer/unsloth/grpo_lora.py --model_name unsloth/Qwen2.5-3B-instruct --run_name 2way_3b_v103

DATA_VERSION=1k SUPERVISION_RATE=0.5 PROMPT_VERSION=v1 DIVERSITY_REWARD=mmr NECESSITY_AGGREGATION=min PYTHONPATH=. uv run decomposer/unsloth/grpo_lora.py --model_name unsloth/Qwen2.5-3B-instruct --run_name 2way_3b_v104

########################################
# update the prompts
- not question penalization
- better user prompt
# 5k data but 1 epoch only
########################################

DATA_VERSION=5k PROMPT_VERSION=v1 DIVERSITY_REWARD=mmr NECESSITY_AGGREGATION=min PYTHONPATH=. uv run decomposer/unsloth/grpo_lora.py --num_train_epochs 1 --model_name unsloth/Qwen2.5-3B-instruct --run_name 2way_3b_v121

########################################
# update the prompts
- not question penalization
- better user prompt
# 2k data but 1 epoch only
########################################

DATA_VERSION=2k PROMPT_VERSION=v1 DIVERSITY_REWARD=mmr NECESSITY_AGGREGATION=min PYTHONPATH=. uv run decomposer/unsloth/grpo_lora.py --num_train_epochs 1 --model_name unsloth/Qwen2.5-3B-instruct --run_name 2way_3b_v141

########################################
# update the prompts
- not question penalization
- better user prompt
# 1k data but 1 epoch only
########################################

DATA_VERSION=1k PROMPT_VERSION=v1 DIVERSITY_REWARD=mmr NECESSITY_AGGREGATION=min PYTHONPATH=. uv run decomposer/unsloth/grpo_lora.py --num_train_epochs 1 --model_name unsloth/Qwen2.5-3B-instruct --run_name 2way_3b_v161

"""
