"""One-off: merge LoRA adapter from checkpoint-5100 and push to HF privately."""

import shutil
from pathlib import Path

import torch
from huggingface_hub import HfApi
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ADAPTER_DIR = Path(
    "/umbc/ada/ferraro/users/sroydip1/DecomposeRL/outputs/2way_7b_v41/checkpoint-5100"
)
MERGED_DIR = Path(
    "/umbc/ada/ferraro/users/sroydip1/DecomposeRL/outputs/2way_7b_v41/merged-5100"
)
BASE_MODEL = "unsloth/Qwen2.5-7B-instruct"
HF_REPO = "dipta007/decomposerl-7b"


def merge() -> None:
    MERGED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model {BASE_MODEL} in bf16...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    print(f"Loading adapter from {ADAPTER_DIR}...")
    model = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
    print("Merging adapter...")
    merged = model.merge_and_unload()

    print(f"Saving merged model to {MERGED_DIR}...")
    merged.save_pretrained(str(MERGED_DIR), safe_serialization=True)

    print("Saving tokenizer (from adapter dir, includes chat template)...")
    tok = AutoTokenizer.from_pretrained(str(ADAPTER_DIR))
    tok.save_pretrained(str(MERGED_DIR))

    chat_template_src = ADAPTER_DIR / "chat_template.jinja"
    if chat_template_src.exists():
        shutil.copy2(chat_template_src, MERGED_DIR / "chat_template.jinja")
        print("Copied chat_template.jinja")

    print("Files written:")
    for p in sorted(MERGED_DIR.iterdir()):
        print(f"  {p.name}  ({p.stat().st_size / 1e6:.1f} MB)")


def push() -> None:
    api = HfApi()
    print(f"Creating repo {HF_REPO} (private)...")
    api.create_repo(repo_id=HF_REPO, private=True, exist_ok=True)
    print(f"Uploading folder {MERGED_DIR} -> {HF_REPO}...")
    api.upload_folder(
        repo_id=HF_REPO,
        folder_path=str(MERGED_DIR),
        commit_message=f"Upload merged model from {ADAPTER_DIR.name}",
    )
    print("Push complete.")


if __name__ == "__main__":
    import sys

    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step in ("merge", "all"):
        merge()
    if step in ("push", "all"):
        push()
