"""Merge a LoRA checkpoint into the base model and push to the HF Hub as private."""

import argparse

from unsloth import FastLanguageModel


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", required=True, help="Path to LoRA adapter checkpoint dir")
    ap.add_argument("--base", default="unsloth/Qwen2.5-7B-instruct", help="Base model name")
    ap.add_argument("--repo", required=True, help="Target HF repo id (e.g., user/name)")
    ap.add_argument("--max-seq-length", type=int, default=4096)
    args = ap.parse_args()

    print(f"Loading adapter {args.adapter} on base {args.base} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        dtype=None,
    )

    print(f"Pushing merged 16-bit model to {args.repo} (private) ...")
    model.push_to_hub_merged(
        args.repo, tokenizer, save_method="merged_16bit", private=True
    )
    print("done.")


if __name__ == "__main__":
    main()
