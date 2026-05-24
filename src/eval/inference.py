"""
Inference script for running a single claim through a LoRA checkpoint.

Usage:
    # With evidence provided
    PYTHONPATH=. python decomposer/inference.py -c outputs/2way_7b/checkpoint-100 --claim "The drug is effective" --evidence "Study showed 80% improvement..."

    # Without evidence (auto-searches Google + Wikipedia)
    PYTHONPATH=. python decomposer/inference.py -c outputs/2way_7b/checkpoint-100 --claim "Coffee improves cognitive function"

    # Interactive mode (will prompt for claim)
    PYTHONPATH=. python decomposer/inference.py -c outputs/2way_7b/checkpoint-100 --interactive

    # Custom number of search results
    PYTHONPATH=. python decomposer/inference.py -c outputs/2way_7b/checkpoint-100 --claim "..." --num_web_results 10 --num_wiki_results 3
"""

import argparse
import json
import os
import re
import textwrap
from typing import Dict, List, Optional, Tuple

import torch
import wikipedia
from ddgs import DDGS
from openai import OpenAI
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from decomposer.prompts import USER_PROMPT_2WAY_TEMPLATE
from decomposer.unsloth.rewards import extract_qa_pairs


# ============================================================================
# Evidence Search Functions
# ============================================================================


def search_wikipedia(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """Search Wikipedia and return top results with summaries."""
    results = []

    try:
        # Search for page titles
        search_results = wikipedia.search(query, results=num_results)

        for title in search_results:
            try:
                # Get page summary
                page = wikipedia.page(title, auto_suggest=False)
                summary = page.summary[:1500]  # Limit length

                results.append(
                    {
                        "title": title,
                        "source": "Wikipedia",
                        "content": summary,
                    }
                )
            except wikipedia.exceptions.DisambiguationError as e:
                # If disambiguation, try the first option
                if e.options:
                    try:
                        page = wikipedia.page(e.options[0], auto_suggest=False)
                        summary = page.summary[:1500]
                        results.append(
                            {
                                "title": e.options[0],
                                "source": "Wikipedia",
                                "content": summary,
                            }
                        )
                    except Exception:
                        pass
            except wikipedia.exceptions.PageError:
                # Page not found, skip
                pass
            except Exception:
                pass
    except Exception as e:
        print(f"  Warning: Wikipedia search failed: {e}")

    return results


def search_duckduckgo(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """Search DuckDuckGo and return top results."""
    results = []

    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=num_results):
            results.append(
                {
                    "title": r.get("title", ""),
                    "source": r.get("href", "Web"),
                    "content": r.get("body", ""),
                }
            )

    return results


def summarize_evidence(claim: str, raw_evidence: str) -> str:
    """Summarize evidence with context to the claim using GPT-4.1-mini."""
    print("  Summarizing evidence with GPT-4.1-mini...")

    client = OpenAI()  # Uses OPENAI_API_KEY env var

    prompt = f"""You are an expert summarizer. Given a search query and raw search results gathered from wikipedia and online search, create a concise, contextualized summary of the search results that are relevant to the search query.

Search Query: {claim}

RAW Search Results:
{raw_evidence}

Instructions:
1. Focus only on information that is directly relevant to the claim
2. Organize the summary by key points that either support or refute the claim
3. Include specific facts, statistics, or findings when available
4. Keep the summary concise but comprehensive (around 500 words)

CONTEXTUALIZED SUMMARY:"""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1500,
        )
        summary = response.choices[0].message.content.strip()
        print(f"    Summary generated ({len(summary)} characters)")
        return summary
    except Exception as e:
        print(f"    Warning: Summarization failed: {e}")
        print("    Using raw evidence instead.")
        return raw_evidence


def search_for_evidence(claim: str, num_web: int = 5, num_wiki: int = 5) -> str:
    """Search for evidence related to the claim from multiple sources."""
    print(f"\nSearching for evidence related to: {claim}")
    print("-" * 60)

    all_results = []

    # Search Wikipedia
    print(f"  Searching Wikipedia (top {num_wiki})...")
    wiki_results = search_wikipedia(claim, num_wiki)
    all_results.extend(wiki_results)
    print(f"    Found {len(wiki_results)} Wikipedia results")

    # Search Web (DuckDuckGo)
    print(f"  Searching Web (top {num_web})...")
    web_results = search_duckduckgo(claim, num_web)
    all_results.extend(web_results)
    print(f"    Found {len(web_results)} web results")

    if not all_results:
        print("  No evidence found from searches!")
        return ""

    # Format evidence document
    evidence_parts = []

    # Add Wikipedia results
    wiki_items = [r for r in all_results if r["source"] == "Wikipedia"]
    if wiki_items:
        evidence_parts.append("## WIKIPEDIA SOURCES")
        for i, item in enumerate(wiki_items, 1):
            evidence_parts.append(f"\n### {i}. {item['title']}")
            evidence_parts.append(item["content"])

    # Add web results
    web_items = [r for r in all_results if r["source"] != "Wikipedia"]
    if web_items:
        evidence_parts.append("\n## WEB SOURCES")
        for i, item in enumerate(web_items, 1):
            evidence_parts.append(f"\n### {i}. {item['title']}")
            evidence_parts.append(f"Source: {item['source']}")
            evidence_parts.append(item["content"])

    raw_evidence = "\n".join(evidence_parts)
    print(f"\n  Total raw evidence length: {len(raw_evidence)} characters")

    # Summarize evidence with context to the claim
    evidence = summarize_evidence(claim, raw_evidence)
    print("-" * 60)

    return evidence


def get_base_model_from_adapter_config(checkpoint_path: str) -> str:
    """Read base model name from adapter_config.json."""
    adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
    if not os.path.exists(adapter_config_path):
        raise FileNotFoundError(
            f"adapter_config.json not found in {checkpoint_path}. "
            "Make sure the path points to a valid LoRA checkpoint directory."
        )
    with open(adapter_config_path, "r") as f:
        config = json.load(f)
    return config.get("base_model_name_or_path", "Qwen/Qwen2.5-7B-Instruct")


def format_evidence(evidence: str) -> str:
    """Format evidence with section headers if not already formatted."""
    if "##" in evidence:
        return evidence
    return f"## EVIDENCE\n{evidence}"


def create_prompt(claim: str, evidence: str) -> List[Dict]:
    """Create the prompt for verification."""
    formatted_evidence = format_evidence(evidence)
    prompt_text = USER_PROMPT_2WAY_TEMPLATE.format(
        evidence_doc=formatted_evidence,
        claim=claim,
    )
    return [{"role": "user", "content": prompt_text}]


def extract_thinking(generation: str) -> List[str]:
    """Extract thinking blocks from generation."""
    pattern = r"<think>\s*(.*?)\s*</think>"
    return [t.strip() for t in re.findall(pattern, generation, re.DOTALL)]


def extract_verification(generation: str) -> Optional[str]:
    """Extract final verification label."""
    try:
        label = (
            generation.split("<verification>")[1].split("</verification>")[0].strip()
        )
        if label.lower() in ["supported", "refuted", "mixed"]:
            return label
        return None
    except Exception:
        return None


def format_output_pretty(
    claim: str,
    qa_pairs: List[Dict[str, str]],
    verification: Optional[str],
) -> str:
    """Format output in a pretty way for terminal display."""
    lines = []

    # Header
    lines.append("=" * 80)
    lines.append("CLAIM VERIFICATION RESULTS")
    lines.append("=" * 80)

    # Claim
    lines.append("\n[CLAIM]")
    lines.append("-" * 40)
    wrapped_claim = textwrap.fill(
        claim, width=76, initial_indent="  ", subsequent_indent="  "
    )
    lines.append(wrapped_claim)

    # Questions and Answers
    if qa_pairs:
        lines.append(f"\n[QUESTIONS & ANSWERS] ({len(qa_pairs)} total)")
        lines.append("-" * 40)
        for i, qa in enumerate(qa_pairs, 1):
            lines.append(f"\n  Q{i}: {qa['question']}")
            wrapped_answer = textwrap.fill(
                qa["answer"],
                width=72,
                initial_indent=f"  A{i}: ",
                subsequent_indent="      ",
            )
            lines.append(wrapped_answer)
    else:
        lines.append("\n[QUESTIONS & ANSWERS]")
        lines.append("-" * 40)
        lines.append("  No questions extracted")

    # Final Verification
    lines.append("\n" + "=" * 80)
    if verification:
        status = "SUPPORTED" if verification.lower() == "supported" else "REFUTED"
        lines.append(f"[FINAL VERDICT]: {status}")
    else:
        lines.append("[FINAL VERDICT]: Could not extract verification")
    lines.append("=" * 80)

    return "\n".join(lines)


def format_output_markdown(
    claim: str,
    evidence: str,
    qa_pairs: List[Dict[str, str]],
    verification: Optional[str],
    raw_output: str,
) -> str:
    """Format output in markdown for copy-pasting."""
    lines = []

    lines.append("# Claim Verification Results\n")

    # Claim
    lines.append("## Claim")
    lines.append(f"> {claim}\n")

    # Evidence (truncated)
    lines.append("## Evidence")
    evidence_preview = evidence[:500] + "..." if len(evidence) > 500 else evidence
    lines.append(f"```\n{evidence_preview}\n```\n")

    # Questions and Answers
    lines.append("## Questions & Answers")
    if qa_pairs:
        for i, qa in enumerate(qa_pairs, 1):
            lines.append(f"### Q{i}: {qa['question']}")
            lines.append(f"**Answer:** {qa['answer']}\n")
    else:
        lines.append("*No questions extracted*\n")

    # Verification
    lines.append("## Final Verdict")
    if verification:
        status = "SUPPORTED" if verification.lower() == "supported" else "REFUTED"
        lines.append(f"**{status}**\n")
    else:
        lines.append("**Could not extract verification**\n")

    # Raw output (collapsible)
    lines.append("<details>")
    lines.append("<summary>Raw Model Output</summary>")
    lines.append("")
    lines.append("```")
    lines.append(raw_output)
    lines.append("```")
    lines.append("</details>")

    return "\n".join(lines)


def load_model_and_tokenizer(
    checkpoint_path: str,
    device: str = "auto",
) -> Tuple[PeftModel, AutoTokenizer]:
    """Load base model with LoRA adapter."""
    base_model_name = get_base_model_from_adapter_config(checkpoint_path)

    print(f"Loading base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter from: {checkpoint_path}")
    model = PeftModel.from_pretrained(model, checkpoint_path)
    model.eval()

    return model, tokenizer


def run_inference(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    claim: str,
    evidence: str,
    max_tokens: int = 6000,
    temperature: float = 0.0,
) -> Tuple[str, List[Dict], Optional[str], List[str]]:
    """Run inference and return raw output, QA pairs, verification, and thinking blocks."""
    messages = create_prompt(claim, evidence)

    # Apply chat template
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            top_p=0.95 if temperature > 0 else None,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    # Decode only the new tokens
    input_length = inputs["input_ids"].shape[1]
    raw_output = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

    qa_pairs = extract_qa_pairs(raw_output)
    verification = extract_verification(raw_output)
    thinking = extract_thinking(raw_output)

    return raw_output, qa_pairs, verification, thinking


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on a single claim using a LoRA checkpoint"
    )
    parser.add_argument(
        "--checkpoint_dir",
        "-c",
        type=str,
        required=True,
        help="Path to LoRA checkpoint directory",
    )
    parser.add_argument(
        "--claim",
        type=str,
        help="The claim to verify",
    )
    parser.add_argument(
        "--evidence",
        type=str,
        help="The evidence document",
    )
    parser.add_argument(
        "--evidence_file",
        type=str,
        help="Path to file containing evidence",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Interactive mode - prompt for claim and evidence",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=6000,
        help="Maximum tokens to generate (default: 6000)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)",
    )
    parser.add_argument(
        "--output_markdown",
        "-o",
        type=str,
        help="Save markdown output to file",
    )
    parser.add_argument(
        "--show_raw",
        action="store_true",
        help="Also print raw model output",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (default: auto)",
    )
    parser.add_argument(
        "--num_web_results",
        type=int,
        default=10,
        help="Number of web search results when auto-searching (default: 5)",
    )
    parser.add_argument(
        "--num_wiki_results",
        type=int,
        default=10,
        help="Number of Wikipedia results when auto-searching (default: 5)",
    )

    args = parser.parse_args()

    # Get claim and evidence
    claim = args.claim
    evidence = args.evidence

    # Load evidence from file if specified
    if args.evidence_file:
        with open(args.evidence_file, "r") as f:
            evidence = f.read()

    # Interactive mode for claim only
    if args.interactive or not claim:
        print("\n" + "=" * 60)
        print("INTERACTIVE MODE")
        print("=" * 60)

        if not claim:
            print("\nEnter the claim to verify:")
            claim = input("> ").strip()

    if not claim:
        print("Error: Claim is required")
        return

    # Auto-search for evidence if not provided
    if not evidence:
        print("\nNo evidence provided. Auto-searching...")
        evidence = search_for_evidence(
            claim,
            num_web=args.num_web_results,
            num_wiki=args.num_wiki_results,
        )

        if not evidence:
            print(
                "Error: Could not find any evidence. Please provide evidence manually."
            )
            return

        print("\n" + "=" * 60)
        print("EVIDENCE FOUND:")
        print("=" * 60)
        print(evidence[:2000] + "..." if len(evidence) > 2000 else evidence)
        print("=" * 60)

    # Load model
    print(f"\nCheckpoint: {args.checkpoint_dir}")
    model, tokenizer = load_model_and_tokenizer(args.checkpoint_dir, args.device)

    print("\nRunning inference...")
    raw_output, qa_pairs, verification, thinking = run_inference(
        model=model,
        tokenizer=tokenizer,
        claim=claim,
        evidence=evidence,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    # Pretty print output
    pretty_output = format_output_pretty(
        claim=claim,
        qa_pairs=qa_pairs,
        verification=verification,
    )
    print("\n" + pretty_output)

    # Show raw output if requested
    if args.show_raw:
        print("\n" + "=" * 80)
        print("RAW MODEL OUTPUT:")
        print("=" * 80)
        print(raw_output)

    # Generate markdown
    markdown_output = format_output_markdown(
        claim=claim,
        evidence=evidence,
        qa_pairs=qa_pairs,
        verification=verification,
        raw_output=raw_output,
    )

    # Save markdown to file if requested
    if args.output_markdown:
        with open(args.output_markdown, "w") as f:
            f.write(markdown_output)
        print(f"\nMarkdown saved to: {args.output_markdown}")


if __name__ == "__main__":
    main()
