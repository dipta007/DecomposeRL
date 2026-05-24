import logging
import re
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


def is_idk(answer: str) -> bool:
    return answer.strip(" .").lower() == "i don't know" or answer.strip() == ""


def format_reward(generation: str, verbose=True) -> Tuple[float, Dict[str, any]]:
    """Evaluates if a generation follows the required verification format."""

    details = {
        "has_initial_think": False,
        "has_questions": False,
        "has_answers": False,
        "has_intermediate_thinks": False,
        "has_final_verification": False,
        "valid_verification_label": False,
        "no_trailing_content": False,
        "proper_ordering": False,
        "num_qa_cycles": 0,
        "errors": [],
    }

    score = 0.0
    max_score = 8.0  # Total number of checks

    # Extract all tags with their positions
    think_tags = list(re.finditer(r"<think>(.*?)</think>", generation, re.DOTALL))
    question_tags = list(
        re.finditer(r"<question>(.*?)</question>", generation, re.DOTALL)
    )
    answer_tags = list(re.finditer(r"<answer>(.*?)</answer>", generation, re.DOTALL))
    verification_tags = list(
        re.finditer(r"<verification>(.*?)</verification>", generation, re.DOTALL)
    )

    # 1. Check for initial <think> tag (should be at the very start, allowing leading whitespace)
    if think_tags and generation[:think_tags[0].start()].strip() == "":
        details["has_initial_think"] = True
        score += 1.0
    else:
        details["errors"].append("Missing or misplaced initial <think> tag")

    # 2. Check for at least one <question> tag
    if question_tags:
        details["has_questions"] = True
        score += 1.0
    else:
        details["errors"].append("No <question> tags found")

    # 3. Check for at least one <answer> tag
    if answer_tags:
        details["has_answers"] = True
        score += 1.0
    else:
        details["errors"].append("No <answer> tags found")

    # 4. Check for intermediate <think> tags (more than just the initial one)
    if len(think_tags) >= 2:
        details["has_intermediate_thinks"] = True
        score += 1.0
    else:
        details["errors"].append("Missing intermediate <think> tags for evaluation")

    # 5. Check for final <verification> tag
    if verification_tags:
        details["has_final_verification"] = True
        score += 1.0
    else:
        details["errors"].append("Missing final <verification> tag")

    # 6. Check if verification label is valid
    if verification_tags:
        verification_content = verification_tags[-1].group(1).strip()
        valid_labels = ["Supported", "Refuted", "Mixed"]
        if verification_content in valid_labels:
            details["valid_verification_label"] = True
            score += 1.0
        else:
            details["errors"].append(
                f"Invalid verification label: '{verification_content}'. Must be one of {valid_labels}"
            )

    # 7. Check no trailing content after final </verification>
    if verification_tags:
        last_verification_end = verification_tags[-1].end()
        trailing_content = generation[last_verification_end:].strip()
        if trailing_content == "":
            details["no_trailing_content"] = True
            score += 1.0
        else:
            details["errors"].append(
                f"Trailing content found after final </verification>: '{trailing_content[:50]}...'"
            )

    # 8. Check proper ordering and structure
    # Extract positions of all tags
    tag_positions = []

    for match in think_tags:
        tag_positions.append(("think", match.start()))
    for match in question_tags:
        tag_positions.append(("question", match.start()))
    for match in answer_tags:
        tag_positions.append(("answer", match.start()))
    for match in verification_tags:
        tag_positions.append(("verification", match.start()))

    # Sort by position
    tag_positions.sort(key=lambda x: x[1])

    # Check structure:
    # 1. Should start with 'think'
    # 2. Should have alternating question->answer pairs (with thinks in between)
    # 3. Should end with 'verification'

    if tag_positions:
        # First tag should be think
        if tag_positions[0][0] == "think" and tag_positions[-1][0] == "verification":
            # Check for Q&A cycles
            qa_cycle_valid = True
            i = 1  # Skip initial think
            qa_cycles = 0

            while i < len(tag_positions) - 1:  # Stop before verification
                tag_type = tag_positions[i][0]

                if tag_type == "question":
                    # Next should be answer
                    if (
                        i + 1 < len(tag_positions)
                        and tag_positions[i + 1][0] == "answer"
                    ):
                        qa_cycles += 1
                        i += 2
                        # Optionally followed by think
                        if i < len(tag_positions) and tag_positions[i][0] == "think":
                            i += 1
                    else:
                        qa_cycle_valid = False
                        details["errors"].append(
                            f"Question at position {i} not followed by answer"
                        )
                        break
                elif tag_type == "think":
                    # Think can appear between cycles
                    i += 1
                else:
                    qa_cycle_valid = False
                    details["errors"].append(f"Unexpected tag ordering at position {i}")
                    break

            details["num_qa_cycles"] = qa_cycles

            if qa_cycle_valid and qa_cycles >= 1:
                details["proper_ordering"] = True
                score += 1.0
            else:
                details["errors"].append(
                    "Invalid tag ordering or no complete Q&A cycles"
                )
        else:
            details["errors"].append(
                "Does not start with <think> or end with <verification>"
            )

    # Calculate final reward (normalized to 0-1)
    reward = score / max_score

    details["raw_score"] = score
    details["max_score"] = max_score
    details["reward"] = reward

    if verbose:
        logger.debug("Format Reward Details:")
        for key, value in details.items():
            logger.debug(f"  {key}: {value}")

    return reward, details


def get_format_reward(generation) -> float:
    reward, _ = format_reward(generation, verbose=False)
    return reward
