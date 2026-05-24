import jsonlines
from pathlib import Path

MIN_QUESTIONS = 2


def filter_less_complex(data, min_questions):
    """Filter out records where evidence has fewer than min_questions questions."""
    filtered = []
    for item in data:
        decomposed_questions = item["decomposed_questions"]
        num_questions = len(decomposed_questions)
        if num_questions >= min_questions:
            filtered.append(item)
    return filtered


if __name__ == "__main__":
    input_dir = Path("data/combined/step_5")
    output_dir = Path("data/combined/step_6")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Filtering records with more than {MIN_QUESTIONS} questions in {input_dir}..., {len(list(input_dir.glob('*.jsonl')))} files found."
    )

    for input_path in sorted(input_dir.glob("*.jsonl")):
        print(f"Processing {input_path.name}...")
        output_path = output_dir / input_path.name

        with jsonlines.open(input_path, "r") as reader:
            data = [item for item in reader]

        filtered_data = data
        if "train" in input_path.name:
            filtered_data = filter_less_complex(data, min_questions=MIN_QUESTIONS)
            print(
                f"[{input_path.name}] Original records: {len(data)}. After filtering: {len(filtered_data)}."
            )

        with jsonlines.open(output_path, "w") as writer:
            for item in filtered_data:
                writer.write(item)

        print(
            f"[{input_path.name}] Filtered {len(data) - len(filtered_data)} records. Remaining: {len(filtered_data)}."
        )
