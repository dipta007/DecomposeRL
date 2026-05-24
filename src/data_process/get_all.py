import os
import ast
import json
import uuid
import sqlite3
from collections import Counter
import jsonlines
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
from py_markdown_table.markdown_table import markdown_table


def make_row(id, src, claim, evidence, label, orig_label, metadata):
    return {
        "id": id,
        "src": src,
        "claim": claim.strip(),
        "evidence": evidence.strip(),
        "label": label,
        "orig_label": str(orig_label),
        "metadata": json.dumps(metadata),
    }


def get_pubmedclaim(splits):
    label_map = {"yes": "Supported", "no": "Refuted"}
    data = []
    for split in splits:
        datasets = []
        ds1 = None
        try:
            ds1 = load_dataset("umbc-scify/PubMedClaim", "pqa_labeled", split=split)
            datasets.append(ds1)
        except Exception as e:
            raise RuntimeError(
                f"Error loading PubMedClaim {split} with pqa_labeled config: {e}"
            )
            pass
        # ds2 = load_dataset("umbc-scify/PubMedClaim", "pqa_artificial", split=split)
        # datasets.append(ds2)
        # ds = concatenate_datasets(datasets)
        ds = ds1
        for row in ds:
            orig_label = row["final_decision"]
            if orig_label not in label_map:
                continue

            contexts = row["context"]["contexts"]
            headers = row["context"]["labels"]
            evidence = ""
            for h, c in zip(headers, contexts):
                evidence += f"## {h}\n{c}\n\n"
            evidence_text = evidence.strip()

            item = make_row(
                id=str(uuid.uuid4()),
                src="pubmedclaim",
                claim=row["claim"],
                evidence=evidence_text,
                label=label_map[orig_label],
                orig_label=orig_label,
                metadata=row,
            )
            data.append(item)

    return data


def get_claimdecomp(splits):
    data = []
    # pants-fire, false, barely-true, half-true, mostly-true, true
    label_map = {
        "true": "Supported",
        # "mostly-true": "Refuted",
        "barely-true": "Refuted",
        "false": "Refuted",
        "pants-fire": "Refuted",
    }
    for split in splits:
        with jsonlines.open(f"data/raw/claimdecomp/{split}.jsonl") as f:
            for line in f:
                if line["label"] not in label_map:
                    continue
                item = make_row(
                    id=str(uuid.uuid4()),
                    src="claimdecomp",
                    claim=line["claim"],
                    evidence=line["justification"],
                    label=label_map[line["label"]],
                    orig_label=line["label"],
                    metadata=line,
                )
                data.append(item)
    return data


def get_wice(splits):
    data = []
    label_map = {"supported": "Supported", "not_supported": "Refuted"}
    for split in splits:
        with jsonlines.open(
            f"data/raw/wice/entailment_retrieval/claim/{split}.jsonl"
        ) as f:
            for line in f:
                if line["label"] not in label_map:
                    continue
                evidence = "\n".join(line["evidence"])
                item = make_row(
                    id=str(uuid.uuid4()),
                    src="wice",
                    claim=line["claim"],
                    evidence=evidence,
                    label=label_map[line["label"]],
                    orig_label=line["label"],
                    metadata=line,
                )
                data.append(item)
    return data


def get_sci_fact(splits):
    data = []
    label_map = {"SUPPORT": "Supported", "CONTRADICT": "Refuted"}
    for split in splits:
        with jsonlines.open(f"data/raw/scifact/{split}.jsonl") as f:
            for line in f:
                if line["evidence_label"] not in label_map:
                    continue

                item = make_row(
                    id=str(uuid.uuid4()),
                    src="scifact",
                    claim=line["claim"],
                    evidence=line["evidence"],
                    label=label_map[line["evidence_label"]],
                    orig_label=line["evidence_label"],
                    metadata=line,
                )
                data.append(item)
    return data


def get_matter_of_fact(splits):
    data = []
    for split in splits:
        orig_data = json.load(open(f"data/raw/matter_of_fact/{split}.json"))
        for line in orig_data:
            evidence = "\n".join(line["metadata"]["supporting_facts_from_paper"])
            claim = line["claim_text"]
            original_label = line["gold_label"]
            line["evidence"] = evidence

            item = make_row(
                id=str(uuid.uuid4()),
                src="matter_of_fact",
                claim=claim,
                evidence=evidence,
                label="Supported" if original_label else "Refuted",
                orig_label=original_label,
                metadata=line,
            )
            data.append(item)
    return data


def get_ex_fever(splits):
    data = []
    label_map = {"SUPPORT": "Supported", "REFUTE": "Refuted"}
    conn = sqlite3.connect("data/raw/ex_fever/wiki_db.db")
    cur = conn.cursor()
    # Build case-insensitive lookup: lowercase id -> actual id
    cur.execute("SELECT id FROM documents")
    id_lookup = {row[0].lower(): row[0] for row in cur.fetchall()}
    for split in splits:
        df = pd.read_csv(f"data/raw/ex_fever/{split}.csv")
        for _, row in tqdm(
            df.iterrows(), total=len(df), desc=f"Processing ex_fever {split}"
        ):
            if row["label"] not in label_map:
                continue

            if pd.isna(row["golden entity"]):
                row["golden entity"] = "[]"
            if pd.isna(row["mention"]):
                row["mention"] = "[]"
            if pd.isna(row["result entity"]):
                row["result entity"] = "[]"

            entities = (
                ast.literal_eval(row["golden entity"])
                + ast.literal_eval(row["mention"])
                + ast.literal_eval(row["result entity"])
            )
            entities = set(e.replace("_", " ") for e in entities)

            evidence_parts = []
            for entity in entities:
                # Try exact match first, then case-insensitive
                actual_id = id_lookup.get(entity.lower())
                if actual_id:
                    cur.execute("SELECT text FROM documents WHERE id = ?", (actual_id,))
                    result = cur.fetchone()
                    if result:
                        evidence_parts.append(result[0])
            evidence = "\n".join(evidence_parts)

            item = make_row(
                id=str(uuid.uuid4()),
                src="ex_fever",
                claim=row["claim"],
                evidence=evidence,
                label=label_map[row["label"]],
                orig_label=row["label"],
                metadata=row.to_dict(),
            )
            data.append(item)

    conn.close()
    return data


def get_healthver(splits):
    data = []
    label_map = {"Supports": "Supported", "Refutes": "Refuted"}
    for split in splits:
        df = pd.read_csv(f"data/raw/healthver/{split}.csv")

        for _, row in df.iterrows():
            if row["label"] not in label_map:
                continue

            item = make_row(
                id=str(uuid.uuid4()),
                src="healthver",
                claim=row["claim"],
                evidence=row["evidence"],
                label=label_map[row["label"]],
                orig_label=row["label"],
                metadata=row.to_dict(),
            )
            data.append(item)
    return data


def get_pubhealthfact(splits):
    data = []
    for split in splits:
        label_map = {0: "Supported", 1: "Refuted"}
        with jsonlines.open(f"data/raw/pubhealth/{split}.jsonl") as f:
            for row in f:
                if row["label"] not in label_map:
                    continue
                claim = row["claim"]
                evidence = row["main_text"]
                label = label_map[row["label"]]
                item = make_row(
                    id=str(uuid.uuid4()),
                    src="pubhealthfact",
                    claim=claim,
                    evidence=evidence,
                    label=label,
                    orig_label=row["label"],
                    metadata=row,
                )
                data.append(item)
    return data


def get_fever(splits):
    data = []
    label_map = {"entailment": "Supported", "not_entailment": "Refuted"}
    for split in splits:
        df = pd.read_csv(f"data/raw/fever/{split}.tsv", sep="\t")

        for _, row in df.iterrows():
            if row["label"] not in label_map:
                continue
            item = make_row(
                id=str(uuid.uuid4()),
                src="fever",
                claim=row["sent2"],
                evidence=row["sent1"],
                label=label_map[row["label"]],
                orig_label=row["label"],
                metadata=row.to_dict(),
            )
            data.append(item)
    return data


def get_covidfact(splits):
    data = []
    label_map = {"entailment": "Supported", "not_entailment": "Refuted"}
    for split in splits:
        df = pd.read_csv(f"data/raw/covidfact/{split}.tsv", sep="\t")
        for _, row in df.iterrows():
            if row["label"] not in label_map:
                continue
            item = make_row(
                id=str(uuid.uuid4()),
                src="covidfact",
                claim=row["sent2"],
                evidence=row["sent1"],
                label=label_map[row["label"]],
                orig_label=row["label"],
                metadata=row.to_dict(),
            )
            data.append(item)
    return data


def get_fool_me_twice(splits):
    data = []
    label_map = {"SUPPORTS": "Supported", "REFUTES": "Refuted"}
    for split in splits:
        with jsonlines.open(f"data/raw/fool_me_twice/{split}.jsonl") as f:
            for row in f:
                if row["label"] not in label_map:
                    continue

                evidence = ""
                for ev in row["retrieved_evidence"]:
                    evidence += f"## {ev['section_header']}\n{ev['text']}\n\n"

                item = make_row(
                    id=str(uuid.uuid4()),
                    src="fool_me_twice",
                    claim=row["text"],
                    evidence=evidence.strip(),
                    label=label_map[row["label"]],
                    orig_label=row["label"],
                    metadata=row,
                )
                data.append(item)
    return data


def get_coverbench(splits):
    ds = load_dataset("google/coverbench")
    data = []
    for split in splits:
        for row in ds[split]:
            claim = row["claim"]
            evidence = row["context"]
            # TODO: find out ways to have all data
            if len(evidence.split()) > 16000:
                continue
            label = "Supported" if row["label"] else "Refuted"
            item = make_row(
                id=str(uuid.uuid4()),
                src="coverbench",
                claim=claim,
                evidence=evidence,
                label=label,
                orig_label=row["label"],
                metadata=row,
            )
            data.append(item)

    return data


def get_llmaggrefact(splits):
    ds = load_dataset("lytang/LLM-AggreFact")
    data = []
    label_map = {1: "Supported", 0: "Refuted"}
    for split in splits:
        for row in ds[split]:
            if row["label"] not in label_map:
                continue
            claim = row["claim"]
            evidence = row["doc"]
            label = label_map[row["label"]]
            item = make_row(
                id=str(uuid.uuid4()),
                src="llmaggrefact",
                claim=claim,
                evidence=evidence,
                label=label,
                orig_label=row["label"],
                metadata=row,
            )
            data.append(item)
    return data


def get_pubhealthtab(splits):
    data = []
    label_map = {"SUPPORTS": "Supported", "REFUTES": "Refuted"}
    for split in splits:
        with jsonlines.open(f"data/raw/pubhealthtab/{split}.jsonl") as f:
            for row in f:
                if row["label"] not in label_map:
                    continue
                item = make_row(
                    id=str(uuid.uuid4()),
                    src="pubhealthtab",
                    claim=row["claim"],
                    evidence=row["evidence"],
                    label=label_map[row["label"]],
                    orig_label=row["label"],
                    metadata=row,
                )
                data.append(item)
    return data


def get_faviq(splits):
    data = []
    label_map = {"SUPPORTS": "Supported", "REFUTES": "Refuted"}
    for split in splits:
        with jsonlines.open(f"data/raw/faviq/{split}_with_evidence.jsonl") as f:
            for row in f:
                if row["label"] not in label_map:
                    continue
                item = make_row(
                    id=str(uuid.uuid4()),
                    src=f"faviq_{split.split('/')[0]}",
                    claim=row["claim"],
                    evidence=row["evidence"],
                    label=label_map[row["label"]],
                    orig_label=row["label"],
                    metadata=row,
                )
                data.append(item)
    return data


def get_snopes(splits):
    data = []
    label_map = {"True": "Supported", "False": "Refuted"}
    for split in splits:
        df = pd.read_csv(f"data/raw/snopes/{split}.csv")
        for _, line in df.iterrows():
            if line["rate"] not in label_map:
                continue
            item = make_row(
                id=str(uuid.uuid4()),
                src="snopes",
                claim=line["claim"],
                evidence=line["origin"],
                label=label_map[line["rate"]],
                orig_label=line["rate"],
                metadata=line.to_dict(),
            )
            data.append(item)
    return data


def get_uphill(splits):
    data = []
    label_map = {"true": "Supported", "false": "Refuted"}
    for split in splits:
        with jsonlines.open(f"data/raw/uphill/{split}.jsonl") as f:
            for row in f:
                if row["claim_veracity"] not in label_map:
                    continue
                item = make_row(
                    id=str(uuid.uuid4()),
                    src="uphill",
                    claim=row["claim"],
                    evidence=row["main_text"],
                    label=label_map[row["claim_veracity"]],
                    orig_label=row["claim_veracity"],
                    metadata=row,
                )
                data.append(item)
    return data


def get_ambifc(splits):
    data = []
    label_map = {"supporting": "Supported", "refuting": "Refuted"}
    for split in splits:
        with jsonlines.open(f"data/raw/ambifc/{split}.jsonl") as f:
            for row in f:
                if row["labels"]["passage"] not in label_map:
                    continue
                evidence = ""
                for ev in row["sentences"].values():
                    evidence += f"{ev}\n"

                item = make_row(
                    id=str(uuid.uuid4()),
                    src="ambifc",
                    claim=row["claim"],
                    evidence=evidence.strip(),
                    label=label_map[row["labels"]["passage"]],
                    orig_label=row["labels"]["passage"],
                    metadata=row,
                )
                data.append(item)
    return data


def get_hover(splits):
    data = []
    label_map = {"supports": "Supported", "refutes": "Refuted"}
    for split in splits:
        all_data = json.load(open(f"data/raw/hover/{split}.json"))
        for row in all_data:
            if row["label"] not in label_map:
                continue
            item = make_row(
                id=str(uuid.uuid4()),
                src="hover",
                claim=row["claim"],
                evidence=row["evidence"],
                label=label_map[row["label"]],
                orig_label=row["label"],
                metadata=row,
            )
            data.append(item)
    return data


def get_feverous(splits):
    data = []
    label_map = {"supports": "Supported", "refutes": "Refuted"}
    for split in splits:
        all_data = json.load(open(f"data/raw/feverous/{split}.json"))
        for row in all_data:
            if row["label"] not in label_map:
                continue
            item = make_row(
                id=str(uuid.uuid4()),
                src="feverous",
                claim=row["claim"],
                evidence=row["evidence"],
                label=label_map[row["label"]],
                orig_label=row["label"],
                metadata=row,
            )
            data.append(item)
    return data


def get_scitab(splits):
    data = []
    label_map = {"supports": "Supported", "refutes": "Refuted"}
    for split in splits:
        raw_data = json.load(open(f"data/raw/scitab/{split}.json"))
        for row in raw_data:
            if row["label"] not in label_map:
                continue
            evidence = row["table_caption"].strip()
            ev_data = []
            columns = row["table_column_names"]
            for values in row["table_content_values"]:
                curr_ev = {}
                for col, val in zip(columns, values):
                    curr_ev[col] = val
                ev_data.append(curr_ev)

            evidence += markdown_table(ev_data).get_markdown().strip("`") + "\n\n"

            item = make_row(
                id=str(uuid.uuid4()),
                src="scitab",
                claim=row["claim"],
                evidence=evidence.strip(),
                label=label_map[row["label"]],
                orig_label=row["label"],
                metadata=row,
            )
            data.append(item)
    return data


# ---------------------------------------------------------------------------
# Dataset → function mapping
# ---------------------------------------------------------------------------
DATASET_FN = {
    "pubmedclaim": get_pubmedclaim,
    "claimdecomp": get_claimdecomp,
    "wice": get_wice,
    "scifact": get_sci_fact,
    "matter_of_fact": get_matter_of_fact,
    "fever": get_fever,
    "ex_fever": get_ex_fever,
    "pubhealthfact": get_pubhealthfact,
    "coverbench": get_coverbench,
    "llmaggrefact": get_llmaggrefact,
    "healthver": get_healthver,
    "covidfact": get_covidfact,
    "fool_me_twice": get_fool_me_twice,
    "pubhealthtab": get_pubhealthtab,
    "faviq_a_set": get_faviq,
    "faviq_r_set": get_faviq,
    "snopes": get_snopes,
    "uphill": get_uphill,
    "ambifc": get_ambifc,
    "hover": get_hover,
    "feverous": get_feverous,
    "scitab": get_scitab,
}

# Split args per dataset.  None = no args (test-only fixed split).
TRAIN_SPLITS = {
    "llmaggrefact": ["dev"],
    "fool_me_twice": ["train", "dev"],
    "pubmedclaim": ["val"],
    "claimdecomp": ["train", "dev"],
    "wice": ["train", "dev"],
    "scifact": ["train", "validation"],
    # "matter_of_fact": ["train", "validation"],
    # "fever": ["train", "dev"],
    "ex_fever": ["train", "dev"],
    "pubhealthfact": ["train", "validation"],
    # "healthver": ["train", "dev"],
    # "covidfact": ["train", "dev"],
    "pubhealthtab": ["train_md", "dev_md"],
    "faviq_a_set": ["a_set/train"],
    # "faviq_r_set": ["r_set/train", "r_set/dev"],
    # "snopes": ["train"],
    # "uphill": ["train"],
    # "ambifc": ["train.certain", "train.uncertain", "dev.certain", "dev.uncertain"],
    "ambifc": ["train.certain", "dev.certain"],
    "hover": ["train"],
    "feverous": ["train"],
    "scitab": ["train"],
}

TEST_SPLITS = {
    "fool_me_twice": ["test"],
    "pubmedclaim": ["test"],
    "claimdecomp": ["test"],
    "wice": ["test"],
    "matter_of_fact": ["test"],
    "fever": ["test"],
    "ex_fever": ["test"],
    "pubhealthfact": ["test"],
    "healthver": ["test"],
    "covidfact": ["test"],
    "coverbench": ["eval"],
    "llmaggrefact": ["test"],
    "pubhealthtab": ["test_md"],
    "faviq_a_set": ["a_set/dev"],
    "faviq_r_set": ["r_set/test"],
    "ambifc": ["test.certain", "test.uncertain"],
    "hover": ["dev"],
    "feverous": ["dev"],
}

TRAIN_DATASETS = list(TRAIN_SPLITS)
TEST_DATASETS = list(TEST_SPLITS)


def _fetch(name, splits):
    """Call the dataset function with the given splits (or no args if None)."""
    fn = DATASET_FN[name]
    return fn() if splits is None else fn(splits)


def train():
    combined_data = []
    for name in TRAIN_DATASETS:
        data = _fetch(name, TRAIN_SPLITS[name])
        lbl = Counter(item["label"] for item in data)
        print(
            f"{name} train: {len(data)} ({', '.join(f'{l}: {c}' for l, c in sorted(lbl.items()))})"
        )
        combined_data.extend(data)

    print(f"Total train size: {len(combined_data)}")
    print("====================")
    print("====================")

    with jsonlines.open("data/combined/step_0/train.jsonl", mode="w") as writer:
        for item in combined_data:
            writer.write(item)


def test():
    all_data = {}
    for name in TEST_DATASETS:
        data = _fetch(name, TEST_SPLITS[name])
        lbl = Counter(item["label"] for item in data)
        print(
            f"{name} test: {len(data)} ({', '.join(f'{l}: {c}' for l, c in sorted(lbl.items()))})"
        )
        all_data[name] = data

    total = sum(len(d) for d in all_data.values())
    print(f"Total test size: {total}")
    print("====================")
    print("====================")

    for name, data in all_data.items():
        with jsonlines.open(
            f"data/combined/step_0/test_{name}.jsonl", mode="w"
        ) as writer:
            for item in data:
                writer.write(item)


if __name__ == "__main__":
    os.makedirs("data/combined/step_0", exist_ok=True)
    train()
    test()
