# /// script
# dependencies = [
#   "datasets<4.0",
#   "jsonlines",
#   "markdownify",
# ]
# ///

import re
import html
from markdownify import markdownify as md
import os
import jsonlines
from datasets import load_dataset


def scifact():
    claims = load_dataset("allenai/scifact", "claims", trust_remote_code=True)
    corpus = load_dataset("allenai/scifact", "corpus", trust_remote_code=True)
    print(claims)
    print(corpus)

    did2doc = {str(row["doc_id"]): row for row in corpus["train"]}

    for split in ["train", "validation", "test"]:
        cnt = 0
        with jsonlines.open(f"data/scifact/{split}.jsonl", mode="w") as writer:
            for item in claims[split]:
                if len(item["evidence_doc_id"]) == 0:
                    continue
                doc_item = did2doc[str(item["evidence_doc_id"])]
                evidence = doc_item["title"] + "\n\n" + " ".join(doc_item["abstract"])
                evidence = evidence.strip()

                item["evidence"] = evidence
                item["doc"] = doc_item

                writer.write(item)
                cnt += 1
        print(
            f"Split: {split}, total claims: {len(claims[split])}, claims with evidence: {cnt}"
        )


def fever():
    dataset = load_dataset("fever/fever", trust_remote_code=True, name="wiki_pages")
    print(dataset)

    for split_name in dataset:
        print(f"\n--- {split_name} ---")
        print(dataset[split_name][0])
        break

    for row in dataset["wikipedia_pages"]:
        if len(row["text"]):
            print(row)
            break


def fevorous():
    ds = load_dataset("fever/feverous", trust_remote_code=True)
    print(ds)

    for split in ds:
        print(f"\n--- {split} ---")
        from pprint import pprint

        pprint(ds[split][0])
        break


def pubhealth():
    ds = load_dataset("bigbio/pubhealth", "pubhealth_source", trust_remote_code=True)
    print(ds)

    os.makedirs("data/raw/pubhealth", exist_ok=True)
    for split in ds:
        labels = set(item["label"] for item in ds[split])
        print(f"Split: {split}, total records: {len(ds[split])}, labels: {labels}")
        with jsonlines.open(f"data/raw/pubhealth/{split}.jsonl", mode="w") as writer:
            for item in ds[split]:
                writer.write(item)


def pubhealthtab():
    def _html_table_to_markdown(html_string: str) -> str:
        """Convert HTML table to clean markdown with comprehensive post-processing."""

        # ── 1. PRE-PROCESSING: fix common HTML issues before conversion ──

        # Decode HTML entities (&amp; &lt; &gt; &nbsp; &#123; &#x2014; etc.)
        html_string = html.unescape(html_string)

        # ── 2. CONVERT ──
        result = md(html_string)

        # ── 3. POST-PROCESSING ──

        # ── 3a. Mojibake fixes (UTF-8 bytes misread as Latin-1/CP1252) ──
        mojibake_map = {
            # Dashes
            "\u00e2\u0080\u0093": "–",  # en-dash  (â\x80\x93)
            "\u00e2\u0080\u0094": "—",  # em-dash
            "\u00e2\u0080\u0090": "-",  # hyphen
            "\u00e2\u0080\u0091": "-",  # non-breaking hyphen
            "\u00e2\u0080\u0092": "–",  # figure dash
            # Quotes
            "\u00e2\u0080\u0099": "\u2019",  # right single quote '
            "\u00e2\u0080\u0098": "\u2018",  # left single quote  '
            "\u00e2\u0080\u009c": "\u201c",  # left double quote  "
            "\u00e2\u0080\u009d": "\u201d",  # right double quote "
            "\u00e2\u0080\u009e": "\u201e",  # double low-9 quote „
            # Misc symbols
            "\u00e2\u0080\u00a6": "…",  # ellipsis
            "\u00e2\u0080\u00a2": "•",  # bullet
            "\u00e2\u0082\u00ac": "€",  # euro
            "\u00e2\u0084\u00a2": "™",  # trademark
            "\u00c2\u00a9": "©",  # copyright
            "\u00c2\u00ae": "®",  # registered
            "\u00c2\u00b0": "°",  # degree
            "\u00c2\u00b1": "±",  # plus-minus
            "\u00c2\u00b7": "·",  # middle dot
            "\u00c3\u0097": "×",  # multiplication
            "\u00c3\u00b7": "÷",  # division
            # Accented characters (Latin-1 mojibake of UTF-8)
            "\u00c3\u00a9": "é",
            "\u00c3\u00a8": "è",
            "\u00c3\u00aa": "ê",
            "\u00c3\u00ab": "ë",
            "\u00c3\u00a1": "á",
            "\u00c3\u00a0": "à",
            "\u00c3\u00a2": "â",
            "\u00c3\u00a3": "ã",
            "\u00c3\u00ad": "í",
            "\u00c3\u00ac": "ì",
            "\u00c3\u00ae": "î",
            "\u00c3\u00af": "ï",
            "\u00c3\u00b3": "ó",
            "\u00c3\u00b2": "ò",
            "\u00c3\u00b4": "ô",
            "\u00c3\u00b5": "õ",
            "\u00c3\u00ba": "ú",
            "\u00c3\u00b9": "ù",
            "\u00c3\u00bb": "û",
            "\u00c3\u00bc": "ü",
            "\u00c3\u00b1": "ñ",
            "\u00c3\u00a7": "ç",
            "\u00c3\u009f": "ß",
            "\u00c3\u0085": "Å",
            "\u00c3\u0086": "Æ",
            "\u00c3\u0098": "Ø",
            "\u00c3\u00a6": "æ",
            "\u00c3\u00b8": "ø",
            "\u00c3\u00a5": "å",
            # Uppercase accented
            "\u00c3\u0089": "É",
            "\u00c3\u0081": "Á",
            "\u00c3\u008d": "Í",
            "\u00c3\u0093": "Ó",
            "\u00c3\u009a": "Ú",
            "\u00c3\u0091": "Ñ",
        }
        for bad, good in mojibake_map.items():
            result = result.replace(bad, good)

        # Catch remaining â + non-word char patterns (likely minus sign in numeric context)
        result = re.sub(r"â(?=\d)", "−", result)  # â followed by digit → minus sign

        # ── 3b. Stray Â artifact (Latin-1 decode of UTF-8 leading byte C2) ──
        result = re.sub(r"Â\s*", " ", result)  # Â + optional space → single space
        result = result.replace("\u00c2", " ")  # raw Â char

        # ── 3c. Unicode whitespace normalization ──
        result = result.replace("\u00a0", " ")  # NBSP
        result = result.replace("\u200b", "")  # zero-width space
        result = result.replace("\u200c", "")  # zero-width non-joiner
        result = result.replace("\u200d", "")  # zero-width joiner
        result = result.replace("\ufeff", "")  # BOM / zero-width no-break space
        result = result.replace("\u2028", "\n")  # line separator
        result = result.replace("\u2029", "\n\n")  # paragraph separator
        result = result.replace("\t", " ")  # tabs to spaces

        # ── 3d. Unicode typography normalization ──
        # Normalize fancy quotes to ASCII
        result = result.replace("\u2018", "'")  # '
        result = result.replace("\u2019", "'")  # '
        result = result.replace("\u201c", '"')  # "
        result = result.replace("\u201d", '"')  # "
        result = result.replace("\u201e", '"')  # „
        result = result.replace("\u2013", "–")  # en-dash (keep)
        result = result.replace("\u2014", "—")  # em-dash (keep)

        # ── 3e. Collapse whitespace ──
        result = re.sub(r"[ ]{2,}", " ", result)

        # ── 3f. Table-specific: merge broken rows from <br/> tags ──
        # markdownify turns <br/> into \n which splits table header cells across lines
        lines = result.split("\n")
        merged = []
        for line in lines:
            stripped = line.strip()
            if (
                merged
                and merged[-1].strip().startswith("|")
                and stripped
                and not stripped.startswith("|")
                and not stripped.startswith("---")
            ):
                # This line is a continuation of the previous table row
                prev = merged[-1].rstrip()
                if prev.endswith("|"):
                    # Insert before the trailing pipe — find last cell
                    last_pipe = prev.rstrip(" |").rfind("|")
                    merged[-1] = (
                        prev[: last_pipe + 1]
                        + prev[last_pipe + 1 :].rstrip(" |")
                        + " "
                        + stripped
                        + " |"
                    )
                else:
                    merged[-1] = prev + " " + stripped
            else:
                merged.append(line)
        result = "\n".join(merged)

        # ── 3g. Clean up table pipes spacing ──
        result = re.sub(r"\|\s{2,}", "| ", result)
        result = re.sub(r"\s{2,}\|", " |", result)

        # ── 3h. Fix separator row column count ──
        lines = result.strip().split("\n")
        if len(lines) >= 2:
            header_cols = lines[0].count("|") - 1
            sep_line = lines[1].strip()
            if re.match(r"^[\|\s\-:]+$", sep_line):
                sep_cols = sep_line.count("|") - 1
                if header_cols != sep_cols:
                    lines[1] = "| " + " | ".join(["---"] * header_cols) + " |"
                    result = "\n".join(lines)

        # ── 3i. Final cleanup ──
        result = re.sub(r" +$", "", result, flags=re.MULTILINE)  # trailing spaces
        result = re.sub(r"\n{3,}", "\n\n", result)  # excessive blank lines

        return result.strip()

    for split in ["train", "dev", "test"]:
        with jsonlines.open(f"data/raw/pubhealthtab/{split}.jsonl", mode="r") as reader:
            all_data = list(reader)
            data = []
            for item in all_data:
                item["evidence"] = item["table"]["html_code"]
                data.append(item)

            for item in all_data:
                item["evidence"] = _html_table_to_markdown(item["table"]["html_code"])
                data.append(item)

            print(
                f"Split: {split}, total rows: {len(all_data)}, after processing: {len(data)}"
            )

        with jsonlines.open(
            f"data/raw/pubhealthtab/{split}_md.jsonl", mode="w"
        ) as writer:
            for item in data:
                writer.write(item)


def faviq():
    import json
    import sqlite3
    from tqdm import tqdm

    wiki_jsonl = "data/raw/faviq/wikipedia_20190801.jsonl"
    db_path = "data/raw/faviq/wikipedia_20190801.db"

    # Build SQLite DB from wiki JSONL if it doesn't already exist
    if not os.path.exists(db_path):
        print(f"Building SQLite DB at {db_path} ...")
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("CREATE TABLE wiki (id TEXT PRIMARY KEY, data TEXT)")
        batch = []
        with jsonlines.open(wiki_jsonl, mode="r") as reader:
            for item in tqdm(reader, desc="Indexing wiki"):
                batch.append((item["id"], json.dumps(item)))
                if len(batch) >= 10000:
                    cur.executemany("INSERT OR IGNORE INTO wiki VALUES (?, ?)", batch)
                    batch = []
        if batch:
            cur.executemany("INSERT OR IGNORE INTO wiki VALUES (?, ?)", batch)
        conn.commit()
        conn.close()
        print("SQLite DB built successfully.")
    else:
        print(f"SQLite DB already exists at {db_path}, skipping build.")

    # Look up evidence via SQLite
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    for dataset in ["a_set", "r_set"]:
        for split in ["train", "dev", "test"]:
            if not os.path.exists(f"data/raw/faviq/{dataset}/{split}.jsonl"):
                continue

            processed_data = []
            with jsonlines.open(
                f"data/raw/faviq/{dataset}/{split}.jsonl", mode="r"
            ) as reader:
                tot, found = 0, 0
                for item in tqdm(reader):
                    doc_id = item["positive_evidence"]["id"]
                    cur.execute("SELECT data FROM wiki WHERE id = ?", (doc_id,))
                    row = cur.fetchone()
                    tot += 1
                    if row:
                        doc = json.loads(row[0])
                        item["evidence"] = doc["text"].strip()
                        found += 1
                        processed_data.append(item)
                    else:
                        continue

            print(
                f"Dataset: {dataset}, Split: {split}, total claims: {tot}, found evidence: {found}, missing: {tot - found}"
            )
            with jsonlines.open(
                f"data/raw/faviq/{dataset}/{split}_with_evidence.jsonl", mode="w"
            ) as writer:
                for item in processed_data:
                    writer.write(item)

    conn.close()


if __name__ == "__main__":
    # fevorous()
    # fever()
    # scifact()
    # pubhealth()
    # pubhealthtab()
    # faviq()
    pass
