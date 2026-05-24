# Baseline Adaptations — Caveats to Flag in the Paper

This file records every place where our re-implementation of a published
baseline (in `decomposer/baselines/prompts.py`) deviates from the original
paper. Reviewers will look for these; they need to appear in the paper's
Experimental Setup or a footnote so the comparison is honest and reproducible.

For each baseline we cover:
1. **What the original paper does** (in one paragraph).
2. **What we do instead**, and why.
3. **Where to mention it in the paper.**

If you update a prompt template, update this file in the same commit.

---

## Publication venues

Canonical citation + repo for each of the eight re-implemented baselines.
Use these references in the paper's Related Work / Baselines paragraph.

| Method | Code key | Authors | Venue | Paper | Code |
|---|---|---|---|---|---|
| Self-Ask | `self_ask` | Press, Zhang, Min, Schmidt, Smith, Lewis | Findings of EMNLP 2023 | [arXiv:2210.03350](https://arxiv.org/abs/2210.03350) | [ofir.io/Self-ask-prompting](https://ofir.io/Self-ask-prompting/) |
| Decomposed Prompting (DecomP) | `decomposed_prompting` | Khot, Trivedi, Finlayson, Fu, Richardson, Clark, Sabharwal | ICLR 2023 | [arXiv:2210.02406](https://arxiv.org/abs/2210.02406) | [allenai/DecomP](https://github.com/allenai/DecomP) |
| HiSS | `hiss` | Xuan Zhang, Wei Gao | IJCNLP-AACL 2023 | [ACL Anthology](https://aclanthology.org/2023.ijcnlp-main.64/) (arXiv:2310.00305) | [jadeCurl/HiSS](https://github.com/jadeCurl/HiSS) |
| FOLK | `folk` | Haoran Wang, Kai Shu | Findings of EMNLP 2023 | [ACL Anthology](https://aclanthology.org/2023.findings-emnlp.416/) (arXiv:2310.05253) | [wang2226/FOLK](https://github.com/wang2226/FOLK) |
| ProgramFC | `programfc` | Pan, Wu, Lu, Luu, Wang, Kan, Nakov | ACL 2023 | [ACL Anthology](https://aclanthology.org/2023.acl-long.386/) (arXiv:2305.12744) | [teacherpeterpan/ProgramFC](https://github.com/teacherpeterpan/ProgramFC) |
| Complex Claim Verification (Chen-2024, decomp-only) | `chen_complex` | Jifan Chen, Grace Kim, Aniruddh Sriram, Greg Durrett, Eunsol Choi | NAACL 2024 | [ACL Anthology](https://aclanthology.org/2024.naacl-long.196/) (arXiv:2305.11859) | — |
| ClaimDecomp | `claimdecomp` | Jifan Chen, Aniruddh Sriram, Eunsol Choi, Greg Durrett | EMNLP 2022 | [ACL Anthology](https://aclanthology.org/2022.emnlp-main.229/) (arXiv:2205.06938) | [jifan-chen/subquestions-for-fact-checking](https://github.com/jifan-chen/subquestions-for-fact-checking) |
| QACheck | `qacheck` | Liangming Pan, Xinyuan Lu, Min-Yen Kan, Preslav Nakov | EMNLP 2023 System Demonstrations | [ACL Anthology](https://aclanthology.org/2023.emnlp-demo.23/) (arXiv:2310.07609) | [XinyuanLu00/QACheck](https://github.com/XinyuanLu00/QACheck) |

For completeness — these are *used as-is* (no prompt adaptation) and so are
out of scope for this file, but listed here as the other published baselines
in our comparison:

| Method | Code key | Authors | Venue | Paper |
|---|---|---|---|---|
| MiniCheck-7B | `minicheck` | Liyan Tang, Philippe Laban, Greg Durrett | EMNLP 2024 | [ACL Anthology](https://aclanthology.org/2024.emnlp-main.499/) (arXiv:2404.10774) |

---

## Cross-cutting adaptations (apply to ALL eight prompted baselines)

### A1. Zero-shot vs. few-shot

- **Original**: All six methods use few-shot in-context exemplars (HiSS: 4-shot;
  Self-Ask, DecomP, FOLK, ProgramFC, Chen-2024: varying multi-shot setups).
  These exemplars carry a lot of weight — they implicitly teach the model the
  exact decomposition style.
- **Ours**: Zero-shot. We rely on instructions + structured templates instead
  of demonstrations.
- **Why**: Apples-to-apples with our trained policy, which is zero-shot at
  test time. Few-shot exemplars would inflate baselines unfairly on some
  datasets and deflate them on others (selection bias), and engineering
  per-dataset exemplar pools is a substantial side project.
- **Risk**: ProgramFC and FOLK in particular were designed around exemplars
  that show the exact syntactic format (`def program(): fact_1 = Verify(...)`,
  `Predicate(...) ::: Verify ...`). Zero-shot Qwen / GPT-4.1-mini / Claude
  Haiku may produce syntactically off-spec outputs that the parser drops.
  Monitor the `unparsed` count in the metrics JSON.
- **In the paper**: Experimental Setup paragraph for baselines:
  > "We re-implement eight prompted baselines (Self-Ask, Decomposed Prompting,
  > HiSS, FOLK, ProgramFC, Chen-2024, ClaimDecomp, QACheck) in a unified
  > zero-shot setting against the same base policies
  > (Qwen2.5-{3B,7B,14B,32B}-Instruct, gpt-4.1-mini, claude-haiku-4-5) over
  > the same retrieved evidence. We omit the original papers' few-shot
  > exemplars to keep all systems on the same starting line as our zero-shot
  > policy; this is a deliberate simplification we expect to slightly
  > under-state baseline numbers in proportion to the model's sensitivity to
  > demonstration formats."

### A2. Closed evidence vs. retrieval-in-the-loop

- **Original**: HiSS uses Google Search when the LLM is "not confident";
  FOLK runs a Google Search per predicate; Chen-2024 uses sub-questions as
  retrieval queries against the open web; ProgramFC's `Verify(...)` calls
  invoke a separate QA model. Self-Ask and DecomP are agnostic but typically
  used with retrieval in their evaluation.
- **Ours**: All six baselines operate on the SAME pre-retrieved evidence
  document we feed to our policy. No retrieval inside the baseline.
- **Why**: Controls for retrieval quality as a confound — we want to measure
  decomposition / verification skill, not retrieval. If a baseline could
  retrieve and we couldn't, any win could be attributed to retrieval.
- **Risk**: Methods that were designed around retrieval may under-perform
  because they cannot consult an external source when the evidence is silent.
  Each prompt template now instructs "if evidence is silent, write
  'I don't know' / mark Unknown / mark False" depending on the method's
  native handling.
- **In the paper**: Experimental Setup, same paragraph as above:
  > "All baselines receive the same retrieved evidence as our policy (the
  > evidence document associated with each test claim); no baseline performs
  > additional retrieval. This isolates decomposition/verification capability
  > from retrieval quality."

### A3. Binary verdict vs. native label space

- **Original**: ProgramFC outputs SUPPORTS / REFUTES / NEI;
  FOLK outputs `[SUPPORTED] / [NOT_SUPPORTED]`;
  HiSS uses the dataset's native label set (e.g. RAWFC = True / False / Half;
  LIAR = 6-way);
  Chen-2024 outputs a 6-way veracity scale (true → pants-on-fire) via a
  separate DeBERTa classifier.
- **Ours**: All baselines output exactly one of Supported / Refuted, encoded
  as `<verdict>Supported</verdict>` / `<verdict>Refuted</verdict>` for
  uniform parsing. We map:
  - FOLK `[NOT_SUPPORTED]` → Refuted (and `[SUPPORTED]` → Supported)
  - ProgramFC `label==True` → Supported, `False` → Refuted (NEI not produced)
  - HiSS `Among [Supported, Refuted], the claim is classified as ...` →
    we substitute the dataset's full label set with `[Supported, Refuted]`
    rather than the paper's native multi-class set.
- **Why**: Our paper's task is 2-way (Supported / Refuted) across all 11
  benchmarks — that's the experimental design choice that already lives in
  `prompts.py::USER_PROMPT_2WAY_V1` and `data/combined/step_9/`. Forcing all
  baselines into the same label space is necessary for any apples-to-apples
  table.
- **In the paper**: Footnote near the baseline table:
  > "We constrain every baseline to a 2-way (Supported / Refuted) output
  > space, matching our policy's label set. Multi-class methods (HiSS,
  > Chen-2024) and methods with a NEI label (ProgramFC) are collapsed to
  > 2-way; FOLK's `[NOT_SUPPORTED]` is mapped to Refuted."

---

## Auditability / Inspectable Traces

A key claim of the paper is that our trained policy produces **inspectable
verification traces** — a reader can examine the `<think>` reasoning block,
the explicit `<question>` decompositions, the `<answer>` evidence grounding,
and the final `<verification>` verdict to see which parts of the claim were
checked, against what evidence, and why the verdict came out the way it did.

This table shows which other baselines preserve that auditability and which
collapse to an opaque verdict. The first three columns ask whether the
baseline exposes per-sub-question structure; the fourth asks whether there is
a separate reasoning channel; the fifth describes how the per-sub-question
signals are combined into the final verdict.

| Baseline | Decomposition visible? | Per-Q grounded answer? | Per-Q verdict? | Reasoning trace? | Aggregation logic |
|---|---|---|---|---|---|
| **Ours (trained policy)** | ✓ `<question>` | ✓ `<answer>` (cited passages) | implicit (per-Q "I don't know" allowed) | ✓ `<think>` blocks between cycles | LLM holistic in final `<think>` → `<verification>` |
| Self-Ask | ✓ `Follow up:` | ✓ `Intermediate answer:` | implicit | ✗ | LLM holistic ("So the final answer is:") |
| Decomposed Prompting | ✓ `QS:` | ✓ `A:` | implicit | ✗ | LLM holistic after `QS: [EOQ]` |
| HiSS | ✓ explicit sub-claims + `Question:` per sub-claim | ✓ `Answer:` | implicit per sub-claim | ✗ | LLM holistic in finale |
| FOLK | ✓ `Predicate(args) ::: Verify <gloss>` | ✓ `Prediction(args) is <T/F> because <reason>` | **✓ explicit per-predicate True/False** | ✗ | **strict conjunction (`&&`)** |
| ProgramFC | ✓ `fact_N = Verify("…")` | ✓ `fact_N = True/False # because …` | **✓ explicit per-fact True/False** | ✗ | **boolean `Predict(fact_1 and fact_2 …)`** |
| Chen-2024 | ✓ yes/no sub-questions | ✓ free-text answer per Q | implicit | ✗ | LLM holistic |
| ClaimDecomp | ✓ T5 sub-questions (separate `subquestions` field in JSONL) | ✓ `A<i>: <Yes/No/Unknown> -- <reason>` (separate `parsed_answers` field) | **✓ explicit Yes/No/Unknown per Q** | ✗ | **rule-based: `#Yes / (#Yes + #No) ≥ 0.5`** (Chen 2022 Sec 5.3) — fully transparent, no LLM call needed |
| QACheck | ✓ multi-turn `Question:` per turn (separate `qa_history` field) | ✓ per-turn `Answer:` | implicit | sufficiency-check turns act as a coarse trace | LLM holistic in final-verdict prompt |
| ───── *end-to-end baselines* ───── | | | | | |
| `simple` (Qwen one-shot) | ✗ | ✗ | ✗ | ✗ | direct Supported/Refuted |
| `cot` (Qwen CoT) | ✗ | ✗ | ✗ | ✓ `<reasoning>` block | LLM in single span |
| `iterative` (Qwen iterative) | ✓ `<question>` | ✓ `<answer>` | implicit | ✓ `<think>` | LLM holistic in final `<think>` |
| MiniCheck-7B | ✗ | ✗ | ✗ | ✗ | learned classifier head — fully opaque |

Reading the table:

- **The 8 prompted baselines and our policy all produce some form of
  inspectable trace.** A reader can quote which sub-question / predicate /
  fact a verdict hinges on. This is the structural property the paper
  asserts is missing from end-to-end checkers.
- **FOLK, ProgramFC, and ClaimDecomp go a step further** by exposing
  *explicit* per-sub-element verdicts (predicate True/False, fact True/False,
  yes/no/unknown). These three are the most easily diff-able when comparing
  outputs across systems — a reviewer can scan the per-fact column without
  reading natural-language justifications.
- **ClaimDecomp is the only baseline whose aggregation is rule-based** (Chen
  2022 Sec 5.3: `v̂ = #Yes / (#Yes + #No)`; Supported iff `v̂ ≥ 0.5`). For
  every other baseline including ours, the aggregation step is an LLM call,
  which itself is inspectable but not formally specified.
- **Only `simple` and `MiniCheck` are fully opaque.** `cot` exposes
  reasoning but no decomposition; `iterative` is the natural ancestor of the
  trained policy and produces the same trace structure.

### What's in the result JSONL for each method

The per-claim trace fields preserved on disk (so the inspectability above
survives serialization):

| Method | Fields beyond `id`/`claim`/`evidence`/`gt_label`/`pred_label`/`generation` |
|---|---|
| Self-Ask, DecomP, HiSS, FOLK, ProgramFC, Chen-2024 | `generation` carries the full trace; no separate fields needed |
| ClaimDecomp | `subquestions` (T5 output), `parsed_answers` (per-Q Yes/No/Unknown list), `frac_yes_decisive` (the aggregation score), `num_subquestions` |
| QACheck | `qa_history` (list of `(question, answer)` tuples), `num_turns`, `exited_early` (True if sufficiency check terminated before `max_turns`) |
| Ours (iterative policy) | `questions`, `answers`, `num_of_questions`, `format_reward`, `verification_reward`, plus optional `*_reward` LLM-rated fields |

### What to put in the paper

A short paragraph for the discussion / qualitative-analysis section:

> **Auditability of the baselines.** Among the eleven comparison systems,
> the two end-to-end fine-tuned classifiers (`simple` Qwen, MiniCheck-7B)
> emit only a verdict and are not inspectable. `cot` exposes reasoning but
> no explicit decomposition. The remaining nine — Self-Ask, Decomposed
> Prompting, HiSS, FOLK, ProgramFC, Chen-2024, ClaimDecomp, QACheck, and
> our policy — all produce per-sub-question traces a reader can audit;
> three of them (FOLK, ProgramFC, ClaimDecomp) additionally surface
> explicit per-sub-element verdicts, and ClaimDecomp is the only baseline
> whose aggregation step is a transparent rule rather than an LLM call.
> Our policy is the only system that combines an explicit decomposition,
> a separate reasoning channel (`<think>` blocks), and a final
> verdict in one trace, while also being trained end-to-end against
> outcome-grounded rewards rather than imitating a target decomposition.

---

## Per-method adaptations

### 1. Self-Ask (Press et al., 2022) — `self_ask`

- **Paper**: arXiv:2210.03350. Method generates explicit follow-up questions
  with the markers `"Are follow up questions needed here:"` / `"Follow up:"`
  / `"Intermediate answer:"`, terminated by `"So the final answer is:"`.
  Originally evaluated with retrieval (Google Search via SerpAPI) feeding
  intermediate answers.
- **Our adaptation**: Surface anchors preserved verbatim from Table 10 of
  the paper. Intermediate answers come from the provided evidence instead
  of search.
- **Risk**: Low. Self-Ask is one of the most retrieval-agnostic decomposers;
  it should transfer cleanly.

### 2. Decomposed Prompting (Khot et al., 2022) — `decomposed_prompting`

- **Paper**: arXiv:2210.02406. Repo: github.com/allenai/DecomP. Format:
  `QC: <complex query>` followed by iterated `QS: <sub-q>` / `A: <answer>`,
  terminated by a literal `QS: [EOQ]` line. (Verified from
  `configs/prompts/commaqa_e/.../decomp_fine.txt`.)
- **Our adaptation**: Preserved `QC:` / `QS:` / `A:` / `QS: [EOQ]` markers.
  Dropped the original `(operator)` / `[sub_handler]` annotations on each
  `QS:` line — those are specific to DecomP's CommaQA setup where sub-
  questions were dispatched to specialized handlers; for fact-checking the
  sub-question is just answered by the same LLM.
- **Risk**: Low–medium. The operator/handler annotations are central to
  DecomP's "modular" framing; we are using DecomP only as a decomposition
  prompt format. Be honest about this in the paper:
  > "We re-implement Decomposed Prompting in single-handler mode: the
  > sub-questions are answered by the same LLM rather than dispatched to
  > specialized sub-handlers as in Khot et al.'s original CommaQA setup."

### 3. HiSS (Zhang & Gao, IJCNLP-AACL 2023) — `hiss`

- **Paper**: arXiv:2310.00305. Three-stage prompt verified from Figure 2:
  (i) `"A fact checker will decompose the claim into N subclaims that are
  easier to verify:"` with numbered list; (ii) per-subclaim
  `"To verify subclaim N, a fact-checker will go through a step-by-step
  process to ask and answer a series of questions relevant to its
  factuality. Here are the specific steps he/she raise each question and
  look for an answer:"`, then `"Question:"` / `"Tell me if you are confident
  to answer the question or not. Answer with \`\`yes'' or \`\`no'':"` /
  `"Answer:"`; (iii) finale
  `"Among [label set], the claim is classified as ..."`.
- **Our adaptation**: Three-stage structure preserved verbatim. We
  **dropped the "confident yes/no" probe entirely**: in the original, the
  probe was a stop-sequence trigger that routed "no" answers to a Google
  Search and re-injected the result. Since we operate in a closed-evidence
  regime with no search fallback, keeping the probe would only add a noisy
  inert line ("yes"/"no" emitted but not gating anything). Each Question is
  followed directly by an Answer field that the model fills from the
  evidence, with "I don't know" when the evidence is silent. The dataset's
  native label set is substituted with `[Supported, Refuted]` in the finale.
- **Risk**: Medium. The confidence probe is the most distinctive part of
  HiSS and was the main lever in their RAWFC/LIAR ablations (Figure 3,
  "HiSS w/o search" loses ~5 F1 points). Without retrieval-as-fallback we
  are running HiSS in its weaker `w/o search` setting. Note this in the
  paper:
  > "Our HiSS reimplementation operates in the closed-evidence regime
  > corresponding to the `HiSS w/o search` ablation in Zhang & Gao (Figure
  > 3); we do not provide a search-engine fallback when the model reports
  > low confidence."

### 4. FOLK (Wang & Shu, Findings of EMNLP 2023) — `folk`

- **Paper**: arXiv:2310.05253. Repo: github.com/wang2226/Folk. Format from
  Listings 2 & 6:
  - Decomposition: `Predicate(arguments) ::: Verify <gloss>`
  - Per-predicate verification: `Prediction(args) is [True/False] because
    <grounded answer>`
  - Conjunction: `Predicate1 && Predicate2 && ... is [True/False]`
  - Final: `The claim is [SUPPORTED]` / `[NOT_SUPPORTED]`
- **Our adaptation**: Anchors verbatim. FOLK's original verification step
  retrieves a grounded answer per predicate via Google Search; we replace
  the grounded-answer source with the provided evidence document. We map
  `[NOT_SUPPORTED]` → Refuted.
- **Risk**: Medium. FOLK was designed around per-predicate retrieval; in
  closed-evidence, multiple predicates may share the same evidence span,
  which reduces the value of the per-predicate decomposition. Acceptable
  if mentioned.

### 5. ProgramFC (Pan et al., ACL 2023) — `programfc`

- **Paper**: arXiv:2305.12744. Repo: github.com/teacherpeterpan/ProgramFC.
  Verified prompt format from `models/prompts.py` (HOVER & FEVEROUS):
  ```
  def program():
      fact_1 = Verify("...")
      fact_2 = Verify("...")
      label = Predict(fact_1 and fact_2)
  ```
- **Our adaptation**: Anchors verbatim. Original ProgramFC executes the
  program by routing each `Verify(...)` call to a separate QA / NLI model
  (FLAN-T5 in the paper) and `Predict(...)` is composed by deterministic
  boolean conjunction. We instead ask the same LLM to (i) emit the program
  and (ii) trace its execution against the evidence in a single forward
  pass. ProgramFC's `NEI` (Not Enough Info) label is dropped: we force
  Supported / Refuted.
- **Risk**: Medium. Single-pass execution is a substantial simplification;
  the original paper specifically argues that executing each `Verify` call
  with a dedicated module is what gives ProgramFC its accuracy. Mention this
  explicitly in the paper as "Single-LLM ProgramFC":
  > "We use ProgramFC's prompt format but execute the generated program
  > end-to-end with the same LLM, rather than dispatching each `Verify(...)`
  > call to a dedicated QA module as in Pan et al. (2023). This isolates
  > the program-as-decomposition contribution from the multi-model pipeline."

### 8. QACheck (Pan et al., EMNLP 2023 demo) — `qacheck`

- **Paper**: https://aclanthology.org/2023.emnlp-demo.23/. Repo:
  github.com/XinyuanLu00/QACheck.
- **Pipeline (original)**: Multi-turn loop driven by GPT-3.5:
  1. Sufficiency check: "Have we gathered enough information?"
  2. If no: generate the next sub-question; answer it (the original demo
     consults a web search engine if needed); append `(q, a)` to history.
  3. If yes: emit final verdict.
- **Our adaptation (see `qacheck.py`)**: Same four-prompt state machine, but:
  - **Closed evidence**: each `Answer` step uses ONLY the provided evidence
    document. If the evidence is silent, the model writes `"I don't know"`
    instead of triggering a search.
  - **Same Qwen / frontier-API LLM** as every other prompted baseline (not
    GPT-3.5).
  - **Batched across claims**: at each turn we collect every still-active
    claim and issue one batched LLM call per step (sufficiency, next-question,
    answer). A 1500-claim dataset with N=5 turns is 3N+1 = 16 batched calls
    instead of 1500 × (3N + 1) sequential calls. Per-claim semantics are
    unchanged — each claim's loop continues until it self-declares
    `<enough>Yes</enough>` or exhausts `--max_turns`.
- **Risk**: Low–medium. The architecture is the same as the paper's; the
  closed-evidence simplification is mild because QACheck was always designed
  for cases where the evidence document already contains the needed facts.
  Mention in setup:
  > "Our QACheck reimplementation preserves the four-prompt state machine
  > (sufficiency check, next-question generation, answer, final verdict) but
  > operates in the closed-evidence regime: each Answer step is grounded in
  > the provided evidence rather than a live search engine. We bound the
  > Q-A loop at `max_turns=5` (default) and batch all still-active claims at
  > each turn for efficient inference."

---

### 7. ClaimDecomp (Chen et al., EMNLP 2022) — `claimdecomp`

- **Paper**: arXiv:2205.06938. Repo:
  github.com/jifan-chen/subquestions-for-fact-checking.
- **Pipeline (original)**: A T5 model fine-tuned on the ClaimDecomp corpus
  generates yes/no sub-questions from a complex claim. A separately-trained
  RoBERTa veracity classifier consumes the original claim plus the sub-
  questions and their retrieved evidence to produce the final 6-way PolitiFact
  label (true / mostly-true / half-true / barely-true / false / pants-on-fire).
- **Our adaptation**: Three-stage pipeline (see `claimdecomp.py`):
  1. **Decomposition** with the released T5 decomposer
     `Factiverse/T5-3B-ClaimDecomp` (HF Hub). Raw claim as input; we parse the
     generation as newline / numbered sub-questions and keep up to 6 of them
     (Chen 2022 reports ~6 sub-questions per claim on average).
  2. **Per-sub-question answering** with the same Qwen / frontier-API LLM
     used by the other prompted baselines. The aggregator prompt
     (`CLAIMDECOMP_AGGREGATOR_TEMPLATE` in `prompts.py`) instructs the model
     to emit exactly one `A<i>: <Yes|No|Unknown> -- <reason>` line per
     sub-question and nothing else.
  3. **Rule-based aggregation** in Python (`aggregate_claimdecomp` in
     `prompts.py`): faithful to Chen 2022 Section 5.3 / Table 6
     ("question aggregation"). Compute
     v̂ = (# Yes) / (# Yes + # No), excluding Unknowns from the denominator;
     v̂ ≥ 0.5 → Supported, else Refuted. (The original paper maps v̂ to a
     6-way label via uniform bins; we collapse to 2-way at v̂=0.5.)
- **Why a rule-based aggregator, not the paper's downstream classifier?**
  Chen 2022 deliberately does NOT train a learned veracity head; the
  "question aggregation" rule above is their reported method in Section 5.3
  / Table 6 (Macro-F1 0.30 vs random 0.16 on the 6-way PolitiFact split).
  Implementing the rule-based aggregator is faithful and domain-agnostic,
  and removes the LLM-aggregator confound that affected our earlier draft of
  this baseline (where strict-AND LLM aggregation biased toward Refuted).
- **Why not use ClaimDecomp's gold sub-questions on its own test split?**
  Doing so would create an asymmetric comparison: on `claimdecomp` test, the
  baseline would benefit from gold annotations; on every other dataset it
  would use T5 generations. We use T5 generations everywhere for consistency.
- **Risk**: Medium. T5-3B-ClaimDecomp's input format is undocumented (the
  HF model card is empty); we feed the raw claim, which is the most-common
  usage pattern but possibly suboptimal. Chen 2022 Figure 3 shows the
  paper's own QG-MULTIPLE fine-tuning used `(c, N)` (claim + target count)
  as input and `[S]`-separated questions as output; the Factiverse checkpoint
  may or may not preserve those conventions. The per-claim `subquestions`,
  `parsed_answers`, and `frac_yes_decisive` fields in the result JSONL make
  it easy to audit; if decompositions look noisy, switch the T5 input
  format in `decompose_all()`.
- **In the paper**:
  > "Our ClaimDecomp reimplementation runs the released supervised T5
  > decomposer (`Factiverse/T5-3B-ClaimDecomp`) over every test claim, asks
  > the same LLM used for the other prompted baselines to answer each
  > sub-question Yes / No / Unknown from the provided evidence, and applies
  > the rule-based 'question aggregation' verdict of Chen et al. (2022,
  > Section 5.3 / Table 6): v̂ = #Yes / (#Yes + #No), Supported iff
  > v̂ ≥ 0.5. This matches the original paper's reported aggregator and
  > avoids using a learned PolitiFact-specific classifier that would not
  > transfer to our 11 cross-domain benchmarks."

---

### 6. Complex Claim Verification (Chen et al., NAACL 2024) — `chen_complex`

- **Paper**: arXiv:2305.11859. Section 3.1 "Claim Decomposition": the LLM is
  prompted with 4 in-context decomposition pairs to generate ~10 yes/no
  sub-questions per claim. The sub-questions are then used as retrieval
  queries against the open web; retrieved documents are summarized; a
  fine-tuned DeBERTa-large classifier consumes all summaries to produce a
  6-way veracity label (true / mostly-true / half-true / barely-true /
  false / pants-on-fire).
- **Our adaptation**: We retain only the sub-question decomposition stage,
  preserving the yes/no format and the ~5–10 question target. In our closed-
  evidence 2-way setting we answer each yes/no sub-question from the provided
  evidence and have the LLM aggregate to Supported / Refuted directly,
  skipping both the web-retrieval-as-search-query stage and the DeBERTa
  classifier. The aggregation instruction is **holistic, not strict-AND**:
  the prompt asks the model to weigh sub-question answers on balance and not
  refute purely because the evidence is silent on one sub-question. We
  considered using their DeBERTa classifier directly, but it is fine-tuned on
  PolitiFact-style claims and 6-way labels — running it on our 11 benchmarks
  (FEVER, HoVer, SciFact, PubMedClaim, …) would be far out-of-distribution
  and would produce noise rather than a fair baseline.
- **Risk**: HIGH — this is the biggest spiritual departure of the six. The
  original method's accuracy comes substantially from (a) open-web retrieval
  driven by the decomposed queries and (b) the supervised DeBERTa classifier
  trained for veracity. We use neither. What's left is essentially "ask the
  LLM yes/no sub-questions over the given evidence then aggregate" — close in
  spirit to QACheck / Self-Ask. Be explicit in the paper:
  > "Our reimplementation of Chen et al. (2024) retains only the yes/no
  > claim-decomposition stage from Section 3.1 and adapts it to the closed-
  > evidence binary-verdict setting: each sub-question is answered from the
  > provided evidence by the LLM, and the LLM aggregates the answers to a
  > Supported / Refuted verdict. The original method's open-web retrieval
  > driven by the sub-questions and its fine-tuned DeBERTa veracity classifier
  > are not used. We report this as `Chen-2024 (decomp-only)` to avoid
  > implying we are reproducing their full pipeline."

---

## What to put in the Experimental Setup

A draft paragraph that covers all the cross-cutting flags in one place:

> **Baselines.** We compare against eight prompted decomposition methods —
> Self-Ask, Decomposed Prompting, HiSS, FOLK, ProgramFC, Chen-2024
> (decomp-only), ClaimDecomp (T5 decomposer + rule-based question
> aggregation), and QACheck (multi-turn QA loop) — plus our existing
> end-to-end baselines (Qwen2.5-Instruct simple/CoT/iterative and
> MiniCheck-7B). All eight prompted methods are re-implemented in a unified
> harness (`decomposer/baselines/`) against the same set of base models —
> Qwen2.5-{3B,7B,14B,32B}-Instruct, gpt-4.1-mini, and claude-haiku-4-5 —
> using the same retrieved evidence as our policy. Surface anchors of each
> method are preserved verbatim from the original paper or its official
> repo (see `decomposer/baselines/ADAPTATIONS.md` for the per-method
> audit). We deliberately run all baselines zero-shot to match our policy's
> test-time setting; this likely under-states some baseline numbers
> relative to their few-shot configurations in the original papers.
> ProgramFC is run as a single-LLM trace rather than via its original
> multi-model execution pipeline. Chen-2024 retains only the yes/no
> decomposition stage; the original's open-web retrieval and DeBERTa
> classifier are not used. ClaimDecomp uses the released
> `Factiverse/T5-3B-ClaimDecomp` decomposer and Chen 2022's rule-based
> question-aggregation verdict (v̂ ≥ 0.5).

---

## Maintenance checklist when you change a prompt

1. Update the template in `prompts.py`.
2. Update the corresponding section in this file (which anchors changed and why).
3. If you change the verdict-output convention (e.g. add a third label),
   update `extract_verdict_tag` and `compute_classification_metrics`.
4. Re-run the affected baseline on `data/combined_5k/test_pubmedclaim.jsonl`
   first and eyeball ~5 generations before launching the full sweep — prompt
   changes are easy to typo into degenerate outputs.
