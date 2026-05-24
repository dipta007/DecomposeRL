import re
from typing import Optional

SELF_ASK_TEMPLATE = """\
You are verifying a factual claim against the evidence document below using
Self-Ask (Press et al., 2022). Pose follow-up questions one at a time and answer
each one using ONLY the evidence. If the evidence does not answer a follow-up,
write "I don't know" as the intermediate answer and move on. When you have
gathered enough information, output the final answer.

<evidence>
{evidence}
</evidence>

Question: Is the claim "{claim}" Supported or Refuted by the evidence?
Are follow up questions needed here: Yes.
Follow up: [first specific question targeting one aspect of the claim]
Intermediate answer: [from evidence only, or "I don't know"]
Follow up: [next question]
Intermediate answer: [...]
...
So the final answer is: [Supported or Refuted]

After "So the final answer is:" output exactly one of:
<verdict>Supported</verdict>
<verdict>Refuted</verdict>"""


DECOMPOSED_PROMPTING_TEMPLATE = """\
You are verifying a factual claim using Decomposed Prompting (Khot et al., 2022).
Decompose the verification task into a sequence of simpler sub-questions; emit
each as "QS:" followed by its answer "A:"; terminate the decomposition with
"QS: [EOQ]"; then output the final verdict. Answer every sub-question using
ONLY the evidence document below.

<evidence>
{evidence}
</evidence>

QC: Is the claim "{claim}" Supported or Refuted by the evidence?
QS: [first sub-question targeting one piece of information needed]
A: [from evidence only, or "I don't know"]
QS: [next sub-question]
A: [...]
...
QS: [EOQ]

After "QS: [EOQ]" output exactly one of:
<verdict>Supported</verdict>
<verdict>Refuted</verdict>"""


HISS_TEMPLATE = """\
You are verifying a factual claim using HiSS (Hierarchical Step-by-Step;
Zhang & Gao, EMNLP 2023). Use ONLY the evidence below. The original method
also includes a "confident yes/no" probe that triggers a web search when the
model is not confident; since we operate in a closed-evidence regime, we drop
the probe entirely — if the evidence does not answer a question, simply write
"I don't know" in the Answer field.

<evidence>
{evidence}
</evidence>

Claim: {claim}

A fact checker will decompose the claim into N subclaims that are easier to verify:
   1. [first subclaim]
   2. [second subclaim]
   ...

To verify subclaim 1, a fact-checker will go through a step-by-step process to ask and answer a series of questions relevant to its factuality. Here are the specific steps he/she raise each question and look for an answer:
   Question: [probing question about subclaim 1]
   Answer: [from evidence only, or "I don't know"]
   Question: [next probing question]
   Answer: [...]

To verify subclaim 2, a fact-checker will go through a step-by-step process to ask and answer a series of questions relevant to its factuality. Here are the specific steps he/she raise each question and look for an answer:
   Question: ...
   Answer: ...
   ...

(Repeat for each remaining subclaim.)

Among [Supported, Refuted], the claim is classified as [Supported / Refuted].

After "the claim is classified as" line, output exactly one of:
<verdict>Supported</verdict>
<verdict>Refuted</verdict>"""


FOLK_TEMPLATE = """\
You are verifying a factual claim using FOLK (First-Order-Logic decomposition;
Wang & Shu, NAACL 2024). Translate the claim into a conjunction of first-order
predicates, verify each from the evidence, then combine. Use ONLY the evidence
below; if it does not answer a predicate, mark that predicate Unknown.

<evidence>
{evidence}
</evidence>

Claim: {claim}

Step 1 — Decompose the claim into first-order-logic predicates of the form
"Predicate(arguments) ::: Verify <natural-language gloss>":
Predicate1(arg, ...) ::: Verify [what to check]
Predicate2(arg, ...) ::: Verify [what to check]
...

Step 2 — For each predicate, ground its truth value in the evidence using the
form "Prediction(arguments) is [True/False] because <grounded answer from evidence>":
Prediction1(arg, ...) is [True/False] because [grounded reason from evidence]
Prediction2(arg, ...) is [True/False] because [grounded reason from evidence]
...

Step 3 — Combine via conjunction:
Predicate1 && Predicate2 && ... is [True/False]

Step 4 — Final decision:
The claim is [SUPPORTED] if every predicate is True, else [NOT_SUPPORTED].
The claim is [SUPPORTED / NOT_SUPPORTED]

Then output exactly one of (map [NOT_SUPPORTED] -> Refuted):
<verdict>Supported</verdict>
<verdict>Refuted</verdict>"""


PROGRAMFC_TEMPLATE = """\
You are verifying a factual claim using ProgramFC (Pan et al., ACL 2023).
Generate a short verification program in the format below, then trace its
execution against the evidence. Use ONLY the evidence; if the evidence does not
answer a Verify(...) call, set that fact to False.

<evidence>
{evidence}
</evidence>

Claim: {claim}

Step 1 — Program (use the exact function names Verify and Predict):
def program():
    fact_1 = Verify("[first atomic fact to check]")
    fact_2 = Verify("[second atomic fact to check]")
    ...
    label = Predict(fact_1 and fact_2 and ...)

Step 2 — Execution trace (resolve each Verify call against the evidence):
fact_1 = [True/False]  # because: <grounded reason from evidence>
fact_2 = [True/False]  # because: <grounded reason from evidence>
...
label = [True/False]

Step 3 — Map label to a verdict (label==True -> Supported, label==False -> Refuted),
then output exactly one of:
<verdict>Supported</verdict>
<verdict>Refuted</verdict>"""


CHEN_COMPLEX_TEMPLATE = """\
You are verifying a factual claim using the Claim-Decomposition approach of
Chen et al., NAACL 2024. Decompose the claim into yes/no sub-questions a
human fact-checker would investigate (target ~5-10), answer each from the
evidence ONLY, then aggregate into a verdict.

<evidence>
{evidence}
</evidence>

Claim: {claim}

Step 1 — Yes/no sub-questions:
Q1: [yes/no question, e.g. "Did X do Y?"]
Q2: [yes/no question]
...

Step 2 — Answers (Yes / No / "I don't know" if the evidence is silent):
A1: [Yes/No/I don't know] — [brief grounded reason from evidence]
A2: [Yes/No/I don't know] — [brief grounded reason from evidence]
...

Step 3 — Aggregation: weigh the sub-question answers HOLISTICALLY. The claim is
Supported when, on balance, the evidence supports it across the most important
sub-questions. The claim is Refuted when, on balance, the evidence contradicts
the claim or fails to support it on the most important sub-questions. Treat
"I don't know" answers as weakly evidence-light, not as automatic refutation
— do not refute purely because the evidence is silent on one sub-question.

Output exactly one of:
<verdict>Supported</verdict>
<verdict>Refuted</verdict>"""


PROMPTED_TEMPLATES = {
    "self_ask": SELF_ASK_TEMPLATE,
    "decomposed_prompting": DECOMPOSED_PROMPTING_TEMPLATE,
    "hiss": HISS_TEMPLATE,
    "folk": FOLK_TEMPLATE,
    "programfc": PROGRAMFC_TEMPLATE,
    "chen_complex": CHEN_COMPLEX_TEMPLATE,
}


CLAIMDECOMP_AGGREGATOR_TEMPLATE = """\
You are verifying a factual claim using ClaimDecomp (Chen et al., NAACL 2022).
A supervised T5 decomposer has already split the claim into the yes/no sub-
questions below. Your only job is to answer each yes/no sub-question from the
evidence; the verdict will be computed rule-based from your answers using
Chen 2022's "question aggregation" heuristic (Section 5.3 / Table 6 of the
paper), so adhere strictly to the output format.

<evidence>
{evidence}
</evidence>

Claim: {claim}

Sub-questions (from the ClaimDecomp T5 decomposer):
{subquestions_block}

For each sub-question, output exactly one line of the form:
    A<i>: <Yes|No|Unknown> -- <brief grounded reason from evidence>
Use "Unknown" iff the evidence is silent or ambiguous. Do NOT output anything
else; the rule-based aggregator parses these lines and ignores the rest.

{answer_template}"""


def build_claimdecomp_aggregator_prompt(
    claim: str, evidence: str, subquestions: list
) -> str:
    """Render the ClaimDecomp aggregator prompt."""
    if not subquestions:
        # Decomposer produced nothing — fall back to a single trivial sub-Q.
        subquestions = [f"Is the claim '{claim}' supported by the evidence?"]
    subquestions_block = "\n".join(
        f"Q{i + 1}: {q}" for i, q in enumerate(subquestions)
    )
    answer_template = "\n".join(
        f"A{i + 1}: <Yes|No|Unknown> -- <reason>"
        for i in range(len(subquestions))
    )
    return CLAIMDECOMP_AGGREGATOR_TEMPLATE.format(
        claim=claim,
        evidence=evidence,
        subquestions_block=subquestions_block,
        answer_template=answer_template,
    )


_ANSWER_LINE_RE = re.compile(
    r"^\s*A\d+\s*:\s*(yes|no|unknown)\b", re.IGNORECASE | re.MULTILINE
)


def aggregate_claimdecomp(generation: str) -> tuple:
    answers = [m.group(1).lower() for m in _ANSWER_LINE_RE.finditer(generation)]
    if not answers:
        return None, []
    decisive = [a for a in answers if a in ("yes", "no")]
    if not decisive:
        # All Unknowns -> Refuted (silence-on-everything in closed evidence).
        return "refuted", answers
    n_yes = sum(1 for a in decisive if a == "yes")
    frac_yes = n_yes / len(decisive)
    verdict = "supported" if frac_yes >= 0.5 else "refuted"
    return verdict, answers


def build_prompted_prompt(mode: str, claim: str, evidence: str) -> str:
    template = PROMPTED_TEMPLATES[mode]
    return template.format(claim=claim, evidence=evidence)


def extract_verdict_tag(generation: str) -> Optional[str]:
    try:
        label = generation.split("<verdict>")[-1].split("</verdict>")[0].strip()
        if label.lower() in ("supported", "refuted"):
            return label.lower()
    except Exception:
        pass

    g = generation.lower()
    # FOLK's native final marker.
    if "[not_supported]" in g or "not_supported" in g:
        return "refuted"
    if "[supported]" in g:
        return "supported"

    last_sup = g.rfind("supported")
    last_ref = g.rfind("refuted")
    if last_sup == -1 and last_ref == -1:
        return None
    return "supported" if last_sup > last_ref else "refuted"
