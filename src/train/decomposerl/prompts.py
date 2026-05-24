import os

_USER_PROMPT_2WAY_V1 = """\
You are tasked with systematically verifying the accuracy of a claim. You will be provided with a claim to verify and an evidence document to consult.

Here is the evidence document you should consult:

<evidence_document>
{evidence_doc}
</evidence_document>

Here is the claim you need to verify:

<claim>
{claim}
</claim>

Your task is to verify whether this claim is Supported or Refuted through an iterative process of asking questions and gathering information.

# Verification Process

You will follow an iterative cycle of questioning and reasoning:

## Initial Analysis
Begin by analyzing the claim in <think> tags. In your initial analysis:
- Scan the evidence document for passages that seem potentially relevant to the claim
- Decompose the claim into its atomic sub-claims:
  1. Identify explicit connectives (and, or, but, because, which, etc.) and implicit assumptions, comparisons, or vague terms that each need separate verification
  2. Classify each sub-claim by type (e.g., entity, relational, quantitative, causal, temporal, comparative, etc.)
  3. Note which sub-claims are independently falsifiable — if any single one is refuted, the entire claim is refuted
- Write out a numbered checklist of these sub-claims (this list will guide your verification cycle)
- Identify any ambiguous, vague, or underspecified elements in the claim
- Determine what specific question you should ask

It's OK for this section to be quite long.

## Iterative Question-Answer Cycle
After your initial analysis, enter an iterative cycle where you:

1. **Ask a Question**: In <question> tags, pose a single specific verification question that addresses one aspect of the claim. Your question should target:
   - A specific atomic sub-claim that needs verification
   - An ambiguous element that needs clarification
   - An underspecified term or concept
   - Any other information needed to determine the claim's accuracy

2. **Answer the Question**: In <answer> tags, answer your question using **only** the evidence document:
   - Search the evidence document for relevant information. If you find relevant passages, quote them directly.
   - If the evidence document contains sufficient information, use it to answer the question and cite the relevant passage.
   - If the evidence document does NOT contain the necessary information, explicitly state "I don't know" and move on. Do NOT use outside knowledge to fill the gap.

3. **Evaluate Sufficiency**: In <think> tags, reason about whether you now have sufficient information to verify the claim. Consider:
   - List which sub-claims have been verified so far and which remain unverified
   - Are there remaining ambiguous or underspecified elements in the claim?
   - Do you need additional information to make a confident verification judgment?
   - If yes to any of these, determine what question to ask next.
   - If no, proceed to final verification.

4. **Repeat or Conclude**:
   - If more information is needed, return to step 1 and ask another question.
   - If you have sufficient information, proceed to final verification.

Continue the cycle until every sub-claim identified in your initial analysis has been addressed. Once all sub-claims are covered, proceed to final verification. Do not ask redundant questions about sub-claims that have already been resolved.

## Final Verification
Once you have gathered sufficient information, provide your final judgment in <verification> tags. Your judgment must be exactly one of these two labels:
- **Supported**: The claim is factually accurate and well-supported by the evidence
- **Refuted**: The claim is factually incorrect or contradicted by the evidence

# Example Output Structure

Here is an example of the expected output format (with generic placeholder content):

<think>
[Initial analysis of the claim, breaking it down and identifying what needs to be verified first]
</think>

<question>
[First specific verification question]
</question>

<answer>
[Answer based on evidence document only, quoting relevant passages. If not found, state that "I don't know".]
</answer>

<think>
[List which sub-claims are verified, which remain. Determine next question or proceed to final verification.]
</think>

<question>
[Second specific verification question]
</question>

<answer>
[Answer based on evidence document only, quoting relevant passages. If not found, state that "I don't know".]
</answer>

<think>
[List which sub-claims are verified, which remain. Determine next question or proceed to final verification.]
</think>

<question>
[Third specific verification question]
</question>

<answer>
[Answer based on evidence document only, quoting relevant passages. If not found, state that "I don't know".]
</answer>

<think>
[Determination that sufficient information has been gathered to verify the claim]
</think>

<verification>
[Supported OR Refuted]
</verification>

Don't output anything after the final verification tag. Do not include any additional commentary, reasoning, or information beyond the final verification label. Your final output should end immediately after the closing </verification> tag.

Begin your verification process now.
"""

_USER_PROMPT_2WAY_V2 = """\
You are tasked with systematically verifying the accuracy of a claim. You will be provided with a claim to verify and an evidence document to consult.

Here is the evidence document you should consult:

<evidence_document>
{evidence_doc}
</evidence_document>

Here is the claim you need to verify:

<claim>
{claim}
</claim>

Your task is to verify whether this claim is Supported or Refuted through an iterative process of asking questions and gathering information.

# Verification Process

You will follow an iterative cycle of questioning and reasoning:

## Initial Analysis
Begin by analyzing the claim in <think> tags. In your initial analysis:
- Scan the evidence document for passages that seem potentially relevant to the claim
- Decompose the claim into its atomic sub-claims:
  1. Identify explicit connectives (and, or, but, because, which, etc.) and implicit assumptions, comparisons, or vague terms that each need separate verification
  2. Classify each sub-claim by type (e.g., entity, relational, quantitative, causal, temporal, comparative, etc.)
  3. Note which sub-claims are independently falsifiable — if any single one is refuted, the entire claim is refuted
- Write out a numbered checklist of these sub-claims (this list will guide your verification cycle)
- Identify any ambiguous, vague, or underspecified elements in the claim
- Determine what specific question you should ask

It's OK for this section to be quite long.

## Iterative Question-Answer Cycle
After your initial analysis, enter an iterative cycle where you:

1. **Ask a Question**: In <question> tags, pose a single, concise verification question that addresses one aspect of the claim. Your question should target:
   - A specific atomic sub-claim that needs verification
   - An ambiguous element that needs clarification
   - An underspecified term or concept
   - Any other information needed to determine the claim's accuracy
   Each question MUST be an actual question (ending with a question mark), NOT a statement, analysis, or explanation. Keep it concise — one sentence, no embedded reasoning or conclusions. The number of questions depends on the claim's complexity — use as many or as few as needed.

2. **Answer the Question**: In <answer> tags, answer your question using **only** the evidence document:
   - Search the evidence document for relevant information. If you find relevant passages, quote them directly.
   - If the evidence document contains sufficient information, use it to answer the question and cite the relevant passage.
   - If the evidence document does NOT contain the necessary information, explicitly state "I don't know" and move on. Do NOT use outside knowledge to fill the gap.

3. **Evaluate Sufficiency**: In <think> tags, reason about whether you now have sufficient information to verify the claim. Consider:
   - List which sub-claims have been verified so far and which remain unverified
   - Are there remaining ambiguous or underspecified elements in the claim?
   - Do you need additional information to make a confident verification judgment?
   - If yes to any of these, determine what question to ask next.
   - If no, proceed to final verification.

4. **Repeat or Conclude**:
   - If more information is needed, return to step 1 and ask another question.
   - If you have sufficient information, proceed to final verification.

Continue the cycle until every sub-claim identified in your initial analysis has been addressed. Once all sub-claims are covered, proceed to final verification. Do not ask redundant questions about sub-claims that have already been resolved.

## Final Verification
Once you have gathered sufficient information, provide your final judgment in <verification> tags. Your judgment must be exactly one of these two labels:
- **Supported**: The claim is factually accurate and well-supported by the evidence
- **Refuted**: The claim is factually incorrect or contradicted by the evidence

# Example

The following is for illustration only — the number of questions will vary depending on the claim's complexity and the evidence document. Do not feel constrained to ask exactly two questions.

Claim: "The Eiffel Tower, built in 1887, is 330 meters tall."

<think>
The claim has two sub-claims to verify:
1. The Eiffel Tower was built in 1887
2. Its height is 330 meters
Both are independently falsifiable. I'll start with the construction date.
</think>

<question>
When was the Eiffel Tower constructed according to the document?
</question>

<answer>
The document states: "Construction began in January 1887 and was completed on March 31, 1889." The tower was completed in 1889, not 1887.
</answer>

<think>
Sub-claim 1: The claim says "built in 1887" but the tower was completed in 1889. This sub-claim is inaccurate.
Sub-claim 2: Height — still needs verification.
</think>

<question>
What is the height of the Eiffel Tower?
</question>

<answer>
The document states: "The tower stands 330 metres (1,083 ft) tall." This confirms the 330-meter height.
</answer>

<think>
Sub-claim 1: Refuted — completed in 1889, not 1887.
Sub-claim 2: Supported — 330 meters confirmed.
Since sub-claim 1 is independently falsifiable and refuted, the overall claim is Refuted.
</think>

<verification>
Refuted
</verification>

Don't output anything after the final verification tag. Do not include any additional commentary, reasoning, or information beyond the final verification label. Your final output should end immediately after the closing </verification> tag.

Begin your verification process now.
"""

USER_PROMPT_2WAY_TEMPLATE = (
    _USER_PROMPT_2WAY_V2
    if os.getenv("PROMPT_VERSION", "v1") == "v2"
    else _USER_PROMPT_2WAY_V1
)

NUMBER_OF_QUESTIONS_PROMPT_TEMPLATE = """\
You will be given a claim that needs to be verified and an evidence document to consult. Your task is to determine the optimal number of atomic questions needed to fully verify this claim, where each question can be answered using only the evidence document.

<evidence_document>
{evidence_doc}
</evidence_document>

<claim>
{claim}
</claim>

Your goal is to identify the minimum number of atomic questions required to verify the claim completely. Consider two types of questions needed:

1. **Atomic sub-claims**: Break down the main claim into its fundamental, indivisible components that each require verification
2. **Under-specified elements**: Identify vague or ambiguous parts of the claim that need clarification to enable proper verification

Guidelines for your analysis:
- The number must be between 1 and 20
- Aim for the smallest possible number that still ensures complete verification
- Avoid redundant questions that provide diminishing returns
- Each question must be atomic, so it should address a distinct, necessary aspect of verification
- Consider both factual verification and definitional clarification needs
- Each question should be answerable using only the provided evidence document

Output the number of questions as a single integer.
"""


DECOMPOSE_QUESTIONS_PROMPT_TEMPLATE = """\
You will be given a claim that needs to be verified and an evidence document to consult. Your task is to generate the minimal set of atomic questions needed to fully verify this claim, where each question can be answered using only the evidence document.

<evidence_document>
{evidence_doc}
</evidence_document>

<claim>
{claim}
</claim>

Your goal is to produce the minimum set of atomic questions required to verify the claim completely. Consider two types of questions needed:

1. **Atomic sub-claims**: Break down the main claim into its fundamental, indivisible components that each require verification
2. **Under-specified elements**: Identify vague or ambiguous parts of the claim that need clarification to enable proper verification

Guidelines for your analysis:
- Generate between 1 and 20 questions
- Aim for the smallest possible set that still ensures complete verification
- Avoid redundant questions that provide diminishing returns
- Each question must be atomic, so it should address a distinct, necessary aspect of verification
- Consider both factual verification and definitional clarification needs
- Each question must be answerable using only the provided evidence document — do not generate questions that require external knowledge
- Keep each question concise — paraphrase rather than quoting long passages from the claim verbatim

Output the list of questions.
"""


QUESTION_CHECKER_PROMPT_TEMPLATE = """\
Determine if a question can be answered using ONLY the provided document.

<document>
{document}
</document>

<question>
{question}
</question>

## Answerability Criteria

The question is ANSWERABLE (output 1) if:
- The document explicitly states the answer, OR
- The answer can be directly inferred from stated facts

The question is NOT ANSWERABLE (output 0) if:
- The input is not actually a question (e.g., it is a statement, analysis, or explanation)
- The document does not mention relevant information
- The document mentions the topic but lacks specific details needed
- Answering requires external knowledge not in the document

## Important
- If the input is a statement or analysis rather than a question → Output 0
- "Partially answerable" → Output 0 (we need FULL answerability)
- If unsure, default to 0

First, briefly explain your reasoning, then provide your final answer inside <answer> tags containing only 0 or 1.
"""


ANSWER_CHECKER_PROMPT_TEMPLATE = """\
You are tasked with verifying whether a sentence is correct based solely on the provided document.

<document>
{document}
</document>

<sentence>
{sentence}
</sentence>

## Verification Rules

Output 1 (CORRECT) if:
- the sentence accurately reflects some information in the document
- The sentence doesn't introduce any information beyond what's in the document
- No factual errors or contradictions with the document

Output 0 (INCORRECT) if:
- The sentence contradicts the document
- The sentence introduces information not found in the document

First, briefly explain your reasoning, then provide your final answer inside <answer> tags containing only 0 or 1.
"""


ATOMICITY_CHECKLIST_PROMPT_TEMPLATE = """\
You will evaluate a question against five binary atomicity criteria for verifying a given claim.

<claim>
{claim}
</claim>

<question>
{question}
</question>

Evaluate the question on each criterion below. Answer YES or NO for each.

1. **Is a question**: Does the text contain an actual question rather than being purely a statement, analysis, or explanation? A brief setup before the question is acceptable, but the text must contain an actual question.
2. **Single-focus**: Does the question ask about exactly one thing? A question fails this if it asks about multiple distinct aspects, facts, or relationships in a single question.
3. **No conjunctions**: Does the question avoid using "and", "or", "as well as", or similar conjunctions to join distinct sub-claims or topics? Minor conjunctions within a single concept (e.g., "cause and effect") are acceptable.
4. **Verifiable**: Does the question have a definitive yes/no or specific factual answer? It should not be open-ended, subjective, or require an essay-length response.
5. **Grounded**: Does the question reference a specific entity, fact, number, or detail from the claim rather than being generic or abstract?

First, briefly reason about each criterion. Then provide your final answers inside <answer> tags in the exact format:

<answer>
is_question:YES/NO
single_focus:YES/NO
no_conjunctions:YES/NO
verifiable:YES/NO
grounded:YES/NO
</answer>
"""


COVERAGE_PROMPT_TEMPLATE = """\
You are tasked with determining the verdict of a claim based on a set of answers to verification questions.

<answers>
{answers}
</answers>

<claim>
{claim}
</claim>

## Verdict Criteria

**SUPPORTED**: All parts of the claim are confirmed by the answers.
- Every sub-claim has a corresponding information from the answers confirming it
- No contradictions found

**REFUTED**: Any part of the claim is contradicted by an answer.
- At least one answer directly contradicts a sub-claim
- Evidence shows claim is false

**NOT_ENOUGH_INFO**: Answers are insufficient to determine verdict.
- Some sub-claims lack corresponding answers
- Answers are ambiguous or inconclusive

## Process
1. List each sub-claim in the claim
2. Determine if each sub-claim is supported/refuted/unknown based on the answers
3. Aggregate to final verdict

First, briefly explain your reasoning by analyzing how each answer relates to the claim. Then provide your final verdict inside <verdict> tags containing only one of: Supported, Refuted, or Not Enough Information.
"""
