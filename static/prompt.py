"""Prompt templates.

Two flavours for the summarizer and writer:

* ``BASELINE_*`` — loosely worded, close to typical tutorial prompts ("add citations").
* ``GROUNDED_*`` — every factual sentence must carry a citation drawn from an explicit
  list of valid ids, no outside knowledge, explicit "sources don't say" escape hatch,
  and no model-written reference list (we append a canonical one).

The eval harness runs both to measure the effect on unsupported claims.
"""

PLANNER_SYSTEM = "You are a senior research planner. You turn an open-ended topic into a small set of searchable sub-questions."

PLANNER_USER = """Today's date: {date}
Research topic: {topic}

Break the topic into {min_tasks}-{max_tasks} complementary sub-tasks that together answer it.
Each sub-task needs:
- "title": short, specific heading (max 10 words)
- "intent": one sentence on why this sub-task matters for the overall question
- "query": a web search query (English keywords, 4-10 words, include a year if recency matters)

Avoid overlapping sub-tasks. Order them from foundational to forward-looking.
Return ONLY JSON in this exact shape:
{{"tasks": [{{"title": "...", "intent": "...", "query": "..."}}]}}"""

PLANNER_REPAIR = "Your previous answer could not be used: {error}. Return ONLY the JSON object, with {min_tasks}-{max_tasks} tasks."

# ------------------------------------------------------------------------------------------

SUMMARIZER_SYSTEM = "You are a meticulous research analyst who writes concise, well-sourced notes."

BASELINE_SUMMARIZER_USER = """Sub-task: {title}
Intent: {intent}

Search results:
{sources}

Summarize the search results above, extracting the key points, important numbers, dates and names.
Add source citations like [1], [2]. Use Markdown."""

GROUNDED_SUMMARIZER_USER = """Sub-task: {title}
Intent: {intent}

Numbered sources (the ONLY material you may use):
{sources}

Write research notes answering the sub-task using ONLY the sources above.
Rules:
1. Every sentence that states a fact must end with its citation(s), e.g. "... in 2023 [2]." or "... [1][3]."
2. Valid citation numbers are exactly: {valid_ids}. Never cite any other number.
3. Cite a source only if it explicitly states the fact. Do not combine facts from different sources into one claim unless you cite all of them.
4. No outside knowledge. If the sources do not cover an important aspect, write one sentence: "The sources do not address <aspect>." (no citation).
5. Prefer specifics (numbers, dates, names) over generic statements.
6. 120-250 words, short bullet points, no headings, no reference list."""

# ------------------------------------------------------------------------------------------

WRITER_SYSTEM = (
    "You are an expert research writer who produces clear, well-structured, well-cited reports."
)

BASELINE_WRITER_USER = """Research topic: {topic}

Notes from each sub-task (citations use the global source numbers below):
{sections}

Sources:
{source_list}

Integrate the notes into a structured Markdown research report with: a title, an overview,
one section per sub-task, a conclusion, and references. Remove duplicated information and
keep source citations."""

GROUNDED_WRITER_USER = """Research topic: {topic}

Notes from each sub-task. Citation numbers are GLOBAL and already correct:
{sections}

Write the final Markdown report.
Structure:
# <title>
## Overview  (2-4 sentences framing the question and the main finding)
## <one section per sub-task, in the order given, titled for a reader>
## Key takeaways  (3-5 bullets)

Hard rules:
1. Use ONLY facts present in the notes. Do not add facts, numbers or examples from your own knowledge.
2. Every factual sentence keeps the citation markers of the note sentences it came from, exactly as written ([n]).
3. Valid citation numbers: {valid_ids}. Never invent or renumber citations.
4. When you merge or paraphrase sentences, carry over all of their citations.
5. If notes say the sources do not address something, you may say so without a citation.
6. Do NOT write a References/Sources section — it is appended automatically."""

# ------------------------------------------------------------------------------------------

JUDGE_SYSTEM = "You are a strict fact-checking judge. You verify claims only against the provided source excerpts."

JUDGE_USER = """Sources (excerpts the report was allowed to use):
{sources}

Claims extracted from a research report, each with the source numbers it cites:
{claims}

For each claim decide one label:
- "supported": the cited source(s) clearly state everything the claim asserts.
- "partially_supported": the cited sources support part of the claim, but some detail (a number, date, name, scope) is missing or different.
- "unsupported": the cited sources do not support the claim, contradict it, or the claim cites nothing although it states a fact.
- "not_a_claim": the sentence is framing, a transition, or states that sources are silent — nothing to verify.

Judge ONLY against the cited sources, not your own knowledge, and not against uncited sources.
Return ONLY JSON: {{"verdicts": [{{"claim": <number>, "label": "...", "reason": "<= 15 words"}}]}}"""
