"""
Stage 8B: Novel Discovery Engine
Finds genuinely NEW insights that no individual paper stated — only discoverable
by combining findings across all uploaded papers.

This is the core value: what a researcher couldn't see by reading papers one-by-one.
"""
import json
import re
from itertools import combinations

FINGERPRINT_PROMPT = """You are a research analyst creating a precise fingerprint of one paper.

Identify:
1. The single most unique contribution of this paper
2. The specific gap/limitation this paper CANNOT address alone
3. The key variable or finding most needing external validation or combination
4. Any unexplored combination this paper hints at but doesn't pursue

Return ONLY JSON:
{
  "paper_title": "title",
  "unique_contribution": "one specific sentence — what only this paper provides",
  "cannot_address_alone": "what limitation or question this paper leaves open",
  "key_finding_for_combination": "the specific finding most useful to combine with other papers",
  "unexplored_hint": "what the paper gestures toward but doesn't do"
}"""

COMBINATION_PROMPT = """You are a scientific discovery engine. Two research papers have been summarised.
Your task: find what NEW hypothesis emerges ONLY by combining them — something NEITHER paper stated.

Paper A fingerprint: {fp_a}
Paper B fingerprint: {fp_b}

Rules for a valid novel hypothesis:
- Must be SPECIFIC and TESTABLE (not "more research needed")
- Must NOT be stated in either paper
- Must be scientifically plausible given both papers' findings
- Must combine something unique from each paper

Return ONLY JSON:
{
  "hypothesis": "specific testable claim in one sentence",
  "mechanism": "why this is scientifically plausible (2-3 sentences)",
  "evidence_from_a": "specific finding from Paper A that supports this",
  "evidence_from_b": "specific finding from Paper B that supports this",
  "what_test_would_confirm": "specific experiment or study design to validate",
  "novelty_reason": "why no single paper could have said this",
  "impact_if_true": "clinical or scientific consequence if hypothesis is correct",
  "confidence": "high|medium|low",
  "feasibility": "high|medium|low"
}"""

SYNTHESIS_PROMPT = """You are a chief scientific officer reviewing novel hypotheses discovered
by combining findings from {n} research papers.

Hypotheses generated:
{hypotheses}

All paper fingerprints:
{fingerprints}

Tasks:
1. Rank the hypotheses by novelty + impact + feasibility
2. Identify if any two hypotheses combine into an even bigger finding
3. Write one HEADLINE DISCOVERY — the single most important new thing found

Return ONLY JSON:
{
  "headline_discovery": "The single most important novel finding in 2 sentences",
  "ranked_hypotheses": [
    {
      "rank": 1,
      "hypothesis": "full hypothesis text",
      "novelty_score": <0-10>,
      "impact_score": <0-10>,
      "feasibility_score": <0-10>,
      "combined_score": <0-10>,
      "papers_involved": ["paper title 1", "paper title 2"],
      "category": "clinical|methodological|computational|translational"
    }
  ],
  "meta_finding": "if two hypotheses combine into something bigger, state it here or null",
  "recommended_next_step": "single most actionable recommendation for a researcher",
  "commercial_potential": "brief note on any patent/product/trial opportunity",
  "confidence": "high|medium|low"
}"""


def _robust_parse(raw: str, expected: type = dict):
    """Multi-attempt JSON parser."""
    raw = (raw or "").strip()
    for attempt in [
        raw,
        re.sub(r"^```(?:json)?\s*", "", raw),
        re.sub(r"\s*```$", "", re.sub(r"^```(?:json)?\s*", "", raw)),
    ]:
        try:
            parsed = json.loads(attempt.strip())
            if isinstance(parsed, expected):
                return parsed
        except Exception:
            pass
    pattern = r'\{.*\}' if expected == dict else r'\[.*\]'
    match = re.search(pattern, raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, expected):
                return parsed
        except Exception:
            pass
    return {} if expected == dict else []


def _get_paper_fingerprint(page: dict, client) -> dict:
    """Generate a fingerprint for one paper."""
    title = page.get("title", page.get("source_file", "Unknown"))
    content = {
        "title": title,
        "contributions": page.get("contributions", []),
        "methods": page.get("methods", []),
        "key_findings": page.get("key_findings", []),
        "limitations": page.get("limitations", []),
        "future_work": page.get("future_work", []),
        "key_concepts": page.get("key_concepts", []),
    }
    user = f"Create a fingerprint for this paper:\n{json.dumps(content, indent=2)}"
    raw = client.chat_json(FINGERPRINT_PROMPT, user, max_tokens=1000)
    fp = _robust_parse(raw, dict)
    fp["paper_title"] = fp.get("paper_title") or title
    return fp


def _generate_pair_hypothesis(fp_a: dict, fp_b: dict, client) -> dict:
    """Generate novel hypothesis from a pair of paper fingerprints."""
    prompt = COMBINATION_PROMPT.format(
        fp_a=json.dumps(fp_a, indent=2),
        fp_b=json.dumps(fp_b, indent=2),
    )
    raw = client.chat_json(
        "You are a scientific discovery engine. Find novel hypotheses by combining research findings.",
        prompt,
        max_tokens=1500,
    )
    hyp = _robust_parse(raw, dict)
    hyp["paper_a"] = fp_a.get("paper_title", "Paper A")
    hyp["paper_b"] = fp_b.get("paper_title", "Paper B")
    return hyp


def run_novel_discovery(wiki: dict, client) -> dict:
    """
    Full novel discovery pipeline:
    1. Fingerprint each paper
    2. Generate hypotheses for each pair
    3. Synthesise into ranked discoveries
    """
    pages = wiki.get("pages", [])
    if len(pages) < 2:
        return {
            "fingerprints": [],
            "pair_hypotheses": [],
            "synthesis": {},
            "headline_discovery": "Need at least 2 papers for cross-paper discovery.",
            "ranked_hypotheses": [],
        }

    # Step 1: Fingerprint all papers
    fingerprints = []
    for page in pages:
        fp = _get_paper_fingerprint(page, client)
        fingerprints.append(fp)

    # Step 2: Generate hypothesis for each unique pair
    pair_hypotheses = []
    paper_indices = list(range(len(fingerprints)))
    for i, j in combinations(paper_indices, 2):
        if i < len(fingerprints) and j < len(fingerprints):
            hyp = _generate_pair_hypothesis(fingerprints[i], fingerprints[j], client)
            if hyp.get("hypothesis"):
                pair_hypotheses.append(hyp)

    # Step 3: Synthesise
    if not pair_hypotheses:
        return {
            "fingerprints": fingerprints,
            "pair_hypotheses": [],
            "synthesis": {},
            "headline_discovery": "No novel cross-paper hypotheses could be generated.",
            "ranked_hypotheses": [],
        }

    synthesis_prompt = SYNTHESIS_PROMPT.format(
        n=len(pages),
        hypotheses=json.dumps(
            [{"hypothesis": h.get("hypothesis"), "papers": [h.get("paper_a"), h.get("paper_b")]}
             for h in pair_hypotheses],
            indent=2,
        ),
        fingerprints=json.dumps(
            [{"title": f.get("paper_title"), "unique": f.get("unique_contribution")}
             for f in fingerprints],
            indent=2,
        ),
    )

    raw_synth = client.chat_json(
        "You are a chief scientific officer synthesising novel research discoveries.",
        synthesis_prompt,
        max_tokens=3000,
    )
    synthesis = _robust_parse(raw_synth, dict)

    return {
        "fingerprints": fingerprints,
        "pair_hypotheses": pair_hypotheses,
        "synthesis": synthesis,
        "headline_discovery": synthesis.get("headline_discovery", ""),
        "ranked_hypotheses": synthesis.get("ranked_hypotheses", []),
        "meta_finding": synthesis.get("meta_finding"),
        "recommended_next_step": synthesis.get("recommended_next_step", ""),
        "commercial_potential": synthesis.get("commercial_potential", ""),
        "confidence": synthesis.get("confidence", "medium"),
    }
