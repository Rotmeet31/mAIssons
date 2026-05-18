"""
MediaLens Fact-Check Agent.

Phase-locked four-phase orchestrator: orientation → database evidence →
web evidence → structured verdict. Verdict is validated via FactCheckVerdict
Pydantic model — no regex extraction.
"""
import json
import os
from typing import Any

from loguru import logger
from openai import OpenAI
from pydantic import ValidationError

from agent import (
    _TOOL_FUNCTIONS as _AGENT_TOOL_FUNCTIONS,
    TOOL_SCHEMAS as _AGENT_TOOL_SCHEMAS,
)
from agent_utils import run_tool_phase
from config import LLM_MODEL, LLM_TEMPERATURE, OPENROUTER_BASE_URL
from schemas import FactCheckVerdict, RunTrace


# ── Web search tool ────────────────────────────────────────────────────────

def search_web(query: str, max_results: int = 5) -> list[dict]:
    """Search the live web via DuckDuckGo. Returns list of {title, url, snippet}."""
    from ddgs import DDGS
    try:
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=max_results))
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
            }
            for r in raw
        ]
    except Exception as exc:
        logger.warning("search_web failed for {!r}: {}", query, exc)
        return []


# ── Fact-check tool registry (extends research agent tools) ───────────────

_FACT_CHECK_TOOL_FUNCTIONS: dict[str, Any] = {
    **_AGENT_TOOL_FUNCTIONS,
    "search_web": search_web,
}

_WEB_SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": (
            "Search the live web via DuckDuckGo for a specific factual claim or topic. "
            "Use this to find real-world evidence outside the internal article database. "
            "Returns up to max_results results with title, url, and snippet."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Specific search query — use exact claim wording.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max results to return (default 5).",
                },
            },
            "required": ["query"],
        },
    },
}

FACT_CHECK_TOOL_SCHEMAS: list[dict] = _AGENT_TOOL_SCHEMAS + [_WEB_SEARCH_SCHEMA]


# ── Phase system prompts ───────────────────────────────────────────────────

_FC_PHASE1_SYSTEM = """\
You are the MediaLens fact-check orientation agent.
Given a claim to verify, determine what database coverage exists.
- You MUST call get_database_stats() first.
- You MUST call search_stories() with the claim's key topic words as the query.
- Do NOT produce prose. Stop after both tool calls.
"""

_FC_PHASE2_SYSTEM = """\
You are the MediaLens fact-check database evidence agent.
Candidate clusters from Phase 1 are in your context.
- You MUST call get_lean_breakdown() for the most relevant cluster(s) (1–2 clusters).
  Look explicitly for articles that SUPPORT and CONTRADICT the claim.
- You MAY call get_story_detail() to access body snippets for specific wording verification.
- Do NOT call get_database_stats or search_stories again.
- Do NOT produce prose. Stop after gathering evidence.
"""

_FC_PHASE3_SYSTEM = """\
You are the MediaLens web verification agent.
- You MUST call search_web() with the exact claim as the query.
- You MAY call search_web() a second time with a refined query if first results are off-topic.
- Do NOT call database tools. Stop after web search(es).
"""

_FC_PHASE4_SYSTEM = """\
You are the MediaLens fact-check synthesis agent. All evidence is in your context.
Produce a structured JSON verdict. Base your verdict ONLY on evidence in this conversation.
Do NOT invent sources. Do NOT conflate topic coverage with direct claim verification.

Confidence guidance (0–1 scale):
  1.0 = multiple concordant primary sources across political leans
  0.7 = strong evidence from one direction, weak or absent from the other
  0.4 = ambiguous, contradictory, or indirect evidence
  0.1 = almost no relevant coverage found
"""


# ── Helpers ────────────────────────────────────────────────────────────────

def _get_client() -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY is not set. Add it to your .env file.")
    return OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)


def _evidence_summary(trace: RunTrace) -> str:
    """Concatenate all tool result summaries from the trace."""
    parts = []
    for step in trace.steps:
        if step.tool_name.startswith("__"):
            continue
        parts.append(f"[{step.tool_name}]\n{step.result_summary}")
    return "\n\n---\n\n".join(parts) if parts else "No tool results available."


def _verdict_to_markdown(v: FactCheckVerdict) -> str:
    """Render a validated FactCheckVerdict to readable markdown."""
    lean = v.lean_emphasis
    confidence_pct = f"{v.confidence:.0%}"

    evidence_for = "\n".join(f"- {e}" for e in v.evidence_for) or "- None found."
    evidence_against = "\n".join(f"- {e}" for e in v.evidence_against) or "- None found."

    return (
        f"**{v.verdict.upper()}** — {v.one_line_explanation}\n\n"
        f"**Evidence for the claim**\n{evidence_for}\n\n"
        f"**Evidence against or complicating context**\n{evidence_against}\n\n"
        f"**What each side emphasises**\n"
        f"- Left: {lean.left or 'No coverage found.'}\n"
        f"- Center: {lean.center or 'No coverage found.'}\n"
        f"- Right: {lean.right or 'No coverage found.'}\n\n"
        f"**Confidence**: {confidence_pct}\n"
        f"{v.database_coverage_note}"
    )


# ── Phase-locked fact-check agent ──────────────────────────────────────────

def check(claim: str, context: str | None = None) -> dict:
    """
    Run the fact-check agent through four locked phases.

    Args:
        claim: The specific claim to fact-check.
        context: Optional story headline so the agent prioritises the right cluster.

    Returns:
        verdict          — lowercase string: "confirmed" / "disputed" / "misleading" / "unverifiable"
        confidence       — float 0–1 (new)
        structured_verdict — FactCheckVerdict as dict (new)
        response         — markdown string
        tools_called     — list of tool names
        run_trace        — RunTrace as dict
        error            — error message or None
    """
    try:
        client = _get_client()
    except EnvironmentError as exc:
        return {
            "verdict": None, "confidence": None, "structured_verdict": None,
            "response": "", "tools_called": [], "run_trace": None, "error": str(exc),
        }

    user_message = claim
    if context:
        user_message = f"Context: {context}\n\nClaim to fact-check: {claim}"

    trace = RunTrace()
    logger.info("Fact-check agent — phase-locked claim: {!r}", claim)

    # ── Phase 1: Orientation ───────────────────────────────────────────────
    # Required: get_database_stats, search_stories
    phase1_results = run_tool_phase(
        client=client,
        trace=trace,
        system=_FC_PHASE1_SYSTEM,
        user_content=user_message,
        allowed_tool_names={"get_database_stats", "search_stories"},
        required_tool_names={"get_database_stats", "search_stories"},
        required_fallback_args={"search_stories": {"query": claim}},
        tool_schemas=_AGENT_TOOL_SCHEMAS,
        tool_registry=_FACT_CHECK_TOOL_FUNCTIONS,
        max_iters=3,
    )

    candidates: list[dict] = phase1_results.get("search_stories") or []
    top_cluster_ids = [c["cluster_id"] for c in candidates[:2] if isinstance(c, dict)]

    # ── Phase 2: Database evidence ─────────────────────────────────────────
    # Required: get_lean_breakdown; optional: get_story_detail
    phase2_user = (
        f"Claim: {claim}\n\n"
        f"Top story clusters found:\n{json.dumps(candidates[:3], default=str, indent=2)}\n\n"
        "Now call get_lean_breakdown for the most relevant cluster(s). "
        "Look for articles that both support AND contradict the claim."
    )
    lean_fallback = (
        {"cluster_id": top_cluster_ids[0]} if top_cluster_ids else {}
    )
    phase2_schemas = [
        s for s in _AGENT_TOOL_SCHEMAS
        if s["function"]["name"] in ("get_lean_breakdown", "get_story_detail")
    ]

    run_tool_phase(
        client=client,
        trace=trace,
        system=_FC_PHASE2_SYSTEM,
        user_content=phase2_user,
        allowed_tool_names={"get_lean_breakdown", "get_story_detail"},
        required_tool_names={"get_lean_breakdown"} if top_cluster_ids else set(),
        required_fallback_args={"get_lean_breakdown": lean_fallback},
        tool_schemas=phase2_schemas,
        tool_registry=_FACT_CHECK_TOOL_FUNCTIONS,
        max_iters=4,
    )

    # ── Phase 3: Web evidence ──────────────────────────────────────────────
    # Required: search_web (always)
    run_tool_phase(
        client=client,
        trace=trace,
        system=_FC_PHASE3_SYSTEM,
        user_content=f"Verify this claim against live web sources: {claim}",
        allowed_tool_names={"search_web"},
        required_tool_names={"search_web"},
        required_fallback_args={"search_web": {"query": claim}},
        tool_schemas=[_WEB_SEARCH_SCHEMA],
        tool_registry=_FACT_CHECK_TOOL_FUNCTIONS,
        max_iters=3,
    )

    # ── Phase 4: Structured verdict ────────────────────────────────────────
    evidence = _evidence_summary(trace)
    phase4_messages = [
        {"role": "system", "content": _FC_PHASE4_SYSTEM},
        {"role": "user", "content": (
            f"Claim: {claim}\n\nAll gathered evidence:\n{evidence}"
        )},
    ]

    verdict_obj: FactCheckVerdict | None = None
    try:
        parsed = client.beta.chat.completions.parse(
            model=LLM_MODEL,
            messages=phase4_messages,
            response_format=FactCheckVerdict,
            temperature=LLM_TEMPERATURE,
        )
        verdict_obj = parsed.choices[0].message.parsed
        usage = parsed.usage
        if verdict_obj is None:
            raw = parsed.choices[0].message.content or ""
            verdict_obj = FactCheckVerdict.model_validate_json(raw)
        trace.add_step(
            "__verdict__", {}, verdict_obj.model_dump(),
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
        )
    except (ValidationError, ValueError) as exc:
        logger.error("Phase 4 verdict validation failed: {}", exc)
        return {
            "verdict": None, "confidence": None, "structured_verdict": None,
            "response": "",
            "tools_called": [s.tool_name for s in trace.steps if not s.tool_name.startswith("__")],
            "run_trace": trace.model_dump(mode="json"),
            "error": f"Verdict validation error: {exc}",
        }
    except Exception as exc:
        logger.error("Phase 4 API call failed: {}", exc)
        return {
            "verdict": None, "confidence": None, "structured_verdict": None,
            "response": "",
            "tools_called": [s.tool_name for s in trace.steps if not s.tool_name.startswith("__")],
            "run_trace": trace.model_dump(mode="json"),
            "error": str(exc),
        }

    logger.info(
        "Fact-check done. Verdict: {} (confidence={:.0%}). {} steps.",
        verdict_obj.verdict, verdict_obj.confidence, len(trace.steps),
    )

    return {
        "verdict": verdict_obj.verdict,
        "confidence": verdict_obj.confidence,
        "structured_verdict": verdict_obj.model_dump(),
        "response": _verdict_to_markdown(verdict_obj),
        "tools_called": [s.tool_name for s in trace.steps if not s.tool_name.startswith("__")],
        "run_trace": trace.model_dump(mode="json"),
        "error": None,
    }
