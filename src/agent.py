"""
MediaLens Research Agent.

Phase-locked orchestrator: three mandatory phases ensure the LLM always
orients on the data, gathers lean-breakdown evidence, and synthesises a
validated structured response — regardless of LLM tool-calling behaviour.
"""
import json
import os
from typing import Any

import numpy as np
from loguru import logger
from openai import OpenAI
from pydantic import ValidationError

from agent_utils import run_tool_phase
from clustering import _get_model, embed
from config import LLM_MODEL, LLM_TEMPERATURE, OPENROUTER_BASE_URL, RSS_SOURCES
from database import (
    get_articles_by_entity,
    get_cluster_analysis,
    get_cluster_with_articles_and_analysis,
    get_db,
    get_ready_cluster_centroids,
    get_ready_clusters,
    get_top_entities,
)
from schemas import AgentResponse, RunTrace


# ── Tool implementations ───────────────────────────────────────────────────

def search_stories(query: str, top_k: int = 5) -> list[dict]:
    """Semantic search against stored cluster centroids (article-body embeddings).
    Falls back to headline encoding for clusters that pre-date centroid storage."""
    with get_db() as conn:
        rows = get_ready_cluster_centroids(conn)

    if not rows:
        return []

    query_vec = embed(query)
    scored = []
    no_centroid_rows = []

    for row in rows:
        if row["centroid"]:
            centroid = np.array(json.loads(row["centroid"]))
            sim = float(np.dot(query_vec, centroid))
            lean_cov = json.loads(row["lean_coverage"])
            scored.append({
                "cluster_id": row["id"],
                "headline": row["representative_headline"],
                "lean_coverage": lean_cov,
                "article_count": sum(lean_cov.values()),
                "similarity": round(sim, 3),
            })
        else:
            no_centroid_rows.append(row)

    # Encode headlines in one batch for legacy clusters (no stored centroid)
    if no_centroid_rows:
        model = _get_model()
        headlines = [r["representative_headline"] for r in no_centroid_rows]
        vecs = model.encode(headlines, normalize_embeddings=True)
        for i, row in enumerate(no_centroid_rows):
            lean_cov = json.loads(row["lean_coverage"])
            sim = float(np.dot(query_vec, vecs[i]))
            scored.append({
                "cluster_id": row["id"],
                "headline": row["representative_headline"],
                "lean_coverage": lean_cov,
                "article_count": sum(lean_cov.values()),
                "similarity": round(sim, 3),
            })

    scored.sort(key=lambda x: x["similarity"], reverse=True)
    relevant = [s for s in scored if s["similarity"] >= 0.3]
    return (relevant or scored)[:top_k]


def get_story_detail(cluster_id: int) -> dict:
    with get_db() as conn:
        data = get_cluster_with_articles_and_analysis(cluster_id, conn)
        cluster_analysis = get_cluster_analysis(cluster_id, conn)

    if not data:
        return {"error": f"Cluster {cluster_id} not found"}

    articles = []
    for art in data.get("articles", []):
        articles.append({
            "id": art["id"],
            "title": art["title"],
            "source_name": art["source_name"],
            "lean": art["source_lean"],
            "url": art["url"],
            "published_at": art.get("published_at", ""),
        })

    return {
        "cluster_id": cluster_id,
        "headline": data["representative_headline"],
        "lean_coverage": data["lean_coverage"],
        "cluster_analysis": cluster_analysis,
        "articles": articles,
    }


def list_recent_stories(limit: int = 10, lean: str | None = None) -> list[dict]:
    with get_db() as conn:
        clusters = get_ready_clusters(conn)

    result = []
    for c in clusters:
        lean_cov = json.loads(c["lean_coverage"])
        if lean and lean_cov.get(lean, 0) == 0:
            continue
        result.append({
            "cluster_id": c["id"],
            "headline": c["representative_headline"],
            "lean_coverage": lean_cov,
            "article_count": sum(lean_cov.values()),
            "created_at": c["created_at"],
        })

    return result[:limit]


def get_lean_breakdown(cluster_id: int) -> dict:
    with get_db() as conn:
        data = get_cluster_with_articles_and_analysis(cluster_id, conn)

    if not data:
        return {"error": f"Cluster {cluster_id} not found"}

    breakdown: dict[str, list] = {"left": [], "center": [], "right": []}
    for art in data.get("articles", []):
        lean = art["source_lean"]
        entry = {
            "title": art["title"],
            "source": art["source_name"],
            "url": art["url"],
        }
        if art.get("analysis"):
            entry["framing"] = art["analysis"]["framing_summary"]
            entry["bias_score"] = art["analysis"]["bias_score"]
        breakdown.setdefault(lean, []).append(entry)

    return {
        "cluster_id": cluster_id,
        "headline": data["representative_headline"],
        "breakdown": breakdown,
    }


def get_trending_topics(limit: int = 5) -> list[dict]:
    with get_db() as conn:
        clusters = get_ready_clusters(conn)

    result = []
    for c in clusters:
        lean_cov = json.loads(c["lean_coverage"])
        result.append({
            "cluster_id": c["id"],
            "headline": c["representative_headline"],
            "lean_coverage": lean_cov,
            "article_count": sum(lean_cov.values()),
            "leans_covered": sum(1 for v in lean_cov.values() if v > 0),
        })

    result.sort(key=lambda x: (x["leans_covered"], x["article_count"]), reverse=True)
    return result[:limit]


def get_database_stats() -> dict:
    with get_db() as conn:
        total_articles = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        total_clusters = conn.execute("SELECT COUNT(*) FROM clusters").fetchone()[0]
        ready_clusters = conn.execute(
            "SELECT COUNT(*) FROM clusters WHERE ready_for_analysis = 1"
        ).fetchone()[0]
        analyzed_articles = conn.execute(
            "SELECT COUNT(*) FROM articles WHERE analyzed = 1"
        ).fetchone()[0]
        last_fetched = conn.execute(
            "SELECT MAX(fetched_at) FROM articles"
        ).fetchone()[0]

    return {
        "total_articles": total_articles,
        "total_clusters": total_clusters,
        "ready_clusters": ready_clusters,
        "analyzed_articles": analyzed_articles,
        "last_fetched_at": last_fetched or "never",
    }


def list_sources() -> list[dict]:
    return [{"name": s["name"], "lean": s["lean"], "url": s["url"]} for s in RSS_SOURCES]


def search_by_entity(name: str, limit: int = 10) -> list[dict]:
    with get_db() as conn:
        rows = get_articles_by_entity(name.lower(), conn, limit=limit)
    return [
        {
            "article_id": r["id"],
            "title": r["title"],
            "source_name": r["source_name"],
            "lean": r["source_lean"],
            "url": r["url"],
            "published_at": r["published_at"],
            "entity_text": r["entity_text"],
            "entity_label": r["entity_label"],
        }
        for r in rows
    ]


def get_top_mentioned_entities(limit: int = 15, label: str | None = None) -> list[dict]:
    with get_db() as conn:
        rows = get_top_entities(conn, limit=limit, label=label)
    return [
        {
            "entity": r["text"],
            "normalized": r["normalized"],
            "type": r["label"],
            "article_count": r["articles"],
        }
        for r in rows
    ]


# ── Tool registry & schemas ────────────────────────────────────────────────

_TOOL_FUNCTIONS: dict[str, Any] = {
    "search_stories": search_stories,
    "get_story_detail": get_story_detail,
    "list_recent_stories": list_recent_stories,
    "get_lean_breakdown": get_lean_breakdown,
    "get_trending_topics": get_trending_topics,
    "get_database_stats": get_database_stats,
    "list_sources": list_sources,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_stories",
            "description": (
                "Search for news story clusters semantically related to a query. "
                "Returns clusters ranked by relevance with lean coverage counts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search topic or keywords"},
                    "top_k": {"type": "integer", "description": "Max results to return (default 5)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_story_detail",
            "description": (
                "Get the full article list and bias analyses for a specific story cluster. "
                "Use after search_stories to drill into a promising result."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cluster_id": {"type": "integer", "description": "The cluster ID from search_stories"},
                },
                "required": ["cluster_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_recent_stories",
            "description": (
                "List recent ready clusters, optionally filtered by political lean. "
                "Use for open-ended queries like 'what's in the news?' or 'what is right media covering?'"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Max results (default 10)"},
                    "lean": {
                        "type": "string",
                        "enum": ["left", "center", "right"],
                        "description": "Filter to clusters covered by this lean",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_lean_breakdown",
            "description": (
                "Get articles for a cluster grouped by political lean (left/center/right) "
                "with framing notes. Use to compare how different outlets cover the same story."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cluster_id": {"type": "integer", "description": "The cluster ID"},
                },
                "required": ["cluster_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_trending_topics",
            "description": (
                "Return the most-covered story clusters sorted by leans represented "
                "and total article count. Shows what is most broadly debated right now."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Max results (default 5)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_database_stats",
            "description": (
                "Return article counts, cluster counts, and when data was last fetched. "
                "Call this to check data freshness before answering substantive questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_sources",
            "description": (
                "List all configured RSS news sources with their political lean. "
                "Use to answer questions about which outlets are covered."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]


# ── Phase system prompts ───────────────────────────────────────────────────

_PHASE1_SYSTEM = """\
You are the MediaLens data orientation agent.
Your ONLY task: determine what data is available to answer the user's question.
- You MUST call get_database_stats() first.
- You MUST call search_stories(query=<user_query>) immediately after.
- Do NOT produce prose. After both tool calls, stop (finish_reason=stop).
- Do NOT call any other tools in this phase.
"""

_PHASE3_SYSTEM = """\
You are the MediaLens synthesis agent. Pre-computed differential analyses for matched story clusters are in your context.
Produce a cross-cluster briefing.

Return ONLY valid JSON with exactly these fields:

topic_overview
  String. 2-3 sentence neutral summary of what is happening across the matched stories.

shared_ground
  JSON array of 1-4 strings. Facts all leans consistently report across the stories.
  Each string must include inline outlet citations, e.g. "(Reuters, Fox News, Guardian)".

left_emphasis
  JSON array of 0-3 strings. Angles, figures, or frames that left outlets add across
  the stories that right outlets omit or downplay. Be specific. Cite outlets.
  Use [] if no significant pattern found.

right_emphasis
  JSON array of 0-3 strings. Same structure for right outlets vs left.

center_angle
  String. What center sources (AP, BBC, Reuters, Al Jazeera) uniquely add or frame
  differently from both left and right, across the matched stories.
  Empty string "" if center simply follows both sides.

Rules:
- Base your response ONLY on evidence in this conversation.
- Do NOT invent outlet names, headlines, or claims not present in the evidence.
- Be specific. Name angles and cite outlets.
- If only one lean is represented, set the missing side's field to [].
"""


# ── Helpers ────────────────────────────────────────────────────────────────

def _get_client() -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY is not set. Add it to your .env file.")
    return OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)


def _fetch_cluster_contexts(cluster_ids: list[int]) -> list[dict]:
    """
    For each cluster: fetch pre-computed cluster_analysis if available.
    Falls back to get_lean_breakdown (raw article titles) for unanalyzed clusters.
    Returns a list of context dicts ready to be formatted into the synthesis prompt.
    """
    contexts = []
    for cid in cluster_ids:
        with get_db() as conn:
            ca = get_cluster_analysis(cid, conn)
            data = get_cluster_with_articles_and_analysis(cid, conn)
        if not data:
            continue
        lean_cov = data.get("lean_coverage", {})
        entry: dict = {
            "cluster_id": cid,
            "headline": data.get("representative_headline", ""),
            "lean_coverage": lean_cov,
            "has_analysis": ca is not None,
        }
        if ca:
            entry["analysis"] = ca
        else:
            entry["analysis"] = get_lean_breakdown(cid)
        contexts.append(entry)
    return contexts


def _format_cluster_contexts(contexts: list[dict]) -> str:
    """Render cluster contexts as structured text for the Phase 3 synthesis prompt."""
    parts = []
    for ctx in contexts:
        lines = [f"### Story: {ctx['headline']} (cluster {ctx['cluster_id']})"]
        if ctx.get("has_analysis"):
            ca = ctx["analysis"]
            lines.append(f"Summary: {ca.get('summary', '')}")
            if ca.get("shared_ground"):
                lines.append("Shared ground:")
                for s in ca["shared_ground"]:
                    lines.append(f"  - {s}")
            if ca.get("left_not_right"):
                lines.append("Left emphasizes (right omits/downplays):")
                for item in ca["left_not_right"]:
                    lines.append(f"  - [{item.get('coverage', '?')}] {item.get('claim', '')}")
            if ca.get("right_not_left"):
                lines.append("Right emphasizes (left omits/downplays):")
                for item in ca["right_not_left"]:
                    lines.append(f"  - [{item.get('coverage', '?')}] {item.get('claim', '')}")
            if ca.get("center_angle"):
                lines.append(f"Center angle: {ca['center_angle']}")
        else:
            bd = ctx.get("analysis", {})
            if isinstance(bd, dict) and "breakdown" in bd:
                for lean, articles in bd["breakdown"].items():
                    if articles:
                        titles = "; ".join(a["title"] for a in articles[:3])
                        lines.append(f"{lean.capitalize()} articles: {titles}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts) if parts else "No cluster data available."


def _to_markdown(r: AgentResponse) -> str:
    lines = [
        "## Overview",
        r.topic_overview,
        "",
        "## What All Sides Report",
        *[f"- {point}" for point in r.shared_ground],
        "",
    ]
    if r.left_emphasis:
        lines += ["## Left Covers — Right Doesn't", *[f"- {point}" for point in r.left_emphasis], ""]
    if r.right_emphasis:
        lines += ["## Right Covers — Left Doesn't", *[f"- {point}" for point in r.right_emphasis], ""]
    if r.center_angle:
        lines += ["## Center Angle", r.center_angle, ""]
    return "\n".join(lines)


# ── Phase-locked agent ─────────────────────────────────────────────────────

def ask(query: str) -> dict:
    """
    Run the research agent through three locked phases.

    Returns:
        response     — markdown string (always present on success)
        structured   — AgentResponse as dict
        tools_called — list of tool names called
        run_trace    — RunTrace as dict (steps, token counts)
        error        — error message or None
    """
    try:
        client = _get_client()
    except EnvironmentError as exc:
        return {
            "response": "", "structured": None,
            "tools_called": [], "run_trace": None, "error": str(exc),
        }

    trace = RunTrace()
    logger.info("Research agent — phase-locked query: {!r}", query)

    # ── Phase 1: Orientation ───────────────────────────────────────────────
    # Required: get_database_stats, search_stories
    phase1_results = run_tool_phase(
        client=client,
        trace=trace,
        system=_PHASE1_SYSTEM,
        user_content=query,
        allowed_tool_names={"get_database_stats", "search_stories"},
        required_tool_names={"get_database_stats", "search_stories"},
        required_fallback_args={"search_stories": {"query": query}},
        tool_schemas=TOOL_SCHEMAS,
        tool_registry=_TOOL_FUNCTIONS,
        max_iters=3,
    )

    candidates: list[dict] = phase1_results.get("search_stories") or []
    top_cluster_ids = [c["cluster_id"] for c in candidates[:4] if isinstance(c, dict)]

    # ── Phase 2: Evidence (Python, no LLM) ────────────────────────────────
    # Fetch pre-computed cluster_analysis for each matched cluster.
    # Falls back to raw lean breakdown for clusters not yet analyzed.
    logger.debug("Fetching cluster contexts for {} cluster(s).", len(top_cluster_ids))
    cluster_contexts = _fetch_cluster_contexts(top_cluster_ids)
    for ctx in cluster_contexts:
        trace.add_step(
            "get_cluster_analysis",
            {"cluster_id": ctx["cluster_id"]},
            ctx.get("analysis") or {},
        )

    # ── Phase 3: Synthesis ─────────────────────────────────────────────────
    # Single structured call — no tools. Returns validated AgentResponse.
    cluster_evidence = _format_cluster_contexts(cluster_contexts)
    phase3_messages = [
        {"role": "system", "content": _PHASE3_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Question: {query}\n\n"
                f"Matched story clusters:\n{cluster_evidence}"
            ),
        },
    ]

    agent_response: AgentResponse | None = None
    try:
        parsed = client.beta.chat.completions.parse(
            model=LLM_MODEL,
            messages=phase3_messages,
            response_format=AgentResponse,
            temperature=LLM_TEMPERATURE,
        )
        agent_response = parsed.choices[0].message.parsed
        usage = parsed.usage
        if agent_response is None:
            # OpenRouter didn't honour the JSON schema mode; fall back to manual parse
            raw = parsed.choices[0].message.content or ""
            agent_response = AgentResponse.model_validate_json(raw)
        trace.add_step(
            "__synthesis__", {}, agent_response.model_dump(),
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
        )
    except (ValidationError, ValueError) as exc:
        logger.error("Phase 3 structured synthesis failed: {}", exc)
        return {
            "response": "",
            "structured": None,
            "tools_called": [s.tool_name for s in trace.steps],
            "run_trace": trace.model_dump(mode="json"),
            "error": f"Synthesis validation error: {exc}",
        }
    except Exception as exc:
        logger.error("Phase 3 API call failed: {}", exc)
        return {
            "response": "",
            "structured": None,
            "tools_called": [s.tool_name for s in trace.steps],
            "run_trace": trace.model_dump(mode="json"),
            "error": str(exc),
        }

    logger.info(
        "Research agent done. {} steps, {} prompt + {} completion tokens.",
        len(trace.steps),
        trace.total_prompt_tokens,
        trace.total_completion_tokens,
    )

    matched_clusters = [
        {
            "cluster_id": ctx["cluster_id"],
            "headline": ctx["headline"],
            "lean_coverage": ctx["lean_coverage"],
            "has_analysis": ctx["has_analysis"],
            "similarity": next(
                (c.get("similarity") for c in candidates if c.get("cluster_id") == ctx["cluster_id"]),
                None,
            ),
        }
        for ctx in cluster_contexts
    ]

    return {
        "response": _to_markdown(agent_response),
        "structured": agent_response.model_dump(),
        "matched_clusters": matched_clusters,
        "tools_called": [s.tool_name for s in trace.steps if not s.tool_name.startswith("__")],
        "run_trace": trace.model_dump(mode="json"),
        "error": None,
    }
