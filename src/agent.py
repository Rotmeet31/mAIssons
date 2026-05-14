"""
MediaLens Research Agent.

Tool-calling loop via OpenRouter: the LLM decides which tools to invoke,
in what order, to answer cross-spectrum news questions.
"""
import json
import os
from typing import Any

import numpy as np
from loguru import logger
from openai import OpenAI

from clustering import _get_model, embed
from config import LLM_MODEL, LLM_TEMPERATURE, OPENROUTER_BASE_URL, RSS_SOURCES
from database import (
    get_articles_by_entity,
    get_cluster_analysis,
    get_cluster_with_articles_and_analysis,
    get_db,
    get_ready_clusters,
    get_top_entities,
)


# ── Tool implementations ───────────────────────────────────────────────────

def search_stories(query: str, top_k: int = 5) -> list[dict]:
    with get_db() as conn:
        clusters = get_ready_clusters(conn)

    if not clusters:
        return []

    query_vec = embed(query)
    model = _get_model()
    headlines = [c["representative_headline"] for c in clusters]
    headline_vecs = model.encode(headlines, normalize_embeddings=True)

    scored = []
    for i, cluster in enumerate(clusters):
        lean_cov = json.loads(cluster["lean_coverage"])
        sim = float(np.dot(query_vec, headline_vecs[i]))
        scored.append({
            "cluster_id": cluster["id"],
            "headline": cluster["representative_headline"],
            "lean_coverage": lean_cov,
            "article_count": sum(lean_cov.values()),
            "similarity": round(sim, 3),
        })

    scored.sort(key=lambda x: x["similarity"], reverse=True)
    relevant = [s for s in scored if s["similarity"] >= 0.3]
    if not relevant:
        relevant = scored
    return relevant[:top_k]


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


# ── Tool registry ──────────────────────────────────────────────────────────

_TOOL_FUNCTIONS: dict[str, Any] = {
    "search_stories": search_stories,
    "get_story_detail": get_story_detail,
    "list_recent_stories": list_recent_stories,
    "get_lean_breakdown": get_lean_breakdown,
    "get_trending_topics": get_trending_topics,
    "get_database_stats": get_database_stats,
    "list_sources": list_sources,
}

# OpenAI-format tool schemas (used by OpenRouter)
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

_SYSTEM_PROMPT = """\
You are a cross-spectrum news research assistant for MediaLens.

Your job: help users understand how news stories are covered across the political spectrum \
(left, center, right). You have access to a database of articles ingested from outlets \
across that spectrum, clustered into stories, and analyzed for political framing.

Guidelines:
- Call get_database_stats first to check data freshness before answering substantive questions.
- Use search_stories for specific topics, list_recent_stories for open-ended browsing.
- Use get_lean_breakdown to compare how different outlets frame the same story.
- Cite specific outlets and article headlines in your response.
- Identify what sources agree on, where they diverge, and notable framing differences.
- If the database has no relevant stories, say so — do not invent coverage.
- Format your response in markdown: use headers, bullet points, bold source names.
"""


# ── Agent loop ─────────────────────────────────────────────────────────────

def _execute_tool(name: str, args: dict) -> Any:
    fn = _TOOL_FUNCTIONS.get(name)
    if fn is None:
        return {"error": f"Unknown tool: {name}"}
    try:
        return fn(**args)
    except Exception as exc:
        logger.error("Tool {} failed with args {}: {}", name, args, exc)
        return {"error": str(exc)}


def _get_client() -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY is not set. Add it to your .env file.")
    return OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)


def ask(query: str) -> dict:
    """
    Run the research agent loop.

    Returns: { response: str, tools_called: list[str], error: str | None }
    """
    try:
        client = _get_client()
    except EnvironmentError as exc:
        return {"response": "", "tools_called": [], "error": str(exc)}

    messages: list[dict] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    tools_called: list[str] = []

    logger.info("Research agent: {!r}", query)

    for iteration in range(10):  # safety cap on loop depth
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=LLM_TEMPERATURE,
        )

        choice = response.choices[0]
        logger.debug("Iteration {}: finish_reason={}", iteration + 1, choice.finish_reason)

        if choice.finish_reason == "stop":
            text = choice.message.content or ""
            logger.info(
                "Agent finished after {} iteration(s). Tools: {}",
                iteration + 1,
                tools_called,
            )
            return {"response": text, "tools_called": tools_called, "error": None}

        if choice.finish_reason != "tool_calls":
            break

        # Append the assistant turn (with its tool_calls) and execute each one
        messages.append(choice.message)

        for tool_call in choice.message.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            tools_called.append(name)
            logger.info("  → {}({})", name, args)
            result = _execute_tool(name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, default=str),
            })

    return {
        "response": "",
        "tools_called": tools_called,
        "error": "Agent loop ended without producing a final response.",
    }
