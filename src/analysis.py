"""
Per-cluster LLM analysis: one call per cluster, all articles fed together.
Returns consensus points, framing disagreements, and coverage gaps.
"""
import json
import os
import re

from openai import OpenAI
from loguru import logger

from config import LLM_MODEL, LLM_TEMPERATURE, MAX_RETRIES, OPENROUTER_BASE_URL
from database import (
    get_articles_by_cluster,
    get_cluster_analysis,
    get_db,
    get_unanalyzed_ready_clusters,
    insert_cluster_analysis,
)


# ── Client ─────────────────────────────────────────────────────────────────

def _get_client() -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY is not set. Add it to your .env file.")
    return OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)


# ── Prompt ─────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a cross-spectrum news analyst. You will be given articles about the same story "
    "from outlets across the political spectrum (LEFT / CENTER / RIGHT). "
    "Identify: what they agree on, how they frame it differently, and what each side omits.\n"
    "Return ONLY valid JSON with exactly these fields:\n"
    "  consensus      — JSON array of 2-4 strings; each string is one point most sources "
    "agree on, with at least one inline outlet citation like (Reuters, BBC)\n"
    "  disagreements  — JSON array of 2-4 strings; each string describes a framing or "
    "emphasis difference between leans, with inline outlet citations\n"
    "  gaps           — single string: notable omissions, misleading claims, or facts "
    "ignored by one side; or exactly \"None identified.\" if none found"
)


def _build_prompt(headline: str, articles: list) -> str:
    lines = [f'Story cluster: "{headline}"\n\nARTICLES:']
    for art in articles:
        lean_tag = art["source_lean"].upper()
        snippet = (art["body"] or "")[:300].strip()
        lines.append(f'\n[{lean_tag}] {art["source_name"]}: "{art["title"]}"\n{snippet}')
    return "\n".join(lines)


# ── Parsing ────────────────────────────────────────────────────────────────

_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _parse(raw: str) -> dict:
    cleaned = _THINK_RE.sub("", raw).strip()
    match = _FENCE_RE.search(cleaned)
    data = json.loads(match.group(1) if match else cleaned)

    consensus = data.get("consensus", [])
    if isinstance(consensus, str):
        consensus = [consensus]

    disagreements = data.get("disagreements", [])
    if isinstance(disagreements, str):
        disagreements = [disagreements]

    gaps = data.get("gaps", "")
    if not isinstance(gaps, str):
        gaps = str(gaps)

    return {"consensus": consensus, "disagreements": disagreements, "gaps": gaps}


# ── Per-cluster analysis ───────────────────────────────────────────────────

def analyze_cluster(
    client: OpenAI,
    cluster_id: int,
    headline: str,
    articles: list,
) -> dict | None:
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _build_prompt(headline, articles)},
    ]
    logger.debug("Sending cluster #{} '{}' to LLM", cluster_id, headline[:60])

    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                temperature=LLM_TEMPERATURE,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or ""
            logger.debug("Cluster {} raw (attempt {}):\n{}", cluster_id, attempt + 1, raw)
            return _parse(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            last_exc = exc
            logger.warning("Parse error cluster {} attempt {}: {}", cluster_id, attempt + 1, exc)
        except Exception as exc:
            logger.error("API error cluster {}: {}", cluster_id, exc)
            raise

    logger.error(
        "Giving up on cluster #{} after {} attempt(s). Last: {}",
        cluster_id, MAX_RETRIES + 1, last_exc,
    )
    return None


# ── Batch runner ───────────────────────────────────────────────────────────

def run_analysis() -> tuple[int, str | None]:
    """
    Analyze every ready cluster that has no cluster_analysis row yet.
    Returns (count_analyzed, error_message_or_None).
    """
    try:
        client = _get_client()
    except EnvironmentError as exc:
        return 0, str(exc)

    analyzed = 0
    last_error: str | None = None

    with get_db() as conn:
        clusters = get_unanalyzed_ready_clusters(conn)
        if not clusters:
            logger.info("No clusters pending analysis.")
            return 0, None

        logger.info("{} cluster(s) pending analysis.", len(clusters))

        for cluster in clusters:
            cluster_id = cluster["id"]
            headline = cluster["representative_headline"]
            articles = list(get_articles_by_cluster(cluster_id, conn))

            if len(articles) < 2:
                logger.debug("Cluster #{} has < 2 articles, skipping.", cluster_id)
                continue

            logger.info("Analyzing cluster #{} '{}' ({} articles)", cluster_id, headline[:60], len(articles))

            try:
                result = analyze_cluster(client, cluster_id, headline, articles)
            except Exception as exc:
                last_error = str(exc)
                logger.error("Skipping cluster #{}: {}", cluster_id, exc)
                continue

            if result is None:
                continue

            insert_cluster_analysis(
                cluster_id=cluster_id,
                consensus=result["consensus"],
                disagreements=result["disagreements"],
                gaps=result["gaps"],
                conn=conn,
            )
            logger.info(
                "Cluster #{} → {} consensus, {} disagreements",
                cluster_id, len(result["consensus"]), len(result["disagreements"]),
            )
            analyzed += 1

    logger.info("Analysis complete. {}/{} clusters processed.", analyzed, len(clusters))
    return analyzed, last_error


if __name__ == "__main__":
    from database import init_db
    init_db()
    run_analysis()
