"""
Per-cluster LLM analysis: one call per cluster, all articles fed together.
Returns consensus points, framing disagreements, and coverage gaps.
"""
import json
import os
import re

from openai import OpenAI
from loguru import logger
from pydantic import ValidationError

from config import ANALYSIS_BODY_CHARS, LLM_MODEL, LLM_TEMPERATURE, MAX_RETRIES, OPENROUTER_BASE_URL
from database import (
    get_articles_by_cluster,
    get_cluster_analysis,
    get_db,
    get_unanalyzed_ready_clusters,
    insert_cluster_analysis,
)
from schemas import ClusterAnalysisResult


# ── Client ─────────────────────────────────────────────────────────────────

def _get_client() -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY is not set. Add it to your .env file.")
    return OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)


# ── Prompt ─────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a cross-spectrum news analyst. You will be given articles about the same story
from outlets tagged [LEFT], [CENTER], or [RIGHT].

Your task: produce a differential analysis — not just what each side says, but what each
side says that the other side does NOT say, or barely says.

Return ONLY valid JSON with exactly these fields:

summary
  String. 2-3 sentences describing what happened. Neutral, fact-based.
  Anchored in what ALL sources report. No editorial framing.

shared_ground
  JSON array of 2-4 strings. Facts that most leans report consistently.
  Each string must include inline outlet citations, e.g. "(Reuters, Fox News, Guardian)".

left_not_right
  JSON array of 0-3 objects. Each object: {"claim": "...", "coverage": "omitted"|"downplayed"}
  "claim" = something LEFT articles emphasize that RIGHT articles either:
    - "omitted":    have zero mention of this topic, angle, or figure.
    - "downplayed": mention it but not prominently — buried after paragraph 3,
                    one passing sentence, absent from headline and lede.
  Be specific: name the angle and cite the source lean.
  Example: {"claim": "Guardian and NPR lead with civilian casualty figures ($X);
             Fox and Breitbart do not mention this number.", "coverage": "omitted"}
  Use [] if no significant asymmetry is found.

right_not_left
  JSON array of 0-3 objects. Same structure.
  "claim" = something RIGHT articles emphasize that LEFT articles omit or downplay.

center_angle
  String. What CENTER sources (AP, BBC, Reuters, Al Jazeera) uniquely add or frame
  differently from both left and right — context, data, international reaction, etc.
  Write "" (empty string) if center simply repeats what left and right both cover.

Rules:
- "omitted" requires that you checked all articles on the other side and found ZERO mention.
- Do not invent coverage. Only use what the article snippets actually contain.
- Be specific in claims. Name angles, figures, and outlets — not vague patterns.
- If only left or only right articles are present, set the missing-side field to [].
"""


def _build_prompt(headline: str, articles: list) -> str:
    lines = [f'Story cluster: "{headline}"\n\nARTICLES:']
    for art in articles:
        lean_tag = art["source_lean"].upper()
        snippet = (art["body"] or "")[:ANALYSIS_BODY_CHARS].strip()
        lines.append(f'\n[{lean_tag}] {art["source_name"]}: "{art["title"]}"\n{snippet}')
    return "\n".join(lines)


# ── Parsing ────────────────────────────────────────────────────────────────

_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _parse_and_validate(raw: str) -> ClusterAnalysisResult:
    """Strip thinking tags and fences, then validate against ClusterAnalysisResult."""
    cleaned = _THINK_RE.sub("", raw).strip()
    fence_match = _FENCE_RE.search(cleaned)
    json_str = fence_match.group(1) if fence_match else cleaned
    data = json.loads(json_str)

    # Remap flat lean keys if the model returns them at the top level instead of nested
    if "lean_summaries" not in data and any(k in data for k in ("left", "center", "right")):
        data["lean_summaries"] = {k: data.pop(k, "") for k in ("left", "center", "right")}

    return ClusterAnalysisResult.model_validate(data)


# ── Per-cluster analysis ───────────────────────────────────────────────────

def analyze_cluster(
    client: OpenAI,
    cluster_id: int,
    headline: str,
    articles: list,
) -> dict | None:
    base_messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _build_prompt(headline, articles)},
    ]

    json_schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "ClusterAnalysisResult",
            "schema": ClusterAnalysisResult.model_json_schema(),
            "strict": False,
        },
    }

    logger.debug("Sending cluster #{} '{}' to LLM", cluster_id, headline[:60])

    last_exc: str | None = None
    for attempt in range(MAX_RETRIES + 1):
        messages = list(base_messages)

        # On retry, inject the validation error as a correction prompt
        if attempt > 0 and last_exc:
            messages.append({
                "role": "user",
                "content": (
                    f"Your previous response failed validation with this error:\n{last_exc}\n\n"
                    "Please fix your JSON and return a valid response matching the schema exactly."
                ),
            })

        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                temperature=LLM_TEMPERATURE,
                response_format=json_schema,
            )
            raw = response.choices[0].message.content or ""
            logger.debug("Cluster {} raw (attempt {}):\n{}", cluster_id, attempt + 1, raw)
            result = _parse_and_validate(raw)
            return result.model_dump()
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            last_exc = str(exc)
            logger.warning(
                "Parse/validation error cluster {} attempt {}: {}", cluster_id, attempt + 1, exc
            )
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

            logger.info(
                "Analyzing cluster #{} '{}' ({} articles)",
                cluster_id, headline[:60], len(articles),
            )

            try:
                result = analyze_cluster(client, cluster_id, headline, articles)
            except Exception as exc:
                last_error = str(exc)
                logger.error("Skipping cluster #{}: {}", cluster_id, exc)
                continue

            if result is None:
                continue

            # result is ClusterAnalysisResult.model_dump() — all keys guaranteed
            insert_cluster_analysis(
                cluster_id=cluster_id,
                summary=result["summary"],
                shared_ground=result["shared_ground"],
                left_not_right=result["left_not_right"],
                right_not_left=result["right_not_left"],
                center_angle=result["center_angle"],
                conn=conn,
            )
            logger.info(
                "Cluster #{} → {} shared, {} left≠right, {} right≠left",
                cluster_id,
                len(result["shared_ground"]),
                len(result["left_not_right"]),
                len(result["right_not_left"]),
            )
            analyzed += 1

    logger.info("Analysis complete. {}/{} clusters processed.", analyzed, len(clusters))
    return analyzed, last_error


if __name__ == "__main__":
    from database import init_db
    init_db()
    run_analysis()
