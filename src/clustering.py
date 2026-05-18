"""
Cluster articles about the same story using sentence-transformer embeddings
and cosine similarity against existing cluster centroids.

Two-tier matching:
  1. Strong cosine (≥ CLUSTER_SIMILARITY_THRESHOLD)   → join immediately
  2. Moderate cosine (≥ CLUSTER_ENTITY_RESCUE_COSINE)
     + entity Jaccard (≥ CLUSTER_ENTITY_RESCUE_JACCARD) → rescue into cluster

Entity Jaccard uses only high-salience labels (PERSON, ORG, GPE) to avoid
noise from generic location/event mentions.
"""
import json
from typing import Any

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from config import (
    BODY_SNIPPET_CHARS,
    CLUSTER_ENTITY_LABELS,
    CLUSTER_ENTITY_RESCUE_COSINE,
    CLUSTER_ENTITY_RESCUE_JACCARD,
    CLUSTER_SIMILARITY_THRESHOLD,
    EMBEDDING_MODEL,
)
from database import (
    assign_article_to_cluster,
    get_all_clusters,
    get_articles_by_cluster,
    get_db,
    get_entities_for_article,
    get_entities_for_cluster,
    get_unclustered_articles,
    insert_cluster,
    update_cluster_centroid,
    update_cluster_lean_coverage,
    update_cluster_ready,
)

# Lazy-loaded model (singleton)
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info("Loading embedding model: {}", EMBEDDING_MODEL)
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


# ── Embedding helpers ──────────────────────────────────────────────────────

def _make_text_for_embedding(title: str, body: str) -> str:
    snippet = body[:BODY_SNIPPET_CHARS].strip()
    return f"{title}. {snippet}"


def embed(text: str) -> np.ndarray:
    model = _get_model()
    vec = model.encode(text, normalize_embeddings=True)
    return vec  # type: ignore[return-value]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity for pre-normalised vectors (dot product)."""
    return float(np.dot(a, b))


# ── Entity helpers ─────────────────────────────────────────────────────────

def _entity_set_from_rows(rows: list) -> set[str]:
    """Return normalized entity strings filtered to high-salience labels."""
    return {r["normalized"] for r in rows if r["label"] in CLUSTER_ENTITY_LABELS}


def _entity_jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _cluster_entity_set(cluster_id: int, conn: Any) -> set[str]:
    rows = get_entities_for_cluster(cluster_id, conn)
    return _entity_set_from_rows(rows)


def _article_entity_set(article_id: int, conn: Any) -> set[str]:
    rows = get_entities_for_article(article_id, conn)
    return _entity_set_from_rows(rows)


# ── Centroid management ────────────────────────────────────────────────────

def _compute_cluster_centroid(
    cluster_id: int, conn: Any
) -> np.ndarray | None:
    """Average embedding of all articles in the cluster."""
    articles = get_articles_by_cluster(cluster_id, conn)
    if not articles:
        return None
    model = _get_model()
    texts = [_make_text_for_embedding(a["title"], a["body"]) for a in articles]
    embeddings = model.encode(texts, normalize_embeddings=True)
    centroid = embeddings.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm
    return centroid  # type: ignore[return-value]


# ── Lean coverage ──────────────────────────────────────────────────────────

def _refresh_lean_coverage(cluster_id: int, conn: Any) -> dict:
    articles = get_articles_by_cluster(cluster_id, conn)
    coverage: dict[str, int] = {"left": 0, "center": 0, "right": 0}
    for art in articles:
        lean = art["source_lean"]
        if lean in coverage:
            coverage[lean] += 1
    update_cluster_lean_coverage(cluster_id, coverage, conn)
    distinct_leans = sum(1 for v in coverage.values() if v > 0)
    ready = distinct_leans >= 2
    update_cluster_ready(cluster_id, ready, conn)
    return coverage


# ── Main clustering logic ──────────────────────────────────────────────────

def cluster_new_articles() -> int:
    """
    Embed all unclustered articles and assign them to an existing cluster
    or create a new one.

    Matching is two-tier:
      - Strong cosine (≥ threshold)           → join
      - Moderate cosine + entity Jaccard agree → join (entity rescue)
      - Otherwise                              → new cluster

    Returns the number of articles processed.
    """
    with get_db() as conn:
        unclustered = get_unclustered_articles(conn)
        if not unclustered:
            logger.info("No unclustered articles found.")
            return 0

        logger.info("Clustering {} unclustered article(s)...", len(unclustered))

        # Build caches: centroid vectors and entity sets per cluster
        existing_clusters = get_all_clusters(conn)
        centroid_cache: dict[int, np.ndarray] = {}
        entity_cache: dict[int, set[str]] = {}

        for cluster_row in existing_clusters:
            cid = cluster_row["id"]
            centroid = _compute_cluster_centroid(cid, conn)
            if centroid is not None:
                centroid_cache[cid] = centroid
            entity_cache[cid] = _cluster_entity_set(cid, conn)

        rescued = 0
        processed = 0

        for article in unclustered:
            text = _make_text_for_embedding(article["title"], article["body"])
            try:
                vec = embed(text)
            except Exception as exc:
                logger.error("Embedding failed for article #{}: {}", article["id"], exc)
                continue

            article_entities = _article_entity_set(article["id"], conn)

            # Find best candidate cluster by cosine similarity
            best_cluster_id: int | None = None
            best_cosine: float = -1.0

            for cid, centroid in centroid_cache.items():
                sim = cosine_similarity(vec, centroid)
                if sim > best_cosine:
                    best_cosine = sim
                    best_cluster_id = cid

            # Two-tier decision
            join_cluster_id: int | None = None
            match_reason: str = ""

            if best_cosine >= CLUSTER_SIMILARITY_THRESHOLD and best_cluster_id is not None:
                join_cluster_id = best_cluster_id
                match_reason = f"cosine={best_cosine:.3f}"

            elif best_cosine >= CLUSTER_ENTITY_RESCUE_COSINE and best_cluster_id is not None:
                jaccard = _entity_jaccard(
                    article_entities, entity_cache.get(best_cluster_id, set())
                )
                if jaccard >= CLUSTER_ENTITY_RESCUE_JACCARD:
                    join_cluster_id = best_cluster_id
                    match_reason = f"entity-rescue cosine={best_cosine:.3f} jaccard={jaccard:.2f}"
                    rescued += 1

            if join_cluster_id is not None:
                assign_article_to_cluster(article["id"], join_cluster_id, conn)
                logger.debug(
                    "Article #{} '{}' → cluster #{} ({})",
                    article["id"], article["title"][:50], join_cluster_id, match_reason,
                )
                # Update both caches and persist new centroid
                new_centroid = _compute_cluster_centroid(join_cluster_id, conn)
                if new_centroid is not None:
                    centroid_cache[join_cluster_id] = new_centroid
                    update_cluster_centroid(
                        join_cluster_id, json.dumps(new_centroid.tolist()), conn
                    )
                entity_cache[join_cluster_id] = (
                    entity_cache.get(join_cluster_id, set()) | article_entities
                )
                _refresh_lean_coverage(join_cluster_id, conn)

            else:
                new_cluster_id = insert_cluster(article["title"], conn)
                assign_article_to_cluster(article["id"], new_cluster_id, conn)
                logger.info(
                    "Article #{} → NEW cluster #{} '{}' (best cosine was {:.3f})",
                    article["id"], new_cluster_id, article["title"][:60], best_cosine,
                )
                centroid_cache[new_cluster_id] = vec
                update_cluster_centroid(new_cluster_id, json.dumps(vec.tolist()), conn)
                entity_cache[new_cluster_id] = article_entities
                _refresh_lean_coverage(new_cluster_id, conn)

            processed += 1

        if rescued:
            logger.info(
                "Clustering complete. {} article(s) processed ({} entity-rescued).",
                processed, rescued,
            )
        else:
            logger.info("Clustering complete. {} article(s) processed.", processed)

    return processed


if __name__ == "__main__":
    from database import init_db
    init_db()
    cluster_new_articles()
