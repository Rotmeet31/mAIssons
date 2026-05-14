"""
Cluster articles about the same story using sentence-transformer embeddings
and cosine similarity against existing cluster centroids.
"""
import json
from typing import Any

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from config import BODY_SNIPPET_CHARS, CLUSTER_SIMILARITY_THRESHOLD, EMBEDDING_MODEL
from database import (
    assign_article_to_cluster,
    get_all_clusters,
    get_articles_by_cluster,
    get_db,
    get_unclustered_articles,
    insert_cluster,
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
    # Re-normalise after averaging
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
    (if cosine similarity ≥ threshold) or create a new one.

    Returns the number of articles processed.
    """
    with get_db() as conn:
        unclustered = get_unclustered_articles(conn)
        if not unclustered:
            logger.info("No unclustered articles found.")
            return 0

        logger.info("Clustering {} unclustered article(s)...", len(unclustered))

        # Build current centroid cache: {cluster_id: centroid_vector}
        existing_clusters = get_all_clusters(conn)
        centroid_cache: dict[int, np.ndarray] = {}
        for cluster_row in existing_clusters:
            cid = cluster_row["id"]
            centroid = _compute_cluster_centroid(cid, conn)
            if centroid is not None:
                centroid_cache[cid] = centroid

        processed = 0
        for article in unclustered:
            text = _make_text_for_embedding(article["title"], article["body"])
            try:
                vec = embed(text)
            except Exception as exc:
                logger.error(
                    "Embedding failed for article #{}: {}", article["id"], exc
                )
                continue

            # Find best matching cluster
            best_cluster_id: int | None = None
            best_sim: float = -1.0
            for cid, centroid in centroid_cache.items():
                sim = cosine_similarity(vec, centroid)
                if sim > best_sim:
                    best_sim = sim
                    best_cluster_id = cid

            if best_sim >= CLUSTER_SIMILARITY_THRESHOLD and best_cluster_id is not None:
                # Add to existing cluster
                assign_article_to_cluster(article["id"], best_cluster_id, conn)
                logger.debug(
                    "Article #{} -> cluster #{} (sim={:.3f})",
                    article["id"],
                    best_cluster_id,
                    best_sim,
                )
                # Update centroid in cache
                new_centroid = _compute_cluster_centroid(best_cluster_id, conn)
                if new_centroid is not None:
                    centroid_cache[best_cluster_id] = new_centroid
                _refresh_lean_coverage(best_cluster_id, conn)
            else:
                # Create a new cluster
                new_cluster_id = insert_cluster(article["title"], conn)
                assign_article_to_cluster(article["id"], new_cluster_id, conn)
                logger.info(
                    "Article #{} -> NEW cluster #{} '{}'",
                    article["id"],
                    new_cluster_id,
                    article["title"][:60],
                )
                centroid_cache[new_cluster_id] = vec
                _refresh_lean_coverage(new_cluster_id, conn)

            processed += 1

    logger.info("Clustering complete. {} article(s) processed.", processed)
    return processed


if __name__ == "__main__":
    from database import init_db
    init_db()
    cluster_new_articles()
