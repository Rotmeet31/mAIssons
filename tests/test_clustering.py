"""
Tests for clustering.py
"""
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestCosineSimilarity:
    def test_identical_vectors(self):
        from clustering import cosine_similarity
        v = np.array([1.0, 0.0, 0.0])
        assert pytest.approx(cosine_similarity(v, v), abs=1e-6) == 1.0

    def test_orthogonal_vectors(self):
        from clustering import cosine_similarity
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert pytest.approx(cosine_similarity(a, b), abs=1e-6) == 0.0

    def test_opposite_vectors(self):
        from clustering import cosine_similarity
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert pytest.approx(cosine_similarity(a, b), abs=1e-6) == -1.0


class TestMakeTextForEmbedding:
    def test_combines_title_and_snippet(self):
        from clustering import _make_text_for_embedding
        text = _make_text_for_embedding("Title", "A" * 500)
        assert text.startswith("Title.")
        # body snippet should be truncated to BODY_SNIPPET_CHARS
        from config import BODY_SNIPPET_CHARS
        assert len(text) <= len("Title. ") + BODY_SNIPPET_CHARS


class TestLeanCoverageRefresh:
    def test_ready_when_two_leans_present(self, tmp_path):
        """Cluster marked ready when articles from ≥2 leans are present."""
        from database import init_db, get_db, insert_article, insert_cluster, assign_article_to_cluster
        from clustering import _refresh_lean_coverage
        import database
        from datetime import datetime

        db = tmp_path / "test.db"
        database.DB_PATH = db
        init_db(db)

        now = datetime.utcnow().isoformat()
        with get_db(db) as conn:
            cid = insert_cluster("Test headline", conn)
            a1 = insert_article(
                url="https://a.com/1", url_hash="h1", title="T1", body="B1",
                source_name="S1", source_lean="left",
                published_at=None, fetched_at=now, conn=conn,
            )
            a2 = insert_article(
                url="https://a.com/2", url_hash="h2", title="T2", body="B2",
                source_name="S2", source_lean="right",
                published_at=None, fetched_at=now, conn=conn,
            )
            assign_article_to_cluster(a1, cid, conn)
            assign_article_to_cluster(a2, cid, conn)
            coverage = _refresh_lean_coverage(cid, conn)

        assert coverage["left"] == 1
        assert coverage["right"] == 1

        with get_db(db) as conn:
            row = conn.execute(
                "SELECT ready_for_analysis FROM clusters WHERE id=?", (cid,)
            ).fetchone()
        assert row["ready_for_analysis"] == 1

    def test_not_ready_with_single_lean(self, tmp_path):
        from database import init_db, get_db, insert_article, insert_cluster, assign_article_to_cluster
        from clustering import _refresh_lean_coverage
        import database
        from datetime import datetime

        db = tmp_path / "test2.db"
        database.DB_PATH = db
        init_db(db)

        now = datetime.utcnow().isoformat()
        with get_db(db) as conn:
            cid = insert_cluster("Headline only left", conn)
            a1 = insert_article(
                url="https://b.com/1", url_hash="hx1", title="T", body="B",
                source_name="S", source_lean="left",
                published_at=None, fetched_at=now, conn=conn,
            )
            assign_article_to_cluster(a1, cid, conn)
            _refresh_lean_coverage(cid, conn)

        with get_db(db) as conn:
            row = conn.execute(
                "SELECT ready_for_analysis FROM clusters WHERE id=?", (cid,)
            ).fetchone()
        assert row["ready_for_analysis"] == 0
