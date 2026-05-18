"""
Tests for ingestion.py
"""
import hashlib
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from database import init_db, get_db, url_hash_exists


@pytest.fixture()
def tmp_db(tmp_path):
    """Initialise a fresh test database and patch DB_PATH."""
    db = tmp_path / "test.db"
    with patch("database.DB_PATH", db), patch("ingestion.get_db") as mock_get_db:
        # We still need a real DB for these tests
        import database
        database.DB_PATH = db
        init_db(db)
        yield db


def make_hash(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()


class TestUrlHashExists:
    def test_returns_false_for_new_url(self, tmp_db):
        with get_db(tmp_db) as conn:
            assert url_hash_exists(make_hash("https://example.com/a"), conn) is False

    def test_returns_true_after_insert(self, tmp_db):
        from database import insert_article
        with get_db(tmp_db) as conn:
            h = make_hash("https://example.com/b")
            insert_article(
                url="https://example.com/b",
                url_hash=h,
                title="Test",
                body="Body text",
                source_name="TestSource",
                source_lean="center",
                published_at=None,
                fetched_at=datetime.utcnow().isoformat(),
                conn=conn,
            )
        with get_db(tmp_db) as conn:
            assert url_hash_exists(h, conn) is True


class TestParsePublished:
    def test_rfc_date_string(self):
        from ingestion import _parse_published
        entry = MagicMock()
        entry.published = "Mon, 01 Jan 2024 12:00:00 +0000"
        entry.updated = None
        entry.published_parsed = None
        entry.updated_parsed = None
        result = _parse_published(entry)
        assert result is not None
        assert result.year == 2024

    def test_returns_none_for_missing_dates(self):
        from ingestion import _parse_published
        entry = MagicMock()
        entry.published = None
        entry.updated = None
        entry.published_parsed = None
        entry.updated_parsed = None
        result = _parse_published(entry)
        assert result is None


class TestIsTooOld:
    def test_recent_article_is_not_too_old(self):
        from ingestion import _is_too_old
        recent = datetime.now(tz=timezone.utc) - timedelta(hours=1)
        assert _is_too_old(recent) is False

    def test_old_article_is_too_old(self):
        from ingestion import _is_too_old
        old = datetime.now(tz=timezone.utc) - timedelta(hours=100)
        assert _is_too_old(old) is True

    def test_none_is_not_too_old(self):
        from ingestion import _is_too_old
        assert _is_too_old(None) is False


class TestExtractText:
    def test_falls_back_to_feed_summary_on_failure(self):
        from ingestion import _extract_text
        with patch("trafilatura.fetch_url", return_value=None):
            text, used_fallback = _extract_text("https://example.com", "fallback summary")
        assert text == "fallback summary"
        assert used_fallback is True

    def test_uses_trafilatura_when_successful(self):
        from ingestion import _extract_text
        with (
            patch("trafilatura.fetch_url", return_value=b"<html>content</html>"),
            patch("trafilatura.extract", return_value="Extracted full article text " * 10),
        ):
            text, used_fallback = _extract_text("https://example.com", "fallback")
        assert "Extracted" in text
        assert used_fallback is False
