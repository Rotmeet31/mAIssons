"""
RSS ingestion: fetch feeds, extract full text, dedup, store to DB.
"""
import hashlib
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

import feedparser
import trafilatura
from loguru import logger

from config import ARTICLE_MAX_AGE_HOURS, FETCH_TIMEOUT_SECONDS, RSS_SOURCES
from database import get_db, insert_article, insert_article_entities, url_hash_exists
from entities import extract_entities


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_url_hash(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()


def _parse_published(entry: feedparser.FeedParserDict) -> datetime | None:
    """Return a timezone-aware datetime or None."""
    for attr in ("published", "updated"):
        raw = getattr(entry, attr, None)
        if raw:
            try:
                dt = parsedate_to_datetime(raw)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except Exception:
                continue
    # feedparser sometimes parses into _parsed tuples
    for attr in ("published_parsed", "updated_parsed"):
        parsed = getattr(entry, attr, None)
        if parsed:
            try:
                return datetime(*parsed[:6], tzinfo=timezone.utc)
            except Exception:
                continue
    return None


def _is_too_old(published_at: datetime | None) -> bool:
    if published_at is None:
        return False  # unknown age → keep
    cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=ARTICLE_MAX_AGE_HOURS)
    return published_at < cutoff


def _extract_text(url: str, feed_summary: str) -> str:
    """Try trafilatura first; fall back to feed summary."""
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded, include_comments=False)
            if text and len(text.strip()) > 100:
                return text.strip()
    except Exception as exc:
        logger.debug("trafilatura failed for {}: {}", url, exc)
    return feed_summary or ""


# ── Core fetch logic ───────────────────────────────────────────────────────

def fetch_feed(source: dict) -> list[dict]:
    """
    Parse a single RSS source and return a list of article dicts ready for DB insertion.
    Already-seen URLs are filtered out.
    """
    name = source["name"]
    url = source["url"]
    lean = source["lean"]

    logger.info("Fetching feed: {} ({})", name, lean)

    try:
        parsed = feedparser.parse(url, request_headers={"User-Agent": "MediaLens/1.0"})
    except Exception as exc:
        logger.error("Failed to parse feed {}: {}", name, exc)
        return []

    if parsed.bozo and parsed.bozo_exception:
        logger.warning("Feed {} has parse issues: {}", name, parsed.bozo_exception)

    articles: list[dict] = []
    fetched_at = datetime.utcnow().isoformat()

    with get_db() as conn:
        for entry in parsed.entries:
            link = getattr(entry, "link", None)
            if not link:
                continue

            url_hash = _make_url_hash(link)
            if url_hash_exists(url_hash, conn):
                logger.debug("Skipping duplicate: {}", link)
                continue

            published_at = _parse_published(entry)
            if _is_too_old(published_at):
                logger.debug("Skipping old article ({}): {}", published_at, link)
                continue

            title = getattr(entry, "title", "").strip()
            if not title:
                continue

            feed_summary = getattr(entry, "summary", "") or ""
            body = _extract_text(link, feed_summary)
            if not body:
                logger.debug("No body extracted for {}", link)
                continue

            article_id = insert_article(
                url=link,
                url_hash=url_hash,
                title=title,
                body=body,
                source_name=name,
                source_lean=lean,
                published_at=published_at.isoformat() if published_at else None,
                fetched_at=fetched_at,
                conn=conn,
            )
            entities = extract_entities(body)
            if entities:
                insert_article_entities(article_id, entities, conn)
                logger.debug("Extracted {} entities from article #{}", len(entities), article_id)

            logger.info("Stored article #{} '{}' from {}", article_id, title[:60], name)
            articles.append({"id": article_id, "title": title, "source": name})

    return articles


def fetch_all_feeds() -> list[dict]:
    """Fetch every configured RSS source and return all new articles stored."""
    all_articles: list[dict] = []
    for source in RSS_SOURCES:
        try:
            new = fetch_feed(source)
            all_articles.extend(new)
        except Exception as exc:
            logger.error("Unexpected error fetching {}: {}", source["name"], exc)
    logger.info(
        "Ingestion complete. {} new articles stored across {} sources.",
        len(all_articles),
        len(RSS_SOURCES),
    )
    return all_articles


if __name__ == "__main__":
    from database import init_db
    init_db()
    fetch_all_feeds()
