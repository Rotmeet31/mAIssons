"""
RSS ingestion: fetch feeds, extract full text, dedup, store to DB.
"""
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

import feedparser
import trafilatura
from loguru import logger

from config import ARTICLE_MAX_AGE_HOURS, FETCH_TIMEOUT_SECONDS, RSS_SOURCES
from database import get_db, insert_article, insert_article_entities, url_hash_exists
from entities import extract_entities


# ── Per-source stats ───────────────────────────────────────────────────────

@dataclass
class FeedStats:
    name: str
    lean: str
    entries_in_feed: int = 0
    stored: int = 0
    skipped_dupe: int = 0
    skipped_old: int = 0
    skipped_nobody: int = 0
    trafilatura_fallbacks: int = 0
    parse_error: str | None = None

    @property
    def healthy(self) -> bool:
        return self.parse_error is None and self.entries_in_feed > 0

    def summary_line(self) -> str:
        if self.parse_error:
            return f"  ERROR  {self.parse_error}"
        if self.entries_in_feed == 0:
            return "  EMPTY FEED — check URL"
        parts = [f"feed:{self.entries_in_feed}"]
        parts.append(f"+{self.stored} stored")
        if self.skipped_dupe:
            parts.append(f"{self.skipped_dupe} dupes")
        if self.skipped_old:
            parts.append(f"{self.skipped_old} old")
        if self.skipped_nobody:
            parts.append(f"{self.skipped_nobody} no-body")
        if self.trafilatura_fallbacks:
            parts.append(f"{self.trafilatura_fallbacks} fallback-text")
        return "  " + " | ".join(parts)


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


def _extract_text(url: str, feed_summary: str) -> tuple[str, bool]:
    """
    Try trafilatura first; fall back to feed summary.
    Returns (text, used_fallback).
    """
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded, include_comments=False)
            if text and len(text.strip()) > 100:
                return text.strip(), False
    except Exception as exc:
        logger.debug("trafilatura failed for {}: {}", url, exc)
    summary = feed_summary or ""
    return summary, bool(summary)


# ── Core fetch logic ───────────────────────────────────────────────────────

def fetch_feed(source: dict) -> list[dict]:
    """
    Parse a single RSS source and return a list of newly stored article dicts.
    Logs a per-source summary line at completion.
    """
    name = source["name"]
    url = source["url"]
    lean = source["lean"]

    stats = FeedStats(name=name, lean=lean)
    logger.info("[{}] Fetching {} …", lean.upper(), name)

    try:
        parsed = feedparser.parse(url, request_headers={"User-Agent": "MediaLens/1.0"})
    except Exception as exc:
        stats.parse_error = str(exc)
        logger.error("[{}] {} — feed parse failed: {}", lean.upper(), name, exc)
        _log_source_summary(stats)
        return []

    if parsed.bozo and parsed.bozo_exception:
        logger.warning("[{}] {} — feed has parse issues: {}", lean.upper(), name, parsed.bozo_exception)

    stats.entries_in_feed = len(parsed.entries)

    if stats.entries_in_feed == 0:
        logger.warning("[{}] {} — feed returned 0 entries (dead URL or empty feed)", lean.upper(), name)
        _log_source_summary(stats)
        return []

    articles: list[dict] = []
    fetched_at = datetime.now(tz=timezone.utc).isoformat()

    with get_db() as conn:
        for entry in parsed.entries:
            link = getattr(entry, "link", None)
            if not link:
                continue

            url_hash = _make_url_hash(link)
            if url_hash_exists(url_hash, conn):
                stats.skipped_dupe += 1
                logger.debug("[{}] {} — dupe: {}", lean.upper(), name, link)
                continue

            published_at = _parse_published(entry)
            if _is_too_old(published_at):
                stats.skipped_old += 1
                logger.debug("[{}] {} — too old ({}): {}", lean.upper(), name, published_at, link)
                continue

            title = getattr(entry, "title", "").strip()
            if not title:
                continue

            feed_summary = getattr(entry, "summary", "") or ""
            body, used_fallback = _extract_text(link, feed_summary)
            if not body:
                stats.skipped_nobody += 1
                logger.debug("[{}] {} — no body extracted: {}", lean.upper(), name, link)
                continue

            if used_fallback:
                stats.trafilatura_fallbacks += 1

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
                logger.debug("[{}] {} — {} entities in article #{}", lean.upper(), name, len(entities), article_id)

            stats.stored += 1
            logger.debug("[{}] {} — stored #{}: {}", lean.upper(), name, article_id, title[:60])
            articles.append({"id": article_id, "title": title, "source": name})

    _log_source_summary(stats)
    return articles


def _log_source_summary(stats: FeedStats) -> None:
    line = stats.summary_line()
    if not stats.healthy or stats.stored == 0:
        logger.warning("[{}] {}{}", stats.lean.upper(), stats.name, line)
    else:
        logger.info("[{}] {}{}", stats.lean.upper(), stats.name, line)


# ── Batch fetch ────────────────────────────────────────────────────────────

def fetch_all_feeds() -> list[dict]:
    """Fetch every configured RSS source. Logs a cross-source summary table."""
    results: list[tuple[dict, list[dict]]] = []

    for source in RSS_SOURCES:
        try:
            new = fetch_feed(source)
        except Exception as exc:
            logger.error("Unexpected error fetching {}: {}", source["name"], exc)
            new = []
        results.append((source, new))

    _log_run_summary(results)
    return [art for _, arts in results for art in arts]


def _log_run_summary(results: list[tuple[dict, list[dict]]]) -> None:
    """Log a grouped table showing contribution per source and lean totals."""
    by_lean: dict[str, list[tuple[str, int]]] = {}
    for source, arts in results:
        by_lean.setdefault(source["lean"], []).append((source["name"], len(arts)))

    total_new = sum(len(arts) for _, arts in results)
    active_sources = sum(1 for _, arts in results if arts)

    logger.info("─" * 56)
    logger.info("Ingestion complete — {} new articles, {}/{} sources active",
                total_new, active_sources, len(results))

    for lean in ("left", "center", "right"):
        entries = by_lean.get(lean, [])
        lean_total = sum(n for _, n in entries)
        parts = []
        for name, n in entries:
            parts.append(f"{name} +{n}" if n > 0 else f"{name} ✗")
        logger.info("  {:6s} ({}) | {:>3} new — {}",
                    lean.upper(), len(entries), lean_total, ", ".join(parts))

    logger.info("─" * 56)


if __name__ == "__main__":
    from database import init_db
    init_db()
    fetch_all_feeds()
