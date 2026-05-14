"""
Database setup and all query helpers for MediaLens (SQLite).
"""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator

from loguru import logger

from config import DB_PATH


# ── Connection management ──────────────────────────────────────────────────

def _get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def get_db(db_path: Path = DB_PATH) -> Generator[sqlite3.Connection, None, None]:
    """Context manager that yields a connection and commits/rolls back automatically."""
    conn = _get_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Schema creation ────────────────────────────────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS articles (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    url           TEXT    NOT NULL,
    url_hash      TEXT    NOT NULL UNIQUE,
    title         TEXT    NOT NULL,
    body          TEXT    NOT NULL,
    source_name   TEXT    NOT NULL,
    source_lean   TEXT    NOT NULL CHECK(source_lean IN ('left','center','right')),
    published_at  TEXT,
    fetched_at    TEXT    NOT NULL,
    cluster_id    INTEGER REFERENCES clusters(id) ON DELETE SET NULL,
    analyzed      INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_articles_url_hash  ON articles(url_hash);
CREATE INDEX IF NOT EXISTS idx_articles_cluster_id ON articles(cluster_id);
CREATE INDEX IF NOT EXISTS idx_articles_analyzed   ON articles(analyzed);

CREATE TABLE IF NOT EXISTS clusters (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    representative_headline TEXT NOT NULL,
    lean_coverage         TEXT NOT NULL DEFAULT '{"left":0,"center":0,"right":0}',
    ready_for_analysis    INTEGER NOT NULL DEFAULT 0,
    created_at            TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS analysis (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id       INTEGER NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
    bias_label       TEXT    NOT NULL CHECK(bias_label IN ('left','center','right')),
    bias_score       REAL    NOT NULL,
    confidence       REAL    NOT NULL,
    framing_summary  TEXT    NOT NULL,
    prompt_tokens    INTEGER,
    response_tokens  INTEGER,
    analyzed_at      TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_analysis_article_id ON analysis(article_id);

CREATE TABLE IF NOT EXISTS cluster_analysis (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    cluster_id      INTEGER NOT NULL UNIQUE REFERENCES clusters(id) ON DELETE CASCADE,
    consensus       TEXT    NOT NULL DEFAULT '[]',
    disagreements   TEXT    NOT NULL DEFAULT '[]',
    gaps            TEXT    NOT NULL DEFAULT '',
    analyzed_at     TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_cluster_analysis_cluster_id ON cluster_analysis(cluster_id);

CREATE TABLE IF NOT EXISTS article_entities (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id  INTEGER NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
    text        TEXT    NOT NULL,
    normalized  TEXT    NOT NULL,
    label       TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_entities_normalized  ON article_entities(normalized);
CREATE INDEX IF NOT EXISTS idx_entities_article_id  ON article_entities(article_id);

CREATE TABLE IF NOT EXISTS lean_audit (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    source_name   TEXT    NOT NULL,
    config_lean   TEXT    NOT NULL,
    computed_lean TEXT,
    confidence    REAL,
    sample_size   INTEGER,
    reasoning     TEXT,
    audited_at    TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_lean_audit_source ON lean_audit(source_name);
"""


def init_db(db_path: Path = DB_PATH) -> None:
    """Create database file and tables if they don't already exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with get_db(db_path) as conn:
        conn.executescript(SCHEMA_SQL)
    logger.info("Database initialised at {}", db_path)


# ── Article helpers ────────────────────────────────────────────────────────

def url_hash_exists(url_hash: str, conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT 1 FROM articles WHERE url_hash = ?", (url_hash,)
    ).fetchone()
    return row is not None


def insert_article(
    *,
    url: str,
    url_hash: str,
    title: str,
    body: str,
    source_name: str,
    source_lean: str,
    published_at: str | None,
    fetched_at: str,
    conn: sqlite3.Connection,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO articles
            (url, url_hash, title, body, source_name, source_lean,
             published_at, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (url, url_hash, title, body, source_name, source_lean,
         published_at, fetched_at),
    )
    return cur.lastrowid  # type: ignore[return-value]


def get_unclustered_articles(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM articles WHERE cluster_id IS NULL ORDER BY fetched_at DESC"
    ).fetchall()


def get_articles_by_cluster(
    cluster_id: int, conn: sqlite3.Connection
) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM articles WHERE cluster_id = ?", (cluster_id,)
    ).fetchall()


def assign_article_to_cluster(
    article_id: int, cluster_id: int, conn: sqlite3.Connection
) -> None:
    conn.execute(
        "UPDATE articles SET cluster_id = ? WHERE id = ?",
        (cluster_id, article_id),
    )


def get_unanalyzed_articles_in_ready_clusters(
    conn: sqlite3.Connection,
) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT a.*
        FROM   articles a
        JOIN   clusters c ON a.cluster_id = c.id
        WHERE  c.ready_for_analysis = 1
          AND  a.analyzed = 0
        ORDER  BY a.fetched_at
        """
    ).fetchall()


def mark_article_analyzed(article_id: int, conn: sqlite3.Connection) -> None:
    conn.execute(
        "UPDATE articles SET analyzed = 1 WHERE id = ?", (article_id,)
    )


# ── Cluster helpers ────────────────────────────────────────────────────────

def insert_cluster(
    representative_headline: str,
    conn: sqlite3.Connection,
) -> int:
    now = datetime.utcnow().isoformat()
    cur = conn.execute(
        """
        INSERT INTO clusters (representative_headline, lean_coverage, created_at)
        VALUES (?, ?, ?)
        """,
        (
            representative_headline,
            json.dumps({"left": 0, "center": 0, "right": 0}),
            now,
        ),
    )
    return cur.lastrowid  # type: ignore[return-value]


def get_all_clusters(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM clusters ORDER BY created_at DESC").fetchall()


def get_ready_clusters(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM clusters WHERE ready_for_analysis = 1 ORDER BY created_at DESC"
    ).fetchall()


def update_cluster_lean_coverage(
    cluster_id: int, lean_coverage: dict, conn: sqlite3.Connection
) -> None:
    conn.execute(
        "UPDATE clusters SET lean_coverage = ? WHERE id = ?",
        (json.dumps(lean_coverage), cluster_id),
    )


def update_cluster_ready(
    cluster_id: int, ready: bool, conn: sqlite3.Connection
) -> None:
    conn.execute(
        "UPDATE clusters SET ready_for_analysis = ? WHERE id = ?",
        (1 if ready else 0, cluster_id),
    )


def search_clusters(query: str, conn: sqlite3.Connection) -> list[sqlite3.Row]:
    pattern = f"%{query}%"
    return conn.execute(
        """
        SELECT * FROM clusters
        WHERE representative_headline LIKE ?
        ORDER BY created_at DESC
        """,
        (pattern,),
    ).fetchall()


# ── Analysis helpers ───────────────────────────────────────────────────────

def insert_analysis(
    *,
    article_id: int,
    bias_label: str,
    bias_score: float,
    confidence: float,
    framing_summary: str,
    prompt_tokens: int | None,
    response_tokens: int | None,
    conn: sqlite3.Connection,
) -> int:
    now = datetime.utcnow().isoformat()
    cur = conn.execute(
        """
        INSERT INTO analysis
            (article_id, bias_label, bias_score, confidence,
             framing_summary, prompt_tokens, response_tokens, analyzed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            article_id, bias_label, bias_score, confidence,
            framing_summary, prompt_tokens, response_tokens, now,
        ),
    )
    return cur.lastrowid  # type: ignore[return-value]


def get_analysis_for_article(
    article_id: int, conn: sqlite3.Connection
) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM analysis WHERE article_id = ?", (article_id,)
    ).fetchone()


def get_cluster_with_articles_and_analysis(
    cluster_id: int, conn: sqlite3.Connection
) -> dict:
    """Return a dict with cluster info, its articles, and analysis rows."""
    cluster = conn.execute(
        "SELECT * FROM clusters WHERE id = ?", (cluster_id,)
    ).fetchone()
    if not cluster:
        return {}

    articles = get_articles_by_cluster(cluster_id, conn)
    result: dict = dict(cluster)
    result["lean_coverage"] = json.loads(result["lean_coverage"])
    result["articles"] = []
    for art in articles:
        art_dict = dict(art)
        art_dict["analysis"] = None
        row = get_analysis_for_article(art["id"], conn)
        if row:
            art_dict["analysis"] = dict(row)
        result["articles"].append(art_dict)
    return result


# ── Cluster analysis helpers ───────────────────────────────────────────────

def get_unanalyzed_ready_clusters(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT c.*
        FROM   clusters c
        LEFT   JOIN cluster_analysis ca ON ca.cluster_id = c.id
        WHERE  c.ready_for_analysis = 1
          AND  ca.id IS NULL
        ORDER  BY c.created_at
        """
    ).fetchall()


def insert_cluster_analysis(
    *,
    cluster_id: int,
    consensus: list,
    disagreements: list,
    gaps: str,
    conn: sqlite3.Connection,
) -> int:
    now = datetime.utcnow().isoformat()
    cur = conn.execute(
        """
        INSERT INTO cluster_analysis
            (cluster_id, consensus, disagreements, gaps, analyzed_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (cluster_id, json.dumps(consensus), json.dumps(disagreements), gaps, now),
    )
    return cur.lastrowid  # type: ignore[return-value]


def get_cluster_analysis(
    cluster_id: int, conn: sqlite3.Connection
) -> dict | None:
    row = conn.execute(
        "SELECT * FROM cluster_analysis WHERE cluster_id = ?", (cluster_id,)
    ).fetchone()
    if not row:
        return None
    result = dict(row)
    result["consensus"] = json.loads(result["consensus"])
    result["disagreements"] = json.loads(result["disagreements"])
    return result


# ── Entity helpers ─────────────────────────────────────────────────────────

def insert_article_entities(
    article_id: int,
    entities: list[dict],
    conn: sqlite3.Connection,
) -> None:
    conn.executemany(
        "INSERT INTO article_entities (article_id, text, normalized, label) VALUES (?, ?, ?, ?)",
        [(article_id, e["text"], e["normalized"], e["label"]) for e in entities],
    )


def get_articles_by_entity(
    normalized: str, conn: sqlite3.Connection, limit: int = 20
) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT a.*, ae.text as entity_text, ae.label as entity_label
        FROM   article_entities ae
        JOIN   articles a ON a.id = ae.article_id
        WHERE  ae.normalized LIKE ?
        ORDER  BY a.fetched_at DESC
        LIMIT  ?
        """,
        (f"%{normalized}%", limit),
    ).fetchall()


def get_top_entities(
    conn: sqlite3.Connection,
    limit: int = 20,
    label: str | None = None,
) -> list[sqlite3.Row]:
    if label:
        return conn.execute(
            """
            SELECT normalized, text, label, COUNT(*) as mentions,
                   COUNT(DISTINCT article_id) as articles
            FROM   article_entities
            WHERE  label = ?
            GROUP  BY normalized
            ORDER  BY articles DESC
            LIMIT  ?
            """,
            (label, limit),
        ).fetchall()
    return conn.execute(
        """
        SELECT normalized, text, label, COUNT(*) as mentions,
               COUNT(DISTINCT article_id) as articles
        FROM   article_entities
        GROUP  BY normalized
        ORDER  BY articles DESC
        LIMIT  ?
        """,
        (limit,),
    ).fetchall()


def get_entities_for_cluster(
    cluster_id: int, conn: sqlite3.Connection
) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT ae.normalized, ae.text, ae.label, COUNT(*) as mentions
        FROM   article_entities ae
        JOIN   articles a ON a.id = ae.article_id
        WHERE  a.cluster_id = ?
        GROUP  BY ae.normalized
        ORDER  BY mentions DESC
        LIMIT  15
        """,
        (cluster_id,),
    ).fetchall()


# ── Lean audit helpers ─────────────────────────────────────────────────────

def insert_lean_audit(
    *,
    source_name: str,
    config_lean: str,
    computed_lean: str | None,
    confidence: float | None,
    sample_size: int,
    reasoning: str,
    conn: sqlite3.Connection,
) -> int:
    now = datetime.utcnow().isoformat()
    cur = conn.execute(
        """
        INSERT INTO lean_audit
            (source_name, config_lean, computed_lean, confidence,
             sample_size, reasoning, audited_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (source_name, config_lean, computed_lean, confidence,
         sample_size, reasoning, now),
    )
    return cur.lastrowid  # type: ignore[return-value]


def get_latest_lean_audits(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Most recent audit result per source."""
    return conn.execute(
        """
        SELECT la.*
        FROM   lean_audit la
        INNER  JOIN (
            SELECT source_name, MAX(audited_at) as latest
            FROM   lean_audit
            GROUP  BY source_name
        ) mx ON la.source_name = mx.source_name AND la.audited_at = mx.latest
        ORDER  BY la.source_name
        """
    ).fetchall()
