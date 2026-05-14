"""
MediaLens configuration: RSS sources, lean labels, and application settings.
"""
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "medialens.db"

# ── RSS Sources ────────────────────────────────────────────────────────────
RSS_SOURCES: list[dict] = [
    # Left-leaning
    {
        "name": "The Guardian",
        "url": "https://www.theguardian.com/world/rss",
        "lean": "left",
    },
    {
        "name": "HuffPost",
        "url": "https://www.huffpost.com/section/front-page/feed",
        "lean": "left",
    },
    {
        "name": "NPR",
        "url": "https://feeds.npr.org/1001/rss.xml",
        "lean": "left",
    },
    # Center
    {
        "name": "Reuters",
        "url": "https://feeds.reuters.com/reuters/topNews",
        "lean": "center",
    },
    {
        "name": "BBC",
        "url": "http://feeds.bbci.co.uk/news/world/rss.xml",
        "lean": "center",
    },
    {
        "name": "AP",
        "url": "https://feeds.apnews.com/rss/apf-topnews",
        "lean": "center",
    },
    # Right-leaning
    {
        "name": "Fox News",
        "url": "https://moxie.foxnews.com/google-publisher/world.xml",
        "lean": "right",
    },
    {
        "name": "Daily Wire",
        "url": "https://www.dailywire.com/feeds/rss.xml",
        "lean": "right",
    },
    {
        "name": "Breitbart",
        "url": "http://feeds.feedburner.com/breitbart",
        "lean": "right",
    },
]

# ── Ingestion settings ─────────────────────────────────────────────────────
ARTICLE_MAX_AGE_HOURS: int = 72          # Skip articles older than this
FETCH_TIMEOUT_SECONDS: int = 15          # HTTP timeout per feed

# ── Clustering settings ────────────────────────────────────────────────────
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
CLUSTER_SIMILARITY_THRESHOLD: float = 0.75   # cosine sim to join existing cluster
BODY_SNIPPET_CHARS: int = 300                # chars of body used for embedding

# ── Analysis settings ─────────────────────────────────────────────────────
OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
LLM_MODEL: str = "deepseek/deepseek-v4-flash:free"
LLM_TEMPERATURE: float = 0.2
MAX_RETRIES: int = 1                    # retry count on JSON parse failure

# ── Scheduler settings ────────────────────────────────────────────────────
INGEST_INTERVAL_MINUTES: int = 30
CLUSTER_INTERVAL_MINUTES: int = 31
ANALYSIS_INTERVAL_MINUTES: int = 35

# ── Lean display ──────────────────────────────────────────────────────────
LEAN_COLORS: dict[str, str] = {
    "left": "#3b82f6",    # blue
    "center": "#6b7280",  # gray
    "right": "#ef4444",   # red
}
