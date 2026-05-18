"""
MediaLens configuration: RSS sources, lean labels, and application settings.
"""
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = Path(os.environ.get("MEDIALENS_DB_PATH", str(DATA_DIR / "medialens.db")))

# ── RSS Sources ────────────────────────────────────────────────────────────
RSS_SOURCES: list[dict] = [
    # ── Left-leaning ───────────────────────────────────────────────────────
    {
        "name": "The Guardian",
        "url": "https://www.theguardian.com/world/rss",
        "lean": "left",
    },
    {
        "name": "NPR",
        "url": "https://feeds.npr.org/1001/rss.xml",
        "lean": "left",
    },
    {
        "name": "HuffPost",
        # chaski endpoint is more stable than the /section/front-page/feed path
        "url": "https://chaski.huffpost.com/us/auto/vertical/front-page",
        "lean": "left",
    },
    {
        "name": "Vox",
        "url": "https://www.vox.com/rss/index.xml",
        "lean": "left",
    },
    # ── Center ─────────────────────────────────────────────────────────────
    {
        "name": "BBC",
        "url": "https://feeds.bbci.co.uk/news/world/rss.xml",
        "lean": "center",
    },
    {
        "name": "AP",
        "url": "https://feeds.apnews.com/rss/apf-topnews",
        "lean": "center",
    },
    {
        "name": "Reuters",
        # Reuters shut down their direct RSS in 2020; Google News RSS filtered
        # to reuters.com gives us the same articles via a stable endpoint.
        "url": "https://news.google.com/rss/search?q=site:reuters.com&hl=en-US&gl=US&ceid=US:en",
        "lean": "center",
    },
    {
        "name": "Al Jazeera",
        "url": "https://www.aljazeera.com/xml/rss/all.xml",
        "lean": "center",
    },
    # ── Right-leaning ──────────────────────────────────────────────────────
    {
        "name": "Fox News",
        # feedburner endpoint is confirmed working; google-publisher/world.xml is CDN-unreliable
        "url": "https://moxie.foxnews.com/feedburner/latest.xml",
        "lean": "right",
    },
    {
        "name": "Daily Wire",
        "url": "https://www.dailywire.com/feeds/rss.xml",
        "lean": "right",
    },
    {
        "name": "Breitbart",
        # Using direct domain feed; FeedBurner is deprecated infrastructure
        "url": "https://www.breitbart.com/feed/",
        "lean": "right",
    },
    {
        "name": "Washington Examiner",
        "url": "https://www.washingtonexaminer.com/feed/",
        "lean": "right",
    },
]

# ── Ingestion settings ─────────────────────────────────────────────────────
ARTICLE_MAX_AGE_HOURS: int = 72          # Skip articles older than this
FETCH_TIMEOUT_SECONDS: int = 15          # HTTP timeout per feed

# ── Clustering settings ────────────────────────────────────────────────────
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
CLUSTER_SIMILARITY_THRESHOLD: float = 0.75   # cosine sim to join existing cluster (primary)
CLUSTER_ENTITY_RESCUE_COSINE: float = 0.60   # lower cosine bound when entities also agree
CLUSTER_ENTITY_RESCUE_JACCARD: float = 0.25  # min entity Jaccard to rescue a borderline match
CLUSTER_ENTITY_LABELS: frozenset = frozenset({"PERSON", "ORG", "GPE"})  # high-salience labels only
BODY_SNIPPET_CHARS: int = 300                # chars of body used for clustering embeddings only
ANALYSIS_BODY_CHARS: int = 2000             # chars of body sent to LLM for lean/bias analysis

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
