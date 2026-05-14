# MediaLens

A news bias analyzer that fetches RSS articles across left/center/right outlets,
clusters stories about the same event, and uses Gemini to score and explain
political framing.

## Project structure

```
medialens/
  data/             SQLite database
  src/
    config.py       RSS sources, lean labels, all settings
    database.py     Schema creation and all query helpers
    ingestion.py    RSS fetching, text extraction, dedup, DB storage
    clustering.py   Embed articles, group by story, mark ready clusters
    analysis.py     Gemini bias scoring for each article
  app.py            Streamlit UI
  requirements.txt
  tests/
    test_ingestion.py
    test_clustering.py
    test_analysis.py
```

## Setup

### 1. Create and activate a virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your Gemini API key

```bash
export GEMINI_API_KEY="your-key-here"
```

Get a free key at <https://aistudio.google.com/app/apikey>.

## Running

### Streamlit UI (recommended)

```bash
streamlit run app.py
```

Use the sidebar buttons to **Ingest → Cluster → Analyze**.

### Individual pipeline steps

```bash
# from the repo root
python src/ingestion.py   # fetch all feeds
python src/clustering.py  # cluster new articles
python src/analysis.py    # analyze ready clusters
```

## Running tests

```bash
pytest tests/ -v
```

## Pipeline overview

```
RSS feeds
    │
    ▼
ingestion.py  ──► articles table (url_hash dedup, 72-hr window)
    │
    ▼
clustering.py ──► clusters table (cosine sim ≥ 0.75, centroid averaging)
    │                ready_for_analysis=True when ≥2 leans covered
    ▼
analysis.py   ──► analysis table (Gemini JSON: bias_label, score, summary)
    │
    ▼
app.py        ──► Streamlit: search, cluster cards, side-by-side articles,
                             bias spectrum bars
```

## Settings (src/config.py)

| Variable | Default | Description |
|---|---|---|
| `ARTICLE_MAX_AGE_HOURS` | 72 | Skip articles older than this |
| `CLUSTER_SIMILARITY_THRESHOLD` | 0.75 | Cosine sim to join an existing cluster |
| `EMBEDDING_MODEL` | all-MiniLM-L6-v2 | Sentence-transformers model |
| `GEMINI_MODEL` | gemini-1.5-flash | Gemini model for analysis |
| `GEMINI_TEMPERATURE` | 0.2 | Lower = more deterministic |
| `MAX_RETRIES` | 1 | Retry count on JSON parse failure |

## Background scheduling (optional)

`APScheduler` is included. To run the full pipeline automatically,
add a scheduler script:

```python
from apscheduler.schedulers.blocking import BlockingScheduler
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))
from ingestion import fetch_all_feeds
from clustering import cluster_new_articles
from analysis import run_analysis
from database import init_db
from config import INGEST_INTERVAL_MINUTES, CLUSTER_INTERVAL_MINUTES, ANALYSIS_INTERVAL_MINUTES

init_db()
scheduler = BlockingScheduler()
scheduler.add_job(fetch_all_feeds,    "interval", minutes=INGEST_INTERVAL_MINUTES)
scheduler.add_job(cluster_new_articles, "interval", minutes=CLUSTER_INTERVAL_MINUTES)
scheduler.add_job(run_analysis,       "interval", minutes=ANALYSIS_INTERVAL_MINUTES)
scheduler.start()
```
