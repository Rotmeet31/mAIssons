"""
MediaLens Pipeline Tester

Full pipeline on synthetic seed articles — no RSS fetching.

Stages:
  1. Seed   — insert 9 hand-crafted articles (3 clusters × 3 leans) directly into DB
  2. Cluster — run the real embedding + entity-rescue clustering
  3. Analyze — run LLM analysis on formed clusters (real OpenRouter calls)
  4. Ask     — run the 3-phase ask agent
  5. Fact    — run the 4-phase fact-check agent (includes web search)

Run from project root:
    streamlit run tests/pipeline_test.py
"""

import os
from pathlib import Path

# Must be set BEFORE any src imports so config.DB_PATH picks it up at definition time.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TEST_DB_PATH = _PROJECT_ROOT / "data" / "test_medialens.db"
os.environ["MEDIALENS_DB_PATH"] = str(_TEST_DB_PATH)

import sys
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
from datetime import datetime

import streamlit as st
from loguru import logger

import database
import clustering
import analysis
import agent
import fact_check
from seed_data import SEED_ARTICLES, SEED_ENTITIES, TEST_QUERIES

# ── Loguru sink ────────────────────────────────────────────────────────────────
# Append pipeline log messages to session state so the log panel stays live.

if "log_lines" not in st.session_state:
    st.session_state.log_lines = []

if "log_sink_id" not in st.session_state:
    sid = logger.add(
        lambda msg: st.session_state.log_lines.append(
            f"[{msg.record['time'].strftime('%H:%M:%S')}] {msg.record['message']}"
        ),
        colorize=False,
        format="{message}",
        level="DEBUG",
    )
    st.session_state.log_sink_id = sid


def _render_logs(placeholder):
    lines = st.session_state.log_lines[-60:]
    placeholder.code("\n".join(lines) if lines else "(no logs yet)", language=None)


# ── Pipeline stage functions ───────────────────────────────────────────────────

def _seed(log_ph) -> int:
    logger.info("=== Stage 1: Seeding {} articles ===", len(SEED_ARTICLES))
    _render_logs(log_ph)
    with database.get_db(_TEST_DB_PATH) as conn:
        for art in SEED_ARTICLES:
            article_id = database.insert_article(
                url=art["url"],
                url_hash=art["url_hash"],
                title=art["title"],
                body=art["body"],
                source_name=art["source_name"],
                source_lean=art["source_lean"],
                published_at=art["published_at"],
                fetched_at=datetime.utcnow().isoformat(),
                conn=conn,
            )
            entities = SEED_ENTITIES.get(art["url_hash"], [])
            if entities:
                database.insert_article_entities(article_id, entities, conn)
            short = art["title"][:52] + ("…" if len(art["title"]) > 52 else "")
            logger.info("  + {} | {} | {}", art["source_name"], art["source_lean"], short)
            _render_logs(log_ph)
    logger.info("Stage 1 done — {} articles seeded.", len(SEED_ARTICLES))
    _render_logs(log_ph)
    return len(SEED_ARTICLES)


def _cluster(log_ph) -> list[dict]:
    logger.info("=== Stage 2: Clustering ===")
    _render_logs(log_ph)
    count = clustering.cluster_new_articles()
    logger.info("Clustering processed {} articles.", count)
    _render_logs(log_ph)
    with database.get_db(_TEST_DB_PATH) as conn:
        rows = database.get_all_clusters(conn)
        clusters = []
        for r in rows:
            c = dict(r)
            c["lean_coverage"] = json.loads(c["lean_coverage"])
            clusters.append(c)
    logger.info("Stage 2 done — {} clusters formed.", len(clusters))
    _render_logs(log_ph)
    return clusters


def _analyze(log_ph) -> tuple[int, str | None]:
    logger.info("=== Stage 3: LLM Analysis ===")
    _render_logs(log_ph)
    count, error = analysis.run_analysis()
    if error:
        logger.error("Analysis error: {}", error)
    else:
        logger.info("Stage 3 done — {} clusters analyzed.", count)
    _render_logs(log_ph)
    return count, error


def _ask(log_ph) -> dict:
    query = TEST_QUERIES["ask"]
    logger.info("=== Stage 4: Ask Agent ===")
    logger.info('Query: "{}"', query)
    _render_logs(log_ph)
    result = agent.ask(query)
    if result.get("error"):
        logger.error("Ask agent error: {}", result["error"])
    else:
        logger.info("Stage 4 done. Tools called: {}", result.get("tools_called", []))
    _render_logs(log_ph)
    return result


def _fact_check(log_ph) -> dict:
    claim = TEST_QUERIES["fact_check"]
    logger.info("=== Stage 5: Fact Check ===")
    logger.info('Claim: "{}"', claim)
    _render_logs(log_ph)
    result = fact_check.check(claim)
    if result.get("error"):
        logger.error("Fact check error: {}", result["error"])
    else:
        logger.info(
            "Stage 5 done. Verdict: {} ({:.0%} confidence)",
            result.get("verdict"), result.get("confidence", 0),
        )
    _render_logs(log_ph)
    return result


# ── Results renderers ──────────────────────────────────────────────────────────

def _render_clusters(clusters: list[dict]):
    if not clusters:
        st.info("No clusters formed.")
        return
    for c in clusters:
        cov = c.get("lean_coverage", {})
        left, center, right = cov.get("left", 0), cov.get("center", 0), cov.get("right", 0)
        total = (left + center + right) or 1
        ready = bool(c.get("ready_for_analysis", 0))
        badge = "✅ ready" if ready else "⏳ not ready"
        with st.expander(f"Cluster {c['id']}: {c['representative_headline']}", expanded=True):
            st.caption(f"Status: {badge} · created {c.get('created_at', '')[:19]}")
            col_l, col_c, col_r = st.columns(3)
            col_l.metric("Left", left)
            col_c.metric("Center", center)
            col_r.metric("Right", right)
            st.progress(left / total, text=f"Left {left/total:.0%}")
            st.progress(center / total, text=f"Center {center/total:.0%}")
            st.progress(right / total, text=f"Right {right/total:.0%}")


def _render_analysis(clusters: list[dict]):
    any_found = False
    with database.get_db(_TEST_DB_PATH) as conn:
        for c in clusters:
            ca = database.get_cluster_analysis(c["id"], conn)
            if not ca:
                continue
            any_found = True
            with st.expander(f"Cluster {c['id']}: {c['representative_headline']}", expanded=True):
                st.write(f"**Summary:** {ca['summary']}")

                sg = ca.get("shared_ground", [])
                if sg:
                    st.write("**Shared ground (all sides agree):**")
                    for item in sg:
                        st.write(f"- {item}")

                col_l, col_r = st.columns(2)
                with col_l:
                    st.markdown("**Left emphasizes — right omits/downplays:**")
                    items = ca.get("left_not_right", [])
                    for item in items:
                        if isinstance(item, dict):
                            badge = f" _{item.get('coverage', '')}_"
                            st.write(f"- {item.get('claim', '')}{badge}")
                        else:
                            st.write(f"- {item}")
                    if not items:
                        st.caption("_(none)_")

                with col_r:
                    st.markdown("**Right emphasizes — left omits/downplays:**")
                    items = ca.get("right_not_left", [])
                    for item in items:
                        if isinstance(item, dict):
                            badge = f" _{item.get('coverage', '')}_"
                            st.write(f"- {item.get('claim', '')}{badge}")
                        else:
                            st.write(f"- {item}")
                    if not items:
                        st.caption("_(none)_")

                if ca.get("center_angle"):
                    st.write(f"**Center angle:** {ca['center_angle']}")

    if not any_found:
        st.info("No cluster analyses found. Run Stage 3 first.")


def _render_ask(result: dict):
    if not result:
        st.info("No ask-agent result yet.")
        return
    if result.get("error"):
        st.error(f"Error: {result['error']}")
        return

    st.write(f"**Query:** {TEST_QUERIES['ask']}")
    st.caption(f"Tools called: {', '.join(result.get('tools_called', []))}")

    structured = result.get("structured", {})
    if structured:
        st.subheader("Topic Overview")
        st.write(structured.get("topic_overview", ""))

        shared = structured.get("shared_ground", [])
        if shared:
            st.subheader("Shared Ground")
            for item in shared:
                st.write(f"- {item}")

        col_l, col_r = st.columns(2)
        with col_l:
            st.subheader("Left Emphasis")
            for item in structured.get("left_emphasis", []):
                st.write(f"- {item}")
        with col_r:
            st.subheader("Right Emphasis")
            for item in structured.get("right_emphasis", []):
                st.write(f"- {item}")

        if structured.get("center_angle"):
            st.subheader("Center Angle")
            st.write(structured["center_angle"])

    matched = result.get("matched_clusters", [])
    if matched:
        st.subheader("Matched Clusters")
        for mc in matched:
            sim = mc.get("similarity", 0)
            st.write(f"- **{mc['headline']}** (similarity: {sim:.2f})")


def _render_fact_check(result: dict):
    if not result:
        st.info("No fact-check result yet.")
        return
    if result.get("error"):
        st.error(f"Error: {result['error']}")
        return

    st.write(f"**Claim:** {TEST_QUERIES['fact_check']}")
    st.caption(f"Tools called: {', '.join(result.get('tools_called', []))}")

    verdict = result.get("verdict", "unknown")
    confidence = result.get("confidence", 0)
    icons = {"confirmed": "🟢", "disputed": "🟡", "misleading": "🟠", "unverifiable": "🔵"}
    st.subheader(f"Verdict: {icons.get(verdict, '⚪')} {verdict.upper()} ({confidence:.0%} confidence)")
    st.progress(confidence)

    structured = result.get("structured_verdict", {})
    if structured:
        st.write(f"**{structured.get('one_line_explanation', '')}**")

        col_l, col_r = st.columns(2)
        with col_l:
            st.write("**Evidence for:**")
            for e in structured.get("evidence_for", []):
                st.write(f"- {e}")
        with col_r:
            st.write("**Evidence against:**")
            for e in structured.get("evidence_against", []):
                st.write(f"- {e}")

        lean_emp = structured.get("lean_emphasis", {})
        if lean_emp:
            st.write("**Coverage by lean:**")
            for lean, text in lean_emp.items():
                if text:
                    st.write(f"- **{lean.title()}:** {text}")

        if structured.get("database_coverage_note"):
            st.caption(f"DB coverage: {structured['database_coverage_note']}")


# ── Main UI ────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="MediaLens Pipeline Tester", layout="wide", page_icon="🔬")
st.title("🔬 MediaLens Pipeline Tester")
st.caption(
    f"Test DB: `{_TEST_DB_PATH.relative_to(_PROJECT_ROOT)}`  ·  "
    f"3 clusters × 3 leans = {len(SEED_ARTICLES)} articles  ·  "
    f"No RSS fetching — direct DB seed"
)

# ── Controls ───────────────────────────────────────────────────────────────────
col_reset, col_run, col_info = st.columns([1, 1, 3])
reset_clicked = col_reset.button("🗑️ Reset Test DB", type="secondary")
run_clicked = col_run.button("▶ Run Full Pipeline", type="primary")

if _TEST_DB_PATH.exists():
    col_info.caption(f"DB exists ({_TEST_DB_PATH.stat().st_size // 1024} KB)")
else:
    col_info.caption("DB does not exist — will be created on first run")

st.divider()

# ── Two-column layout: stages (left) + live logs (right) ──────────────────────
col_stages, col_logs = st.columns([3, 2])

with col_logs:
    st.subheader("Live Logs")
    log_ph = st.empty()
    _render_logs(log_ph)

# ── Reset action ───────────────────────────────────────────────────────────────
if reset_clicked:
    if _TEST_DB_PATH.exists():
        _TEST_DB_PATH.unlink()
    database.init_db(_TEST_DB_PATH)
    st.session_state.log_lines = []
    st.session_state.pop("results", None)
    logger.info("Test database reset and re-initialised.")
    _render_logs(log_ph)
    with col_stages:
        st.success("Test database reset. Ready for a fresh run.")

# ── Pipeline run ───────────────────────────────────────────────────────────────
if run_clicked:
    if not _TEST_DB_PATH.exists():
        database.init_db(_TEST_DB_PATH)

    # Check for stale data from a previous run
    with database.get_db(_TEST_DB_PATH) as _chk:
        existing = _chk.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    if existing > 0:
        with col_stages:
            st.warning(
                f"Test DB already has {existing} articles. "
                "Click **Reset Test DB** first for a clean run."
            )
        st.stop()

    results: dict = {}

    with col_stages:
        with st.status("Stage 1: Seeding articles…", expanded=True) as s1:
            try:
                n = _seed(log_ph)
                results["seed_count"] = n
                s1.update(label=f"Stage 1 ✓ — {n} articles seeded", state="complete")
            except Exception as exc:
                logger.error("Stage 1 failed: {}", exc)
                _render_logs(log_ph)
                s1.update(label=f"Stage 1 ✗ — {exc}", state="error")
                st.stop()

        with st.status("Stage 2: Clustering…", expanded=True) as s2:
            try:
                clusters = _cluster(log_ph)
                results["clusters"] = clusters
                s2.update(label=f"Stage 2 ✓ — {len(clusters)} clusters formed", state="complete")
            except Exception as exc:
                logger.error("Stage 2 failed: {}", exc)
                _render_logs(log_ph)
                s2.update(label=f"Stage 2 ✗ — {exc}", state="error")
                st.stop()

        with st.status("Stage 3: LLM Analysis…", expanded=True) as s3:
            try:
                count, error = _analyze(log_ph)
                results["analysis_count"] = count
                if error:
                    s3.update(label=f"Stage 3 ⚠ — {error}", state="error")
                else:
                    s3.update(label=f"Stage 3 ✓ — {count} clusters analyzed", state="complete")
            except Exception as exc:
                logger.error("Stage 3 failed: {}", exc)
                _render_logs(log_ph)
                s3.update(label=f"Stage 3 ✗ — {exc}", state="error")

        with st.status("Stage 4: Ask Agent…", expanded=True) as s4:
            try:
                ask_result = _ask(log_ph)
                results["ask"] = ask_result
                if ask_result.get("error"):
                    s4.update(label="Stage 4 ⚠ — ask agent error", state="error")
                else:
                    s4.update(label="Stage 4 ✓ — ask agent complete", state="complete")
            except Exception as exc:
                logger.error("Stage 4 failed: {}", exc)
                _render_logs(log_ph)
                s4.update(label=f"Stage 4 ✗ — {exc}", state="error")

        with st.status("Stage 5: Fact Check…", expanded=True) as s5:
            try:
                fc_result = _fact_check(log_ph)
                results["fact_check"] = fc_result
                if fc_result.get("error"):
                    s5.update(label="Stage 5 ⚠ — fact-check error", state="error")
                else:
                    v = fc_result.get("verdict", "?")
                    conf = fc_result.get("confidence", 0)
                    s5.update(
                        label=f"Stage 5 ✓ — verdict: {v} ({conf:.0%})",
                        state="complete",
                    )
            except Exception as exc:
                logger.error("Stage 5 failed: {}", exc)
                _render_logs(log_ph)
                s5.update(label=f"Stage 5 ✗ — {exc}", state="error")

        st.success("Pipeline run complete.")

    st.session_state["results"] = results

# ── Results section ────────────────────────────────────────────────────────────

if "results" in st.session_state:
    r = st.session_state["results"]
    st.divider()
    st.subheader("Results")
    tab1, tab2, tab3, tab4 = st.tabs(["Clusters", "Analysis", "Ask Agent", "Fact Check"])

    with tab1:
        n_art = r.get("seed_count", "?")
        n_cl = len(r.get("clusters", []))
        st.caption(f"{n_art} articles → {n_cl} clusters")
        _render_clusters(r.get("clusters", []))

    with tab2:
        st.caption(f"{r.get('analysis_count', 0)} clusters analyzed by LLM")
        _render_analysis(r.get("clusters", []))

    with tab3:
        _render_ask(r.get("ask", {}))

    with tab4:
        _render_fact_check(r.get("fact_check", {}))
