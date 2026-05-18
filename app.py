"""
MediaLens – Streamlit UI entry point.

Run with:
    streamlit run app.py
"""
import json
import sys
from pathlib import Path

import streamlit as st

# Make `src/` importable
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agent import ask as agent_ask
from fact_check import check as factcheck
from config import LEAN_COLORS
from database import (
    get_cluster_analysis,
    get_cluster_with_articles_and_analysis,
    get_db,
    get_ready_clusters,
    init_db,
    search_clusters,
)

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediaLens",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .lean-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        color: white;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 4px;
    }
    .cluster-card {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
        cursor: pointer;
    }
    .bias-bar-container {
        width: 100%;
        height: 16px;
        background: linear-gradient(to right, #3b82f6 0%, #6b7280 50%, #ef4444 100%);
        border-radius: 8px;
        position: relative;
        margin: 8px 0;
    }
    .bias-marker {
        position: absolute;
        top: -4px;
        width: 6px;
        height: 24px;
        background: black;
        border-radius: 3px;
        transform: translateX(-50%);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Initialise DB ──────────────────────────────────────────────────────────
init_db()


# ── Helpers ────────────────────────────────────────────────────────────────

def lean_badge(lean: str) -> str:
    color = LEAN_COLORS.get(lean, "#6b7280")
    return f'<span class="lean-badge" style="background:{color}">{lean.capitalize()}</span>'



def lean_distribution_bar(lean_coverage: dict) -> str:
    total = sum(lean_coverage.values()) or 1
    parts = ""
    for lean in ("left", "center", "right"):
        pct = lean_coverage.get(lean, 0) / total * 100
        color = LEAN_COLORS[lean]
        parts += f'<div style="width:{pct:.1f}%;background:{color};height:100%"></div>'
    return f"""
    <div style="display:flex;width:100%;height:12px;border-radius:6px;overflow:hidden">
        {parts}
    </div>
    <div style="display:flex;justify-content:space-between;font-size:0.7rem;color:#9ca3af;margin-top:2px">
        <span>L:{lean_coverage.get("left",0)}</span>
        <span>C:{lean_coverage.get("center",0)}</span>
        <span>R:{lean_coverage.get("right",0)}</span>
    </div>
    """


def _render_agent_response(result: dict) -> None:
    if result.get("error"):
        st.error(result["error"])
        return

    run_trace = result.get("run_trace")
    if run_trace and run_trace.get("steps"):
        steps = run_trace["steps"]
        visible = [s for s in steps if not s["tool_name"].startswith("__")]
        total_p = run_trace.get("total_prompt_tokens", 0)
        total_c = run_trace.get("total_completion_tokens", 0)
        token_label = f", {total_p + total_c:,} tokens" if (total_p + total_c) else ""
        with st.expander(f"Execution trace ({len(visible)} steps{token_label})"):
            for step in visible:
                p = step.get("prompt_tokens") or 0
                c = step.get("completion_tokens") or 0
                summary = step["result_summary"][:80]
                st.markdown(f"- `{step['tool_name']}` — {summary}…  *{p}p / {c}c tokens*")
            if total_p or total_c:
                st.caption(
                    f"Total: {total_p:,} prompt + {total_c:,} completion"
                    f" = {total_p + total_c:,} tokens"
                )
    elif result.get("tools_called"):
        with st.expander(f"Tools used ({len(result['tools_called'])})"):
            for tool in result["tools_called"]:
                st.markdown(f"- `{tool}`")

    st.markdown(result.get("response", ""))

    matched = result.get("matched_clusters", [])
    if matched:
        st.markdown("---")
        st.markdown("**Related stories — click to dive in**")
        for cl in matched:
            lean_cov = cl.get("lean_coverage", {})
            lean_str = "  ".join(
                f"**{k.capitalize()}** {v}" for k, v in lean_cov.items() if v
            )
            sim = cl.get("similarity")
            sim_str = f"  ·  sim {sim:.2f}" if sim is not None else ""
            label = cl["headline"]
            with st.container():
                col_info, col_btn = st.columns([5, 1])
                with col_info:
                    st.markdown(f"**{label[:90]}{'…' if len(label) > 90 else ''}**")
                    st.caption(f"{lean_str}{sim_str}")
                with col_btn:
                    if st.button("View →", key=f"agent_view_{cl['cluster_id']}"):
                        st.query_params["cluster_id"] = cl["cluster_id"]
                        st.rerun()


# ── Verdict helpers (defined here so Stories tab can call _render_factcheck) ──

_VERDICT_COLORS = {
    "confirmed":    ("#4ade80", "#052e16"),
    "disputed":     ("#fbbf24", "#1c1200"),
    "misleading":   ("#fb923c", "#1c0a00"),
    "unverifiable": ("#94a3b8", "#1e293b"),
}


def _verdict_badge(verdict: str | None) -> str:
    label = (verdict or "unknown").upper()
    color, bg = _VERDICT_COLORS.get(verdict or "", ("#94a3b8", "#1e293b"))
    return (
        f'<span style="background:{bg};color:{color};border:1px solid {color};'
        f'padding:4px 14px;border-radius:12px;font-size:0.8rem;font-weight:700;'
        f'letter-spacing:0.08em">{label}</span>'
    )


def _render_factcheck(result: dict) -> None:
    if result.get("error"):
        st.error(result["error"])
        return

    confidence = result.get("confidence")
    st.markdown(_verdict_badge(result.get("verdict")), unsafe_allow_html=True)
    if confidence is not None:
        st.progress(float(confidence), text=f"Confidence: {confidence:.0%}")
    st.write("")

    run_trace = result.get("run_trace")
    if run_trace and run_trace.get("steps"):
        visible = [s for s in run_trace["steps"] if not s["tool_name"].startswith("__")]
        with st.expander(f"Evidence steps ({len(visible)})"):
            for step in visible:
                summary = step["result_summary"][:80]
                st.markdown(f"- `{step['tool_name']}`: {summary}…")
    elif result.get("tools_called"):
        with st.expander(f"Tools used ({len(result['tools_called'])})"):
            for t in result["tools_called"]:
                st.markdown(f"- `{t}`")

    st.markdown(result.get("response", ""))


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("MediaLens")
    st.caption("News bias at a glance")
    st.divider()
    search_query = st.text_input("Search stories", placeholder="e.g. climate, economy…")
    st.divider()
    st.markdown("**Quick actions**")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Ingest now", use_container_width=True):
            with st.spinner("Fetching feeds…"):
                from ingestion import fetch_all_feeds
                new = fetch_all_feeds()
            st.success(f"{len(new)} new articles stored")
    with col2:
        if st.button("Cluster", use_container_width=True):
            with st.spinner("Clustering…"):
                from clustering import cluster_new_articles
                n = cluster_new_articles()
            st.success(f"{n} articles clustered")

    if st.button("Analyze clusters", use_container_width=True):
        with st.spinner("Analyzing bias…"):
            from analysis import run_analysis
            n, err = run_analysis()
        if err:
            st.error(f"Analysis error: {err}")
        if n:
            st.success(f"{n} articles analyzed")
        elif not err:
            st.info("No articles pending analysis")

    st.divider()
    st.caption("Only clusters with ≥2 leans are analyzed and shown.")

# ── Main area ──────────────────────────────────────────────────────────────
if "selected_cluster_id" not in st.session_state:
    st.session_state.selected_cluster_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "factcheck_history" not in st.session_state:
    st.session_state.factcheck_history = []  # list of {claim, result}

tab_stories, tab_ask, tab_factcheck = st.tabs(["Stories", "Ask", "Fact Check"])

# ── Stories tab ────────────────────────────────────────────────────────────
with tab_stories:
    # Back button
    if st.session_state.selected_cluster_id is not None:
        if st.button("← Back to story list"):
            st.session_state.selected_cluster_id = None
            st.rerun()

    # Story detail view
    if st.session_state.selected_cluster_id is not None:
        cluster_id = st.session_state.selected_cluster_id

        with get_db() as conn:
            data = get_cluster_with_articles_and_analysis(cluster_id, conn)

        if not data:
            st.error("Cluster not found.")
            st.stop()

        st.subheader(data["representative_headline"])

        lean_cov = data["lean_coverage"]
        badges = "".join(
            lean_badge(ln) for ln in ("left", "center", "right") if lean_cov.get(ln, 0) > 0
        )
        st.markdown(f"**Coverage:** {badges}", unsafe_allow_html=True)
        st.markdown(lean_distribution_bar(lean_cov), unsafe_allow_html=True)
        st.divider()

        articles = data.get("articles", [])
        if not articles:
            st.info("No articles in this cluster yet.")
            st.stop()

        # Cluster-level analysis
        with get_db() as conn:
            ca = get_cluster_analysis(cluster_id, conn)

        if ca:
            st.markdown("**Summary**")
            st.markdown(ca.get("summary", ""))

            shared = ca.get("shared_ground", [])
            if shared:
                st.markdown("**What all sides report**")
                for point in shared:
                    st.markdown(f"- {point}")

            lnr = ca.get("left_not_right", [])
            rnl = ca.get("right_not_left", [])
            if lnr or rnl:
                col_left, col_right = st.columns(2)

                def _coverage_badge(coverage: str) -> str:
                    if coverage == "omitted":
                        return '<span style="background:#ef4444;color:#fff;padding:1px 6px;border-radius:4px;font-size:0.72em">omitted</span>'
                    return '<span style="background:#f59e0b;color:#fff;padding:1px 6px;border-radius:4px;font-size:0.72em">downplayed</span>'

                with col_left:
                    left_color = LEAN_COLORS["left"]
                    st.markdown(
                        f'<span style="color:{left_color};font-weight:700">Left says — Right doesn\'t</span>',
                        unsafe_allow_html=True,
                    )
                    if lnr:
                        for item in lnr:
                            badge = _coverage_badge(item.get("coverage", "downplayed"))
                            st.markdown(
                                f'{badge} {item["claim"]}',
                                unsafe_allow_html=True,
                            )
                    else:
                        st.caption("No significant asymmetry found.")

                with col_right:
                    right_color = LEAN_COLORS["right"]
                    st.markdown(
                        f'<span style="color:{right_color};font-weight:700">Right says — Left doesn\'t</span>',
                        unsafe_allow_html=True,
                    )
                    if rnl:
                        for item in rnl:
                            badge = _coverage_badge(item.get("coverage", "downplayed"))
                            st.markdown(
                                f'{badge} {item["claim"]}',
                                unsafe_allow_html=True,
                            )
                    else:
                        st.caption("No significant asymmetry found.")

            center_angle = ca.get("center_angle", "")
            if center_angle:
                center_color = LEAN_COLORS["center"]
                st.markdown(
                    f'<span style="color:{center_color};font-weight:700">Center angle</span>',
                    unsafe_allow_html=True,
                )
                st.markdown(center_angle)
        else:
            st.caption("Cluster analysis pending — click Analyze in the sidebar.")

        st.divider()

        # Inline fact-check
        st.markdown("**Fact-check a claim**")
        fc_key = f"fc_{cluster_id}"
        claim_text = st.text_input(
            "Enter a claim from this story",
            key=f"fc_input_{cluster_id}",
            placeholder="e.g. 'Trump says Europe doesn’t pay for defence'",
        )
        if st.button("Check this claim", key=f"fc_btn_{cluster_id}"):
            if claim_text.strip():
                with st.spinner("Fact-checking…"):
                    fc_result = factcheck(claim_text.strip(), context=data["representative_headline"])
                st.session_state[fc_key] = fc_result
            else:
                st.warning("Please enter a claim to check.")
        if fc_key in st.session_state:
            _render_factcheck(st.session_state[fc_key])

        st.divider()

        # Article list grouped by lean
        order = {"left": 0, "center": 1, "right": 2}
        articles = sorted(articles, key=lambda a: order.get(a["source_lean"], 9))

        cols = st.columns(max(len(articles), 1))
        for col, art in zip(cols, articles):
            with col:
                badge_html = lean_badge(art["source_lean"])
                st.markdown(
                    f"{badge_html} **{art['source_name']}**", unsafe_allow_html=True
                )
                st.markdown(f"**{art['title']}**")
                pub = art.get("published_at", "")
                if pub:
                    st.caption(f"Published: {pub[:10]}")
                st.markdown(f"[Read full article]({art['url']})")

    # Story list view
    else:
        st.header("Stories")

        with get_db() as conn:
            if search_query and search_query.strip():
                clusters = search_clusters(search_query.strip(), conn)
            else:
                clusters = get_ready_clusters(conn)

        if not clusters:
            st.info(
                "No stories found. "
                "Use the sidebar buttons to ingest feeds and cluster articles."
            )
        else:
            st.caption(f"{len(clusters)} cluster(s) found")

            for cluster in clusters:
                lean_cov = json.loads(cluster["lean_coverage"])
                badges_html = "".join(
                    lean_badge(ln)
                    for ln in ("left", "center", "right")
                    if lean_cov.get(ln, 0) > 0
                )
                total_articles = sum(lean_cov.values())

                with st.container():
                    col_text, col_btn = st.columns([5, 1])
                    with col_text:
                        st.markdown(
                            f"**{cluster['representative_headline']}**  \n"
                            f"{badges_html} &nbsp; "
                            f"<span style='color:#9ca3af;font-size:0.8rem'>"
                            f"{total_articles} article(s)</span>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(lean_distribution_bar(lean_cov), unsafe_allow_html=True)
                    with col_btn:
                        if st.button("View", key=f"view_{cluster['id']}"):
                            st.session_state.selected_cluster_id = cluster["id"]
                            st.rerun()
                    st.divider()


# ── Ask tab ────────────────────────────────────────────────────────────────
with tab_ask:
    st.header("Ask about a topic")
    st.caption(
        "The agent searches stories already in the database and synthesizes "
        "what left, center, and right sources agree and disagree on."
    )

    # Render chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.write(message["content"])
            else:
                result = message["result"]
                _render_agent_response(result)

    # Chat input
    user_query = st.chat_input("Ask about a topic, e.g. 'Ukraine aid' or 'inflation'")
    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query, "result": None})
        with st.chat_message("user"):
            st.write(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Searching and synthesizing…"):
                result = agent_ask(user_query)
            _render_agent_response(result)

        st.session_state.chat_history.append({"role": "assistant", "content": "", "result": result})
        st.rerun()


# ── Fact Check tab ─────────────────────────────────────────────────────────
with tab_factcheck:
    st.header("Fact Check")
    st.caption(
        "State a specific claim. The agent searches the database for evidence "
        "that supports or contradicts it, then delivers a verdict."
    )

    for entry in st.session_state.factcheck_history:
        st.markdown(f"**Claim:** _{entry['claim']}_")
        _render_factcheck(entry["result"])
        st.divider()

    claim_input = st.chat_input(
        "Enter a claim, e.g. 'The Fed raised rates by 0.5%' or 'Ukraine aid was blocked'"
    )
    if claim_input:
        with st.spinner("Checking claim…"):
            result = factcheck(claim_input)
        st.session_state.factcheck_history.append({"claim": claim_input, "result": result})
        st.rerun()
