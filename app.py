"""
Multi-Agent Research & Report Generation System
Streamlit UI — Premium pastel-themed interface with real-time agent progress.
"""

from __future__ import annotations

import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime

import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration — MUST be the first Streamlit command
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Imports (after page config)
# ---------------------------------------------------------------------------
from dotenv import load_dotenv

load_dotenv()

from config import GROQ_API_KEY, TAVILY_API_KEY
from graph import build_graph
from utils.pdf_generator import generate_pdf

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom CSS — Premium dark theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* ── Import Google Font ───────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

    /* ── Animated background gradient keyframes ──────────────────── */
    @keyframes gradientShift {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes floatBlob {
        0%, 100% { transform: translateY(0px) scale(1); }
        50%      { transform: translateY(-20px) scale(1.05); }
    }
    @keyframes shimmer {
        0%   { background-position: -200% center; }
        100% { background-position: 200% center; }
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* ── Global overrides ─────────────────────────────────────────── */
    .stApp {
        font-family: 'Outfit', sans-serif;
        background: linear-gradient(-45deg, #fdf2f8, #ede9fe, #e0f2fe, #fef3c7, #fce7f3);
        background-size: 400% 400%;
        animation: gradientShift 18s ease infinite;
        color: #3b2e4a;
    }

    /* ── Streamlit native element overrides for pastel theme ──────── */
    .stApp [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.55) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(196, 181, 253, 0.3) !important;
    }
    .stApp [data-testid="stSidebar"] * {
        color: #4a3b5c !important;
    }

    /* ── Hero header ──────────────────────────────────────────────── */
    .hero-container {
        background: rgba(255, 255, 255, 0.5);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2.5rem 2rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(196, 181, 253, 0.35);
        box-shadow: 0 8px 32px rgba(168, 148, 214, 0.12),
                    inset 0 1px 0 rgba(255, 255, 255, 0.6);
        position: relative;
        overflow: hidden;
        animation: fadeInUp 0.6s ease-out;
    }
    .hero-container::before {
        content: '';
        position: absolute;
        top: -30%;
        right: -20%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(249, 168, 212, 0.25) 0%, transparent 70%);
        border-radius: 50%;
        animation: floatBlob 6s ease-in-out infinite;
        pointer-events: none;
    }
    .hero-container::after {
        content: '';
        position: absolute;
        bottom: -20%;
        left: -10%;
        width: 250px;
        height: 250px;
        background: radial-gradient(circle, rgba(165, 180, 252, 0.25) 0%, transparent 70%);
        border-radius: 50%;
        animation: floatBlob 8s ease-in-out infinite reverse;
        pointer-events: none;
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #7c3aed 0%, #db2777 40%, #f59e0b 100%);
        background-size: 200% auto;
        animation: shimmer 4s linear infinite;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
        position: relative;
        z-index: 1;
    }
    .hero-subtitle {
        color: #6b5b7b;
        font-size: 1.05rem;
        font-weight: 400;
        line-height: 1.7;
        position: relative;
        z-index: 1;
    }

    /* ── Agent status cards ───────────────────────────────────────── */
    .agent-card {
        background: rgba(255, 255, 255, 0.55);
        backdrop-filter: blur(14px);
        border: 1px solid rgba(196, 181, 253, 0.25);
        border-radius: 14px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.75rem;
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        animation: fadeInUp 0.5s ease-out;
        box-shadow: 0 2px 12px rgba(168, 148, 214, 0.08);
    }
    .agent-card:hover {
        border-color: rgba(167, 139, 250, 0.5);
        box-shadow: 0 6px 24px rgba(167, 139, 250, 0.15);
        transform: translateY(-2px);
    }
    .agent-card.active {
        border-color: rgba(167, 139, 250, 0.6);
        background: rgba(237, 233, 254, 0.65);
        box-shadow: 0 0 28px rgba(167, 139, 250, 0.18);
    }
    .agent-card.done {
        border-color: rgba(134, 239, 172, 0.5);
        background: rgba(220, 252, 231, 0.45);
    }
    .agent-name {
        font-weight: 600;
        font-size: 0.95rem;
        margin-bottom: 2px;
        color: #3b2e4a;
    }
    .agent-desc {
        color: #8b7fa0;
        font-size: 0.82rem;
    }

    /* ── Status badges ────────────────────────────────────────────── */
    .badge {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-waiting { background: rgba(148, 140, 165, 0.12); color: #8b7fa0; }
    .badge-active  { background: rgba(167, 139, 250, 0.18); color: #7c3aed; }
    .badge-done    { background: rgba(74, 222, 128, 0.18);   color: #16a34a; }
    .badge-error   { background: rgba(251, 113, 133, 0.18);  color: #e11d48; }

    /* ── Metric cards ─────────────────────────────────────────────── */
    .metric-card {
        background: rgba(255, 255, 255, 0.55);
        backdrop-filter: blur(14px);
        border: 1px solid rgba(196, 181, 253, 0.25);
        border-radius: 16px;
        padding: 1.4rem;
        text-align: center;
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 2px 12px rgba(168, 148, 214, 0.08);
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 28px rgba(167, 139, 250, 0.15);
        border-color: rgba(167, 139, 250, 0.4);
    }
    .metric-value {
        font-size: 1.9rem;
        font-weight: 700;
        background: linear-gradient(135deg, #7c3aed, #db2777);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .metric-label {
        color: #8b7fa0;
        font-size: 0.82rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 4px;
    }

    /* ── Sidebar styling ──────────────────────────────────────────── */
    .sidebar-section {
        background: rgba(255, 255, 255, 0.5);
        border: 1px solid rgba(196, 181, 253, 0.2);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(8px);
    }
    .sidebar-title {
        font-weight: 700;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #7c6f94 !important;
        margin-bottom: 0.75rem;
    }

    /* ── Key indicator dots ───────────────────────────────────────── */
    .key-status {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 4px 0;
        font-size: 0.88rem;
    }
    .dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
    }
    .dot-green { background: #4ade80; box-shadow: 0 0 8px rgba(74, 222, 128, 0.5); }
    .dot-red   { background: #fb7185; box-shadow: 0 0 8px rgba(251, 113, 133, 0.5); }

    /* ── Hide default Streamlit branding ──────────────────────────── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ── Custom button ────────────────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #a78bfa 0%, #f472b6 50%, #fb923c 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.65rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        font-family: 'Outfit', sans-serif;
        letter-spacing: 0.3px;
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 18px rgba(167, 139, 250, 0.3);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(167, 139, 250, 0.45);
        filter: brightness(1.05);
    }

    /* ── Text input styling ───────────────────────────────────────── */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.6) !important;
        border: 1px solid rgba(196, 181, 253, 0.35) !important;
        border-radius: 12px !important;
        color: #3b2e4a !important;
        font-family: 'Outfit', sans-serif !important;
        backdrop-filter: blur(10px);
    }
    .stTextInput > div > div > input:focus {
        border-color: rgba(167, 139, 250, 0.6) !important;
        box-shadow: 0 0 0 3px rgba(167, 139, 250, 0.15) !important;
    }
    .stTextInput > div > div > input::placeholder {
        color: #a89bbd !important;
    }

    /* ── Expander styling ─────────────────────────────────────────── */
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 0.95rem;
        color: #3b2e4a !important;
    }
    [data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.45) !important;
        border: 1px solid rgba(196, 181, 253, 0.25) !important;
        border-radius: 14px !important;
        backdrop-filter: blur(12px);
    }

    /* ── Progress bar override ────────────────────────────────────── */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #a78bfa, #f472b6, #fb923c) !important;
        border-radius: 10px;
    }

    /* ── Download button styling ──────────────────────────────────── */
    .stDownloadButton > button {
        background: rgba(255, 255, 255, 0.6) !important;
        border: 1px solid rgba(196, 181, 253, 0.35) !important;
        color: #5b4a72 !important;
        border-radius: 12px !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        backdrop-filter: blur(10px);
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .stDownloadButton > button:hover {
        background: rgba(237, 233, 254, 0.7) !important;
        border-color: rgba(167, 139, 250, 0.5) !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 16px rgba(167, 139, 250, 0.15);
    }

    /* ── Markdown text color ──────────────────────────────────────── */
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #3b2e4a !important;
    }

    /* ── Divider ──────────────────────────────────────────────────── */
    hr {
        border-color: rgba(196, 181, 253, 0.25) !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Agent definitions (for UI display)
# ---------------------------------------------------------------------------
AGENTS = [
    {
        "key": "planner",
        "name": "🧠 Planner Agent",
        "desc": "Decomposes topic into focused subtopics",
        "color": "#a78bfa",  # pastel violet
    },
    {
        "key": "search",
        "name": "🔍 Search Agent",
        "desc": "Searches Tavily & ArXiv for sources",
        "color": "#f9a8d4",  # pastel pink
    },
    {
        "key": "analyst",
        "name": "📊 Analyst Agent",
        "desc": "Extracts structured insights per source",
        "color": "#93c5fd",  # pastel blue
    },
    {
        "key": "writer",
        "name": "✍️ Writer Agent",
        "desc": "Generates the structured research report",
        "color": "#fdba74",  # pastel orange
    },
    {
        "key": "critic",
        "name": "🔎 Critic Agent",
        "desc": "Evaluates quality & requests revisions",
        "color": "#86efac",  # pastel green
    },
]

STATUS_MAP = {
    "planning_complete": 1,
    "search_complete": 2,
    "analysis_complete": 3,
    "writing_complete": 4,
    "revision_requested": 4,  # Writer→Critic loop
    "complete": 5,
}

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "run_history" not in st.session_state:
    st.session_state.run_history = []

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <div style="font-size: 2.5rem;">🔬</div>
        <div style="font-weight: 700; font-size: 1.1rem; margin-top: 4px; color: #4a3b5c;">
            AI Research Agent
        </div>
        <div style="color: #8b7fa0; font-size: 0.82rem;">
            Multi-Agent System v1.0
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # API Key status
    st.markdown('<div class="sidebar-title">🔑 API Status</div>', unsafe_allow_html=True)

    groq_ok = bool(GROQ_API_KEY and GROQ_API_KEY != "your_groq_api_key_here")
    tavily_ok = bool(TAVILY_API_KEY and TAVILY_API_KEY != "your_tavily_api_key_here")

    dot_g = "dot-green" if groq_ok else "dot-red"
    dot_t = "dot-green" if tavily_ok else "dot-red"

    st.markdown(f"""
    <div class="sidebar-section">
        <div class="key-status">
            <span class="dot {dot_g}"></span>
            <span>Groq API {'Connected' if groq_ok else 'Not configured'}</span>
        </div>
        <div class="key-status">
            <span class="dot {dot_t}"></span>
            <span>Tavily API {'Connected' if tavily_ok else 'Not configured'}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not groq_ok or not tavily_ok:
        st.warning("⚠️ Add your API keys to `.env` before running.")

    st.markdown("---")

    # Architecture info
    st.markdown('<div class="sidebar-title">🏗️ Architecture</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-section">
        <div style="font-size: 0.85rem; line-height: 1.7; color: #6b5b7b;">
            <b>Framework:</b> LangGraph<br>
            <b>LLM:</b> Llama 3.3 70B (Groq)<br>
            <b>Web Search:</b> Tavily API<br>
            <b>Papers:</b> ArXiv API<br>
            <b>PDF Engine:</b> ReportLab<br>
            <b>Max Revisions:</b> 2 loops
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Run history
    if st.session_state.run_history:
        st.markdown('<div class="sidebar-title">📜 History</div>', unsafe_allow_html=True)
        for entry in reversed(st.session_state.run_history[-5:]):
            st.markdown(f"""
            <div class="sidebar-section" style="padding: 0.7rem;">
                <div style="font-size: 0.82rem; font-weight: 600; color: #4a3b5c;">{entry['topic'][:40]}...</div>
                <div style="font-size: 0.72rem; color: #8b7fa0;">{entry['time']}</div>
            </div>
            """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

# Hero header
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🔬 Multi-Agent Research System</div>
    <div class="hero-subtitle">
        Enter any research topic below. A team of 5 autonomous AI agents will search the web
        and ArXiv, analyze findings, write a comprehensive report, and critique it for quality
        — all without any further input from you.
    </div>
</div>
""", unsafe_allow_html=True)

# Topic input
col_input, col_btn = st.columns([4, 1])
with col_input:
    topic = st.text_input(
        "Research Topic",
        placeholder="e.g., Impact of Large Language Models on healthcare diagnostics",
        label_visibility="collapsed",
    )
with col_btn:
    run_clicked = st.button("🚀 Generate", use_container_width=True, disabled=st.session_state.is_running)

# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------
if run_clicked and topic:
    if not groq_ok:
        st.error("❌ Please configure your Groq API key in `.env`")
        st.stop()

    st.session_state.is_running = True
    st.session_state.pipeline_result = None

    # Progress tracking containers
    progress_bar = st.progress(0, text="Initializing pipeline...")
    status_container = st.container()

    agent_outputs: dict = {}
    current_step = 0
    total_steps = 5

    def update_progress(step: int, agent_name: str, detail: str = ""):
        progress = step / total_steps
        progress_bar.progress(progress, text=f"{agent_name}: {detail}")

    try:
        # Build the graph
        app = build_graph()

        # Stream through the graph to track progress
        initial_state = {
            "topic": topic,
            "subtopics": [],
            "search_results": [],
            "analyzed_sources": [],
            "draft_report": "",
            "critique": {},
            "final_report": "",
            "revision_count": 0,
            "status": "starting",
            "errors": [],
        }

        with status_container:
            st.markdown("### 🔄 Pipeline Progress")

            # Create placeholder columns for agent status
            agent_placeholders = {}
            for agent in AGENTS:
                agent_placeholders[agent["key"]] = st.empty()

            # Render initial state
            for agent in AGENTS:
                with agent_placeholders[agent["key"]]:
                    st.markdown(f"""
                    <div class="agent-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <div class="agent-name">{agent['name']}</div>
                                <div class="agent-desc">{agent['desc']}</div>
                            </div>
                            <span class="badge badge-waiting">Waiting</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        # Execute the graph
        final_state = None
        for step_output in app.stream(initial_state, stream_mode="updates"):
            for node_name, node_output in step_output.items():
                agent_outputs[node_name] = node_output
                status = node_output.get("status", "")

                # Update agent cards
                step_idx = STATUS_MAP.get(status, 0)

                with status_container:
                    for agent in AGENTS:
                        key = agent["key"]
                        with agent_placeholders[key]:
                            if key in agent_outputs:
                                badge_class = "badge-done"
                                badge_text = "Done"
                                card_class = "done"
                            elif key == node_name and status not in ("complete",):
                                badge_class = "badge-active"
                                badge_text = "Running"
                                card_class = "active"
                            else:
                                badge_class = "badge-waiting"
                                badge_text = "Waiting"
                                card_class = ""

                            st.markdown(f"""
                            <div class="agent-card {card_class}">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <div>
                                        <div class="agent-name">{agent['name']}</div>
                                        <div class="agent-desc">{agent['desc']}</div>
                                    </div>
                                    <span class="badge {badge_class}">{badge_text}</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                # Update progress bar
                progress_val = min(step_idx / total_steps, 1.0)
                agent_display = next(
                    (a["name"] for a in AGENTS if a["key"] == node_name),
                    node_name,
                )
                progress_bar.progress(progress_val, text=f"{agent_display} — {status}")

                # Track final state
                if status == "complete":
                    final_state = node_output

        # ── Pipeline finished ───────────────────────────────────────────
        progress_bar.progress(1.0, text="✅ Pipeline complete!")

        # Merge all agent outputs into a result
        merged = {**initial_state}
        for output in agent_outputs.values():
            merged.update(output)

        st.session_state.pipeline_result = merged
        st.session_state.is_running = False

        # Add to history
        st.session_state.run_history.append({
            "topic": topic,
            "time": datetime.now().strftime("%b %d, %H:%M"),
        })

    except Exception as e:
        st.session_state.is_running = False
        progress_bar.progress(0, text="❌ Pipeline failed")
        st.error(f"Pipeline error: {e}")
        logger.exception("Pipeline failed")

# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------
result = st.session_state.pipeline_result

if result:
    st.markdown("---")

    # ── Metrics row ─────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(result.get('subtopics', []))}</div>
            <div class="metric-label">Subtopics</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(result.get('search_results', []))}</div>
            <div class="metric-label">Sources Found</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(result.get('analyzed_sources', []))}</div>
            <div class="metric-label">Analyzed</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        quality = result.get("critique", {}).get("quality_score", "—")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{quality}/10</div>
            <div class="metric-label">Quality Score</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Agent detail expanders ──────────────────────────────────────────
    with st.expander("🧠 Planner Agent — Subtopics", expanded=False):
        subtopics = result.get("subtopics", [])
        for i, st_item in enumerate(subtopics, 1):
            st.markdown(f"**{i}.** {st_item}")

    with st.expander("🔍 Search Agent — Sources Discovered", expanded=False):
        for src in result.get("search_results", [])[:20]:
            type_badge = "🌐" if src.get("source_type") == "web" else "📄"
            st.markdown(f"{type_badge} **{src.get('title', 'Untitled')}**")
            st.caption(f"{src.get('source_url', 'N/A')}")

    with st.expander("📊 Analyst Agent — Source Analyses", expanded=False):
        for analysis in result.get("analyzed_sources", []):
            score = analysis.get("relevance_score", "?")
            st.markdown(f"**{analysis.get('title', 'Untitled')}** — Relevance: `{score}/10`")
            findings = analysis.get("key_findings", [])
            for f in findings:
                st.markdown(f"  - {f}")
            st.markdown(f"  *Methodology:* {analysis.get('methodology', 'N/A')}")
            st.divider()

    with st.expander("🔎 Critic Agent — Quality Assessment", expanded=False):
        critique = result.get("critique", {})
        st.metric("Quality Score", f"{critique.get('quality_score', '—')}/10")
        issues = critique.get("issues_found", [])
        if issues:
            st.markdown("**Issues found:**")
            for issue in issues:
                st.markdown(f"- {issue}")
        else:
            st.success("No issues found!")
        st.markdown(f"**Revisions performed:** {result.get('revision_count', 0)}")

    # ── Final report display ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📄 Final Research Report")

    final_report = result.get("final_report", "") or result.get("draft_report", "")

    if final_report:
        st.markdown(final_report)

        st.markdown("---")

        # ── Download buttons ────────────────────────────────────────────
        dl_col1, dl_col2 = st.columns(2)

        with dl_col1:
            st.download_button(
                label="📥 Download Markdown",
                data=final_report,
                file_name=f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True,
            )

        with dl_col2:
            try:
                pdf_path = generate_pdf(
                    final_report,
                    result.get("topic", "Research Report"),
                )
                with open(pdf_path, "rb") as pdf_file:
                    st.download_button(
                        label="📥 Download PDF",
                        data=pdf_file.read(),
                        file_name=os.path.basename(pdf_path),
                        mime="application/pdf",
                        use_container_width=True,
                    )
            except Exception as e:
                st.error(f"PDF generation failed: {e}")

        # ── Errors (if any) ─────────────────────────────────────────────
        errors = result.get("errors", [])
        if errors:
            with st.expander("⚠️ Warnings & Errors", expanded=False):
                for err in errors:
                    st.warning(err)
    else:
        st.info("No report was generated. Check for errors above.")

# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------
elif not st.session_state.is_running:
    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)
    examples = [
        ("🏥", "Impact of LLMs on healthcare diagnostics"),
        ("🤖", "Recent advances in autonomous robotics"),
        ("🌍", "Climate change mitigation through AI"),
    ]
    for col, (icon, ex_topic) in zip([col_a, col_b, col_c], examples):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="cursor: pointer; min-height: 100px;">
                <div style="font-size: 1.8rem; margin-bottom: 8px;">{icon}</div>
                <div style="font-size: 0.88rem; color: #6b5b7b; font-weight: 500;">
                    {ex_topic}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; padding: 2rem; color: #6b5b7b;">
        <p style="font-size: 0.92rem;">
            Enter a research topic above and click <b>Generate</b> to start the autonomous research pipeline.
        </p>
    </div>
    """, unsafe_allow_html=True)
