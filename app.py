"""
Research Gap Analyzer
An AI-powered tool for identifying research gaps across academic papers.
Architecture: LLM Wiki + PageIndex + LazyGraphRAG + Academic Validation
"""
import streamlit as st
import tempfile
import os
import json
import time
from pathlib import Path

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Research Gap Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — dark editorial theme ────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

  /* ── Base ── */
  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d1117;
    color: #e6edf3;
  }
  .stApp { background-color: #0d1117; }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
  }
  [data-testid="stSidebar"] .stMarkdown h1,
  [data-testid="stSidebar"] .stMarkdown h2,
  [data-testid="stSidebar"] .stMarkdown h3 {
    color: #58a6ff;
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
  }

  /* ── Main header ── */
  .main-header {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    line-height: 1.1;
    background: linear-gradient(135deg, #58a6ff 0%, #79c0ff 50%, #a5d6ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
  }
  .sub-header {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.95rem;
    color: #8b949e;
    font-weight: 300;
    letter-spacing: 0.02em;
    margin-bottom: 2rem;
  }

  /* ── Stage cards ── */
  .stage-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
  }
  .stage-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #58a6ff;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
  }
  .stage-body { font-size: 0.9rem; color: #c9d1d9; }

  /* ── Gap cards ── */
  .gap-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-left: 3px solid #58a6ff;
    border-radius: 6px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
  }
  .gap-card.open { border-left-color: #3fb950; }
  .gap-card.partial { border-left-color: #d29922; }
  .gap-card.solved { border-left-color: #f85149; }
  .gap-card.pending { border-left-color: #8b949e; }
  .gap-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.1rem;
    color: #e6edf3;
    margin-bottom: 0.5rem;
  }
  .gap-meta {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #8b949e;
    margin-bottom: 0.6rem;
  }
  .gap-desc { font-size: 0.88rem; color: #c9d1d9; line-height: 1.6; }

  /* ── Proposal cards ── */
  .proposal-card {
    background: #0d2136;
    border: 1px solid #1f4f7a;
    border-radius: 8px;
    padding: 1.4rem 1.8rem;
    margin-bottom: 1.2rem;
  }
  .proposal-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.25rem;
    color: #79c0ff;
    margin-bottom: 0.8rem;
  }
  .proposal-section {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: #58a6ff;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.8rem;
    margin-bottom: 0.3rem;
  }
  .proposal-body { font-size: 0.88rem; color: #c9d1d9; line-height: 1.6; }

  /* ── Badges ── */
  .badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 12px;
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    font-weight: 500;
    margin-right: 0.4rem;
  }
  .badge-open { background: #0f3d1a; color: #3fb950; border: 1px solid #238636; }
  .badge-partial { background: #2d1f00; color: #d29922; border: 1px solid #9e6a03; }
  .badge-solved { background: #3d0f0f; color: #f85149; border: 1px solid #da3633; }
  .badge-high { background: #0f2d3d; color: #79c0ff; border: 1px solid #1f6feb; }
  .badge-medium { background: #2d2500; color: #e3b341; border: 1px solid #9e6a03; }
  .badge-low { background: #1c1c1c; color: #8b949e; border: 1px solid #30363d; }

  /* ── Wiki card ── */
  .wiki-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.8rem;
  }
  .wiki-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.05rem;
    color: #e6edf3;
    margin-bottom: 0.6rem;
  }
  .wiki-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: #58a6ff;
    text-transform: uppercase;
    letter-spacing: 0.1em;
  }
  .wiki-item { font-size: 0.83rem; color: #c9d1d9; margin-left: 0.8rem; }

  /* ── Graph stats ── */
  .stat-box {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 1rem;
    text-align: center;
  }
  .stat-number {
    font-family: 'DM Mono', monospace;
    font-size: 2rem;
    color: #58a6ff;
    display: block;
  }
  .stat-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.75rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }

  /* ── Buttons ── */
  .stButton > button {
    background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.05em !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s !important;
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, #388bfd, #58a6ff) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(31, 111, 235, 0.3) !important;
  }

  /* ── Inputs ── */
  .stTextInput > div > div > input,
  .stTextArea > div > div > textarea,
  .stSelectbox > div > div {
    background-color: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.85rem !important;
    border-radius: 6px !important;
  }

  /* ── File uploader ── */
  [data-testid="stFileUploader"] {
    background: #161b22;
    border: 2px dashed #30363d;
    border-radius: 8px;
    padding: 1rem;
  }
  [data-testid="stFileUploader"]:hover { border-color: #58a6ff; }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: #161b22;
    border-radius: 8px 8px 0 0;
    border-bottom: 1px solid #30363d;
    padding: 0 0.5rem;
  }
  .stTabs [data-baseweb="tab"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.05em !important;
    color: #8b949e !important;
    padding: 0.7rem 1.2rem !important;
    text-transform: uppercase !important;
  }
  .stTabs [aria-selected="true"] {
    color: #58a6ff !important;
    border-bottom: 2px solid #58a6ff !important;
  }
  .stTabs [data-baseweb="tab-panel"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-top: none;
    border-radius: 0 0 8px 8px;
    padding: 1.5rem;
  }

  /* ── Expanders ── */
  .streamlit-expanderHeader {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
    color: #c9d1d9 !important;
  }
  .streamlit-expanderContent {
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    border-top: none !important;
  }

  /* ── Progress ── */
  .stProgress > div > div { background: linear-gradient(90deg, #1f6feb, #58a6ff) !important; }

  /* ── Divider ── */
  hr { border-color: #30363d !important; }

  /* ── Code ── */
  code { font-family: 'DM Mono', monospace !important; color: #79c0ff !important; }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: #0d1117; }
  ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: #58a6ff; }
</style>
""", unsafe_allow_html=True)

# ── Imports ──────────────────────────────────────────────────────────────────
import streamlit as st
import tempfile, os, json, time, traceback

from utils.azure_client import AzureOpenAIClient
from utils.export import to_json, to_markdown_report
from pipeline.pdf_parser import parse_pdf
from pipeline.page_index import build_tree, tree_to_summary
from pipeline.wiki_compiler import build_wiki
from pipeline.graph_builder import build_knowledge_graph
from pipeline.gap_detector import detect_gaps
from pipeline.academic_search import validate_gaps
from pipeline.proposal_generator import generate_proposals
from pipeline.param_extractor import (
    extract_parameters, derive_cross_paper_parameters,
    check_sufficiency, params_to_csv, group_params_by_category,
)
from pipeline.discovery_engine import run_discovery_engine
from pipeline.llm_council import council_select_model, council_synthesise_insights
from pipeline.simulation_engine import run_simulation, auto_select_models
from pipeline.multi_run import run_multi_council
from pipeline.doc_generator import build_zip
from domain_config import (
    get_domain_display_options, parse_domain_selection, get_domain_config,
)

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ("results", None), ("running", False), ("comp_results", None),
    ("pipeline_trace", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Helpers ───────────────────────────────────────────────────────────────────
def _badge(text, color="#58a6ff", bg="#0f2d3d", border="#1f6feb"):
    return (f'<span style="display:inline-block;padding:0.15rem 0.5rem;border-radius:10px;'
            f'font-family:\'DM Mono\',monospace;font-size:0.65rem;font-weight:500;'
            f'color:{color};background:{bg};border:1px solid {border};margin-right:0.3rem;">'
            f'{text}</span>')

def _card(title, body, color="#58a6ff"):
    return (f'<div class="stage-card"><div class="stage-title" style="color:{color};">{title}</div>'
            f'<div class="stage-body">{body}</div></div>')

def _section_label(text):
    st.markdown(f'<div class="wiki-label">{text}</div>', unsafe_allow_html=True)

def _item(text, color="#c9d1d9"):
    st.markdown(f'<div class="wiki-item" style="color:{color};margin-bottom:0.25rem;">{text}</div>',
                unsafe_allow_html=True)

def _add_trace(stage, inputs_summary, outputs_summary, status="✅"):
    st.session_state.pipeline_trace.append({
        "stage": stage, "inputs": inputs_summary,
        "outputs": outputs_summary, "status": status,
    })

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔬 Research Gap Analyzer")
    st.markdown("---")

    st.markdown("##### API Configuration")
    st.markdown("""
    <div style="background:#0d2136;border:1px solid #1f4f7a;border-radius:6px;
                padding:0.8rem 1rem;margin-bottom:0.8rem;">
      <div style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#58a6ff;
                  text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem;">Model Stack</div>
      <div style="font-size:0.78rem;color:#e6edf3;font-weight:500;">🧠 Master — gpt-oss-120b</div>
      <div style="font-family:'DM Mono',monospace;font-size:0.62rem;color:#484f58;line-height:1.9;">
        Council A1 · gpt-oss-20b<br>
        Council A2 · llama-3.2-3b-instruct<br>
        Council A3 · nemotron-nano-omni-30b<br>
        Chair · gpt-oss-120b · via NVIDIA NIM
      </div>
    </div>""", unsafe_allow_html=True)

    api_key = st.text_input("NVIDIA API Key", type="password",
                             placeholder="nvapi-••••••••••••••••")
    st.markdown('<div style="font-family:\'DM Mono\',monospace;font-size:0.62rem;color:#484f58;">'
                '🔗 <a href="https://build.nvidia.com" target="_blank" style="color:#58a6ff;">'
                'Get key → build.nvidia.com</a></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### Research Domain")
    domain_options  = get_domain_display_options()
    default_idx     = next((i for i, d in enumerate(domain_options) if "Healthcare" in d), 0)
    selected_display = st.selectbox("Select Domain", domain_options, index=default_idx)
    selected_domain  = parse_domain_selection(selected_display)
    domain_cfg       = get_domain_config(selected_domain)

    st.markdown(f"""
    <div style="background:#0d2136;border:1px solid #1f4f7a;border-radius:6px;
                padding:0.7rem 0.9rem;margin-top:0.3rem;">
      <div style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#58a6ff;
                  text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.3rem;">Active</div>
      <div style="font-size:0.78rem;color:#c9d1d9;">{domain_cfg['description']}</div>
      <div style="font-family:'DM Mono',monospace;font-size:0.62rem;color:#484f58;margin-top:0.4rem;line-height:1.7;">
        ✦ Wiki prompts tuned &nbsp; ✦ Graph entity types tuned<br>
        ✦ Gap priorities tuned &nbsp; ✦ Academic search boosted
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### Research Context")
    interest = st.text_area("Your Specific Interest",
                             placeholder=f"e.g., {domain_cfg.get('gap_examples','')[:80]}...",
                             height=70)
    gap_type = st.selectbox("Gap Type Focus",
                             ["Any","Methodology","Application","Dataset","Evaluation","Theory","Benchmark"])

    st.markdown("---")
    st.markdown("##### Options")
    use_vision = st.checkbox("🖼️ Vision Analysis (figures/tables)", value=False)
    show_prompts = st.checkbox("🔍 Show prompts (explainability)", value=False)
    n_council_runs = st.select_slider(
        "🔄 Council runs (insight stage)",
        options=[1, 2, 3],
        value=2,
        help="Run the insight council N times and merge all findings. More runs = more coverage, longer wait.",
    )
    st.markdown(
        f'<div style="font-family:\'DM Mono\',monospace;font-size:0.65rem;color:#484f58;'
        f'margin-top:0.3rem;">Each run may surface different insights.<br>'
        f'Merged result shows all unique findings.</div>',
        unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""<div style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#484f58;line-height:1.8;">
PIPELINE<br>
① PDF Parse &amp; PageIndex<br>② LLM Wiki Compile<br>③ LazyGraphRAG<br>
④ Gap Detection<br>⑤ Academic Validation<br>⑥ Proposal Generation<br>
⑦ Parameter Extraction<br>⑧ Cross-Paper Discovery<br>⑨ Council: Model Select<br>
⑩ Multi-Algorithm Simulation<br>⑪ Council: Novel Findings<br>⑫ Report Package
</div>""", unsafe_allow_html=True)

# ── Main header ───────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">Research Gap Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload papers → AI discovers what the field hasn\'t found yet</div>',
            unsafe_allow_html=True)

uploaded_files = st.file_uploader("Upload up to 5 research papers (PDF)",
                                   type=["pdf"], accept_multiple_files=True)

if uploaded_files and len(uploaded_files) > 5:
    uploaded_files = uploaded_files[:5]

if uploaded_files:
    cols = st.columns(min(len(uploaded_files), 5))
    for i, (col, f) in enumerate(zip(cols, uploaded_files)):
        with col:
            st.markdown(
                f'<div class="stage-card" style="text-align:center;padding:0.8rem;">'
                f'<div class="stage-title">Paper {i+1}</div>'
                f'<div class="stage-body" style="font-size:0.78rem;word-break:break-word;">{f.name}</div>'
                f'<div style="font-family:\'DM Mono\',monospace;font-size:0.65rem;color:#8b949e;margin-top:0.3rem;">'
                f'{len(f.getvalue())//1024} KB</div></div>',
                unsafe_allow_html=True)

st.markdown("")
run_col, _ = st.columns([1, 3])
with run_col:
    run_btn = st.button("🚀 Run Analysis", disabled=st.session_state.running)

# ── Main Pipeline ─────────────────────────────────────────────────────────────
if run_btn and uploaded_files:
    if not api_key:
        st.error("Please enter your NVIDIA API key in the sidebar.")
        st.stop()
    if len(uploaded_files) < 2:
        st.warning("Upload at least 2 papers for meaningful gap analysis.")
        st.stop()

    st.session_state.running = True
    st.session_state.pipeline_trace = []

    client  = AzureOpenAIClient(api_key=api_key.strip())
    context = {
        "domain":   selected_domain,
        "interest": interest or "General research improvements",
        "gap_type": gap_type,
    }

    prog = st.progress(0)
    stat = st.empty()

    def upd(msg, pct):
        prog.progress(pct)
        stat.markdown(_card("Pipeline Running", f"⟳ {msg}"), unsafe_allow_html=True)

    try:
        all_papers, all_trees = [], []

        upd("Stage 1 — Parsing PDFs & building PageIndex trees...", 5)
        for i, uf in enumerate(uploaded_files):
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(uf.getvalue())
                tmp_path = tmp.name
            paper = parse_pdf(tmp_path, uf.name,
                               client=client if use_vision else None,
                               use_vision=use_vision)
            tree  = build_tree(paper)
            all_papers.append(paper)
            all_trees.append(tree)
            os.unlink(tmp_path)

        _add_trace("Stage 1 — PDF Parse + PageIndex",
                   f"{len(uploaded_files)} PDFs uploaded",
                   f"{sum(p.get('char_count',0) for p in all_papers):,} chars, "
                   f"{sum(len(p.get('tables',[])) for p in all_papers)} tables extracted")
        prog.progress(15)

        upd("Stage 2 — Compiling LLM Wiki pages...", 17)
        wiki = build_wiki(all_papers, all_trees, client, domain_config=domain_cfg)
        _add_trace("Stage 2 — LLM Wiki Compiler",
                   f"{len(all_papers)} paper trees",
                   f"{len(wiki.get('pages',[]))} wiki pages, "
                   f"{len(wiki.get('cross_links',{}).get('shared_concepts',[]))} cross-links")
        prog.progress(28)

        upd("Stage 3 — Building knowledge graph (LazyGraphRAG)...", 30)
        graph = build_knowledge_graph(wiki, client, domain_config=domain_cfg)
        _add_trace("Stage 3 — Knowledge Graph",
                   "Wiki pages + cross-links",
                   f"{graph['stats']['entity_count']} entities, "
                   f"{graph['stats']['relationship_count']} relationships, "
                   f"{graph['stats']['community_count']} communities")
        prog.progress(42)

        upd("Stage 4 — Detecting research gaps...", 44)
        gaps = detect_gaps(wiki, graph, context, client, domain_config=domain_cfg)
        prog.progress(55)

        upd("Stage 5 — Validating gaps (Semantic Scholar + arXiv)...", 57)
        validated_gaps = validate_gaps(gaps, client, domain_config=domain_cfg)
        _add_trace("Stages 4-5 — Gap Detection + Validation",
                   "Wiki + graph + domain context",
                   f"{len(validated_gaps)} gaps: "
                   f"{sum(1 for g in validated_gaps if g.get('validation_status')=='open')} open, "
                   f"{sum(1 for g in validated_gaps if g.get('validation_status')=='partial')} partial")
        prog.progress(65)

        upd("Stage 6 — Generating research proposals...", 67)
        proposals = generate_proposals(validated_gaps, wiki, context, client,
                                       domain_config=domain_cfg)
        _add_trace("Stage 6 — Proposals",
                   f"{len(validated_gaps)} validated gaps",
                   f"{len(proposals)} proposals generated")
        prog.progress(75)

        st.session_state.results = {
            "papers": [
                {
                    "filename":   p["filename"],
                    "wiki":       w,
                    "char_count": p.get("char_count", 0),
                    "sections":   p.get("sections", {}),
                    "tables":     p.get("tables", []),
                    "tree_summary": tree_to_summary(t, max_chars=2000),
                }
                for p, w, t in zip(all_papers, wiki["pages"], all_trees)
            ],
            "wiki":       wiki,
            "graph":      graph,
            "gaps":       validated_gaps,
            "proposals":  proposals,
            "communities": graph.get("communities", []),
            "context":    context,
        }

        prog.progress(80)
        prog.empty(); stat.empty()
        st.success(f"✅ Analysis complete — {len(validated_gaps)} gaps, {len(proposals)} proposals.")

    except Exception as e:
        prog.empty(); stat.empty()
        st.error(f"Pipeline error: {e}")
        with st.expander("Error details"):
            st.code(traceback.format_exc())
    finally:
        st.session_state.running = False

# ── Results ────────────────────────────────────────────────────────────────────
if st.session_state.results:
    res        = st.session_state.results
    gaps       = res.get("gaps", [])
    proposals  = res.get("proposals", [])
    graph      = res.get("graph", {})
    wiki       = res.get("wiki", {})
    wiki_pages = wiki.get("pages", [])
    context    = res.get("context", {})

    # Domain banner
    res_cfg = get_domain_config(context.get("domain","General AI/ML"))
    st.markdown(f"""
    <div style="background:#0d2136;border:1px solid #1f4f7a;border-radius:8px;
                padding:0.8rem 1.2rem;margin:1rem 0;">
      <span style="font-size:1.4rem;">{res_cfg['icon']}</span>
      <span style="font-size:0.9rem;color:#e6edf3;font-weight:500;margin-left:0.8rem;">
        {context.get('domain','')} — Domain-Tuned Analysis</span>
      <span style="font-size:0.75rem;color:#8b949e;margin-left:0.8rem;">{res_cfg['description']}</span>
    </div>""", unsafe_allow_html=True)

    # Summary metrics
    m1,m2,m3,m4,m5 = st.columns(5)
    for col,(val,lbl) in zip([m1,m2,m3,m4,m5],[
        (len(res.get("papers",[])), "Papers"),
        (graph.get("stats",{}).get("entity_count",0), "Entities"),
        (graph.get("stats",{}).get("community_count",0), "Communities"),
        (len(gaps), "Gaps Found"),
        (len(proposals), "Proposals"),
    ]):
        with col:
            st.markdown(f'<div class="stat-box"><span class="stat-number">{val}</span>'
                        f'<span class="stat-label">{lbl}</span></div>', unsafe_allow_html=True)

    st.markdown("")

    tab1,tab2,tab3,tab4,tab5,tab6,tab7 = st.tabs([
        "📄 Wiki Pages","🕸️ Knowledge Graph","🔍 Research Gaps",
        "💡 Proposals","📊 Computational Lab","🔎 Pipeline Trace","📦 Export",
    ])

    # ── Tab 1: Wiki Pages ─────────────────────────────────────────────────────
    with tab1:
        st.markdown("### Compiled Wiki Pages")
        st.markdown('<div style="font-size:0.82rem;color:#8b949e;margin-bottom:1rem;">'
                    'Each paper compiled into structured knowledge. '
                    'Expand to see what was extracted and the PageIndex tree structure.</div>',
                    unsafe_allow_html=True)

        for i, paper_res in enumerate(res.get("papers", [])):
            page  = paper_res.get("wiki", {})
            title = page.get("title", paper_res.get("filename", f"Paper {i+1}"))

            with st.expander(f"📄 {title}"):
                # Explainability: what was extracted
                c1, c2 = st.columns(2)
                with c1:
                    _section_label("Key Contributions")
                    for c in page.get("contributions", [])[:5]:
                        _item(f"• {c}")
                    st.markdown("")
                    _section_label("Methods Used")
                    for m in page.get("methods", [])[:5]:
                        _item(f"• {m}")
                    st.markdown("")
                    _section_label("Datasets")
                    for d in page.get("datasets", [])[:4]:
                        _item(f"• {d}")
                with c2:
                    _section_label("Key Findings")
                    for f in page.get("key_findings", [])[:4]:
                        _item(f"• {f}")
                    st.markdown("")
                    _section_label("Limitations ⚠️")
                    for lim in page.get("limitations", [])[:4]:
                        _item(f"• {lim}", "#f85149")
                    st.markdown("")
                    _section_label("Future Work 🔭")
                    for fw in page.get("future_work", [])[:4]:
                        _item(f"• {fw}", "#3fb950")

                # PageIndex tree
                tree_summary = paper_res.get("tree_summary", "")
                if tree_summary:
                    with st.expander("🌲 PageIndex Tree — how the paper was navigated"):
                        st.code(tree_summary, language=None)

                # Extracted sections list
                sections = paper_res.get("sections", {})
                if sections:
                    with st.expander(f"📑 Sections detected ({len(sections)} sections)"):
                        for sec_name, sec_text in sections.items():
                            wc = len(str(sec_text).split())
                            status = "✅" if wc > 50 else "⚠️ thin"
                            st.markdown(f'<div class="wiki-item">{status} <b>{sec_name}</b> — {wc} words</div>',
                                        unsafe_allow_html=True)

                # Tables
                tables = paper_res.get("tables", [])
                if tables:
                    with st.expander(f"📊 {len(tables)} table(s) extracted"):
                        for t in tables[:4]:
                            st.markdown(f'**Table (page {t["page"]}, {t["rows"]}×{t["cols"]}):**')
                            st.code(t["content"][:500])

        # Cross-links
        cross_links = wiki.get("cross_links", {})
        if cross_links.get("shared_concepts"):
            st.markdown("### Cross-Paper Links (auto-detected)")
            for link in cross_links["shared_concepts"][:6]:
                papers_str = ", ".join(link.get("papers", []))
                st.markdown(
                    f'<div class="stage-card" style="margin-bottom:0.5rem;">'
                    f'<div class="stage-title">Shared Concept</div>'
                    f'<div class="stage-body"><strong>{link.get("concept","")}</strong>'
                    f' — {link.get("context","")}<br>'
                    f'<span style="font-family:\'DM Mono\',monospace;font-size:0.7rem;color:#58a6ff;">'
                    f'{papers_str}</span></div></div>',
                    unsafe_allow_html=True)

    # ── Tab 2: Knowledge Graph ────────────────────────────────────────────────
    with tab2:
        st.markdown("### Knowledge Graph (LazyGraphRAG)")
        stats = graph.get("stats", {})
        gc1,gc2,gc3,gc4 = st.columns(4)
        for col,(lbl,val) in zip([gc1,gc2,gc3,gc4],[
            ("Entities", stats.get("entity_count",0)),
            ("Relations", stats.get("relationship_count",0)),
            ("Communities", stats.get("community_count",0)),
            ("Orphan Signals", stats.get("orphan_count",0)),
        ]):
            with col:
                st.markdown(f'<div class="stat-box"><span class="stat-number" style="font-size:1.5rem;">'
                            f'{val}</span><span class="stat-label">{lbl}</span></div>',
                            unsafe_allow_html=True)
        st.markdown("")

        # Network diagram
        network_data = graph.get("network_data", {})
        nodes = network_data.get("nodes", [])
        edges = network_data.get("edges", [])

        if nodes:
            try:
                import plotly.graph_objects as go
                import math

                # Spring layout approximation
                n = len(nodes)
                angles = [2 * math.pi * i / n for i in range(n)]
                radius = max(1, n / (2 * math.pi))
                pos_x  = [radius * math.cos(a) for a in angles]
                pos_y  = [radius * math.sin(a) for a in angles]

                edge_x, edge_y = [], []
                for e in edges:
                    si, ti = e["source_idx"], e["target_idx"]
                    if si < n and ti < n:
                        edge_x += [pos_x[si], pos_x[ti], None]
                        edge_y += [pos_y[si], pos_y[ti], None]

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                    line=dict(width=1, color="#30363d"), hoverinfo="none"))
                fig.add_trace(go.Scatter(
                    x=pos_x, y=pos_y, mode="markers+text",
                    text=[nd["name"][:20] for nd in nodes],
                    textposition="top center",
                    textfont=dict(size=9, color="#c9d1d9"),
                    marker=dict(
                        size=[nd["size"] for nd in nodes],
                        color=[nd["color"] for nd in nodes],
                        line=dict(width=1, color="#0d1117"),
                    ),
                    hovertext=[f"{nd['name']}<br>Type: {nd['type']}<br>Papers: {', '.join(nd['papers'][:2])}"
                               for nd in nodes],
                    hoverinfo="text",
                ))
                fig.update_layout(
                    title="Knowledge Graph — Entity Network",
                    showlegend=False,
                    paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
                    font=dict(color="#c9d1d9", size=10),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    height=450, margin=dict(l=10,r=10,t=40,b=10),
                )
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.info("pip install plotly for network diagram")

        # Communities
        st.markdown("#### Communities")
        for comm in graph.get("communities", []):
            with st.expander(f"🏘️ {comm.get('theme','Community')}"):
                st.markdown(f'<div class="stage-body">{comm.get("summary","")}</div>',
                            unsafe_allow_html=True)
                if comm.get("gap_signal"):
                    st.markdown(
                        f'<div style="background:#2d1f00;border:1px solid #9e6a03;border-radius:4px;'
                        f'padding:0.5rem 0.8rem;margin-top:0.6rem;font-size:0.82rem;color:#d29922;">'
                        f'⚠️ Gap Signal: {comm["gap_signal"]}</div>', unsafe_allow_html=True)

        # Entities table (explainability)
        entities = graph.get("entities", [])
        if entities:
            with st.expander(f"📋 All {len(entities)} entities (explainability view)"):
                import pandas as pd
                df = pd.DataFrame([{
                    "Name": e.get("name",""), "Type": e.get("type",""),
                    "Importance": e.get("importance",""),
                    "Papers": ", ".join(e.get("papers",[]))[:60],
                } for e in entities])
                st.dataframe(df, use_container_width=True, height=280)

        # Relationships table
        rels = graph.get("relationships", [])
        if rels:
            with st.expander(f"🔗 All {len(rels)} relationships (explainability view)"):
                import pandas as pd
                df2 = pd.DataFrame([{
                    "Source": r.get("source",""), "Relation": r.get("relation",""),
                    "Target": r.get("target",""),
                    "Evidence": r.get("evidence","")[:60],
                    "Paper": r.get("paper","")[:40],
                } for r in rels])
                st.dataframe(df2, use_container_width=True, height=250)

        # Orphan concepts
        orphans = graph.get("orphan_concepts", [])
        if orphans:
            st.markdown("#### Orphan Concepts — Unexplored Territory")
            st.markdown('<div style="font-size:0.82rem;color:#8b949e;margin-bottom:0.8rem;">'
                        'Concepts mentioned in only one paper with no cross-paper connections. '
                        'High-value gap signals.</div>', unsafe_allow_html=True)
            for o in orphans[:8]:
                ent = o.get("entity", {})
                st.markdown(
                    f'<div class="stage-card" style="padding:0.6rem 1rem;margin-bottom:0.3rem;">'
                    f'<span style="font-family:\'DM Mono\',monospace;color:#79c0ff;">{ent.get("name","")}</span>'
                    f'<span style="font-family:\'DM Mono\',monospace;color:#484f58;font-size:0.7rem;margin-left:0.6rem;">'
                    f'[{ent.get("type","")}]</span>'
                    f'<div style="font-size:0.78rem;color:#8b949e;margin-top:0.2rem;">{o.get("signal","")}</div>'
                    f'</div>', unsafe_allow_html=True)

    # ── Tab 3: Research Gaps ──────────────────────────────────────────────────
    with tab3:
        st.markdown("### Identified Research Gaps")
        open_n    = sum(1 for g in gaps if g.get("validation_status")=="open")
        partial_n = sum(1 for g in gaps if g.get("validation_status")=="partial")
        solved_n  = sum(1 for g in gaps if g.get("validation_status")=="solved")

        f1,f2,f3 = st.columns(3)
        for col,(lbl,cnt,color) in zip([f1,f2,f3],[
            ("🟢 Open",open_n,"#3fb950"),("🟡 Partial",partial_n,"#d29922"),("🔴 Solved",solved_n,"#f85149"),
        ]):
            with col:
                st.markdown(f'<div class="stat-box"><span class="stat-number" style="color:{color};'
                            f'font-size:1.5rem;">{cnt}</span><span class="stat-label">{lbl}</span></div>',
                            unsafe_allow_html=True)
        st.markdown("")

        status_badge = {
            "open": _badge("Open","#3fb950","#0f3d1a","#238636"),
            "partial": _badge("Partial","#d29922","#2d1f00","#9e6a03"),
            "solved": _badge("Solved","#f85149","#3d0f0f","#da3633"),
            "pending": _badge("Pending","#8b949e","#1c1c1c","#30363d"),
        }
        conf_badge = {
            "high": _badge("High Confidence","#79c0ff","#0f2d3d","#1f6feb"),
            "medium": _badge("Medium","#e3b341","#2d2500","#9e6a03"),
            "low": _badge("Low","#8b949e","#1c1c1c","#30363d"),
        }

        for gap in gaps:
            status = gap.get("validation_status","pending")
            conf   = gap.get("confidence","medium")
            st.markdown(
                f'<div class="gap-card {status}">'
                f'<div class="gap-title">{gap.get("title","Untitled")}</div>'
                f'<div class="gap-meta">{status_badge.get(status,"")} {conf_badge.get(conf,"")}'
                f' {_badge(gap.get("gap_type",""),"#8b949e","#1c1c1c","#30363d")}</div>'
                f'<div class="gap-desc">{gap.get("description","")}</div></div>',
                unsafe_allow_html=True)

            with st.expander("Evidence trail + validation"):
                ec1, ec2 = st.columns(2)
                with ec1:
                    _section_label("Evidence from Papers")
                    for ev in gap.get("evidence", []):
                        _item(f"📎 {ev}")
                    if gap.get("missing_connection"):
                        st.markdown("")
                        _section_label("Missing Connection")
                        _item(gap["missing_connection"])
                    if gap.get("search_query_used"):
                        st.markdown("")
                        _section_label("Search Query Used")
                        st.code(gap["search_query_used"], language=None)
                    if gap.get("validation_reasoning"):
                        st.markdown("")
                        _section_label("Validation Note")
                        _item(gap["validation_reasoning"], "#8b949e")
                with ec2:
                    existing = gap.get("existing_papers", [])
                    if existing:
                        _section_label("Related Existing Work")
                        for ep in existing[:4]:
                            _item(f"📄 {ep.get('title','N/A')} ({ep.get('year','')}) "
                                  f"[{ep.get('source','')}]")
                    else:
                        _item("No directly relevant papers found — gap appears open.", "#3fb950")

    # ── Tab 4: Proposals ──────────────────────────────────────────────────────
    with tab4:
        st.markdown("### Research Proposals")
        for i, proposal in enumerate(proposals, 1):
            conf   = proposal.get("confidence","medium")
            effort = proposal.get("effort_estimate","")
            risk   = proposal.get("risk","")
            effort_color = {"short":"#3fb950","medium":"#d29922","long":"#f85149"}.get(
                effort.split()[0].lower() if effort else "", "#8b949e")

            st.markdown(
                f'<div class="proposal-card">'
                f'<div class="proposal-title">#{i}  {proposal.get("title","Untitled")}</div>'
                f'<div>{_badge(conf.title()+" Confidence","#79c0ff","#0f2d3d","#1f6feb")}'
                f'{_badge(effort,effort_color,"#1a2a1a",effort_color)}'
                f'{_badge("Risk: "+risk,"#8b949e","#1c1c1c","#30363d")}</div>'
                f'<div class="proposal-section">Problem Statement</div>'
                f'<div class="proposal-body">{proposal.get("problem_statement","")}</div>'
                f'<div class="proposal-section">Proposed Methodology</div>'
                f'<div class="proposal-body">{proposal.get("methodology","")}</div>'
                f'<div class="proposal-section">Novelty</div>'
                f'<div class="proposal-body">{proposal.get("novelty","")}</div>'
                f'</div>',
                unsafe_allow_html=True)

            with st.expander("Experiments, datasets & source papers"):
                e1,e2 = st.columns(2)
                with e1:
                    _section_label("Suggested Experiments")
                    for exp in proposal.get("suggested_experiments",[]):
                        _item(f"🧪 {exp}")
                    st.markdown("")
                    _section_label("Potential Datasets")
                    for ds in proposal.get("potential_datasets",[]):
                        _item(f"📊 {ds}")
                with e2:
                    _section_label("Builds On")
                    for src in proposal.get("builds_on",[]):
                        _item(f"📎 {src}")
                    if proposal.get("addresses_gap"):
                        st.markdown("")
                        _section_label("Addresses Gap")
                        _item(proposal["addresses_gap"], "#79c0ff")

    # ── Tab 5: Computational Lab ──────────────────────────────────────────────
    with tab5:
        st.markdown("### 📊 Computational Lab")
        st.markdown(
            '<div style="font-size:0.82rem;color:#8b949e;margin-bottom:1.5rem;">'
            'Extract parameters → Cross-paper discovery → Council selects model → '
            'Multi-algorithm simulation → Novel findings → Download package.'
            '</div>', unsafe_allow_html=True)

        col_run, col_info = st.columns([1, 3])
        with col_run:
            run_comp = st.button("🧬 Run Computational Lab", key="run_comp")
        with col_info:
            st.markdown(
                '<div style="font-size:0.78rem;color:#8b949e;padding-top:0.5rem;">'
                'Stages 7–12 · ~4–6 min · Uses same NVIDIA API key</div>',
                unsafe_allow_html=True)

        if run_comp:
            comp_prog = st.progress(0)
            comp_stat = st.empty()

            def cupd(msg, pct):
                comp_prog.progress(pct)
                comp_stat.markdown(_card("Computational Lab", f"⟳ {msg}"), unsafe_allow_html=True)

            try:
                client = AzureOpenAIClient(api_key=api_key.strip())

                papers_raw = [
                    {"sections": p.get("sections",{}),
                     "tables":   p.get("tables",[]),
                     "filename": p.get("filename","")}
                    for p in res.get("papers",[])
                ]

                # Stage 8A: Parameter Extraction
                cupd("Stage 8A — Extracting quantitative parameters...", 8)
                params = extract_parameters(wiki, papers_raw, client)

                cupd("Stage 8A — Deriving cross-paper computed values...", 14)
                derived = derive_cross_paper_parameters(params, client)
                params  = params + derived

                sufficiency = check_sufficiency(params, client)
                csv_data    = params_to_csv(params)
                _add_trace("Stage 8A — Parameter Extraction",
                           f"{len(papers_raw)} papers",
                           f"{len(params)} params ({len(derived)} derived), "
                           f"score {sufficiency.get('coverage_score',0)}/10")
                comp_prog.progress(22)

                # Stage 8B: Cross-Paper Discovery
                cupd("Stage 8B — Cross-paper discovery engine (novel hypotheses)...", 24)
                discovery = run_discovery_engine(wiki, params, context, client, domain_cfg)
                _add_trace("Stage 8B — Cross-Paper Discovery",
                           f"{len(wiki.get('pages',[]))} paper fingerprints",
                           f"{len(discovery.get('combinations',[]))} combinations, "
                           f"{len(discovery.get('hypotheses',[]))} novel hypotheses")
                comp_prog.progress(40)

                # Stage 9: Council model selection
                cupd(f"Stage 9 — LLM Council debating model ({len(params)} params)...", 42)
                model_council = council_select_model(params, sufficiency, context, client)
                comp_prog.progress(55)

                # Stage 10: Multi-algorithm simulation
                models_auto   = auto_select_models(params, sufficiency)
                cupd(f"Stage 10 — Running {len(models_auto)} simulation algorithms: {', '.join(models_auto)}...", 57)
                sim_results   = run_simulation(params, model_council, sufficiency)
                _add_trace("Stage 10 — Multi-Algorithm Simulation",
                           f"Council decision + {len(params)} params",
                           f"Models run: {sim_results.get('models_run', [])}")
                comp_prog.progress(72)

                # Stage 11: Multi-run council insight synthesis
                def _council_progress(run_num, total):
                    cupd(f"Stage 11 — Council run {run_num}/{total} (different model perspectives)...",
                         74 + int((run_num - 1) / total * 12))

                cupd(f"Stage 11 — Running insight council {n_council_runs}× to capture all findings...", 74)
                merged_council = run_multi_council(
                    n_runs=n_council_runs,
                    sim_results=sim_results,
                    discovery=discovery,
                    params=params,
                    gaps=gaps,
                    context=context,
                    client=client,
                    progress_callback=_council_progress,
                )
                comp_prog.progress(88)

                # Stage 12: Document package
                cupd("Stage 12 — Building report package...", 90)
                st.session_state.comp_results = {
                    "params": params, "derived_count": len(derived),
                    "sufficiency": sufficiency, "csv_data": csv_data,
                    "discovery": discovery, "model_council": model_council,
                    "sim_results": sim_results,
                    "insight_council": merged_council,   # merged multi-run result
                    "n_runs": n_council_runs,
                    "zip_bytes": build_zip(
                        params_csv=csv_data, sim_results=sim_results,
                        model_council=model_council, insight_council=merged_council,
                        context=context, params=params, sufficiency=sufficiency,
                        discovery=discovery,
                    ),
                }
                comp_prog.empty(); comp_stat.empty()
                st.success(
                    f"✅ Done — {len(params)} params, {len(models_auto)} simulations, "
                    f"{len(discovery.get('hypotheses',[]))} novel hypotheses.")

            except Exception as e:
                comp_prog.empty(); comp_stat.empty()
                st.error(f"Error: {e}")
                with st.expander("Error details"):
                    st.code(traceback.format_exc())

        # ── Display comp results ──────────────────────────────────────────────
        if st.session_state.comp_results:
            cr             = st.session_state.comp_results
            params         = cr["params"]
            sufficiency    = cr["sufficiency"]
            discovery      = cr["discovery"]
            model_council  = cr["model_council"]
            sim_results    = cr["sim_results"]
            insight_council = cr["insight_council"]

            st.markdown("---")

            # Sufficiency banner with explanation
            score       = sufficiency.get("coverage_score", 0)
            suff_color  = "#3fb950" if score >= 7 else "#d29922" if score >= 4 else "#f85149"
            st.markdown(f"""
            <div style="background:#161b22;border:1px solid {suff_color};border-radius:8px;
                        padding:1rem 1.2rem;margin-bottom:1.2rem;">
              <div style="font-family:'DM Mono',monospace;font-size:0.65rem;color:{suff_color};
                          text-transform:uppercase;letter-spacing:0.1em;">Data Availability</div>
              <div style="font-size:1.2rem;color:{suff_color};font-family:'DM Mono',monospace;
                          font-weight:500;">{score}/10</div>
              <div style="font-size:0.85rem;color:#c9d1d9;margin-top:0.3rem;">
                {len(params)} parameters extracted ({cr.get('derived_count',0)} cross-paper derived) ·
                {sufficiency.get('recommendation','').replace('_',' ').title()}
              </div>
              <div style="font-size:0.78rem;color:#8b949e;margin-top:0.4rem;">
                {sufficiency.get('why_score_is_low','')}
              </div>
            </div>""", unsafe_allow_html=True)

            if sufficiency.get("what_would_improve_score"):
                with st.expander("How to improve data availability score"):
                    for item in sufficiency["what_would_improve_score"]:
                        _item(f"• {item}", "#d29922")

            # Stage 8A: Parameters
            st.markdown("#### Stage 8A — Extracted Parameters")
            grouped = group_params_by_category(params)
            for cat, cat_params in list(grouped.items())[:10]:
                orig  = [p for p in cat_params if not p.get("is_derived")]
                deriv = [p for p in cat_params if p.get("is_derived")]
                label = f"📊 {cat.replace('_',' ').title()} ({len(orig)} extracted"
                label += f", {len(deriv)} derived)" if deriv else ")"
                with st.expander(label):
                    for p in cat_params:
                        ci_str = (f" (CI: {p['ci_lower']}–{p['ci_upper']})"
                                  if p.get("ci_lower") is not None and p.get("ci_upper") is not None else "")
                        derived_tag = " 🔀 derived" if p.get("is_derived") else ""
                        conf_color = {"high":"#3fb950","medium":"#d29922","low":"#f85149"}.get(
                            p.get("confidence","medium"), "#8b949e")
                        st.markdown(
                            f'<div class="wiki-item" style="margin-bottom:0.4rem;">'
                            f'<span style="color:#79c0ff;">{p.get("name","")}</span> = '
                            f'<span style="color:{conf_color};font-family:\'DM Mono\',monospace;">'
                            f'{p.get("value","")}{ci_str}</span> {p.get("unit","")}'
                            f'<span style="color:#484f58;font-size:0.7rem;"> '
                            f'[{p.get("source_paper","")[:45]}]{derived_tag}</span>'
                            f'</div>', unsafe_allow_html=True)
                    if any(p.get("derivation_note") for p in cat_params):
                        for p in cat_params:
                            if p.get("derivation_note"):
                                st.markdown(f'<div class="wiki-item" style="color:#8b949e;font-size:0.75rem;">'
                                            f'ℹ️ {p["name"]}: {p["derivation_note"]}</div>',
                                            unsafe_allow_html=True)

            st.download_button("⬇ Download parameters.csv", data=cr["csv_data"],
                               file_name="parameters.csv", mime="text/csv")

            # Stage 8B: Cross-Paper Discovery
            st.markdown("#### Stage 8B — Cross-Paper Discovery Engine")
            hypotheses = discovery.get("hypotheses", [])
            combinations = discovery.get("combinations", [])

            if hypotheses:
                st.markdown(f'<div style="font-size:0.82rem;color:#8b949e;margin-bottom:1rem;">'
                            f'{len(hypotheses)} novel hypotheses generated · '
                            f'{len(combinations)} cross-paper combinations identified</div>',
                            unsafe_allow_html=True)
                for h in hypotheses:
                    nov   = h.get("novelty_score", 5)
                    imp   = h.get("impact_score", 5)
                    conf  = h.get("confidence","medium")
                    cconf = {"high":"#3fb950","medium":"#d29922","low":"#f85149"}.get(conf,"#8b949e")
                    st.markdown(
                        f'<div style="background:#0d2136;border:1px solid #1f4f7a;border-left:4px solid {cconf};'
                        f'border-radius:8px;padding:1.2rem 1.5rem;margin-bottom:1rem;">'
                        f'<div style="font-family:\'DM Mono\',monospace;font-size:0.65rem;color:#58a6ff;'
                        f'text-transform:uppercase;letter-spacing:0.1em;">'
                        f'{h.get("hypothesis_id","H?")} — {h.get("combination_type","cross-paper")}</div>'
                        f'<div style="font-size:1.05rem;color:#e6edf3;font-weight:500;margin:0.4rem 0;">'
                        f'{h.get("title","")}</div>'
                        f'<div style="font-size:0.87rem;color:#c9d1d9;margin-bottom:0.6rem;">'
                        f'{h.get("claim","")}</div>'
                        f'{_badge(f"Novelty {nov}/10","#79c0ff","#0f2d3d","#1f6feb")}'
                        f'{_badge(f"Impact {imp}/10","#3fb950","#0f3d1a","#238636")}'
                        f'{_badge(conf.title()+" confidence",cconf,"#161b22",cconf)}'
                        f'</div>', unsafe_allow_html=True)

                    with st.expander(f"Evidence chain + validation for {h.get('hypothesis_id','')}"):
                        hc1, hc2 = st.columns(2)
                        with hc1:
                            _section_label("Evidence Chain")
                            for ev in h.get("evidence_chain",[]):
                                _item(f"📎 {ev}")
                            st.markdown("")
                            _section_label("Mechanism")
                            _item(h.get("mechanism",""))
                        with hc2:
                            _section_label("Validation Study")
                            _item(h.get("validation_study",""), "#79c0ff")
                            st.markdown("")
                            _section_label("Clinical Opportunity")
                            _item(h.get("clinical_opportunity",""), "#3fb950")
                            st.markdown("")
                            _section_label("Commercial Opportunity")
                            _item(h.get("commercial_opportunity",""), "#d29922")

                if combinations:
                    with st.expander(f"🔗 {len(combinations)} cross-paper combinations identified"):
                        for combo in combinations[:6]:
                            st.markdown(
                                f'<div class="stage-card" style="margin-bottom:0.4rem;">'
                                f'<div class="stage-title">{combo.get("combination_type","")}</div>'
                                f'<div class="stage-body">{combo.get("novel_combination","")}<br>'
                                f'<span style="color:#484f58;font-size:0.72rem;">'
                                f'Papers: {", ".join(combo.get("papers_involved",[]))}'
                                f' · Evidence: {combo.get("evidence_strength","")} '
                                f'· Feasibility: {combo.get("feasibility","")}</span></div></div>',
                                unsafe_allow_html=True)
            else:
                st.info("No cross-paper hypotheses generated — papers may be too similar or insufficient data.")

            # Stage 9: Council debate
            st.markdown("#### Stage 9 — LLM Council: Model Selection")
            fd = model_council.get("final_decision", {})
            st.markdown(f"""
            <div style="background:#0d2136;border:1px solid #1f4f7a;border-radius:8px;
                        padding:1rem 1.2rem;margin-bottom:1rem;">
              <div style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#58a6ff;
                          text-transform:uppercase;letter-spacing:0.1em;">Council Decision</div>
              <div style="font-size:1.05rem;color:#e6edf3;font-weight:500;margin-top:0.3rem;">
                {fd.get('selected_model','').replace('_',' ').title()}</div>
              <div style="font-size:0.83rem;color:#c9d1d9;margin-top:0.3rem;">{fd.get('rationale','')}</div>
              <div style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#8b949e;margin-top:0.4rem;">
                Confidence: {fd.get('confidence','')} · Scenarios: {', '.join(fd.get('scenarios',[]))}</div>
            </div>""", unsafe_allow_html=True)

            for agent in model_council.get("agents", []):
                with st.expander(f"🤖 {agent['agent']} ({agent['role']}) — {agent.get('model','')}"):
                    st.markdown(f'<div style="font-size:0.85rem;color:#c9d1d9;line-height:1.7;">'
                                f'{agent["response"]}</div>', unsafe_allow_html=True)

            # Stage 10: Multi-algorithm results
            st.markdown(f"#### Stage 10 — Simulation: {sim_results.get('model_used','')}")
            models_run = sim_results.get("models_run", [])
            if models_run:
                st.markdown(
                    f'<div style="font-size:0.82rem;color:#8b949e;margin-bottom:1rem;">'
                    f'Algorithms run: ' +
                    " ".join([_badge(m) for m in models_run]) +
                    '</div>', unsafe_allow_html=True)

            # Headline numbers per scenario
            scenarios_run = sim_results.get("scenarios_run", [])
            if scenarios_run:
                sc_cols = st.columns(len(scenarios_run))
                headline = sim_results.get("headline_numbers", {})
                for col, sc in zip(sc_cols, scenarios_run):
                    sk   = sc.lower().replace(" ", "_")
                    vals = [v for k, v in headline.items() if sk in k]
                    val  = round(vals[0], 3) if vals else "N/A"
                    with col:
                        st.markdown(f'<div class="stat-box"><span class="stat-number" '
                                    f'style="font-size:1.4rem;">{val}</span>'
                                    f'<span class="stat-label">{sc}</span></div>',
                                    unsafe_allow_html=True)

            # Sensitivity ranking
            sens = sim_results.get("sensitivity_ranking", [])
            if sens:
                st.markdown("**Parameter Sensitivity — what drives outcomes most:**")
                for i, s in enumerate(sens[:6], 1):
                    bar = int(min(s["importance"], 1.0) * 100)
                    st.markdown(
                        f'<div style="margin-bottom:0.5rem;">'
                        f'<div style="font-family:\'DM Mono\',monospace;font-size:0.75rem;color:#c9d1d9;">'
                        f'{i}. {s["parameter"]}</div>'
                        f'<div style="background:#30363d;border-radius:3px;height:6px;margin-top:2px;">'
                        f'<div style="background:#58a6ff;width:{bar}%;height:6px;border-radius:3px;"></div>'
                        f'</div></div>', unsafe_allow_html=True)

            # Per-algorithm results (explainability)
            algo_results = sim_results.get("algorithm_results", {})
            if algo_results:
                with st.expander(f"🔬 Per-algorithm results ({len(algo_results)} algorithms run)"):
                    for model_key, model_res in algo_results.items():
                        st.markdown(f"**{model_res.get('model_used', model_key)}**")
                        hl = model_res.get("headline_numbers", {})
                        if hl:
                            for k, v in list(hl.items())[:3]:
                                st.markdown(f'<div class="wiki-item">• {k}: {v}</div>',
                                            unsafe_allow_html=True)

                        # Markov trajectory
                        sc_res = model_res.get("scenario_results", {})
                        if sc_res and model_res.get("model_used","").startswith("Markov"):
                            first = list(sc_res.values())[0]
                            traj  = first.get("trajectories", {})
                            states = first.get("states", [])
                            if traj:
                                try:
                                    import plotly.graph_objects as go
                                    colors_st = ["#3fb950","#d29922","#f85149","#58a6ff","#8b949e"]
                                    fig = go.Figure()
                                    for state, color in zip(states, colors_st):
                                        vals = traj.get(state, [])
                                        fig.add_trace(go.Scatter(
                                            x=list(range(len(vals))), y=vals,
                                            name=state, line=dict(color=color), mode="lines+markers"))
                                    fig.update_layout(
                                        title="Markov — State Trajectories",
                                        paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                                        font=dict(color="#c9d1d9",size=10), height=320,
                                        xaxis=dict(title="Year",gridcolor="#30363d"),
                                        yaxis=dict(title="Cohort (10k)",gridcolor="#30363d"),
                                        margin=dict(l=10,r=10,t=40,b=10))
                                    st.plotly_chart(fig, use_container_width=True)
                                except ImportError:
                                    pass

                        # Survival curve
                        if model_res.get("model_used","").startswith("Survival"):
                            for sc, sc_data in list(sc_res.items())[:2]:
                                curve = sc_data.get("survival_curve", [])
                                tp    = sc_data.get("time_points", [])
                                if curve and tp:
                                    try:
                                        import plotly.graph_objects as go
                                        fig = go.Figure()
                                        fig.add_trace(go.Scatter(
                                            x=tp, y=curve, name=sc,
                                            line=dict(color="#58a6ff"), mode="lines",
                                            fill="tozeroy", fillcolor="rgba(88,166,255,0.1)"))
                                        fig.update_layout(
                                            title="Survival Curve (Weibull)",
                                            paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                                            font=dict(color="#c9d1d9",size=10), height=280,
                                            xaxis=dict(title="Years",gridcolor="#30363d"),
                                            yaxis=dict(title="Survival Probability",gridcolor="#30363d",range=[0,1]),
                                            margin=dict(l=10,r=10,t=40,b=10))
                                        st.plotly_chart(fig, use_container_width=True)
                                    except ImportError:
                                        pass

                        # Bayesian posterior
                        if model_res.get("model_used","").startswith("Bayesian"):
                            for sc, sc_data in list(sc_res.items())[:1]:
                                base = sc_data.get("baseline_probability", 0)
                                post = sc_data.get("posterior_probability", 0)
                                st.markdown(
                                    f'<div class="stage-card">'
                                    f'<div class="stage-title">Bayesian Posterior ({sc})</div>'
                                    f'<div class="stage-body">'
                                    f'Baseline: {base:.1%} → Posterior (with risk factors): <strong>{post:.1%}</strong>'
                                    f' (Δ {sc_data.get("absolute_increase",0):.1%})</div></div>',
                                    unsafe_allow_html=True)

                        st.markdown("---")

            # Stage 11: Multi-run merged results
            st.markdown(f"#### Stage 11 — Merged Council Findings ({cr.get('n_runs',1)}× runs)")

            n_runs   = cr.get("n_runs", 1)
            ic       = cr["insight_council"]
            coverage = ic.get("coverage_stats", {})

            # Coverage banner
            st.markdown(f"""
            <div style="background:#0d2136;border:1px solid #1f4f7a;border-radius:8px;
                        padding:0.9rem 1.2rem;margin-bottom:1.2rem;">
              <div style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#58a6ff;
                          text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.4rem;">
                Multi-Run Coverage — {n_runs} council runs merged</div>
              <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0.8rem;">
                <div style="text-align:center;">
                  <div style="font-size:1.4rem;color:#3fb950;font-family:'DM Mono',monospace;">
                    {coverage.get('total_novel_findings',0)}</div>
                  <div style="font-size:0.68rem;color:#8b949e;text-transform:uppercase;">Novel Findings</div>
                </div>
                <div style="text-align:center;">
                  <div style="font-size:1.4rem;color:#79c0ff;font-family:'DM Mono',monospace;">
                    {coverage.get('total_clinical_insights',0)}</div>
                  <div style="font-size:0.68rem;color:#8b949e;text-transform:uppercase;">Clinical Insights</div>
                </div>
                <div style="text-align:center;">
                  <div style="font-size:1.4rem;color:#d29922;font-family:'DM Mono',monospace;">
                    {coverage.get('total_research_insights',0)}</div>
                  <div style="font-size:0.68rem;color:#8b949e;text-transform:uppercase;">Research Insights</div>
                </div>
              </div>
              <div style="font-size:0.72rem;color:#484f58;margin-top:0.5rem;">
                🟢 Consensus = appeared in all runs &nbsp;
                🟡 Likely = most runs &nbsp;
                🔵 Unique = one run — still captured
              </div>
            </div>""", unsafe_allow_html=True)

            # Novel findings — all groups shown
            novel_groups = ic.get("novel_finding_groups", [])
            if novel_groups:
                st.markdown("##### Novel Findings")
                for g in novel_groups:
                    nf         = g["best"]
                    conf_label = g["confidence"]
                    conf_color = g["confidence_color"]
                    freq       = g["frequency"]

                    st.markdown(f"""
                    <div style="background:#0a1f0a;border:2px solid {conf_color};
                                border-radius:10px;padding:1.2rem 1.5rem;margin-bottom:1rem;">
                      <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                        <div style="font-family:'DM Mono',monospace;font-size:0.62rem;
                                    color:{conf_color};text-transform:uppercase;letter-spacing:0.12em;">
                          🔬 Novel Finding</div>
                        <div>
                          {_badge(conf_label, conf_color, '#161b22', conf_color)}
                          {_badge(freq, '#8b949e', '#1c1c1c', '#30363d')}
                        </div>
                      </div>
                      <div style="font-size:1.05rem;color:#e6edf3;font-weight:600;
                                  margin:0.5rem 0 0.4rem;">{nf.get('title','')}</div>
                      <div style="font-size:0.88rem;color:#c9d1d9;line-height:1.7;
                                  margin-bottom:0.6rem;">{nf.get('claim','')}</div>
                      <div style="font-size:0.75rem;color:#8b949e;font-style:italic;">
                        What's new: {nf.get('what_makes_it_new','')}</div>
                    </div>""", unsafe_allow_html=True)

                    with st.expander(f"Full evidence + opportunity for this finding"):
                        fc1, fc2 = st.columns(2)
                        with fc1:
                            _section_label("Mechanism")
                            _item(nf.get("mechanism",""))
                            st.markdown("")
                            _section_label("Validation Study")
                            _item(nf.get("validation_study",""), "#79c0ff")
                            st.markdown("")
                            _section_label("Evidence Chain")
                            for ev in nf.get("evidence_chain",[]):
                                _item(f"📎 {ev}")
                        with fc2:
                            _section_label("Clinical Opportunity")
                            _item(nf.get("clinical_opportunity",""), "#3fb950")
                            st.markdown("")
                            _section_label("Commercial Opportunity")
                            _item(nf.get("commercial_opportunity",""), "#d29922")
                            st.markdown("")
                            conf = nf.get("confidence","medium")
                            _section_label(f"Confidence: {conf.title()}")
                            _item(nf.get("confidence_reasoning",""), "#8b949e")

                        if g["count"] > 1 and len(g["all_versions"]) > 1:
                            with st.expander(f"See all {g['count']} versions from different runs"):
                                for v in g["all_versions"]:
                                    run_idx = v.get("run_index","?")
                                    st.markdown(
                                        f'<div style="border-left:2px solid #30363d;'
                                        f'padding-left:0.8rem;margin-bottom:0.6rem;">'
                                        f'<div style="font-family:\'DM Mono\',monospace;'
                                        f'font-size:0.65rem;color:#484f58;">Run {run_idx}</div>'
                                        f'<div style="font-size:0.83rem;color:#c9d1d9;">'
                                        f'{v.get("claim","")}</div></div>',
                                        unsafe_allow_html=True)
            else:
                st.warning("No novel findings generated. Try increasing council runs or check paper quality.")

            # Grouped insights — all shown with confidence badges
            def _show_insight_groups(groups: list, icon: str):
                for g in groups:
                    conf_color = g["confidence_color"]
                    freq       = g["frequency"]
                    st.markdown(
                        f'<div style="background:#161b22;border:1px solid #30363d;'
                        f'border-left:3px solid {conf_color};border-radius:5px;'
                        f'padding:0.6rem 0.9rem;margin-bottom:0.35rem;">'
                        f'<div style="display:flex;justify-content:space-between;">'
                        f'<div style="font-size:0.85rem;color:#c9d1d9;">{icon} {g["representative"]}</div>'
                        f'<div style="margin-left:0.8rem;white-space:nowrap;">'
                        f'{_badge(g["confidence"], conf_color, "#161b22", conf_color)}'
                        f'{_badge(freq, "#8b949e", "#1c1c1c", "#30363d")}</div>'
                        f'</div></div>',
                        unsafe_allow_html=True)

            ic1, ic2 = st.columns(2)
            with ic1:
                clinical_groups = ic.get("clinical_insight_groups", [])
                if clinical_groups:
                    st.markdown("##### Clinical Insights")
                    _show_insight_groups(clinical_groups, "🏥")
                st.markdown("")
                steps_groups = ic.get("next_step_groups", [])
                if steps_groups:
                    st.markdown("##### Actionable Next Steps")
                    _show_insight_groups(steps_groups, "→")
            with ic2:
                research_groups = ic.get("research_insight_groups", [])
                if research_groups:
                    st.markdown("##### Research Insights")
                    _show_insight_groups(research_groups, "🔬")
                st.markdown("")
                limits_groups = ic.get("limitation_groups", [])
                if limits_groups:
                    st.markdown("##### Limitations")
                    _show_insight_groups(limits_groups, "⚠️")

            # All agent debates from all runs
            all_agents = ic.get("all_agents", [])
            if all_agents:
                st.markdown("##### Agent Debates (all runs)")
                for agent in all_agents:
                    run_label = f"Run {agent.get('run_index','?')}"
                    with st.expander(f"🤖 {agent['agent']} ({agent['role']}) [{agent.get('model','')}] — {run_label}"):
                        st.markdown(
                            f'<div style="font-size:0.85rem;color:#c9d1d9;line-height:1.7;">'
                            f'{agent["response"]}</div>', unsafe_allow_html=True)

            # Stage 12: Download
            st.markdown("---")
            st.markdown("#### Stage 12 — Download Full Package")
            st.markdown(
                '<div class="stage-card"><div class="stage-title">Package Contents</div>'
                '<div class="stage-body">'
                '📄 parameters.csv · 📊 simulation_results.json · 🏛️ council_debate.md · '
                '💡 insights.md · 📋 full_report.md · 🔬 novel_hypotheses.json'
                f' · ({cr.get("n_runs",1)} council runs merged)'
                '</div></div>', unsafe_allow_html=True)
            st.download_button("⬇ Download ZIP", data=cr["zip_bytes"],
                               file_name="computational_lab_output.zip",
                               mime="application/zip")

    # ── Tab 6: Pipeline Trace ─────────────────────────────────────────────────
    with tab6:
        st.markdown("### 🔎 Pipeline Trace — Explainability View")
        st.markdown(
            '<div style="font-size:0.82rem;color:#8b949e;margin-bottom:1.5rem;">'
            'Full audit trail of what data went in and came out at each stage.</div>',
            unsafe_allow_html=True)

        trace = st.session_state.pipeline_trace
        if not trace:
            st.info("Run the main analysis first — trace will appear here.")
        else:
            for i, entry in enumerate(trace, 1):
                status = entry.get("status","✅")
                st.markdown(
                    f'<div style="background:#161b22;border:1px solid #30363d;border-left:3px solid #58a6ff;'
                    f'border-radius:6px;padding:0.8rem 1.2rem;margin-bottom:0.5rem;">'
                    f'<div style="font-family:\'DM Mono\',monospace;font-size:0.7rem;color:#58a6ff;">'
                    f'{status} STAGE {i} — {entry["stage"]}</div>'
                    f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-top:0.5rem;">'
                    f'<div><span style="font-family:\'DM Mono\',monospace;font-size:0.62rem;color:#484f58;">IN:</span> '
                    f'<span style="font-size:0.8rem;color:#c9d1d9;">{entry["inputs"]}</span></div>'
                    f'<div><span style="font-family:\'DM Mono\',monospace;font-size:0.62rem;color:#484f58;">OUT:</span> '
                    f'<span style="font-size:0.8rem;color:#3fb950;">{entry["outputs"]}</span></div>'
                    f'</div></div>',
                    unsafe_allow_html=True)

        # Show prompts if enabled
        if show_prompts:
            st.markdown("---")
            st.markdown("#### Active Prompts (debug view)")
            from pipeline.wiki_compiler import WIKI_SYSTEM_PROMPT
            from pipeline.gap_detector import GAP_DETECTION_PROMPT
            from pipeline.graph_builder import ENTITY_EXTRACTION_PROMPT

            with st.expander("Wiki Compiler Prompt"):
                st.code(WIKI_SYSTEM_PROMPT, language=None)
            with st.expander("Gap Detection Prompt"):
                st.code(GAP_DETECTION_PROMPT, language=None)
            with st.expander("Graph Entity Extraction Prompt"):
                st.code(ENTITY_EXTRACTION_PROMPT, language=None)

    # ── Tab 7: Export ─────────────────────────────────────────────────────────
    with tab7:
        st.markdown("### Export Results")
        ex1, ex2 = st.columns(2)
        with ex1:
            st.markdown('<div class="stage-card"><div class="stage-title">Markdown Report</div>'
                        '<div class="stage-body">Full analysis — wiki, gaps, proposals.</div></div>',
                        unsafe_allow_html=True)
            st.download_button("⬇ Download Markdown", data=to_markdown_report(res, context),
                               file_name="research_gap_report.md", mime="text/markdown")
        with ex2:
            st.markdown('<div class="stage-card"><div class="stage-title">JSON Export</div>'
                        '<div class="stage-body">Complete structured data for programmatic use.</div></div>',
                        unsafe_allow_html=True)
            st.download_button("⬇ Download JSON", data=to_json(res),
                               file_name="research_gap_analysis.json", mime="application/json")
        st.markdown("---")
        with st.expander("View raw JSON"):
            st.json(res)
