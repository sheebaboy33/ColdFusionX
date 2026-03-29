import sys
import os
from pathlib import Path
import streamlit as st

# Ensure the app can find the backend modules and utils
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ui.utils.theme import load_custom_css, COLORS
from navigation import render_top_nav


# PAGE CONFIGURATION
st.set_page_config(
    page_title="ColdFusionX | Research Hub",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="❄️"
)

# Load the global theme (Gradient H1s, Backgrounds, Metric Cards)
render_top_nav(page_name="home")
load_custom_css()


# CLEAN CUSTOM CSS
st.markdown("""
<style>
    /* Bumping up the subtitle size and adjusting the margin to sit perfectly under the title */
    .hero-subtitle { 
        font-size: 1.85rem; /* Increased from 1.5rem */
        font-weight: 400; 
        margin-top: -10px; 
        padding-top: 0px; 
        color: #D0D4DA; 
        line-height: 1.4;
    }
    .highlight-text { color: #06B6D4; font-weight: 700; }

    /* Abstract Box Styling */
    .abstract-box {
        background-color: rgba(255, 255, 255, 0.03);
        border-left: 4px solid #06B6D4;
        padding: 20px;
        border-radius: 8px;
        margin-top: 15px;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }

    /* Custom Card for Research Gaps (Subtle Red Accent) */
    .gap-card {
        background-color: rgba(239, 68, 68, 0.05);
        border-left: 4px solid #EF4444;
        padding: 16px 20px;
        border-radius: 6px;
        margin-bottom: 16px;
        transition: transform 0.3s ease, background-color 0.3s ease;
    }
    .gap-card:hover {
        transform: translateX(8px);
        background-color: rgba(239, 68, 68, 0.1);
    }
    
    /* Custom Card for Contributions (Subtle Green Accent) */
    .cont-card {
        background-color: rgba(16, 185, 129, 0.05);
        border-left: 4px solid #10B981;
        padding: 16px 20px;
        border-radius: 6px;
        margin-bottom: 16px;
        transition: transform 0.3s ease, background-color 0.3s ease;
    }
    .cont-card:hover {
        transform: translateX(8px);
        background-color: rgba(16, 185, 129, 0.1);
    }

    .card-title { font-weight: 700; color: #F8FAFC; margin-bottom: 6px; font-size: 1.05rem; }
    .card-text { color: #d8d6d6; font-size: 0.9rem; line-height: 1.4; }

    /* Transform Primary Buttons into Gradient Action Buttons */
    button[kind="primary"] {
        background: linear-gradient(90deg, #3B82F6, #06B6D4) !important;
        border-radius: 8px !important;
        border: none !important;
        transition: all 0.3s ease !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 0.25rem 1rem !important; /* Keeps the button sleek */
    }
    
    button[kind="primary"]:hover {
        transform: scale(1.03) !important;
        box-shadow: 0 4px 15px rgba(6, 182, 212, 0.4) !important;
        background: linear-gradient(90deg, #3B82F6, #06B6D4) !important;
    }
    
    /* Ensure the text inside the button pops */
    button[kind="primary"] div > p {
        font-weight: 600 !important;
        font-size: 1.05rem !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)


# HERO SECTION & ABSTRACT
st.title("ColdFusionX")
st.markdown('<p class="hero-subtitle">An Explainable Multi-Modal CTR Prediction Framework for <span class="highlight-text">Cold-Start Recommendation</span>.</p>', unsafe_allow_html=True)

st.write("") 

# Academic Context
st.caption("**Author:** Sharm Johan Ashley Fernando (w1985557)")
st.caption("**Supervisor:** Ms. Nethmi Wijesinghe")
st.caption("**Degree:** BSc (Hons) in Computer Science ")
st.caption("**Institution:** Informatics Institute of Technology (IIT) & University of Westminster, UK")

# Project Abstract
st.markdown("""
<div class="abstract-box">
    <strong>Project Abstract:</strong> <br>
            To overcome the severe performance degradation of Click-Through Rate (CTR) models on cold-start items, we developed ColdFusionX: an adaptive multi-modal architecture that leverages Stochastic Alpha Masking and Micro-Drift Temporal FiLM to dynamically synthesize proxy embeddings from vision, text, and time data. Evaluated on the AntM2C dataset, the prototype achieved state-of-the-art robustness by outperforming established industry baselines with a Cold-Start AUC of 0.8545 and an Overall AUC of 0.9255.
</div>
""", unsafe_allow_html=True)

st.divider()


# THE RESEARCH LANDSCAPE
st.markdown("### The Research Landscape")
st.markdown("Addressing structural bottlenecks in state-of-the-art (SOTA) Click-Through Rate prediction.")

col_gap, col_cont = st.columns(2)

with col_gap:
    st.markdown("#### Identified Research Gaps")
    st.markdown("""
    <div class="gap-card">
        <div class="card-title">1. Insufficient Multi-Modal Integration</div>
        <div class="card-text">Standard models rely heavily on ID embeddings, failing to utilize rich textual and visual features to prevent predictive collapse in zero-shot scenarios.</div>
    </div>
    <div class="gap-card">
        <div class="card-title">2. Static Fusion & Gradient Starvation</div>
        <div class="card-text">Existing frameworks lack dynamic modality adjustment at the instance level, leaving networks vulnerable to gradient starvation when collaborative signals dominate.</div>
    </div>
    <div class="gap-card">
        <div class="card-title">3. Neglect of Temporal Drift</div>
        <div class="card-text">Item relevance and the semantic weight of content features naturally drift as an item ages, a dynamic that is under-modelled in sparse environments.</div>
    </div>
    <div class="gap-card">
        <div class="card-title">4. The Black-Box Opacity</div>
        <div class="card-text">Foundational CTR models prioritize predictive interaction over visibility, failing to expose how disparate signals jointly shape outcomes.</div>
    </div>
    """, unsafe_allow_html=True)

with col_cont:
    st.markdown("#### ColdFusionX Contributions")
    st.markdown("""
    <div class="cont-card">
        <div class="card-title">1. Stop-Gradient Cosine Alignment</div>
        <div class="card-text">Geometrically pulls multi-modal content representations closer to the collaborative latent space, bridging disjoint manifolds without collapsing the encoders.</div>
    </div>
    <div class="cont-card">
        <div class="card-title">2. Stochastic Alpha Masking (SAM)</div>
        <div class="card-text">Reduces over-reliance on dominant ID pathways during training by mathematically governing the transition from Content-Based routing to Collaborative routing.</div>
    </div>
    <div class="cont-card">
        <div class="card-title">3. Residual Temporal FiLM</div>
        <div class="card-text">Incorporates lightweight temporal conditioning directly into the fusion pipeline to capture behavioural drift without overwhelming the core representation.</div>
    </div>
    <div class="cont-card">
        <div class="card-title">4. Intrinsic Explainability</div>
        <div class="card-text">Exposes internal tensor routing through modality-attentive Meta-Embeddings, demonstrating the True Routing Mass utilized for every prediction.</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()


# SYSTEM ARCHITECTURE
st.markdown("### System Architecture")

col_arch_img, col_arch_text = st.columns([2, 1])

with col_arch_img:
    with st.container(border=True):
        arch_img_path = project_root / "docs" / "ColdFusionX-Archtecture.png"
        if arch_img_path.exists():
            st.image(str(arch_img_path), use_container_width=True)
        else:
            st.info("Architecture Diagram Placeholder: Add image to `docs/ColdFusionX-Archtecture.png`")

with col_arch_text:
    st.markdown("#### Processing Pipeline")
    st.markdown("""
    * **Multi-Modal Encoders:** Extract semantic representations from both raw textual and visual inputs.
    * **Alignment Module:** Utilizes Stop-Gradient Cosine Alignment to bridge disjoint latent manifolds.
    * **Temporal Integration:** Injects item-age and periodic dynamics via Residual FiLM conditioning.
    * **Fusion Engine:** Synthesizes the final Content-Based Proxy ID using an independent Sigmoid Attention mechanism.
    * **SAM Gateway:** Dynamically shifts predictive reliance from the Content Proxy to historical Collaborative IDs.
    """)

st.divider()


# INTERACTIVE NAVIGATION HUB
st.markdown("### Interactive Artifact Modules")

nav1, nav2, nav3 = st.columns(3)

with nav1:
    with st.container(border=True):
        st.markdown("#### System Performance")
        st.caption("Review offline evaluation metrics, ROC-AUC curves, and comparative baselines.")
        st.write("")
        if st.button("View Performance", type="primary", use_container_width=True):
            st.switch_page("pages/performance.py")

with nav2:
    with st.container(border=True):
        st.markdown("#### Campaign Simulator")
        st.caption("Test the model in a live business context. Adjust budgets and target scenes.")
        st.write("")
        if st.button("Launch Simulator", type="primary", use_container_width=True):
            st.switch_page("pages/campaign_optimiser.py")

with nav3:
    with st.container(border=True):
        st.markdown("#### Model X-Ray")
        st.caption("Inspect intrinsic diagnostics, tensor attention, and SAM trajectories.")
        st.write("")
        if st.button("Open X-Ray", type="primary", use_container_width=True):
            st.switch_page("pages/model_xray.py")

st.write("")
st.write("")


# FOOTER
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.85em; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 20px;">
    <strong>ColdFusionX Architecture</strong> <br>
    <em>Software Artifact optimized for academic review and conference-level demonstration.</em>
</div>
""", unsafe_allow_html=True)