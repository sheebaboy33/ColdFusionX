# ui/app_streamlit.py
import sys
import os
from pathlib import Path

# Add root to path BEFORE other project imports
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from ui.navigation import render_top_nav
import plotly.graph_objects as go
import numpy as np


# PAGE CONFIGURATION
st.set_page_config(
    page_title="ColdFusionX | Research Hub",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# RENDER TOP NAVIGATION
render_top_nav("home")

# --- CSS FOR ACADEMIC TYPOGRAPHY ---
st.title("❄️ ColdFusionX")
st.markdown("""
<style>
    .big-font { font-size:20px !important; }
    .hero-title { font-size: 3rem; font-weight: 800; color: #0d47a1; margin-bottom: 0px; }
    .hero-subtitle { font-size: 1.5rem; color: #424242; font-weight: 300; margin-top: 0px; }
    .highlight-box { 
        background-color: #f0f8ff; 
        padding: 20px; 
        border-radius: 10px; 
        border-left: 5px solid #0d47a1; 
        margin-bottom: 20px;
    }
    .research-gap-box {
        background-color: #fff5f5;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #ffcdd2;
        margin-bottom: 10px;
    }
    .contribution-box {
        background-color: #e8f5e9;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #c8e6c9;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- HERO SECTION ---
col_hero_text, col_hero_img = st.columns([5, 3])

with col_hero_text:
    st.markdown("""
    <p class="hero-subtitle">
    Explainable Multi-Modal CTR Prediction via Modality-Attentive <br>
    Meta-Embeddings and Temporal Drift Adaptations
    </p>
    """, unsafe_allow_html=True)
    
    # Student / Supervisor Details (Academic Context)
    st.markdown("""
    **Author:** Sharm Johan Ashley Fernando (w1985557)  
    **Supervisor:** Ms. Nethmi Wijesinghe  
    **Institution:** Informatics Institute of Technology (IIT) in collaboration with University of Westminster, UK
    """)

    st.markdown("---")
    
    # The "Elevator Pitch" (Abstract Summary)
    st.markdown("""
    ### 🎯 Project Abstract
    Modern CTR prediction struggles with **Cold-Start** scenarios where new items lack interaction history. 
    **ColdFusionX** is a unified architecture that fuses **Text, Image, ID, and Time** features using 
    **Scenario-Aware Meta-Learning**. It integrates a temporal encoder and **Instance-Wise Explainability**, 
    ensuring recommendations are not only accurate but transparent and adaptable to concept drift.
    """)
    
    if st.button("👉 View Live Simulator", type="primary"):
        st.switch_page("pages/ad_simulator.py")

with col_hero_img:
    # 3D LATENT SPACE VISUALIZATION (Conceptual)
    # Visualizing the "Fusion" of modalities
    st.markdown("#### Latent Space Visualization")
    st.caption("Conceptual representation of Multi-Modal Embedding Fusion")
    
    # Generating dummy data for visual flair
    n = 100
    x, y, z = np.random.normal(0, 1, n), np.random.normal(0, 1, n), np.random.normal(0, 1, n)
    categories = np.random.choice(['Text', 'Image', 'Time'], n)
    colors = {'Text': '#EF553B', 'Image': '#00CC96', 'Time': '#636EFA'}
    c_map = [colors[c] for c in categories]

    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(size=6, color=c_map, opacity=0.8),
        hovertext=categories
    )])
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(xaxis_title='Semantic', yaxis_title='Visual', zaxis_title='Temporal'),
        height=350
    )
    st.plotly_chart(fig, use_container_width=True)


st.markdown("---")

# SECTION 2: RESEARCH LANDSCAPE (GAPS VS CONTRIBUTIONS)
st.header("🔍 Research Landscape")
st.markdown("Addressing critical limitations in state-of-the-art (SOTA) CTR prediction.")

# We use tabs to switch between the "Problem" (Gaps) and the "Solution" (Contributions)
tab_gaps, tab_contrib, tab_impact = st.tabs(["❌ Research Gaps", "✅ Novel Contributions", "🚀 Real-World Impact"])

with tab_gaps:
    st.markdown("### Unresolved Challenges in Current Literature ")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="research-gap-box">
            <strong>1. Insufficient Multi-Modal Signals</strong><br>
            Existing models (e.g., MetaEmb, EmerG) rely heavily on ID embeddings. They neglect rich 
            <strong>textual and visual features</strong>, which are critical for connecting similar items 
            in cold-start settings.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="research-gap-box">
            <strong>2. Lack of Scenario-Awareness</strong><br>
            Current Multi-Task frameworks (MMoE, PLE) treat all scenarios uniformly. They fail to adapt 
            fusion strategies for specific contexts (e.g., <em>Coupons vs. Videos</em>), ignoring the 
            distinct user intents in each scenario.
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="research-gap-box">
            <strong>3. Neglect of Temporal Drift</strong><br>
            Baseline cold-start models (MeLU, CLCRec) ignore <strong>Temporal context</strong> or 
            <strong>Item Age</strong>. In fast-paced markets (Flash Sales), a static model becomes 
            obsolete quickly as user behavior drifts.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="research-gap-box">
            <strong>4. The "Black Box" Problem</strong><br>
            Deep Learning CTR models lack transparency. Stakeholders cannot verify <em>why</em> a 
            prediction was made, leading to a lack of trust in regulated business environments.
        </div>
        """, unsafe_allow_html=True)

with tab_contrib:
    st.markdown("### ColdFusionX Architectural Contributions")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="contribution-box">
            <strong>1. Modality-Attentive Meta-Embeddings</strong><br>
            A novel generator that dynamically fuses Text, Image, and Time. It assigns 
            <strong>instance-wise attention weights</strong> to each modality, addressing the 
            modality imbalance problem.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="contribution-box">
            <strong>2. Scenario-Aware MMoE Routing</strong><br>
            Extends Mixture-of-Experts with <strong>Scene-Specific Gating</strong>. The model 
            selectively activates expert sub-networks based on the business scenario (e.g., Ads, Mini-Programs).
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="contribution-box">
            <strong>3. Temporal Drift Adaptation </strong><br>
            Integrates a lightweight <strong>Time2Vec</strong> encoder directly into the fusion pipeline. 
            This allows the model to capture periodic trends (e.g., weekend spikes) and adaptability.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="contribution-box">
            <strong>4. Post-Hoc Explainability Layer</strong><br>
            A dedicated module capable of attributing CTR predictions to specific modalities 
            (e.g., <em>"Click driven 60% by Image, 40% by Text"</em>), even for unseen items.
        </div>
        """, unsafe_allow_html=True)

with tab_impact:
    st.markdown("### 🌍 Significance of the Study")
    st.info("""
    **Problem Domain Contribution:**
    * **Commercial Viability:** Reduces wasted ad spend on new items by accurately predicting their potential without history.
    * **User Trust:** Provides transparent reasoning for recommendations, essential for "Responsible AI."
    
    **Research Domain Contribution:**
    * **Unified Pipeline:** Bridges the gap between *Meta-Learning* (for Cold-Start) and *Multi-Modal Fusion*.
    * **Reproducibility:** Validated on the large-scale **AntM2C** dataset (Huan et al., 2024).
    """)

st.markdown("---")

# --- SECTION 3: SYSTEM OVERVIEW ---
st.header("🏗️ System Architecture")

col_arch_text, col_arch_img = st.columns([1, 2])

with col_arch_text:
    st.markdown("""
    **The ColdFusionX Pipeline:**
    
    1.  **Multi-Modal Input:** Processing ID, Text (BERT), Image (ViT), and Time.
    2.  **Cold-Start Detector:** Routes traffic based on user history.
    3.  **Meta-Embedding Generator:** Synthesizes proxies for new items.
    4.  **Adaptive Fusion:** Weights modalities based on context.
    5.  **MMoE Routing:** Scenario-specific expert selection.
    """)
    st.caption("See 'Performance' page for full details.")

with col_arch_img:
    arch_img_path = ROOT_DIR / "docs" / "ColdFusionX-Architecture.jpg"
    if arch_img_path.exists():
        st.image(str(arch_img_path), caption="High-Level Architecture of ColdFusionX", use_container_width=True)
    else:
        st.warning(f"Image not found at: {arch_img_path}")


# --- FOOTER ---
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.8em; border-top: 1px solid #ddd; padding-top: 20px;">
    <strong>ColdFusionX</strong> | Informatics Institute of Technology | University of Westminster<br>
    <em>Submitted in partial fulfilment of the requirements for the BSc (Hons) in Computer Science.</em>
</div>
""", unsafe_allow_html=True)