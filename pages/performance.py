import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
import os
from navigation import render_top_nav

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="ColdFusionX | Performance",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. RENDER TOP NAVIGATION
render_top_nav("performance")

# --- HELPER: LOAD METRICS ---
@st.cache_data
def load_data():
    # Paths to your uploaded files
    # Ensure these files are in the root directory or update path accordingly
    try:
        with open('eval_metrics.json', 'r') as f:
            eval_data = json.load(f)
        
        ablation_df = pd.read_csv('ablation_metrics.csv')
        scenario_df = pd.read_csv('scenario_metrics.csv')
        
        return eval_data, ablation_df, scenario_df
    except FileNotFoundError as e:
        st.error(f"Missing Data File: {e.filename}. Please ensure 'eval_metrics.json', 'ablation_metrics.csv', and 'scenario_metrics.csv' are in the app directory.")
        return None, None, None

eval_data, ablation_df, scenario_df = load_data()

if not eval_data:
    st.stop()

# --- CSS STYLING ---
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .metric-value { font-size: 2.5rem; font-weight: 700; color: #0d47a1; }
    .metric-label { font-size: 1rem; color: #6c757d; font-weight: 500; }
    .insight-box {
        background-color: #e3f2fd;
        border-left: 5px solid #2196f3;
        padding: 15px;
        margin-top: 20px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)


# SECTION 1: SYSTEM ARCHITECTURE
st.title("📈 Model Performance & Evaluation")
st.markdown("Comprehensive evaluation of **ColdFusionX** on the AntM2C dataset, featuring multi-modal ablation studies and scenario-specific benchmarking.")

with st.expander("🏗️ View System Architecture", expanded=False):
    # Check for image
    if os.path.exists("ColdFusionX-Architecture.jpg"):
        st.image("ColdFusionX-Architecture.jpg", caption="Figure 1: ColdFusionX Scenario-Aware MMoE Architecture", use_container_width=True)
    else:
        st.warning("Architecture image 'ColdFusionX-Architecture.jpg' not found.")


# SECTION 2: GLOBAL METRICS (JSON DATA)
st.markdown("### 1. Global Performance Metrics")
st.markdown("Performance comparison across **Overall**, **Cold-Start** (Unseen Items), and **Warm-Start** splits.")

# Extract metrics from JSON
m_overall = eval_data['metrics']['overall']
m_cold = eval_data['metrics']['cold']
m_warm = eval_data['metrics']['warm']

# Helper to render custom metric card
def render_metric(label, auc, logloss, acc):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label} AUC</div>
        <div class="metric-value">{auc:.4f}</div>
        <div style="margin-top: 10px; font-size: 0.9em; color: #555;">
            LogLoss: <strong>{logloss:.4f}</strong><br>
            Accuracy: <strong>{acc:.2%}</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    render_metric("Overall", m_overall['auc'], m_overall['logloss'], m_overall['accuracy'])
with c2:
    render_metric("❄️ Cold-Start", m_cold['auc'], m_cold['logloss'], m_cold['accuracy'])
with c3:
    render_metric("🔥 Warm-Start", m_warm['auc'], m_warm['logloss'], m_warm['accuracy'])

st.divider()


# SECTION 3: ABLATION STUDY (CSV DATA)
st.header("2. Ablation Study: Component Impact")
st.markdown("Analyzing the contribution of **Meta-Embeddings**, **Routing**, and **Modality Attention**.")

col_ablation_chart, col_ablation_text = st.columns([2, 1])

with col_ablation_chart:
    # Prepare data for plotting (Long Format)
    # We want to compare AUC_Cold vs AUC_Warm for each variant
    plot_df = ablation_df[['variant', 'auc_cold', 'auc_warm']].melt(id_vars='variant', var_name='Split', value_name='AUC')
    
    # Rename for readability
    plot_df['Split'] = plot_df['Split'].replace({'auc_cold': 'Cold Start', 'auc_warm': 'Warm Start'})
    
    fig_ablation = px.bar(
        plot_df, 
        x='variant', 
        y='AUC', 
        color='Split', 
        barmode='group',
        text_auto='.4f',
        color_discrete_map={'Cold Start': '#00CC96', 'Warm Start': '#EF553B'},
        height=400,
        title="Impact of Components on AUC (Cold vs. Warm)"
    )
    fig_ablation.update_layout(yaxis_range=[0.75, 0.95]) # Zoom in to see differences
    st.plotly_chart(fig_ablation, use_container_width=True)

with col_ablation_text:
    st.markdown("#### 💡 Key Findings")
    st.markdown("""
    The ablation results confirm the theoretical hypothesis:
    
    1.  **Meta-Embeddings (No Meta):** Removing them causes a massive drop in **Cold-Start AUC** (from `0.8418` to `0.8258`), confirming they are critical for unseen items.
    2.  **Scenario Routing (No Routing):** Removing expert routing hurts **Both** warm and cold performance, as the model loses its ability to adapt to different traffic patterns.
    3.  **Modality Attention:** Essential for balancing the signal from Text/Image vs. ID features.
    """)
    
    with st.expander("View Raw Ablation Data"):
        st.dataframe(ablation_df[['variant', 'auc_all', 'auc_cold', 'auc_warm', 'delta_auc_cold']], hide_index=True)


st.divider()


# SECTION 4: SCENARIO ANALYSIS (CSV DATA)
st.header("3. Scenario-Specific Performance")
st.markdown("Evaluating performance across the 5 business scenarios (0-4) defined in AntM2C.")

# Prepare Scenario Data
# We will compare AUC_Cold and AUC_Warm per scene
scene_plot_df = scenario_df[['scene', 'AUC_cold', 'AUC_warm']].melt(id_vars='scene', var_name='Split', value_name='AUC')
scene_plot_df['Split'] = scene_plot_df['Split'].replace({'AUC_cold': 'Cold Start', 'AUC_warm': 'Warm Start'})
scene_plot_df['scene'] = "Scene " + scene_plot_df['scene'].astype(str)

fig_scene = px.bar(
    scene_plot_df,
    x='scene',
    y='AUC',
    color='Split',
    barmode='group',
    title="AUC by Business Scenario",
    color_discrete_map={'Cold Start': '#636EFA', 'Warm Start': '#FFA15A'},
    height=400
)
fig_scene.update_layout(yaxis_range=[0.5, 1.0])
st.plotly_chart(fig_scene, use_container_width=True)

# Scenario Insights
st.info("""
**Observation:**
* **Scene 1 & 2** show the highest performance (>0.9 AUC), indicating rich data availability.
* **Scene 3** appears to be the most challenging (AUC ~0.70), likely due to high sparsity or high noise.
* **Cold-Start Gap:** The gap between Warm and Cold performance varies significantly by scenario, validating the need for **Scenario-Aware Routing**.
""")

with st.expander("View Detailed Scenario Metrics"):
    st.dataframe(scenario_df, hide_index=True)

# FOOTER
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.8em;">
    Metrics constructured using 3 million samples from the AntM2C Dataset Evaluation
</div>
""", unsafe_allow_html=True)