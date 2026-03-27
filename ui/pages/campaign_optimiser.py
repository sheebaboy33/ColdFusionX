import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.graph_objects as go
import os
import sys

# Ensure the app can find the backend modules and utils
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from ui.utils.theme import load_custom_css, COLORS
from coldfusionx.inference.manager import InferenceManager
from navigation import render_top_nav


# RENDER THE TOP NAVIGATION BAR
render_top_nav(page_name="campaign_optimiser")

# PAGE CONFIGURATION
st.set_page_config(
    page_title="ColdFusionX | Campaign Optimizer", 
    page_icon="🎯", 
    layout="wide",
    initial_sidebar_state="expanded"
)

load_custom_css()


# BACKEND INITIALIZATION (CACHED)
@st.cache_resource(show_spinner="Loading ColdFusionX Model Weights...")
def load_system():
    artifact_path = os.path.join(project_root, "artifacts", "release_demo")
    return InferenceManager(artifact_dir=artifact_path, device="cpu")

try:
    manager = load_system()
    demo_df = manager.store.list_cases()
except Exception as e:
    st.error(f"System Initialization Failed. Check artifact paths. Error: {e}")
    st.stop()


# HEADER
st.title("Campaign Optimizer")
st.markdown("Simulate ad performance, test temporal contexts, and forecast campaign ROI before spending real budget.")
st.divider()


# UI LAYOUT: TOP CONTROLS
# Create a nice dropdown for marketers using the English translated titles
demo_df['dropdown_label'] = demo_df['item_title_en'].fillna("Unknown Campaign") + " (" + demo_df['scene_name'] + ")"
selected_label = st.selectbox("Select Ad Creative to Simulate:", demo_df['dropdown_label'].tolist())

# Get the specific local index for the selected ad
selected_idx = demo_df[demo_df['dropdown_label'] == selected_label].index[0]
ad_metadata = manager.store.get_raw_metadata(selected_idx)


# UI LAYOUT: THE CREATIVE PROFILE (FULL WIDTH)
st.markdown("### Creative Profile")

# Internal columns to keep the image reasonably sized next to the text
prof_col1, prof_col2 = st.columns([1, 3])

with prof_col1:
    # Try to load the actual image
    img_path = manager.store.get_image_path(ad_metadata.get('item_id', '0'))
    if os.path.exists(img_path):
        st.image(img_path, use_container_width=True, caption="Raw Creative Image")
    else:
        st.info("Raw image file not available in subset. Model will use cached Vision embeddings.")

with prof_col2:
    st.markdown(f"**Campaign Copy:** {ad_metadata.get('item_title_en', 'N/A')}")
    st.markdown(f"**Category Tag:** {ad_metadata.get('item_category_en', 'N/A')}")
    st.markdown(f"**Target Scene:** {ad_metadata.get('scene_name', 'N/A')}")
    
    # System Tracking IDs
    user_id = ad_metadata.get('user_id', str(ad_metadata.get('user_id_idx', 'N/A')))
    item_id = ad_metadata.get('item_id', str(ad_metadata.get('item_id_idx', 'N/A')))
    scene_id = ad_metadata.get('scene_id', str(ad_metadata.get('scene_idx', 'N/A')))
    st.caption(f"**Tracking** | User ID: `{user_id}` | Item ID: `{item_id}` | Scene ID: `{scene_id}`")
    
    # Cold Start Badge
    hist_clicks = ad_metadata.get('historical_click_count', 0)
    if hist_clicks == 0:
        st.error("**COLD START:** This creative has 0 historical clicks.")
    elif hist_clicks < 10:
        st.warning(f"**COOL ITEM:** Minimal history ({hist_clicks} clicks).")
    else:
        st.success(f"**WARM ITEM:** Established history ({hist_clicks} clicks).")

st.divider()


# UI LAYOUT: 2-COLUMN DASHBOARD
col1, col2 = st.columns(2)

# COLUMN 1: DYNAMIC CONTEXT SHIFTER
with col1:
    st.markdown("### Context Simulator")
    st.markdown("Shift the environment to see how the AI adapts.")
    
    # Context Controls
    sim_traffic = st.radio("Traffic Condition:", ["Standard Weekday", "Weekend / Holiday Spike"])
    
    # Injecting the Context Override
    base_batch = manager.store.get_case_by_index(selected_idx)
    override_time = base_batch["time_vec"].clone()
    
    # Assuming index 2 corresponds to the 'is_weekend' boolean flag
    if sim_traffic == "Weekend / Holiday Spike":
        override_time[0, 2] = 1.0 
    else:
        override_time[0, 2] = 0.0

    # Run Live Inference
    with st.spinner("Calculating CTR..."):
        prediction = manager.predict_demo_case(selected_idx, override_time_vec=override_time)
    
    ctr_score = prediction["probability"]
    
    # Beautiful Plotly Gauge for CTR
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = ctr_score * 100,
        number = {'suffix': "%", 'font': {'color': COLORS['text_main']}},
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Predicted Click-Through Rate", 'font': {'color': COLORS['text_muted'], 'size': 14}},
        gauge = {
            'axis': {'range': [0, 15], 'tickwidth': 1, 'tickcolor': COLORS['text_muted']}, 
            'bar': {'color': COLORS['primary']},
            'bgcolor': "rgba(0,0,0,0)",
            'steps': [
                {'range': [0, 2], 'color': COLORS['danger']},
                {'range': [2, 5], 'color': COLORS['card_bg']},
                {'range': [5, 15], 'color': COLORS['success']}
            ]
        }
    ))
    fig_gauge.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)", font={'color': COLORS['text_main']})
    st.plotly_chart(fig_gauge, use_container_width=True, theme=None)
    
    # Explainability Translation
    # "Why this score?" Explainability Translation
    f_attn = prediction["fusion_attention"]
    total_weight = sum(f_attn.values())
    
    if total_weight > 0:
        # Normalize the Sigmoid values to a 100% scale for the marketer
        normalized_attn = {k: (v / total_weight) * 100 for k, v in f_attn.items()}
        dominant_modality = max(normalized_attn, key=normalized_attn.get)
        dom_percentage = normalized_attn[dominant_modality]
        
        st.info(f"**AI Insight:** Among the creative features, the model relies most heavily on the **{dominant_modality.upper()}** ({dom_percentage:.1f}%) to predict engagement in this context.")
    else:
        st.info("**AI Insight:** The model is relying purely on historical ID tracking for this prediction.")

        
# COLUMN 2: ROI FORECASTER
with col2:
    st.markdown("### ROI Forecaster")
    st.markdown("Project business impact based on live AI prediction.")
    
    # Budget Inputs
    budget = st.number_input("Campaign Budget ($)", min_value=100, max_value=100000, value=5000, step=500)
    cpm = st.number_input("Estimated CPM ($/1000 impressions)", min_value=0.50, max_value=20.0, value=2.50, step=0.50)
    
    # The Math
    impressions = (budget / cpm) * 1000
    expected_clicks = int(impressions * ctr_score)
    
    cpc = budget / expected_clicks if expected_clicks > 0 else 0.0

    st.divider()
    
    # Display the Business Metrics
    st.metric(label="Total Ad Impressions", value=f"{int(impressions):,}")
    st.metric(label="Expected Clicks", value=f"{expected_clicks:,}")
    
    if cpc < 1.00:
        st.metric(label="Effective Cost-Per-Click (eCPC)", value=f"${cpc:.2f}", delta="Highly Efficient", delta_color="normal")
    else:
        st.metric(label="Effective Cost-Per-Click (eCPC)", value=f"${cpc:.2f}", delta="Expensive Traffic", delta_color="inverse")


# UI LAYOUT: CROSS-CHANNEL PLACEMENT OPTIMIZER
st.divider()
st.markdown("### Cross-Channel Placement Optimizer")
st.markdown("Discover the optimal ad placement by simulating performance across different platform environments.")

# Dictionary mapping human-readable scenes to their AntM2C tensor indices
scene_options = {
    "Search Results": 1,
    "Banner Ad": 2,
    "Recommendation Feed": 3,
    "Category Page": 4
}

if st.button("Run Placement Analysis", type="primary"):
    with st.spinner("Simulating multi-channel performance..."):
        placement_results = []
        
        for scene_name, s_idx in scene_options.items():
            # Inject the specific scene tensor
            override_scene = torch.tensor([s_idx], dtype=torch.long, device="cpu")
            
            # Run inference while keeping the currently selected Time context
            pred = manager.predict_demo_case(
                selected_idx, 
                override_time_vec=override_time, 
                override_scene_idx=override_scene
            )
            placement_results.append({"Placement": scene_name, "Predicted CTR": pred["probability"]})
            
        # Find the absolute best placement
        df_places = pd.DataFrame(placement_results).sort_values(by="Predicted CTR", ascending=True)
        best_placement = df_places.iloc[-1]["Placement"]
        
        st.success(f"**Optimal Placement Found:** Running this creative in the **{best_placement}** yields the highest user engagement.")
        
        # Plotly Horizontal Bar Chart
        fig_places = go.Figure(go.Bar(
            x=df_places["Predicted CTR"] * 100, 
            y=df_places["Placement"],
            orientation='h',
            text=[f"{val:.2f}%" for val in df_places["Predicted CTR"] * 100],
            textposition='auto',
            marker_color=[COLORS['primary'] if s == best_placement else COLORS['card_bg'] for s in df_places["Placement"]]
        ))
        
        fig_places.update_layout(
            title="Expected CTR by Platform Placement",
            xaxis_title="Predicted CTR (%)",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLORS['text_muted']),
            height=250,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig_places, use_container_width=True, theme=None)

st.divider()
st.caption("Powered by ColdFusionX Inference Engine. Network serialization bypassed for sub-50ms local execution.")