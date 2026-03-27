import streamlit as st
import plotly.graph_objects as go
import os
import sys

# Ensure the app can find the backend modules and utils
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from ui.utils.theme import load_custom_css, COLORS
from coldfusionx.inference.manager import InferenceManager
from ui.navigation import render_top_nav


# PAGE CONFIGURATION
st.set_page_config(
    page_title="ColdFusionX | Model X-Ray", 
    page_icon="", 
    layout="wide",
    initial_sidebar_state="expanded"
)

load_custom_css()

# Render the dynamic navigation bar
render_top_nav(page_name="model_xray") 


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


# HEADER & SELECTION
st.title("Intrinsic Diagnostics & Model X-Ray")
st.markdown("Inspect the internal tensor mechanics, attention weights, and latent semantic space of ColdFusionX.")
st.divider()

demo_df['dropdown_label'] = demo_df['item_title_en'].fillna("Unknown Campaign") + " (" + demo_df['scene_name'] + ")"
selected_label = st.selectbox("Select Target Tensor for Inspection:", demo_df['dropdown_label'].tolist())

selected_idx = demo_df[demo_df['dropdown_label'] == selected_label].index[0]
ad_metadata = manager.store.get_raw_metadata(selected_idx)


# SECTION 1: MODALITY ATTENTION & CONTRIBUTIONS
st.header("1. Meta-Generator Attention Weights")
st.markdown("Real-time evaluation of multi-modal inputs prior to Proxy ID synthesis")

col1, col2 = st.columns([1.2, 1.8])

with col1:
    # The sleek SaaS Card component
    with st.container(border=True):
        st.markdown("#### Target Creative")
        
        img_path = manager.store.get_image_path(ad_metadata.get('item_id', '0'))
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
            
        st.write("") # Spacer
        st.caption(f"**Copy:** {ad_metadata.get('item_title_en', 'N/A')}")
        st.caption(f"**Category:** {ad_metadata.get('item_category_en', 'N/A')}")

with col2:
    with st.spinner("Analyzing Tensor Routing..."):
        prediction = manager.predict_demo_case(selected_idx)
    
    attn_weights = prediction["meta_attention"]
    
    # Calculate the relative contribution for the new metrics
    total_weight = sum(attn_weights.values())
    rel_contribs = {k: (v / total_weight) * 100 if total_weight > 0 else 0 for k, v in attn_weights.items()}
    
    # Display the Dominant Modality as a top-level metric
    dominant_mod = max(attn_weights, key=attn_weights.get)
    st.metric(
        label="Dominant Feature Contribution", 
        value=f"{dominant_mod.title()} ({rel_contribs[dominant_mod]:.1f}%)"
    )
    
    # Plotly Bar Chart showing BOTH Raw Logits and Relative %
    keys_ordered = ['text', 'image', 'time']
    labels_ordered = ['Text Feature', 'Image Feature', 'Temporal Feature']
    
    fig_attn = go.Figure(go.Bar(
        x=[attn_weights[k] for k in keys_ordered],
        y=labels_ordered,
        orientation='h',
        marker_color=[COLORS['primary'], COLORS['primary'], COLORS['accent']],
        text=[f"Raw: {attn_weights[k]:.4f} | {rel_contribs[k]:.1f}%" for k in keys_ordered],
        textposition='auto',
        # Force the text to be larger and bold so it stands out against the metric card
        textfont=dict(size=14, color="white", family="Arial Black")
    ))
    
    fig_attn.update_layout(
        title="Sigmoid Activation Strength vs. Relative Contribution",
        xaxis_title="Raw Activation Strength (0.0 to 1.0)",
        yaxis_title="",
        xaxis=dict(range=[0, 1.05]),
        yaxis=dict(showticklabels=True),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS['text_muted']),
        height=360, 
        margin=dict(l=100, r=10, t=40, b=30),
        # Decreased bargap to make the actual bars thicker and more prominent
        bargap=0.15 
    )
    st.plotly_chart(fig_attn, use_container_width=True, theme=None)
    
    st.info("💡 Labels display the raw Sigmoid activation alongside the feature's relative contribution to the final Proxy ID.")

st.divider()


# SECTION 2: STOCHASTIC ALPHA MASKING (SAM)
st.header("2. The 'Warm-Up' Trajectory (SAM Engine)")
st.markdown("Watch the SAM engine dynamically shift reliance from the Meta-Embedding to the Collaborative ID as item exposure grows.")

col3, col4 = st.columns([1.2, 1.8])

with col3:
    st.markdown("### Inject Historical Clicks")
    sim_clicks = st.slider("Simulated Exposure Count", min_value=0, max_value=20, value=int(ad_metadata.get('historical_click_count', 0)), step=1)
    
    warmup_pred = manager.predict_demo_case(selected_idx, override_item_count=sim_clicks)
    current_alpha = warmup_pred["alpha"]
    current_ctr = warmup_pred["probability"]
    
    st.write("") # Spacer
    
    # The Interactive Gauge Chart
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = current_ctr * 100,
        number = {'suffix': "%", 'font': {'color': COLORS['text_main']}},
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Simulated Predicted CTR", 'font': {'color': COLORS['text_muted'], 'size': 14}},
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
    fig_gauge.update_layout(height=240, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)", font={'color': COLORS['text_main']})
    st.plotly_chart(fig_gauge, use_container_width=True, theme=None)
    st.caption("Notice how the predicted CTR adjusts as the SAM engine shifts reliance toward the Collaborative ID.")

with col4:
    fig_alpha = go.Figure(data=[go.Pie(
        labels=['Synthesized Meta-Embedding', 'Collaborative History ID'],
        values=[1.0 - current_alpha, current_alpha],
        hole=0.6, 
        marker=dict(colors=[COLORS['primary'], COLORS['accent']]),
        textinfo='percent',
        hoverinfo='label+percent'
    )])

    fig_alpha.update_layout(
        title="Live Tensor Blend (SAM Ratio)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS['text_muted']),
        height=260, # Slightly reduced height to fit the metric card below
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.0) 
    )
    st.plotly_chart(fig_alpha, use_container_width=True, theme=None)
    
    # Moved the Alpha Metric below the Doughnut inside a neat card
    alpha_c1, alpha_c2, alpha_c3 = st.columns([0.5, 2, 0.5]) # Centering columns
    with alpha_c2:
        with st.container(border=True):
            st.metric(label="Calculated Alpha (α) Ratio", value=f"{current_alpha:.4f}")
            st.caption("α = 0.0 (Pure Cold / Meta) ➔ α = 1.0 (Pure Warm / Collab)")

st.divider()


# SECTION 3: SEMANTIC RETRIEVAL (LATENT SPACE)
st.header("3. Semantic Latent Space Alignment")
st.markdown("Validating the multi-modal encoders by performing a Cosine Similarity search across the concatenated Vision and Text embedding space.")

if st.button("🔍 Run Semantic Similarity Search", type="primary"):
    with st.spinner("Scanning feature store..."):
        similar_cases = manager.get_similar_cases(selected_idx, top_k=3)
        
        st.success("Successfully retrieved the top 3 closest mathematical neighbors in the latent space.")
        
        sim_cols = st.columns(3)
        for i, case in enumerate(similar_cases):
            with sim_cols[i]:
                # Wrap each result in a bordered container (Creates a SaaS "Card" look)
                with st.container(border=True):
                    st.markdown(f"#### Rank {i+1}")
                    st.metric(label="Cosine Similarity", value=f"{case['similarity_score']:.4f}")
                    
                    s_img = manager.store.get_image_path(case.get('item_id', '0'))
                    if os.path.exists(s_img):
                        st.image(s_img, use_container_width=True)
                        
                    st.caption(f"**Copy:** {case.get('item_title_en', 'N/A')}")
                    st.caption(f"**Category:** {case.get('item_category_en', 'N/A')}")

st.divider()
st.caption("Diagnostic UI | ColdFusionX PyTorch Backend")