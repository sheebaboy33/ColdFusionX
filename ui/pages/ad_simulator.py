import streamlit as st
import sys
import torch
import plotly.graph_objects as go
from pathlib import Path
import pandas as pd
import plotly.express as px
import os

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from coldfusionx.inference.manager import InferenceManager
    from coldfusionx.utils import get_file_path
except ImportError as e:
    st.error(f"Import Error: {e}")

from ui.navigation import render_top_nav

# PAGE CONFIGURATION
st.set_page_config(
    page_title="ColdFusionX | Ad Simulator", 
    page_icon="❄️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# RENDER TOP NAV
render_top_nav("ad_simulator")

# CSS for "Academic" look
st.markdown("""
<style>
    /* Modern Metric Cards */
    .metric-box {
        background: linear-gradient(145deg, #ffffff, #f8f9fa);
        border: 1px solid #e2e8f0;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        text-align: center;
        transition: all 0.3s ease;
    }
    /* Subtle lift on hover */
    .metric-box:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        border-color: #0052FF;
    }
    .metric-box h4 {
        color: #64748b;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 10px;
    }
    .metric-box p {
        color: #0f172a;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    /* Style the tabs to look like modern pills */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0px 0px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


# INITIALIZE SYSTEM (Cached)
@st.cache_resource
def load_system():
    # Points to the extracted artifacts
    ARTIFACT_PATH = os.path.join(os.getcwd(), "artifacts", "release_demo")
    
    if not os.path.exists(ARTIFACT_PATH):
        st.error(f"Artifacts not found at {ARTIFACT_PATH}. Did you unzip the demo bundle?")
        return None
        
    manager = InferenceManager(artifact_dir=ARTIFACT_PATH, device="cpu")
    return manager

manager = load_system()

if not manager:
    st.stop()

# Load the demo metadata dataframe for the sidebar list
demo_df = manager.store.list_cases()


def get_live_attribution(model, batch_data):
    """
    Calculates 'Gradient * Input' attribution for the current batch.
    Matches PyTorch training identically (NO manual L2 normalization).
    """
    # 1. Detach and Enable Gradients for Raw Feature Tensors
    text_v = batch_data["text_vec"].clone().detach().requires_grad_(True)
    image_v = batch_data["image_vec"].clone().detach().requires_grad_(True)
    time_v = batch_data["time_vec"].clone().detach().requires_grad_(True)
    
    # 2. Run Forward Pass (Raw embeddings, exactly like training)
    results = model(
        user_idx=batch_data["user_idx"],
        item_idx=batch_data["item_idx"],
        scene_idx=batch_data["scene_idx"],
        text_vec=text_v,
        image_vec=image_v,
        time_vec=time_v,
        item_count=batch_data["item_count"],
        cold_mask=batch_data.get("cold_mask", None) # Pass cold mask if expected
    )
    
    # 3. Backward Pass (Calculate Gradients)
    prob = torch.sigmoid(results)
    prob.backward()
    
    # 4. Compute Importance (Gradient * Input)
    with torch.no_grad():
        imp_text = (text_v.grad * text_v).abs().sum().item()
        imp_image = (image_v.grad * image_v).abs().sum().item()
        imp_time = (time_v.grad * time_v).abs().sum().item()
        
    return {"Text": imp_text, "Image": imp_image, "Time": imp_time}


# SIDEBAR: CONTROLS
st.sidebar.title("📻 Campaign Simulator")

st.sidebar.header("1. Select Case Study")
demo_df['label'] = demo_df.apply(lambda x: f"User {x.user_id} | Item {x.item_id} | {x.scene_id}", axis=1)
selected_label = st.sidebar.selectbox("Choose a test sample:", demo_df['label'])

selected_index = demo_df[demo_df['label'] == selected_label].index[0]
case_data = manager.store.get_raw_metadata(selected_index)

st.sidebar.divider()

st.sidebar.header("2. Simulation Parameters")
st.sidebar.info("Modify these to test model robustness.")

# Force Cold Start Toggle
#force_cold = st.sidebar.toggle("❄️ Force Cold-Start", value=False, 
#    help="Treat item as zero-shot. Activates Meta-Embeddings and Stochastic Alpha Masking.")
st.sidebar.markdown("### Exposure Simulator")
sim_count = st.sidebar.slider(
    "Simulate Item Exposure Count:", 
    min_value=0, max_value=20, value=15,
    help="0 = Zero-Shot Cold Start. >=10 = Fully Warm. In between = Dynamic SAM Blending."
)
force_cold = (sim_count == 0)

st.sidebar.divider()

# SCENE SIMULATOR
st.sidebar.header("3. Context Injection")
unique_scenes = sorted(demo_df['scene_id'].unique())
current_scene = case_data.get('scene_id', 'Unknown') 

idx_default = unique_scenes.index(current_scene) if current_scene in unique_scenes else 0
override_scene = st.sidebar.selectbox("Simulate Scenario:", unique_scenes, index=idx_default)

scene_row = demo_df[demo_df['scene_id'] == override_scene].iloc[0]
override_scene_idx = scene_row.get('scene_idx', 0)

st.sidebar.divider()
st.sidebar.markdown("### Thesis Info")
st.sidebar.caption("Project: ColdFusionX \n\nStudent: Sharm Fernando\n\nUniversity of Westminster")


## MAIN LAYOUT
col1, col2 = st.columns([3, 1])
with col1:
    st.title("📢 Ad Campaign Diagnostic")
    st.markdown(f"**Scenario Context:** `{override_scene}`")
with col2:
    if force_cold:
        st.error("MODE: COLD START (Zero-Shot)")
    else:
        st.success("MODE: WARM START (Collaborative)")

st.divider()

left_col, right_col = st.columns([1, 1])

# 1. PREPARE DATA 
batch = manager.store.get_case_by_index(selected_index)

# Apply Overrides
#batch['scene_idx'] = torch.tensor([override_scene_idx], dtype=torch.long)
#if force_cold:
#    batch['item_count'] = torch.tensor([0], dtype=torch.long)
#    batch['cold_mask'] = torch.tensor([True], dtype=torch.bool)
#else:
#    if batch['item_count'].item() == 0:
#        batch['item_count'] = torch.tensor([10], dtype=torch.long) # Simulate some history
#    batch['cold_mask'] = torch.tensor([False], dtype=torch.bool)
# Apply Overrides
batch['scene_idx'] = torch.tensor([override_scene_idx], dtype=torch.long)
batch['item_count'] = torch.tensor([sim_count], dtype=torch.long)

if force_cold:
    batch['cold_mask'] = torch.tensor([True], dtype=torch.bool)
else:
    batch['cold_mask'] = torch.tensor([False], dtype=torch.bool)

# 2. RUN INFERENCE 
with torch.no_grad():
    results = manager.model(
        user_idx=batch["user_idx"],
        item_idx=batch["item_idx"],
        scene_idx=batch["scene_idx"],
        text_vec=batch["text_vec"],   # RAW EMBEDDINGS
        image_vec=batch["image_vec"], # RAW EMBEDDINGS
        time_vec=batch["time_vec"],   # RAW EMBEDDINGS
        item_count=batch["item_count"],
        cold_mask=batch["cold_mask"]
    )
    
    prob = torch.sigmoid(results).item()
    
    # Safe Extraction of Explainability specific to v10 Architecture
    meta_attn = getattr(manager.model, 'get_last_meta_attention', lambda: None)()
    
    # Extract the SAM Blend Ratio directly from the Blender module
    alpha_tensor = getattr(manager.model.blender, '_last_alpha', None)
    alpha_val = alpha_tensor.item() if alpha_tensor is not None else 1.0


with left_col:
    st.subheader("High-Level Campaign Details")
    selected_row = demo_df.iloc[selected_index]

    with st.container(border=True):
        st.markdown(f"**Item ID:** `{selected_row['item_id']}`")
        st.markdown(f"**User ID:** `{selected_row['user_id']}`")
        st.markdown(f"**Time:** `{selected_row.get('timestamp', case_data.get('timestamp', 'Unknown'))}`")
        
        st.markdown("### 🖼️ Ad Creative")
        
        view_mode = st.radio("Visual Mode", ["Human Vision", "Neural View"], horizontal=True, label_visibility="collapsed")

        if view_mode == "Human Vision":
            item_id_str = str(selected_row['item_id'])
            img_path = manager.store.get_image_path(item_id_str)
            
            if os.path.exists(img_path):
                c1, c2, c3 = st.columns([1, 4, 1]) 
                with c2:
                    st.image(img_path, caption=f"Raw Input: {item_id_str}.png", use_container_width=True)
            else:
                st.warning(f"⚠️ Image not found for Item {item_id_str}")

        else:
            st.caption("ViT Embedding (What the AI sees - 768 dim)")
            img_tensor = batch['image_vec'].squeeze().cpu().numpy()
            
            if img_tensor.shape[0] == 768:
                img_grid = img_tensor.reshape(24, 32)
                fig_img = px.imshow(img_grid, color_continuous_scale='Magma', aspect="auto")
                fig_img.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=300, coloraxis_showscale=False)
                fig_img.update_xaxes(showticklabels=False)
                fig_img.update_yaxes(showticklabels=False)
                st.plotly_chart(fig_img, use_container_width=True)
            else:
                st.warning(f"Embedding shape {img_tensor.shape} not standard (768).")

        st.markdown("### 📝 Text Context")
        txt_tensor = batch['text_vec'].squeeze().cpu().numpy()
        st.bar_chart(txt_tensor[:50]) 
        st.caption(f"BERT Text Embedding (First 50 dims). Mean Activation: {txt_tensor.mean():.4f}")

with right_col:
    st.subheader("🚀 Model Prediction")
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prob * 100, 
        title = {'text': "Predicted CTR %"},
        gauge = {
            'axis': {'range': [0, 40]}, 
            'bar': {'color': "#2e86c1"},
            'steps': [
                {'range': [0, 10], 'color': "#f9e79f"},   
                {'range': [10, 20], 'color': "#abebc6"},  
                {'range': [20, 30], 'color': "#82e0aa"},        
                {'range': [30, 40], 'color': "#abebc6"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 10
            }
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

    confidence = "High" if prob > 0.7 or prob < 0.3 else "Uncertain"
    st.markdown(f"""
    <div class="metric-box">
        <h4>Model Confidence</h4>
        <p>{confidence}</p>
    </div>
    """, unsafe_allow_html=True)


# EXPLAINABILITY
st.divider()
st.header("🔍 Explainability Diagnostics")
st.markdown("Examine the internal mechanisms and dynamic routing of ColdFusionX.")

tab1, tab2, tab3 = st.tabs(["Live Attribution", "Meta-Embedding Focus", "SAM Blend Strategy"])

with tab1:
    st.markdown("### ⚡ Live Feature Attribution")
    st.markdown("Calculates **instance-wise importance** using Gradient * Input. Try toggling 'Force Cold-Start'!")
    st.caption("🎯 **Global Network Focus:** Measures end-to-end impact (Gradient × Input) across all Transformer pathways, answering: *'What raw features drove the final CTR probability?'*")
    
    try:
        attr_scores = get_live_attribution(manager.model, batch)
        df_explain = pd.DataFrame({"Modality": list(attr_scores.keys()), "Importance": list(attr_scores.values())})
        
        total_imp = df_explain["Importance"].sum()
        df_explain["Contribution %"] = (df_explain["Importance"] / total_imp) * 100 if total_imp > 0 else 0

        fig_explain = px.bar(
            df_explain, x="Contribution %", y="Modality", orientation='h',
            title=f"Feature Contribution (Scenario {override_scene})", color="Modality", text_auto='.1f',
            color_discrete_map={'Text':'#1f77b4', 'Image':'#aec7e8', 'Time':'#ff7f0e'}
        )
        fig_explain.update_layout(xaxis_title="Relative Importance (%)")
        st.plotly_chart(fig_explain, use_container_width=True)
        
        top_mod = df_explain.sort_values("Importance", ascending=False).iloc[0]['Modality']
        st.info(f"💡 For this specific prediction, the model relied most heavily on **{top_mod}**.")
        
    except Exception as e:
        st.error(f"Could not calculate attribution: {str(e)}")


# --- TAB 2: META-EMBEDDING FOCUS ---
# --- TAB 2: META-EMBEDDING FOCUS ---
with tab2:
    st.markdown("### 🧬 Proxy ID Synthesis")
    
    # We check if Alpha has reached the Few-Shot cap (1.0)
    # Using 0.999 to avoid any floating point precision issues
    if alpha_val >= 0.999:
        st.success("🔥 **Fully Warm Item (Threshold Reached)**")
        st.markdown(f"This item has accumulated enough historical exposure that the SAM Blender has assigned it an **Alpha of 1.00**.")
        st.markdown("Because the network has total confidence in the historical **Collaborative ID**, the multi-modal Meta-Generator is completely masked out and has **zero impact** on this prediction.")
        st.info("💡 **Slide the 'Exposure Simulator' to the left** to see how the model synthesizes a proxy ID when historical data is missing.")
    else:
        st.markdown(f"**Blend Active:** The Meta-Generator synthesized a proxy ID using this modality mix. Because Alpha is **{alpha_val:.2f}**, this proxy makes up **{(1.0 - alpha_val)*100:.0f}%** of the final fused ID.")
        
        if meta_attn:
            # Standard Visualization 
            meta_raw = {k: v.item() for k, v in meta_attn.items()}
            total_weight = sum(meta_raw.values())
            
            if total_weight == 0: 
                meta_norm = {k: 0.33 for k in meta_raw}
            else:
                meta_norm = {k: v / total_weight for k, v in meta_raw.items()}
                
            df_meta = pd.DataFrame({'Modality': list(meta_norm.keys()), 'Contribution': list(meta_norm.values())})
            
            fig_meta = px.pie(
                df_meta, 
                values='Contribution', names='Modality', title="Meta-Embedding Composition",
                color='Modality',
                color_discrete_map={'text':'#1f77b4', 'image':'#aec7e8', 'time':'#ff7f0e'}
            )
            fig_meta.update_traces(textinfo='percent+label')
            st.plotly_chart(fig_meta, use_container_width=True)
            
            with st.expander("View Raw Sigmoid Attention Weights"):
                 st.json(meta_raw)
        else:
            st.warning("Meta-Attention extraction is not currently configured in the active PyTorch model.")

with tab3:
    st.markdown("### ⚖️ Stochastic Alpha Masking (SAM)")
    st.markdown("How is the final representation blended between the historical **Collaborative ID** and the synthesized **Meta-Embedding**?")
    
    if force_cold:
        st.info("❄️ **Zero-Shot Mode Active:** The model is forced to rely 100% on the multi-modal Meta-Embedding (Alpha = 0.0).")
    else:
        st.success(f"🔥 **Warm Mode Active:** The model dynamically blends historical ID with multi-modal features based on the exposure count.")
        
    # Visualize the Alpha Blend Ratio
    blend_df = pd.DataFrame({
        'Source': ['Collaborative ID (Alpha)', 'Meta-Embedding (1 - Alpha)'],
        'Weight': [alpha_val, 1.0 - alpha_val]
    })
    
    fig_blend = px.pie(
        blend_df, 
        values='Weight', 
        names='Source', 
        title=f"Current Blend Ratio (Alpha: {alpha_val:.4f})",
        color='Source', 
        color_discrete_map={'Collaborative ID (Alpha)':'#2ca02c', 'Meta-Embedding (1 - Alpha)':'#1f77b4'}
    )
    fig_blend.update_traces(textinfo='percent+label', hole=0.5)
    st.plotly_chart(fig_blend, use_container_width=True)

st.divider()

with st.expander("🛠️ Developer / Tensor Inspector"):
    st.markdown("### Raw Input Tensors (Live Simulation)")
    t_col1, t_col2 = st.columns(2)
    with t_col1:
        st.code(f"User ID Idx: {batch['user_idx'].item()}")
        st.code(f"Item ID Idx: {batch['item_idx'].item()} (Count: {batch['item_count'].item()})")
        st.code(f"Scene Idx: {batch['scene_idx'].item()} (Override Active)")
    with t_col2:
        st.code(f"Text Emb Shape: {list(batch['text_vec'].shape)}")
        st.code(f"Image Emb Shape: {list(batch['image_vec'].shape)}")
        st.code(f"Time Emb Shape: {list(batch['time_vec'].shape)}")