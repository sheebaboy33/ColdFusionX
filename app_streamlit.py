import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import torch
import os

# Import our backend system
from coldfusionx.inference.manager import InferenceManager


# PAGE CONFIGURATION
st.set_page_config(
    page_title="ColdFusionX | Campaign Simulator",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# CSS for "Academic" look
st.markdown("""
<style>
    .reportview-container { background: #f0f2f6 }
    .big-font { font-size:24px !important; font-weight: bold; }
    .metric-box {
        padding: 20px;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    div[data-testid="stMetricValue"] { font-size: 36px; color: #2e86c1; }
</style>
""", unsafe_allow_html=True)


# INITIALIZE SYSTEM (Cached)
@st.cache_resource
def load_system():
    # Points to our extracted artifacts
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


# SIDEBAR: CONTROLS
st.sidebar.title("❄️ ColdFusionX")
st.sidebar.markdown("### Campaign Simulator")

st.sidebar.header("1. Select Case Study")
# Create a readable label for the dropdown
demo_df['label'] = demo_df.apply(lambda x: f"User {x.user_id} | Item {x.item_id} | {x.scene_id}", axis=1)
selected_label = st.sidebar.selectbox("Choose a test sample:", demo_df['label'])

# Get the index of the selected row
selected_index = demo_df[demo_df['label'] == selected_label].index[0]
case_data = manager.store.get_raw_metadata(selected_index)

st.sidebar.divider()

st.sidebar.header("2. Simulation Parameters")
st.sidebar.info("Modify these to test model robustness.")

# The "Killer Feature": Force Cold Start
force_cold = st.sidebar.toggle("❄️ Force Cold-Start", value=False, 
    help="Ignore historical ID data and force model to use Meta-Embeddings (Text/Image/Time) only.")

# SCENE SIMULATOR
st.sidebar.header("3. Context Injection")
unique_scenes = sorted(demo_df['scene_id'].unique())
current_scene = case_data.get('scene_id', 'Unknown') # Safe Get

# If current scene isn't in unique list (rare), default to first
idx_default = unique_scenes.index(current_scene) if current_scene in unique_scenes else 0
override_scene = st.sidebar.selectbox("Simulate Scenario:", unique_scenes, index=idx_default)

# Look up the numeric index for the selected scene string
# We filter the dataframe to find the matching scene_id, then grab the scene_idx
scene_row = demo_df[demo_df['scene_id'] == override_scene].iloc[0]
override_scene_idx = scene_row.get('scene_idx', 0)

st.sidebar.divider()
st.sidebar.markdown("### Thesis Info")
st.sidebar.caption("Project: ColdFusionX\nStudent: Sharm Fernando\nUniversity of Westminster")


## MAIN LAYOUT
# -- HEADER --
col1, col2 = st.columns([3, 1])
with col1:
    st.title("📢 Ad Campaign Diagnostic")
    st.markdown(f"**Scenario Context:** `{override_scene}`")
with col2:
    if force_cold:
        st.error("MODE: COLD START (Meta-Embedding)")
    else:
        st.success("MODE: WARM START (Hybrid)")

st.divider()

# -- CONTENT PREVIEW (Left) & PREDICTION (Right) --
left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("High-Level Campaign Details")
    
    # 1 -> Prepare Data for Display
    # We grab the batch *before* inference so we can visualize inputs
    batch = manager.store.get_case_by_index(selected_index)
    
    with st.container(border=True):
        st.markdown(f"**Item ID:** `{case_data.get('item_id', 'Unknown')}`")
        st.markdown(f"**User ID:** `{case_data.get('user_id', 'Unknown')}`")
        st.markdown(f"**Time:** `{case_data.get('timestamp', 'Unknown')}`")
        
        st.markdown("### 🖼️ Ad Creative (Neural View)")
        st.caption("Since raw images are private/large, we visualize the **ViT Embedding** (What the AI sees).")
        
        # 2 -> VISUALIZE THE EMBEDDING
        # The image_vec is size [1, 768]. We reshape it to [24, 32] to make a 2D grid.
        img_tensor = batch['image_vec'].squeeze().cpu().numpy()
        
        if img_tensor.shape[0] == 768:
            # Reshape 1D vector -> 2D Grid for visualization
            img_grid = img_tensor.reshape(24, 32)
            
            # Create a cool heatmap
            fig_img = px.imshow(
                img_grid, 
                color_continuous_scale='Magma',
                aspect="auto",
                labels=dict(x="Feature Dim X", y="Feature Dim Y", color="Activation")
            )
            fig_img.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                height=300,
                coloraxis_showscale=False # Clean look
            )
            fig_img.update_xaxes(showticklabels=False)
            fig_img.update_yaxes(showticklabels=False)
            
            st.plotly_chart(fig_img, use_container_width=True)
        else:
            st.warning(f"Embedding shape {img_tensor.shape} not standard (768). Cannot visualize.")

        st.markdown("### 📝 Text Context")
        # BERT Embedding Stats (since we don't have raw text)
        txt_tensor = batch['text_vec'].squeeze().cpu().numpy()
        st.bar_chart(txt_tensor[:50]) # Show first 50 dims of BERT
        st.caption(f"BERT Text Embedding (First 50/768 dims). Mean Activation: {txt_tensor.mean():.4f}")

with right_col:
    st.subheader("🚀 Model Prediction")
    
    # RUN INFERENCE LIVE
    # We pass the index and the "Force Cold" toggle state
    # Note: InferenceManager needs to support the 'force_cold' flag logic. (double check on this)
    # We will hack it here: if force_cold is True, we modify the batch inside the manager or passed args.
    
    # Actually, let's just update the manager call to handle this override if we didn't add it yet.
    # For now, we assume the manager handles standard inference. 
    # To support the toggle, we'll manually patch the 'item_count' in the logic below.
    
    batch = manager.store.get_case_by_index(selected_index)
    if force_cold:
        batch['item_count'] = torch.tensor([0]) # Force Zero Count -> Cold Start
    
    # Run Prediction
    with torch.no_grad():
        results = manager.model(
            user_idx=batch["user_idx"],
            item_idx=batch["item_idx"],
            scene_idx=batch["scene_idx"],
            text_vec=batch["text_vec"],
            image_vec=batch["image_vec"],
            time_vec=batch["time_vec"],
            item_count=batch["item_count"]
        )
        prob = torch.sigmoid(results).item()
        attn = manager.model.get_last_fusion_attention()
        meta_attn = manager.model.get_last_meta_attention()
        gates = manager.model.get_last_gates()

    # Gauge Chart
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prob * 100,
        title = {'text': "Predicted CTR %"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "#2e86c1"},
            'steps': [
                {'range': [0, 20], 'color': "#f9e79f"},
                {'range': [20, 50], 'color': "#abebc6"},
                {'range': [50, 100], 'color': "#82e0aa"}],
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

    # Key Insight Box
    confidence = "High" if prob > 0.7 or prob < 0.3 else "Uncertain"
    st.markdown(f"""
    <div class="metric-box">
        <h4>Model Confidence</h4>
        <p>{confidence}</p>
    </div>
    """, unsafe_allow_html=True)


# EXPLAINABILITY (The "Why")
st.divider()
st.header("🔍 Explainability Diagnostics")
st.markdown("Why did ColdFusionX make this prediction?")

tab1, tab2, tab3 = st.tabs(["Modality Importance", "Cold-Start Meta-Weights", "Scenario Routing"])

# --- TAB 1: MODALITY ATTENTION ---
with tab1:
    st.markdown("Which data source contributed most to the decision?")
    
    # Convert attention dict to DF
    attn_data = {k: v.item() for k, v in attn.items()}
    df_attn = pd.DataFrame([attn_data])
    
    fig_attn = px.bar(df_attn.T, orientation='h', 
                      labels={'value': 'Attention Weight', 'index': 'Modality'},
                      title="Fusion Layer Attention Weights")
    fig_attn.update_layout(showlegend=False)
    st.plotly_chart(fig_attn, use_container_width=True)
    
    if force_cold:
        st.info("💡 Notice: ID embedding might still have weight, but it is purely synthesized from content!")

# --- TAB 2: META-EMBEDDING ---
with tab2:
    st.markdown("How was the **Cold-Start Item ID** generated?")
    st.markdown("The Meta-Generator synthesizes a proxy ID from these features:")
    
    meta_data = {k: v.item() for k, v in meta_attn.items()}
    df_meta = pd.DataFrame([meta_data])
    
    fig_meta = px.pie(values=list(meta_data.values()), names=list(meta_data.keys()), 
                      title="Meta-Embedding Composition")
    st.plotly_chart(fig_meta, use_container_width=True)

# --- TAB 3: EXPERT GATES ---
with tab3:
    st.markdown("How did the **Scenario-Aware MMoE** route this sample?")
    
    gate_vals = gates.cpu().numpy().flatten()
    df_gates = pd.DataFrame({'Expert': [f'Expert {i+1}' for i in range(len(gate_vals))], 'Weight': gate_vals})
    
    fig_gates = px.bar(df_gates, x='Expert', y='Weight', 
                       title=f"Expert Activation for Scenario: {case_data['scene_id']}")
    st.plotly_chart(fig_gates, use_container_width=True)