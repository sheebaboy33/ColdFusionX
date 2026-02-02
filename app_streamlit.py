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

st.sidebar.divider()

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
st.sidebar.caption("Project: ColdFusionX \n\nStudent: Sharm Fernando\n\nUniversity of Westminster")


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

# 1. PREPARE DATA (Centralized)
batch = manager.store.get_case_by_index(selected_index)

# Apply Overrides
batch['scene_idx'] = torch.tensor([override_scene_idx], dtype=torch.long)
if force_cold:
    batch['item_count'] = torch.tensor([0], dtype=torch.long)
else:
    if batch['item_count'].item() == 0:
        batch['item_count'] = torch.tensor([1], dtype=torch.long)

# 2. RUN INFERENCE (With STRICT Normalization)
with torch.no_grad():
    # Explicit L2 Normalization
    # We normalize Text and Image to Unit Vectors (Length=1.0).
    # This prevents the "loudest" vector from drowning out the others.
    # Note: We do NOT normalize Time, matching your Debugger success.
    norm_text = torch.nn.functional.normalize(batch["text_vec"], p=2, dim=1)
    norm_image = torch.nn.functional.normalize(batch["image_vec"], p=2, dim=1)
    norm_time = torch.nn.functional.normalize(batch["time_vec"], p=2, dim=1)
    
    results = manager.model(
        user_idx=batch["user_idx"],
        item_idx=batch["item_idx"],
        scene_idx=batch["scene_idx"],
        text_vec=norm_text,   # PASSING NORMALIZED
        image_vec=norm_image, # PASSING NORMALIZED
        time_vec=norm_time, # PASSING NORMALIZED
        item_count=batch["item_count"]
    )
    
    prob = torch.sigmoid(results).item()
    attn = manager.model.get_last_fusion_attention()
    meta_attn = manager.model.get_last_meta_attention()
    gates = manager.model.get_last_gates()

with left_col:
    st.subheader("High-Level Campaign Details")
    
    # Get the specific row from the DataFrame for static metadata
    selected_row = demo_df.iloc[selected_index]

    with st.container(border=True):
        st.markdown(f"**Item ID:** `{selected_row['item_id']}`")
        st.markdown(f"**User ID:** `{selected_row['user_id']}`")
        st.markdown(f"**Time:** `{selected_row.get('timestamp', case_data.get('timestamp', 'Unknown'))}`")
        
        st.markdown("### 🖼️ Ad Creative (Neural View)")
        st.caption("Since raw images are private/large, we visualize the **ViT Embedding** (What the AI sees).")
        
        # VISUALIZE THE EMBEDDING
        # NEW INTERACTIVE IMAGE COMPONENT
        st.markdown("### 🖼️ Ad Creative")
        
        # 1. View Mode Switcher
        view_mode = st.radio(
            "Visual Mode",
            ["Human Vision", "Neural View"],
            horizontal=True,
            label_visibility="collapsed"
        )

        # 2. Conditional Rendering
        if view_mode == "Human Vision":
            # Retrieve path from store
            item_id_str = str(selected_row['item_id'])
            img_path = manager.store.get_image_path(item_id_str)
            
            if os.path.exists(img_path):
                st.image(img_path, caption=f"Raw Input: {item_id_str}.png", use_container_width=True)
            else:
                # Fallback if image is missing
                st.warning(f"⚠️ Image not found for Item {item_id_str}")
                st.caption(f"Checked path: `{img_path}`")
                st.info("Ensure the 'images' folder exists in your artifacts directory.")

        else:
            # NEURAL VIEW (Heatmap)
            st.caption("ViT Embedding (What the AI sees - 768 dim)")
            
            img_tensor = batch['image_vec'].squeeze().cpu().numpy()
            
            if img_tensor.shape[0] == 768:
                img_grid = img_tensor.reshape(24, 32)
                fig_img = px.imshow(
                    img_grid, 
                    color_continuous_scale='Magma',
                    aspect="auto",
                    labels=dict(x="Feature Dim X", y="Feature Dim Y", color="Activation")
                )
                fig_img.update_layout(
                    margin=dict(l=0, r=0, t=0, b=0),
                    height=300,
                    coloraxis_showscale=False 
                )
                fig_img.update_xaxes(showticklabels=False)
                fig_img.update_yaxes(showticklabels=False)
                st.plotly_chart(fig_img, use_container_width=True)
            else:
                st.warning(f"Embedding shape {img_tensor.shape} not standard (768). Cannot visualize.")

        st.markdown("### 📝 Text Context")
        txt_tensor = norm_text.squeeze().cpu().numpy()
        st.bar_chart(txt_tensor[:50]) 
        st.caption(f"BERT Text Embedding (First 50/768 dims). Mean Activation: {txt_tensor.mean():.4f}")

with right_col:
    st.subheader("🚀 Model Prediction")
    
    # GAUGE CHART 
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prob * 100, 
        title = {'text': "Predicted CTR %"},
        gauge = {
            'axis': {'range': [0, 30]}, 
            'bar': {'color': "#2e86c1"},
            'steps': [
                {'range': [0, 5], 'color': "#f9e79f"},   
                {'range': [5, 15], 'color': "#abebc6"},  
                {'range': [15, 30], 'color': "#82e0aa"}  
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 25
            }
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
    if force_cold:
        st.warning("⚠️ **Cold-Start Active:** The 'id' weight below represents the **Synthetic Meta-Embedding**, not the historical ID!")

    # Convert attention dict to DF
    attn_data = {k: v.item() for k, v in attn.items()}
    df_attn = pd.DataFrame([attn_data])
    
    fig_attn = px.bar(df_attn.T, orientation='h', 
                      labels={'value': 'Attention Weight', 'index': 'Modality'},
                      title="Fusion Layer Attention Weights")
    fig_attn.update_layout(showlegend=False)
    st.plotly_chart(fig_attn, use_container_width=True)

# --- TAB 2: META-EMBEDDING (Standard Normalization) ---
with tab2:
    is_meta_active = force_cold or (batch['item_count'].item() == 0)

    if not is_meta_active:
        st.info("🔥 **Warm Start Active**")
        st.markdown("To see how the **Meta-Generator** would synthesize a new ID for this item, toggle **❄️ Force Cold-Start**.")
    else:
        st.markdown("How was the **Cold-Start Item ID** generated?")
        st.markdown("The Meta-Generator synthesized a proxy ID using this modality mix:")
        
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
        
        with st.expander("View Raw Weights"):
             st.json(meta_raw)

# --- TAB 3: EXPERT GATES ---
with tab3:
    st.markdown("How did the **Scenario-Aware MMoE** route this sample?")
    
    gate_vals = gates.cpu().numpy().flatten()
    # Create labels Expert 0, Expert 1...
    expert_labels = [f'Expert {i}' for i in range(len(gate_vals))]
    
    df_gates = pd.DataFrame({'Expert': expert_labels, 'Weight': gate_vals})
    
    fig_gates = px.bar(df_gates, x='Expert', y='Weight', 
                       title=f"Expert Activation for Scenario: {override_scene}")
    st.plotly_chart(fig_gates, use_container_width=True)


# DEVELOPER INSPECTOR
with st.expander("🛠️ Developer / Tensor Inspector"):
    st.markdown("### Raw Input Tensors (Live Simulation)")
    st.write("These are the actual vectors flowing into the PyTorch model:")
    
    t_col1, t_col2 = st.columns(2)
    with t_col1:
        st.code(f"User ID Idx: {batch['user_idx'].item()}")
        st.code(f"Item ID Idx: {batch['item_idx'].item()} (Count: {batch['item_count'].item()})")
        st.code(f"Scene Idx: {batch['scene_idx'].item()} (Override Active)")
    with t_col2:
        st.code(f"Text Emb Shape: {list(batch['text_vec'].shape)}")
        st.code(f"Image Emb Shape: {list(batch['image_vec'].shape)}")
        st.code(f"Time Emb Shape: {list(batch['time_vec'].shape)}")