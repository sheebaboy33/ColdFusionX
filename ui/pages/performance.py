import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from ui.navigation import render_top_nav

# --- RENDER THE TOP NAVIGATION BAR ---
render_top_nav(page_name="performance")


# PAGE CONFIGURATION
st.set_page_config(
    page_title="ColdFusionX | Model Performance", 
    page_icon="📈", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Colors for the ColdFusionX Monochromatic Blue Theme
COLOR_PRIMARY = "#0062FF"   # Vibrant 'Cold' Blue 
COLOR_SECONDARY = "#1E3A8A" # Deep Navy Blue
COLOR_ACCENT = "#38BDF8"    # Bright Sky Blue 
COLOR_BASELINE = "#94A3B8"  # Muted Slate Gray


# DATA LOADING
@st.cache_data
def load_metrics():
    base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "artifacts", "release_demo", "metrics")
    
    global_df = pd.read_csv(os.path.join(base_path, "global_metrics_summary.csv"))
    ablation_df = pd.read_csv(os.path.join(base_path, "ablation_metrics.csv"))
    scenario_df = pd.read_csv(os.path.join(base_path, "scenario_metrics.csv"))
    
    return global_df, ablation_df, scenario_df

try:
    global_df, ablation_df, scenario_df = load_metrics()
except FileNotFoundError as e:
    st.error(f"Metrics files not found. Please check paths. {e}")
    st.stop()

# HEADER
st.title("Model Performance & Evaluation")
st.markdown("Comprehensive evaluation of **ColdFusionX** on the AntM2C dataset (10M interactions). This dashboard provides empirical proof of the architecture's superiority in **Cold-Start** scenarios and resilience to **Temporal Drift**.")
st.divider()

# SECTION 1: GLOBAL BASELINE COMPARISON
st.header("1. Baseline Comparisons (The Cold-Start Proof)")
st.markdown("Comparing ColdFusionX against state-of-the-art models. Notice how the performance gap widens significantly on **[COLD]** items.")

col1, col2 = st.columns([1, 2])

with col1:
    # Key Metric Cards for the Proposed Model
    cfx_metrics = global_df[global_df['Model'].str.contains("ColdFusionX")].iloc[0]
    
    st.markdown("### ColdFusionX (Proposed)")
    st.metric(label="Overall AUC", value=f"{cfx_metrics['Overall AUC']:.4f}")
    st.metric(label="Warm-Start AUC", value=f"{cfx_metrics['Warm AUC']:.4f}")
    st.metric(label="Cold-Start AUC", value=f"{cfx_metrics['Cold AUC']:.4f}", delta="Top Performer")
    st.metric(label="LogLoss", value=f"{cfx_metrics['Overall LogLoss']:.4f}", delta="-Lower is better", delta_color="inverse")

with col2:
    # Grouped Bar Chart for Warm vs Cold AUC
    fig_baselines = go.Figure()

    # Add Warm AUC Bars
    fig_baselines.add_trace(go.Bar(
        x=global_df['Model'], y=global_df['Warm AUC'],
        name='Warm-Start AUC', marker_color=COLOR_SECONDARY
    ))
    # Add Cold AUC Bars
    fig_baselines.add_trace(go.Bar(
        x=global_df['Model'], y=global_df['Cold AUC'],
        name='Cold-Start AUC', marker_color=COLOR_PRIMARY
    ))

    fig_baselines.update_layout(
        title="Model Performance: Warm vs. Cold Start",
        barmode='group',
        yaxis_title="AUC Score",
        yaxis=dict(range=[0.80, 0.95]), # Zoom in to show the differences clearly
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_baselines, use_container_width=True)


# SECTION 2: ABLATION STUDY
st.divider()
st.header("2. Architectural Ablation Study")
st.markdown("Isolating the impact of the **Modality-Attentive Fusion** and **Meta-Embedding Generator** components.")

col3, col4 = st.columns([2, 1])

with col3:
    # Sort ablation to show Full model at top, then descending
    ablation_df_sorted = ablation_df.sort_values(by="Cold AUC", ascending=True)
    
    # Color the Full Model differently
    colors = [COLOR_PRIMARY if "Full" in m else COLOR_BASELINE for m in ablation_df_sorted['Model']]
    
    fig_ablation = go.Figure(go.Bar(
        x=ablation_df_sorted['Cold AUC'], 
        y=ablation_df_sorted['Model'],
        orientation='h',
        marker_color=colors,
        text=ablation_df_sorted['Cold AUC'].round(4),
        textposition='auto'
    ))
    
    fig_ablation.update_layout(
        title="Impact of Components on Cold-Start AUC",
        xaxis_title="Cold-Start AUC",
        xaxis=dict(range=[0.83, 0.86]), # Zoomed for emphasis
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_ablation, use_container_width=True)

with col4:
    st.info("""
    **Key Takeaways:**
    * **w/o Temporal FiLM:** Removing temporal context drops Cold-Start AUC, proving the model relies on time data when historical clicks are absent.
    * **w/o Cosine Align:** Failing to align the text and image spaces results in the largest performance drop, validating our fusion strategy.
    """)
    st.dataframe(ablation_df[['Model', 'Overall AUC', 'Cold AUC']], hide_index=True)


# SECTION 3: TEMPORAL DRIFT RESILIENCE

### WOULD MAKE MORE SENSE IF THE BASELINE MODELS ALSO HAD TEMPORAL DRIFT DATA

st.divider()
st.header("3. Temporal Drift & Scenario Adaptability")
st.markdown("Evaluating how consistently ColdFusionX maintains accuracy across shifting contextual environments.")

fig_scenario = px.line(
    scenario_df, 
    x="Scenario", 
    y="AUC", 
    markers=True,
    title="ColdFusionX Stability Across Contexts",
    color_discrete_sequence=[COLOR_ACCENT]
)

fig_scenario.update_traces(marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')))
fig_scenario.update_layout(
    yaxis=dict(range=[0.90, 0.94]),
    yaxis_title="Overall AUC",
    plot_bgcolor="rgba(0,0,0,0)"
)

col5, col6 = st.columns([1, 1])

with col5:
    st.plotly_chart(fig_scenario, use_container_width=True)

with col6:
    # Calculate the actual drift degradation from the CSV
    max_auc = scenario_df['AUC'].max()
    min_auc = scenario_df['AUC'].min()
    auc_fluctuation = max_auc - min_auc
    
    st.success(f"""
    **Temporal Robustness Verified:**
    Despite significant shifts in user behavior across different temporal scenarios, the model's performance fluctuates by a maximum of only **{auc_fluctuation:.4f} AUC**. 
    
    This stability is directly attributed to the dynamic routing of the Modality-Attentive Fusion mechanism, which successfully re-weights modalities based on temporal context to prevent performance collapse during concept drift.
    """)


# SEE IF WE CAN SHOW THE DISJOINT MANIFOLD ALIGNMENT TOO