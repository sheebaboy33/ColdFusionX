import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
from ui.navigation import render_top_nav
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.theme import load_custom_css, COLORS 

# RENDER THE TOP NAVIGATION BAR
render_top_nav(page_name="performance")

# LOAD CUSTOM CSS FOR THEME
load_custom_css()   

# PAGE CONFIGURATION
st.set_page_config(
    page_title="ColdFusionX | Model Performance", 
    page_icon="📈", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# DATA LOADING
@st.cache_data
def load_metrics():
    base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "artifacts", "release_demo", "metrics")
    
    global_df = pd.read_csv(os.path.join(base_path, "global_metrics_summary.csv"))
    ablation_df = pd.read_csv(os.path.join(base_path, "ablation_metrics.csv"))
    
    # Try to load the ALL models scenario file
    try:
        scenario_df = pd.read_csv(os.path.join(base_path, "scenario_metrics_all.csv"))
    except FileNotFoundError:
        # Fallback
        scenario_df = pd.read_csv(os.path.join(base_path, "scenario_metrics.csv"))
        
    return global_df, ablation_df, scenario_df

try:
    global_df, ablation_df, scenario_df = load_metrics()
except FileNotFoundError as e:
    st.error(f"Metrics files not found. Ensure artifacts are downloaded. {e}")
    st.stop()


# HEADER
st.title("Model Performance & Evaluation")
st.markdown("**ColdFusionX** evaluation on 3.7M AntM2C interactions. Demonstrates state-of-the-art **Cold-Start** performance and **Temporal Drift** resilience.")
st.divider()


# SECTION 1: GLOBAL BASELINE COMPARISON
st.header("1. Baseline Comparisons (The Cold-Start Proof)")
st.markdown("While baselines succeed on *Warm* items, true performance is measured by the **drop penalty** on *Cold* items. A smaller penalty indicates superior multi-modal generalization.")
# Calculate the Delta (Performance Drop)
global_df['AUC Drop'] = global_df['Warm AUC'] - global_df['Cold AUC']
global_df = global_df.sort_values('AUC Drop', ascending=False)

# Adjusting columns so the chart gets slightly more room (1:2 ratio)
col1, col2 = st.columns([0.75, 2])

with col1:
    # Restored: Key Metric Cards for the Proposed Model on the left
    cfx_metrics = global_df[global_df['Model'].str.contains("ColdFusionX")].iloc[0]
    
    st.markdown("### ColdFusionX (Proposed)")
    st.metric(label="Overall AUC", value=f"{cfx_metrics['Overall AUC']:.4f}")
    st.metric(label="Warm-Start AUC", value=f"{cfx_metrics['Warm AUC']:.4f}")
    st.metric(label="Cold-Start AUC", value=f"{cfx_metrics['Cold AUC']:.4f}", delta="Top Performer")
    st.metric(label="LogLoss", value=f"{cfx_metrics['Overall LogLoss']:.4f}", delta="-Lower is better", delta_color="inverse")

with col2:
    colors = [COLORS['primary'] if "ColdFusionX" in m else COLORS['text_muted'] for m in global_df['Model']]
    
    fig_delta = go.Figure(go.Bar(
        x=global_df['AUC Drop'], 
        y=global_df['Model'],
        orientation='h',
        marker_color=colors,
        text=["Drop: " + str(round(val, 4)) for val in global_df['AUC Drop']],
        textposition='auto',
        textfont=dict(
            # White text for the blue bar, black for the light gray bars
            color=["white" if "ColdFusionX" in m else "black" for m in global_df['Model']],
            family="Arial Bold",
            size=11 # Slightly smaller for mobile screens
        )
    ))
    
    fig_delta.update_layout(
        # Use a dynamic height based on the number of bars
        height=len(global_df) * 100, 
        title={
            'text': "Performance Penalty: Warm vs. Cold Items",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="AUC Drop (Warm - Cold)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)", 
        font=dict(color="#94A3B8"),
        # Reduced margins for responsiveness
        margin=dict(l=10, r=10, t=80, b=40),
        bargap=0.3,
        # Ensure the chart stretches to the container
        autosize=True 
    )
    
    fig_delta.update_yaxes(automargin=True)
    fig_delta.update_xaxes(showgrid=False) # Clean up mobile view
    
    # use_container_width=True is key for responsiveness
    st.plotly_chart(fig_delta, use_container_width=True, config={'displayModeBar': False})

# The Detailed Baseline Table below the columns
st.markdown("#### Comprehensive Baseline Metrics Table")
display_df = global_df[['Model', 'Overall AUC', 'Warm AUC', 'Cold AUC', 'Overall LogLoss']].copy()
st.dataframe(
    display_df.style.set_properties(**{'background-color': 'rgba(72,88,113,0.25)'})
                    .highlight_max(subset=['Overall AUC', 'Warm AUC', 'Cold AUC'], color='rgba(139, 92, 246, 0.3)')
                    .highlight_min(subset=['Overall LogLoss'], color='rgba(139, 92, 246, 0.3)'),
    use_container_width=True,
    hide_index=True,
)


# SECTION 2: ABLATION STUDY (WATERFALL)
st.divider()
st.header("2. Architectural Ablation Study")
st.markdown("Quantifying the **Cold-Start AUC contribution** of each ColdFusionX intervention modules.")

col3, col4 = st.columns([1.75, 1])

with col3:
    # Calculate contributions mathematically
    full_auc = ablation_df[ablation_df['Model'].str.contains("Full")]['Cold AUC'].values[0]
    
    try:
        no_align = ablation_df[ablation_df['Model'].str.contains("Align")]['Cold AUC'].values[0]
        no_film = ablation_df[ablation_df['Model'].str.contains("FiLM")]['Cold AUC'].values[0]
        no_sam = ablation_df[ablation_df['Model'].str.contains("SAM")]['Cold AUC'].values[0]
        
        contrib_align = full_auc - no_align
        contrib_film = full_auc - no_film
        contrib_sam = full_auc - no_sam
    except IndexError:
        st.error("Ablation data is missing expected entries. Ensure the ablation_metrics.csv contains rows for 'No Align', 'No FiLM', and 'No SAM'.")
        st.stop()

    base_auc = full_auc - (contrib_align + contrib_film + contrib_sam)

    fig_waterfall = go.Figure(go.Waterfall(
        name="Ablation", orientation="v",
        measure=["absolute", "relative", "relative", "relative", "total"],
        x=["Base Architecture", "+ Cosine Alignment", "+ Temporal FiLM", "+ SAM Optimization", "ColdFusionX (Full)"],
        textposition="outside",
        text=[f"{base_auc:.4f}", f"+{contrib_align:.4f}", f"+{contrib_film:.4f}", f"+{contrib_sam:.4f}", f"{full_auc:.4f}"],
        y=[base_auc, contrib_align, contrib_film, contrib_sam, full_auc],
        connector={"line": {"color": "rgb(255, 255, 255)"}},
        decreasing={"marker": {"color": COLORS['danger']}},
        increasing={"marker": {"color": COLORS['accent']}},
        totals={"marker": {"color": COLORS['primary']}}
    ))

    fig_waterfall.update_layout(
        title="Component Contributions to Cold-Start AUC",
        yaxis_title="Cold-Start AUC",
        yaxis=dict(range=[0.82, 0.86]), # Zoomed in to highlight the steps
        plot_bgcolor="rgba(0, 0,0,0)",
        paper_bgcolor="rgba(72,88,113,0.25)", 
        font=dict(color="#FFFFFF"),
    )

    st.plotly_chart(fig_waterfall, use_container_width=True)

with col4:
    # text white for readability
    st.markdown("""
    <style>
        div[data-testid="stAlert"] * {
            color: #F8FAFC !important;
        }
    </style>
    """, unsafe_allow_html=True)
    st.info("""
    **Key Contributions:**
    * **SAM Optimization:** Adaptive routing between the collaborative IDs and multi-modal proxy embeddings.
    * **Cosine Alignment:** Forcing the meta-generator to mimic the established collaborative space.
    * **Temporal FiLM:** Modulating visual and textual features to capture cyclical micro-drift successfully.
    """)
    st.dataframe(
        ablation_df[['Model', 'Cold AUC', 'Overall AUC']].style.set_properties(**{'background-color': 'rgba(72,88,113,0.25)'}),
        hide_index=True,
        use_container_width=True
    )


# SECTION 3: TEMPORAL DRIFT RESILIENCE
st.divider()
st.header("3. Temporal Drift & Scenario Adaptability")
st.markdown("Evaluating how consistently models maintain accuracy across shifting contextual environments (The 'Scissors Effect').")

# 1. CALCULATE DYNAMIC RANGE (The "Zoom" Fix)
# Find the min and max AUC to crop the chart
padding = 0.01
min_auc = scenario_df['AUC'].min() - padding
max_auc = scenario_df['AUC'].max() + padding

# 2. CREATE THE FIGURE
fig_scenario = px.line(
    scenario_df, 
    x="Scenario", 
    y="AUC", 
    color="Model",
    markers=True,
    # Shorten title for mobile compatibility
    title="Temporal Drift: Weekday vs. Weekend",
    color_discrete_sequence=px.colors.qualitative.Safe # Fallback if specific colors aren't set
)

# Customize markers
fig_scenario.update_traces(
    line=dict(width=3),
    marker=dict(size=10, line=dict(width=1, color='white'))
)

# 3. RESPONSIVE LAYOUT & DYNAMIC ZOOM
fig_scenario.update_layout(
    height=450,
    # THE ZOOM: Crop the Y-axis to only show the relevant area
    yaxis=dict(
        range=[min_auc, max_auc], 
        gridcolor="rgba(255,255,255,0.1)",
        zeroline=False
    ),
    yaxis_title="Scenario AUC",
    xaxis_title="",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    
    # THE MOBILE FIX: Legend moves below the chart to prevent overlapping
    legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.2, 
        xanchor="center",
        x=0.5,
        font=dict(size=10)
    ),
    font=dict(color="#FFFFFF"),
    margin=dict(l=10, r=10, t=50, b=80)
)

# 4. DISPLAY
# Using a container ensures that on mobile (where col5/col6 stack), 
# the text appears logically after the chart.
with st.container():
    st.plotly_chart(fig_scenario, use_container_width=True, config={'displayModeBar': False})
    
    st.success("""
    **The Scissors Effect:**
    Unlike baselines that collapse under behavioral shifts, **ColdFusionX** remains stable—validating the Temporal Drift Adaptation module.
    """)