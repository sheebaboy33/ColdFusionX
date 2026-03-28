import streamlit as st

def render_top_nav(page_name=None):
    """
    Renders the custom Top Navigation bar and handles CSS for all pages.
    """
    
    # 1. GLOBAL & PAGE-SPECIFIC CSS
    st.markdown("""
    <style>
        /* Hide Default Streamlit Sidebar and Adjust Top Padding */
        [data-testid="stSidebarNav"] { display: none !important; }
        .block-container { padding-top: 2rem !important; }
        
        /* NAV LINK UI/UX UPGRADES             */
        /* Base styling for the links to look like clean, clickable pills */
        [data-testid="stPageLink-NavLink"] {
            background-color: transparent !important;
            border: 1px solid transparent !important;
            border-radius: 8px !important;
            padding: 6px 12px !important;
            transition: all 0.2s ease-in-out !important;
        }
        
        /* Hover state */
        [data-testid="stPageLink-NavLink"]:hover {
            background-color: rgba(255, 255, 255, 0.05) !important;
            border-color: rgba(255, 255, 255, 0.1) !important;
        }
        
        /* Active Page State */
        [data-testid="stPageLink-NavLink"][aria-current="page"] {
            background-color: rgba(6, 182, 212, 0.1) !important; /* Subtle Cyan background */
            border-color: rgba(6, 182, 212, 0.3) !important;
        }
        [data-testid="stPageLink-NavLink"][aria-current="page"] p {
            color: #06B6D4 !important; /* Cyan text */
            font-weight: 700 !important;
        }
    </style>
    """, unsafe_allow_html=True)
    

    if page_name != "campaign_optimiser":
        st.markdown("""
        <style>
            [data-testid="stHeader"] { display: none !important; }
        </style>
        """, unsafe_allow_html=True)

    # 2. NAVIGATION BAR LAYOUT
    nav_items = [
        {"name": "home", "label": "Home", "path": "app_streamlit.py"},
        {"name": "performance", "label": "Performance", "path": "pages/performance.py"},
        {"name": "campaign_optimiser", "label": "Simulator", "path": "pages/campaign_optimiser.py"},
        {"name": "model_xray", "label": "Model X-Ray", "path": "pages/model_xray.py"}
    ]

    # Native Streamlit Columns (Stable & Safe)
    cols = st.columns([2.5, 0.5, 1, 1.2, 1.2, 1.2])

    with st.container():
        # Render the Logo
        with cols[0]:
            if page_name != "home":
                st.markdown("### ❄️ ColdFusionX")
            else:
                st.empty() # Leaves the space blank on the home page

        # Render the Links
        for i, item in enumerate(nav_items):
            with cols[i + 2]:
                st.page_link(item["path"], label=item["label"])
                
        st.markdown("---")