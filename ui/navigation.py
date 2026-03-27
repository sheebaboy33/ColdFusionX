import streamlit as st

def render_top_nav(page_name=None):
    """
    Renders the custom Top Navigation bar and handles CSS for all pages.
    """
    
    # ==========================================
    # 1. GLOBAL & PAGE-SPECIFIC CSS
    # ==========================================
    # Hides the default Streamlit sidebar and adjusts padding
    common_css = """
    <style>
        [data-testid="stSidebarNav"] { display: none; }
        .block-container { padding-top: 3rem !important; }
    </style>
    """
    st.markdown(common_css, unsafe_allow_html=True)
    
    # Hide the top decoration (Hamburger/Running man) for a cleaner look,
    # unless we are on the simulator page which might need standard headers.
    if page_name != "campaign_optimiser":
        st.markdown("""
        <style>
            [data-testid="stHeader"] { display: none; }
            .block-container { padding-top: 2rem !important; }
        </style>
        """, unsafe_allow_html=True)

    # ==========================================
    # 2. NAVIGATION BAR LAYOUT
    # ==========================================
    # Define your app structure here. Easy to update later!
    nav_items = [
        {"name": "home", "label": "Home", "path": "app_streamlit.py"},
        {"name": "performance", "label": "Performance", "path": "pages/performance.py"},
        {"name": "campaign_optimiser", "label": "Simulator", "path": "pages/campaign_optimiser.py"},
        {"name": "model_xray", "label": "Model X-Ray", "path": "pages/model_xray.py"}
    ]

    # Adjusted column weights to perfectly fit 4 links + Logo
    # col1 (Logo), col2 (Spacer), col3-6 (Links)
    cols = st.columns([2.5, 0.5, 1, 1.2, 1.2, 1.2])

    with st.container():
        # Render the Logo (Hide it on the home page)
        with cols[0]:
            if page_name != "home":
                st.markdown("### ❄️ ColdFusionX")
            else:
                st.empty() # Leaves the space blank on the home page

        # Render the Links Dynamically
        for i, item in enumerate(nav_items):
            with cols[i + 2]:
                # UI UX Upgrade: Only show the emoji icon if it is the ACTIVE page. 
                # This acts as a visual "You are here" indicator.
                
                st.page_link(
                    item["path"], 
                    label=item["label"], 
                )
                
        st.markdown("---")