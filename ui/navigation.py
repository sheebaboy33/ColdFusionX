import streamlit as st

def render_top_nav(page_name=None):
    """
    Renders the custom Top Navigation bar and handles CSS for all pages.
    """
    
    # SHARED CSS (APPLIES TO ALL PAGES)
    common_css = """
    <style>
        /* Hides the default "Pages" menu in the sidebar */
        [data-testid="stSidebarNav"] {
            display: none;
        }
        
        /* Adjust main content padding so titles aren't cut off */
        .block-container {
            padding-top: 4rem !important; 
        }
    </style>
    """
    st.markdown(common_css, unsafe_allow_html=True)
    # --- PAGE SPECIFIC CSS
    if page_name == "ad_simulator":
        # No extra CSS needed here, standard header visibility is fine.
        pass 
        
    else:
        # For Home & Performance: HIDE the top header decoration (Hamburger/Running man)
        # This gives the "Clean App" look
        st.markdown("""
        <style>
            [data-testid="stHeader"] {
                display: none;
            }
            /* Start content higher up since header is gone */
            .block-container {
                padding-top: 2rem !important;
            }
        </style>
        """, unsafe_allow_html=True)

    # --- NAVIGATION BAR CONTENT ---
    # Determine the active page for styling (optional highlight logic can be added here)
    if page_name in ["performance", "ad_simulator"]:
        with st.container():
            # Adjust column weights to center the links effectively
            # col1: Logo, col3-5: Links
            col1, _, col3, col4, col5 = st.columns([2, 0.5, 1, 1, 1])
            with col1:
                st.markdown("### ❄️ ColdFusionX")
            with col3:
                st.page_link("app_streamlit.py", label="Home", icon="🏠")
            with col4:
                st.page_link("pages/performance.py", label="Performance", icon="📈")
            with col5:
                st.page_link("pages/ad_simulator.py", label="Simulator", icon="🚀")
                
            st.markdown("---")

    elif page_name == "home":
        _, _, col3, col4, col5 = st.columns([2, 0.5, 1, 1, 1])

        with st.container():
            with col3:
                st.page_link("app_streamlit.py", label="Home", icon="🏠")
            with col4:
                st.page_link("pages/performance.py", label="Performance", icon="📈")
            with col5:
                st.page_link("pages/ad_simulator.py", label="Simulator", icon="🚀")
                
            st.markdown("---")
    else:
        st.error(f"Unknown page_name: '{page_name}'. Check your render_top_nav call.")