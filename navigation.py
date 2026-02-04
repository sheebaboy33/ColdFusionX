import streamlit as st


def render_top_nav():
    # CSS to hide the default top header and adjust padding
    st.markdown("""
        <style>
            [data-testid="stHeader"] {display: none;}
            .block-container {padding-top: 2rem;}
        </style>
    """, unsafe_allow_html=True)

    # The Navbar Container
    with st.container():
        col1, col2, col3, col4, col5 = st.columns([1, 2, 1, 1, 1])
        
        with col1:
            st.markdown("### ❄️ ColdFusionX")
            
        # Navigation Links using Streamlit's native page_link
        with col3:
            st.page_link("app_streamlit.py", label="Home", icon="🏠")
        with col4:
            st.page_link("pages/performance.py", label="Model Performance", icon="📈")
        with col5:
            st.page_link("pages/ad_simulator.py", label="Campaign Simulator", icon="🚀")
            
        st.markdown("---")