import streamlit as st

# The ColdFusionX Color Palette
COLORS = {
    "background": "#1d3352",      # Deep dark navy (almost black)
    "card_bg": "#697e9d",         # Slate blue for cards
    "primary": "#3B82F6",         # Royal Blue
    "accent": "#06B6D4",          # Icy Cyan (for highlights)
    "text_main": "#F8FAFC",       # Crisp white
    "text_muted": "#D0D4DA",      # Slate gray
    "danger": "#EF4444",          # Red for performance drops
    "success": "#10B981"          # Green for wins
}

def load_custom_css():
    """Injects global custom CSS into the Streamlit app."""
    
    custom_css = f"""
    <style>
        /* 1. Global App Background */
        .stApp {{
            background-color: {COLORS['background']};
            color: {COLORS['text_main']};
        }}

        /* 2. Hide Streamlit Branding */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{visibility: hidden;}} /* Hides the top colored bar */

        /* 3. Reduce Top Padding */
        .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
        }}

        /* 4. Interactive Metric Cards (Breaks the default Streamlit look) */
        [data-testid="stMetric"] {{
            background-color: {COLORS['card_bg']};
            border-radius: 12px;
            padding: 15px 20px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2), 0 2px 4px -1px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(59, 130, 246, 0.2);
            transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
        }}
        
        /* Hover effect for interactivity */
        [data-testid="stMetric"]:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
            border-color: {COLORS['accent']};
        }}

        [data-testid="stMetricLabel"] {{
            color: {COLORS['text_muted']} !important;
            font-size: 1.1rem !important;
            font-weight: 500 !important;
        }}

        [data-testid="stMetricValue"] {{
            color: {COLORS['text_main']} !important;
            font-size: 2.2rem !important;
            font-weight: 700 !important;
        }}

        /* 5. Custom Styling for Dataframes */
        [data-testid="stDataFrame"] {{
            border-radius: 10px;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}

        /* 6. Info and Success Boxes */
        div.stAlert > div {{
            border-radius: 10px;
            border: none;
        }}
        
        /* 7. Typography Adjustments */
        h1, h2, h3 {{
            color: {COLORS['text_main']} !important;
            font-family: 'Inter', sans-serif;
            letter-spacing: -0.02em;
        }}
        
        h1 {{
            background: -webkit-linear-gradient(45deg, {COLORS['primary']}, {COLORS['accent']});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
        }}
        
        /* 8. Horizontal Dividers */
        hr {{
            border-color: rgba(255, 255, 255, 0.1) !important;
        }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)