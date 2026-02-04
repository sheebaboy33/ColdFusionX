import streamlit as st
import plotly.graph_objects as go
import numpy as np
from navigation import render_top_nav

# PAGE CONFIG
st.set_page_config(page_title="ColdFusionX | Model Performance", page_icon="❄️", layout="wide")

# RENDER TOP NAV
render_top_nav()