import streamlit as st

st.set_page_config(
    page_title="OrbitResearch",
    page_icon="🛰️",
    layout="wide"
)

# Navigation
pages = {
    "🛰️ OrbitResearch": [
        st.Page("pages/1_Chat.py", title="Chat", icon="💬"),
        st.Page("pages/2_History.py", title="History", icon="🕘"),
    ]
}

pg = st.navigation(pages)
pg.run()
