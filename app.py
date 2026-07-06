import streamlit as st

st.set_page_config(

    page_title="OrbitResearch",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Navigation

pages = {
    "🛰️ OrbitResearch": [
        st.Page("pages/1_Chat.py", title="Chat", icon="💬"),
    ]
}

pg = st.navigation(pages)
pg.run()