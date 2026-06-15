import streamlit as st

st.title("🕘 History")
st.caption("Your past research conversations.")

sample_sessions = [
    "Impact of inflation on GDP growth",
    "Comparing trade policies of South Asia",
    "Renewable energy adoption trends",
]

for session in sample_sessions:
    st.button(session)
