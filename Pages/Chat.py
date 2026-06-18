import streamlit as st

st.title("💬 OrbitResearch Chat")
st.caption("Ask a question and OrbitResearch will research it for you.")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Ask me anything and I'll research it for you."}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask OrbitResearch something...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = "This is a placeholder response. The research agent is not connected yet."
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
