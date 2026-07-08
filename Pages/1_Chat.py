import streamlit as st

from research_core.core import CoreResearch
from research_core.llm import llmHandler

st.title("💬 OrbitResearch Chat")
st.caption("Ask a question and OrbitResearch will research it across arXiv, GitHub, the web, Wikipedia, and YouTube.")

st.markdown(
    """
    <style>
    .stChatMessage { border-radius: 12px; }
    div[data-testid="stStatusWidget"] { border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

WELCOME = "Hi! Ask me anything and I'll research it for you."

# ---------------------------------------------------------------- sidebar

with st.sidebar:

    st.subheader("Settings")

    model_name = st.selectbox(
        "Model",
        options=llmHandler.ALL_MODELS,
        format_func=lambda m: llmHandler.MODEL_LABELS.get(m, m),
        key="model_select",
    )

    st.divider()

    if st.button("🧹 New conversation", use_container_width=True):

        st.session_state.messages = [{"role": "assistant", "content": WELCOME}]
        agent = st.session_state.get("agent")

        if agent is not None:

            agent.reset()

        st.rerun()

    st.caption(
        "Groq models need `GROQ_API_KEY` set (via `.env` or Streamlit secrets). "
        "The local Qwen model needs an Ollama server running on this machine."
    )

# ------------------------------------------------------------ state setup

if "messages" not in st.session_state:

    st.session_state.messages = [{"role": "assistant", "content": WELCOME}]

if "agent" not in st.session_state or st.session_state.get("agent_model") != model_name:

    try:

        st.session_state.agent = CoreResearch(model_name)
        st.session_state.agent_model = model_name
        st.session_state.agent_error = None

    except Exception as e:

        st.session_state.agent = None
        st.session_state.agent_model = model_name
        st.session_state.agent_error = str(e)

# --------------------------------------------------------------- history

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])

        if msg.get("sources"):

            st.caption("🔎 Sources: " + ", ".join(msg["sources"]))

# ------------------------------------------------------------------ input

prompt = st.chat_input("Ask OrbitResearch something...")

if prompt:

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):

        st.markdown(prompt)

    with st.chat_message("assistant"):

        if st.session_state.agent is None:

            error_text = (
                f"OrbitResearch couldn't start the **{llmHandler.MODEL_LABELS.get(model_name, model_name)}** "
                f"model: {st.session_state.agent_error}"
            )

            st.error(error_text)
            st.session_state.messages.append({"role": "assistant", "content": error_text})

        else:

            result = None

            with st.spinner("Researching..."):

                try:

                    result = st.session_state.agent.run(prompt)

                except Exception as e:

                    st.error(f"Research failed: {e}")

            if result:

                st.markdown(result["summary"])

                if result["sources_used"]:

                    st.caption("🔎 Sources: " + ", ".join(result["sources_used"]))

                meta = result.get("metadata", {})
                if meta.get("elapsed_seconds") is not None:
                    
                    st.caption(f"⏱️ {meta['elapsed_seconds']}s · retries: {meta.get('retries', 0)}")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["summary"],
                    "sources": result["sources_used"],
                })