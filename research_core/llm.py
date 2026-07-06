from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from utils.apiKeys import GROQ_API_KEY
from langchain_ollama import ChatOllama


class llmHandler:

    
    GROQ_MODELS = [

        "meta-llama/llama-4-scout-17b-16e-instruct",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
    ]

    LOCAL_MODELS = [
        
        "qwen2.5:7b",
    ]

    ALL_MODELS = GROQ_MODELS + LOCAL_MODELS

    MODEL_LABELS = {

        "meta-llama/llama-4-scout-17b-16e-instruct": "Llama 4 Scout (Groq)",
        "llama-3.3-70b-versatile": "Llama 3.3 70B Versatile (Groq)",
        "llama-3.1-8b-instant": "Llama 3.1 8B Instant (Groq)",
        "qwen2.5:7b": "Qwen2.5 7B (Local / Ollama)",
    }

    def __init__(self, name: str):


        self.all_models = self.ALL_MODELS
        self.groq_models = self.GROQ_MODELS
        self.local_models = self.LOCAL_MODELS

        self.name = name
        self.llm = self.getLLM(name)

    def getLLM(self, name: str):

        if name in self.GROQ_MODELS:

            if not GROQ_API_KEY:

                raise EnvironmentError(
                    "GROQ_API_KEY is not set. Add it to your .env file or Streamlit secrets."
                )

            return ChatGroq(model=name, api_key=GROQ_API_KEY, temperature=0.1)

        if name in self.LOCAL_MODELS:

            return ChatOllama(model=name, temperature=0.1)

        raise ValueError(f"Model '{name}' is not supported. Available models: {self.ALL_MODELS}")

    def queryLLM(self, prompt: str) -> str:

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content