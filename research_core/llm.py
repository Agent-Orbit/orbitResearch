from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from utils.apiKeys import GROQ_API_KEY
from langchain_ollama import ChatOllama

class llmHandler:

    def __init__(self, name: str):
        
        self.all_models = ["meta-llama/llama-4-scout-17b-16e-instruct","qwen2.5:7b","llama-3.3-70b-versatile","llama-3.1-8b-instant"]
        self.groq_models = ["meta-llama/llama-4-scout-17b-16e-instruct","llama-3.3-70b-versatile","llama-3.1-8b-instant"]
        self.local_models = ["qwen2.5:7b"]
        self.llm = self.getLLM(name)

    def getLLM(self, name: str) -> ChatGroq:

        if name in self.groq_models:

            return ChatGroq(model=name, api_key=GROQ_API_KEY)

        elif name in self.local_models:

            return ChatOllama(model=name)

        raise ValueError(f"Model '{name}' is not supported. Available models: {self.groq_models}")

    def queryLLM(self, prompt: str) -> str:

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content