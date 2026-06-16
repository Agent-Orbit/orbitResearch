from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from utils.apiKeys import GROQ_API_KEY


class llmHandler:

    def __init__(self, name: str):
        
        self.groq_models = ["meta-llama/llama-4-scout-17b-16e-instruct"]
        self.llm = self.getLLM(name)

    def getLLM(self, name: str) -> ChatGroq:

        if name in self.groq_models:

            return ChatGroq(model=name, api_key=GROQ_API_KEY)

        raise ValueError(f"Model '{name}' is not supported. Available models: {self.groq_models}")

    def queryLLM(self, prompt: str) -> str:

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content