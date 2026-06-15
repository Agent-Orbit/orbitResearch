import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

def getAPI(name):

    try:
        return st.secrets[name]
    except:
        return os.getenv(name)

GROQ_API_KEY = getAPI("GROQ_API_KEY")
TAVILY_API_KEY = getAPI("TAVILY_API_KEY")
GITHUB_TOKEN = getAPI("GITHUB_TOKEN")