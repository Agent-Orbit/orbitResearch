from tools import webSearch,githubSearch,wikiSearch,arxivSearch,ytSearch
from research_core import core
from langchain_ollama import ChatOllama

# print(webSearch.searchTool.invoke({"query": "what is 3*3 + 3"}))
#print(githubSearch.githubSearchTool.invoke({"query": "langchain"}))
#print(webSearch.extractWeb.invoke({"url": "https://github.com/Agent-Orbit/PDF-Analyser"}))
#print(wikiSearch.wikipediaSearch.invoke({"query": "Lionel_Messi"}))
#print(arxivSearch.arxivSearch.invoke({"query": "Deep Learning"}))
#print(ytSearch.youtubeVideoDetails.invoke({"url": "https://www.youtube.com/watch?v=dtCS0WGZH4k"}))

c = core.CoreResearch("meta-llama/llama-4-scout-17b-16e-instruct")
#print(c.run("Find me repos on langchain"))

print(c.run("Get me deep analysis of DL from arxiv papers"))

#llm = ChatOllama(model="qwen2.5:7b")
#print(llm.invoke("Hi").content)