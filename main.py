from tools import webSearch,githubSearch,wikiSearch,arxivSearch,ytSearch
from research_core import core
from langchain_ollama import ChatOllama

# print(webSearch.searchTool.invoke({"query": "what is 3*3 + 3"}))
#print(githubSearch.githubSearchTool.invoke({"query": "langchain"}))
#print(webSearch.extractWeb.invoke({"url": "https://github.com/Agent-Orbit/PDF-Analyser"}))
#print(wikiSearch.wikipediaSearch.invoke({"query": "Lionel_Messi"}))
#print(arxivSearch.arxivSearch.invoke({"query": "Deep Learning"}))
#print(ytSearch.youtubeVideoDetails.invoke({"url": "https://www.youtube.com/watch?v=dtCS0WGZH4k"}))

c = core.CoreResearch("qwen2.5:7b")
#print(c.run("Make a deep research on Argentina's 2026 WC team."))

#print(c.run("What is DL"))

#llm = ChatOllama(model="qwen2.5:7b")
#print(llm.invoke("Hi").content)

while True:

    q = input(">> ")
    if q in ["exit","e"]:
        break

    print(f"AI: {c.run(q)}")