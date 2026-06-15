from tools import webSearch,githubSearch,wikiSearch,arxivSearch,ytSearch

# print(webSearch.searchTool.invoke({"query": "what is 3*3 + 3"}))
#print(githubSearch.githubSearchTool.invoke({"query": "langchain"}))
#print(webSearch.extractWeb.invoke({"url": "https://github.com/Agent-Orbit/PDF-Analyser"}))
#print(wikiSearch.wikipediaSearch.invoke({"query": "Lionel_Messi"}))
#print(arxivSearch.arxivSearch.invoke({"query": "Deep Learning"}))
print(ytSearch.youtubeVideoDetails.invoke({"url": "https://www.youtube.com/watch?v=AYW386xCAKw"}))