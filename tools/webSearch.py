from langchain_tavily import TavilySearch,TavilyExtract
from langchain_core.tools import tool
from utils.apiKeys import TAVILY_API_KEY
import os
from urllib.parse import urlparse

os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

searchTool = TavilySearch()
extractTool = TavilyExtract()

@tool
def search(query: str, max_results: int = 5, search_depth = "basic"):
    """ Searches web for query """

    try:

        return searchTool.invoke({
            "query": query,
            "max_results": max_results,
            "search_depth": search_depth
        })

    except Exception as e:

        raise e

@tool
def extractWeb(url: str):
    """ Opens given URL and extracts its contents """

    try:

        if not is_url(url):

            return "Given URL is incorrect"

        # Make URLs list later.
        result = extractTool.invoke({"urls": [url]})

        if result.get("results"):

            content = result["results"][0].get("raw_content", "")
            max_chars = 8000

            if len(content) > max_chars:

                content = content[:max_chars] + "\n\n...[truncated]"

            return content

        if result.get("failed_results"):

            return f"Failed to extract: {result['failed_results']}"

        return "No content extracted"
    
    except Exception as e:

        raise e


def is_url(url: str):
    """ Checkes if given string is an URL or not """

    try:

        result = urlparse(url)

        return all([
            result.scheme,
            result.netloc
        ])
    
    except Exception as e:

        return f"Arxiv search failed: {e}"
