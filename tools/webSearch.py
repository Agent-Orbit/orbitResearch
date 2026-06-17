from langchain_tavily import TavilySearch, TavilyExtract
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from utils.apiKeys import TAVILY_API_KEY
from urllib.parse import urlparse
import os

os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

searchTool = TavilySearch()
extractTool = TavilyExtract()


class WebSearchInput(BaseModel):
    
    query: str = Field(description="Search query to look up on the web (e.g. 'latest LangChain updates 2025')")
    max_results: str = Field(default="5", description="Maximum number of search results to return. Between 1 and 10.")
    search_depth: str = Field(default="basic", description="Search depth: 'basic' for fast results, 'advanced' for deeper research.")


class ExtractWebInput(BaseModel):

    url: str = Field(description="Full URL of the webpage to extract content from (e.g. 'https://example.com/article'). Must include https://.")


@tool("webSearch",args_schema=WebSearchInput)
def search(query: str, max_results: str = "5", search_depth: str = "basic") -> dict:
    """
    Search the web for a given query using Tavily.
    Returns a list of relevant URLs, titles, and content snippets.
    Best used for finding current information, news, documentation, and general research.
    """

    max_results = int(max_results)

    try:

        return searchTool.invoke({
            "query": query,
            "max_results": max_results,
            "search_depth": search_depth
        })

    except Exception as e:

        return {"error": f"Web search failed for query '{query}': {e}"}


@tool(args_schema=ExtractWebInput)
def extractWeb(url: str) -> str:
    """
    Extract and return the full text content of a webpage given its URL.
    Best used after a web search to read the full content of a specific result.
    Returns truncated content for very long pages.
    """

    if not is_url(url):

        return f"Invalid URL provided: '{url}'. Must be a full URL including https://."

    try:

        result = extractTool.invoke({"urls": [url]})

        if result.get("results"):

            content = result["results"][0].get("raw_content", "")
            max_chars = 8000

            if len(content) > max_chars:

                content = content[:max_chars] + "\n\n...[truncated]"
            return content

        if result.get("failed_results"):

            return f"Failed to extract content from URL: {result['failed_results']}"

        return "No content could be extracted from the given URL."

    except Exception as e:

        return f"Web extraction failed for URL '{url}': {e}"


def is_url(url: str) -> bool:
    
    """Check if a given string is a valid URL."""
    try:

        result = urlparse(url)
        return all([result.scheme, result.netloc])
    
    except Exception:

        return False  