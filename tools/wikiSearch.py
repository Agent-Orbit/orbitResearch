from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import tool
from pydantic import BaseModel, Field


class WikipediaSearchInput(BaseModel):

    query: str = Field(description="Search query to look up on Wikipedia (e.g. 'transformer neural network architecture')")
    k_results: str = Field(default="3", description="Number of Wikipedia articles to return. Between 1 and 5.")
    max_content_chars: str = Field(default="4000", description="Maximum characters of content per article. Use higher values for more detail.")


@tool(args_schema=WikipediaSearchInput)
def wikipediaSearch(query: str, k_results: str = "3", max_content_chars: str = "4000") -> str:
    """
    Search Wikipedia for articles matching a given query.
    Returns article summaries and content for general knowledge, concepts, and background information.
    Best used for understanding definitions, historical context, and well-established topics.
    """

    k_results = int(k_results)
    max_content_chars = int(max_content_chars)

    try:

        wikipediaWrapper = WikipediaAPIWrapper(top_k_results=k_results, doc_content_chars_max=max_content_chars,features="lxml")
        return wikipediaWrapper.run(query)
    
    except Exception as e:

        return f"Wikipedia search failed for query '{query}': {e}"