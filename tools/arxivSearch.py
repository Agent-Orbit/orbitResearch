from langchain_core.tools import tool
from langchain_community.utilities import ArxivAPIWrapper
from pydantic import BaseModel, Field


class ArxivSearchInput(BaseModel):

    query: str = Field(description="Search query to find research papers on Arxiv (e.g. 'attention mechanism transformers')")
    k_results: int = Field(default=3, description="Number of papers to return. Keep low (1-5) for focused results.")
    max_content_chars: int = Field(default=4000, description="Maximum characters of content to return per paper. Use higher values for detailed reading.")

@tool(args_schema=ArxivSearchInput)
def arxivSearch(query: str, k_results: int = 3, max_content_chars: int = 4000) -> str:
    """
    Search Arxiv for academic research papers matching a given query.
    Returns paper titles, authors, abstracts, and content summaries.
    Best used for finding cutting-edge research, technical papers, and academic literature.
    """

    try:

        arxivWrapper = ArxivAPIWrapper(top_k_results=k_results, doc_content_chars_max=max_content_chars)
        return arxivWrapper.run(query)
    
    except Exception as e:

        return f"Arxiv search failed for query '{query}': {e}"
