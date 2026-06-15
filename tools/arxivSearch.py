from langchain_core.tools import tool
from langchain_community.utilities import ArxivAPIWrapper

@tool
def arxivSearch(query: str,k_results: int = 3,max_content_chars: int = 4000):
    """ Searches Arxiv for research papers matching given query """

    try:

        arxivWrapper = ArxivAPIWrapper(top_k_results=k_results, doc_content_chars_max=max_content_chars)
        return arxivWrapper.run(query)
    
    except Exception as e:

        return f"Arxiv search failed: {e}"
