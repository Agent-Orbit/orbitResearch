from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import tool



@tool
def wikipediaSearch(query: str,k_results: int = 3,max_content_chars: int = 4000):
    """ Searches Wikipedia for given query and returns summaries """

    try:

        wikipediaWrapper = WikipediaAPIWrapper(top_k_results=k_results, doc_content_chars_max=max_content_chars,features="lxml")
        return wikipediaWrapper.run(query)
    
    except Exception as e:

        return f"Arxiv search failed: {e}"