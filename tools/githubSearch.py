from langchain_core.tools import tool
from github import Github
from utils.apiKeys import GITHUB_TOKEN

g = Github(GITHUB_TOKEN)

@tool
def githubSearchTool(query: str,max_repos: int = 3) -> list:
    """Searches GitHub repositories for a given query"""

    try:

        
        results = g.search_repositories(query=query)

        # All repositories
        repos = []

        for repo in results[:max_repos]:

            repos.append({
                "name": repo.full_name,
                "description": repo.description,
                "url": repo.html_url,
                "stars": repo.stargazers_count
            })

        return repos
    
    except Exception as e:

        return f"Arxiv search failed: {e}"