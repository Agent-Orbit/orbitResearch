from langchain_core.tools import tool
from github import Github, GithubException
from utils.apiKeys import GITHUB_TOKEN
from pydantic import BaseModel, Field

g = Github(GITHUB_TOKEN)

class GithubSearchInput(BaseModel):

    query: str = Field(description="Search query to find GitHub repositories (e.g. 'langchain agent python')")
    max_repos: int = Field(default=3, description="Maximum number of repositories to return. Must be an integer between 1 and 10.")
    sort: str = Field(default="stars", description="Sort results by: 'stars', 'forks', 'updated', or 'best-match'.")
    language: str = Field(default="", description="Filter repositories by programming language (e.g. 'python', 'javascript'). Leave empty for all languages.")


@tool(args_schema=GithubSearchInput)
def githubSearchTool(query: str, max_repos: int = 3, sort: str = "stars", language: str = "") -> list:
    """
    Search GitHub repositories for a given query.
    Returns repository names, descriptions, URLs, and star counts.
    Best used for finding open source projects, code examples, and libraries.
    """

    try:

        full_query = f"{query} language:{language}" if language else query
        results = g.search_repositories(query=full_query, sort=sort)

        repos = []

        for repo in results[:max_repos]:

            repos.append({
                "name": repo.full_name,
                "description": repo.description,
                "url": repo.html_url,
                "stars": repo.stargazers_count,
                "language": repo.language,
                "last_updated": str(repo.updated_at)
            })

        return repos

    except GithubException as e:

        return [{"error": f"GitHub API error: {e.data.get('message', str(e))}"}]

    except Exception as e:

        return [{"error": f"GitHub search failed for query '{query}': {e}"}]