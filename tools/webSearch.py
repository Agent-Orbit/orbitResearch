import requests
from utils.apiKeys import TAVILY_API_KEY

def searchTool(query: str):
    """ Searhces Tavily for given query """

    url = "https://api.tavily.com/search"

    headers = {
        "Authorization": f"{TAVILY_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "query": query,
        "search_depth": "advanced",
        "max_results": 5
    }

    response = requests.post(url, json=payload, headers=headers)

    print(response.json()["results"][0])