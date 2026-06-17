from langchain_core.tools import tool
from langchain_community.tools import YouTubeSearchTool
from youtube_transcript_api import YouTubeTranscriptApi
import requests
import re
import ast
from pydantic import BaseModel, Field



ytSearchTool = YouTubeSearchTool()

class YoutubeSearchInput(BaseModel):

    query: str = Field(description="Search query to find YouTube videos (e.g. 'LangGraph tutorial for beginners')")
    max_results: str = Field(default="5", description="Maximum number of video URLs to return. Between 1 and 10.")


class YoutubeVideoDetailsInput(BaseModel):

    url: str = Field(description="Full YouTube video URL to fetch details for (e.g. 'https://www.youtube.com/watch?v=dQw4w9WgXcQ')")

@tool(args_schema=YoutubeSearchInput)
def youtubeSearch(query: str, max_results: str = "5") -> list:
    """
    Search YouTube for videos matching a given query.
    Returns a list of video URLs. Use youtubeVideoDetails to get full content of a specific video.
    Best used for finding tutorials, talks, and video explanations of a topic.
    """

    max_results = int(max_results)

    try:

        results = ytSearchTool.run(f"{query},{max_results}")

        if isinstance(results, str):

            results = ast.literal_eval(results)

        return results

    except Exception as e:

        return [f"YouTube search failed for query '{query}': {e}"]

@tool(args_schema=YoutubeVideoDetailsInput)
def youtubeVideoDetails(url: str) -> str:
    """
    Fetch the title, author, and transcript of a YouTube video given its URL.
    Returns full transcript text (truncated at 8000 characters for long videos).
    Best used after youtubeSearch to read the actual content of a video.
    """

    try:

        video_id = extractVideoId(url)

        if not video_id:

            return f"Could not extract video ID from URL: '{url}'"

        # Get title and description via oEmbed
        oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
        response = requests.get(oembed_url,timeout=5)

        title = None
        author = None

        if response.status_code == 200:

            data = response.json()
            title = data.get("title")
            author = data.get("author_name")

        # Get transcript
        transcript_text = None

        try:

            ytt_api = YouTubeTranscriptApi()
            transcript = ytt_api.fetch(video_id)
            transcript_text = " ".join([snippet.text for snippet in transcript])

            max_chars = 8000

            if len(transcript_text) > max_chars:

                transcript_text = transcript_text[:max_chars] + "\n\n...[truncated]"

        except Exception as te:
            
            transcript_text = f"Transcript unavailable: {te}"

        return (
            f"Title: {title}\n"
            f"Author: {author}\n"
            f"Video ID: {video_id}\n"
            f"URL: {url}\n\n"
            f"Transcript:\n{transcript_text}"
        )

    except Exception as e:

        return f"Failed to fetch video details for '{url}': {e}"

def extractVideoId(url: str) -> str | None:
    """Extract the 11-character video ID from a YouTube URL."""

    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"(?:embed\/)([0-9A-Za-z_-]{11})",
        r"(?:youtu\.be\/)([0-9A-Za-z_-]{11})"
    ]

    for pattern in patterns:

        match = re.search(pattern, url)

        if match:

            return match.group(1)
        
    return None