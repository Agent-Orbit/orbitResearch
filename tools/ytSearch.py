from langchain_core.tools import tool
from langchain_community.tools import YouTubeSearchTool
from youtube_transcript_api import YouTubeTranscriptApi
import requests
import re

ytSearchTool = YouTubeSearchTool()


@tool
def youtubeSearch(query: str, max_results: int = 5) -> list:
    """ Searches YouTube for videos matching given query and returns video URLs """

    try:

        results = ytSearchTool.run(f"{query},{max_results}")

        if isinstance(results, str):

            results = eval(results)

        return results

    except Exception as e:

        return f"YouTube search failed: {e}"

@tool
def youtubeVideoDetails(url: str) -> dict:
    """ Given a YouTube video URL, returns its title, description and transcript """

    try:

        video_id = extractVideoId(url)

        if not video_id:
            return {"error": "Could not extract video ID from URL"}

        # Get title and description via oEmbed (no API key needed)
        oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
        response = requests.get(oembed_url)

        title = None
        author = None

        if response.status_code == 200:
            
            data = response.json()
            title = data.get("title")
            author = data.get("author_name")

        # Get transcript
        transcript_text = None

        try:

            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            transcript_text = " ".join([entry["text"] for entry in transcript])

            max_chars = 8000

            if len(transcript_text) > max_chars:

                transcript_text = transcript_text[:max_chars] + "\n\n...[truncated]"

        except Exception as te:

            transcript_text = f"Transcript unavailable: {te}"

        return {
            "video_id": video_id,
            "url": url,
            "title": title,
            "author": author,
            "transcript": transcript_text
        }

    except Exception as e:

        return {"error": f"Failed to fetch video details: {e}"}

def extractVideoId(url: str) -> str:
    """ Extracts the video ID from a YouTube URL """

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