import os
import yt_dlp
from pytube import Channel
# Define the output directory for downloaded videos
OUTPUT_DIR = "input/videos"

def download_video(url):
    ydl_opts = {
        'outtmpl': f'{OUTPUT_DIR}/%(title)s.%(ext)s',
        'format': 'mp4'
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

# Function to scrape popular channels from a YouTube search query
def get_popular_channels(query):
    # Use requests library to send an HTTP request to the YouTube search page
    response = Request.get(f"https://www.youtube.com/results?search_query={query}")
    html_content = response.content

    # Parse HTML content using BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find all video entries on the page
    channel_entries = soup.find_all('a', class_='yt-simple-endpoint style-scope ytd-rich-grid-renderer ytd-compact-channel-renderer')

    # List to store popular channel URLs
    popular_channels = []

    # Iterate over each channel entry and extract the URL
    for channel in channel_entries:
        channel_url = channel['href']
        if not channel_url.startswith('#'):
            popular_channels.append(channel_url)

    return popular_channels

# Function to get trending videos from YouTube
def get_trending_videos():
    # Use pytube to fetch trending video URLs and their metrics (view count, like count)
    channel = Channel("https://www.youtube.com/feed/trending")
    videos = channel.videos

    # Sort videos by view count in descending order
    popular_videos = sorted(videos, key=lambda x: x.views, reverse=True)

    return [video.url for video in popular_videos[:10]]  # Get top 10 most viewed trending videos

# Function to detect viral videos from YouTube's trending API
def get_viral_videos():
    # Use pytube to create a Channel object for the trending channel
    channel = Channel("https://www.youtube.com/feed/trending")

    # Fetch video URLs and their metrics (view count, like count)
    videos = channel.videos

    # Sort videos by view count in descending order
    viral_videos = sorted(videos, key=lambda x: x.views, reverse=True)

    return [video.url for video in viral_videos[:5]]  # Get top 5 most viewed trending videos

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Example query to find popular channels related to entertainment
    query = "entertainment"

    # Get popular channels from the search query
    popular_channels = get_popular_channels(query)

    print("Popular Channels:")
    for i, url in enumerate(popular_channels):
        print(f"Downloading Channel {i+1}: {url}")
        download_video(url)

    print("\nTrending Videos:")
    for i, url in enumerate(get_trending_videos()):
        print(f"Downloading Video {i+1}: {url}")
        download_video(url)

    print("\nViral Videos:")
    for i, url in enumerate(get_viral_videos()):
        print(f"Downloading Video {i+1}: {url}")
        download_video(url)
