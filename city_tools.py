import os

import requests
from langchain_core.tools import tool
from tavily import TavilyClient


# ========================================
# TOOL 1: GET WEATHER
# ========================================

@tool
def get_weather(city: str) -> str:
    """Get current weather information for a city,
    including temperature, condition, humidity and wind speed."""

    api_key = os.getenv("OPENWEATHER_API_KEY")

    if not api_key:
        return "Error: OpenWeatherMap API key is missing."

    url = "https://api.openweathermap.org/data/2.5/weather"

    params = {
        "q": city,
        "appid": api_key,
        "units": "metric",
    }

    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        weather = data["weather"][0]
        main = data["main"]
        wind = data["wind"]

        return (
            f"City: {data['name']}, {data['sys']['country']}\n"
            f"Condition: {weather['description']}\n"
            f"Temperature: {main['temp']} °C\n"
            f"Feels like: {main['feels_like']} °C\n"
            f"Humidity: {main['humidity']}%\n"
            f"Wind speed: {wind['speed']} m/s"
        )

    except requests.HTTPError:
        if response.status_code == 401:
            return "Error: Invalid OpenWeatherMap API key."
        elif response.status_code == 404:
            return f"Error: City '{city}' not found."
        elif response.status_code == 429:
            return "Error: Weather API limit exceeded."

        return f"Weather API error: HTTP {response.status_code}"

    except requests.RequestException as e:
        return f"Weather connection error: {e}"

    except (KeyError, IndexError, ValueError):
        return "Error: Could not process weather data."


# ========================================
# TOOL 2: GET NEWS
# ========================================

_tavily_client = None


def _get_tavily() -> TavilyClient:
    """Create the Tavily client lazily so the app can start without a key."""
    global _tavily_client
    if _tavily_client is None:
        _tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    return _tavily_client


@tool
def get_news(city: str) -> str:
    """Search for recent news relevant to a specific city or location."""

    if not os.getenv("TAVILY_API_KEY"):
        return "Error: Tavily API key is missing."

    try:
        response = _get_tavily().search(
            query=(
                f"Recent local news specifically about {city}, Bangladesh. "
                f"Prioritize events, infrastructure, roads, bridges, "
                f"public safety and local government. "
                f"Exclude news about unrelated districts."
            ),
            topic="news",
            max_results=10,
        )

        results = response.get("results", [])

        if not results:
            return f"No news results found for {city}."

        location_words = [
            word.lower().strip()
            for word in city.replace(",", " ").split()
            if word.strip()
        ]

        relevant_news = []

        for item in results:
            title = item.get("title", "")
            content = item.get("content", "")
            url = item.get("url", "")

            text_to_check = f"{title} {content} {url}".lower()

            if any(word in text_to_check for word in location_words):
                relevant_news.append(
                    f"Title: {title}\n"
                    f"URL: {url}\n"
                    f"Content: {content}"
                )

        if not relevant_news:
            return (
                f"No results explicitly mentioning {city} were found. "
                "Try a broader search or a shorter location name."
            )

        return "\n\n".join(relevant_news[:5])

    except Exception as e:
        return f"News search error: {e}"