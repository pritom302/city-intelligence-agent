
from dotenv import load_dotenv
load_dotenv()

import os
import requests
from langchain.tools import tool
from tavily import TavilyClient
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, ToolMessage,SystemMessage
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call


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
        "units": "metric"
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

tavily_client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)


@tool
def get_news(city: str) -> str:
    """Search for recent news relevant to a specific city or location."""

    try:
        response = tavily_client.search(
            query=(
                f"Recent local news specifically about {city}, Bangladesh. "
                f"Prioritize events, infrastructure, roads, bridges, "
                f"public safety and local government. "
                f"Exclude news about unrelated districts."
            ),
            topic="news",
            max_results=10
        )

        results = response.get("results", [])

        if not results:
            return f"No news results found for {city}."

        # Build location keywords from the requested location.
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

            # Keep results that mention at least one location keyword.
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

# ========================================
# MODEL AND TOOL BINDING
# ========================================

llm = ChatMistralAI(model="codestral-latest")

# tools = {
#     "get_weather": get_weather,
#     "get_news": get_news
# }

# llm_with_tool = llm.bind_tools([get_weather, get_news])


# # ========================================
# # MANUAL AGENT LOOP
# # ========================================

# messages = []

# print("City Intelligence System")
# print("Type 'exit' to quit.")

# while True:
#     user_input = input("\nYou: ").strip()

#     if user_input.lower() in ("exit", "quit"):
#         break

#     if not user_input:
#         continue

#     messages.append(HumanMessage(content=user_input))

#     while True:
#         result = llm_with_tool.invoke(messages)
#         messages.append(result)

#         # Check whether the model requested tools
#         if not result.tool_calls:
#             print("\nAssistant:", result.content)
#             break

#         # Execute each requested tool
#         for tool_call in result.tool_calls:
#             tool_name = tool_call["name"]

#             print(f"\nRequested tool: {tool_name}")

#             confirm = input("Approve? (yes/no): ").strip().lower()

#             if confirm in ("yes", "y"):
#                 if tool_name in tools:
#                     tool_result = tools[tool_name].invoke(
#                         tool_call["args"]
#                     )
#                 else:
#                     tool_result = f"Unknown tool: {tool_name}"

#             else:
#                 tool_result = (
#                     f"Tool {tool_name} was denied by the user."
#                 )

#             print(f"Tool result:\n{tool_result}")

#             # Return the result to the model
#             messages.append(
#                 ToolMessage(
#                     content=str(tool_result),
#                     tool_call_id=tool_call["id"]
#                 )
#             )

#         # Let the model use the tool results
#         # and decide whether another call is needed

@wrap_tool_call
def human_approval(request,handler):
    """Ask for human Approval for every tool call."""
    tool_name = request.tool_call["name"]
    confirm = input(f"Agent wants to call '{tool_name}.Approve(yes/no): '")

    if confirm.lower() !="yes":
        return ToolMessage(
            content = "Tool call denied by user",
            tool_call_id = request.tool_call["id"]
        )

    return handler(request)


agent = create_agent(
    llm,
    tools=[get_weather, get_news],
    system_prompt="""
    You are a City Intelligence Assistant.

    You have two tools:
    1. get_weather(city): Gets current weather information.
    2. get_news(city): Searches for recent news and local updates.

    Rules:
    - For current weather questions, use get_weather.
    - For latest news questions, use get_news.
    - For requests about both weather and news, use both tools.
    - Pass the location provided by the user to the tools.
    - For locations in Bangladesh, include the district and country
      when needed to distinguish places with similar names.
    - Summarize the returned information clearly.
    - Never invent weather data or news.
    - If a tool is denied or fails, explain that to the user.
    """,
    middleware=[human_approval]
)


print("City Agent.........(Type Exit to quit)")
while True:
    inp = input("You: ")
    if inp.lower() == "exit":
        break

    result = agent.invoke({
        "messages": [{"role":"user","content":inp}]
    })

    print("Bot: ",result['messages'][-1].content)