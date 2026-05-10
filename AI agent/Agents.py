from dotenv import load_dotenv
load_dotenv()

import os
import requests

from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import ToolMessage
from tavily import TavilyClient
from rich import print
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call


# =========================================================
# 🔧 TOOLS SECTION
# =========================================================

# 🌦️ Weather Tool
@tool
def get_weather(city: str) -> str:
    """Get current weather for a given city"""

    api_key = os.getenv("OPENWEATHER_API_KEY")

    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

    response = requests.get(url)
    data = response.json()

    if data.get("cod") != 200:
        return f"Error: {data.get('message', 'Unable to fetch weather data')}"

    temp = data["main"]["temp"]
    description = data["weather"][0]["description"]

    return f"The current temperature in {city} is {temp}°C with {description}."


# 📰 Tavily News Tool
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


@tool
def get_latest_news(city: str) -> str:
    """Get latest news for a given city"""

    response = tavily_client.search(
        query=f"latest news in {city}",
        search_depth="basic",
        max_results=3
    )

    results = response.get("results", [])

    if not results:
        return f"No news found for {city}."

    news_list = []

    for r in results:
        title = r.get("title", "No title")
        url = r.get("url", "")
        snippet = r.get("content", "")

        news_list.append(
            f"{title}\n - {url}\n {snippet[:100]}..."
        )

    return f"Latest news in {city}:\n\n" + "\n\n".join(news_list)


# =========================================================
# 🤖 LLM + TOOL BINDING
# =========================================================

llm = ChatMistralAI(model="mistral-medium-latest")

#this is middleware which is used to get the human approval for tool calls.
@wrap_tool_call
def human_approval(request, handler):
    """Ask for human approval before executing a tool call"""
    tool_name = request.tool_call["name"]
    confirm = input(f"Agent wants to call tool '{tool_name}'. Approve? (y/n): ")
    
    if confirm.lower() != "y":
        return ToolMessage(
            content = "Tool call denied by user.",
            tool_call_id = request.tool_call["id"]
        )
    
    return handler(request)



tools = [get_weather, get_latest_news]

#create_agent is a helper function to create an agent with llm, tools and middleware.
agent = create_agent(
    llm,
    tools = [get_weather, get_latest_news],
    system_prompt = "You are a helpful city assistant.",
    middleware = [human_approval]
)

print("city Agent || type over to Exit..")

while True: 
    user_input = input("You : ")
    if user_input.lower() == "over":
        break
    result = agent.invoke({
        "messages" : [{
            "role" : "user",
            "content" : user_input
        }]
    })

    print(result['messages'][-1].content)


