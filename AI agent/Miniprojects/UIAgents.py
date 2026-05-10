import streamlit as st
from dotenv import load_dotenv
load_dotenv()

import os
import requests

from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from tavily import TavilyClient


# =========================================================
# 🔧 TOOLS
# =========================================================

@tool
def get_weather(city: str) -> str:
    """Get current weather for a given city"""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

    response = requests.get(url)
    data = response.json()

    if data.get("cod") != 200:
        return f"❌ {data.get('message', 'Unable to fetch weather data')}"

    temp = data["main"]["temp"]
    description = data["weather"][0]["description"]

    return f"🌦️ **{city} Weather**\n\nTemperature: **{temp}°C**\nCondition: *{description}*"


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
        return f"❌ No news found for {city}."

    news_list = []

    for r in results:
        title = r.get("title", "No title")
        snippet = r.get("content", "")

        news_list.append(f"📰 **{title}**\n{snippet[:120]}...")

    return "\n\n".join(news_list)


# =========================================================
# 🤖 LLM
# =========================================================

llm = ChatMistralAI(model="mistral-small-latest")

tools = [get_weather, get_latest_news]
tools_dict = {tool.name: tool for tool in tools}

llm_with_tools = llm.bind_tools(tools)


# =========================================================
# 🎨 UI
# =========================================================

st.set_page_config(page_title="City Intelligence", page_icon="🌍", layout="centered")

# 🔥 Custom CSS (modern look)
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #0f172a, #1e293b);
}
.chat-bubble-user {
    background-color: #2563eb;
    padding: 10px;
    border-radius: 10px;
    color: white;
}
.chat-bubble-ai {
    background-color: #1e293b;
    padding: 10px;
    border-radius: 10px;
    color: #e2e8f0;
}
</style>
""", unsafe_allow_html=True)


# Header
st.markdown("## 🌍 City Intelligence System")
st.caption("💬 Ask about **weather** or **latest news**")

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("### 💡 Examples")
    st.write("• Weather in Delhi")
    st.write("• News in Mumbai")


# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display chat
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(f"💬 {msg.content}")
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)


# Input
user_input = st.chat_input("Ask something...")

if user_input:
    st.session_state.messages.append(HumanMessage(content=user_input))

    with st.chat_message("user"):
        st.markdown(f"💬 {user_input}")

    with st.spinner("🤖 Thinking..."):
        while True:
            result = llm_with_tools.invoke(st.session_state.messages)
            st.session_state.messages.append(result)

            if result.tool_calls:
                for tool_call in result.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    tool_call_id = tool_call["id"]

                    with st.chat_message("assistant"):
                        st.info(f"🔧 Using tool: {tool_name}")

                    tool_result = tools_dict[tool_name].invoke(tool_args)

                    st.session_state.messages.append(
                        ToolMessage(
                            content=str(tool_result),
                            tool_call_id=tool_call_id
                        )
                    )
                continue
            else:
                with st.chat_message("assistant"):
                    st.markdown(result.content)
                break