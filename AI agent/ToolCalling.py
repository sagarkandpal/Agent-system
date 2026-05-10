from dotenv import load_dotenv
load_dotenv()
from rich import print

from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage

#Tool
@tool
def get_text_length(text:str) -> int:
    """Returns the number of characters in a given text"""
    return len(text)

tools = {
    "get_text_length" : get_text_length
}

# LLM
llm = ChatMistralAI(model = "mistral-small-latest")

#Tool binding
llm_with_tool = llm.bind_tools([get_text_length])

message = []
# generating human mmsg
prompt = input("you: ")   
query = HumanMessage(prompt)
message.append(query)

# generating llm msg
result = llm_with_tool.invoke(message)
message.append(result)

# generating tool msg
if result.tool_calls:
    tool_name = result.tool_calls[0]['name']
    tool_message = tools[tool_name].invoke(result.tool_calls[0])
    message.append(tool_message)

result = llm_with_tool.invoke(message)
print(result.content)

