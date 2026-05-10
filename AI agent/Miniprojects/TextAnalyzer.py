from dotenv import load_dotenv
load_dotenv()
from rich import print

from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage


from langchain.tools import tool

@tool
def word_count(text: str) -> int:
    """Returns the number of words in a given text"""
    return len(text.split())


@tool
def char_count(text: str) -> int:
    """Returns the number of characters in a given text"""
    return len(text)


@tool
def to_uppercase(text: str) -> str:
    """Converts text to uppercase"""
    return text.upper()

# tools as list (for LLM)
tools = [word_count, char_count, to_uppercase]

# dict for execution
tools_dict = {tool.name: tool for tool in tools}

# LLM
llm = ChatMistralAI(model = "mistral-small-latest")

#Tool binding
llm_with_tools = llm.bind_tools(tools)

message = []
# user input
prompt = input("you: ")   
message.append(HumanMessage(prompt))

# llm response
result = llm_with_tools.invoke(message)
message.append(result)

# tool execution
if result.tool_calls:
    tool_call = result.tool_calls[0]  # esse andar tool ki details hai

    tool_name = tool_call["name"]    # eske andar tool ka naam ajayega like word_count
    tool_args = tool_call["args"]   # eske andar vo text ayega jo hamne diya hai like "text": "hello world"
    tool_call_id = tool_call["id"] 

    tool_message = tools_dict[tool_name].invoke(tool_args)

    message.append({
        "role": "tool",
        "name": tool_name,
        "tool_call_id": tool_call_id,
        "content": str(tool_message)
    })

    final_result = llm_with_tools.invoke(message)
    print("AI:", final_result.content)

else:
    print("AI:", result.content)

