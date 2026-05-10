from langchain.tools import tool


@tool #decorator for creating a tool
def get_greeting(name: str) -> str: #type hints
    """Generat a greeting message for a user""" #docstring

    return f"Hello {name}, Welcome to the sagar's world of AI agents"

result = get_greeting.invoke({"name": "ronaldo"})
print(result)

print(get_greeting.name)
print(get_greeting.description)
print(get_greeting.args)