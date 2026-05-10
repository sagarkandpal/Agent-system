from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. prompt template
prompt = ChatPromptTemplate.from_messages([
    ("user", "Explain {topic} in simple words.")
])

# 2. Model
model = ChatMistralAI(model="mistral-small-2506")

# 3. Output parser
parser = StrOutputParser()

"""# Step by step manual flow with old methdods

# Format the prompt
formatted_prompt = prompt.format(topic="quantum computing")

# call the model mauanlly
response = model.invoke(formatted_prompt)

# Parse the output manually
final_output = parser.parse(response.content)

print(final_output)"""

# Sequence runnables

chain = prompt | model | parser

result = chain.invoke({"topic": "physics"})
print(result)


