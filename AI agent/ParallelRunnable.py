from click import prompt
from dotenv import load_dotenv
load_dotenv()

import langchain_core
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda

model = ChatMistralAI(model="mistral-small-2506")
parser = StrOutputParser()

# Two diff prompts for parallel execution
short_prompt = ChatPromptTemplate.from_messages([
    ("user", "Explain {topic} in 20 words.")
])

long_prompt = ChatPromptTemplate.from_messages([
    ("user", "Explain {topic} in 200 words.")
])

#Input
chain1 = short_prompt | model | parser
chain2 = long_prompt | model | parser

chain = RunnableParallel({
    "short": RunnableLambda(lambda x:x['short']) | chain1,
    "brief": RunnableLambda(lambda x:x['brief']) | chain2
})

result = chain.invoke({
    "short" : {"topic": "machine learning"},
    "brief": {"topic": "quantum computing"}
})

print(result["short"])
print(result["brief"])


 