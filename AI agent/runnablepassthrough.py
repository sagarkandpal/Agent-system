from click import prompt
from dotenv import load_dotenv
load_dotenv()

import langchain_core
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough

model = ChatMistralAI(model="mistral-small-2506")
parser = StrOutputParser()

code_prompt = ChatPromptTemplate.from_messages([
    ("system", "you are a code generator"),
    ("human", "{topic}")
])

explain_prompt = ChatPromptTemplate.from_messages([
    ("system", "you are a helpful assistance who explains code in simple language"), 
    ("human", "explain the following code in simple words: {code}")    
])

seq = code_prompt | model | parser      

seq2 = RunnableParallel(
    {"code" : RunnablePassthrough(),
    "explanation" : explain_prompt | model | parser
    }
)

chain = seq | seq2

result = chain.invoke({"topic" : "please write a code of addtion of 2 digits in python."})

print(result['code'])
print(result['explanation'])
