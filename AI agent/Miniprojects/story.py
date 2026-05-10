from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.runnables import RunnableParallel
from typer import prompt
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# prompt1 = ChatPromptTemplate.from_messages([
#     ("user", "generate a short 200 words beautiful story for this given {genre}")
# ])

# prompt2 = ChatPromptTemplate.from_messages([
#     ("user", "generate a short 2 words tittle for this given short story {final_output}")
# ])

# prompt3 = ChatPromptTemplate.from_messages([
#     ("user", "generate a short 2 line moral for this {final_output}")
# ])


# model = ChatMistralAI(model="mistral-small-2506")

# parser = StrOutputParser()

# formatted_prompt = prompt1.format(genre="love story")

# response = model.invoke(formatted_prompt)

# final_output = parser.parse(response.content)

# chain1 = prompt1 | model | parser
# chain2 = prompt2 | model | parser
# chain3 = prompt3 | model | parser

# result1 = chain1.invoke({"genre": "love story"})
# result2 = chain2.invoke({"final_output": result1})  
# result3 = chain3.invoke({"final_output": result1})  

# print("Story: ", result1)
# print("Title: ", result2)
# print("Moral: ", result3)

model = ChatMistralAI(model="mistral-small-2506")
parser = StrOutputParser()

story_prompt = ChatPromptTemplate.from_messages([
    ("system", "you are a short beautiful 200 words story generator"),
    ("human", "{topic}")
])

title_prompt = ChatPromptTemplate.from_messages([
    ("system", "you are a helpful assistance who generates a short 2 words tittle for the given story"), 
    ("human", "generate the 2 word tittle for this short story: {story}")    
])

moral_prompt = ChatPromptTemplate.from_messages([
    ("system", "you are a helpful assistance who generates a short 2 line moral for the given story"), 
    ("human", "generate the 2 line moral for this short story: {story}")    
])

seq = story_prompt | model | parser

seq2 = RunnableParallel(
    {"story" : RunnablePassthrough(),
    "title" : title_prompt | model | parser,
    "moral" : moral_prompt | model | parser
    }
)

chain = seq | seq2

result = chain.invoke({"topic" : "write a love story."})
print("Story: ", result['story'])
print("Title: ", result['title'])   
print("Moral: ", result['moral'])   