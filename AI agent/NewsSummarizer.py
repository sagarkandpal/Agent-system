from dotenv import load_dotenv
load_dotenv()
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

#bhai ham yahape tool ka use esliye kar rhe hai kyuki mistral ai hame 2026 ka current data nhi de sakta 
# to ham tavily search tool ka use kar rhe hai taki hame current news ke baare me pata chal ske 
# aur uske baad ham mistral ai ka use kar ke us news ka summary generate karenge.

search_tool = TavilySearchResults(max_result = 5)

llm = ChatMistralAI(model="mistral-small-2506")

search_prompt = ChatPromptTemplate.from_messages([
    ("system", "you are a helpful assistance who generates a clear bullet points for the given news topic"),
    ("human", "generate the news summary for this news topic: {news_topic}")        
])

chain = search_prompt | llm | StrOutputParser()

news_result = search_tool.run("latest news on AI")

result = chain.invoke({"news_topic": news_result})

print("News Summary: ", result)