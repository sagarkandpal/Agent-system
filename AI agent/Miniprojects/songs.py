from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from typer import prompt
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


model = ChatMistralAI(model="mistral-small-2506")
parser = StrOutputParser()

song_genre = ChatPromptTemplate.from_messages([
    ("system", "you are a short beautiful hindi bollywood song generator"),
    ("human", "{topic}")
])

song_title = ChatPromptTemplate.from_messages([
    ("system", "you are a helpful assistance who generates a short 2 words tittle for the given song lyrics"), 
    ("human", "generate the tittle for this short song lyrics: {lyrics}")    
])

album_des = ChatPromptTemplate.from_messages([
    ("system", "you are a helpful assistance who generates a short album description for the given according to given song lyrics and song title."), 
    ("human", "generate the album description for the song lyrics: {lyrics} with title name {song_title}")    
])

seq1 = song_genre | model | parser

seq2 = RunnableParallel(
    {"lyrics" : RunnablePassthrough(),
    "song_title" : song_title | model | parser,
    }
)

seq3 = RunnableParallel({
    "lyrics" : RunnableLambda(lambda x:x['lyrics']),
    "song_title" : RunnableLambda(lambda x:x['song_title']),
    "album_description" : album_des | model | parser
    
})

chain = seq1 | seq2 | seq3

result = chain.invoke({"topic" : "pop"})
print("lyrics: ", result['lyrics'])
print("Title: ", result['song_title'])   
print("Album: ", result['album_description'])   