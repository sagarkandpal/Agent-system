import streamlit as st
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load env
load_dotenv()

# Page config
st.set_page_config(
    page_title="🎵 Bollywood Song Generator",
    page_icon="🎶",
    layout="centered"
)

# Custom UI (modern glass style)
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f172a, #1e293b);
}
.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: white;
}
.subtitle {
    text-align: center;
    color: #cbd5f5;
    margin-bottom: 25px;
}
.card {
    background: rgba(255,255,255,0.05);
    padding: 18px;
    border-radius: 12px;
    margin-bottom: 15px;
    backdrop-filter: blur(10px);
    color: white;
    border: 1px solid rgba(255,255,255,0.1);
}
button[kind="primary"] {
    width: 100%;
    border-radius: 10px;
    height: 45px;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">🎵 Bollywood Song Generator</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Generate lyrics, title & album description</div>', unsafe_allow_html=True)

# Input
topic = st.text_input("🎧 Enter genre / vibe:", placeholder="e.g. romantic, sad, pop, breakup...")

generate = st.button("✨ Generate Song")

# Model + Chains
model = ChatMistralAI(model="mistral-small-2506")
parser = StrOutputParser()

song_genre = ChatPromptTemplate.from_messages([
    ("system", "you are a short beautiful hindi bollywood song generator"),
    ("human", "{topic}")
])

song_title = ChatPromptTemplate.from_messages([
    ("system", "you are a helpful assistant who generates a short 2 words title for the given song lyrics"), 
    ("human", "generate the title for this short song lyrics: {lyrics}")    
])

album_des = ChatPromptTemplate.from_messages([
    ("system", "you are a helpful assistant who generates a short album description according to given song lyrics and song title"), 
    ("human", "generate the album description for the song lyrics: {lyrics} with title name {song_title}")    
])

seq1 = song_genre | model | parser

seq2 = RunnableParallel({
    "lyrics": RunnablePassthrough(),
    "song_title": song_title | model | parser,
})

seq3 = RunnableParallel({
    "lyrics": RunnableLambda(lambda x: x['lyrics']),
    "song_title": RunnableLambda(lambda x: x['song_title']),
    "album_description": album_des | model | parser
})

chain = seq1 | seq2 | seq3

# Generate
if generate:
    if topic.strip() == "":
        st.warning("⚠️ Please enter a genre or vibe")
    else:
        with st.spinner("Creating your song... 🎶"):
            result = chain.invoke({"topic": topic})

        st.markdown("### 🎤 Lyrics")
        st.markdown(f'<div class="card">{result["lyrics"]}</div>', unsafe_allow_html=True)

        st.markdown("### 🏷️ Title")
        st.markdown(f'<div class="card">{result["song_title"]}</div>', unsafe_allow_html=True)

        st.markdown("### 💿 Album Description")
        st.markdown(f'<div class="card">{result["album_description"]}</div>', unsafe_allow_html=True)