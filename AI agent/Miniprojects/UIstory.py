import streamlit as st
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load env
load_dotenv()

# Page config
st.set_page_config(
    page_title="AI Story Generator",
    page_icon="📖",
    layout="centered"
)

# Custom CSS for better UI
st.markdown("""
    <style>
        .main {
            background-color: #0e1117;
        }
        .title {
            text-align: center;
            font-size: 40px;
            font-weight: bold;
            color: #ffffff;
        }
        .subtitle {
            text-align: center;
            font-size: 18px;
            color: #bbbbbb;
            margin-bottom: 20px;
        }
        .output-box {
            background-color: #1c1f26;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 15px;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">📖 AI Story Generator</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Generate story, title & moral instantly</div>', unsafe_allow_html=True)

# Input
topic = st.text_input("Enter your topic:", placeholder="e.g. A love story, friendship, life lesson...")

# Button
generate = st.button("✨ Generate")

# Model + Chains
model = ChatMistralAI(model="mistral-small-2506")
parser = StrOutputParser()

story_prompt = ChatPromptTemplate.from_messages([
    ("system", "you are a short beautiful 200 words story generator"),
    ("human", "{topic}")
])

title_prompt = ChatPromptTemplate.from_messages([
    ("system", "you are a helpful assistant who generates a short 2 word title for the given story"), 
    ("human", "generate the 2 word title for this short story: {story}")    
])

moral_prompt = ChatPromptTemplate.from_messages([
    ("system", "you are a helpful assistant who generates a short 2 line moral for the given story"), 
    ("human", "generate the 2 line moral for this short story: {story}")    
])

seq = story_prompt | model | parser

seq2 = RunnableParallel({
    "story": RunnablePassthrough(),
    "title": title_prompt | model | parser,
    "moral": moral_prompt | model | parser
})

chain = seq | seq2

# Generate output
if generate:
    if topic.strip() == "":
        st.warning("⚠️ Please enter a topic")
    else:
        with st.spinner("Generating..."):
            result = chain.invoke({"topic": topic})

        # Output UI
        st.markdown("### 📝 Story")
        st.markdown(f'<div class="output-box">{result["story"]}</div>', unsafe_allow_html=True)

        st.markdown("### 🏷️ Title")
        st.markdown(f'<div class="output-box">{result["title"]}</div>', unsafe_allow_html=True)

        st.markdown("### 📌 Moral")
        st.markdown(f'<div class="output-box">{result["moral"]}</div>', unsafe_allow_html=True)