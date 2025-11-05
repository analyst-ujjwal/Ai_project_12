import streamlit as st
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()


# --------- Function to generate blog using Hugging Face LLaMA ---------
def getLLamaresponse(topic: str, no_words: int, audience: str) -> str:
    """
    Generate a blog post using LLaMA-3 via Groq API.
    """
    llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-8b-instant")

    template = """
    Write a {no_words}-word blog post for a {audience} audience
    on the topic: "{topic}".
    Make it engaging, clear, and well-structured.
    """
    prompt = PromptTemplate(
        input_variables=["audience", "topic", "no_words"],
        template=template
    )

    formatted_prompt = prompt.format(audience=audience, topic=topic, no_words=no_words)

    response = llm.invoke(formatted_prompt)

    # Extract only the content
    if isinstance(response, dict) and "content" in response:
        return dict(response)["content"]

    return dict(response)["content"]

# --------- Streamlit UI ---------
st.set_page_config(
    page_title="Blog Generator 🤖",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.header("Generate AI Blogs 🤖")

# User input
topic = st.text_input("Enter the Blog Topic")
col1, col2 = st.columns([5, 5])
with col1:
    no_words = st.text_input("Number of Words", "200")
with col2:
    audience = st.selectbox("Writing for", ("Researchers", "Data Scientist", "Common People"))

submit = st.button("Generate Blog")

# Handle button click
if submit:
    if not topic.strip():
        st.error("Please enter a blog topic.")
    else:
        with st.spinner("Generating blog..."):
            try:
                no_words_int = max(50, int(no_words))
                response = getLLamaresponse(topic, no_words_int, audience)
                st.subheader("Generated Blog:")
                st.write(response)
            except ValueError as ve:
                st.error(str(ve))
            except Exception as e:
                st.error(f"Error generating blog: {e}")
