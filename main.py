import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

def get_movie_chain():
    prompt_template = """
    You are a knowledgeable song assistant. Answer song-related questions with detailed recommendations and insights.
    For non-song-related questions, respond with: "I don't know. I only handle song-related queries."

    Question: {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["question"])
    chain = LLMChain(llm=model, prompt=prompt)
    return chain

def main():
    st.set_page_config(page_title="Song-Recommender", layout="wide")
    st.header(" Song-Recommender 2.0 🤖")

    question = st.text_input("Enter your song related question:")
    if question:
        chain = get_movie_chain()
        response = chain.run({"question": question})
        st.write("Response:", response)

if __name__ == "__main__":
    main()
