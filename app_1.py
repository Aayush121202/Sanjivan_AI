# -*- coding: utf-8 -*-
"""Sanjivan_AI.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1GAEbjy7YkfBsR5Bwi4Vz9Qm_UzvZgtag
"""

# Import necessary libraries
import streamlit as st
import google.generativeai as palm

# Initialize Google PaLM API
API_KEY = "AIzaSyCRqg5aq7QARvhhE-UjLI0yVgrN_wZ5FmQ"  # Replace with your API key
palm.configure(api_key=API_KEY)

# Streamlit app
st.title("Interactive Data Analysis of Titanic Dataset")

# Load Titanic dataset
df = pd.read_csv("/content/Titanic.csv")
st.write(df.head())

# Text input for questions
user_query = st.text_input(
    "Ask a question about the Titanic dataset (e.g., 'What is the survival rate of males?')"
)

# Define a function to handle AI-powered responses
def answer_query_with_palm(query, dataframe):
    """
    Uses Google's PaLM API to answer questions related to the dataframe.
    """
    # Convert the dataframe's first few rows to a string for context
    context = f"Here is the data:\n{dataframe.head(10).to_string(index=False)}"

    # Create a prompt
    prompt = f"""
    You are a data analyst. Use the following data to answer the query:
    {context}

    Question: {query}
    Provide the best answer.
    """

    # Get the response from PaLM
    response = palm.generate_text(prompt=prompt)
    return response.result if response else "I'm sorry, I couldn't process your query."

# Handle user query
if user_query:
    with st.spinner("Analyzing your query..."):
        # Get AI response
        ai_response = answer_query_with_palm(user_query, df)
        st.write("### AI's Answer:")
        st.write(ai_response)

# Footer
st.markdown(
    """
    **Note**: This app is powered by Google AI's PaLM API for natural language understanding and dynamic data analysis.
    """
)