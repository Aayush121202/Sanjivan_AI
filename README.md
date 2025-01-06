# Titanic Data Analysis Assistant with Generative AI

## Overview

This project leverages Generative AI to help users perform data analysis on the Titanic dataset. The assistant is powered by **Google Generative AI** (via `langchain_google_genai`) and is designed to assist with querying, generating Python code, and running analysis on Titanic dataset-related questions. Users can ask various questions related to the dataset, and the system will provide Python code along with relevant visualizations (if required).

## Features

- **Question Answering:** Users can ask any relevant question about the Titanic dataset (e.g., survival rates, age distributions, ticket fare distributions).
- **Code Generation:** The system generates Python code to analyze the Titanic dataset based on the user query.
- **Data Visualization:** It can generate plots (e.g., histograms) for visual analysis of different attributes in the dataset (e.g., age, fare, survival rate).
- **Streamlit Interface:** The web interface is built using Streamlit to interact with the AI assistant in real time.

## Requirements

To run this project locally, you'll need the following:

- Python 3.x
- Streamlit
- pandas
- matplotlib
- seaborn
- openai
