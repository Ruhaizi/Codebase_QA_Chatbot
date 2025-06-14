# Codebase QA Chatbot

Codebase QA Chatbot is a local question-answering system that allows you to query your codebase in plain English. Built using LangChain, ChromaDB, Hugging Face Inference API, and Streamlit, this project helps developers understand large codebases faster.

## Features

- Semantic search over local codebase files using language models
- Custom LLM integration with Hugging Face Inference API
- ChromaDB-based vector store using sentence-transformer embeddings
- Clean and simple Streamlit interface for asking questions

## Tech Stack

- Python 3.12
- LangChain
- Hugging Face Inference API
- ChromaDB
- Sentence Transformers (`all-MiniLM-L6-v2`)
- Streamlit
- dotenv

## Setup Instructions

```bash
# Clone the repository
git clone https://github.com/Ruhaizi/Codebase_QA_Chatbot.git
cd Codebase_QA_Chatbot

# Create and activate virtual environment
python3 -m venv env
source env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Add your Hugging Face API token to .env
echo "HUGGINGFACEHUB_API_TOKEN=your_token_here" > .env

# Run the Streamlit app
streamlit run app.py
