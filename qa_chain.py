from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

import os

from hf_llm_wrapper import HuggingFaceInferenceLLM

load_dotenv()  # Load HUGGINGFACEHUB_API_TOKEN from .env

# Initialize embedding model
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load vector database
vectordb = Chroma(
    persist_directory="./repo_db",
    embedding_function=embedding
)

# Initialize custom LLM
llm = HuggingFaceInferenceLLM(
    repo_id="microsoft/Phi-3-mini-4k-instruct",
    token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant for answering questions about a Java codebase.

Use the following context to answer the user's question. Only answer based on the context provided. If the answer isn't clearly mentioned or cannot be inferred from the context, respond with "I don't know."

Context:
{context}

Question:
{question}

Answer:
"""
)

# Create RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(),
    chain_type="stuff",
    chain_type_kwargs={"prompt": custom_prompt}
)

# Wrapper to ask questions
def ask_question(query: str) -> str:
    return qa_chain.invoke(query)
