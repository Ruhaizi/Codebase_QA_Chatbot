from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

import os
from dotenv import load_dotenv
load_dotenv()

def load_files(folder):
    extensions = ["*.py", "*.md", "*.js", "*.java", "*.ts"]
    exclude_dirs = ["node_modules", ".git", ".venv", "__pycache__"]

    files = []
    for ext in extensions:
        files.extend(Path(folder).rglob(ext))

    loaders = []
    for file_path in files:
        # Skip if path includes excluded directory names
        if any(excluded in str(file_path) for excluded in exclude_dirs):
            continue

        # Only load actual files
        if file_path.is_file():
            try:
                loaders.append(TextLoader(str(file_path)))
            except Exception as e:
                print(f"Skipped {file_path}: {e}")

    # Load and flatten all documents
    all_docs_nested = [loader.load() for loader in loaders]
    all_docs = [doc for sublist in all_docs_nested for doc in sublist]
    return all_docs


repo_path = "/Users/ruhaizimopuri/Documents/projects/Crypto_trading"
docs = load_files(repo_path)
print(f"Loaded {len(docs)} documents.")



splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 150
)

chunks = splitter.split_documents(docs)
print(f"Split into {len(chunks)} chunks.")


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma.from_documents(
    documents = chunks,
    embedding = embeddings,
    persist_directory = "./repo_db"
)

vectordb.persist()
print("Embedding complete. Vector DB saved to ./repo_db")

