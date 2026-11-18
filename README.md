# RAGDocu

A local Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask questions based on their contents.

RAGDocu uses locally running language models through Ollama, so it does not require paid LLM APIs or external AI services.

## Features

- Upload and process PDF documents.
- Extract text and split documents into chunks.
- Generate embeddings and store them in ChromaDB.
- Retrieve relevant document chunks using semantic search.
- Generate context-aware answers using a local LLM.
- Display document sources alongside generated answers.
- Prevent duplicate document uploads using SHA-256 hashing.
- Interactive chat interface with Streamlit.

## Tech Stack

| Component | Technology |
|---|---|
| Programming Language | Python 3.11 |
| Dependency Management | uv |
| LLM Runtime | Ollama |
| Chat Model | Qwen3 4B Instruct |
| Embedding Model | nomic-embed-text |
| RAG Framework | LangChain |
| Vector Database | ChromaDB |
| PDF Processing | PyMuPDF |
| Backend | FastAPI |
| Frontend | Streamlit |

## Architecture

```text
                 User
                  |
                  v
           Streamlit Frontend
                  |
                  v
             FastAPI API
                  |
                  v
           LangChain RAG
                  |
          +-------+-------+
          |               |
          v               v
       ChromaDB         Ollama
     (Vector Store)   (Local LLM)
                          |
                   +------+------+
                   |             |
                   v             v
                 Qwen3      nomic-embed-text
```

## Requirements

- Python 3.11
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/)
- Git

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd RAGDocu
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Download Ollama models

Make sure Ollama is installed and running, then execute:

```bash
ollama pull qwen3:4b-instruct
ollama pull nomic-embed-text
```

### 4. Initialize a new ChromaDB

**Important:** This repository's original ChromaDB contains previously uploaded documents and their embeddings. If you want to start with a clean knowledge base, initialize a new database rather than reusing the existing database.

The LangChain vector store uses the following directory:

```text
chroma_db_langchain/
```

If this directory already exists, stop the application and remove it before the first run:

```bash
# Windows Command Prompt
rmdir /s /q chroma_db_langchain
```

The application will automatically create a new persistent ChromaDB directory and collection when it starts.

**Warning:** Removing this directory permanently deletes the documents and embeddings stored in that database. Back it up first if you need its contents.

### 5. Run the backend

Open a terminal:

```bash
uv run uvicorn ragdocu.api.main:app --reload
```

Backend API:

```text
http://localhost:8000
```

API documentation:

```text
http://localhost:8000/docs
```

### 6. Run the frontend

Open another terminal:

```bash
uv run streamlit run frontend/app.py
```

Open the application:

```text
http://localhost:8501
```

## Usage

1. Start Ollama, FastAPI, and Streamlit.
2. Upload a PDF document through the sidebar.
3. Wait for the document processing and embedding process to finish.
4. Enter a question related to the uploaded document.
5. Review the generated answer and its source references.

## Project Scope

This project was built as a learning and portfolio project to understand the fundamentals of Retrieval-Augmented Generation.

It covers document preprocessing, chunking, embedding generation, vector storage, semantic retrieval, LLM integration, retrieval evaluation, and application development.

The project uses a manually implemented RAG pipeline as a learning foundation and a LangChain implementation for comparison and application development.

## Notes

- Ollama must be running locally before using the application.
- The application uses local models and does not require paid LLM APIs.
- The default retrieval configuration uses Top-K = 3. But you can simply adjust it to fulfill your condition/needs
- Uploaded PDFs are processed into chunks and stored in ChromaDB.
- The application currently supports text-based PDFs. Scanned PDFs requiring OCR are not supported.
- The frontend conversation history is session-based and is not persisted across sessions.

## License

This project is intended for educational and portfolio purposes.
