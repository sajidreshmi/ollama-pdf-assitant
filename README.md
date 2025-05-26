# PDF RAG with Ollama and Streamlit

This project demonstrates a Retrieval Augmented Generation (RAG) system using Ollama for local large language models, LangChain for orchestration, ChromaDB for vector storage, and Streamlit for the user interface.

It allows users to ask questions about a PDF document. The Streamlit interface (`pdf-rag-streamlit.py`) provides a GUI to upload a PDF and interact with the RAG system.

## Features

*   Load and process PDF documents.
*   Split documents into manageable chunks.
*   Generate embeddings using a local Ollama model (e.g., `nomic-embed-text`).
*   Store and retrieve document chunks using ChromaDB.
*   Utilize a local LLM via Ollama (e.g., `llama3.2`) for answering questions based on retrieved context.
*   Streamlit GUI for PDF upload and Q&A interaction.
*   A command-line version (`pdf-rag.py`) for basic RAG functionality (currently uses a hardcoded PDF).

## Prerequisites

*   **Python 3.8+**
*   **Ollama**: You need to have Ollama installed and running locally. You can download it from [https://ollama.com/](https://ollama.com/).
    *   After installing Ollama, pull the models you intend to use (e.g., for embeddings and generation):
        ```bash
        ollama pull nomic-embed-text
        ollama pull llama3.2 
        ```
        (Adjust model names as per your `MODEL_NAME` and `EMBED_MODEL_NAME` constants in the scripts if you use different ones).

## Project Structure