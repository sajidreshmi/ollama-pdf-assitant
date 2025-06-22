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

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd ollama-rag-fundatmentals-py
   ```

2. Create a virtual environment (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Streamlit Web Interface (Recommended)

Run the Streamlit application for an interactive web interface:

```bash
# Method 1: Using streamlit command (if available in PATH)
streamlit run pdf-rag-streamlit.py

# Method 2: Using python module (recommended if streamlit command not found)
python3 -m streamlit run pdf-rag-streamlit.py
```

This will start a web server (typically at `http://localhost:8501`) where you can:
1. Upload a PDF document
2. Ask questions about the document
3. Get AI-powered answers based on the document content

### Command Line Interface

For basic RAG functionality without GUI:

```bash
python3 pdf-rag.py
```

**Note**: The command-line version currently uses a hardcoded PDF path. Make sure to update the `DOC_PATH` variable in the script to point to your desired PDF file.

## Troubleshooting

### Streamlit Command Not Found

If you get `command not found: streamlit`, use the module approach:
```bash
python3 -m streamlit run pdf-rag-streamlit.py
```

### Virtual Environment Issues

Make sure your virtual environment is activated and streamlit is installed:
```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install streamlit
```

### Ollama Connection Issues

Ensure Ollama is running and the required models are pulled:
```bash
# Check if Ollama is running
ollama list

# Pull required models if not available
ollama pull nomic-embed-text
ollama pull llama3.2
```

## Project Structure

```
ollama-rag-fundatmentals-py/
├── README.md
├── requirements.txt
├── pdf-rag-streamlit.py    # Streamlit web interface
├── pdf-rag.py             # Command-line interface
├── data/                  # Directory for PDF files
└── chroma_db_streamlit/   # ChromaDB storage (created automatically)
```