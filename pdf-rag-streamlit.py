# app.py

import streamlit as st
import os
import logging
import tempfile # Added for temporary file handling
import shutil # Added for directory removal
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import ollama

# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants
DOC_PATH = "./data/BOI.pdf"
MODEL_NAME = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "simple-rag"
PERSIST_DIRECTORY = "./chroma_db_streamlit" # Changed to avoid conflict if running other scripts


def ingest_pdf(uploaded_file):
    """Load PDF documents from an uploaded file object."""
    if uploaded_file is not None:
        try:
            # Create a temporary file to store the uploaded PDF content
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            loader = UnstructuredPDFLoader(file_path=tmp_file_path)
            data = loader.load()
            logging.info(f"PDF '{uploaded_file.name}' loaded successfully.")
            os.remove(tmp_file_path) # Clean up the temporary file
            return data
        except Exception as e:
            logging.error(f"Error loading PDF '{uploaded_file.name}': {e}")
            st.error(f"Error processing PDF: {e}")
            return None
    return None


def split_documents(documents):
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    chunks = text_splitter.split_documents(documents)
    logging.info("Documents split into chunks.")
    return chunks


@st.cache_resource(show_spinner="Processing PDF and building vector store...")
def load_vector_db(_uploaded_file_name, uploaded_file_content):
    """Load or create the vector database from uploaded PDF content."""
    # Use _uploaded_file_name or hash of content for cache invalidation if needed
    # For simplicity, this version rebuilds if PERSIST_DIRECTORY is cleared or content changes

    # Pull the embedding model if not already available
    ollama.pull(EMBEDDING_MODEL)

    embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)

    # If the persistence directory exists, remove it to ensure fresh processing for new uploads
    # A more sophisticated approach might involve checking if the PDF is the same
    # or using different collection names / persist directories per PDF.
    if os.path.exists(PERSIST_DIRECTORY):
        shutil.rmtree(PERSIST_DIRECTORY)
        logging.info(f"Cleared existing vector database at {PERSIST_DIRECTORY}.")
    
    os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

    # Create a temporary file from the uploaded content for UnstructuredPDFLoader
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file_content)
        tmp_file_path = tmp_file.name

    try:
        loader = UnstructuredPDFLoader(file_path=tmp_file_path)
        data = loader.load()
        if not data:
            st.error("Could not load any content from the PDF.")
            return None
        logging.info("PDF loaded successfully for vectorization.")

        # Split the documents into chunks
        chunks = split_documents(data)
        if not chunks:
            st.error("Could not split the document into chunks.")
            return None

        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            collection_name=VECTOR_STORE_NAME + "_" + _uploaded_file_name.replace(".", "_").replace(" ", "_"), # Unique collection name
            persist_directory=PERSIST_DIRECTORY,
        )
        # vector_db.persist() # Persist is implicitly handled by Chroma usually, but can be explicit.
        logging.info("Vector database created and persisted.")
        return vector_db
    except Exception as e:
        st.error(f"Failed to create vector database: {e}")
        logging.error(f"Failed to create vector database: {e}")
        return None
    finally:
        os.remove(tmp_file_path) # Clean up the temporary file


def create_retriever(vector_db, llm):
    """Create a multi-query retriever."""
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five
different versions of the given user question to retrieve relevant documents from
a vector database. By generating multiple perspectives on the user question, your
goal is to help the user overcome some of the limitations of the distance-based
similarity search. Provide these alternative questions separated by newlines.
Original question: {question}""",
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
    )
    logging.info("Retriever created.")
    return retriever


def create_chain(retriever, llm):
    """Create the chain with preserved syntax."""
    # RAG prompt
    template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    logging.info("Chain created with preserved syntax.")
    return chain


def main():
    st.title("Document Assistant")

    uploaded_file = st.file_uploader("Upload your PDF document", type=["pdf"])

    if uploaded_file is not None:
        st.info(f"File '{uploaded_file.name}' uploaded. Processing...")
        
        # Get file content once
        uploaded_file_content = uploaded_file.getvalue()

        # Initialize the language model
        llm = ChatOllama(model=MODEL_NAME)

        # Load the vector database using the uploaded file's name and content
        # Pass file name for potential use in collection naming or caching key
        vector_db = load_vector_db(uploaded_file.name, uploaded_file_content)

        if vector_db is None:
            st.error("Failed to load or create the vector database from the uploaded PDF.")
            return
        else:
            st.success("PDF processed and vector database is ready!")

        # User input for questions, only if PDF is processed
        user_input = st.text_input("Enter your question about the document:", "")

        if user_input:
            with st.spinner("Generating response..."):
                try:
                    # Create the retriever
                    retriever = create_retriever(vector_db, llm)

                    # Create the chain
                    chain = create_chain(retriever, llm)

                    # Get the response
                    response = chain.invoke(input=user_input)

                    st.markdown("**Assistant:**")
                    st.write(response)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        elif vector_db is not None: # Check if vector_db is not None before prompting for question
             st.info("Please enter a question about the uploaded document.")

    else:
        st.info("Please upload a PDF document to get started.")


if __name__ == "__main__":
    main()