"""
Streamlit app for Aroma College Chatbot
"""
import os
import logging
import tempfile
import streamlit as st
from typing import List, Dict, Any

from src.ingestion.document_loader import DocumentLoader
from src.ingestion.text_processor import TextProcessor
from src.database.vector_store import VectorStore
from src.rag.retriever import RAGRetriever
from src.utils import get_ollama_base_url

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page title and description
st.set_page_config(
    page_title="Aroma College Chatbot",
    page_icon="🎓",
    layout="wide"
)

def check_ollama_connection() -> bool:
    """Check if Ollama server is running"""
    import requests
    ollama_url = get_ollama_base_url()
    try:
        response = requests.get(f"{ollama_url}/api/tags")
        if response.status_code == 200:
            return True
        else:
            st.error(f"Could not connect to Ollama server at {ollama_url}. Please make sure it's running.")
            return False
    except Exception as e:
        st.error(f"Error connecting to Ollama: {e}. Please make sure the server is running at {ollama_url}")
        return False

def initialize_components():
    """Initialize RAG components"""
    if "vector_store" not in st.session_state:
        try:
            st.session_state.vector_store = VectorStore()
            st.session_state.retriever = RAGRetriever(st.session_state.vector_store)
        except Exception as e:
            st.error(f"Error initializing components: {e}")
            return False
    return True

def process_uploaded_files(uploaded_files):
    """Process uploaded files and add to vector store"""
    with st.spinner("Processing uploaded files..."):
        document_loader = DocumentLoader()
        text_processor = TextProcessor()
        
        all_docs = []
        
        # Save uploaded files to temporary directory and process
        with tempfile.TemporaryDirectory() as temp_dir:
            for uploaded_file in uploaded_files:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Load document
                docs = document_loader.load_document(file_path)
                if docs:
                    all_docs.extend(docs)
        
        if all_docs:
            # Split documents
            processed_docs = text_processor.split_documents(all_docs)
            
            # Add to vector store
            st.session_state.vector_store.add_documents(processed_docs)
            
            # Reinitialize retriever with updated vector store
            st.session_state.retriever = RAGRetriever(st.session_state.vector_store)
            
            st.success(f"Successfully processed {len(uploaded_files)} files and added {len(processed_docs)} document chunks to the knowledge base.")
        else:
            st.warning("No content could be extracted from the uploaded files.")

def display_chat_history():
    """Display chat history"""
    for message in st.session_state.chat_history:
        role = message["role"]
        content = message["content"]
        
        with st.chat_message(role):
            st.write(content)

def main():
    """Main function for the Streamlit app"""
    st.title("🎓 Aroma College Chatbot")
    st.markdown("""
    Ask questions about Aroma College and get instant answers! 
    Upload college documents to enhance the chatbot's knowledge base.
    """)
    
    # Check if Ollama server is running
    if not check_ollama_connection():
        st.stop()
    
    # Initialize RAG components
    if not initialize_components():
        st.stop()
    
    # Initialize chat history if it doesn't exist
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "assistant", "content": "Hello! I'm the Aroma College Chatbot. How can I help you today?"}
        ]
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("Document Upload")
        st.markdown("Upload documents about Aroma College to enhance the chatbot's knowledge.")
        
        uploaded_files = st.file_uploader(
            "Upload PDF, DOCX, TXT, or CSV files",
            type=["pdf", "docx", "txt", "csv"],
            accept_multiple_files=True
        )
        
        if uploaded_files and st.button("Process Files"):
            process_uploaded_files(uploaded_files)
        
        st.divider()
        st.markdown("""
        ### About
        This chatbot uses RAG (Retrieval-Augmented Generation) to answer questions about Aroma College based on the provided documents.
        """)
    
    # Display chat messages
    display_chat_history()
    
    # User input
    user_query = st.chat_input("Ask a question about Aroma College")
    
    if user_query:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_query)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.retriever.answer_question(user_query)
                    answer = response.get("answer", "I couldn't find an answer to your question.")
                    st.write(answer)
                    
                    # Add assistant message to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                except Exception as e:
                    error_msg = f"Error generating response: {e}"
                    logger.error(error_msg)
                    st.error(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": "I encountered an error while processing your question. Please try again."})

if __name__ == "__main__":
    main()
