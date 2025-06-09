"""
Vector database functionality for Aroma College Chatbot
"""
import os
import logging
from typing import List, Dict, Any, Optional
from langchain_chroma import Chroma
# Using proper import to avoid deprecation warnings
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    # Fallback to legacy import
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

from src.utils import get_embedding_model, get_vector_db_path

logger = logging.getLogger(__name__)

class VectorStore:
    """Vector database for storing and retrieving embeddings"""
    
    def __init__(self, persist_directory: Optional[str] = None):
        """
        Initialize the vector store
        
        Args:
            persist_directory: Directory to persist vector store
        """
        try:
            # Using sentence transformers for embeddings
            model_name = get_embedding_model()
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'}
            )
        except Exception as e:
            logger.warning(f"Error loading specified model: {e}")
            logger.info("Falling back to default sentence-transformers model")
            # Fallback to a model known to be included with sentence-transformers
            self.embeddings = HuggingFaceEmbeddings(
                model_name="paraphrase-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
        
        self.persist_directory = persist_directory or get_vector_db_path()
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize or load vector store
        if os.path.exists(self.persist_directory) and len(os.listdir(self.persist_directory)) > 0:
            logger.info(f"Loading existing vector store from {self.persist_directory}")
            self.db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        else:
            logger.info(f"Creating new vector store at {self.persist_directory}")
            self.db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the vector store
        
        Args:
            documents: List of documents to add
        """
        if not documents:
            logger.warning("No documents to add to vector store")
            return
        
        logger.info(f"Adding {len(documents)} documents to vector store")
        self.db.add_documents(documents)
        logger.info("Vector store updated")
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        Perform similarity search on the vector store
        
        Args:
            query: Query string
            k: Number of results to return
            
        Returns:
            List of similar documents
        """
        logger.info(f"Performing similarity search for query: {query}")
        return self.db.similarity_search(query, k=k)
    
    def get_relevant_documents(self, query: str, k: int = 5) -> List[Document]:
        """
        Get documents relevant to a query
        
        Args:
            query: Query string
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        return self.similarity_search(query, k=k)
