"""
Text processing utilities for Aroma College Chatbot
"""
import logging
from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

class TextProcessor:
    """Process and split text for efficient retrieval"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the text processor
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks for processing
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of split documents
        """
        logger.info(f"Splitting {len(documents)} documents into chunks")
        try:
            split_docs = self.text_splitter.split_documents(documents)
            logger.info(f"Created {len(split_docs)} document chunks")
            return split_docs
        except Exception as e:
            logger.error(f"Error splitting documents: {e}")
            return documents
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text to improve quality
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Remove extra whitespace
        text = " ".join(text.split())
        return text
