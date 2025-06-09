"""
Document loading utilities for Aroma College Chatbot
"""
import os
import logging
from typing import List, Dict, Any
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    CSVLoader
)
from langchain.schema import Document

logger = logging.getLogger(__name__)

class DocumentLoader:
    """Load and process documents from various formats"""
    
    def __init__(self):
        """Initialize the document loader"""
        self.supported_extensions = {
            '.pdf': PyPDFLoader,
            '.txt': TextLoader,
            '.docx': Docx2txtLoader,
            '.csv': CSVLoader
        }
    
    def load_document(self, file_path: str) -> List[Document]:
        """Load a single document based on file extension"""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext not in self.supported_extensions:
            logger.warning(f"Unsupported file extension: {ext} for file {file_path}")
            return []
        
        try:
            loader_cls = self.supported_extensions[ext]
            loader = loader_cls(file_path)
            return loader.load()
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {e}")
            return []
    
    def load_documents_from_directory(self, directory_path: str) -> List[Document]:
        """Load all documents from a directory"""
        documents = []
        
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                ext = os.path.splitext(file_path)[1].lower()
                
                if ext in self.supported_extensions:
                    logger.info(f"Loading {file_path}")
                    docs = self.load_document(file_path)
                    documents.extend(docs)
        
        return documents
