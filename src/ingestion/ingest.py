"""
Data ingestion script for Aroma College Chatbot
"""
import os
import argparse
import logging

from src.ingestion.document_loader import DocumentLoader
from src.ingestion.text_processor import TextProcessor
from src.database.vector_store import VectorStore

logger = logging.getLogger(__name__)

def ingest_data(data_dir: str):
    """
    Ingest data from a directory into the vector store
    
    Args:
        data_dir: Directory containing data files
    """
    logger.info(f"Starting data ingestion from {data_dir}")
    
    # Validate data directory
    if not os.path.exists(data_dir):
        logger.error(f"Directory not found: {data_dir}")
        return False
    
    try:
        # Initialize components
        document_loader = DocumentLoader()
        text_processor = TextProcessor()
        vector_store = VectorStore()
        
        # Load documents
        logger.info(f"Loading documents from {data_dir}")
        documents = document_loader.load_documents_from_directory(data_dir)
        logger.info(f"Loaded {len(documents)} documents")
        
        if not documents:
            logger.warning("No documents were loaded")
            return False
        
        # Process documents
        processed_docs = text_processor.split_documents(documents)
        logger.info(f"Created {len(processed_docs)} document chunks")
        
        # Add documents to vector store
        vector_store.add_documents(processed_docs)
        logger.info("Documents successfully ingested into vector store")
        
        return True
    
    except Exception as e:
        logger.error(f"Error during data ingestion: {e}")
        return False

def main():
    """Entry point for data ingestion"""
    parser = argparse.ArgumentParser(description="Ingest data into vector store")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing data files")
    args = parser.parse_args()
    
    success = ingest_data(args.data_dir)
    
    if success:
        logger.info("Data ingestion completed successfully")
    else:
        logger.error("Data ingestion failed")

if __name__ == "__main__":
    main()
