#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AromaAI: Aroma College Chatbot
A RAG-based chatbot for answering questions about Aroma College
"""

import os
import argparse
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def run_flask_app():
    """Run the Flask app"""
    try:
        import subprocess
        import sys
        
        logger.info("Starting Flask app from C:\\Users\\User\\Documents\\aromaAI\\src\\ui\\flask_app.py")
        subprocess.run([sys.executable, "src/ui/flask_app.py"])
    except Exception as e:
        logger.error(f"Error running Flask app: {e}")


def ingest_data(data_dir):
    """Ingest data from a directory"""
    from src.ingestion.ingest import ingest_data as run_ingestion
    
    if not os.path.exists(data_dir):
        logger.error(f"Data directory not found: {data_dir}")
        return
    
    logger.info(f"Starting data ingestion from {data_dir}")
    success = run_ingestion(data_dir)
    
    if success:
        logger.info("Data ingestion completed successfully")
    else:
        logger.error("Data ingestion failed")


def run_cli_chat(question):
    """Run CLI chat mode"""
    try:
        logger.info("Initializing CLI chat mode")
        from src.rag.retriever import RAGRetriever
        
        print("\nAroma College Chatbot is thinking...\n")
        retriever = RAGRetriever()
        result = retriever.answer_question(question)
        
        print("\n" + "=" * 60)
        print(f"Question: {result['question']}")
        print("-" * 60)
        print(f"Answer: {result['answer']}")
        print("=" * 60 + "\n")
    except Exception as e:
        logger.error(f"Error in CLI chat mode: {e}")
        print(f"\nError: {str(e)}\n")


def main():
    """Main entry point for the application"""
    parser = argparse.ArgumentParser(
        description="Aroma College Chatbot - A RAG-based chatbot for answering questions about Aroma College"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Streamlit web app
    web_parser = subparsers.add_parser("web", help="Run the web interface")
    
    # Data ingestion
    ingest_parser = subparsers.add_parser("ingest", help="Ingest data into the vector store")
    ingest_parser.add_argument(
        "--data-dir", 
        type=str, 
        required=True, 
        help="Directory containing data files"
    )
    
    # CLI chat
    chat_parser = subparsers.add_parser("chat", help="Chat with the bot using CLI")
    chat_parser.add_argument(
        "--question", 
        type=str, 
        required=True, 
        help="Question to ask"
    )
    
    args = parser.parse_args()
    
    if args.command == "web":
        run_flask_app()
    elif args.command == "ingest" and args.data_dir:
        ingest_data(args.data_dir)
    elif args.command == "chat" and args.question:
        run_cli_chat(args.question)
    else:
        print("Welcome to Aroma College Chatbot!")
        print("Use one of the following commands:")
        print("  python main.py web          - Run the web interface")
        print("  python main.py ingest --data-dir=./docs  - Ingest data")
        print("  python main.py chat --question='What are the admission requirements?'  - Ask a question")


if __name__ == "__main__":
    main()