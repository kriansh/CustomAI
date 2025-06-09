"""
Utility functions for Aroma College Chatbot
"""
import os
import logging
import re
from typing import Optional
from dotenv import load_dotenv, set_key

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables (force reload)
load_dotenv(override=True)

# Path to .env file
ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')

def get_ollama_base_url():
    """Get Ollama base URL from environment variables"""
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

def get_embedding_model():
    """Get embedding model name from environment variables"""
    return os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

def get_llm_model():
    """Get LLM model name from environment variables"""
    return os.getenv("LLM_MODEL", "llama2")

def get_vector_db_path():
    """Get vector database path from environment variables"""
    return os.getenv("VECTOR_DB_PATH", "./data/vectordb")

def set_env_variable(key: str, value: str) -> bool:
    """Set an environment variable in .env file and current process
    
    Args:
        key: Environment variable name
        value: Environment variable value
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Update the .env file
        set_key(ENV_PATH, key, value)
        
        # Also update in current process environment
        os.environ[key] = value
        
        # Force reload environment variables
        load_dotenv(override=True)
        
        logger.info(f"Updated environment variable {key}={value}")
        return True
    except Exception as e:
        logger.error(f"Error updating environment variable: {e}")
        return False
