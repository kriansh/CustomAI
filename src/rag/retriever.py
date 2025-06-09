"""
Retrieval components for Aroma College Chatbot
"""
import logging
from typing import List, Dict, Any, Optional

from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Import our direct Ollama integration
from src.rag.direct_ollama import OllamaGenerator

from src.database.vector_store import VectorStore
from src.utils import get_ollama_base_url, get_llm_model

logger = logging.getLogger(__name__)

# Define a custom prompt template that includes context about Aroma College
QA_PROMPT_TEMPLATE = """You are an AI assistant for Aroma College. You are helpful, respectful, and honest. 
You should answer questions based solely on the provided context about Aroma College, and avoid making up information.

CONTEXT:
{context}

QUESTION: {question}

If you don't know the answer or can't find it in the context, say "I don't have enough information about that in my knowledge base." 
Never make up information about Aroma College that is not present in the context.

ANSWER:"""

class RAGRetriever:
    """Retrieval-Augmented Generation for question answering"""
    
    def __init__(self, vector_store: VectorStore = None):
        """
        Initialize the RAG retriever
        
        Args:
            vector_store: Vector store for embeddings
        """
        # Initialize direct Ollama generator
        self.ollama = OllamaGenerator()
        
        # Initialize or use provided vector store
        self.vector_store = vector_store or VectorStore()
    
    def answer_question(self, question: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Answer a question using RAG
        
        Args:
            question: Question to answer
            
        Returns:
            Dictionary with question and answer
        """
        logger.info(f"Answering question: {question}")
        try:
            # Retrieve relevant documents using vector store
            retriever = self.vector_store.db.as_retriever(
                search_type="similarity", 
                search_kwargs={"k": 5}
            )
            docs = retriever.get_relevant_documents(question)
            
            # Extract the content from documents
            context_docs = [doc.page_content for doc in docs]
            
            # Generate answer using Ollama with retrieved context
            context_text = "\n\n".join(context_docs)
            system_prompt = QA_PROMPT_TEMPLATE.split("CONTEXT:")[0].strip()
            prompt_with_context = f"CONTEXT:\n{context_text}\n\nQUESTION: {question}\n\nANSWER:"
            
            answer = self.ollama.generate(prompt=prompt_with_context, system_prompt=system_prompt, chat_history=chat_history)
            
            return {
                "question": question,
                "answer": answer,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return {
                "question": question,
                "answer": f"I encountered an error while trying to answer your question. Please try again.",
                "success": False,
                "error": str(e)
            }
            
    def stream_answer(self, question: str, chat_history: Optional[List[Dict[str, str]]] = None):
        """
        Stream the answer to a question using RAG
        
        Args:
            question: Question to answer
            
        Returns:
            Generator yielding response chunks as they arrive
        """
        logger.info(f"Streaming answer to question: {question}")
        try:
            # Retrieve relevant documents using vector store
            retriever = self.vector_store.db.as_retriever(
                search_type="similarity", 
                search_kwargs={"k": 5}
            )
            docs = retriever.get_relevant_documents(question)
            
            # Extract the content from documents
            context_docs = [doc.page_content for doc in docs]
            
            # Check if this is a general question
            if self.ollama.is_general_question(question):
                logger.info("Detected general question, streaming with general model")
                system_prompt = """You are AromaAI, an AI assistant for Aroma College. You're helpful, respectful, and concise.
                When answering general questions, you can be helpful, but always mention that your primary purpose is to provide information about Aroma College."""
                
                prompt = f"""QUESTION: {question}

Provide a helpful response. Remember to be concise and mention that your primary purpose is to provide information about Aroma College."""
                
                # Stream the response
                return self.ollama.generate_stream(prompt=prompt, system_prompt=system_prompt, chat_history=chat_history)
            
            # Aroma College specific question - use RAG with context
            logger.info("Streaming answer to Aroma College question using RAG")
            
            # Create a prompt with the context
            context_text = "\n\n".join(context_docs)
            
            system_prompt = """You are an AI assistant for Aroma College. You are helpful, respectful, and honest.
You should answer questions based solely on the provided context about Aroma College, and avoid making up information."""
            
            prompt = f"""
CONTEXT:
{context_text}

QUESTION: {question}

If you don't know the answer or can't find it in the context, say "I don't have enough information about that in my knowledge base." 
Never make up information about Aroma College that is not present in the context.
"""
            
            # Stream the response
            return self.ollama.generate_stream(prompt=prompt, system_prompt=system_prompt, chat_history=chat_history)
            
        except Exception as e:
            logger.error(f"Error streaming answer: {e}")
            yield f"I encountered an error while trying to answer your question. Please try again. Error: {str(e)}"
