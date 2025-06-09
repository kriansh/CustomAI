"""
Direct Ollama integration for Aroma College Chatbot
"""
import logging
import ollama
from typing import Dict, Any, List, Optional

from src.utils import get_ollama_base_url, get_llm_model

logger = logging.getLogger(__name__)

class OllamaGenerator:
    """Direct integration with Ollama API"""
    
    def __init__(self):
        """Initialize the Ollama generator"""
        self.base_url = get_ollama_base_url()
        self.model = get_llm_model()
        # The host is set directly in the client calls
        logger.info(f"Initialized Ollama generator with model {self.model} at {self.base_url}")
    
    def generate(self, prompt: str, system_prompt: str = None, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Generate a response from Ollama
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            
        Returns:
            Generated response as string
        """
        try:
            logger.info(f"Generating response with model {self.model}")
            
            # Create message payload
            messages = []
            if chat_history:
                messages.extend(chat_history)
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            messages.append({
                "role": "user",
                "content": prompt
            })
            
            # Call Ollama API with host URL
            client = ollama.Client(host=self.base_url)
            response = client.chat(
                model=self.model,
                messages=messages
            )
            
            # Extract response text
            if response and "message" in response and "content" in response["message"]:
                return response["message"]["content"]
            else:
                logger.error(f"Unexpected response structure: {response}")
                return "Error: Unexpected response structure from Ollama."
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"
            
    def generate_stream(self, prompt: str, system_prompt: str = None, chat_history: Optional[List[Dict[str, str]]] = None):
        """
        Generate a streaming response from Ollama
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            
        Returns:
            Generator yielding response chunks as they arrive
        """
        try:
            logger.info(f"Generating streaming response with model {self.model}")
            
            # Create message payload
            messages = []
            if chat_history:
                messages.extend(chat_history)
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            messages.append({
                "role": "user",
                "content": prompt
            })
            
            # Call Ollama API with host URL in streaming mode
            client = ollama.Client(host=self.base_url)
            response_stream = client.chat(
                model=self.model,
                messages=messages,
                stream=True
            )
            
            # Extract and yield each chunk
            for chunk in response_stream:
                if chunk and "message" in chunk and "content" in chunk["message"]:
                    yield chunk["message"]["content"]
                
        except Exception as e:
            logger.error(f"Error generating streaming response: {e}")
            yield f"Error generating response: {str(e)}"
    
    def is_general_question(self, question: str) -> bool:
        """
        Check if this is a general question not related to Aroma College
        
        Args:
            question: The question to check
            
        Returns:
            True if this is a general question, False if it's about Aroma College
        """
        # List of keywords related to Aroma College
        college_keywords = [
            "aroma", "college", "university", "program", "degree", "course", 
            "admission", "faculty", "campus", "tuition", "scholarship", "student",
            "class", "lecture", "school", "academic", "education", "nepal"
        ]
        
        # Check for greeting patterns
        greeting_patterns = [
            "hi ", "hello", "hey ", "greetings", "good morning", "good afternoon", 
            "good evening", "how are you", "what's up", "who are you", "what can you do", 
            "help me", "your name", "introduce", "tell me about yourself"
        ]
        
        # Convert to lowercase for matching
        question_lower = question.lower()
        
        # Check if any college keyword is in the question
        has_college_keyword = any(keyword in question_lower for keyword in college_keywords)
        
        # Check if it's a greeting
        is_greeting = any(pattern in question_lower for pattern in greeting_patterns)
        
        # If it's a greeting or doesn't have college keywords, it's a general question
        return is_greeting or not has_college_keyword
    
    def answer_general_question(self, question: str) -> str:
        """
        Answer a general question not related to Aroma College
        
        Args:
            question: The general question
            
        Returns:
            Generated answer
        """
        system_prompt = """You are AromaAI, an AI assistant for Aroma College. You're helpful, respectful, and concise.
        When answering general questions, you can be helpful, but always mention that your primary purpose is to provide information about Aroma College."""
        
        prompt = f"""QUESTION: {question}

Provide a helpful response. Remember to be concise and mention that your primary purpose is to provide information about Aroma College."""
        
        return self.generate(prompt=prompt, system_prompt=system_prompt)

    def answer_with_context(self, question: str, context_docs: List[str]) -> str:
        """
        Answer a question using a list of context documents
        
        Args:
            question: Question to answer
            context_docs: List of context documents
            
        Returns:
            Generated answer
        """
        # Check if this is a general question
        if self.is_general_question(question):
            logger.info("Detected general question, using fallback model")
            return self.answer_general_question(question)
        
        # Aroma College specific question - use RAG with context
        logger.info("Answering Aroma College question using RAG")
        
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
        
        return self.generate(prompt=prompt, system_prompt=system_prompt)
