"""
Flask app for Aroma College Chatbot
"""
import os
import logging
import tempfile
from typing import List, Dict, Any
from flask import Flask, request, jsonify, render_template, send_from_directory, redirect, url_for, session, flash
from functools import wraps
import werkzeug.utils

import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.ingestion.document_loader import DocumentLoader
from src.ingestion.text_processor import TextProcessor
from src.database.vector_store import VectorStore
import shutil
from src.rag.retriever import RAGRetriever
from src.utils import get_ollama_base_url, get_llm_model, set_env_variable

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__, 
            template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates"),
            static_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), "static"))

# Configure app
app.secret_key = "aroma_college_chatbot_secret_key"  # Required for sessions
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "adminkriaroma"

# Global variables
vector_store = None
retriever = None


def check_ollama_connection() -> bool:
    """Check if Ollama server is running"""
    import requests
    ollama_url = get_ollama_base_url()
    try:
        response = requests.get(f"{ollama_url}/api/tags")
        if response.status_code == 200:
            return True
        else:
            logger.error(f"Could not connect to Ollama server at {ollama_url}")
            return False
    except Exception as e:
        logger.error(f"Error connecting to Ollama: {e}")
        return False


def initialize_components() -> bool:
    """Initialize RAG components"""
    global vector_store, retriever
    try:
        vector_store = VectorStore()
        retriever = RAGRetriever(vector_store)
        return True
    except Exception as e:
        logger.error(f"Error initializing components: {e}")
        return False


def process_uploaded_files(files) -> Dict[str, Any]:
    """Process uploaded files and add to vector store"""
    global vector_store, retriever
    
    try:
        document_loader = DocumentLoader()
        text_processor = TextProcessor()
        
        all_docs = []
        
        # Save uploaded files to temporary directory and process
        with tempfile.TemporaryDirectory() as temp_dir:
            for file in files:
                if file and werkzeug.utils.secure_filename(file.filename):
                    file_path = os.path.join(temp_dir, werkzeug.utils.secure_filename(file.filename))
                    file.save(file_path)
                    
                    # Load document
                    docs = document_loader.load_document(file_path)
                    if docs:
                        all_docs.extend(docs)
        
        if all_docs:
            # Split documents
            processed_docs = text_processor.split_documents(all_docs)
            
            # Add to vector store
            vector_store.add_documents(processed_docs)
            
            # Reinitialize retriever with updated vector store
            retriever = RAGRetriever(vector_store)
            
            return {"success": True, "file_count": len(files), "chunk_count": len(processed_docs)}
        else:
            return {"success": False, "error": "No content could be extracted from the uploaded files."}
    
    except Exception as e:
        logger.error(f"Error processing files: {e}")
        return {"success": False, "error": str(e)}


# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def home():
    """Home page route - now serves landing.html"""
    return render_template('landing.html')

@app.route('/chat')
def chat_page():
    """Chat page route"""
    return render_template('index.html')

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    """Admin login route"""
    error = None
    if request.method == 'POST':
        if request.form['username'] == ADMIN_USERNAME and request.form['password'] == ADMIN_PASSWORD:
            session['logged_in'] = True
            return redirect(url_for('admin_dashboard'))
        else:
            error = 'Invalid credentials. Please try again.'
    return render_template('admin_login.html', error=error)

@app.route('/admin')
@login_required
def admin_dashboard():
    """Admin dashboard route"""
    return render_template('admin.html')

@app.route('/admin/logout')
def admin_logout():
    """Admin logout route"""
    session.pop('logged_in', None)
    return redirect(url_for('home'))

@app.route('/admin/clear_knowledge_base', methods=['POST'])
@login_required
def clear_knowledge_base():
    """API endpoint to clear the knowledge base (vector database)."""
    global vector_store, retriever
    try:
        db_path = get_vector_db_path()
        if os.path.exists(db_path):
            logger.info(f"Deleting existing vector database at: {db_path}")
            shutil.rmtree(db_path)
        
        logger.info(f"Re-creating empty vector database directory at: {db_path}")
        os.makedirs(db_path, exist_ok=True)
        
        logger.info("Re-initializing VectorStore and RAGRetriever after clearing knowledge base.")
        vector_store = VectorStore(persist_directory=db_path)
        retriever = RAGRetriever(vector_store)
        
        return jsonify({"success": True, "message": "Knowledge base cleared successfully. Please re-upload your files."})
    except Exception as e:
        logger.error(f"Error clearing knowledge base: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/chat', methods=['POST'])
def chat():
    """API endpoint for answering questions"""
    try:
        data = request.json
        question = data.get('question', '')
        stream = data.get('stream', False)
        chat_history = data.get('chat_history', [])
        
        if not question:
            return jsonify({"success": False, "error": "No question provided."})
        
        if not retriever:
            return jsonify({"success": False, "error": "Chatbot not initialized. Try refreshing the page."})
        
        if stream:
            return jsonify({"success": False, "error": "Please use the streaming endpoint /api/chat/stream"})
        
        # Answer the question
        response = retriever.answer_question(question, chat_history=chat_history)
        
        return jsonify({
            "success": True,
            "question": question,
            "answer": response.get("answer", "I couldn't find an answer to your question.")
        })
    
    except Exception as e:
        logger.error(f"Error in chat API: {e}")
        return jsonify({"success": False, "error": str(e)})


def generate_stream_response(generator):
    """Generate a streaming response from a generator"""
    for chunk in generator:
        if chunk:
            # Properly format as Server-Sent Events (SSE)
            yield f"data: {chunk}\n\n"


@app.route('/api/chat/stream', methods=['POST'])
def chat_stream():
    """Streaming API endpoint for answering questions"""
    try:
        data = request.json
        question = data.get('question', '')
        chat_history = data.get('chat_history', [])
        
        if not question:
            return jsonify({"success": False, "error": "No question provided."})
        
        if not retriever:
            return jsonify({"success": False, "error": "Chatbot not initialized. Try refreshing the page."})
        
        # Stream the answer
        response_generator = retriever.stream_answer(question, chat_history=chat_history)
        
        response = app.response_class(
            generate_stream_response(response_generator),
            mimetype='text/event-stream'
        )
        
        # Add necessary headers for SSE
        response.headers['Cache-Control'] = 'no-cache'
        response.headers['X-Accel-Buffering'] = 'no'  # Disable buffering for nginx
        response.headers['Connection'] = 'keep-alive'
        
        return response
    
    except Exception as e:
        logger.error(f"Error in streaming chat API: {e}")
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/upload_files', methods=['POST'])
@login_required
def upload_files():
    """API endpoint for uploading files - requires admin login"""
    if 'files[]' not in request.files:
        return jsonify({"success": False, "error": "No files provided."})
    
    files = request.files.getlist('files[]')
    
    if not files or len(files) == 0:
        return jsonify({"success": False, "error": "No files provided."})
    
    result = process_uploaded_files(files)
    return jsonify(result)


@app.route('/api/get_current_model', methods=['GET'])
def get_current_model():
    """API endpoint to get the current LLM model."""
    try:
        current_model = os.getenv("LLM_MODEL", "llama2") # Default to llama2 if not set
        return jsonify({"success": True, "model": current_model})
    except Exception as e:
        logger.error(f"Error fetching current model: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/update_model', methods=['POST'])
def update_model():
    """API endpoint for updating the model"""
    try:
        data = request.json
        model = data.get('model', '')
        
        if not model:
            return jsonify({"success": False, "error": "No model provided."})
        
        # Update the model in .env
        set_env_variable("LLM_MODEL", model)
        
        # Re-initialize the retriever with the new model
        global retriever
        retriever = RAGRetriever(vector_store)
        
        return jsonify({"success": True, "model": model})
    
    except Exception as e:
        logger.error(f"Error updating model: {e}")
        return jsonify({"success": False, "error": str(e)})


def main():
    """Initialize the app and run it"""
    # Check if Ollama server is running
    if not check_ollama_connection():
        logger.error("Failed to connect to Ollama server. Exiting.")
        return
    
    # Initialize RAG components
    if not initialize_components():
        logger.error("Failed to initialize components. Exiting.")
        return
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)


if __name__ == "__main__":
    main()
