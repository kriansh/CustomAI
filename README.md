# Aroma College Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for answering questions about Aroma College using locally hosted Ollama models.

## Prerequisites

1. [Ollama](https://ollama.ai/download) must be installed and running on your machine
2. Python 3.8+ with pip and virtual environment

## Setup

1. Create and activate the virtual environment:
   ```
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   source .venv/bin/activate  # On macOS/Linux
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download Ollama model (in a separate terminal):
   ```
   ollama pull llama2     # For LLM completion
   ```
   
   Note: For embeddings, we're using the `all-MiniLM-L6-v2` model from the sentence-transformers library.

## Usage

### 1. Add college data

Place your college documents (PDF, DOCX, TXT, CSV) in the `docs` directory.

### 2. Ingest the data

Process and index the documents:
```
python main.py ingest --data-dir=./docs
```

### 3. Run the chatbot

You can use the chatbot in two ways:

- **Web Interface**: Start the Streamlit app
  ```
  python main.py web
  ```
  Then open your browser at http://localhost:8501

- **Command Line**: Ask questions directly from the terminal
  ```
  python main.py chat --question="What are the admission requirements for Aroma College?"
  ```

## Project Structure

- `main.py`: Entry point for the application
- `requirements.txt`: Project dependencies
- `data/`: Directory for storing vector database
- `docs/`: Directory for college documents
- `src/`: Source code
  - `ingestion/`: Document loading and processing
  - `database/`: Vector database components
  - `rag/`: Retrieval and question answering
  - `ui/`: Streamlit web interface
