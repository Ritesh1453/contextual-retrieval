# Contextual Retrieval System

A Python-based document retrieval system that combines ChromaDB for vector storage, Sentence Transformers for embeddings, and Google's Gemini for context generation and question answering.

## Features

- PDF document processing and text extraction using MarkItDown
- Smart document chunking with configurable size and overlap
- Context-aware document retrieval using ChromaDB
- Intelligent context generation using Google Gemini
- Semantic search capabilities with similarity scoring
- Batch processing for efficient document handling

## Prerequisites

- Python 3.12 or higher
- Google Gemini API key

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd contextual-retrieval
```

2. Install dependencies using uv:
```bash
uv pip install sentence-transformers chromadb numpy tqdm google-generativeai pypdf markdown
```

Or install using the provided `pyproject.toml`:
```bash
uv pip install .
```

3. Set up your Google Gemini API key as an environment variable:
```bash
export GEMINI_API_KEY='your-api-key-here'
```

## Usage

### Basic Usage

```python
from contextual_retrieval import ContextualRetrieval

# Initialize the system
retrieval_system = ContextualRetrieval()

# Add a document
retrieval_system.add_document("Document Title", "Your document content here")

# Search for relevant content
results = retrieval_system.search("Your search query")
```

### Processing PDF Documents

```python
from main import PDFProcessor

# Initialize the processor
processor = PDFProcessor('data')

# Process all PDFs in the data folder
processor.process_folder()
```

### Configuration

- Default chunk size: 500 words
- Chunk overlap: 50 words
- Batch size: 32 documents
- Embedding model: 'all-MiniLM-L6-v2'
- Vector similarity metric: cosine similarity

## Project Structure

- `contextual_retrieval.py`: Core retrieval system implementation
- `main.py`: PDF processing and example usage
- `data/`: Directory for storing PDF documents
- `chroma_db/`: Persistent storage for document embeddings

## Features in Detail

### Document Processing
- Automatic document chunking with configurable overlap
- Context generation for each chunk using Gemini
- Metadata extraction and storage

### Search Capabilities
- Semantic search using sentence transformers
- Similarity scoring
- Configurable number of results
- Rich metadata retrieval

### Document Management
- Add new documents
- Delete existing documents
- Retrieve document chunks
- Batch processing support

