# First install dependencies using uv:
# uv pip install sentence-transformers chromadb numpy tqdm google-generativeai

import chromadb
from chromadb.utils import embedding_functions
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict, Any
from tqdm import tqdm
import google.generativeai as genai

class ContextualRetrieval:
    def __init__(self, collection_name: str = "contextual_retrieval"):
        """Initialize ChromaDB and embedding model"""
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        # Initialize the embedding function
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name='all-MiniLM-L6-v2'
        )
        
        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        # Batch size for processing
        self.batch_size = 32
        
        # Configure Google Gemini API key
        genai.configure(api_key='YOUR_API_KEY')

    def chunk_document(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split document into overlapping chunks"""
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        return chunks

    def generate_context(self, document: str, chunk: str, title: str) -> Tuple[str, Dict[str, Any]]:
        """
        Generate context and metadata for a chunk using Google Gemini.
        Returns tuple of (contextualized_chunk, metadata)
        """
        # Construct the prompt using the specified format
        prompt = f"""<document> 
        {document}
        </document> 
        Here is the chunk we want to situate within the whole document: 
        <chunk> 
        {chunk}
        </chunk> 
        Please provide a short, succinct context to situate this chunk within the overall document 
        for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""

        # Ensure Google Gemini API is configured
        import os
        import google.generativeai as genai

        genai.configure(api_key=os.environ["GEMINI_API_KEY"])

        # Create the generation configuration
        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }

        # Create the model
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
        )

        # Use the model to generate context
        chat_session = model.start_chat()
        

        gemini_response = chat_session.send_message(prompt)
        generated_context = gemini_response.text.strip()  # Extract the generated text

        # Generate additional metadata
        

        contextualized_chunk = f"Context: {generated_context}.\nChunk: {chunk}"

        # Create metadata
        metadata = {
            "document_title": title,
            "chunk_length": len(chunk),
            "original_chunk": chunk,
            "generated_context": generated_context,  # Store the generated context in metadata
        }

        return contextualized_chunk, metadata


    def process_chunks_batch(self, chunks: List[str], document: str, title: str) -> Tuple[List[str], List[Dict], List[str]]:
        """Process a batch of chunks to prepare for ChromaDB"""
        contextualized_chunks = []
        metadata_list = []
        ids = []
        
        for i, chunk in enumerate(chunks):
            contextualized_chunk, metadata = self.generate_context(document, chunk, title)
            contextualized_chunks.append(contextualized_chunk)
            metadata_list.append(metadata)
            # Generate a unique ID for each chunk
            ids.append(f"{title}-chunk-{i}")
            
        return contextualized_chunks, metadata_list, ids

    def add_document(self, title: str, content: str):
        """Add a document to ChromaDB with batched processing"""
        # Create chunks
        chunks = self.chunk_document(content)
        
        # Process chunks in batches
        for i in tqdm(range(0, len(chunks), self.batch_size), desc="Processing document chunks"):
            batch_chunks = chunks[i:i + self.batch_size]
            
            # Process the batch
            contextualized_chunks, metadata_list, ids = self.process_chunks_batch(
                batch_chunks, content, title
            )
            
            # Add to ChromaDB
            self.collection.add(
                documents=contextualized_chunks,
                metadatas=metadata_list,
                ids=ids
            )

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks using ChromaDB.
        Returns list of dictionaries containing chunk, metadata, and distance.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["metadatas", "documents", "distances"]
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                'chunk': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'similarity': 1 - results['distances'][0][i]  # Convert distance to similarity
            })
            
        return formatted_results

    def get_document_chunks(self, title: str) -> List[Dict[str, Any]]:
        """Retrieve all chunks for a specific document"""
        results = self.collection.get(
            where={"document_title": title},
            include=["metadatas", "documents"]
        )
        
        return [{
            'chunk': doc,
            'metadata': meta
        } for doc, meta in zip(results['documents'], results['metadatas'])]

    def delete_document(self, title: str):
        """Delete all chunks associated with a document"""
        self.collection.delete(
            where={"document_title": title}
        )

# Example usage:
# retrieval_system = ContextualRetrieval()
# retrieval_system.add_document("Sample Title", "Your long document content goes here.")
# results = retrieval_system.search("Your search query")
