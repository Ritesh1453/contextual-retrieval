# First install additional dependencies using uv:
# uv pip install pypdf markdown

import os
from markitdown import MarkItDown
from contextual_retrieval import ContextualRetrieval

class PDFProcessor:
    def __init__(self, data_folder: str):
        self.data_folder = data_folder
        self.retriever = ContextualRetrieval()
        self.md_converter = MarkItDown()  # Initialize MarkItDown instance

    def convert_pdf_to_text(self, pdf_path: str) -> str:
        """Convert PDF file to text using MarkItDown"""
        try:
            result = self.md_converter.convert(pdf_path)
            return result.text_content  # Get the converted text content
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            return ""

    def process_folder(self):
        """Process all PDF files in the data folder"""
        for filename in os.listdir(self.data_folder):
            if filename.endswith('.pdf'):
                file_path = os.path.join(self.data_folder, filename)
                print(f"Processing {filename}...")
                
                # Extract text from PDF using MarkItDown
                text = self.convert_pdf_to_text(file_path)
                print("Got markdown")
                # Add to database
                title = os.path.splitext(filename)[0]
                self.retriever.add_document(title, text)
                
                print(f"Successfully processed {filename}")

def main():
    # Initialize processor with data folder path
    processor = PDFProcessor('data')
    
    # Process all PDFs
    # processor.process_folder()
    
    # Test the retrieval
    print("\nTesting retrieval...")
    test_query = "hhow does gptuner suggest i tune my db config?"
    import os
    import google.generativeai as genai

    # Configure Google Gemini API
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

    # Use the results to generate an answer
    results = processor.retriever.search(test_query)

    # Prepare a context summary from the search results
    context_summary = "\n\n".join(
        f"Similarity: {result['similarity']:.4f}\nChunk: {result['chunk'][:500]}"
        for result in results
    )

    # Generate a prompt for Gemini
    prompt = f"""
    You are an AI assistant tasked with answering questions based on a document retrieval system. 
    Below are some relevant document chunks retrieved for the query:

    {context_summary}

    Based on the above information, provide a clear and concise answer to the following question:

    '{test_query}'
    """

    # Generate an answer using Gemini
    chat_session = model.start_chat()
    gemini_response = chat_session.send_message(prompt)
    generated_answer = gemini_response.text.strip()  # Extract the generated text

    # Print the generated answer
    print(f"\nGenerated Answer:\n{generated_answer}")


if __name__ == "__main__":
    main()
