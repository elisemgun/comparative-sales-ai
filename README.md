# Search and extract info from PDFs 
This project uses the OpenAI API to extract text from PDF documents, embed the text, and then use the embeddings to search for and answer questions about the content of the PDF. The output is a .csv file with questions, answers and the sources for the answer.

## Usage:
1. pip install -r requirements.txt

2. Place your PDF document in the docs' directory.
3. Run main.py to start the extraction, embedding, and querying process.
4. Use the ask function in search.py to ask questions about the content of the PDF.

  

## Based these tutorials
**Reading PDFs:**  
https://levelup.gitconnected.com/chatgpt-for-pdf-files-with-langchain-ef565c041796  

**Search and ask algorithm:**  
https://github.com/openai/openai-cookbook/blob/3115683f14b3ed9570df01d721a2b01be6b0b066/examples/Embedding_Wikipedia_articles_for_search.ipynb

**Embeddings:**  
https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb


## Modules Overview
**constants.py**: Contains global constants such as model names, token limits, and batch sizes.  

**pdf_processing.py**: Provides functions for extracting and processing text from PDF documents.  

**embedding.py**: Contains functions related to generating and working with embeddings using the OpenAI API.  

**search.py**: Implements the core search functionality using embeddings and GPT.  

**utils.py**: Houses utility functions like token counting and string processing.  

**main.py**: The main entry point that orchestrates the flow of the application.  
