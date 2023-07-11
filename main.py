import openai
import os
from dotenv import load_dotenv
from pathlib import Path
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI

# Load the API key from .env
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)
openai.api_key = os.environ['OPENAI_API_KEY']

# Load and split the documents
pdf_path = "./testfile.pdf"
loader = PyPDFLoader(pdf_path)
pages = loader.load_and_split()

# Create the embeddings
embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(pages, embedding=embeddings, persist_directory=".")
vectordb.persist()
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
pdf_qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.2), vectordb.as_retriever(), memory=memory)

# Query the model
query = "Does the insurance cover birth complications?"
result = pdf_qa({"question": query})
print("Answer:" + result["answer"])





