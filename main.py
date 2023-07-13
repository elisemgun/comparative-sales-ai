import openai
import os
import csv
from dotenv import load_dotenv
from pathlib import Path
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI


class Main:
    def __init__(self):
        # Load the API key from .env
        env_path = Path('.') / '.env'
        load_dotenv(dotenv_path=env_path)
        openai.api_key = os.environ['OPENAI_API_KEY']

    # Load document and create embeddings
    @staticmethod
    def load_doc(pdf):
        loader = PyPDFLoader(pdf)  # PDF-path set as global variable
        pages = loader.load_and_split()

        # Create the embeddings
        embeddings = OpenAIEmbeddings()
        vectordb = Chroma.from_documents(pages, embedding=embeddings, persist_directory=".")
        vectordb.persist()
        return vectordb

    @staticmethod
    def create_llm(vectordb):
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        model = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.1), vectordb.as_retriever(), memory=memory)
        return model

    @staticmethod
    def query(model, pdf, health_service):
        query = f"Using only the information provided in the {pdf} policy document, could you please tell me if" \
                f" the policy provides coverage for {health_service}? If it does, could you specify the extent of the" \
                f" coverage in terms of either a percentage, yes/no or a dollar amount? Please also provide the page " \
                f"numbers in the policy document where this information can be found." \
                f"It is important that the page number references are on the format 'Page: 2, 5, 13'"

        result = model({"question": query})
        return result

    def start(self, pdf):
        # Creates vector database and passes it to language model
        vectordb = self.load_doc(pdf)
        model = self.create_llm(vectordb)
        # List of health services to query
        services = ["dental", "repatriation"]

        # Creates CSV file of answers
        with open('coverage.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Health Service", "Coverage"])

            # Query the model
            for health_service in services:
                result = self.query(model, pdf, health_service=health_service)
                # print(f"{health_service}: " + result["answer"])
                writer.writerow([health_service, result["answer"]])


if __name__ == "__main__":
    main = Main()
    pdf_path = "docs/premium-community-plan-2.0.pdf"
    main.start(pdf_path)
