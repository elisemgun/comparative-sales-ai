"""
Note! This version only works on langchain version 0.0.198 or earlier
"""
import openai
import os
import csv
from dotenv import load_dotenv
from pathlib import Path

from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
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
        model1 = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), vectordb.as_retriever(), memory=memory)
        model = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="refine", retriever=vectordb.as_retriever(),
                                            return_source_documents=True)
        # model = RetrievalQAWithSourcesChain.from_llm(llm=ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo'),
        # retriever=vectordb.as_retriever())

        return model

    @staticmethod
    def query(model, pdf, health_service):
        query = f"Using only the information provided in the {pdf} policy document, give the coverage for " \
                f" {health_service}. If you can't find the answer say you dont know instead of making up an answer." \
                f"Provide coverage in terms of either a percentage, yes/no or a dollar amount. Provide the relevant" \
                f" page numbers as source, and quote the section in the document where this information can be found." \
                f"The references should be on the format 'Answer: [answer to question] Page: [page numbers]'"

        result = model({"query": query})
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
                print(f"{health_service}: " + result["result"])
                print(f"Page: {result['source_documents']}")
                writer.writerow([health_service, result["result"]])


if __name__ == "__main__":
    main = Main()
    pdf_path = "docs/premium-community-plan-2.0.pdf"
    main.start(pdf_path)
