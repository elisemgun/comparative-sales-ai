import openai
import os
import csv
from dotenv import load_dotenv
from pathlib import Path

from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain, RetrievalQAWithSourcesChain, RetrievalQA
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
        loader = PyPDFLoader(pdf)
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
        # model = RetrievalQAWithSourcesChain.from_llm(llm=ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo'), retriever=vectordb.as_retriever())

        return model

    @staticmethod
    def query(model, pdf, health_service):
        response = openai.ChatCompletion.create(
            messages=[
                {'role': 'system',
                 'content': f"Given the insurance policy document {pdf_path}, extract specific coverage details. "
                            f"Return the prices, terms, and the exact sections from where these details were derived "
                            f"for the type of coverage I will specify. For instance, if I say 'dental', your response "
                            f"should be along the lines of: 'Emergency dental care covered up to $1500. Regular care "
                            f"covered $100 per year. No deductibles. Source: Page 3, 23.' Ensure the information is "
                            f"precise and derived directly from the document."
                 },
                {'role': 'user', 'content': "repatriation"},
            ],
            model='gpt-3.5-turbo',
            temperature=0,
        )
        print(response['choices'][0]['message']['content'])

        return response

    def start(self, pdf):
        # Creates vector database and passes it to language model
        vectordb = self.load_doc(pdf)
        model = self.create_llm(vectordb)
        # List of health services to query
        services = ["Maximum cover (overall coverage limit)", "Outpatient limit",
                    "Geographical coverage options",
                    "Network coverage", "Deductible/Co-pay", "Age limit", "Surgeries (inpatient & outpatient)",
                    "Dental", "Vision", "Screenings & Vaccines", "Medical History Disregarded", "Maternity",
                    "Complementary therapies (massage therapy, osteopaths, chiropodists and podiatrists, "
                    "chiropractors, homeopaths, dietitian and acupuncture)",
                    "Outpatient psychiatric, psychologist or therapeutic treatment", "Repatriation & Evacuation",
                    "Prescription Medication (inpatient & outpatient)", "Home health nursing",
                    "Diagnostic study services (CT, PET scans, etc.)", "Hospital cash benefit", "Allergy treatments"]

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
