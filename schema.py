import openai
from langchain.chains import create_citation_fuzzy_match_chain
from langchain.chat_models import ChatOpenAI
from langchain.chains import create_extraction_chain, create_extraction_chain_pydantic
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma


pdf_path = "docs/premium-community-plan-2.0.pdf"
loader = PyPDFLoader(pdf_path)
pages = loader.load_and_split()

# Create the embeddings
embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(pages, embedding=embeddings, persist_directory=".")
vectordb.persist()

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")


schema = {
    "properties": {
        "dental": {"type": "string"},
        "dental_source": {"type": "string"},
        "repatriation": {"type": "string"},
        "repatriation_source": {"type": "string"},
    },
    "required": [],
}
input = vectordb
context = f"Given the insurance policy document {pdf_path}, extract specific coverage details. Return the prices, " \
          f"terms, and the exact sections from where these details were derived for the type of coverage I will " \
          f"specify. For instance, if I say 'dental', your response should be along the lines of: 'Emergency dental " \
          f"care covered up to $1500. Regular care covered $100 per year. No deductibles. Source: Page 3, 23.' Ensure " \
          f"the information is precise and derived directly from the document."

chain = create_extraction_chain(schema, llm)
chain.run(input)

services = ["Maximum cover (overall coverage limit)", "Outpatient limit",
            "Geographical coverage options",
            "Network coverage", "Deductible/Co-pay", "Age limit", "Surgeries (inpatient & outpatient)",
            "Dental", "Vision", "Screenings & Vaccines", "Medical History Disregarded", "Maternity",
            "Complementary therapies (massage therapy, osteopaths, chiropodists and podiatrists, "
            "chiropractors, homeopaths, dietitian and acupuncture)",
            "Outpatient psychiatric, psychologist or therapeutic treatment", "Repatriation & Evacuation",
            "Prescription Medication (inpatient & outpatient)", "Home health nursing",
            "Diagnostic study services (CT, PET scans, etc.)", "Hospital cash benefit", "Allergy treatments"]

print(response['choices'][0]['message']['content'])
