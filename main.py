from constants import EMBEDDINGS_PATH
from pdf_processing import extract_pages_from_pdf
from embedding import generate_embeddings
from prompts import health_services
from search import ask
import pandas as pd
import csv
import ast

split_pages, num_pages = extract_pages_from_pdf('docs/premium-community-plan-2.0.pdf')
# print(f"{num_pages} PDF pages split into {len(split_pages)} strings.")

df = generate_embeddings(split_pages)
df.to_csv(EMBEDDINGS_PATH, index=False)

# Reload the dataframe
df = pd.read_csv(EMBEDDINGS_PATH)
df['embedding'] = df['embedding'].apply(ast.literal_eval)

with open('coverage.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Health Service", "Coverage"])

    for service in health_services:
        query = f"Using only the information provided in the insurance policy document, give the coverage for " \
                f" {service}. If you can't find the answer say you dont know instead of making up an answer." \
                f"Provide coverage in terms of either a percentage, yes/no or a dollar amount. Provide the relevant" \
                f" page numbers as source, and quote the section in the document where this information can be found." \
                f"The references should be on the format 'Answer: [answer to question] Page: [page numbers]'"

        response = ask(query, df)
        writer.writerow([service, response])

