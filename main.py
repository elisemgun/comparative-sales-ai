from constants import EMBEDDINGS_PATH
from pdf_processing import extract_pages_from_pdf
from embedding import generate_embeddings
from prompts import health_services
from search import ask
import pandas as pd
import ast

split_pages, num_pages = extract_pages_from_pdf('docs/premium-community-plan-2.0.pdf')
# print(f"{num_pages} PDF pages split into {len(split_pages)} strings.")

df = generate_embeddings(split_pages)
df.to_csv(EMBEDDINGS_PATH, index=False)

# Reload the dataframe
df = pd.read_csv(EMBEDDINGS_PATH)
df['embedding'] = df['embedding'].apply(ast.literal_eval)

response = ask('What is the level of coverage for dental?', df, print_message=True)
print(response)
