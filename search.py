"""
Contains the core logic for the search functionality.
Provides functions to rank strings based on relatedness to a query,
generate messages for the GPT model, and obtain answers using the
GPT model and a dataframe of relevant texts and embeddings.
"""

import openai
import pandas as pd
import ast

from constants import *
from utils import strings_ranked_by_relatedness, num_tokens


def query_message(
        query: str,
        df: pd.DataFrame,
        model: str,
        token_budget: int
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    strings, relatedness = strings_ranked_by_relatedness(query, df)
    introduction = ('Use the below insurance policy document to answer the subsequent question. If the answer cannot '
                    'be found in the articles, write "I could not find an answer."')
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        next_article = f'\n\nInsurance policy document:\n"""\n{string}\n"""'
        if (
                num_tokens(message + next_article + question, model=model)
                > token_budget
        ):
            break
        else:
            message += next_article
    return message + question


def ask(
        query: str,
        df: pd.DataFrame = None,
        model: str = GPT_MODEL,
        token_budget: int = 4096 - 500,
        print_message: bool = False,
) -> str:
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""

    if df is None:
        df = pd.read_csv(EMBEDDINGS_PATH)
        df['embedding'] = df['embedding'].apply(ast.literal_eval)

    message = query_message(query, df, model=model, token_budget=token_budget)
    if print_message:
        print(message)

    messages = [
        {"role": "system",
         "content": "You find the level of coverage of different health services in insurance policies."},
        {"role": "user", "content": message},
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0
    )
    response_message = response["choices"][0]["message"]["content"]
    return response_message
