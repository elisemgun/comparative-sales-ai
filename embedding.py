"""
Contains functions related to generating and using embeddings.
Includes functionality to convert chunks of text into embeddings using
the OpenAI API and also provides utility functions for working with embeddings.
"""

import openai
import pandas as pd
from utils import num_tokens, truncated_string, halved_by_delimiter

from constants import *


def generate_embeddings(split_pages):
    embeddings = []
    for batch_start in range(0, len(split_pages), BATCH_SIZE):
        batch_end = batch_start + BATCH_SIZE
        batch = split_pages[batch_start:batch_end]

        response = openai.Embedding.create(model=EMBEDDING_MODEL, input=batch)

        for i, be in enumerate(response["data"]):
            assert i == be["index"]
        batch_embeddings = [e["embedding"] for e in response["data"]]
        embeddings.extend(batch_embeddings)

    df = pd.DataFrame({"text": split_pages, "embedding": embeddings})
    return df


def split_strings_from_subsection(
        subsection: tuple[list[str], str],
        max_tokens: int = 1000,
        model: str = GPT_MODEL,
        max_recursion: int = 5,
) -> list[str]:
    """
    Split a subsection into a list of subsections, each with no more than max_tokens.
    Each subsection is a tuple of parent titles [H1, H2, ...] and text (str).
    """
    titles, text = subsection
    string = "\n\n".join(titles + [text])
    num_tokens_in_string = num_tokens(string)
    # if length is fine, return string
    if num_tokens_in_string <= max_tokens:
        return [string]
    # if recursion hasn't found a split after X iterations, just truncate
    elif max_recursion == 0:
        return [truncated_string(string, model=model, max_tokens=max_tokens)]
    # otherwise, split in half and recurse
    else:
        titles, text = subsection
        for delimiter in ["\n\n", "\n", ". "]:
            left, right = halved_by_delimiter(text, delimiter=delimiter)
            if left == "" or right == "":
                # if either half is empty, retry with a more fine-grained delimiter
                continue
            else:
                # recurse on each half
                results = []
                for half in [left, right]:
                    half_subsection = (titles, half)
                    half_strings = split_strings_from_subsection(
                        half_subsection,
                        max_tokens=max_tokens,
                        model=model,
                        max_recursion=max_recursion - 1,
                    )
                    results.extend(half_strings)
                return results
    # otherwise no split was found, so just truncate (should be very rare)
    return [truncated_string(string, model=model, max_tokens=max_tokens)]
