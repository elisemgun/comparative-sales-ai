"""
Functionality related to processing PDF documents.
Contains functions for extracting text from PDF pages and splitting
the extracted text into manageable chunks for further processing.
"""

import PyPDF2
from constants import MAX_TOKENS
from embedding import split_strings_from_subsection


def extract_pages_from_pdf(pdf_path):
    file_object = open(pdf_path, 'rb')
    pdf_reader = PyPDF2.PdfReader(file_object)
    page_texts = [page.extract_text() for page in pdf_reader.pages]

    split_pages = []
    for i, text in enumerate(page_texts):
        titles = [f"Page {i + 1}"]
        section = (titles, text)
        split_pages.extend(split_strings_from_subsection(section, max_tokens=MAX_TOKENS))

    return split_pages, len(pdf_reader.pages)
