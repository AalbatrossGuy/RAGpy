# Created by AG on 30-08-2025

"""
    Objectives:
        -> Extract text from pdfs using PyPDF2
        -> Convert the texts into chunks
        -> Convert them into a vector array using TF-IDF method
        -> Create a search module where top-k chunks are retrieved by the given
            query using TF-IDF method and cosine similarity
        -> Run a very small NLP model that converts the found paragraphs into
            a more human sentence structure.
"""

import sys
import click
from pypdf import PdfReader
from typing import List, Tuple
from transformers import pipeline
from dataclasses import dataclass
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class Chunk:
    content: str
    index: int


# Extract text from PDFs
def extract_text(
        file_path: str
) -> str:
    pdfreader = PdfReader(file_path)
    return "\n".join(page.extract_text() or "" for page in pdfreader.pages)


# Split the texts into a chunks array
def create_chunks(
    text: str,
    chunk_size: int = 350,
    intersected_texts: int = 55
) -> List[str]:
    get_words = text.split()
    chunks_array, index = [], 0
    while index < len(get_words):
        chunk = get_words[index:index + chunk_size]
        if not chunk:
            break
        chunks_array.append(" ".join(chunk))
        index += chunk_size - intersected_texts
    return chunks_array


class RAGRetriever:
    def __init__(
        self,
        chunks: List[Chunk]
    ):
        # TfidfVectorizer() converts text, etc. into vectors using TF-IDF
        # method.This allows storing data in a high-dimensional space from
        # where we can calculate their similarity by measuring their vector
        # distance or cosine similarity which is basically finding the angle
        # between the two vectors. TfidfVectorizer() tokenizes texts into
        # words and ignores common end-words by default. It also assigns
        # weight to each word based on it's importance, repetition, etc.
        self.vectorizer = TfidfVectorizer()
        self.document_contents = [chunk.content for chunk in chunks]
        self.chunks = chunks
        # It converts the chunks into a TF-IDF vector matrix where each row is
        # one chunk and each column is one word and their value is the tf-idf
        # weight of that word in that certain chunk. Therefore, the shape of
        #
        self.matrix = self.vectorizer.fit_transform(
            self.document_contents
        )
        # Normalize each row vector in the matrix because when comparing using
        # cosine similarity, the normalization would ensure fairness in the
        # comparisons regardless of the chunk's length.
        self.matrix = normalize(self.matrix)

    def search(
        self,
        search_query: str,
        top_k: int = 6
    ) -> List[Tuple[Chunk, float]]:
        # Convert query to a vector matrix the same way so we can measure
        # their distance using cosine similarity to find their similarity
        vectorized_query = self.vectorizer.transform([search_query])
        # The above vectorized_query can now be compared with the matrix.
        # Normalize vectorized_query so each row is same in order to have
        # fairness during cosine similarity.
        vectorized_query = vectorized_query / \
            (vectorized_query.power(2).sum() ** 0.5)
        # The above formula resizes the vector to have a unit length of 1
        similarity_index = (self.matrix @ vectorized_query.T).toarray().ravel()
        # The above gives a similarity index between search_query and each
        # chunk. @ -> does matrix multiplication. A High index score
        # would mean there's more occurance of important words
        sorted_similarity_index = similarity_index.argsort()[::-1]
        return [
            (self.chunks[i], float(similarity_index[i])) for i in
            sorted_similarity_index[:top_k]
        ]


@click.command()
@click.option("--pdf", "file_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--query", required=True)
@click.option("--k", default=6, show_default=True)
def script(file_path: str, query: str, k: int):
    extracted_text = extract_text(file_path=file_path)
    chunks = [
        Chunk(content=chunk, index=index)
        for index, chunk in enumerate(create_chunks(extracted_text))
    ]
    if not chunks:
        click.echo("Text couldn't be extracted", err=True)
        sys.exit(-1)

    rag_retriever = RAGRetriever(chunks)
    score = rag_retriever.search(search_query=query, top_k=k)
    context = " ".join([chunk.content for chunk, _ in score])
    # The above created the context that we will feed to the simple NLP
    # to make the output more human.
    local_model = pipeline("question-answering",
                           model="distilbert-base-uncased-distilled-squad")
    # Get the answer back from the model. The model only modifies the messages
    # to make them more structured. It doesn't generate any data.
    answer = local_model(question=query, context=context)
    click.echo(f"Query: {query}")
    click.echo(f"Response: {answer['answer']}")
    click.echo("\nContext: \n")
    for chunk, index in score:
        click.echo(
            f"[Chunk {chunk.index}, Score {
                index:.3f}]\n{chunk.content}\n"
        )


if __name__ == "__main__":
    script()
