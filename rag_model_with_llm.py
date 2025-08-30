# Created by AG on 30-08-2025

import os
import sys
import re
import math
import click
from dataclasses import dataclass
from typing import List, Tuple
from pathlib import Path
from groq import Groq
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


@dataclass
class Chunk:
    id: str
    content: str
    metadata: dict


# Extracts texts from pdf, breaks them into chunks where each chunk
# is a Chunk object having Chunk(id=id, content=content, metadata=metadata).
# Each Chunk essentially has a chunk id of the format "p00x-o0y" where,
# x = page number and y = offset. Each Chunk also has a metadata field where,
# metadata has {"page": x, "offset": y}
def create_chunks(
    file_path: str,
    max_characters: int = 500,
    intersections: int = 100
) -> List[Chunk]:
    pdfreader = PdfReader(file_path)
    chunks = []
    for page_no, page in enumerate(pdfreader.pages, start=1):
        raw = page.extract_text() or ""
        text = re.sub(r"\s+\n|\n+", "\n", raw).strip()
        if not text:
            continue
        index = 0
        # Sliding Window Technique
        while index < len(text):
            window = text[index: index + max_characters]
            cid = f"p{page_no:03d}-o{index:05d}"
            chunks.append(Chunk(id=cid, content=window.strip(),
                          metadata={"page": page_no, "offset": index}))
            if index + max_characters >= len(text):
                break
            index += max_characters - intersections
    return chunks


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


def create_retrieval_context(
    ChunkBlocks: List[Tuple[Chunk, float]],
    max_characters=2000
) -> str:
    buffer, used, total = [], set(), 0
    for character, score in ChunkBlocks:
        if character.id in used:
            continue
        create_line = f"[Chunk_id: {character.id} page: {
            character.metadata.get('page')}] {character.content.strip()}\n"
        if total + len(create_line) > max_characters:
            break
        buffer.append(create_line)
        used.add(character.id)
        total += len(create_line)
    return "".join(buffer)


def answer_pass_threshold(
    ChunkBlocks,
    min_similarity=0.14,
    min_support=2,
    support_similarity=0.06
) -> bool:
    if not ChunkBlocks:
        return False
    top_similarity = ChunkBlocks[0][1]
    if top_similarity < min_similarity:
        return False
    supporter = sum(1 for _, s in ChunkBlocks if s >= support_similarity)
    return supporter >= min_support


fallback = "Don't know the answer"

PROMPT = """You are a strict RAG assistant.
RULES:
- Answer ONLY using the provided CONTEXT.
- If the answer is not fully contained in CONTEXT, reply exactly:
  "I don't know from the answer."
- Keep answers concise and cite chunk IDs like [Chunk_id: id] inline for claims
"""

CONTEXT_WITH_QUESTION = """QUESTION:
{question}

CONTEXT (verbatim excerpts):
{context}
"""


def groq() -> Groq:
    key = os.getenv("GROQ_API_KEY")
    return Groq(api_key=key)


def chat(
    client: Groq,
    user_prompt: str,
    model="qwen/qwen3-32b",
) -> str:
    response = client.chat.completions.create(
        model=model,
        temperature=0.1,
        messages=[
            {
                "role": "system", "content": PROMPT
            },
            {
                "role": "user", "content": user_prompt
            }
        ],
        reasoning_format="hidden"
    )
    return response.choices[0].message.content.strip()


def get_response(
    query: str,
    retriever: RAGRetriever,
    client: Groq,
    min_similarity=0.14,
    min_support=2,
    support_similarity=0.06
) -> str:
    ChunkBlocks = retriever.search(query, top_k=6)

    if not answer_pass_threshold(
        ChunkBlocks,
        min_similarity,
        min_support,
        support_similarity
    ):
        return fallback

    context = create_retrieval_context(ChunkBlocks, max_characters=4000)
    response = chat(
        client,
        CONTEXT_WITH_QUESTION.format(question=query, context=context)
    )

    if fallback in response:
        return fallback
    if "[Chunk_id:" not in response:
        return fallback
    return response


@click.command()
@click.argument("pdf_file_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("question", type=str)
# @click.option("--model", default=lambda: os.getenv("GROQ_MODEL", "qwen2.5-32b-instruct"),
#               help="Groq model name (e.g., qwen2.5-32b-instruct)")
@click.option("--max-characters", default=500)
@click.option("--intersection", default=100)
@click.option("--min-similarity", default=0.14)
@click.option("--support-similarity", default=0.06)
@click.option("--min-support", default=2)
def script(
        pdf_file_path,
        question,
        max_characters,
        intersection,
        min_similarity,
        support_similarity,
        min_support
):
    chunks = create_chunks(
        pdf_file_path,
        max_characters=max_characters,
        intersections=intersection
    )
    if not chunks:
        click.echo("No text extracted from PDF.", err=True)
        sys.exit(1)

    retriever = RAGRetriever(chunks)
    client = groq()
    ans = get_response(
        query=question,
        retriever=retriever,
        client=client,
        min_similarity=min_similarity,
        min_support=min_support,
        support_similarity=support_similarity
    )
    click.echo(ans)


if __name__ == "__main__":
    script()
