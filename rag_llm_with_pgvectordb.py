# Created by AG on 31-08-2025

import os
import sys
import re
import math
import click
import numpy
import psycopg
from groq import Groq
from pathlib import Path
from pypdf import PdfReader
from psycopg.rows import dict_row
from dataclasses import dataclass
from pgvector.psycopg import register_vector
from typing import List, Tuple, Dict, Optional
from sentence_transformers import SentenceTransformer


@dataclass
class Chunk:
    id: str
    content: str
    metadata: Dict[str, int]


def create_chunks(
        file_path: str,
        max_characters: int = 500,
        intersections: int = 100
) -> List[Chunk]:
    pdfreader = PdfReader(file_path)
    chunks: List = []
    for page_no, page in enumerate(pdfreader.pages, start=1):
        extracted_text = page.extract_text() or ""
        text = re.sub(r"\s+\n|\n+", "\n", extracted_text).strip()
        if not text:
            continue
        index = 0
        while index < len(text):
            sliding_window = text[index: index + max_characters]
            chunk_id = f"P{page_no:03d} -O{index:05d}"
            chunks.append(
                Chunk(
                    id=chunk_id,
                    content=sliding_window.strip(),
                    metadata={
                        "page": page_no,
                        "character_offset": index
                    }
                )
            )
            if index + max_characters >= len(text):
                break
            index += max_characters - intersections
    return chunks


class VectorDBStore:
    def __init__(
        self,
        url: str,
        table: str = "chunks",
        text_dim: int = 384
    ):
        self.url = url
        self.table = table
        self.dim = text_dim
        self._verify_schema()

    def _verify_schema(self):
        with psycopg.connect(self.url, autocommit=True) as connection:
            with connection.cursor() as cursor:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cursor.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.table} (
                        chunk_id TEXT PRIMARY KEY,
                        page INTEGER NOT NULL,
                        character_offset INTEGER NOT NULL,
                        content TEXT NOT NULL,
                        text_embedding VECTOR({self.dim}) NOT NULL
                    );
                    """
                )

                cursor.execute(
                    f"""
                    DO $$
                    BEGIN
                        IF NOT EXISTS(
                            SELECT 1 FROM pg_class info
                            JOIN pg_namespace name on name.oid = info.relnamespace
                            WHERE info.relname = '{self.table}_embedding_index'
                        ) THEN
                            CREATE INDEX {self.table}_embedding_index
                            ON {self.table}
                            USING ivfflat (text_embedding vector_cosine_ops)
                            WITH (lists = 100);
                        END IF;
                    END$$;
                    """
                )

    def upsert_chunks(
        self,
        chunks: List[Chunk],
        text_embeddings: numpy.ndarray
    ):
        assert len(chunks) == text_embeddings.shape[0], "Length mismatch"
        chunk_values: List[Tuple] = [
            (
                chunk.id,
                int(chunk.metadata.get("page", 0)),
                int(chunk.metadata.get("character_offset", 0)),
                chunk.content,
                text_embeddings[index].tolist(),
            )
            for index, chunk in enumerate(chunks)
        ]

        with psycopg.connect(self.url, autocommit=True) as connection:
            with connection.cursor() as cursor:
                cursor.executemany(
                    f"""
                    INSERT INTO {self.table} (chunk_id, page, character_offset, content, text_embedding)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        page = EXCLUDED.page,
                        character_offset = EXCLUDED.character_offset,
                        content = EXCLUDED.content,
                        text_embedding = EXCLUDED.text_embedding
                    """,
                    chunk_values
                )

    def search(
        self,
        embedded_query: numpy.ndarray,
        top_k: int = 6
    ) -> List[Tuple[Chunk, float]]:
        query_vector = embedded_query.tolist()
        with psycopg.connect(self.url, row_factory=dict_row) as connection:
            register_vector(connection)
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT chunk_id, page, character_offset, content,
                        (text_embedding <=> %s::vector) AS cosine_distance
                    FROM {self.table}
                    ORDER BY text_embedding <=> %s::vector
                    LIMIT %s;
                    """,
                    (query_vector, query_vector, top_k)
                )
                rows = cursor.fetchall()
                print(f"ROWS = {rows}\n\n")

        get_chunks: List[Tuple[Chunk, float]] = []
        for row in rows:
            chunk = Chunk(
                id=row["chunk_id"],
                content=row["content"],
                metadata={
                    "page": row["page"],
                    "character_offset": row["character_offset"]
                },
            )
            cosine_similarity = float(1.0 - row["cosine_distance"])
            get_chunks.append(
                (chunk, cosine_similarity)
            )
        return get_chunks


class Embed:
    def __init__(
        self,
        model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.model = SentenceTransformer(model)
        validation_vector = self.model.encode(
            ["validate"],
            normalize_embeddings=True
        )
        self.dim = int(validation_vector.shape[1])

    def encode_embed(
        self,
        text: List[str]
    ) -> numpy.ndarray:
        embeds = self.model.encode(
            text,
            normalize_embeddings=True
        )
        return numpy.asarray(embeds, dtype=numpy.float32)


class RAGRetrieverModel:
    def __init__(
        self,
        vectorStore: VectorDBStore,
        embedder: Embed
    ):
        self.vectorStore = vectorStore
        self.embedder = embedder

    def search(
        self,
        query: str,
        top_k: int = 6
    ) -> List[Tuple[Chunk, float]]:
        query = self.embedder.encode_embed([query])[0]
        return self.vectorStore.search(embedded_query=query, top_k=top_k)


def create_retrieve_context(
    ChunkBlock: List[Tuple[Chunk, float]],
    max_characters=1000
) -> str:
    buffer, used, total = [], set(), 0
    for character, score in ChunkBlock:
        if character.id in used:
            continue
        line = f"[Chunk_id: {character.id} page: {character.metadata.get('page')}] {
            character.content.strip()}\n"
        if total + len(line) > max_characters:
            break
        buffer.append(line)
        used.add(character.id)
        total += len(line)
    return "".join(buffer)


def answer_pass_threshold(
    ChunkBlocks,
    min_similarity=0.14,
    min_support=1,
    support_similarity=0.06
) -> bool:
    if not ChunkBlocks:
        return False
    top_similarity = ChunkBlocks[0][1]
    if top_similarity >= 0.40:
        return True
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
    retriever: RAGRetrieverModel,
    client: Groq,
    min_similarity=0.05,
    min_support=1,
    support_similarity=0.02
) -> str:
    ChunkBlocks = retriever.search(query, top_k=6)

    if not answer_pass_threshold(
        ChunkBlocks,
        min_similarity,
        min_support,
        support_similarity
    ):
        print('could not pass answer_pass_threshold')
        return fallback

    context = create_retrieve_context(ChunkBlocks, max_characters=1000)
    response = chat(
        client,
        CONTEXT_WITH_QUESTION.format(question=query, context=context)
    )
    # print(f"UNFILTERED RESPONSE = {response}")
    if fallback in response:
        print('fallback encountered')
        return fallback
    if "[Chunk_id:" not in response:
        print('Chunk_id not found in response')
        return fallback
    return response


@click.command()
@click.argument("pdf_file_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("question", type=str)
# @click.option("--model", default=lambda: os.getenv("GROQ_MODEL", "qwen2.5-32b-instruct"),
#               help="Groq model name (e.g., qwen2.5-32b-instruct)")
@click.option("--database-url", default="postgres://aalbatrossguy:pgadmin%40123@localhost:5432/vector_db")
@click.option("--max-characters", default=500)
@click.option("--intersection", default=100)
@click.option("--min-similarity", default=0.14)
@click.option("--support-similarity", default=0.06)
@click.option("--min-support", default=2)
@click.option("--skip-indexing", is_flag=True, default=False)
def script(
    pdf_file_path,
    question,
    database_url,
    max_characters,
    intersection,
    min_similarity,
    support_similarity,
    min_support,
    skip_indexing,
):
    embedder = Embed()
    create_vector_store = VectorDBStore(
        url=database_url,
        text_dim=embedder.dim
    )

    if not skip_indexing:
        chunks = create_chunks(
            file_path=pdf_file_path,
            max_characters=max_characters,
            intersections=intersection
        )
        if not chunks:
            click.echo("Text couldn't be extracted from PDF.", err=True)
            sys.exit(-1)
        content = [chunk.content for chunk in chunks]
        embeds = embedder.encode_embed(content)
        create_vector_store.upsert_chunks(
            chunks=chunks,
            text_embeddings=embeds
        )
        retriever = RAGRetrieverModel(
            vectorStore=create_vector_store,
            embedder=embedder
        )
        client = groq()
        response = get_response(
            query=question,
            retriever=retriever,
            client=client,
            min_similarity=min_similarity,
            min_support=min_support,
            support_similarity=support_similarity
        )
        click.echo(response)


if __name__ == "__main__":
    script()
