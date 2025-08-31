# Created by AG on 31-08-2025

import os
import sys
import re
import math
import click
import numpy
import psycopg2
from groq import Groq
from pathlib import Path
from pypdf import PdfReader
from psycopg2 import dict_row
from dataclasses import dataclass
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
        with psycopg2.connect(self.url, autocommit=True) as connection:
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
                            USING ivfflat (embedding vector_cosine_similarity)
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

        with psycopg2.connect(self.url, autocommit=True) as connection:
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
        with psycopg2.connect(self.url, row_factory=dict_row) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT chunk_id, page, character_offset, content,
                        (text_embedding <=> %s) AS cosine_distance
                    FROM {self.table} ORDER BY text_embedding <=> %s
                    LIMIT %s;
                    """,
                    (query_vector, query_vector, top_k)
                )
                rows = cursor.fetchall()

        get_chunks: List[Tuple[Chunk, float]] = []
        for row in rows:
            chunk = Chunk(
                id=row["id"],
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
        return numpy.ndarray(embeds, dtype=numpy.float32)


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
