import os
import sys
import click
import numpy
import psycopg
from groq import Groq
from psycopg.rows import dict_row
from transfomers import AutoTokenizer
from pgvector.psycopg import register_vector
from typing import List, Tuple, Generator, Iterable
from sentence_transformers import SentenceTransformer
from xmlchunker import Chunk, XMLChunker


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
        self._ensure_schema()

    def _ensure_schema(self):
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

    def upsert_batch(
        self,
        batch_rows: List[Tuple[str, int, int, str, list]]
    ) -> None:
        if not batch_rows:
            return
        with psycopg.connect(self.url) as connection:
            with connection.cursor() as cursor:
                cursor.executemany(
                    f"""
                    INSERT INTO {self.table}
                        (chunk_id, page, character_offset, content, text_embedding)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        page = EXCLUDED.page,
                        character_offset = EXCLUDED.character_offset,
                        content = EXCLUDED.content,
                        text_embedding = EXCLUDED.text_embedding;
                    """,
                    batch_rows
                )
            connection.commit()

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

        out: List[Tuple[Chunk, float]] = []
        for r in rows:
            _chunk = Chunk(
                id=r["chunk_id"],
                content=r["content"],
                metadata={
                    "page": r["page"],
                    "character_offset": r["character_offset"]
                }
            )
            similarity = float(1.0 - r["cosine_distance"])
            out.append((_chunk, similarity))
        return out


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
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model)
        except Exception:
            self.tokenizer = None

    def encode_embed(
        self,
        text: List[str],
        batch_size: int = 50
    ) -> numpy.ndarray:
        embeds = self.model.encode(
            text,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        # numpy.set_printoptions(threshold=numpy.inf,
        #                        precision=6, suppress=True, linewidth=180)
        # print(embeds[0])
        return numpy.asarray(embeds, dtype=numpy.float32)

    def count_tokens(
        self,
        text: str
    ) -> int:
        if not text:
            return 0

        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text, add_special_tokens=False))

        return max(1, len(text.split()))


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
