# Created by AG on 04-08-2025
import re
import numpy
import blingfire
import xml.etree.ElementTree as ETree
from dataclasses import dataclass
from typing import List, Generator, Dict, Optional, Tuple

# Import Blingfire, if not found, fallback to simple regex
# try:
#     import blingfire
#     USE_BLING: bool = True
# except Exception:
#     USE_BLING = False


@dataclass
class Chunk:
    id: str
    content: str
    metadata: Dict[str, int]


def clean_whitespace(text: str) -> str:
    return re.compile(r"\s+").sub(" ", text).strip()


# Splits sentence in following cases - terminating punctuations, trailing
# quotes/brackets, whitespace gaps and if next sentence starts with uppercase
# or numbers. Open to modification as required.
FALLBACK_REGEX = re.compile(
    r"""
    (?<=\S[.!?])
    ["'\)\]]*
    \s+
    (?=[A-Z0-9\(])
    """,
    re.X
)


def cosine_similarity(
    array1: numpy.ndarray,
    array2: numpy.ndarray
):
    # The vector arrays need to be L2-normalised vectors for it to give
    # consistent values.
    return float(numpy.dot(array1, array2))


class XMLChunker:
    def __init__(
        self,
        # TODO: Add type hinting for embedder
        embedder,
        soft_target_token: int = 200,
        max_tokens: int = 400,
        intersections: int = 10,
        similarity_floor: float = 0.212,
        batch_size_embed: int = 256
    ):
        self.embedder = embedder
        self.target_tokens = soft_target_token
        self.max_tokens = max_tokens
        self.intersections = intersections
        self.similarity_floor = similarity_floor
        self.embed_per_batch = batch_size_embed

    @staticmethod
    def _sentence_start_position(
        sentences: List[str]
    ) -> List[int]:
        starts, counter = [], 0
        for start in sentences:
            starts.append(counter)
            counter += len(start.strip()) + 1
        return starts

    @staticmethod
    def _normalise_embeds(
            embeds
    ) -> numpy.ndarray:
        embed_array = numpy.asarray(embeds, dtype=numpy.float32)
        normalise = numpy.linalg.norm(
            embed_array, axis=1, keepdims=True
        ) + 1e-12
        return embed_array / normalise

    def _too_big(
        self,
        current_tokens: int,
        next_tokens: int
    ) -> bool:
        return current_tokens and (current_tokens + next_tokens > self.max_tokens)

    def _hit_boundary(
        self,
        current_tokens: int,
        similarity: Optional[float]
    ) -> bool:
        return (
            current_tokens >= self.target_tokens
            and similarity is not None
            and similarity < self.similarity_floor
        )

    def _flush_current_buffer(
        self,
        block_index: int,
        sentences: List[str],
        sentence_starts: List[int],
        indexes: List[int],
        current_tokens: int,
    ) -> Tuple[Optional[Chunk], List[int], int]:
        if not indexes:
            return None, indexes, current_tokens

        piece = " ".join(sentences[index].strip() for index in indexes).strip()
        start_char = sentence_starts[indexes[0]]
        chunk_id = f"[Chunk_id: {block_index:03d} page: {block_index}]"
        chunk = Chunk(
            id=chunk_id,
            content=piece,
            metadata={
                "page": block_index,
                "character_offset": int(start_char)
            },
        )

        if self.intersections > 0:
            keep = indexes[-self.intersections:]
            new_tokens = sum(self.embedder.count_tokens(
                sentences[i]) for i in keep
            )
            return chunk, keep[:], new_tokens

        return chunk, [], 0

    def iterate_chunks(
        self,
        xml_file_path: str
    ) -> Generator["Chunk", None, None]:

        block_index = 0
        bs = max(1, min(self.embed_per_batch, 16))

        for raw_text in self._iterate_through_content(xml_file_path):
            pure_text = clean_whitespace(raw_text)
            if not pure_text:
                block_index += 1
                continue

            sentences = self._semantic_sentences(pure_text)
            if not sentences:
                block_index += 1
                continue

            starts = self._sentence_start_position(sentences)
            current_indexes: List[int] = []
            current_tokens = 0
            previous_embed = None

            for i in range(0, len(sentences), bs):
                batch_sentences = sentences[i: i + bs]
                batch_embeds = self._normalise_embeds(
                    self.embedder.encode_embed(batch_sentences, batch_size=bs)
                )

                for k, embed in enumerate(batch_embeds):
                    index = i + k
                    token = self.embedder.count_tokens(sentences[index])

                    if self._too_big(current_tokens, token):
                        chunk, current_indexes, current_tokens = self._flush_current_buffer(
                            block_index,
                            sentences,
                            starts,
                            current_indexes,
                            current_tokens
                        )
                        if chunk:
                            yield chunk

                    cosine_similarity = float(
                        numpy.dot(previous_embed, embed)
                    ) if previous_embed is not None else None
                    if self._hit_boundary(current_tokens, cosine_similarity):
                        chunk, current_indexes, current_tokens = self._flush_current_buffer(
                            block_index,
                            sentences,
                            starts,
                            current_indexes,
                            current_tokens
                        )
                        if chunk:
                            yield chunk

                    current_indexes.append(index)
                    current_tokens += token
                    previous_embed = embed

            chunk, current_indexes, current_tokens = self._flush_current_buffer(
                block_index,
                sentences,
                starts,
                current_indexes,
                current_tokens
            )
            if chunk:
                yield chunk

            block_index += 1

    def _iterate_through_content(
        self,
        xml_file_path: str
    ) -> Generator[str, None, None]:
        for _, element in ETree.iterparse(
            xml_file_path,
            events=("end",)
        ):
            if element.tag.rsplit("}", 1)[-1].lower() == "content":
                content_parts: list = []
                for t in element.itertext():
                    if t and (string := t.strip()):
                        content_parts.append(string)
                if content_parts:
                    yield " ".join(content_parts)
            element.clear()

    def _semantic_sentences(
        self,
        text: str
    ) -> List[str]:
        if not text:
            return []
        # if USE_BLING:
        sentences = blingfire.text_to_sentences(text).strip().splitlines()
        return [sentence.strip() for sentence in sentences if sentence.strip()]
        # else:
        # parts = FALLBACK_REGEX.split(text)
        # stripped_sentence: List[str] = []
        # for part in parts:
        #   part = part.strip()
        #   if part:
        #       out.append(part)
        # return out

    def _pack_sentences_semantically(
        self,
        sentences: List[str],
        sentence_embeds: numpy.ndarray
    ) -> List[List[int]]:
        n: int = len(sentences)
        if n == 0:
            return []

        tokens = [
            self.embedder.count_tokens(sentence) for sentence in sentences
        ]

        chunks: List[List[int]] = []
        iter1 = 0
        while iter1 < n:
            start_pos = max(0, iter1 - self.intersections) \
                if (chunks and self.intersections > 0) else iter1
            current_indexes: List[int] = []
            current_tokens = 0

            iter2 = start_pos
            while iter2 < iter1:
                current_indexes.append(iter2)
                current_tokens += tokens[iter2]
                iter2 += 1

            while iter2 < n:
                if current_tokens + tokens[iter2] > self.max_tokens and current_indexes:
                    break

                if current_indexes:
                    previous_index = current_indexes[-1]
                    similarity = cosine_similarity(
                        sentence_embeds[previous_index],
                        sentence_embeds[iter2]
                    )
                    if current_tokens >= self.target_tokens and similarity < self.similarity_floor:
                        break

                current_indexes.append(iter2)
                current_tokens += tokens[iter2]
                iter2 += 1

                if current_tokens >= self.target_tokens and iter2 < n:
                    next_similarity = cosine_similarity(
                        sentence_embeds[current_indexes[-1]],
                        sentence_embeds[iter2]
                    )
                    if next_similarity < self.similarity_floor:
                        break

            if not current_indexes:
                current_indexes = [iter1]
                iter2 = iter + 1

            chunks.append(current_indexes)
            iter1 = max(current_indexes[-1] + 1, iter2)

        return chunks
