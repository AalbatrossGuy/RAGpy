# Created by AG on 04-08-2025
import re
import numpy
import blingfire
import xml.etree.ElementTree as ETree
from dataclasses import dataclass
from typing import List, Generator, Dict

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

    def iterate_chunks(
        self,
        xml_file_path: str
    ) -> Generator["Chunk", None, None]:
        block_index = 0
        for raw_text in self._iterate_through_content(xml_file_path):
            pure_text = clean_whitespace(raw_text)

            if not pure_text:
                block_index += 1
                continue

            sentences = self._semantic_sentences(pure_text)
            if not sentences:
                block_index += 1
                continue

            embed_sentences = self.embedder.encode_embed(
                sentences, batch_size=self.embed_per_batch)
            index_group = self._pack_sentences_semantically(
                sentences, embed_sentences)

            sentence_start_list: List[int] = []
            pos = 0
            for sentence in sentences:
                sentence_start_list.append(pos)
                pos += len(sentence.strip()) + 1

            for group in index_group:
                sentence_piece = " ".join(
                    sentences[index].strip() for index in group
                ).strip()
                starting_character = sentence_start_list[group[0]] \
                    if group else 0
                chunk_id = f"Chunk-{block_index:06d} Offset-{starting_character:08d}"
                yield Chunk(
                    id=chunk_id,
                    content=sentence_piece,
                    metadata={
                        "page": block_index,
                        "character_offset": int(starting_character)
                    }
                )
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
        iter = 0
        while iter < n:
            start_pos = max(0, iter - self.intersections) \
                if (chunks and self.intersections > 0) else iter
            current_indexes: List[int] = []
            current_tokens = 0

            iter2 = start_pos
            while iter2 < iter:
                current_indexes.append(iter2)
                current_tokens += tokens[iter2]

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
                current_indexes = list(iter)
                iter2 = iter + 1

            chunks.append(current_indexes)
            iter = max(current_indexes[-1] + 1, iter2)

        return chunks
