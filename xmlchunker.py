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

    # def iterate_chunks(
    #     self,
    #     xml_file_path: str
    # ) -> Generator["Chunk", None, None]:
    #     block_index = 0
    #     for raw_text in self._iterate_through_content(xml_file_path):
    #         pure_text = clean_whitespace(raw_text)
    #
    #         if not pure_text:
    #             block_index += 1
    #             continue
    #
    #         sentences = self._semantic_sentences(pure_text)
    #         if not sentences:
    #             block_index += 1
    #             continue
    #
    #         embed_sentences = self.embedder.encode_embed(
    #             sentences, batch_size=self.embed_per_batch)
    #         index_group = self._pack_sentences_semantically(
    #             sentences, embed_sentences)
    #
    #         sentence_start_list: List[int] = []
    #         pos = 0
    #         for sentence in sentences:
    #             sentence_start_list.append(pos)
    #             pos += len(sentence.strip()) + 1
    #
    #         for group in index_group:
    #             sentence_piece = " ".join(
    #                 sentences[index].strip() for index in group
    #             ).strip()
    #             starting_character = sentence_start_list[group[0]] \
    #                 if group else 0
    #             chunk_id = f"Chunk-{block_index:06d} Offset-{starting_character:08d}"
    #             yield Chunk(
    #                 id=chunk_id,
    #                 content=sentence_piece,
    #                 metadata={
    #                     "page": block_index,
    #                     "character_offset": int(starting_character)
    #                 }
    #             )
    #         block_index += 1

    def iterate_chunks(
        self,
        xml_file_path: str
    ) -> Generator["Chunk", None, None]:
        import numpy as np  # local import to avoid global dep if you swap backends
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

            # Precompute starts once (for offsets)
            sent_starts: List[int] = []
            acc = 0
            for s in sentences:
                sent_starts.append(acc)
                acc += len(s.strip()) + 1  # +1 for the join-space

            # Packing state (we build one chunk at a time)
            current_idxs: List[int] = []
            current_tokens = 0
            prev_embed = None  # last sentence embedding added to the chunk

            def flush_chunk():
                nonlocal current_idxs, current_tokens
                if not current_idxs:
                    return
                piece = " ".join(sentences[i].strip()
                                 for i in current_idxs).strip()
                start_char = sent_starts[current_idxs[0]]
                # chunk_id = f"Chunk-{block_index:06d} Offset-{start_char:08d}"
                chunk_id = f"[Chunk_id: {block_index:03d} page: {block_index}]"
                yield Chunk(
                    id=chunk_id,
                    content=piece,
                    metadata={"page": block_index,
                              "character_offset": int(start_char)},
                )
                # prepare overlap for next chunk
                if self.intersections > 0:
                    keep = current_idxs[-self.intersections:]
                    current_idxs = keep[:]
                    current_tokens = sum(self.embedder.count_tokens(
                        sentences[i]) for i in current_idxs)
                else:
                    current_idxs = []
                    current_tokens = 0

            # Stream the embeddings in micro-batches (keeps RAM flat)
            # tiny batch for small VMs
            bs = max(1, min(self.embed_per_batch, 16))
            i = 0
            while i < len(sentences):
                j = min(i + bs, len(sentences))
                batch_sents = sentences[i:j]
                batch_embeds = self.embedder.encode_embed(
                    batch_sents, batch_size=bs)

                # ensure L2-normalized embeddings (if your backend didn’t)
                if isinstance(batch_embeds, np.ndarray):
                    norms = np.linalg.norm(
                        batch_embeds, axis=1, keepdims=True) + 1e-12
                    batch_embeds = batch_embeds / norms

                for k, emb in enumerate(batch_embeds):
                    idx = i + k
                    tok = self.embedder.count_tokens(sentences[idx])

                    # hard cap → flush now
                    if current_idxs and current_tokens + tok > self.max_tokens:
                        for ch in flush_chunk():
                            yield ch

                    # semantic boundary preference (once soft target reached)
                    boundary = False
                    if current_idxs and prev_embed is not None:
                        # cos sim (embeddings are unit-normalized)
                        sim = float(np.dot(prev_embed, emb))
                        if current_tokens >= self.target_tokens and sim < self.similarity_floor:
                            boundary = True

                    if boundary and current_idxs:
                        for ch in flush_chunk():
                            yield ch

                    # add sentence
                    current_idxs.append(idx)
                    current_tokens += tok
                    prev_embed = emb

                i = j

            # flush the tail
            for ch in flush_chunk():
                yield ch

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
