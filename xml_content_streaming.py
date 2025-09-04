# Created by AG on 04-08-2025

import os
import sys
import re
import click
import numpy
import psycopg
import xml.etree.ElementTree as ETree
from groq import Groq
from psycopg.rows import dict_row
from dataclasses import dataclass
from pgvector.psycopg import register_vector
from typing import List, Tuple, Dict, Iterable, Generator, Optional
from sentence_transformers import SentenceTransformer

@dataclass 
class Chunk:
    id: str
    content: str
    metadata: Dict[str, int]
    
def _generate_text_from_xml(
    xml_file_path: str
    ) -> Generator[str, None, None]:
        for _, element in ETree.iterparse(
            xml_file_path,
            events=("end", )
            ):
                if element.tag.rsplit("}", 1)[-1].lower() == "content":
                    body_text = []
                    for text in element.itertext():
                        if text and (string := text.strip()):
                            body_text.append(string)
                    if body_text:
                        yield " ".join(body_text)
                element.clear()
                
