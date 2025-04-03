import os
import json
import hashlib
from bs4 import BeautifulSoup
from typing import List

def generate_checksum(file_path: str) -> str:
    """Generates an MD5 checksum for a given file."""
    hash_md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def extract_text_from_body_xml(body_xml_file: str) -> str:
    """Extracts plain text content from a body-only XML file."""
    with open(body_xml_file, 'r') as file:
        soup = BeautifulSoup(file.read(), features="xml")
    body_section = soup.find('body')
    return body_section.get_text() if body_section else ''


def strip_xml_tags(body_xml_file: str) -> str:
    """Strips XML tags and returns plain text from a body XML file."""
    with open(body_xml_file, 'r') as file:
        soup = BeautifulSoup(file.read(), features="xml")
    return soup.get_text()


def split_text_into_chunks(text: str, chunk_size: int = 2500, overlap: int = 250) -> List[str]:
    """Splits text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def update_json_with_chunks(json_output_path: str, chunks_data: List[dict]) -> None:
    """Appends chunk data to a JSON file, creating it if necessary."""
    if os.path.exists(json_output_path):
        try:
            with open(json_output_path, 'r') as file:
                existing_data = json.load(file)
        except (json.JSONDecodeError, ValueError):
            existing_data = []
    else:
        existing_data = []

    existing_data.extend(chunks_data)

    with open(json_output_path, 'w') as file:
        json.dump(existing_data, file, indent=4)