import os
import json
from .xml_utils import read_tei, find_all_body_text
from .file_utils import (
    extract_text_from_body_xml,
    strip_xml_tags,
    split_text_into_chunks,
    update_json_with_chunks,
)



def process_tei_xml_files(
    tei_directory: str = 'data/xml_output',
    body_xml_directory: str = 'data/body_xml',
    json_output_path: str = 'data/JSON/body_chunks.json',
) -> None:
    """Processes TEI XML files into body-only XML and updates the JSON with chunked text."""

    os.makedirs(body_xml_directory, exist_ok=True)
    os.makedirs(os.path.dirname(json_output_path), exist_ok=True)

    tei_files = [f for f in os.listdir(tei_directory) if f.endswith('.xml')]
    body_xml_files = [f for f in os.listdir(body_xml_directory) if f.endswith('.xml')]

    for tei_file in tei_files:
        base_name = os.path.splitext(tei_file)[0]
        body_file_name = base_name + '_body.xml'
        body_file_path = os.path.join(body_xml_directory, body_file_name)

        if os.path.exists(body_file_path):
            print(f'Skipping {body_file_name} (already exists).')
            continue

        with open(os.path.join(tei_directory, tei_file), 'r') as file:
            tei_xml = file.read()

        title, body_text = find_all_body_text(tei_xml)
        soup = read_tei(tei_xml)
        body_section = soup.find('text').find('body')
        body_xml = f'<?xml version="1.0" encoding="UTF-8"?>\n<body>{str(body_section)}</body>'

        with open(body_file_path, 'w') as file:
            file.write(body_xml)

        print(f'Generated: {body_file_name}')

    # Re-fetch all body files including the newly created ones
    body_xml_files = [f for f in os.listdir(body_xml_directory) if f.endswith('.xml')]
    chunks_data = []

    for body_file in body_xml_files:
        path = os.path.join(body_xml_directory, body_file)
        body_text = extract_text_from_body_xml(path)
        chunks = split_text_into_chunks(body_text)
        for chunk in chunks:
            chunks_data.append({'Title': os.path.splitext(body_file)[0], 'Chunk': chunk})

    update_json_with_chunks(json_output_path, chunks_data)

    if not os.path.exists(json_output_path) or os.stat(json_output_path).st_size == 0:
        with open(json_output_path, 'w') as json_file:
            json.dump([], json_file, indent=4)
        print(f'Created empty JSON at {json_output_path}')


def process_body_xml_to_plain_text(
    body_xml_directory: str = 'data/body_xml',
    text_output_directory: str = 'data/plain_text',
) -> None:
    """Converts all body-only XML files to plain text files."""
    os.makedirs(text_output_directory, exist_ok=True)
    body_xml_files = [f for f in os.listdir(body_xml_directory) if f.endswith('.xml')]

    for body_xml_file in body_xml_files:
        path = os.path.join(body_xml_directory, body_xml_file)
        plain_text = strip_xml_tags(path)

        output_file = os.path.join(
            text_output_directory,
            os.path.splitext(body_xml_file)[0] + '.txt'
        )

        with open(output_file, 'w') as file:
            file.write(plain_text)

        print(f'Saved plain text: {output_file}')
