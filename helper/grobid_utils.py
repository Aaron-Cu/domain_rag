import os
import requests
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def grobid_process_pdf(pdf_file_path, grobid_url="http://localhost:8070/api/processFulltextDocument"):
    """
    Sends a PDF file to the GROBID REST API and returns the TEI XML response.

    Args:
        pdf_file_path (str): Path to the PDF file to be processed.
        grobid_url (str): URL of the GROBID API endpoint.

    Returns:
        str: TEI XML string response from GROBID.

    Raises:
        RuntimeError: If GROBID fails to process the PDF.
    """
    with open(pdf_file_path, 'rb') as pdf_file:
        response = requests.post(grobid_url, files={'input': pdf_file})
    if response.status_code == 200:
        return response.text
    else:
        raise RuntimeError(f'GROBID failed to process {pdf_file_path}, status code: {response.status_code}')


def process_pdfs_in_directory(
    input_directory: str,
    output_directory: str,
    grobid_url: str = "http://localhost:8070/api/processFulltextDocument",
    sleep_time: float = 1.0,
    overwrite_existing: bool = False
):
    """
    Processes all PDF files in the input directory using GROBID and saves the resulting TEI XML files.

    Args:
        input_directory (str): Path to the directory containing PDF files.
        output_directory (str): Path where the TEI XML files should be saved.
        grobid_url (str): URL of the GROBID API endpoint. Defaults to local server.
        sleep_time (float): Time to wait (in seconds) between API calls to avoid overloading the server. Default is 1.0.
        overwrite_existing (bool): If True, overwrite existing XML files. Defaults to False.

    Returns:
        None
    """
    os.makedirs(output_directory, exist_ok=True)
    
    for root, _, files in os.walk(input_directory):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_path = os.path.join(root, file)
                
                # Build output path with same relative structure
                relative_path = os.path.relpath(root, input_directory)
                output_subdir = os.path.join(output_directory, relative_path)
                os.makedirs(output_subdir, exist_ok=True)
                output_file_path = os.path.join(output_subdir, os.path.splitext(file)[0] + '.xml')

                if not overwrite_existing and os.path.exists(output_file_path):
                    logger.info(f'Skipping {pdf_path}, XML already exists.')
                    continue

                logger.info(f'Processing {pdf_path}')
                try:
                    tei_xml = grobid_process_pdf(pdf_path, grobid_url=grobid_url)
                    with open(output_file_path, 'w', encoding='utf-8') as xml_file:
                        xml_file.write(tei_xml)
                    logger.info(f'Successfully wrote {output_file_path}')
                except Exception as e:
                    logger.error(f'Failed to process {pdf_path}: {e}')

                time.sleep(sleep_time)


# Example usage (uncomment to run directly):
# process_pdfs_in_directory("pdfs", "xml_output")
