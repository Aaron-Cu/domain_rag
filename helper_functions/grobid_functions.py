import os
import requests
import time

# Function to call GROBID REST API and get TEI XML
def grobid_process_pdf(pdf_file_path):
    url = "http://localhost:8070/api/processFulltextDocument"
    with open(pdf_file_path, 'rb') as pdf_file:
        response = requests.post(url, files={'input': pdf_file})
    if response.status_code == 200:
        return response.text
    else:
        raise RuntimeError(f'GROBID failed to process {pdf_file_path}')

# Function to process all PDFs in a directory and save the XML to a different directory
def process_pdfs_in_directory(input_directory, output_directory):
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)
    
    for root, _, files in os.walk(input_directory):
        for file in files:
            if file.endswith('.pdf'):
                pdf_path = os.path.join(root, file)
                
                # Construct the output file path (same name as the PDF, but .xml and in the output directory)
                relative_path = os.path.relpath(root, input_directory)
                output_subdir = os.path.join(output_directory, relative_path)
                os.makedirs(output_subdir, exist_ok=True)
                output_file_path = os.path.join(output_subdir, os.path.splitext(file)[0] + '.xml')
                
                # Check if the XML file already exists
                if os.path.exists(output_file_path):
                    print(f'Skipping {pdf_path}, XML already exists.')
                    continue  # Skip processing if XML already exists
                
                print(f'Processing {pdf_path}')
                try:
                    # Get the TEI XML content
                    tei_xml = grobid_process_pdf(pdf_path)
                    
                    # Write the TEI XML content to the output file
                    with open(output_file_path, 'w', encoding='utf-8') as xml_file:
                        xml_file.write(tei_xml)
                    
                    print(f'Successfully wrote {output_file_path}')
                except Exception as e:
                    print(f'Failed to process {pdf_path}: {e}')
                
                # Avoid overwhelming the GROBID server
                time.sleep(1)

# Example usage:
# process_pdfs_in_directory("/path/to/pdf_directory", "/path/to/output_xml_directory")