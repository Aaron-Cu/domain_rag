# Function to read TEI XML and return BeautifulSoup object
def read_tei(tei_xml):
    soup = BeautifulSoup(tei_xml, features="xml")
    return soup

# Function to parse a section of the TEI XML
def parse_section(section):
    head = section.find('head').getText() if section.find('head') else 'No title'
    paragraphs = section.find_all('p')
    text = ''.join([p.getText() for p in paragraphs])
    return head, text

# Function to extract the title of the paper from TEI XML
def extract_title(soup):
    title = soup.find('title').getText() if soup.find('title') else 'No title'
    return title

# Function to find all body sections and concatenate text
def find_all_body_text(tei_xml):
    soup = read_tei(tei_xml)
    title = extract_title(soup)
    sections = soup.find('text').find('body').find_all('div', xmlns="http://www.tei-c.org/ns/1.0")
    body_text = ''.join([parse_section(section)[1] for section in sections])
    return title, body_text

# Function to generate the checksum of a file
def generate_checksum(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

# Function to split text into chunks of 2500 characters with 250 overlap
def split_text_into_chunks(text, chunk_size=2500, overlap=250):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# Function to extract text from body XML
def extract_text_from_body_xml(body_xml_file):
    with open(body_xml_file, 'r') as file:
        body_xml = file.read()
    soup = BeautifulSoup(body_xml, features="xml")
    body_section = soup.find('body')
    if body_section:
        return body_section.get_text()
    return ''

# Function to update JSON file with chunks
def update_json_with_chunks(json_output_path, chunks_data):
    if os.path.exists(json_output_path):
        try:
            with open(json_output_path, 'r') as json_file:
                existing_data = json.load(json_file)
        except (json.JSONDecodeError, ValueError):
            existing_data = []
    else:
        existing_data = []

    existing_data.extend(chunks_data)
    
    with open(json_output_path, 'w') as json_file:
        json.dump(existing_data, json_file, indent=4)

# Main processing function
def process_tei_xml_files(tei_directory, body_xml_directory, json_output_path):
    tei_files = [f for f in os.listdir(tei_directory) if f.endswith('.xml')]
    existing_body_xml_files = [f for f in os.listdir(body_xml_directory) if f.endswith('.xml')]
    
    # Process TEI XML files
    for tei_file in tei_files:
        tei_file_path = os.path.join(tei_directory, tei_file)
        base_name = os.path.splitext(tei_file)[0]
        body_file_name = base_name + '_body.xml'
        body_file_path = os.path.join(body_xml_directory, body_file_name)
        
        if os.path.exists(body_file_path):
            print(f'Skipping {body_file_name} as it already exists.')
            continue
        
        with open(tei_file_path, 'r') as file:
            tei_xml = file.read()
        title, body_text = find_all_body_text(tei_xml)
        
        # Generate body XML
        soup = read_tei(tei_xml)
        body_section = soup.find('text').find('body')
        body_xml = f'<?xml version="1.0" encoding="UTF-8"?>\n<body>{str(body_section)}</body>'
        
        with open(body_file_path, 'w') as file:
            file.write(body_xml)
        
        print(f'Generated {body_file_name}')
    
    # Chunk newly generated body XML files
    chunks_data = []
    for body_file in existing_body_xml_files:
        body_file_path = os.path.join(body_xml_directory, body_file)
        body_text = extract_text_from_body_xml(body_file_path)
        chunks = split_text_into_chunks(body_text)
        for chunk in chunks:
            chunks_data.append({'Title': os.path.splitext(body_file)[0], 'Chunk': chunk})

    # Update JSON with chunks from existing body XML files
    update_json_with_chunks(json_output_path, chunks_data)
    
    # Check if JSON file exists and create if not
    if not os.path.exists(json_output_path) or os.stat(json_output_path).st_size == 0:
        with open(json_output_path, 'w') as json_file:
            json.dump([], json_file, indent=4)
        print(f'Created empty JSON file at {json_output_path}')

# Example usage
tei_directory = 'TEI_XML'
body_xml_directory = 'Body_XML'
json_output_path = 'JSON/fulltexts.json'

process_tei_xml_files(tei_directory, body_xml_directory, json_output_path)

# Function to strip XML tags and get plain text
def strip_xml_tags(body_xml_file):
    with open(body_xml_file, 'r') as file:
        body_xml = file.read()
    soup = BeautifulSoup(body_xml, features="xml")
    body_text = soup.get_text()
    return body_text

# Function to process Body_XML files and save plain text
def process_body_xml_files(body_xml_directory, text_output_directory):
    if not os.path.exists(text_output_directory):
        os.makedirs(text_output_directory)
    
    body_xml_files = [f for f in os.listdir(body_xml_directory) if f.endswith('.xml')]
    
    for body_xml_file in body_xml_files:
        body_file_path = os.path.join(body_xml_directory, body_xml_file)
        plain_text = strip_xml_tags(body_file_path)
        
        # Create output file path
        base_name = os.path.splitext(body_xml_file)[0]
        text_file_name = base_name + '.txt'
        text_file_path = os.path.join(text_output_directory, text_file_name)
        
        with open(text_file_path, 'w') as file:
            file.write(plain_text)
        
        print(f'Processed {body_xml_file} and saved plain text to {text_file_name}')

# Example usage
body_xml_directory = 'Body_XML'
text_output_directory = 'Plain_Text'

process_body_xml_files(body_xml_directory, text_output_directory)