# domain_rag

**domain_rag** is a utility pipeline for processing domain-specific documents (e.g., academic papers, reports) and converting them into structured and plain-text formats suitable for Retrieval-Augmented Generation (RAG) systems. It leverages GROBID for PDF to XML transformation and custom tools for extracting and cleaning the text for downstream tasks such as document retrieval or fine-tuning.

---

## 📆 Project Structure

```
domain_rag/
|
├── data/
│   ├── pdfs/               # Input folder for raw PDF documents
│   └── xml_output/         # Output folder for GROBID XML files
|
├── helper/
│   ├── grobid_utils.py     # Functions to run GROBID on PDFs
│   └── tei_utils.py        # Functions to extract and clean TEI XML
|
├── README.md               # This file
└── ...
```

---

## ⚙️ Utilities

### GROBID PDF Processing

```python
from helper.grobid_utils import process_pdfs_in_directory, grobid_process_pdf

# Process a directory of PDFs
process_pdfs_in_directory("data/pdfs", "data/xml_output")
```

Uses [GROBID](https://github.com/kermitt2/grobid) to convert academic PDF documents into structured TEI XML format.

---

### TEI XML Processing

```python
from helper.tei_utils import process_tei_xml_files, process_body_xml_to_plain_text

# Parse and clean TEI XML files into plain text
process_tei_xml_files()
process_body_xml_to_plain_text()
```

- `process_tei_xml_files()`: Extracts relevant metadata and structural content.
- `process_body_xml_to_plain_text()`: Converts the body text of XML documents into clean, retrievable plain text format for downstream RAG or NLP pipelines.

---

## 🚀 Use Case

This toolchain is especially useful in domain-specific RAG systems where clean, structured data from academic or technical PDFs is needed for building vector databases, knowledge bases, or context-aware LLM applications.

---

## ✅ Requirements

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

### `requirements.txt`

```
lxml
beautifulsoup4
tqdm
requests
```

You’ll also need a running [GROBID](https://github.com/kermitt2/grobid) server (Java-based). Start it using:

```bash
./gradlew run
```

Or with Docker:

```bash
docker run -t --rm -p 8070:8070 lfoppiano/grobid
```

---

## 💬 Flask-Based RAG Chat Server

After running the data pre-processing pipeline (using GROBID and TEI utilities), you can launch an interactive RAG (Retrieval-Augmented Generation) system using the included Flask app.

### 🧠 Features

- Uses [LangChain](https://github.com/langchain-ai/langchain) with Chroma for vector search  
- Embedding model: `BAAI/bge-m3` via `sentence-transformers`  
- Automatically loads and splits `.pdf` and `.txt` files  
- Domain-specific prompt templates (e.g., dementia care)  
- LM Studio support with both streaming and non-streaming chat  
- Context citations and basic emergency tagging  

### 🗂️ Structure

```
rag_app/
├── app.py                      # Main Flask application
├── services/                   # Modular service files
│   ├── config.py
│   ├── embedding_utils.py
│   ├── file_utils.py
│   ├── vectorstore_utils.py
│   └── llm_utils.py
└── Plain_Text/                 # Input files for processing (PDFs or TXT)
```

### 🚀 Getting Started

```bash
cd rag_app
pip install -r requirements.txt
python app.py
```

### 🛠 Additional Requirements

These packages are needed for the RAG app portion:

```
Flask
torch
langchain
langchain-community
chromadb
sentence-transformers
PyPDF2
requests
```

Install them with:

```bash
pip install Flask torch langchain langchain-community chromadb sentence-transformers PyPDF2 requests
```

### ▶️ How to Start the RAG App

1. Navigate to the app folder:
   ```bash
   cd rag_app
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Add your `.txt` or `.pdf` files to the `Plain_Text/` folder:
   ```
   rag_app/
   └── Plain_Text/
       ├── example1.txt
       └── example2.pdf
   ```

4. Start LM Studio (or another local LLM endpoint) at:
   ```
   http://localhost:1234/v1/chat/completions
   ```

5. Launch the Flask app:
   ```bash
   python app.py
   ```

6. Send requests to the API:
   ```json
      curl --location 'http://localhost:5000/v1/chat/completions' \
      --header 'x-api-key: your_secure_api_key' \
      --header 'Content-Type: application/json' \
      --data '{"messages": [{"role": "user", "content": "What are tiny pointers?"}], "stream": false, "include_citations": true}'
   ```

---

## 🧐 Future Extensions

- Add support for metadata indexing (authors, year, abstract).
- Integrate chunking and embedding pipeline.
- Add CLI and GUI wrappers for easier batch processing.

---

## 🤝 Contributing

Feel free to open issues or submit PRs if you’d like to contribute!

---

## 📄 License

MIT License

