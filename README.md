# domain_rag

**domain_rag** is a utility pipeline for processing domain-specific documents (e.g., academic papers, reports) and converting them into structured and plain-text formats suitable for Retrieval-Augmented Generation (RAG) systems. It leverages GROBID for PDF to XML transformation and custom tools for extracting and cleaning the text for downstream tasks such as document retrieval or fine-tuning.

---

## 📦 Project Structure

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

