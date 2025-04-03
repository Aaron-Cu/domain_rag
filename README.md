# RAG Pipeline for Domain Specific Applications

**RAG Pipeline for Domain Specific Applications** is a modular pipeline for building domain-specific Retrieval-Augmented Generation (RAG) systems that integrate data preprocessing, question-answer generation, and fine-tuning. Developed as part of a case study on disseminating dementia care practices, the framework addresses the challenges of closed-domain QA, including data scarcity, computational constraints, and lack of domain-aligned conversational data.

The pipeline leverages tools like GROBID for structured PDF parsing, RAG for context-aware generation, and PEFT techniques like LoRA/QLoRA for fine-tuning LLMs on specialized datasets.

---

## 📚 Motivation

General-purpose LLMs lack the specificity needed for niche domains like healthcare. This pipeline addresses that gap by:

- Converting unstructured documents into retrievable formats
- Generating synthetic domain-specific QA pairs via few-shot prompting
- Fine-tuning smaller LLMs (e.g., LLaMA3-8B) for improved response relevance
- Combining retrieval and generation for up-to-date, contextually grounded outputs

> This project powers the dementia care assistant presented in our CHASE ’25 paper.

---

## 📦 Project Structure

```
domain_rag/
├── data/
│   ├── pdfs/               # Raw PDF documents
│   └── xml_output/         # Output from GROBID (TEI XML)
│
├── helper/
│   ├── grobid_utils.py     # GROBID PDF to XML conversion
│   ├── tei_utils.py        # TEI XML to clean plain-text chunks
│   ├── finetune_utils.py   # Fine-tuning scripts using PEFT (LoRA/QLoRA)
│   └── qag_utils.py        # Question-answer generation utilities
│
├── rag_app/                # Flask-based RAG interface with vector DB
├── requirements.txt
└── README.md               # This file
```

---

## ⚙️ Preprocessing Pipeline

### PDF → TEI XML → Plain Text Chunks

```python
from helper.grobid_utils import process_pdfs_in_directory
from helper.tei_utils import process_tei_xml_files, process_body_xml_to_plain_text

process_pdfs_in_directory("data/pdfs", "data/xml_output")
process_tei_xml_files()
process_body_xml_to_plain_text()
```

- Converts academic PDFs into structured TEI XML via GROBID
- Extracts and chunks domain-specific text for vector storage and training

Each chunk includes source context, aiding retrieval and grounding.

---

## 💬 RAG Chat System (Flask App)

A local RAG system powered by:

- `sentence-transformers` (`BAAI/bge-m3`) for embeddings
- Chroma vector DB
- LM Studio-compatible LLM (e.g., LLaMA3-8B)
- Domain-aware prompt templates (e.g., dementia caregiving scenarios)

Launch with:

```bash
cd rag_app
pip install -r requirements.txt
python app.py
```

RAG dynamically integrates external context, enabling precise, up-to-date answers with minimal hallucination.

---

## 🤖 Question-Answer Generation Module

Generates high-quality domain-specific QA pairs from labeled chunks:

```python
from helper.qag_util import process_all
df = process_all("qa_generation/input/")
df.to_csv("qa_output.csv", index=False)
```

- Uses few-shot prompting tailored to the domain (e.g., dementia care)
- Supports auto-fix for malformed generations
- Output: Pandas DataFrame with question, answer, context chunk

These QA pairs feed into fine-tuning or evaluation workflows.

---

## 🧠 Fine-Tuning + LoRA/QLoRA

This pipeline fine-tunes compact LLMs with LoRA/QLoRA using synthetic QA:

1. Load a base model (e.g., LLaMA3-8B)
2. Inject LoRA adapters
3. Tokenize QA pairs:
   ```
   Question: <q> Answer: <a>
   ```
4. Train with Hugging Face `Trainer` (supports fp16 + early stopping)
5. Merge adapters and save

Optional: Push to Hugging Face or convert to `.gguf`:

```bash
python convert-hf-to-gguf.py \
  --input_dir ./models/merged \
  --output_dir ./models/gguf \
  --dtype q8_0
```

---

## 📊 Evaluation

Our CHASE ’25 paper benchmarks this framework with:

- BLEU
- ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-LSUM
- F-measure (precision + recall)

> RAG + Med-LLaMA (our fine-tuned model) achieved the best scores across all metrics, demonstrating the effectiveness of combining retrieval with parameter-efficient fine-tuning.

---

## 🔭 Future Work

- Integrate metadata indexing (authors, publication year)
- Expand template library for more domains
- Develop GUI + CLI wrappers for non-technical users

---

## 🧪 Use Cases

- Domain-specific chat assistants (e.g., dementia caregiving, legal aid)
- Dataset synthesis for low-resource fields
- QA benchmarking for fine-tuned LLMs

---

## 📄 License

MIT License

---

# ⚙️ More on the Utilities 

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

## 🧩 Question-Answer Generation Pipeline Helper Module

This module provides a fully automated pipeline for generating scenario-based **questions and answers** using a **local LLM (via LM Studio)** and a lightweight RAG setup. The output is structured and saved as a Pandas DataFrame for further use or export.

---

### 📦 Features

- ✅ Generate **casual, practical questions** from context
- ✅ Generate **concise answers** using a RAG model/client
- ✅ Process `.json` files in bulk
- ✅ Handles malformed JSON responses with a built-in fixer
- ✅ Optionally includes annotation fields for manual review

---

### 📁 Input Format

Each input `.json` file must contain a list of dictionaries with at least a `Chunk` field. Optionally include a `Title`.

```json
[
  {
    "Title": "Understanding ML",
    "Chunk": "Machine learning is a method of data analysis that automates analytical model building."
  }
]
```

---

### 🚀 How to Use

#### 1. **Import the module**
```python
from helper.qag_util import process_all
```

#### 2. **Run the pipeline**
```python
df = process_all("path/to/your/json/files")
df.to_csv("generated_qa.csv", index=False)
```

---

### 📝 Output Columns

| Column      | Description                                                   |
|-------------|---------------------------------------------------------------|
| `question`  | Generated scenario-based question                             |
| `answer`    | Answer generated using a second client (e.g. local RAG app)   |
| `head`      | Title of the context chunk (if provided)                      |
| `text`      | Original input text chunk                                     |

---

## 🧪 Fine-Tuning, Hub Upload, and GGUF Conversion

This project supports fine-tuning a LLaMA-based model with LoRA adapters, saving the merged model, uploading it to the Hugging Face Hub, and converting it to the GGUF format for use with [LM Studio](https://lmstudio.ai/) or other `llama.cpp`-compatible runtimes.

### 🛠️ Training Workflow

The training pipeline includes the following steps:

1. **Model Loading**  
   Load a LLaMA model (quantized or full precision) from the Hugging Face Hub or local path, adding a `pad_token` if needed.

2. **LoRA Adapter Injection**  
   Apply lightweight Low-Rank Adaptation (LoRA) layers using `peft` to enable efficient fine-tuning.

3. **Dataset Preparation**  
   A custom CSV is tokenized using the format:
   ```
   Question: <question> Answer: <answer>
   ```
   Tokenized examples are padded and truncated to a max length of 512 tokens, and split into train/test sets.

4. **Training with Hugging Face Trainer**  
   The `Trainer` API handles epoch-level evaluation, early stopping, and saving the best-performing checkpoint. Mixed-precision (`fp16`) training is supported.

5. **Merging LoRA Adapters**  
   After training, LoRA weights are merged into the base model for deployment using `merge_and_unload()`.

6. **Saving and Uploading to Hugging Face Hub**  
   The merged model and tokenizer are saved locally and optionally pushed to your Hugging Face repo for sharing and deployment:
   ```python
   model.push_to_hub("your-org/your-model-name")
   tokenizer.push_to_hub("your-org/your-model-name")
   ```

---

### 🔁 Converting to GGUF for llama.cpp and LM Studio

After training and merging, the model can be converted to the `.gguf` format used by `llama.cpp`, making it compatible with LM Studio and other local inference tools.

To convert:

1. Clone the `llama.cpp` repo (or download the converter script):
   ```bash
   git clone https://github.com/ggerganov/llama.cpp
   cd llama.cpp
   ```

2. Run the conversion script:
   ```bash
   python convert-hf-to-gguf.py \
       --input_dir ./path/to/merged-model \
       --output_dir ./path/to/output-gguf \
       --dtype q8_0
   ```

3. Load the `.gguf` file in LM Studio or with `llama.cpp`.

> ✅ Supports common quantization formats like `q4_0`, `q8_0`, and `bf16`.

---

### 📦 Output Directory Structure

After full execution, you'll have:

```
models/
├── merged-model/               # Merged model with LoRA adapters fused
├── gguf-out/                   # Converted GGUF file for inference
├── logs/                       # Trainer logs
```

---

### 🔧 Additional Setup Tips

- Make sure `nltk` data is downloaded in any notebook using `punkt`:
  ```python
  import nltk
  nltk.download("punkt")
  ```
- `llama-cpp-python` (if you're running GGUF inference locally):
  ```bash
  pip install llama-cpp-python
  ```

---