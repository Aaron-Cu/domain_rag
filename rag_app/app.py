from flask import Flask, request, jsonify, Response, stream_with_context
import logging
import torch

# === Local services ===
from services.config import FILES_DIRECTORY, PERSIST_DIRECTORY, CHECKSUM_FILE, LLM_URL, DEFAULT_MODE
from services.embedding_utils import CustomHuggingFaceEmbeddings
from services.file_utils import generate_checksum, load_checksum, save_checksum, clear_vector_store, load_files_from_directory
from services.vectorstore_utils import split_documents, build_or_load_vectorstore
from services.llm_utils import rag_chain, llm_chat

from langchain_community.vectorstores import Chroma

# === Flask setup ===
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Device selection ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Embedding model ===
embedding_function = CustomHuggingFaceEmbeddings("BAAI/bge-m3", device=device)

# === Checksum logic ===
checksum_now = generate_checksum(FILES_DIRECTORY)
checksum_stored = load_checksum(CHECKSUM_FILE)

if checksum_now != checksum_stored:
    logger.info("Changes detected in the source directory. Rebuilding vector store.")
    clear_vector_store(PERSIST_DIRECTORY)
    documents = load_files_from_directory(FILES_DIRECTORY)
    chunks = split_documents(documents)
    vectorstore = build_or_load_vectorstore(chunks, PERSIST_DIRECTORY, embedding_function)
    vectorstore.persist()
    save_checksum(CHECKSUM_FILE, checksum_now)
else:
    logger.info("No file changes detected. Loading existing vector store.")
    vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding_function)

# === Flask endpoint ===
@app.route('/v1/chat/completions', methods=['POST'])
def chat():
    data = request.json
    messages = data.get('messages', [])
    if not messages or not isinstance(messages, list):
        return jsonify({'error': 'Invalid messages format'}), 400
    
    stream = data.get('stream', False)
    include_citations = data.get('include_citations', True)
    user_input = messages[-1]['content']

    # RAG processing
    prompt, citations = rag_chain(user_input, vectorstore, mode=DEFAULT_MODE)
    messages[-1]['content'] = prompt

    # LLM call
    response = llm_chat(messages, LLM_URL, stream=stream)

    if stream:
        return Response(
            stream_with_context((f"data: {line.decode('utf-8')}\n\n" for line in response.iter_lines() if line)),
            mimetype="text/event-stream"
        )

    result_text = response.json()['choices'][0]['message']['content']
    result = {
        "choices": [{"message": {"role": "assistant", "content": result_text}, "finish_reason": "stop", "index": 0}]
    }
    if include_citations:
        result["citations"] = citations
    if user_input == "The patient is out of control, what do I do?":
        result["emergency"] = True

    return jsonify(result)

# === Entry point ===
if __name__ == '__main__':
    logger.info("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000)
