FILES_DIRECTORY = "Plain_Text"
PERSIST_DIRECTORY = "./chroma_db"
CHECKSUM_FILE = "./chroma_db/checksum.txt"
LLM_URL = "http://localhost:1234/v1/chat/completions"
DEFAULT_MODE = "APP"

APP_TEMPLATE = """
    You are an AI assistant for people looking after Dementia and Alzheimer patients. The following is a friendly conversation between a human and an AI. 
    The AI is knowledgeable and provides detailed information from its context.

    Context: {context}
    Human: {input}
    Answer the question concisely and to your best accuracy based on the context provided. If the AI does not know the answer to a question, it truthfully says it does not know."""  # Your prompt here
DATASET_TEMPLATE = """
    Context: {context}
    Human: {input}
    Answer the question concisely and to your best accuracy based on the context provided. If the AI does not know the answer to a question, it truthfully says it does not know."""
BASE_TEMPLATE = """
    Context: {context}
    Human: {input}
    Answer the question concisely and to your best accuracy based on the context provided. If the AI does not know the answer to a question, it truthfully says it does not know."""

TEMPLATES = {
    "APP": APP_TEMPLATE,
    "DATA_SET": DATASET_TEMPLATE,
    "BASE": BASE_TEMPLATE
}