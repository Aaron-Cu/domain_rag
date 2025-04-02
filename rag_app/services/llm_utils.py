# llm_utils.py
from services.config import BASE_TEMPLATE, TEMPLATES
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import requests
import logging

logger = logging.getLogger(__name__)

def get_prompt_template(mode: str) -> str:
    return TEMPLATES.get(mode, BASE_TEMPLATE)

def rag_chain(message: str, vectorstore, mode="APP"):
    docs = vectorstore.similarity_search(message)
    context = "\n\n".join(doc.page_content for doc in docs)
    citations = [{"title": doc.metadata["title"], "content": doc.page_content} for doc in docs]
    
    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(get_prompt_template(mode)),
        HumanMessagePromptTemplate.from_template("{input}")
    ])
    prompt = chat_prompt.format(context=context, input=message)
    return prompt, citations

def llm_chat(messages, url, stream=False):
    logger.info("Calling LLM endpoint...")
    return requests.post(url, json={"model": "llama3", "messages": messages, "stream": stream}, stream=stream)
