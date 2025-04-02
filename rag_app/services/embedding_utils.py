# embedding_utils.py
from sentence_transformers import SentenceTransformer
import torch

class CustomHuggingFaceEmbeddings:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.device = device
        self.model = SentenceTransformer(model_name).to(device)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, convert_to_tensor=True, device=self.device).cpu().numpy().tolist()

    def embed_query(self, query: str) -> list[float]:
        return self.model.encode([query], convert_to_tensor=True, device=self.device).cpu().numpy().tolist()[0]
