import cohere
import fitz
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class VectorStore:
    def __init__(self, pdf_path: str, cohere_api_key: str):
        self.pdf_path = pdf_path
        self.co = cohere.ClientV2(cohere_api_key)
        self.chunks = []
        self.rerank_top_k = 3
        self.load_pdf()
        self.split_text()
        self.embed_chunks()

    def load_pdf(self):
        self.pdf_text = self.extract_text_from_pdf(self.pdf_path)

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        text = ""
        with fitz.open(pdf_path) as pdf:
            for page_num in range(pdf.page_count):
                page = pdf.load_page(page_num)
                text += page.get_text("text")
        return text

    def split_text(self, chunk_size=1000):
        sentences = self.pdf_text.split(". ")
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                self.chunks.append(current_chunk)
                current_chunk = sentence + ". "
        if current_chunk:
            self.chunks.append(current_chunk)

    def embed_chunks(self):
        self.vectorizer = TfidfVectorizer()
        self.embeddings = self.vectorizer.fit_transform(self.chunks)

    def retrieve(self, query: str) -> list:
        query_emb = self.vectorizer.transform([query])
        scores = cosine_similarity(query_emb, self.embeddings)[0]
        top_k_indices = np.argsort(scores)[::-1][:self.rerank_top_k]
        return [{'text': self.chunks[i]} for i in top_k_indices]
