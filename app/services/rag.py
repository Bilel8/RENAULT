import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer


class RAGService:
    def __init__(self, csv_path: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        df = pd.read_csv(csv_path, sep=";")
        self.rows = df.astype(str).agg(";".join, axis=1).tolist()

        # df = df.fillna("")

        # # Prefix each cell with its column name (including empty strings)
        # for col in df.columns:
        #     df[col] = col + " : " + df[col].astype(str)

        # self.rows = df.astype(str).agg(";".join, axis=1).tolist()

        self.model = SentenceTransformer(model_name)

        self.embeddings = self.model.encode(self.rows, convert_to_numpy=True, normalize_embeddings=True)

    def top_k(self, query: str, k: int = 5):
        if not query:
            return []
        q_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]

        scores = np.dot(self.embeddings, q_emb)
        top_idx = np.argsort(scores)[::-1][:k]
        return [(self.rows[i], float(scores[i])) for i in top_idx]

    def format_context(self, query: str, k: int = 5) -> str:
        hits = self.top_k(query, k)
        return "".join([row + "|" for (row, _) in hits])