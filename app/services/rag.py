import pandas as pd
from rank_bm25 import BM25Okapi


class RAGService:
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path, sep=";")
        self.rows = df.astype(str).agg(" | ".join, axis=1).tolist()
        self.corpus = [r.lower().split() for r in self.rows]
        self.bm25 = BM25Okapi(self.corpus)

    def top_k(self, query: str, k: int = 5):
        scores = self.bm25.get_scores(query.lower().split())
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.rows[i], float(scores[i])) for i in top]

    def format_context(self, query: str, k: int = 5) -> str:
        hits = self.top_k(query, k)
        # return "\n".join(
        #     [f"[CSV {i+1} | score={s:.2f}] {row}" for i, (row, s) in enumerate(hits)]
        # )
        return "\n".join(
            sum([row + ' | ' for (row, _) in hits], "")
        )
