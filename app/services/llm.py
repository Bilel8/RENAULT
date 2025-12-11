class LLMService:
    def generate(self, question: str, docs: list) -> str:
        return f"Réponse basée sur {docs}"