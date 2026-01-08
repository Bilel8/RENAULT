from dataclasses import dataclass
from typing import Any, Iterable, Optional
import os

try:
    from llama_cpp import Llama
except ImportError as e:
    raise ImportError(
        "Installe llama-cpp-python : pip install llama-cpp-python"
    ) from e


@dataclass
class LLMConfig:
    model_path: str
    n_ctx: int = 2048
    n_threads: int = max(os.cpu_count() or 4, 4)
    n_gpu_layers: int = 0
    temperature: float = 0.2
    top_p: float = 0.95
    max_tokens: int = 256
    system_prompt: str = (
        "Tu es un assistant vocal embarqué sur un Jetson Orin Nano. "
        "Réponds en français, de façon concise, claire, et utile. "
        "Si tu ne sais pas, dis-le."
    )


class LLMService:
    def __init__(self, config: LLMConfig):
        if not os.path.isfile(config.model_path):
            raise FileNotFoundError(f"Modèle GGUF introuvable: {config.model_path}")

        self.cfg = config
        self.llm = Llama(
            model_path=self.cfg.model_path,
            n_ctx=self.cfg.n_ctx,
            n_threads=self.cfg.n_threads,
            n_gpu_layers=self.cfg.n_gpu_layers,
            verbose=False,
        )

    def _format_docs(self, docs: Optional[Iterable[Any]]) -> str:
        if not docs:
            return ""
        parts = []
        for i, d in enumerate(docs, start=1):
            if isinstance(d, str):
                txt = d
            elif isinstance(d, dict):
                txt = d.get("text") or d.get("content") or str(d)
            else:
                txt = str(d)
            txt = txt.strip()
            if txt:
                parts.append(f"[Doc {i}] {txt}")
        return "\n".join(parts)

    def generate(self, question: str, docs: Optional[list] = None) -> str:
        question = (question or "").strip()
        if not question:
            return "Je n'ai pas reçu de question."

        context = self._format_docs(docs)
        user_content = (
            f"Question: {question}\n"
            + (f"\nContexte:\n{context}\n" if context else "\n")
            + "\nRéponds maintenant."
        )

        messages = [
            {"role": "system", "content": self.cfg.system_prompt},
            {"role": "user", "content": user_content},
        ]

        out = self.llm.create_chat_completion(
            messages=messages,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            max_tokens=self.cfg.max_tokens,
        )

        try:
            return out["choices"][0]["message"]["content"].strip()
        except Exception:
            return str(out)
