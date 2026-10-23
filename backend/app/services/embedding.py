from collections.abc import Sequence


class SentenceTransformerEmbedder:
    """Loads a small local model only when the first document/query is embedded."""
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def encode(self, texts: Sequence[str]):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model.encode(
            list(texts), convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False,
        )

