"""Local persistent document registry and FAISS index for a single-user prototype."""
import json
from pathlib import Path
from threading import RLock

import numpy as np

from backend.app.schemas import ChunkRecord, DocumentRecord, EvidenceResult


class LocalVectorStore:
    def __init__(self, root: Path):
        self.root = root
        self.docs_dir = root / "documents"
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.docs_path = root / "documents.json"
        self.chunks_path = root / "chunks.json"
        self.index_path = root / "vectors.faiss"
        self._lock = RLock()
        self.documents = self._read_models(self.docs_path, DocumentRecord)
        self.chunks = self._read_models(self.chunks_path, ChunkRecord)
        self._index = None
        if self.index_path.exists():
            import faiss
            self._index = faiss.read_index(str(self.index_path))

    @staticmethod
    def _read_models(path: Path, model):
        if not path.exists():
            return []
        try:
            return [model.model_validate(item) for item in json.loads(path.read_text("utf-8"))]
        except (ValueError, OSError) as exc:
            raise RuntimeError(f"Local index metadata is invalid: {path.name}") from exc

    def find_by_hash(self, sha256: str) -> DocumentRecord | None:
        return next((doc for doc in self.documents if doc.sha256 == sha256), None)

    def add_document(self, record: DocumentRecord, chunks: list[ChunkRecord], vectors) -> None:
        import faiss
        vectors = np.asarray(vectors, dtype="float32")
        if vectors.ndim != 2 or len(vectors) != len(chunks):
            raise ValueError("Embedding output does not match chunk count")
        with self._lock:
            old_index = self._index
            if old_index is None:
                index = faiss.IndexFlatIP(vectors.shape[1])
            else:
                if old_index.d != vectors.shape[1]:
                    raise ValueError("Embedding dimension changed; rebuild the local index")
                index = old_index
            index.add(vectors)
            new_docs = [*self.documents, record]
            new_chunks = [*self.chunks, *chunks]
            # Write metadata first to temporary files, then atomically replace each file.
            self._write_json(self.docs_path, [x.model_dump(mode="json") for x in new_docs])
            self._write_json(self.chunks_path, [x.model_dump(mode="json") for x in new_chunks])
            temp_index = self.root / "vectors.faiss.tmp"
            faiss.write_index(index, str(temp_index))
            temp_index.replace(self.index_path)
            self.documents, self.chunks, self._index = new_docs, new_chunks, index

    @staticmethod
    def _write_json(path: Path, data) -> None:
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
        temp.replace(path)

    def search(self, vector, top_k: int) -> list[EvidenceResult]:
        with self._lock:
            if self._index is None or not self.chunks:
                return []
            query = np.asarray(vector, dtype="float32").reshape(1, -1)
            scores, ids = self._index.search(query, min(top_k, len(self.chunks)))
            return [
                EvidenceResult(**self.chunks[int(i)].model_dump(), similarity=float(score))
                for score, i in zip(scores[0], ids[0]) if i >= 0
            ]

