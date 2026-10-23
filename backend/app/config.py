"""Small environment-backed settings object; no settings package is required."""

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    max_upload_bytes: int
    verifier: str
    llm_api_key: str | None
    llm_base_url: str
    llm_model: str

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            data_dir=Path(os.getenv("VERIRAG_DATA_DIR", "./data/storage")).resolve(),
            embedding_model=os.getenv(
                "VERIRAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            chunk_size=max(100, int(os.getenv("VERIRAG_CHUNK_SIZE", "900"))),
            chunk_overlap=max(0, int(os.getenv("VERIRAG_CHUNK_OVERLAP", "120"))),
            top_k=max(1, int(os.getenv("VERIRAG_TOP_K", "5"))),
            max_upload_bytes=max(1, int(os.getenv("VERIRAG_MAX_UPLOAD_MB", "20"))) * 1024 * 1024,
            verifier=os.getenv("VERIRAG_VERIFIER", "heuristic").lower(),
            llm_api_key=os.getenv("LLM_API_KEY") or None,
            llm_base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        )

