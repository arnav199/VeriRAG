from datetime import datetime, timezone
import hashlib
from pathlib import Path

from backend.app.schemas import DocumentRecord
from backend.app.services.chunking import chunk_pages
from backend.app.services.storage import LocalVectorStore


def extract_pdf_pages(content: bytes) -> list[tuple[int, str]]:
    import pymupdf
    try:
        pdf = pymupdf.open(stream=content, filetype="pdf")
    except Exception as exc:
        raise ValueError("The uploaded file is not a readable PDF") from exc
    with pdf:
        if pdf.is_encrypted:
            raise ValueError("Password-protected PDFs are not supported")
        return [(number, page.get_text("text")) for number, page in enumerate(pdf, 1)]


class DocumentService:
    def __init__(self, store: LocalVectorStore, embedder, chunk_size: int, overlap: int):
        self.store, self.embedder = store, embedder
        self.chunk_size, self.overlap = chunk_size, overlap

    def ingest(self, filename: str, content: bytes) -> tuple[DocumentRecord, bool]:
        digest = hashlib.sha256(content).hexdigest()
        existing = self.store.find_by_hash(digest)
        if existing:
            return existing, True
        pages = extract_pdf_pages(content)
        if not any(text.strip() for _, text in pages):
            raise ValueError("No extractable text was found. Scanned PDFs need OCR, which is not included.")
        document_id = digest
        safe_filename = Path(filename).name[:180] or "document.pdf"
        chunks = chunk_pages(pages, document_id, safe_filename, self.chunk_size, self.overlap)
        vectors = self.embedder.encode([chunk.text for chunk in chunks])
        record = DocumentRecord(
            document_id=document_id, filename=safe_filename, sha256=digest,
            page_count=len(pages), chunk_count=len(chunks),
            uploaded_at=datetime.now(timezone.utc).isoformat(),
        )
        target = self.store.docs_dir / f"{document_id}.pdf"
        target.write_bytes(content)
        try:
            self.store.add_document(record, chunks, vectors)
        except Exception:
            target.unlink(missing_ok=True)
            raise
        return record, False

