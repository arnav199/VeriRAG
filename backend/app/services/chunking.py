import re

from backend.app.schemas import ChunkRecord


def chunk_pages(pages: list[tuple[int, str]], document_id: str, filename: str,
                chunk_size: int = 900, overlap: int = 120) -> list[ChunkRecord]:
    """Split each page independently so every chunk has an honest page citation."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0")
    result: list[ChunkRecord] = []
    for page_number, text in pages:
        normalized = re.sub(r"\s+", " ", text).strip()
        if not normalized:
            continue
        start = 0
        while start < len(normalized):
            end = min(start + chunk_size, len(normalized))
            # Prefer ending at a word boundary without making tiny chunks.
            if end < len(normalized):
                boundary = normalized.rfind(" ", start + chunk_size // 2, end)
                if boundary > start:
                    end = boundary
            piece = normalized[start:end].strip()
            if piece:
                result.append(ChunkRecord(
                    chunk_id=f"{document_id}-p{page_number}-c{len(result) + 1}",
                    document_id=document_id, filename=filename,
                    page_number=page_number, text=piece,
                ))
            if end >= len(normalized):
                break
            start = max(start + 1, end - overlap)
    return result

