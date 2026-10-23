"""Create the bundled evaluation PDF from the human-readable sample policy."""
from pathlib import Path
import pymupdf


def main():
    source = Path("data/sample_documents/sample_policy.txt")
    target = Path("data/sample_documents/sample_policy.pdf")
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = source.read_text(encoding="utf-8").splitlines()
    document = pymupdf.open()
    page = document.new_page()
    y = 72
    for line in lines:
        if not line:
            y += 11
            continue
        spare = page.insert_textbox(pymupdf.Rect(72, y, 540, 760), line, fontsize=11)
        if spare < 0:
            page = document.new_page(); y = 72
            spare = page.insert_textbox(pymupdf.Rect(72, y, 540, 760), line, fontsize=11)
        y = 760 - spare + 8
    document.save(target)
    document.close()
    print(f"Created {target}")


if __name__ == "__main__":
    main()

