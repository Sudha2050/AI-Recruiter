from pathlib import Path
from docx import Document

def parse_jd_docx(filepath: Path) -> str:
    doc = Document(filepath)
    return '\n'.join([p.text for p in doc.paragraphs])