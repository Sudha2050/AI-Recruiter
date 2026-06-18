def clean_text(text: str) -> str:
    return ' '.join(text.strip().lower().split())