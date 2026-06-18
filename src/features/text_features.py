from typing import List
from collections import Counter

def extract_keywords(text: str, top_k: int = 20) -> List[str]:
    words = text.lower().split()
    common = Counter(words).most_common(top_k)
    return [w for w, _ in common]