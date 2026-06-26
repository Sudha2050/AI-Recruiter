import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any
from ..ingestion.loader import load_candidates

def aggregate_profile_text(candidate: Dict[str, Any]) -> str:
    profile = candidate.get('profile', {})
    parts = [
        profile.get('headline', ''),
        profile.get('summary', '')
    ]
    for career in candidate.get('career_history', []):
        parts.append(career.get('title', ''))
        # Repeat the career description to give it higher importance/weight in the semantic embeddings
        parts.append(career.get('description', ''))
        parts.append(career.get('description', ''))
    for skill in candidate.get('skills', []):
        parts.append(skill.get('name', ''))
    return ' '.join(parts)

def build_embeddings(candidates: List[Dict], model_name: str, output_path: Path, batch_size: int = 64):
    texts = [aggregate_profile_text(c) for c in tqdm(candidates)]
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_norm = (embeddings / norms).astype(np.float16)
    np.save(output_path, embeddings_norm)
    return embeddings_norm