from typing import List, Dict, Any

RELEVANT_SKILLS = {
    'sentence-transformers', 'openai embeddings', 'bge', 'e5', 'vector search',
    'pinecone', 'weaviate', 'qdrant', 'milvus', 'faiss', 'elasticsearch',
    'opensearch', 'rag', 'llm', 'nlp', 'recommendation systems', 'learning-to-rank',
    'xgboost', 'evaluation frameworks', 'a/b testing', 'production ml',
    'mlops', 'prompt engineering', 'langchain', 'haystack', 'retrieval', 'ranking',
    'information retrieval', 'embeddings', 'fine-tuning', 'peft', 'lora', 'transformers',
    'pytorch', 'tensorflow'
}

def skill_depth_score(skills: List[Dict]) -> float:
    total = 0.0
    count = 0
    for s in skills:
        name = s.get('name', '').lower()
        if not any(term in name for term in RELEVANT_SKILLS):
            continue
        prof = {'beginner': 0.4, 'intermediate': 0.7, 'advanced': 1.0, 'expert': 1.2}.get(s.get('proficiency'), 0.5)
        dur = min(1.0, s.get('duration_months', 0) / 24) if s.get('duration_months', 0) > 0 else 0.3
        end = min(1.5, 1 + s.get('endorsements', 0) / 30)
        total += prof * dur * end
        count += 1
    return total / count if count else 0.0