from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
DATA_EMBEDDINGS = BASE_DIR / "data" / "embeddings"
DATA_MODELS = BASE_DIR / "data" / "models"

CANDIDATES_FILE = DATA_RAW / "candidates.jsonl.gz"
JD_FILE = DATA_RAW / "job_description.docx"

# Online ranking thresholds
RUNTIME_LIMIT_SEC = 300
MEMORY_LIMIT_GB = 16

# Stage sizes
TOP_K_COARSE = 5000      # after stage 2
TOP_K_SEMANTIC = 500     # after stage 3
TOP_K_FINAL = 100        # output

# Embedding model
EMBEDDING_MODEL = "all-mpnet-base-v2"
EMBEDDING_DIM = 768
EMBEDDING_BATCH_SIZE = 64   # <-- ADD THIS LINE (adjust based on your RAM)
EMBEDDING_NORM = True

# FAISS index type
FAISS_INDEX_TYPE = "FlatIP"  # or "IVF"

# Behavioural signal thresholds
MIN_RESPONSE_RATE = 0.2
MAX_INACTIVE_DAYS = 90

# Honeypot detection
HONEYPOT_FICTIONAL_COMPANIES = {
    "Stark Industries", "Wayne Enterprises", "Pied Piper",
    "Dunder Mifflin", "Initech", "Acme Corp", "Globex Inc", "Hooli"
}

PRODUCT_COMPANIES = {
    'Uber', 'Swiggy', 'Zomato', 'Razorpay', 'CRED', 'Flipkart', 'Ola',
    'Google', 'Microsoft', 'Amazon',
    'Meta', 'Apple', 'Netflix', 'Mad Street Den', 'Redrob AI'
}

CONSULTING_COMPANIES = {
    'TCS', 'Infosys', 'Wipro', 'Accenture', 'Cognizant', 'HCL',
    'Tech Mahindra', 'Capgemini', 'Deloitte', 'PwC', 'EY', 'KPMG',
    'Mindtree', 'LTI', 'Mphasis'
}

AI_CORE_SKILLS = {
    'sentence-transformers', 'openai embeddings', 'bge', 'e5', 'vector search',
    'pinecone', 'weaviate', 'qdrant', 'milvus', 'faiss', 'elasticsearch',
    'opensearch', 'rag', 'llm', 'nlp', 'recommendation systems', 'learning-to-rank',
    'xgboost', 'evaluation frameworks', 'a/b testing', 'production ml',
    'mlops', 'prompt engineering', 'langchain', 'haystack', 'retrieval', 'ranking',
    'information retrieval', 'embeddings', 'fine-tuning', 'peft', 'lora', 'transformers',
    'pytorch', 'tensorflow'
}

PREFERRED_LOCATIONS = ['bangalore', 'hyderabad', 'delhi', 'mumbai', 'pune', 'chennai']

BEHAVIORAL_MULTIPLIERS = {
    "open_to_work": 1.18,
    "high_response_rate": 1.10,
    "low_response_rate": 0.65,
    "recent_activity": 1.05,
    "inactive_6mo": 0.35,
    "saved_by_recruiters": 1.05,
    "verified_profile": 1.05,
    "long_notice_period": 0.95,
    "ghost_profile": 0.20,
}