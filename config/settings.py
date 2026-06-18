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
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
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