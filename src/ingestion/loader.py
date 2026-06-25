import gzip
import json
from pathlib import Path
from typing import List, Dict, Any

def load_candidates(filepath: Path) -> List[Dict[str, Any]]:
    candidates = []
    open_func = gzip.open if str(filepath).endswith('.gz') else open
    with open_func(filepath, 'rt', encoding='utf-8-sig') as f:
        for line in f:
            if line.strip():
                candidates.append(json.loads(line))
    return candidates