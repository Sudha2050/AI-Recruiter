import pandas as pd
import json
import gzip
from pathlib import Path

def main():
    df = pd.read_csv('outputs/submission.csv')
    top_10_ids = list(df.head(10)['candidate_id'])
    
    candidates = {}
    with gzip.open('data/raw/candidates.jsonl.gz', 'rt', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                c = json.loads(line)
                if c['candidate_id'] in top_10_ids:
                    candidates[c['candidate_id']] = c
                    
    for idx, row in df.head(10).iterrows():
        cand_id = row['candidate_id']
        c = candidates[cand_id]
        print(f"Rank {row['rank']}: {cand_id} (Score: {row['score']})")
        print(f"  Title: {c['profile'].get('current_title')} at {c['profile'].get('current_company')} ({c['profile'].get('years_of_experience')} yrs exp)")
        print(f"  Skills: {[(s['name'], s['proficiency'], s.get('duration_months')) for s in c['skills'][:5]]}")
        print(f"  Signals: Response: {c['redrob_signals'].get('recruiter_response_rate')}, Active: {c['redrob_signals'].get('last_active_date')}, Open: {c['redrob_signals'].get('open_to_work_flag')}")
        print(f"  Reasoning: {row['reasoning']}")
        print("-" * 50)

if __name__ == '__main__':
    main()
