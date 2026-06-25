# AI Recruiter & Candidate Ranker

## Overview

**AI Recruiter** is an intelligent, multi-stage candidate discovery and ranking pipeline. It parses job descriptions, evaluates candidate profiles, and generates ranked shortlists with detailed, automated reasoning.

The system features a **Dynamic Category-Based Scoring System** that adapts evaluation weights for experienced candidates versus freshers, ensuring fair and realistic shortlisting.

---

## Key Features

- **Multi-Stage Ranking Pipeline**
  1. **Stage 1 – Honeypot & Trap Filter**: Removes fictional companies, impossible timelines, and ghost profiles. Freshers bypass strict keyword-stuffer checks.
  2. **Stage 2 – Coarse Ranking**: High-speed heuristic scoring using job-specific criteria (titles, company tiers, skill depth) to select the top 5,000 candidates.
  3. **Stage 3 – Semantic Ranking**: BM25 lexical filtering followed by semantic similarity search with Sentence-Transformer embeddings (`all-MiniLM-L6-v2`) to pick the top 500 candidates.
  4. **Stage 4 – Fine Ranking**: Category-aware weighted evaluation integrating work experience, academic background, assessments, and soft-skill metrics.
- **Interactive UI** – Gradio dashboard (`sandbox/app.py`) for uploading candidate JSON, entering job descriptions, and downloading ranking spreadsheets.
- **Automated Explanations** – Generates concise, honest reasoning statements for each ranked candidate.

---

## Category-Aware Scoring Design

### Experienced Candidates (>= 1.5 years)
- **Work Experience & Leadership (45%)** – Tenure, role relevance, company tier, progression speed, semantic JD alignment.
- **Technical Assessments (35%)** – Skill depth and coding test scores.
- **Behavioral Interviews (20%)** – Recruiter response rates, profile completeness, interview completion.

### Freshers / Entry-Level (< 1.5 years)
- **Academic & Foundational Knowledge (50%)** – University tier, field relevance, internships, skill depth, semantic JD match.
- **Aptitude & Soft Skills (50%)** – Standardized tests, GitHub activity, response time, overall profile completeness.

---

## Repository Structure

```
ai-recruiter/
├── config/
│   ├── settings.py       # Pipeline thresholds, embedding models
│   └── jd_config.py      # Scoring weights, company categories
├── data/
│   ├── raw/              # Raw candidate JSON & JD docs
│   └── embeddings/       # Pre-computed candidate vectors
├── src/
│   ├── features/         # Feature extraction (career, education, skills)
│   └── ranking/
│       ├── stage1_filter.py
│       ├── stage2_coarse.py
│       ├── stage3_semantic.py
│       ├── stage4_fine.py
│       └── utils.py      # Category classifier helpers
├── sandbox/
│   └── app.py            # Gradio web UI
├── rank.py               # CLI entry point
└── README.md
```

---

## Getting Started

### 1. Setup Environment
Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the Ranking Pipeline (CLI)
You can run the ranker CLI using customizable arguments. The script dynamically handles both `.jsonl` (uncompressed) and `.jsonl.gz` (compressed) formats.

```bash
python rank.py --candidates data/raw/candidates.jsonl.gz --out outputs/submission.csv --jd data/raw/job_description.docx
```

#### CLI Flags:
* `--candidates`: Path to the candidate pool file (defaults to `data/raw/candidates.jsonl.gz`).
* `--out`: Path to save the output CSV file (defaults to `outputs/submission.csv`).
* `--jd`: Path to the job description document (defaults to `data/raw/job_description.docx`).

#### Robustness Features:
* **Dynamic Embedding Generation**: The script checks if the input candidate size matches the precomputed `data/embeddings/candidate_embeddings.npy` file. If they match, it loads the embeddings instantly. If they do not match (e.g. when testing with a smaller subset), it automatically computes candidate embeddings on-the-fly so the process finishes without failing.
* **BOM Protection (`utf-8-sig`)**: Handles line-by-line reading of `.jsonl` files even when they include a Byte Order Mark (BOM) common on Windows/PowerShell platforms.

### 3. Launch the Interactive UI
Launch the Gradio Web App to view a dashboard for candidate exploration:
```bash
python sandbox/app.py
```
Open the printed URL (e.g., `http://127.0.0.1:7860`) in your browser to interactively upload candidates and search.

---

## Contributing

Contributions are welcome! Please fork the repository, create a feature branch, and submit a pull request. Follow the existing code style and update documentation as needed.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.