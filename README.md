# Opposing-Argument Simulator

A jurisdiction-grounded RAG system that helps self-represented litigants prepare for Illinois landlord-tenant hearings by simulating the arguments, precedents, and counterevidence the opposing side is likely to raise.

> Internship project — ITSOLERA

---

## What it does

1. **Retrieves** relevant Illinois case law from 9,408 real court opinions (Harvard CAP corpus, 1825–2011)
2. **Simulates** the opposing side's strongest arguments — grounded strictly in retrieved citations, no hallucinated case law
3. **Presents** a Streamlit UI where a self-represented litigant describes their situation and receives prepared counterarguments with source citations

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Data | Harvard Caselaw Access Project (CAP) — Illinois bulk, via Kaggle |
| Chunking | NLTK sentence tokenizer + recursive overlap splitter |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers, local GPU) |
| Vector DB | pgvector on Amazon RDS (PostgreSQL 16.4) |
| Sparse search | BM25 via rank_bm25 (in-memory) |
| Retrieval | Hybrid dense+sparse with Reciprocal Rank Fusion |
| LLM | Claude (Anthropic) via Amazon Bedrock |
| UI | Streamlit |

---

## Project Structure

```
Opposing-Argument-Simulator/
├── data_collection/
│   └── filter_cap_data.py       # Filter Illinois CAP bulk download → legal_corpus_final.jsonl
├── processing/
│   ├── chunk_corpus.py          # Split cases into 300-500 token chunks
│   ├── embed_chunks.py          # Embed chunks with all-MiniLM-L6-v2
│   ├── load_to_rds.py           # Bulk-load embeddings into RDS pgvector
│   └── merge_corpus.py          # Merge CAP + optional CourtListener supplement
├── retrieval/
│   └── retrieve.py              # Hybrid BM25 + vector search with RRF fusion
├── app/
│   └── app.py                   # Streamlit UI (Phase 3)
├── archive/                     # Documented attempts — see Data Sourcing Notes below
│   ├── collect_legal_data.py
│   ├── collect_legal_data_courtlistener.py
│   ├── download_cap_states.py
│   ├── check_hf_access.py
│   ├── check_hf_dataset.py
│   └── validate_data.py
├── kaggle_dataset/              # Metadata for Kaggle dataset upload
├── env.example
├── requirements.txt
└── README.md
```

---

## Pipeline Overview

### Phase 1 — Data Collection

```
Kaggle (harvardlil/caselaw-dataset-illinois)
    ↓ download (929MB .xz)
data_collection/filter_cap_data.py
    ↓ whole-word keyword filter (landlord, tenant, eviction, lease, habitability)
legal_corpus_final.jsonl  ← 9,408 Illinois landlord-tenant cases
```

**Filter design:**
- Whole-word regex (`\bkeyword\b`) — eliminates `released` ≠ `lease`, `lieutenant` ≠ `tenant`
- Match rule: ≥3 primary keyword hits OR (≥1 primary + ≥2 support terms)
- Result: 9,408 / 183,149 total Illinois cases (5.1% match rate)

### Phase 2 — Chunking, Embedding, Vector DB

```
legal_corpus_final.jsonl
    ↓ processing/chunk_corpus.py (NLTK sentence split, 400 tokens, 60 token overlap)
chunked_corpus.jsonl  ← 110,163 chunks
    ↓ processing/embed_chunks.py (all-MiniLM-L6-v2, 384d, GPU)
embedded_chunks.jsonl
    ↓ processing/load_to_rds.py (psycopg2 batch insert + pgvector HNSW index)
Amazon RDS PostgreSQL (case_chunks table, vector(384))
```

### Phase 3 — UI + LLM Generation

```
User query (Streamlit)
    ↓ retrieval/retrieve.py
    ├── Dense: pgvector cosine similarity (top-20)
    ├── Sparse: BM25 keyword match (top-20)
    └── Fused: Reciprocal Rank Fusion → top-5
    ↓ Claude via Amazon Bedrock
    ↓ Opposing arguments grounded in retrieved citations
```

---

## Setup

```bash
# 1. Clone
git clone https://github.com/ImranRafique847/Opposing-Argument-Simulator.git
cd Opposing-Argument-Simulator

# 2. Create venv
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt
# Install torch separately (see requirements.txt for CUDA version options)

# 4. Configure secrets
copy env.example .env        # Windows
# cp env.example .env        # macOS/Linux
# Fill in all tokens in .env

# 5. Download Illinois CAP data
kaggle datasets download -d harvardlil/caselaw-dataset-illinois -p data --unzip

# 6. Run the pipeline
python data_collection/filter_cap_data.py    # Phase 1
python processing/chunk_corpus.py            # Phase 2a
python processing/embed_chunks.py            # Phase 2b (GPU recommended)
python processing/load_to_rds.py             # Phase 2c (requires RDS)
streamlit run app/app.py                     # Phase 3
```

---

## Environment Variables

Copy `env.example` to `.env` and fill in:

```
HF_TOKEN=                    # Hugging Face token
AWS_ACCESS_KEY_ID=           # AWS credentials (Bedrock + RDS)
AWS_SECRET_ACCESS_KEY=
AWS_REGION=us-east-1
BEDROCK_MODEL=us.anthropic.claude-haiku-4-5-20251001-v1:0
COURTLISTENER_API_TOKEN=     # Optional
KAGGLE_USERNAME=
KAGGLE_KEY=
DB_HOST=                     # RDS endpoint
DB_PORT=5432
DB_NAME=opposingargdb
DB_USER=postgres
DB_PASSWORD=
```

---

## Data Sourcing Notes

Getting the corpus data required navigating several blocked paths — documented here as part of the real development story:

1. **Hugging Face `free-law/Caselaw_Access_Project`** — the ideal source (parquet per state, full text). Requires manual gated access approval from repo authors. Access was requested but not approved in time. Scripts in `archive/collect_legal_data.py` and `archive/download_cap_states.py` document this path.

2. **CourtListener API** — attempted as a fallback. The free tier was severely rate-limited (125 req/day as of May 2026), yielding effectively 0 usable full-text records. `archive/collect_legal_data_courtlistener.py` documents the retry/backoff logic built for this.

3. **Kaggle `harvardlil/caselaw-dataset-illinois`** — the path that worked. Harvard published the Illinois state file (929MB) as a public Kaggle dataset. Downloaded via the Kaggle CLI, filtered with whole-word regex, and validated for quality before use.

The CA/NM scope expansion was also explored (see `archive/`) but abandoned when no Kaggle equivalents existed for those states and the HF gated access remained pending.

---

## Security

- `.env` is gitignored — never committed
- All credentials loaded via `python-dotenv`
- `data/`, `*.jsonl`, `venv/` are gitignored — regenerate locally
