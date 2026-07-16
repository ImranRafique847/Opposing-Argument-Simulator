# Opposing-Argument Simulator

A jurisdiction-grounded RAG system that helps self-represented litigants prepare for hearings by simulating the arguments, precedents, and counterevidence the opposing side is likely to raise.

> Internship project — ITSOLERA

---

## Scope

- **Jurisdiction:** Illinois (US state)
- **Case type:** Landlord-tenant / tenancy disputes
- **Architecture:** RAG pipeline (not fine-tuning) — every argument is traceable to a real statute or case citation

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Data | Harvard Caselaw Access Project (CAP) via Kaggle |
| Vector DB | pgvector on Amazon RDS |
| Embeddings | Amazon Titan Text Embeddings V2 (Bedrock) |
| LLM | Claude via Amazon Bedrock |
| RAG Framework | LlamaIndex |
| Local Dev | Python venv, 4GB NVIDIA GPU |

---

## Project Phases

- **Phase 1 — Data Collection** ✅ Complete
- **Phase 2 — Chunking + Embedding** 🔜 In progress
- **Phase 3 — UI** 🔜 Planned

---

## Phase 1: Data Collection

### Scripts

| Script | Purpose |
|--------|---------|
| `collect_legal_data.py` | Original HF CAP streaming script (gated dataset — access pending) |
| `collect_legal_data_courtlistener.py` | CourtListener API alternative (rate-limited on free tier) |
| `filter_cap_data.py` | **Active** — filters CAP Kaggle download for IL landlord-tenant cases |
| `validate_data.py` | Data quality validation with keyword context checks |

### Running Phase 1

1. **Install dependencies:**
   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   pip install datasets huggingface_hub pandas requests tqdm python-dotenv hf_transfer kaggle
   ```

2. **Set up `.env`** (copy from `env.example`):
   ```
   HF_TOKEN=
   COURTLISTENER_API_TOKEN=
   KAGGLE_USERNAME=
   KAGGLE_KEY=
   ```

3. **Download the Illinois CAP dataset from Kaggle:**
   ```bash
   kaggle datasets download -d harvardlil/caselaw-dataset-illinois -p data --unzip
   ```

4. **Run the filter:**
   ```bash
   python filter_cap_data.py
   ```
   Outputs `legal_corpus_raw.jsonl` — ~9,400 Illinois landlord-tenant cases.

### Filter Design

- **Whole-word regex** (`\bkeyword\b`) eliminates substring false positives (e.g. `released` ≠ `lease`, `lieutenant` ≠ `tenant`)
- **Primary keywords:** landlord, tenant, eviction, lease, habitability
- **Support terms:** rent, premises, possession, lessee, lessor, unlawful detainer, etc.
- **Match rule:** ≥3 primary keyword hits OR (≥1 primary + ≥2 support terms)
- **Result:** 9,408 clean cases from 183,149 total Illinois cases (5.1% match rate)

### Output Schema

```jsonl
{
  "id": 4263941,
  "source": "CAP",
  "type": "case_law",
  "jurisdiction": "Illinois",
  "case_type": "tenancy",
  "citation": "365 Ill. App. 3d 621",
  "date": "2006-06-02",
  "text": "..."
}
```

---

## Setup

```bash
# Clone
git clone https://github.com/ImranRafique847/Opposing-Argument-Simulator.git
cd Opposing-Argument-Simulator

# Create venv
python -m venv venv
venv\Scripts\activate

# Install deps
pip install -r requirements.txt

# Configure secrets
copy env.example .env
# Fill in your tokens in .env
```

---

## Security

- `.env` is in `.gitignore` — never committed
- All tokens loaded via `python-dotenv` — never hardcoded
- `legal_corpus_raw.jsonl` and `data/` are excluded (regenerate locally)
