# Illinois Landlord-Tenant Case Law (Filtered CAP Subset)

A clean, filtered subset of Illinois case law focused on landlord-tenant / tenancy disputes, derived from the Harvard Caselaw Access Project (CAP).

---

## Source

- **Original dataset:** [Harvard Caselaw Access Project — Illinois](https://www.kaggle.com/datasets/harvardlil/caselaw-dataset-illinois)
- **Published by:** Harvard Library Innovation Lab (Harvard LIL)
- **License:** CC0-1.0 (public domain — no restrictions on use)

This dataset is a filtered derivative of the Harvard CAP Illinois corpus. All source data is CC0-licensed and freely usable without restriction.

---

## Filtering Method

Starting from 183,149 total Illinois cases in the CAP corpus, cases were filtered using **whole-word regex matching** (`\bkeyword\b`) to avoid substring false positives (e.g. `released` matching `lease`, `lieutenant` matching `tenant`).

**Primary keywords** (at least one must match as a whole word):
- landlord, tenant, eviction, lease, habitability

**Support terms** (used for signal strength):
- rent, premises, possession, lessee, lessor, sublease, subletting,
  security deposit, notice to quit, unlawful detainer, holdover,
  month-to-month, tenancy

**Match rule:** A case is included if:
- Primary keyword appears **3 or more times** (whole-word), **OR**
- At least **1 primary keyword** + at least **2 support terms** appear as whole words

This two-condition rule filters out cases where a tenancy keyword appears only incidentally (e.g. a criminal case that mentions "possession" once in passing).

---

## Record Count

**9,408 cases** — from 183,149 total Illinois cases (5.1% match rate)

Date range: **1825 — 2011**

---

## Schema

Each line in `legal_corpus_raw.jsonl` is a JSON object with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `id` | int | Original CAP case ID |
| `source` | string | Always `"CAP"` |
| `type` | string | Always `"case_law"` |
| `jurisdiction` | string | Always `"Illinois"` |
| `case_type` | string | Always `"tenancy"` |
| `citation` | string | Official case citation (e.g. `"365 Ill. App. 3d 621"`) |
| `date` | string | Decision date in `YYYY-MM-DD` or `YYYY-MM` format |
| `text` | string | Full opinion text |

### Example record

```json
{
  "id": 4263941,
  "source": "CAP",
  "type": "case_law",
  "jurisdiction": "Illinois",
  "case_type": "tenancy",
  "citation": "365 Ill. App. 3d 621",
  "date": "2006-06-02",
  "text": "...option to cancel a commercial lease..."
}
```

---

## Intended Use

This dataset was created for the **Opposing-Argument Simulator** — a RAG (Retrieval-Augmented Generation) system that helps self-represented litigants prepare for Illinois landlord-tenant hearings by surfacing the arguments, precedents, and counterevidence the opposing side is likely to raise.

- **Project repo:** https://github.com/ImranRafique847/Opposing-Argument-Simulator
- **Architecture:** LlamaIndex + Amazon Titan Embeddings V2 + pgvector on RDS + Claude via Bedrock

---

## Reproducing This Dataset

To regenerate from scratch:

```bash
# 1. Download the source Illinois CAP data
kaggle datasets download -d harvardlil/caselaw-dataset-illinois -p data --unzip

# 2. Run the filter script (from the project repo)
python filter_cap_data.py
```

See the [project README](https://github.com/ImranRafique847/Opposing-Argument-Simulator) for full setup instructions.
