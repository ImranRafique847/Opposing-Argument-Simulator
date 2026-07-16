"""
Opposing-Argument Simulator — Phase 1: Data Collection (CourtListener version, v2)
-------------------------------------------------------------------------------------
IMPORTANT CONTEXT: CourtListener cut its free-tier API limits in May 2026.
A plain new account now gets only 5 requests/minute, 50/hour, 125/day
(rolling window) — down from the old 5,000/hour default. This version is
redesigned around that reality:

1. Uses SEARCH RESULT SNIPPETS directly as the case text, instead of a
   separate detail-fetch call per case. This halves the request cost per
   case (1 request per ~20 results via pagination, instead of 1+1 per case).

2. Tracks your daily request budget in a local file (request_budget.json)
   so the script can be re-run across multiple days without exceeding the
   limit, and picks up where it left off automatically.

3. Saves progress incrementally (checkpointing) so a run that stops
   partway never loses what it already collected.

TRADE-OFF: snippets are ~500 characters, not full opinion text. This is
enough for a working RAG prototype. Full text for your best/most relevant
cases can be backfilled selectively later (see fetch_full_text_for_case()
at the bottom, meant for manual use on a small subset, not a bulk pass).

RECOMMENDED PARALLEL STEP: apply for CourtListener's free EDU/student
membership (https://www.courtlistener.com/help/membership/) — it raises
these limits significantly and is worth doing today since it's a one-time,
5-minute signup that benefits your whole project going forward.

Install dependencies (already in your venv):
    pip install pandas requests tqdm python-dotenv

.env file (same folder as this script):
    COURTLISTENER_API_TOKEN=your_token_here
"""

import os
import json
import time
import datetime
import requests
import pandas as pd
from dotenv import load_dotenv

# =========================================================
# LOAD SECRETS FROM .env
# =========================================================

load_dotenv()

COURTLISTENER_API_TOKEN = os.getenv("COURTLISTENER_API_TOKEN", "")

if not COURTLISTENER_API_TOKEN:
    raise SystemExit(
        "ERROR: COURTLISTENER_API_TOKEN not found in .env file.\n"
        "Get a free token at https://www.courtlistener.com/profile/api/ "
        "and add it to your .env file before running this script."
    )

HEADERS = {"Authorization": f"Token {COURTLISTENER_API_TOKEN}"}
BASE_URL = "https://www.courtlistener.com/api/rest/v4"

# =========================================================
# CONFIG
# =========================================================

TARGET_STATE        = "Illinois"
CASE_TYPE_LABEL     = "tenancy"
KEYWORDS            = ["landlord", "tenant", "eviction", "lease", "habitability"]
MAX_CASES_TOTAL     = 300          # realistic target across several days at 125/day budget
RESULTS_PER_KEYWORD = 60           # cap per keyword so no single keyword eats the whole daily budget
DAILY_REQUEST_BUDGET = 110         # stay under the real 125/day limit, leave a safety margin
REQUEST_DELAY_SECONDS = 3          # stay under 5/minute comfortably
REQUEST_TIMEOUT_SECONDS = 60

OUTPUT_CORPUS_JSONL = "legal_corpus_raw.jsonl"
BUDGET_FILE         = "request_budget.json"
PROGRESS_FILE       = "collection_progress.json"  # tracks which keywords are already fully processed


# =========================================================
# DAILY BUDGET TRACKER (persists across script runs / days)
# =========================================================

def load_budget():
    today = datetime.date.today().isoformat()
    if os.path.exists(BUDGET_FILE):
        with open(BUDGET_FILE, "r") as f:
            data = json.load(f)
        if data.get("date") == today:
            return data
    return {"date": today, "requests_used": 0}

def save_budget(budget):
    with open(BUDGET_FILE, "w") as f:
        json.dump(budget, f)

def can_make_request(budget):
    return budget["requests_used"] < DAILY_REQUEST_BUDGET

def record_request(budget):
    budget["requests_used"] += 1
    save_budget(budget)


# =========================================================
# PROGRESS TRACKER (which keywords are done, so re-runs skip them)
# =========================================================

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {"completed_keywords": [], "seen_cluster_ids": []}

def save_progress(progress):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f)


# =========================================================
# STEP 1 — Search for candidates, using snippets as the text field directly
# =========================================================

def search_and_collect(keyword, max_results, budget, progress):
    print(f"\nSearching CourtListener for keyword: '{keyword}'...")

    collected = []
    url = f"{BASE_URL}/search/"
    params = {"q": keyword, "type": "o", "order_by": "score desc", "highlight": "on"}

    while url and len(collected) < max_results:
        if not can_make_request(budget):
            print(f"  Daily request budget ({DAILY_REQUEST_BUDGET}) reached. "
                  f"Stopping here — re-run this script tomorrow to continue.")
            break

        try:
            response = requests.get(
                url,
                headers=HEADERS,
                params=params,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            record_request(budget)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"  Request failed: {e}")
            break

        data = response.json()
        results = data.get("results", [])

        for r in results:
            court_name = (r.get("court") or "")
            cluster_id = r.get("cluster_id")

            if cluster_id in progress["seen_cluster_ids"]:
                continue  # already collected in a previous keyword/run

            if TARGET_STATE.lower() in court_name.lower():
                snippet = r.get("snippet", "") or r.get("text", "") or ""
                citations = r.get("citation", [])
                collected.append({
                    "id": cluster_id,
                    "source": "CourtListener",
                    "type": "case_law",
                    "jurisdiction": TARGET_STATE,
                    "case_type": CASE_TYPE_LABEL,
                    "citation": citations[0] if citations else "",
                    "date": r.get("dateFiled", ""),
                    "text": snippet,
                })
                progress["seen_cluster_ids"].append(cluster_id)

            if len(collected) >= max_results:
                break

        url = data.get("next")
        params = {}  # cursor URL already includes query params
        time.sleep(REQUEST_DELAY_SECONDS)

    print(f"  Collected {len(collected)} new {TARGET_STATE} cases for '{keyword}'.")
    return collected


# =========================================================
# STEP 2 — Append records to the output file (checkpointing)
# =========================================================

def append_records(records):
    with open(OUTPUT_CORPUS_JSONL, "a", encoding="utf-8") as f:
        for rec in records:
            if rec["text"]:  # skip empty-text records, useless for chunking
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# =========================================================
# OPTIONAL — manual full-text backfill for a small subset (not run automatically)
# =========================================================

def fetch_full_text_for_case(cluster_id, budget):
    """
    Call this manually (e.g. from a Python shell) on a handful of your most
    important cases once you've reviewed the corpus — NOT meant to run in a
    bulk loop, since each call costs 1 request from your daily budget.
    """
    if not can_make_request(budget):
        print("Daily budget exhausted, cannot fetch full text right now.")
        return ""
    url = f"{BASE_URL}/opinions/"
    params = {"cluster": cluster_id}
    response = requests.get(url, headers=HEADERS, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
    record_request(budget)
    response.raise_for_status()
    results = response.json().get("results", [])
    if not results:
        return ""
    return results[0].get("plain_text") or results[0].get("html_with_citations") or ""


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    start = time.time()

    budget   = load_budget()
    progress = load_progress()

    print(f"Daily budget status: {budget['requests_used']}/{DAILY_REQUEST_BUDGET} requests used today.")

    total_existing = len(progress["seen_cluster_ids"])

    if total_existing >= MAX_CASES_TOTAL:
        print(f"Already collected {total_existing} cases (target: {MAX_CASES_TOTAL}). Nothing more to do.")
    else:
        for keyword in KEYWORDS:
            if keyword in progress["completed_keywords"]:
                print(f"\nSkipping '{keyword}' — already completed in a previous run.")
                continue

            if len(progress["seen_cluster_ids"]) >= MAX_CASES_TOTAL:
                print(f"\nReached target of {MAX_CASES_TOTAL} total cases. Stopping.")
                break

            if not can_make_request(budget):
                print(f"\nDaily budget reached before finishing '{keyword}'. "
                      f"Re-run this script tomorrow — it will resume automatically.")
                break

            new_records = search_and_collect(keyword, RESULTS_PER_KEYWORD, budget, progress)
            append_records(new_records)
            save_progress(progress)

            # Only mark a keyword complete if we stopped because we ran out of
            # results, not because we hit the budget mid-keyword
            if can_make_request(budget):
                progress["completed_keywords"].append(keyword)
                save_progress(progress)

    elapsed = time.time() - start
    total_collected = len(progress["seen_cluster_ids"])

    print(f"\nAll done in {elapsed:.1f}s.")
    print(f"Total unique cases collected so far (across all runs): {total_collected}")
    print(f"Requests used today: {budget['requests_used']}/{DAILY_REQUEST_BUDGET}")

    if os.path.exists(OUTPUT_CORPUS_JSONL):
        with open(OUTPUT_CORPUS_JSONL, "r", encoding="utf-8") as f:
            lines = f.readlines()
        print(f"\n{OUTPUT_CORPUS_JSONL} contains {len(lines)} records with usable text.")
        if lines:
            print("\nSample record (first entry):")
            print(lines[0][:800] + "...")

    if total_collected < MAX_CASES_TOTAL:
        print(f"\nNOTE: Under the {MAX_CASES_TOTAL}-case target. "
              f"Re-run this script again (today if budget remains, or tomorrow) to continue collecting.")

    print(f"\nNext step once collection is complete: chunk + embed {OUTPUT_CORPUS_JSONL} for the RAG pipeline.")
