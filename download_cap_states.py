"""
Download California and New Mexico CAP parquet files from
free-law/Caselaw_Access_Project (HF gated — requires approved access).

Files:
  cal/cal.parquet  ~1.28 GB
  nm/nm.parquet    ~148 MB

Run this once access is approved (you'll get an email from HF).
Check approval status at:
  https://huggingface.co/datasets/free-law/Caselaw_Access_Project
"""
import os
from dotenv import load_dotenv
load_dotenv()

from huggingface_hub import hf_hub_download, login

token = os.getenv("HF_TOKEN")
login(token=token, add_to_git_credential=False)

REPO_ID = "free-law/Caselaw_Access_Project"

downloads = [
    ("cal/cal.parquet", "d:\\Opposing-Argument Simulator\\data_cal\\cal.parquet"),
    ("nm/nm.parquet",   "d:\\Opposing-Argument Simulator\\data_nm\\nm.parquet"),
]

for hf_path, local_path in downloads:
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    if os.path.exists(local_path):
        size_mb = os.path.getsize(local_path) / 1024 / 1024
        print(f"Already exists: {local_path} ({size_mb:.0f}MB) — skipping.")
        continue

    print(f"Downloading {hf_path} ...")
    hf_hub_download(
        repo_id=REPO_ID,
        filename=hf_path,
        repo_type="dataset",
        token=token,
        local_dir=os.path.dirname(local_path),
        local_dir_use_symlinks=False,
    )
    size_mb = os.path.getsize(local_path) / 1024 / 1024
    print(f"Saved: {local_path}  ({size_mb:.0f}MB)")

print("\nDone. Now run: python filter_cap_data.py")
