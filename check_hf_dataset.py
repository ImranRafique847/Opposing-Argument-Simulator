import os
from dotenv import load_dotenv
load_dotenv()

from huggingface_hub import login, list_repo_files
token = os.getenv("HF_TOKEN")
login(token=token, add_to_git_credential=False)

# Count files per state to understand dataset size
from huggingface_hub import list_repo_tree
cal_files = list(list_repo_tree(
    "endomorphosis/Caselaw_Access_Project_JSON",
    path_in_repo="cal",
    repo_type="dataset",
    recursive=False,
))
nm_files = list(list_repo_tree(
    "endomorphosis/Caselaw_Access_Project_JSON",
    path_in_repo="nm",
    repo_type="dataset",
    recursive=False,
))
print(f"California files: {len(cal_files):,}")
print(f"New Mexico files: {len(nm_files):,}")
print(f"\nSample CA file path: {cal_files[0].path if cal_files else 'none'}")
print(f"Sample NM file path: {nm_files[0].path if nm_files else 'none'}")
