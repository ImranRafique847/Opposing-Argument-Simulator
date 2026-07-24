import os, requests
from dotenv import load_dotenv
load_dotenv()
token = os.getenv("HF_TOKEN")
headers = {"Authorization": f"Bearer {token}"}

# Check data/ subfolder and look for CA/NM files
r = requests.get("https://huggingface.co/api/datasets/HFforLegal/case-law/tree/main/data",
                 headers=headers, timeout=15)
print(f"data/ HTTP: {r.status_code}")
if r.status_code == 200:
    items = r.json()
    print(f"Files in data/ ({len(items)}):")
    for item in items[:30]:
        path = item.get("path", "")
        size_mb = round(item.get("size", 0) / 1024 / 1024, 1)
        print(f"  {path}  ({size_mb}MB)")
    # Look for california/new mexico
    ca = [i for i in items if "california" in i.get("path","").lower() or "/cal" in i.get("path","").lower()]
    nm = [i for i in items if "new_mexico" in i.get("path","").lower() or "new-mexico" in i.get("path","").lower() or "/nm" in i.get("path","").lower()]
    print(f"\nCA matches: {[i['path'] for i in ca]}")
    print(f"NM matches: {[i['path'] for i in nm]}")
