"""Download the smoke dataset: ~100 CC0 photos from iNaturalist.

Groups: butterflies (Papilionoidea), moths (lookalike distractors), plants
(no-butterfly photos). Saved into code/before_golden_set/data/ — disposable,
kept forever separate from the team's official 400.

Run: python code/before_golden_set/download_smoke.py
"""
import csv
import os
import sys
import time
from pathlib import Path

# The lab proxy breaks iNaturalist/S3 requests — go direct.
for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(k, None)

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.io import SMOKE_DIR

API = "https://api.inaturalist.org/v1"
TARGETS = [
    # (group, taxon_id, n_photos)
    ("butterfly", 47224, 70),   # Papilionoidea, given in the task
    ("moth", 47607, 15),        # Noctuoidea — classic moth lookalikes
    ("plant", 47125, 15),       # Angiospermae — flowering plants, no butterflies
]
IMG_DIR = SMOKE_DIR / "images"


def fetch_page(taxon_id, page=1):
    r = requests.get(
        f"{API}/observations",
        params={"taxon_id": taxon_id, "photos": True, "photo_license": "cc0",
                "per_page": 200, "page": page, "quality_grade": "research"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["results"]


def pick_photos(results, need, seen):
    picked = []
    for obs in results:
        for photo in obs.get("photos", []):
            if photo.get("license_code") != "cc0" or photo["id"] in seen:
                continue
            seen.add(photo["id"])
            picked.append({
                "photo_id": photo["id"],
                "observation_id": obs["id"],
                "taxon_name": obs.get("taxon", {}).get("name", ""),
                "group": None,  # filled by caller
                "url": photo["url"].replace("/square.", "/large."),
                "license": "cc0",
            })
            if len(picked) >= need:
                return picked
    return picked


def download(urls_rows):
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    ok = []
    for row in urls_rows:
        dest = IMG_DIR / f"{row['photo_id']}.jpg"
        if dest.exists():
            ok.append(row)
            continue
        try:
            r = requests.get(row["url"], timeout=60)
            r.raise_for_status()
            if not r.content.startswith(b"\xff\xd8"):  # not a JPEG
                continue
            dest.write_bytes(r.content)
            ok.append(row)
            print(f"ok {row['group']}/{row['photo_id']}", flush=True)
        except Exception as e:
            print(f"skip {row['photo_id']}: {e}", flush=True)
    return ok


def main():
    seen, rows = set(), []
    for group, taxon_id, need in TARGETS:
        got = []
        for page in (1, 2, 3):
            if len(got) >= need:
                break
            results = fetch_page(taxon_id, page)
            if not results:
                break
            got += pick_photos(results, need - len(got), seen)
            time.sleep(1.5)  # be polite to the API
        for r in got:
            r["group"] = group
        rows += got
        print(f"{group}: {len(got)} photo refs", flush=True)

    rows = download(rows)
    with open(SMOKE_DIR / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    by = {}
    for r in rows:
        by[r["group"]] = by.get(r["group"], 0) + 1
    print(f"manifest: {SMOKE_DIR / 'manifest.csv'} | {by}", flush=True)


if __name__ == "__main__":
    main()
