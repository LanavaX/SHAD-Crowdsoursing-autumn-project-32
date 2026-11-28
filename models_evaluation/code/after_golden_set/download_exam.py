"""Download the 400 exam images by URL from test_dataset/dataset.jsonl.

S3 originals may vary in extension: the dataset gives original.jpg — on 404 we
retry original.jpeg / original.png. Proxy is disabled (lab proxy breaks S3).

Usage: python code/after_golden_set/download_exam.py
"""
import json
import os
import sys
import time
from pathlib import Path

for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(k, None)

import requests

PROJECT = Path(__file__).resolve().parents[2]
DATASET = PROJECT / "test_dataset" / "dataset.jsonl"
OUT = PROJECT / "data" / "exam" / "images"
EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def fetch(url, sess):
    candidates = [url]  # the dataset URL is authoritative (case-sensitive extensions!)
    base = url.rsplit("/", 1)[0]
    candidates += [base + f"/original{ext}" for ext in EXTS]
    for candidate in dict.fromkeys(candidates):
        try:
            r = sess.get(candidate, timeout=90)
        except requests.RequestException:
            continue
        if r.status_code == 200 and (r.content[:2] == b"\xff\xd8" or r.content[:4] == b"\x89PNG"):
            return (".jpg" if r.content[:2] == b"\xff\xd8" else ".png"), r.content
    return None, None


def main():
    rows = [json.loads(l) for l in open(DATASET) if l.strip()]
    OUT.mkdir(parents=True, exist_ok=True)
    sess = requests.Session()
    missing, done = [], 0
    for i, row in enumerate(rows):
        pid = row["photo_id"]
        dests = [OUT / f"{pid}{ext}" for ext in EXTS]
        if any(d.exists() for d in dests):
            done += 1
            continue
        ext, content = fetch(row["photo_url"], sess)
        if content:
            (OUT / f"{pid}{ext}").write_bytes(content)
            done += 1
        else:
            missing.append(pid)
        if (i + 1) % 50 == 0:
            print(f"{i+1}/{len(rows)} done, missing so far: {len(missing)}", flush=True)
        time.sleep(0.1)
    print(f"TOTAL: {done}/{len(rows)} downloaded, missing: {missing}", flush=True)
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
