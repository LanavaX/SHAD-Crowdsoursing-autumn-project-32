"""Download all experiment checkpoints into the shared HF cache (/home/hf_cache).

/home/hf_cache itself is root-owned and read-only for us, but hub/ is world-writable,
so we point HF_HUB_CACHE straight at it and disable xet (which logs into the read-only
cache root). Models then land next to the existing models--* entries and are visible
to everyone using HF_HOME=/home/hf_cache.
"""
import os

os.environ.setdefault("HF_HOME", "/home/hf_cache")
os.environ["HF_HUB_CACHE"] = "/home/hf_cache/hub"
os.environ["HF_HUB_DISABLE_XET"] = "1"
# The lab proxy caps HF at ~270 KB/s; direct is ~7.7 MB/s.
for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(k, None)

from huggingface_hub import snapshot_download

CHECKPOINTS = [
    "IDEA-Research/grounding-dino-tiny",
    "google/owlv2-base-patch16-ensemble",
    "Qwen/Qwen3-VL-2B-Instruct",
    "iSEE-Laboratory/llmdet_large",
    "openmmlab-community/mm_grounding_dino_large_all",
    "Qwen/Qwen3-VL-4B-Instruct",
    "microsoft/Florence-2-large-ft",
    "omlab/omdet-turbo-swin-tiny-hf",
]

if __name__ == "__main__":
    for repo in CHECKPOINTS:
        print(f"=== downloading {repo} ===", flush=True)
        try:
            path = snapshot_download(repo)
            print(f"OK {repo} -> {path}", flush=True)
        except Exception as e:
            print(f"FAIL {repo}: {e}", flush=True)
    print("ALL DONE", flush=True)
