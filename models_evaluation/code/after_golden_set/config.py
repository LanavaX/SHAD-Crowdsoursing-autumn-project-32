"""Fixed final model configurations — THE single source of truth for chosen settings.

The task requires frozen, reproducible settings for the chosen solution. Thresholds
below are the OPERATING thresholds applied to saved scores at evaluation/submission
time (inference always runs at a low 0.05 threshold and saves scores so any
operating point is reachable without re-running models).

Values here are placeholders until tuned on the real golden set
(scout_thresholds gives the first orientation, experiments.py finalizes).
"""
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.models import QWEN_PROMPT_DETAILED

# ФИНАЛЬНЫЙ ВЫБОР (18.11, argmax F1@0.5 на полном golden set): qwen3_vl_4b
# с подробным промптом (эксперимент D: 0.947 -> 0.959; у qwen3_vl_2b база 0.955).
WINNER = "qwen3_vl_4b"

PHRASE_PROMPT = "adult butterfly. small butterfly. distant butterfly. camouflaged butterfly."

CFG = {
    # Пороги = лучший результат свипа на НАСТОЯЩЕМ golden set (эксперимент B, 18.11):
    # gd_tiny 0.35/F1 .644, owlv2 0.40/.773, llmdet 0.50/.215, mm_gd 0.35/.025 (артефакт), omdet 0.45/.572.
    # Qwen (VLM) порога не имеют — F1: 2B .955, 4B .947.
    "grounding_dino_tiny": {"box_threshold": 0.35},
    "owlv2_base": {"box_threshold": 0.40},
    "qwen3_vl_2b": {},
    "llmdet_large": {"box_threshold": 0.50},
    "mm_grounding_dino_large": {"box_threshold": 0.35},
    "qwen3_vl_4b": {"prompt": QWEN_PROMPT_DETAILED},
    "omdet_turbo": {"box_threshold": 0.45},
}

MANDATORY = ["grounding_dino_tiny", "owlv2_base", "qwen3_vl_2b"]

CONTRASTIVE_PROMPT = ("butterfly. moth. small butterfly. distant butterfly. "
                      "camouflaged butterfly. caterpillar. pupa.")
NMS_IOU = 0.5


def apply_threshold(records, name):
    """Filter score-bearing boxes down to the operating threshold from CFG."""
    thr = CFG[name].get("box_threshold")
    if not thr:
        return records
    return {pid: {"boxes": [b for b in p["boxes"] if b.get("score", 1.0) >= thr],
                  "error": p["error"]}
            for pid, p in records.items()}
