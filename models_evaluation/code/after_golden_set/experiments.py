"""Mandatory one-factor experiments on the FULL golden set (A: prompt, B: threshold, C: NMS).

Hypotheses are written to runs/experiments.md BEFORE any numbers (task requirement).
Requires runs/golden/<model>.jsonl from run_golden.py.

Usage:
  python code/after_golden_set/experiments.py B C      # post-processing only (cheap)
  HF_HOME=/home/hf_cache python code/after_golden_set/experiments.py A   # re-inference
"""
import sys
from pathlib import Path

import torch
import torchvision

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import io
from lib.metrics import evaluate
from lib.runner import sanitize
from config import CFG, CONTRASTIVE_PROMPT, NMS_IOU, apply_threshold

HYPOTHESES = {
    "A": ("Контрастивный промпт на grounding_dino_tiny: базовый (включает moth как цель) "
          "vs базовый + негативные классы 'caterpillar. pupa.' с фильтром рамок по меткам "
          "butterfly/moth. Гипотеза: перечисление личиночных стадий как отдельных классов "
          "снизит FP на группе other_insects и пустых кадрах, Precision вырастет, Recall "
          "почти не изменится, F1@IoU 0.5 вырастет."),
    "B": ("Калибровка порога уверенности: дефолтные пороги (низкие, рассчитаны на recall) vs "
          "повышенный порог на каждой детекторной модели. Гипотеза: у повышенного порога меньше "
          "FP (в т.ч. на пустых кадрах), F1@IoU 0.5 выше; после оптимума начинает падать Recall."),
    "C": ("Удаление дублирующихся рамок: NMS(iou=0.5) on/off при фиксированном пороге и промпте. "
          "Гипотеза: NMS убирает дубли на крупных бабочках, TP/R почти не меняются, Precision и "
          "F1@IoU 0.5 растут."),
    "D": ("Промпт Qwen3-VL: краткая инструкция (базовая) vs подробная (QWEN_PROMPT_DETAILED: "
          "описание сложных случаев — мелкий/дальний/замаскированный/перекрытый объект, цели "
          "butterfly+moth, отрицания гусениц/куколок/других насекомых) на qwen3_vl_4b. "
          "Гипотеза: подробная повысит Recall на группах small/cropped/overlap, немного добавив "
          "FP на other_insects; итоговый F1@IoU 0.5 сравним с базовым или выше."),
}

SWEEP = [round(0.05 * i, 2) for i in range(1, 13)]


def _load(name):
    return io.read_jsonl(io.RUNS_DIR / "golden" / f"{name}.jsonl")


def _gt():
    return io.read_jsonl(io.GOLDEN_DIR / "golden.jsonl")


def _fmt(tag, r):
    return (f"| {tag} | {r['f1']:.3f} | {r['precision']:.3f} | {r['recall']:.3f} | "
            f"{r['fp']} | {r['fn']} | {r['fp_on_empty']} |")


def run_B(lines):
    gt = _gt()
    lines += ["## B. Калибровка порога уверенности", "", HYPOTHESES["B"], "",
              "| модель | порог | F1@0.5 | P | R | FP | FN | FP на пустых |", "|---|---|---|---|---|---|---|---|"]
    for name, c in CFG.items():
        preds = _load(name)
        if not any(b.get("score") is not None for p in preds.values() for b in p["boxes"]):
            lines.append(f"| {name} | — (нет score) | | | | | | |")
            continue
        best = None
        for t in SWEEP:
            filt = {pid: {"boxes": [b for b in p["boxes"] if b.get("score", 1.0) >= t],
                          "error": p["error"]} for pid, p in preds.items()}
            r = evaluate(gt, filt)
            lines.append(_fmt(f"{name} thr={t:.2f}", r))
            if best is None or r["f1"] > best[1]["f1"]:
                best = (t, r)
        lines.append(_fmt(f"**{name} BEST thr={best[0]:.2f}**", best[1]))
    lines.append("")
    return lines


def _nms(boxes):
    if len(boxes) < 2:
        return boxes
    xyxy = torch.tensor([[b["left"], b["top"], b["left"] + b["width"], b["top"] + b["height"]]
                         for b in boxes])
    scores = torch.tensor([b.get("score", 1.0) for b in boxes])
    keep = torchvision.ops.nms(xyxy, scores, NMS_IOU)
    return [boxes[i] for i in keep.tolist()]


def run_C(lines):
    gt = _gt()
    lines += ["## C. NMS (дублирующиеся рамки)", "", HYPOTHESES["C"], "",
              "| конфиг | F1@0.5 | P | R | FP | FN | FP на пустых |", "|---|---|---|---|---|---|---|"]
    for name in CFG:
        preds = apply_threshold(_load(name), name)
        off = evaluate(gt, preds)
        on = evaluate(gt, {pid: {**p, "boxes": _nms(p["boxes"])} for pid, p in preds.items()})
        lines.append(_fmt(f"{name} без NMS", off))
        lines.append(_fmt(f"{name} NMS iou={NMS_IOU}", on))
    lines.append("")
    return lines


def run_A(lines):
    from lib.models import Detector
    gt = _gt()
    name = "grounding_dino_tiny"
    lines += ["## A. Контрастивный промпт", "", HYPOTHESES["A"], "",
              "| конфиг | F1@0.5 | P | R | FP | FN | FP на пустых |", "|---|---|---|---|---|---|---|"]
    from lib.models import DEFAULTS
    base_prompt = CFG[name].get("prompt") or DEFAULTS[name]["prompt"]
    base = apply_threshold(_load(name), name)
    lines.append(_fmt(f"{name} базовый промпт v2", evaluate(gt, base)))

    det = Detector(name, prompt=CONTRASTIVE_PROMPT, filter_labels=["butterfly", "moth"])
    raw = {}
    for pid in gt:
        img = io.load_image(io.image_path(pid, io.GOLDEN_DIR))
        raw[pid] = {"boxes": sanitize(det.predict(img)), "error": None}
    io.write_jsonl(io.RUNS_DIR / "golden" / f"{name}_contrastive_raw.jsonl", raw)
    treat = apply_threshold(raw, name)
    lines.append(_fmt(f"{name} '{CONTRASTIVE_PROMPT}' + фильтр 'butterfly'", evaluate(gt, treat)))
    lines.append("")
    return lines


def run_D(lines):
    from lib.models import Detector, QWEN_PROMPT_DETAILED
    gt = _gt()
    name = "qwen3_vl_4b"
    lines += ["## D. Промпт Qwen3-VL (краткий vs подробный)", "", HYPOTHESES["D"], "",
              "| конфиг | F1@0.5 | P | R | FP | FN | FP на пустых |", "|---|---|---|---|---|---|---|"]
    lines.append(_fmt(f"{name} краткий (базовый)", evaluate(gt, _load(name))))
    det = Detector(name, prompt=QWEN_PROMPT_DETAILED)
    raw = {}
    for pid in gt:
        img = io.load_image(io.image_path(pid, io.GOLDEN_DIR))
        raw[pid] = {"boxes": sanitize(det.predict(img)), "error": None}
    io.write_jsonl(io.RUNS_DIR / "golden" / f"{name}_detailed_prompt_raw.jsonl", raw)
    lines.append(_fmt(f"{name} подробный", evaluate(gt, raw)))
    lines.append("")
    return lines


RUNNERS = {"A": run_A, "B": run_B, "C": run_C, "D": run_D}


def main():
    which = [a.upper() for a in (sys.argv[1:] or ["A", "B", "C", "D"])]
    lines = ["# Однофакторные эксперименты на полном golden set", ""]
    lines += [f"- Гипотеза {k}: {v.split('Гипотеза: ')[-1]}" for k, v in HYPOTHESES.items() if k in which]
    lines.append("")
    for k in which:
        lines = RUNNERS[k](lines)
    out = io.RUNS_DIR / "experiments.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
