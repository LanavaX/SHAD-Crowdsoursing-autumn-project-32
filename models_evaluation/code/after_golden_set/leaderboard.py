"""Leaderboard: all models on the golden set at fixed configs -> runs/golden/leaderboard.md.

Includes a per-group slice (manifest.jsonl `group`): F1@IoU 0.5 by group and FP counts
(the `empty` group has no positive boxes — FP only, F1 undefined there).

Usage: python code/after_golden_set/leaderboard.py
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import io
from lib.metrics import evaluate
from config import CFG, apply_threshold


def load_timing():
    t = {}
    path = io.RUNS_DIR / "golden" / "timing.csv"
    if path.exists():
        with open(path) as f:
            for row in csv.DictReader(f):
                t[row["model"]] = row
    return t


def load_groups():
    path = io.GOLDEN_DIR / "manifest.jsonl"
    if not path.exists():
        return {}
    out = {}
    for l in open(path):
        if l.strip():
            m = json.loads(l)
            out[m["photo_id"]] = m.get("group", "?")
    return out


def main():
    gt = io.read_jsonl(io.GOLDEN_DIR / "golden.jsonl")
    timing, groups = load_timing(), load_groups()
    group_names = sorted(set(groups.values()))

    rows = []
    for f in sorted((io.RUNS_DIR / "golden").glob("*.jsonl")):
        name = f.stem
        if name.endswith("_raw"):  # experiment artifacts (contrastive/detailed)
            continue
        preds = apply_threshold(io.read_jsonl(f), name)
        r5 = evaluate(gt, preds, 0.5)
        r75 = evaluate(gt, preds, 0.75)
        t = timing.get(name, {})
        by_group = {}
        for g in group_names:
            sub_gt = {pid: gt[pid] for pid in gt if groups.get(pid) == g}
            sub_pred = {pid: preds.get(pid, {"boxes": [], "error": None}) for pid in sub_gt}
            e = evaluate(sub_gt, sub_pred, 0.5)
            by_group[g] = e
        rows.append({
            "model": name,
            "f1_05": r5["f1"], "f1_075": r75["f1"],
            "precision": r5["precision"], "recall": r5["recall"],
            "fp_on_empty": r5["fp_on_empty"], "errors": r5["errors"],
            "per_img_s": float(t.get("per_img_s", 0) or 0),
            "mandatory": name in CFG,
            "by_group": by_group,
        })
    rows.sort(key=lambda r: -r["f1_05"])

    lines = [
        "# Лидерборд на golden set (400 фото, фиксированные конфиги config.CFG)",
        "",
        "| model | F1@0.5 | F1@0.75 | P | R | FP на пустых | ошибки | сек/кадр | очки контеста* |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        points = int(min(10, 10 * (r["f1_05"] + 0.3)))
        star = "" if r["mandatory"] else " (+доп.)"
        lines.append(
            f"| {r['model']}{star} | {r['f1_05']:.3f} | {r['f1_075']:.3f} | "
            f"{r['precision']:.3f} | {r['recall']:.3f} | {r['fp_on_empty']} | "
            f"{r['errors']} | {r['per_img_s']:.2f} | {points} |")
    lines += ["", "\\* int(min(10, 10×(F1+0.3))) — шкала из задания (по экзамену, не по этому лидерборду)."]

    if group_names:
        lines += ["", "## Порез по группам (манифест разметки)", ""]
        lines.append("F1@IoU 0.5 внутри группы (прочерк — в группе нет позитивных рамок):")
        lines.append("")
        header = "| группа \\ модель | " + " | ".join(r["model"] for r in rows) + " |"
        lines.append(header)
        lines.append("|" + "---|" * (len(rows) + 1))
        for g in group_names:
            cells = []
            for r in rows:
                e = r["by_group"][g]
                n_pos = (e["tp"] + e["fn"]) > 0
                cells.append(f"{e['f1']:.3f}" if n_pos else "—")
            n = sum(1 for pid in gt if groups.get(pid) == g)
            lines.append(f"| {g} (n={n}) | " + " | ".join(cells) + " |")
        lines += ["", "FP (лишние рамки) по группам — где модели ошибаются на двойниках/пустых:"]
        lines.append("")
        lines.append(header)
        lines.append("|" + "---|" * (len(rows) + 1))
        for g in group_names:
            cells = [str(r["by_group"][g]["fp"]) for r in rows]
            n = sum(1 for pid in gt if groups.get(pid) == g)
            lines.append(f"| {g} (n={n}) | " + " | ".join(cells) + " |")

    report = "\n".join(lines)
    out = io.RUNS_DIR / "golden" / "leaderboard.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    print(report)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
