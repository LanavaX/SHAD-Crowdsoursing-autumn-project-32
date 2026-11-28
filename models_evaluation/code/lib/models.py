"""Unified zero-shot detector adapters.

Detector(name).predict(PIL.Image) -> [{"left","top","width","height","score"?}, ...]
Boxes normalized 0..1 relative to the input image. "score" is absent for VLM models.

Default thresholds are intentionally LOW: runners save boxes with scores, and
threshold calibration happens as post-processing (scout_thresholds / experiments).
"""
import json
import os
import re
import time

os.environ.setdefault("HF_HOME", "/home/hf_cache")
os.environ["HF_HUB_CACHE"] = "/home/hf_cache/hub"
os.environ["HF_HUB_DISABLE_XET"] = "1"
for _k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(_k, None)

import torch
from PIL import Image

# Целевой класс по контракту разметки (STEP2_REPORT): ВСЕ взрослые чешуекрылые —
# дневные бабочки + мотыльки (моль, бражники). Не размечаются: гусеницы, куколки,
# другие насекомые, тени. Промпты обязаны просить мотыльков, иначе потеря Recall.
QWEN_PROMPT = (
    "Detect every adult butterfly or moth (Lepidoptera) visible in the image. "
    "Do NOT box caterpillars, pupae, other insects or shadows. "
    "Reply ONLY with a JSON array, one object per insect: "
    '[{"bbox_2d": [x1, y1, x2, y2]}] where coordinates are integers in 0-1000. '
    "If there are none, reply []."
)

# Подробный вариант — эксперимент D на golden set (мотыльки теперь цель).
QWEN_PROMPT_DETAILED = (
    "You are annotating photos for a biodiversity study. Locate ALL adult butterflies "
    "and moths (any Lepidoptera) in the image. For each one output a tight bounding box "
    "around its visible parts: wings, body, antennae. They can be large or very small, "
    "close or far away, camouflaged against the background, wings open or folded, "
    "partially hidden behind leaves or cut off by the image edge — box only the visible "
    "part, do not guess hidden parts. Do NOT box: caterpillars, pupae, other insects, "
    "birds, flowers, leaves or shadows. If several insects overlap, each one still gets "
    "its own box. "
    'Reply ONLY with a JSON array like [{"bbox_2d": [x1, y1, x2, y2]}, ...] with integer '
    "coordinates in 0-1000. If there are no adult butterflies or moths, reply []."
)

# Базовые промпты: атрибутные фразы у gd_tiny — по дымовому сравнению (FP на
# растениях 19→6), остальным — короткие формы; все включают moth в цель.
DEFAULTS = {
    # mandatory three
    "grounding_dino_tiny": dict(
        ckpt="IDEA-Research/grounding-dino-tiny", kind="grounding_dino",
        prompt="butterfly. moth. small butterfly. distant butterfly. camouflaged butterfly.",
        box_threshold=0.05, text_threshold=0.25),
    "owlv2_base": dict(
        ckpt="google/owlv2-base-patch16-ensemble", kind="owlv2",
        prompt="a photo of a butterfly or moth", box_threshold=0.05),
    "qwen3_vl_2b": dict(
        ckpt="Qwen/Qwen3-VL-2B-Instruct", kind="qwen_vl",
        prompt=None,  # inherits QWEN_PROMPT at load time
        max_new_tokens=1024),
    # additional (florence2_large dropped: transformers 5.19 breaks Florence2Processor
    # at init — RobertaTokenizer has no attribute image_token; adapter code kept below)
    "llmdet_large": dict(
        ckpt="iSEE-Laboratory/llmdet_large", kind="llmdet",
        prompt="butterfly. moth.", box_threshold=0.05),
    "mm_grounding_dino_large": dict(
        ckpt="openmmlab-community/mm_grounding_dino_large_all", kind="grounding_dino",
        prompt="butterfly. moth.", box_threshold=0.05, text_threshold=0.25),
    "qwen3_vl_4b": dict(
        ckpt="Qwen/Qwen3-VL-4B-Instruct", kind="qwen_vl",
        prompt=None,  # inherits QWEN_PROMPT at load time
        max_new_tokens=1024),
    "omdet_turbo": dict(
        ckpt="omlab/omdet-turbo-swin-tiny-hf", kind="omdet",
        prompt=["butterfly", "moth"], box_threshold=0.05),
}

BBOX_RE = re.compile(r'"bbox_2d"\s*:\s*\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]')


def _to_ltwh(x1, y1, x2, y2):
    return {"left": x1, "top": y1, "width": x2 - x1, "height": y2 - y1}


def _freest_cuda():
    """GPU with the most free VRAM — the box is shared and cards fill up unpredictably."""
    if not torch.cuda.is_available():
        return None
    idx = max(range(torch.cuda.device_count()),
              key=lambda i: torch.cuda.mem_get_info(i)[0])
    return f"cuda:{idx}"


class Detector:
    def __init__(self, name, device=None, **overrides):
        self.name = name
        self.cfg = {**DEFAULTS[name], **overrides}
        if self.cfg.get("prompt") is None and self.cfg["kind"] == "qwen_vl":
            self.cfg["prompt"] = QWEN_PROMPT
        self.device = device or _freest_cuda() or "cpu"
        self._load()

    def _load(self):
        from transformers import AutoProcessor
        c = self.cfg
        self.processor = AutoProcessor.from_pretrained(c["ckpt"])
        if c["kind"] == "qwen_vl":
            from transformers import AutoModelForImageTextToText
            self.model = AutoModelForImageTextToText.from_pretrained(
                c["ckpt"], torch_dtype=torch.bfloat16).to(self.device).eval()
        elif c["kind"] == "florence2":
            from transformers import AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(
                c["ckpt"], trust_remote_code=True).to(self.device).eval()
        else:
            from transformers import AutoModelForZeroShotObjectDetection
            dtype = torch.float32  # small models; fp32 avoids bf16 drift in scores
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                c["ckpt"], torch_dtype=dtype).to(self.device).eval()

    # ---- per-kind prediction -------------------------------------------------
    def predict(self, image):
        kind = self.cfg["kind"]
        if kind == "qwen_vl":
            return self._predict_qwen(image)
        if kind == "florence2":
            return self._predict_florence(image)
        return self._predict_detector(image)

    def _predict_detector(self, image):
        c = self.cfg
        if c["kind"] == "owlv2":
            inputs = self.processor(text=[c["prompt"]], images=image, return_tensors="pt").to(self.device)
        elif c["kind"] == "omdet":  # text format: [[label, ...]] per image
            labels = c["prompt"] if isinstance(c["prompt"], list) else [c["prompt"]]
            inputs = self.processor(text=[labels], images=image, return_tensors="pt").to(self.device)
        else:  # grounding_dino, llmdet
            inputs = self.processor(text=c["prompt"], images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]], device=self.device)
        # transformers 5.x: post-processing lives on the processor; signatures differ per family.
        if c["kind"] == "owlv2":  # no input_ids, no text_threshold
            res = self.processor.post_process_grounded_object_detection(
                outputs, threshold=c["box_threshold"], target_sizes=target_sizes)[0]
        else:
            ii = inputs["input_ids"] if "input_ids" in inputs else inputs["tasks_input_ids"]
            kwargs = {"threshold": c["box_threshold"], "target_sizes": target_sizes}
            if c["kind"] in ("grounding_dino", "llmdet"):
                kwargs["text_threshold"] = c.get("text_threshold", 0.25)
            res = self.processor.post_process_grounded_object_detection(outputs, ii, **kwargs)[0]
        W, H = image.size
        boxes = [[x1 / W, y1 / H, x2 / W, y2 / H]
                 for x1, y1, x2, y2 in res["boxes"].float().cpu().tolist()]
        scores = res["scores"].float().cpu().tolist() if "scores" in res else [None] * len(boxes)
        labels = res.get("text_labels")
        if labels is None:
            labels = res.get("labels")
        if labels is not None:
            labels = list(labels.tolist() if hasattr(labels, "tolist") else labels)
        else:
            labels = []
        labels = [str(l) for l in labels]
        out = []
        keep_subs = c.get("filter_labels")  # e.g. ["butterfly", "moth"]
        for (x1, y1, x2, y2), s, lab in zip(boxes, scores, labels + [None] * len(boxes)):
            if keep_subs and lab is not None and not any(sub in lab for sub in keep_subs):
                continue
            b = _to_ltwh(x1, y1, x2, y2)
            if s is not None:
                b["score"] = s
            out.append(b)
        return out

    def _predict_qwen(self, image):
        c = self.cfg
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": c["prompt"]},
        ]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.device)
        with torch.no_grad():
            generated = self.model.generate(**inputs, max_new_tokens=c.get("max_new_tokens", 1024), do_sample=False)
        trimmed = [out_ids[len(inp):] for inp, out_ids in zip(inputs.input_ids, generated)]
        answer = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        out = []
        for x1, y1, x2, y2 in ((int(g) for g in m) for m in BBOX_RE.findall(answer)):
            x1, y1, x2, y2 = (v / 1000.0 for v in (x1, y1, x2, y2))
            out.append(_to_ltwh(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
        return out

    def _predict_florence(self, image):
        c = self.cfg
        task = "<OPEN_VOCABULARY_DETECTION>"
        inputs = self.processor(text=task + c["prompt"], images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            generated = self.model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        answer = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
        res = self.processor.post_process_generation(
            answer, task=task, image_size=(image.width, image.height))
        W, H = image.size
        return [_to_ltwh(x1 / W, y1 / H, x2 / W, y2 / H)
                for x1, y1, x2, y2 in res.get(task, {}).get("bboxes", [])]


def available():
    return list(DEFAULTS)
