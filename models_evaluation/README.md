# models_evaluation — часть 3: выбор, сравнение и настройка моделей

Этот раздел — результат сравнения 7 zero-shot моделей на golden set (400 фото, командная
разметка `human_labeling/`) и выбора финальной конфигурации для Яндекс Контеста.

## Результат

**Победитель: Qwen3-VL-4B-Instruct + подробный промпт + дедупликация точных дублей.**
F1@IoU 0.5 = **0.9606** на полном golden set (P 0.9606, R 0.9606, TP 317 / FP 13 / FN 13).

Итоговый файл для Контеста: **[`submission.jsonl`](submission.jsonl)** — 400 уникальных
photo_id, все правила формата соблюдены (проверено).

## Что здесь лежит

| Что | Где |
|---|---|
| Код инференса, метрик, экспериментов | [`code/`](code/) (lib — ядро, after_golden_set — финал, before_golden_set — отладочная фаза, tests — юнит-тесты) |
| Лидерборд 7 моделей + порез по группам | [`runs/golden/leaderboard.md`](runs/golden/leaderboard.md) |
| Однофакторные эксперименты (A/B/C/D) | [`runs/experiments.md`](runs/experiments.md) |
| Человеко-читаемый отчёт для защиты | [`runs/golden/REPORT.md`](runs/golden/REPORT.md) |
| История промптов (v1/v2, дым) | [`runs/smoke/prompts.md`](runs/smoke/prompts.md) |
| Предсказания всех моделей на golden | [`runs/golden/*.jsonl`](runs/golden/) |
| Предсказания на экзамене (3 модели, перекрёстная проверка) | [`runs/exam/*.jsonl`](runs/exam/) |
| Финальная посылка для Контеста | [`submission.jsonl`](submission.jsonl) |
| Golden set (зеркало `human_labeling/data/`) | [`data/golden/golden.jsonl`](data/golden/golden.jsonl) + `manifest.jsonl` (группы кадров) |
| Зависимости | [`requirements.txt`](requirements.txt) |

## Воспроизводимость

Все запуски — с `HF_HOME=/home/hf_cache` (общий кэш весов). Кэш пополняется
`code/lib/download_models.py`. Лабораторный прокси отключается внутри скриптов.

```bash
cd models_evaluation
pip install -r requirements.txt

# 1. Веса моделей в общий кэш (~20 ГБ)
python code/lib/download_models.py

# 2. Тесты метрик (синтетика, ноль фото)
python -m pytest code/tests/ -q

# 3. (опционально) Скачать 400 экзаменационных фото по URL из test_dataset
python code/after_golden_set/download_exam.py

# 4. Инференс всех 7 моделей на golden set (~15 мин на A100)
HF_HOME=/home/hf_cache python code/after_golden_set/run_golden.py --images data/golden

# 5. Лидерборд с порезом по группам
python code/after_golden_set/leaderboard.py

# 6. Четыре однофакторных эксперимента (A промпт / B порог / C NMS / D подробный промпт Qwen)
HF_HOME=/home/hf_cache python code/after_golden_set/experiments.py A B C D

# 7. Финальная посылка с зафиксированной конфигурацией (winner из config.py)
HF_HOME=/home/hf_cache python code/after_golden_set/make_submission.py qwen3_vl_4b data/exam/images submission.jsonl
```

## Модели

Обязательные по заданию: `IDEA-Research/grounding-dino-tiny`,
`google/owlv2-base-patch16-ensemble`, `Qwen/Qwen3-VL-2B-Instruct`.

Дополнительные: `Qwen/Qwen3-VL-4B-Instruct` (победитель), `iSEE-Laboratory/llmdet_large`,
`openmmlab-community/mm_grounding_dino_large_all`, `omlab/omdet-turbo-swin-tiny-hf`.

Florence-2-large-ft **исключена**: регрессия transformers 5.19 (ломается инициализация
`Florence2Processor`). `mm_grounding_dino_large` исключён из выбора: систематическое сжатие
рамок по вертикали (артефакт конвертации чекпоинта, не качество модели).
