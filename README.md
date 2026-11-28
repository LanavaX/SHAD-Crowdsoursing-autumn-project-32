# SHAD-Crowdsoursing-autumn-project-32

Финальный проект: разметка данных и выбор модели для мониторинга биоразнообразия
(детекция бабочек и мотыльков рамками на фотографиях).

## Результат

- **400 фото** golden set, размечено командой через Яндекс Задания с двойным ревью.
- **7 моделей** сравнены zero-shot по метрике Контеста (micro F1@IoU 0.5).
- **Победитель: Qwen3-VL-4B-Instruct + подробный промпт + дедупликация дублей** →
  **F1@IoU 0.5 = 0.961** на golden set (10/10 по шкале Контеста).
- Финальная посылка: [`models_evaluation/submission.jsonl`](models_evaluation/submission.jsonl).

## Структура

| Папка | Содержимое |
|---|---|
| [`human_labeling/`](human_labeling/) | Разметка: инструкции, настройки Яндекс Заданий (bbox + review), golden set, манифест, расходы |
| [`models_evaluation/`](models_evaluation/) | Часть 3: код сравнения, метрики, эксперименты A/B/C/D, предсказания всех моделей, финальная посылка |
| [`presentation/`](presentation/) | Материалы для защиты (в т.ч. [`SECTION3_MODELS.md`](presentation/SECTION3_MODELS.md) — нарратив «Анализ моделей») |
| [`data/`](data/) | Сведения о датасете |
| [`experiments/`](experiments/) | Эксперименты участников |
| [`models/`](models/) | Описание моделей и конфигурации |
| [`models_labeling/`](models_labeling/) | Разметка моделями (вспомогательная) |

## Быстрый старт (воспроизведение части 3)

```bash
cd models_evaluation
pip install -r requirements.txt
python code/lib/download_models.py        # веса 7 моделей в общий кэш
python -m pytest code/tests/ -q           # юнит-тесты метрик
HF_HOME=/home/hf_cache python code/after_golden_set/run_golden.py --images data/golden
python code/after_golden_set/leaderboard.py
HF_HOME=/home/hf_cache python code/after_golden_set/experiments.py A B C D
HF_HOME=/home/hf_cache python code/after_golden_set/make_submission.py qwen3_vl_4b data/exam/images submission.jsonl
```

Подробнее — [`models_evaluation/README.md`](models_evaluation/README.md).
