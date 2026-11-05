# HW4. Full fine-tuning и LoRA на Banking77

Дообучаем Qwen2.5-1.5B-Instruct маршрутизировать банковские обращения по 77 интентам. Сравниваем Base, Full FT и пять конфигураций LoRA на 770 общих тестовых примерах по accuracy и macro-F1. [Отчёт и ручной разбор](report.md)

Сначала вышла фигня: я взял задачу, которую исходная модель уже неплохо решала, а после дообучения она стала отвечать аккуратнее и ошибаться чаще. [Подробности в `_deprecated`](_deprecated/README.md)

## Воспроизведение

Из папки `hw4_lora` на машине с CUDA. Укажите номер доступной BF16-карты в `CUDA_VISIBLE_DEVICES`.

```bash
(
  set -euo pipefail
  export CUDA_VISIBLE_DEVICES=1
  bash setup.sh
  .venv/bin/python banking.py
  for run in base full lora rank alpha dropout targets; do
    .venv/bin/python -u experiment.py "$run" 2>&1 | tee "run-$run.log"
  done
  .venv/bin/python summarize.py
)
```

Подготовка скачивает официальные CSV, сверяет их контрольные суммы и выбирает по 30 train и 10 test на каждый интент после удаления дубликатов. В `runs/` сохраняются конфиги, ответы, метрики, история обучения и восемь примеров для ручной проверки.
