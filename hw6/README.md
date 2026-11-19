# HW6. Инференс Qwen3-8B

- [Отчёт по трём экспериментам](report.md)
- [Сводка всех 24 прогонов](evidence/runs.csv) и [сравнение при 120 RPM](evidence/tuning_120rpm.csv)
- [График спекуляции](evidence/spec-tokens4096-ngram8.png) и [график SLA](evidence/sla-tokens4096fi-off.png)
- [Данные для перепроверки результатов](evidence/benchmark-records.tar.gz)

## Пересчёт сохранённых результатов

Из каталога `hw6`, с Python 3 и Matplotlib 3.9.4.

```bash
mkdir -p results/review
tar -xzf evidence/benchmark-records.tar.gz -C results/review
python3 analyze.py results/review --spec tokens4096 ngram8 --sla tokens4096fi off
```

Архив содержит параметры и команды запусков, агрегаты AIPerf, числовые метрики и метки времени по запросам. По нему можно пересчитать таблицы и графики на CPU.

## Повторные замеры на GPU

Используется Linux с Docker, NVIDIA Container Toolkit и `flock`. В `data/requests.jsonl` положите оригинальный файл из материалов курса. Скрипт проверяет его SHA-256. Ревизия модели и токенизатора закреплена в `model-revision`.

```bash
docker build -t llm-hw6 .

export HW6_IMAGE=llm-hw6
# Укажите индекс выделенной карты.
export HW6_GPU_INDEX=0
export HW6_GPU_UUID="$(nvidia-smi -i "$HW6_GPU_INDEX" --query-gpu=uuid --format=csv,noheader)"

bash run_vm.sh baseline off benchmark 120 baseline-120
bash run_vm.sh tokens4096fi off benchmark 120 flashinfer-120
bash run_vm.sh tokens4096fi off load 240 flashinfer-sla-240
```

Пресет `tokens4096fi off` использует бюджет планировщика 4096 токенов, FlashInfer и режим `interactivity`. Для исследования CPU n-gram используются `tokens4096 off` и `tokens4096 ngram8`. Режим `benchmark` подаёт 640 запросов, режим `load` держит заданную частоту 300 секунд. Первый запуск скачивает веса в локальный кеш.
