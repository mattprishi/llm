# HW2: ускорение обучения LLM и кольцевые коммуникации

## Обучение на A100

Все эксперименты используют одну модель, токенизатор, выборку, split, длину 512,
BF16, AdamW fused, learning rate, scheduler и seed из исходного шаблона.
Меняются только разрешённые заданием механизмы оптимизации.

Итог: baseline — 7 067 useful tokens/s; FlashAttention 2 + packing — 22 238
tokens/s, `S=3.1465`. Полный разбор находится в [`report.md`](report.md), краткий
текст для формы — в [`submission.md`](submission.md), итоговая метрика — в
[`summary.json`](summary.json), все успешные запуски — в [`results/`](results/).

```bash
docker build -t llm-hw2 .
docker run --rm --gpus all --ipc=host \
  -v "$PWD/data:/app/data" \
  -v "$PWD/results:/app/results" \
  -v "$PWD/logs:/app/logs" \
  llm-hw2 bash scripts/run_suite.sh
```

Первый запуск скачивает и токенизирует данные. Это происходит до измеряемого участка
обучения. Suite выполняет baseline 300 секунд, короткий screening, затем два лучших
кандидата по 300 секунд с новой инициализацией модели.

Для учебной машины с driver 470 автоматический suite не повторяет заведомо
неуспешные конфигурации `batch64`, `torch_compile` и `liger`: первая превысила
80 GB VRAM, а две последние требуют Triton kernels, которые этот драйвер не
загружает. Их ошибки сохраняются в логах как результаты screening.

Один эксперимент можно запустить отдельно:

```bash
EXPERIMENT=fa2_packing RUN_NAME=manual_fa2_packing TRAIN_SECONDS=90 python3 train.py
```

Имена доступных конфигураций определены в `EXPERIMENTS` внутри `train.py`.
Каталог запуска должен отсутствовать: это защищает результаты от случайной перезаписи.

## Кольцевые коммуникации

Notebook использует Gloo и CPU:

```bash
jupyter nbconvert --execute --to notebook --inplace communications.ipynb
```

Он создаёт `collectives.py` и `check.py`, затем сравнивает реализацию с collectives
PyTorch на двух и четырёх процессах.
