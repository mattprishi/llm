# HW2: ускорение обучения LLM и кольцевые коммуникации

## Обучение на A100

Все эксперименты используют одну модель, токенизатор, выборку, split, длину 512,
BF16, AdamW fused, learning rate, scheduler и seed из исходного шаблона.
Меняются только разрешённые заданием механизмы оптимизации.

Итог: baseline — 7 067 useful tokens/s; FlashAttention 2 + packing — 22 238
tokens/s, `S=3.1465`. Полный разбор находится в [`report.md`](report.md), итоговая метрика — в
[`summary.json`](summary.json), все успешные запуски — в [`results/`](results/).
