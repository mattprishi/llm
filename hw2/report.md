# HW2: ускорение обучения Qwen3 на одной A100 80 GB

## Executive summary

Лучшая конфигурация по главной метрике задания = **FlashAttention 2 + sequence
packing**. За 300 секунд она достигла **22 237.6 полезных токенов/с**, тогда как
baseline показал **7 067.4 токенов/с**. Итоговое ускорение = 3.1465

Peak allocated memory почти не изменилась: 40.17 GiB против 41.24 GiB у baseline.
При этом peak reserved выросла до 76.20 GiB, поэтому конфигурация близка к пределу
80 GB и имеет небольшой запас до OOM. Validation loss составил 6.6746 против
6.6006 у baseline.

Практически наиболее сбалансированным вариантом оказался **FlashAttention 2 +
padding-free**: 18 204.0 токенов/с, ускорение 2.5758, всего 15.18 GiB peak
allocated memory и лучший validation loss 5.1540. Он медленнее packing на 22.2%,
но существенно экономнее по памяти и успевает сделать 951 optimizer step вместо
208 за те же пять минут.

## Результаты экспериментов

В таблице приведены все успешные запуски. `S` рассчитано относительно
300-секундного baseline; для коротких запусков это screening-оценка, а не
финальный benchmark. `Allocated / reserved` — пиковая CUDA-память в GiB.

| Запуск | Время, с | Microbatch × accum | Механизмы | Токенов/с | S | Allocated / reserved, GiB | Val loss | Steps |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| baseline | 300.21 | 32 × 2 | SDPA | 7 067.4 | 1.0000 | 41.24 / 43.44 | 6.6006 | 368 |
| batch16_accum4 | 75.29 | 16 × 4 | меньший microbatch | 8 091.1 | 1.1449 | 24.20 / 25.47 | 8.1630 | 104 |
| checkpointing | 75.34 | 32 × 2 | activation checkpointing | 5 986.6 | 0.8471 | 17.39 / 23.97 | 8.3297 | 77 |
| FlashAttention 2 | 75.74 | 32 × 2 | FA2 | 7 671.9 | 1.0855 | 39.17 / 41.60 | 8.1977 | 99 |
| FA2 + packing, screen | 75.31 | 32 × 2 | FA2, packing, padding-free | 22 562.4 | 3.1925 | 40.15 / 76.20 | 8.3129 | 52 |
| FA2 + padding-free, screen | 75.03 | 32 × 2 | FA2, padding-free | 19 325.0 | 2.7344 | 15.11 / 21.24 | 7.2304 | 251 |
| **FA2 + packing, final** | **300.31** | **32 × 2** | **FA2, packing, padding-free** | **22 237.6** | **3.1465** | **40.17 / 76.20** | **6.6746** | **208** |
| FA2 + padding-free, final | 300.27 | 32 × 2 | FA2, padding-free | 18 204.0 | 2.5758 | 15.18 / 25.79 | **5.1540** | **951** |

Полная таблица находится в
[`results/experiments.csv`](results/experiments.csv), исходные метрики — в
`results/<run_name>/summary.json`.

## Неуспешные конфигурации тоже ограничили пространство решений

| Конфигурация | Результат | Диагноз |
|---|---|---|
| batch 64 × accum 1 | OOM | процесс занимал 75.38 GiB и пожадничал еще на 6.14 GiB |
| FA2 + packing, batch 64 | OOM | процесс занимал 73.69 GiB и запросил ещё 6.09 GiB |
| `torch.compile` | ошибка до первого шага | Triton: `device kernel image is invalid` |
| Liger | ошибка до первого шага | тот же сбой загрузки Triton kernel |
| FA2 + packing + Liger, batch 64 | ошибка до первого шага | тот же сбой Triton |
| activation offloading | ошибка backward | несовместимое выравнивание `attn_bias.stride(1)`; дополнительно сработал guard по CPU RAM |

Ошибки `torch.compile` и Liger относятся к конкретному стеку старого driver 470
и Triton из контейнера, а не к алгоритмам как таковым. Они не использовались при
выборе финалиста.

## Что объясняет наблюдаемые эффекты

### 1. FlashAttention 2 даёт умеренный самостоятельный выигрыш

При неизменных batch и accumulation переход SDPA → FlashAttention 2 увеличил
screening-throughput на 8.6% и снизил peak allocated memory примерно на 5.0%.
При длине 512 attention — лишь часть полной стоимости 960.9M-модели, поэтому
одиночный выигрыш ожидаемо намного меньше, чем у методов, устраняющих padding.

### 2. Packing максимизирует именно полезные токены

Исходные параграфы имеют переменную длину, а baseline выполняет вычисления и над
padding. Packing собирает несколько примеров в блоки длины 512; почти вся работа
GPU начинает приходиться на позиции с `labels != -100`. В сочетании с FA2 это
даёт 3.15× ускорение по целевой метрике. Цена — высокая reserved memory
(76.20 GiB), меньше optimizer steps и validation loss немного хуже baseline
(+0.074 абсолютного loss).

### 3. Padding-free — лучший компромисс скорости, памяти и сходимости

Padding-free не разрезает и не объединяет примеры в packed blocks, но разворачивает
batch без padding и передаёт position information в FA2. В финальном запуске это
дало 2.58× throughput, на 63.2% меньшую allocated memory и на 21.9% меньший
validation loss относительно baseline. За пять минут конфигурация выполнила 951
optimizer step, что объясняет более низкий loss при меньшем числе обработанных
полезных токенов, чем у packing.

### 4. Checkpointing меняет память на повторные вычисления

Activation checkpointing снизил peak allocated memory на 57.8%, но уменьшил
throughput на 15.3%. На A100 80 GB baseline уже помещается, поэтому recompute не
открывает полезный более крупный batch и остаётся чистым проигрышем по главной
метрике.

### 5. Меньший microbatch оказался быстрее при том же effective batch

`batch=16, accumulation=4` сохранил effective batch 64, но дал 1.145× throughput
и снизил allocated memory на 41.3% относительно baseline. Вероятное объяснение —
dynamic padding определяется максимумом внутри каждого microbatch: в меньшей
группе средний максимум короче, и GPU реже считает padding. Это интерпретация
одного screening-run, а не отдельный причинный эксперимент.

## Сетап и определения

- GPU: NVIDIA A100 80 GB PCIe; driver 470.182.03; MIG disabled.
- Контейнер: CUDA runtime 11.8; host `nvidia-smi` сообщает CUDA 11.4.
- PyTorch 2.6.0+cu118, Transformers 4.56.2, TRL 0.23.1.
- Модель: Qwen3, 960 881 664 параметра, случайная инициализация каждого запуска.
- Токенизатор: `ai-forever/rugpt3small_based_on_gpt2`, зафиксированная revision.
- Данные: `wikimedia/wikipedia`, `20231101.ru`, зафиксированная revision.
- Train: 244 768 параграфов; validation loss вычисляется на одних и тех же 512
  примерах и 50 694 target tokens.
- Последовательность: максимум 512 токенов.
- Неизменяемые параметры: BF16, TF32, `adamw_torch_fused`, LR `3e-4`,
  `constant_with_warmup`, 200 warmup steps, weight decay 0.01, seed 42.
- Effective batch во всех успешных конфигурациях равен 64 примерам.

Полезный токен — позиция, для которой `label != -100`. Основная метрика:

\[
\text{tokens/s} =
\frac{\sum \text{useful tokens after timing warmup}}
     {\sum \text{step duration after timing warmup}}.
\]

Первые десять optimizer steps исключаются из throughput-метрики. Peak memory
измеряется через `torch.cuda.max_memory_allocated()` и
`torch.cuda.max_memory_reserved()` после сброса статистики в начале train-loop.

## Экспериментальный дизайн

Baseline и два лучших кандидата запускались с нуля по 300 секунд на одной и той
же GPU. Screening длился 75 секунд. Перед каждым запуском создавалась новая модель
с seed 42; датасет, split и порядок данных оставались фиксированными. Evaluation
выполнялся после остановки train timer и не входит в tokens/s.

Исследованы пять семейств механизмов: microbatch/gradient accumulation,
activation checkpointing, attention backend, padding-free batching и sequence
packing. Дополнительно предприняты попытки использовать `torch.compile`, Liger и
activation offloading. Два обязательных сочетания — FA2 + packing и FA2 +
padding-free — успешно проверены в коротком и финальном режимах.

## Устойчивость и ограничения

- Повторных запусков с другими seeds нет, поэтому нельзя оценить дисперсию
  throughput и validation loss.
- У коротких и финальных запусков разная длительность; итоговое `S` использует
  только сопоставимые 300-секундные baseline и final.
- Throughput финального packing-run отличается от screening лишь на −1.4%, а
  padding-free — на −5.8%; ранжирование двух финалистов сохранилось.
- Validation loss сравнивает качество после фиксированного wall-clock, а не после
  одинакового числа optimizer steps или токенов. Это соответствует заданию, но
  не изолирует статистическую эффективность метода.
- Packing меняет группировку исходных последовательностей, хотя набор данных и
  split остаются теми же. Этот механизм явно разрешён заданием.
- Peak reserved memory packing достигает 96% физической VRAM; конфигурация
  чувствительна к любому внешнему GPU-процессу.
- Старый driver не позволил получить валидные метрики для Triton-зависимых
  `torch.compile` и Liger.


## Кольцевые коммуникации

В `collectives.py` реализованы:

1. `ring_reduce_scatter`: за `world_size - 1` фаз передаёт partial chunk правому
   соседу, принимает chunk слева и накапливает сумму; каждый rank получает свой
   редуцированный shard.
2. `ring_all_gather`: распространяет shard каждого rank по тому же кольцу и
   возвращает конкатенацию в порядке rank.
3. `ring_all_reduce`: последовательно вызывает две предыдущие операции; результат
   является суммой, а не средним.

Входные CPU-тензоры не изменяются. `communications.ipynb` сравнивает результаты с
`torch.distributed.reduce_scatter_tensor`, `all_gather` и `all_reduce` для shard
sizes 1, 7 и 31. Сохранённые проверки успешно прошли на 2 и 4 Gloo-процессах.

## Итог

Для критерия задания выбирается **FA2 + packing** с ускорением **S = 3.1465**.
Для более консервативного рабочего режима предпочтительнее **FA2 + padding-free**:
он уступает 22.2% throughput, но имеет большой запас VRAM и лучший validation
loss.
