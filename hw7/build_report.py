#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import random
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from statistics import quantiles
from typing import Any


QUANT_RUNS = {
    "gptq_massive": "GPTQ W4A16 / MASSIVE",
    "gptq_wikitext": "GPTQ W4A16 / WikiText",
    "awq_massive": "AWQ W4A16 / MASSIVE",
    "sq_05_massive": "SmoothQuant + GPTQ W8A8 / MASSIVE, α=0,5",
    "sq_08_massive": "SmoothQuant + GPTQ W8A8 / MASSIVE, α=0,8",
}
METRICS = {
    "quality": "Quality",
    "intent_accuracy": "Intent Accuracy",
    "slot_micro_f1": "Slot Micro F1",
    "frame_exact_match": "Frame Exact Match",
    "valid_json_rate": "Valid JSON Rate",
}
BOOTSTRAP_SEED = 42
BOOTSTRAP_REPLICATES = 10_000


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def read_table(path):
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if len({row["run"] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate runs in {path}")
    return {row["run"]: row for row in rows}


def notebook_metrics(notebook):
    cell = next(cell for cell in read_json(notebook)["cells"] if cell["id"] == "metrics")
    tree = ast.parse("".join(cell["source"]))
    names = {"normalize_value", "parse_answer", "slot_counter", "compute_metrics"}
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in definitions} != names:
        raise ValueError("The notebook metrics cell is incomplete")
    namespace = {"json": json, "Counter": Counter, "Sequence": Sequence, "Any": Any}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(notebook), "exec"), namespace)
    return namespace["compute_metrics"], namespace["parse_answer"], namespace["slot_counter"]


def check_score(actual, saved, source):
    for metric in (*METRICS, "n"):
        if not math.isclose(float(actual[metric]), float(saved[metric]), rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"{source}: {metric} {saved[metric]} != recomputed {actual[metric]}")


def verify_evaluation(artifacts, runs, compute_metrics, labels):
    scores = {}
    dev = None
    for run in runs:
        path = artifacts / f"{run}_dev_predictions.jsonl"
        rows = read_jsonl(path)
        reference = [(str(row["id"]), row["gold"]) for row in rows]
        if len(reference) != 256 or len({row_id for row_id, _ in reference}) != 256:
            raise ValueError(f"{path}: expected 256 unique dev examples")
        if dev is not None and reference != dev:
            raise ValueError(f"{path}: dev IDs, order, or gold labels differ from baseline")
        dev = reference
        scores[run] = compute_metrics(
            [row["prediction"] for row in rows], [gold for _, gold in dev], *labels,
        )
        check_score(scores[run], read_json(artifacts / f"{run}_dev.json"), path)
    return scores, {row_id for row_id, _ in dev}


def quality_terms(rows, parse_answer, slot_counter, labels):
    terms = []
    for row in rows:
        gold = parse_answer(row["gold"], *labels)
        pred = parse_answer(row["prediction"], *labels)
        gold_slots = slot_counter(gold)
        pred_slots = slot_counter(pred) if pred is not None else Counter()
        intent_ok = pred is not None and pred["intent"] == gold["intent"]
        matches = sum((gold_slots & pred_slots).values())
        slot_count = sum(gold_slots.values()) + sum(pred_slots.values())
        terms.append((intent_ok, pred is not None, matches, slot_count))
    return terms


def resample_quality(terms, indices):
    correct_intent = valid = matches = total_slots = 0
    for index in indices:
        intent_ok, json_ok, row_matches, row_slots = terms[index]
        correct_intent += intent_ok
        valid += json_ok
        matches += row_matches
        total_slots += row_slots
    n = len(indices)
    # The total gold plus predicted slots is 2*TP + FP + FN.
    slot_f1 = (2 * matches / total_slots) if total_slots else float(valid == n)
    return 0.5 * (correct_intent / n + slot_f1)


def paired_quality_interval(artifacts, changed, reference, scores, parse_answer, slot_counter, labels):
    terms = []
    for run in (changed, reference):
        rows = read_jsonl(artifacts / f"{run}_dev_predictions.jsonl")
        terms.append(quality_terms(rows, parse_answer, slot_counter, labels))
    left, right = terms
    indices = range(len(left))
    for run, run_terms in zip((changed, reference), (left, right)):
        quality = resample_quality(run_terms, indices)
        if not math.isclose(quality, scores[run]["quality"], rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"{run}: bootstrap statistic disagrees with the notebook's Quality")

    # The same sampled dev row indices are used for both models on each draw.
    rng = random.Random(BOOTSTRAP_SEED)
    effects = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sample = rng.choices(indices, k=len(indices))
        effects.append(resample_quality(left, sample) - resample_quality(right, sample))
    cutpoints = quantiles(effects, n=40, method="inclusive")
    return cutpoints[0], cutpoints[-1]


def verify_teacher(artifacts, dev_ids, parse_answer, labels):
    cohorts = {}
    train_ids = set()
    for cohort in ("teacher_answers", "teacher_answers_extra"):
        rows = read_jsonl(artifacts / cohort / "answers.jsonl")
        ids = {str(row["id"]) for row in rows}
        if len(ids) != len(rows) or ids & train_ids or ids & dev_ids:
            raise ValueError(f"{cohort}: train IDs contain duplicates or overlap with another split")
        valid_count = sum(parse_answer(row["teacher_answer"], *labels) is not None for row in rows)
        cohorts[cohort] = (len(rows), valid_count)
        train_ids.update(ids)
    return cohorts


def verify_checkpoint(path, quantized):
    files = list(path.glob("*.safetensors"))
    config = read_json(path / "config.json")
    if not files or bool(config.get("quantization_config")) != quantized:
        raise ValueError(f"Wrong checkpoint format or weights missing: {path}")
    if quantized and config["quantization_config"].get("quantization_status") != "compressed":
        raise ValueError(f"Quantization status is not compressed: {path}")
    weight_maps = list(path.glob("*.safetensors.index.json"))
    for index in weight_maps:
        if set(read_json(index)["weight_map"].values()) != {weight.name for weight in files}:
            raise ValueError(f"Missing/unexpected shards for {index}")
    return sum(weight.stat().st_size for weight in files) / 2**30


def number(value, decimals=2, signed=False):
    sign = "+" if signed else ""
    return f"{value:{sign}.{decimals}f}".replace(".", ",")


def pct(value):
    return f"{number(100 * value)}%"


def difference(value):
    return f"{number(100 * value, signed=True)} п. п."


def interval_text(interval):
    low, high = (number(100 * value, signed=True) for value in interval)
    zero = ", включает ноль" if interval[0] <= 0 <= interval[1] else ""
    return f"95% ДИ от {low} до {high} п. п.{zero}"


def score_cells(score):
    return [pct(score[key]) for key in METRICS]


def table(headers, rows):
    def row(values):
        return "| " + " | ".join(map(str, values)) + " |"

    return "\n".join([row(headers), row(["---"] * len(headers)), *(row(values) for values in rows)])


def render(artifacts, notebook, out, verify_weights):
    compute_metrics, parse_answer, slot_counter = notebook_metrics(notebook)
    schema = read_json(artifacts / "label_schema.json")
    labels = (schema["intents"], schema["slot_types"])
    quant = read_table(artifacts / "quantization_dev.csv")
    trials = read_table(artifacts / "student_trials.csv")
    if set(quant) != {"baseline", *QUANT_RUNS}:
        raise ValueError("The approved quantization matrix is incomplete or contains unknown runs")
    runs = [*quant, "teacher", *trials, "student_sft"]
    scores, dev_ids = verify_evaluation(artifacts, runs, compute_metrics, labels)
    for run, saved in {**quant, **trials}.items():
        check_score(scores[run], saved, run)

    teacher_counts = verify_teacher(artifacts, dev_ids, parse_answer, labels)
    pilot_raw, pilot_valid = teacher_counts["teacher_answers"]
    full_raw = sum(raw for raw, _ in teacher_counts.values())
    full_valid = sum(valid for _, valid in teacher_counts.values())
    for run, expected in (("student_128", pilot_valid), ("student_128_high_lr", pilot_valid),
                          ("student_1024", full_valid)):
        if int(trials[run]["train_n"]) != expected:
            raise ValueError(f"{run}: training count differs from valid teacher records")

    selected, = (run for run, row in trials.items() if row["selected"].casefold() == "true")
    if scores[selected]["quality"] != max(scores[run]["quality"] for run in trials):
        raise ValueError("The selected trial does not have the highest Quality")
    check_score(scores["student_sft"], scores[selected], selected)

    sizes = {run: float(row["weights_gib"]) for run, row in quant.items()}
    if verify_weights:
        for run in QUANT_RUNS:
            size = verify_checkpoint(artifacts / run, quantized=True)
            if not math.isclose(sizes[run], size, rel_tol=0, abs_tol=1e-9):
                raise ValueError(f"{run}: checkpoint size differs from the experiment table")
        student_gib = verify_checkpoint(artifacts / "student_sft", quantized=False)

    quality = {run: score["quality"] for run, score in scores.items()}
    best_quant = max(QUANT_RUNS, key=quality.__getitem__)
    smallest = min(sizes[run] for run in QUANT_RUNS)
    smallest_names = ", ".join(QUANT_RUNS[run] for run in QUANT_RUNS if sizes[run] == smallest)
    baseline, teacher, student = (quality[run] for run in ("baseline", "teacher", "student_sft"))
    target = 0.9 * teacher
    target_result = "выполнена" if student >= target else "не выполнена"
    ratio = f"{pct(student / teacher)} качества учителя" if teacher else "целевое Quality равно нулю"
    pilot_lr_best = max(("student_128", "student_128_high_lr"), key=quality.__getitem__)
    if float(trials["student_1024"]["lr"]) != float(trials[pilot_lr_best]["lr"]):
        raise ValueError("The extended SFT run does not use the selected pilot learning rate")
    lr_effect = quality["student_128_high_lr"] - quality["student_128"]
    sample_effect = quality["student_1024"] - quality[pilot_lr_best]
    quant_pairs = [
        ("источник калибровки", "gptq_massive", "gptq_wikitext"),
        ("алгоритм W4A16", "awq_massive", "gptq_massive"),
        ("коэффициент сглаживания", "sq_08_massive", "sq_05_massive"),
    ]
    effects = [(name, changed, reference, quality[changed] - quality[reference])
               for name, changed, reference in quant_pairs]
    source_effect, method_effect, alpha_effect = [effect for *_, effect in effects]
    _, source_left, source_right = quant_pairs[0]
    source_interval = paired_quality_interval(
        artifacts, source_left, source_right, scores, parse_answer, slot_counter, labels,
    )
    _, alpha_left, alpha_right = quant_pairs[2]
    alpha_interval = paired_quality_interval(
        artifacts, alpha_left, alpha_right, scores, parse_answer, slot_counter, labels,
    )
    factor, changed, reference, effect = max(effects, key=lambda row: abs(row[-1]))
    intent_effect = scores[changed]["intent_accuracy"] - scores[reference]["intent_accuracy"]
    slot_effect = scores[changed]["slot_micro_f1"] - scores[reference]["slot_micro_f1"]
    max_effect = "Во всех трёх парах Quality совпало."
    if effect:
        max_effect = (
            f"Наибольший по модулю сдвиг Quality связан с фактором «{factor}» ({difference(effect)}). "
            f"В этой паре Intent Accuracy изменилась на {difference(intent_effect)}, "
            f"Slot Micro F1 — на {difference(slot_effect)}"
        )

    quant_table = table(
        ["Конфигурация", "Примеров", "Токенов калибровки", "Веса, ГиБ", *METRICS.values()],
        [["Qwen3-8B BF16", "—", "—", number(sizes["baseline"]), *score_cells(scores["baseline"])],
         *([
             name, int(float(quant[run]["calibration_n"])), int(float(quant[run]["calibration_tokens"])),
             number(sizes[run]), *score_cells(scores[run]),
         ] for run, name in QUANT_RUNS.items())],
    )
    models = table(
        ["Модель", *METRICS.values()],
        [[name, *score_cells(scores[run])] for run, name in (
            ("baseline", "Qwen3-8B, бейзлайн"),
            ("teacher", "Qwen3-32B, учитель"),
            ("student_sft", "Qwen3-8B, лучший ученик"),
        )],
    )
    trial_table = table(
        ["Прогон", "Валидных меток", "Learning rate", "Эпох", *METRICS.values()],
        [[f"**{run}**" if run == selected else run, row["train_n"], row["lr"], row["epochs"],
          *score_cells(scores[run])] for run, row in trials.items()],
    )

    def link(path, caption):
        return f"[{caption}]({os.path.relpath(path, start=out.parent)})"

    versions = read_json(artifacts / "environment.json")
    packages = ", ".join(f"`{package} {value}`" for package, value in versions.items())
    checkpoint_details = (
        f"Проверены файлы весов, размеры пяти сжатых чекпойнтов и состав шардов ученика "
        f"({number(student_gib)} ГиБ). " if verify_weights else ""
    )
    timings = []
    for stage, title in (("quant", "Квантизация"), ("teacher", "Учитель"), ("student", "SFT")):
        path = artifacts / f"run-{stage}.json"
        if path.is_file():
            timing = read_json(path)
            if timing["status"] == "completed":
                timings.append([
                    link(path, title), number(timing["elapsed_seconds"] / 60, 1),
                    number(timing["peak_allocated_gib"], 1), number(timing["peak_reserved_gib"], 1),
                ])
    runtime = ""
    if timings:
        headers = ["Стадия", "Время, мин", "Peak allocated, ГиБ", "Peak reserved, ГиБ"]
        runtime = "\n\nЗапуски на NVIDIA A100-SXM4-80GB. Для каждой стадии указано время от её старта до окончания. Стадии могли выполняться параллельно.\n\n" + table(headers, timings)
    provenance = artifacts / "run_provenance.json"
    provenance_link = (
        "\n- " + link(provenance, "Ревизии моделей и датасетов, параметры генерации и оборудование") + "."
        if provenance.is_file() else ""
    )
    checkpoint_location = (
        read_json(provenance).get("checkpoints_location") if provenance.is_file() else None
    )
    checkpoint_location_note = (
        f" Полные веса сохранены на VM в `{checkpoint_location}`." if checkpoint_location else ""
    )
    runner_path = notebook.with_name("run_vm.py")
    runner_link = (
        "\n- " + link(runner_path, "Запуск ноутбука по стадиям на VM") + "."
        if runner_path.is_file() else ""
    )
    bootstrap_replicates = f"{BOOTSTRAP_REPLICATES:,}".replace(",", " ")
    return f"""# HW7. Квантизация и дистилляция Qwen3

Лучшее Quality среди сжатых моделей дала **{QUANT_RUNS[best_quant]}** — **{pct(quality[best_quant])}** при {number(sizes[best_quant])} ГиБ весов. У BF16-бейзлайна Quality составило {pct(baseline)} при {number(sizes['baseline'])} ГиБ. Разница Quality — {difference(quality[best_quant] - baseline)}

Лучший ученик получил **{pct(student)} Quality** ({ratio}, {difference(student - baseline)} к бейзлайну). Цель ≥90% качества учителя **{target_result}**. Требуемое Quality на этой выборке — {pct(target)}.

## Данные и оценка

Задача — вернуть JSON с `intent` и `slots` для русского запроса голосовому ассистенту. Для оценки всех моделей использована одна и та же подвыборка из 256 записей MASSIVE ru-RU validation, seed 42. Модели получали одинаковый системный промпт со списком допустимых меток. Ответы генерировались жадно, батчами по 8, с лимитом 128 новых токенов и выключенным thinking.

Quality — среднее Intent Accuracy и Slot Micro F1. Slot Micro F1 учитывает каждое вхождение пары `(тип, значение)`, включая повторы, после нормализации регистра и пробельных символов. Frame Exact Match требует одновременно верного intent и совпадения всех пар. Valid JSON Rate учитывает синтаксис, схему и допустимость названий меток.

## Квантизация

Каждый прогон стартовал с Qwen3-8B BF16. Для калибровки взято по 128 записей из MASSIVE train или WikiText-2 train. К обоим источникам применён один шаблон запроса. В опытах квантизированы линейные слои с сохранением `lm_head` в BF16. В таблице указаны размеры файлов весов на диске. W4A16 загружался через `run_compressed=False`. Для W8A8 использован загрузчик `run_compressed=True`, который загружает замороженные INT8-веса, деквантует их при первом проходе и сохраняет динамическую INT8-квантизацию активаций.

{quant_table}

- **Источник калибровки.** В паре GPTQ W4A16 переход с WikiText на MASSIVE дал {difference(source_effect)} Quality ({interval_text(source_interval)}). Одинаковое число примеров даёт сопоставимый протокол. Вместе с предметной областью меняются язык и длины текстов.
- **Алгоритм.** На MASSIVE переход с GPTQ на AWQ при W4A16 дал {difference(method_effect)} Quality. Минимальный размер — {number(smallest)} ГиБ, или в {number(sizes['baseline'] / smallest)} раза меньше BF16. Его дали {smallest_names}.
- **Сглаживание.** Переход от α=0,5 к α=0,8 внутри W8A8 дал {difference(alpha_effect)} Quality ({interval_text(alpha_interval)}). Сравнение W4A16 с W8A8 одновременно меняет точность весов, обработку активаций и применение SmoothQuant.

Интервалы оценены парным процентильным бутстрэпом ({bootstrap_replicates} реплик, seed {BOOTSTRAP_SEED}). В каждой реплике выбирали с возвращением 256 dev ID и использовали эту выборку для обеих моделей. Slot Micro F1 пересчитывали по суммарным счётчикам слотов в реплике. Интервалы отражают вариацию по dev-примерам для фиксированных чекпойнтов, полученных с одним seed калибровки.

{max_effect}

## Дистилляция

{models}

Учитель показал {difference(teacher - baseline)} Quality относительно бейзлайна. Для обучения использовали жадные ответы учителя на MASSIVE train. Проверку JSON-схемы прошли {pilot_valid}/{pilot_raw} ответов в пилотной выборке и {full_valid}/{full_raw} в расширенной. Токены промпта и паддинга были замаскированы в loss. Цель обучения — токены ответа учителя и EOS.

{trial_table}

В каждом прогоне заново загружался Qwen3-8B. Обучались все параметры в BF16 с Adafactor и gradient checkpointing, microbatch 2, accumulation 4. Повышение learning rate на фиксированных {pilot_valid} примерах дало {difference(lr_effect)} Quality. Переход к {full_valid} примерам с лучшим из двух learning rate дал {difference(sample_effect)} Quality. При том же количестве эпох расширенный прогон получил также больше шагов оптимизатора. Dev использован и для выбора чекпойнта, и для приведённых оценок.

## Воспроизведение и файлы

Для расчёта таблиц `build_report.py` повторно применяет функции метрик из ноутбука к сохранённым предсказаниям. Для всех сравнений совпали 256 ID и эталонные ответы dev. В двух выборках train ID уникальны, проверено отсутствие пересечений между выборками и с dev. {checkpoint_details}Версии окружения — {packages}. {link(notebook, 'Ноутбук')}{runtime}

- {link(artifacts / 'quantization_dev.csv', 'Таблица квантизации')} и {link(artifacts / 'student_trials.csv', 'таблица прогонов SFT')}.
- {link(artifacts / 'baseline_dev_predictions.jsonl', 'Бейзлайн')}, {link(artifacts / 'teacher_dev_predictions.jsonl', 'учитель')} и {link(artifacts / 'student_sft_dev_predictions.jsonl', 'выбранный ученик')} на dev.
- {link(artifacts / 'teacher_answers/answers.jsonl', 'Первые ответы учителя')} и {link(artifacts / 'teacher_answers_extra/answers.jsonl', 'дополнительные ответы')} из MASSIVE train.
- Имена чекпойнтов в `artifacts_compression_hw` — {', '.join(f'`{run}`' for run in QUANT_RUNS)} и `student_sft`.{checkpoint_location_note}{provenance_link}{runner_link}
"""


def main():
    folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Verify HW7 artifacts and render the report")
    parser.add_argument("--artifacts", type=Path, default=folder / "artifacts_compression_hw")
    parser.add_argument("--notebook", type=Path, default=folder / "homework_compression_student.ipynb")
    parser.add_argument("--output", type=Path, default=folder / "report.md")
    parser.add_argument("--metrics-only", action="store_true", help="Run against a copy without model shards")
    args = parser.parse_args()
    report = render(args.artifacts, args.notebook, args.output, not args.metrics_only)
    args.output.write_text(report, encoding="utf-8")
    print(f"Verified HW7 artifacts and wrote {args.output}")


if __name__ == "__main__":
    main()
