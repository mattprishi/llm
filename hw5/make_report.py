"""Build the Russian HW5 PDF from audited evaluation and training artifacts."""
import argparse
from html import escape
import json
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory

from audit_results import GROUPS, inspect_run, read_jsonl

LABELS = {"base": "Baseline", "rl1": "RL-1", "rl2": "RL-2", "rl3": "RL-3"}
HYPOTHESES = {
    "rl1": "GRPO + точная награда. Проверяем, достаточно ли различий внутри группы для обучения.",
    "rl2": "GRPO + частичная награда. Проверяем, помогает ли сигнал за верную функцию и часть аргументов.",
    "rl3": "Усреднение по токенам (loss_type=dapo) + та же частичная награда. Проверяем влияние длины ответа.",
}


def load(path):
    return json.loads(path.read_text())


def build(root, output, font):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    examples = read_jsonl(root / "bfcl.jsonl")
    inspected = {name: inspect_run(root / "results" / name, examples) for name in LABELS}
    results = {name: pair[0] for name, pair in inspected.items()}
    if any(result[key] != results["base"][key] for result in results.values() for key in ("versions", "prompt_lengths_sha256")):
        raise ValueError("Evaluation environments or prompt lengths differ")
    summaries = {name: load(root / "runs" / name / "summary.json") for name in HYPOTHESES}
    configs = {name: load(root / "runs" / name / "config.json") for name in HYPOTHESES}
    histories = {name: read_jsonl(root / "runs" / name / "history.jsonl") for name in HYPOTHESES}
    eda_path = root / "data" / "eda.json"
    eda = load(eda_path if eda_path.exists() else root / "evidence" / "toolace_eda.json")
    common = configs["rl1"]["arguments"]
    controls = {key: value for key, value in common.items() if key not in ("experiment", "output")}
    excluded = {"output_dir", "loss_type"}
    trainer_controls = {k: v for k, v in configs["rl1"]["trainer"].items() if k not in excluded}
    expected = {"rl1": ("strict", "grpo"), "rl2": ("shaped", "grpo"), "rl3": ("shaped", "dapo")}
    for name, config in configs.items():
        current = {key: value for key, value in config["arguments"].items() if key not in ("experiment", "output")}
        trainer = {k: v for k, v in config["trainer"].items() if k not in excluded}
        if current != controls or trainer != trainer_controls:
            raise ValueError(f"Training controls differ: {name}")
        if (config["reward"], config["trainer"]["loss_type"]) != expected[name]:
            raise ValueError(f"Unexpected reward or loss: {name}")
        for key in ("runtime", "data_sha256", "validation_ids", "lora", "revision", "gpu"):
            if config[key] != configs["rl1"][key]:
                raise ValueError(f"Training control differs: {name}, {key}")
        steps = [r["step"] for r in histories[name] if "loss" in r]
        if steps != list(range(1, common["steps"] + 1)):
            raise ValueError(f"Training history incomplete: {name}")
        best = max(summaries[name]["validation"], key=lambda r: (r["strict"], r["step"]))
        if best != summaries[name]["best"]:
            raise ValueError(f"Invalid checkpoint selection: {name}")
    pdfmetrics.registerFont(TTFont("Body", str(font)))
    body = ParagraphStyle("body", fontName="Body", fontSize=9.6, leading=14, spaceAfter=8)
    small = ParagraphStyle("small", parent=body, fontSize=8, leading=10, spaceAfter=6)
    heading = ParagraphStyle("heading", parent=body, fontSize=16, leading=21, spaceBefore=6, textColor=colors.HexColor("#123552"))
    title = ParagraphStyle("title", parent=heading, fontSize=25, leading=30, spaceAfter=13)
    story = []

    def p(text, style=body):
        story.append(Paragraph(escape(str(text)), style))

    def table(rows, widths, padding=7):
        element = Table(rows, colWidths=widths, hAlign="LEFT")
        element.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), "Body"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#123552")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#eef5f9"), colors.white]),
            ("TOPPADDING", (0, 0), (-1, -1), padding),
            ("BOTTOMPADDING", (0, 0), (-1, -1), padding),
        ]))
        story.extend([element, Spacer(1, 10)])

    p("HW5 · RL для вызова инструментов", title)
    p(f"Qwen3.5-2B · {configs['rl1']['gpu']} · seed {common['seed']} · {len(examples)} задач BFCL")
    p("Данные и постановка", heading)
    p(f"Источник обучения — Team-ACE/ToolACE ({eda['source_dialogues']:,} диалогов, revision {eda['revision'][:12]}). Схемы приведены к формату инструментов Qwen. "
      "История предыдущих ходов и результатов инструментов входит в промпт; эталон текущего вызова используется в награде. "
      "Сплит сгруппирован по первому запросу до нарезки диалога. Дубликаты промптов удалены. Лимит промпта — 1536 токенов.")
    rows = [["Сплит", "Всего", "Один вызов", "Несколько", "Без вызова", "p50 / p95 токенов"]]
    for split, stats in eda["splits"].items():
        counts, lengths = stats["kinds"], stats["prompt_tokens"]
        rows.append([split, stats["rows"], counts["single_call"], counts["multi_call"], counts["no_call"],
                     f"{lengths['p50']} / {lengths['p95']}"])
    table(rows, [65, 55, 85, 75, 80, 125])
    p(f"Отсев: {eda['filter_counts']['unsupported_schema_header']} диалогов с неподдержанной схемой или пустым списком инструментов, "
      f"{eda['filter_counts']['long_prompt']} длинных промптов. Текст сразу после результата инструмента не служит целью RL.", small)
    p("Награда", heading)
    p("Точный набор вызовов с совпадающими именами и аргументами получает 1. Порядок вызовов свободный, повторы учитываются. "
      "Непустой ответ без вызова получает 1 на примере с эталоном []. Пустой, оборванный или неразбираемый ответ получает 0.")
    p("Частичная награда при неточном наборе: 0,9 × сумма баллов лучшего взаимно однозначного сопоставления / max(число предсказанных, число эталонных). "
      "Разные имена дают 0; одинаковые имена дают 0,25 + 0,75 × долю совпавших аргументов среди объединения ключей. "
      "Строгая награда соответствует выполнению всей задачи; частичная даёт сигнал там, где несколько неполных ответов получили бы один и тот же ноль.")
    p("Пример ToolACE: Market_Trends_API(trend_type=MARKET_INDEXES, country=us) → 1/1. Если country=ca: строгая 0, частичная 0,5625. "
      "Два верных вызова Timezones из трёх: 0/0,6. Неполная разметка <tool_call>: 0/0.")
    p("Итоговый BFCL, %", heading)
    rows = [["Модель", "Overall", "Simple", "Multiple", "Parallel", "Par. mult.", "Irrel."]]
    for name, r in results.items():
        rows.append([LABELS[name], f"{100*r['overall']:.2f}", *(f"{100*r['groups'][g]:.2f}" for g in GROUPS)])
    table(rows, [73, 68, 67, 68, 68, 75, 66])
    p("Overall — среднее пяти групп; Multiple и Irrelevance сначала усредняются по двум подкатегориям. "
      "Все запуски оценены штатным evaluate.py с закреплённым токенизатором, temperature=0, thinking=False, max_new_tokens=1024. "
      "Для прогрева vLLM без nvcc использован нативный sampler во всех четырёх оценках.", small)
    story.append(PageBreak())
    p("Обучение и выбор чекпоинта", title)
    p(f"LoRA rank 16, LR {common['lr']:.1g}, G={common['generations']}, batch={common['batch']}, "
      f"{common['steps']} шагов на запуск, предел ответа {common['max_completion']} токенов. Выбор шага по строгой метрике ToolACE validation. "
      "Для отбора берём фиксированный срез: по 12 примеров каждого типа. BFCL используется только для итогового сравнения.")
    groups = common["steps"] * common["batch"] // common["generations"]
    p(f"На конфигурацию: {groups} групп, {groups * common['generations']} сгенерированных ответов; один seed. "
      "Это короткий контролируемый запуск, который использует небольшую часть train-сплита.", small)
    train_runtime = configs["rl1"]["runtime"]
    eval_runtime = results["base"]["versions"]
    p(f"Обучение: torch {train_runtime['torch']}, TRL {train_runtime['trl']}, Transformers {train_runtime['transformers']}; "
      f"оценка: vLLM {eval_runtime['vllm']}, BFCL {eval_runtime['bfcl-eval']}.", small)
    rows = [["Модель", "Шаг", "Strict val", "N val", "Шагов с градиентом", "Обучение, с"]]
    for name, s in summaries.items():
        p(LABELS[name] + " · " + HYPOTHESES[name], small)
        rows.append([LABELS[name], s["best"]["step"], f"{s['best']['strict']:.1%}", s["best"]["n"],
                     s.get("nonzero_grad_steps", "—"), f"{s['elapsed_seconds']:.0f}"])
    table(rows, [65, 58, 73, 56, 128, 105], padding=4)
    with TemporaryDirectory() as temp:
        plot = Path(temp) / "curves.png"
        fig, axes = plt.subplots(2, 3, figsize=(9.1, 4.7), sharex="col", layout="constrained")
        for j, (name, h) in enumerate(histories.items()):
            steps = [r["step"] for r in h if "loss" in r]
            loss = [float(r["loss"]) for r in h if "loss" in r]
            axes[0, j].plot(steps, loss, color="#294f72", lw=1.2)
            axes[0, j].set_title(LABELS[name])
            for key, color in (("reward", "#a34c2f"), ("toolace/strict", "#297e61")):
                points = [(r["step"], float(r[key])) for r in h if key in r]
                if points:
                    axes[1, j].plot(*zip(*points), color=color, label=key, lw=1.2)
            axes[1, j].set_ylim(-0.05, 1.05)
            axes[1, j].set_xlabel("Optimizer step")
            for ax in axes[:, j]:
                ax.grid(alpha=.2)
                ax.axvline(summaries[name]["best"]["step"], ls=":", color="gray")
        axes[0, 0].set_ylabel("Loss")
        axes[1, 0].set_ylabel("Reward")
        if axes[1, -1].get_legend_handles_labels()[0]:
            axes[1, -1].legend(fontsize=7, loc="lower left")
        fig.savefig(plot, dpi=180)
        plt.close(fig)
        story.append(Image(str(plot), width=460, height=238))
        p("По вертикали отмечен выбранный шаг. Strict — строгая метрика на сэмплированных train-ответах, общая для всех запусков; "
          "Reward — оптимизируемая награда конкретного запуска. При одинаковой награде у всех G ответов advantage равен нулю. "
          "В GRPO loss может быть близок к нулю при ненулевом градиенте: слагаемые с центрированными advantage компенсируются по значению.", small)
        p("Как часто модель вызывает инструмент", heading)
        counts_table = [["Модель", "Вызов нужен (N=300)", "Вызов не нужен (N=50)", "Верно (N=350)"]]
        for name, row in results.items():
            calls = row["with_parsed_call"]
            counts_table.append([LABELS[name], calls["call_required"], calls["irrelevance"], row["micro_correct"]])
        table(counts_table, [94, 132, 149, 112], padding=4)
        p("В двух средних столбцах считаются ответы хотя бы с одним разобранным вызовом. Это признак поведения; правильность аргументов проверяет BFCL.", small)
        story.append(PageBreak())
        p("Что изменилось и где модель ошибается", title)
        base = results["base"]["overall"]
        best = max(HYPOTHESES, key=lambda name: results[name]["overall"])
        reward_effect = 100*(results["rl2"]["overall"] - results["rl1"]["overall"])
        loss_effect = 100*(results["rl3"]["overall"] - results["rl2"]["overall"])
        strongest = "смена награды" if abs(reward_effect) > abs(loss_effect) else (
            "смена усреднения loss" if abs(loss_effect) > abs(reward_effect) else "оба изменения дали одинаковую по модулю разницу")
        p(f"Лучший RL-запуск: {LABELS[best]}, {results[best]['overall']:.2%}, "
          f"разница с базой {100*(results[best]['overall']-base):+.2f} п.п. "
          f"RL-2 минус RL-1: {reward_effect:+.2f} п.п.; RL-3 минус RL-2: {loss_effect:+.2f} п.п. "
          f"Наибольшая разница в этой серии: {strongest}. Сравнение включает общий протокол выбора шага; "
          "один seed и короткое обучение ограничивают вывод об устойчивости.")
        base_flags, best_flags = inspected["base"][1], inspected[best][1]
        fixed = sum(not base_flags[i] and best_flags[i] for i in base_flags)
        lost = sum(base_flags[i] and not best_flags[i] for i in base_flags)
        gains = {group: 100*(results[best]["groups"][group]-results["base"]["groups"][group]) for group in GROUPS}
        p(f"На тех же 350 задачах лучший RL-чекпоинт исправил {fixed} ответов и испортил {lost} верных ответов базы. "
          "Изменения по группам, п.п.: " + "; ".join(f"{g} {v:+.2f}" for g, v in gains.items()) + ".", small)
        for name, h in histories.items():
            rs = [float(r["reward"]) for r in h if "reward" in r]
            if rs:
                n = max(1, len(rs)//3)
                ties = [float(r["frac_reward_zero_std"]) for r in h if "frac_reward_zero_std" in r]
                clipping = [float(r["completions/clipped_ratio"]) for r in h if "completions/clipped_ratio" in r]
                p(f"{LABELS[name]}: награда, первая и последняя треть шагов: {mean(rs[:n]):.3f} → {mean(rs[-n:]):.3f}; "
                  f"BFCL {results[name]['overall']:.2%}." +
                  (f" Группы без разброса: {mean(ties):.1%}; обрезанные ответы: {mean(clipping):.1%}." if ties and clipping else ""), small)
        p("Награда измеряется на ToolACE с обучающим sampling и зависит от формулы; BFCL использует другой набор "
          "задач, жадную генерацию и AST-проверку. В отборочном ToolACE no-call весит 1/3, в BFCL Overall — 1/5. "
          "Сравнение средних reward первой и последней трети дополнительно зависит от сложности попавших в них запросов.")
        predictions = read_jsonl(root / "results" / best / "predictions.jsonl")
        errors = [(ex, row) for ex, row in zip(examples, predictions, strict=True) if not row["check"]["valid"]]
        preferred = {
            "live_irrelevance_617-195-5": "Не было маршрута и даты; вызов подставил недостающие данные вместо уточнения.",
            "simple_5": "Модель решает уравнение в тексте. Задача требует вызова solve_quadratic.",
            "simple_150": "Название функции верно, но ставка 3% передана как 3 вместо 0,03.",
        }
        for label, predicate, case in (
            ("Лишний вызов", lambda r: "irrelevance" in r["category"] and r["calls"], "live_irrelevance_617-195-5"),
            ("Нет разобранного вызова", lambda r: "irrelevance" not in r["category"] and not r["calls"], "simple_5"),
            ("Ошибка в наборе или аргументах", lambda r: "irrelevance" not in r["category"] and r["calls"] and r["finish_reason"] != "length", "simple_150"),
        ):
            p(label, heading)
            matches = [(ex, row) for ex, row in errors if predicate(row)]
            match = next(((ex, row) for ex, row in matches if best == "rl1" and row["id"] == case), matches[0] if matches else None)
            if match:
                ex, row = match
                request = next((x["content"] for x in reversed(ex["question"]) if x["role"] == "user"), "")
                p(f"{row['id']} ({row['category']}), запрос: {str(request)[:260]}", small)
                p(f"Ожидалось: {str(ex['ground_truth'])[:270]}. Ответ: {str(row['calls'] or row['response'])[:270]}. "
                  f"Проверка: {str(row['check'].get('error'))[:150]}.", small)
                if best == "rl1" and row["id"] in preferred:
                    p(preferred[row["id"]], small)
            else:
                p("У лучшей RL-модели такой ошибки в этом срезе не найдено.", small)
        output.parent.mkdir(parents=True, exist_ok=True)

        def footer(canvas, doc):
            canvas.setFont("Body", 8)
            canvas.setFillColor(colors.HexColor("#527086"))
            canvas.drawString(55, 31, "LLM Course · HW5")
            canvas.drawRightString(A4[0]-55, 31, str(doc.page))

        SimpleDocTemplate(str(output), pagesize=A4, leftMargin=55, rightMargin=55,
                          topMargin=46, bottomMargin=48, title="HW5 — RL для вызова инструментов").build(
            story, onFirstPage=footer, onLaterPages=footer
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, default=Path("report.pdf"))
    parser.add_argument("--font", type=Path, required=True, help="Path to a Cyrillic TrueType font")
    cli = parser.parse_args()
    build(cli.root, cli.output, cli.font)
