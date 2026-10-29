"""Embed the real loss plot after a headless run; training is not repeated."""
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import nbformat

from execute_notebook import LocalPythonNotebookClient


def main():
    root = Path(__file__).resolve().parent
    path = root / "sft_function_calling_hw.ipynb"
    notebook = nbformat.read(path, as_version=4)
    status = json.loads((root / "outputs/run_status.json").read_text())
    assert status["status"] == "completed"
    assert all(c.execution_count is not None for c in notebook.cells if c.cell_type == "code")
    assert not any(o.output_type == "error" for c in notebook.cells for o in c.get("outputs", []))
    plot = next(c for c in notebook.cells if c.cell_type == "code" and 'plt.savefig("outputs/loss.png"' in c.source)
    # Run only the plotting cell against the saved training history, in the same venv.
    render = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_code_cell('''
import json
from pathlib import Path
get_ipython().run_line_magic("matplotlib", "inline")
import matplotlib.pyplot as plt
import pandas as pd
training_state = json.loads(Path("outputs/training_state.json").read_text())
'''),
        nbformat.v4.new_code_cell(plot.source),
    ])
    LocalPythonNotebookClient(render, kernel_name="python3", timeout=120).execute(cwd=str(root))
    assert any("image/png" in o.get("data", {}) for o in render.cells[1].outputs)
    plot.outputs = render.cells[1].outputs
    first = next(c for c in notebook.cells if c.cell_type == "code")
    magic = 'get_ipython().run_line_magic("matplotlib", "inline")'
    if magic not in first.source:
        first.source = first.source.replace("import matplotlib.pyplot as plt", magic + "\nimport matplotlib.pyplot as plt")
    notebook.metadata["plot_export"] = {
        "rendered_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "outputs/training_state.json",
        "note": "Only loss plotting was re-executed with the inline backend; training and evaluation were not repeated.",
    }
    state = json.loads((root / "outputs/training_state.json").read_text())
    manifest = json.loads((root / "outputs/run_manifest.json").read_text())
    agents = json.loads((root / "outputs/agent_results.json").read_text())
    scores = list(csv.DictReader((root / "outputs/eval_results.csv").open()))
    table = "\n".join(
        f"| {s['model']} | {s['lang']} | {float(s['tool_set_accuracy']):.1%} | {float(s['exact_match']):.1%} |"
        for s in scores
    )
    eval_losses = [(r["epoch"], r["eval_loss"]) for r in state["log_history"] if "eval_loss" in r]
    losses = " → ".join(f"{loss:.5f}" for _, loss in eval_losses)
    validation_note = (
        "На третьей эпохе validation loss выше, чем на второй: дальнейшее снижение train loss "
        "уже не сопровождается улучшением validation loss. Это совместимо с началом переобучения. "
        "В дальнейшем стоит сохранять и выбирать checkpoint по validation loss. В этом запуске "
        "заранее заданные три эпохи завершены целиком; по результатам теста модель не перенастраивалась."
        if eval_losses[-1][1] > min(loss for _, loss in eval_losses) else
        "Финальный validation loss минимален среди проверенных эпох."
    )
    summary = f"""## Разбор выполненного запуска

Все {state['trainable_parameters']:,} параметров обучены за {state['training_seconds']:.1f} с,
{state['global_step']} шагов, три эпохи, DDP на двух A100. Максимум выделенной памяти PyTorch:
{state['peak_allocated_gib'][0]:.2f} и {state['peak_allocated_gib'][1]:.2f} GiB по GPU.

| Модель | Язык | Tool set accuracy | Exact match |
|---|---|---:|---:|
{table}

В целом exact match вырос с 26/60 до 56/60; выбор функций — с 45/60 до 58/60.
Validation loss по эпохам: {losses}.
{validation_note}

Четыре оставшиеся ошибки SFT: `en_03` — ложный отказ включить свет в гостиной;
`en_14` — 120 минут вместо 60; `en_38` — 25 градусов вместо 20+2=22;
`ru_04` — лишнее уточнение вместо установки допустимых 19 градусов.
Все ответы имели корректный формат; оставшиеся ошибки относятся к содержанию решений.

Все {len(agents)} сценариев агента завершились корректно: проверены результаты каждого вызова,
два вызова в одной реплике и последовательность температуры 21→23→25. При этом успех этих
сценариев не отменяет ошибку 20+2 в отдельном тестовом контексте.

В таблице десяти контрольных запросов видны исправления для `all lights`, гаража и отсутствующей
комнаты. Есть и регрессия: `two cartons of milk` стало `item="cartons of milk"` без `quantity=2`
(симулятор подставил бы 1). Улучшение средних метрик не означает улучшения каждого примера.

Три тестовых контекста совпадают с train, один — с validation. На остальных 56 контекстах SFT
получила exact match 52/56=92.86%, base — 23/56=41.07%. Это дополнительная проверка на точные
совпадения контекстов; она не доказывает независимость всех семантических шаблонов.
"""
    summary_id = "completed-run-analysis"
    existing = next((c for c in notebook.cells if c.get("id") == summary_id), None)
    if existing is None:
        notebook.cells.append(nbformat.v4.new_markdown_cell(summary.strip(), id=summary_id))
    else:
        existing.source = summary.strip()
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    print("Embedded loss plot from the completed run; no training/evaluation repeated.")


if __name__ == "__main__":
    main()
