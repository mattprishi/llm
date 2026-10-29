"""Fill the updated instructor template without changing the supplied source files."""
import textwrap
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent
notebook = nbformat.read(ROOT / "_docs/updated/sft_function_calling_hw_student-2.ipynb", as_version=4)


def code(index, source):
    notebook.cells[index].source = textwrap.dedent(source).strip()


code(2, '''
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

get_ipython().run_line_magic("matplotlib", "inline")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from datasets import Dataset
from IPython.display import display
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer

from chat_ui import launch_chat
from smart_home import BASELINE_PROMPTS, SMART_HOME_TOOLS, SYSTEM_PROMPT, SmartHome

SEED = 42
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = Path("outputs/full_sft")
MAX_LENGTH = 2048
set_seed(SEED)
torch.set_num_threads(8)
assert torch.cuda.is_available(), "Нужен GPU kernel"
assert torch.cuda.device_count() == 2, "Запуск: CUDA_VISIBLE_DEVICES=0,1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
pd.set_option("display.max_colwidth", 200)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.bfloat16, attn_implementation="sdpa"
).to("cuda:0")
provenance = {
    "started_at_utc": datetime.now(timezone.utc).isoformat(),
    "hostname": platform.node(), "seed": SEED,
    "model": MODEL_NAME, "model_revision": model.config._commit_hash,
    "python": platform.python_version(), "cuda_runtime": torch.version.cuda,
    "packages": {p: importlib.metadata.version(p) for p in
                 ["torch", "transformers", "trl", "datasets", "accelerate"]},
    "gpus": [torch.cuda.get_device_name(i) for i in range(2)],
    "data_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(Path("data").glob("*.jsonl"))},
}
print(json.dumps(provenance, ensure_ascii=False, indent=2))
''')

code(6, '''
@torch.inference_mode()
def generate(model, tokenizer, messages, tools, max_new_tokens=256):
    model.eval()
    inputs = tokenizer.apply_chat_template(
        messages, tools=tools, add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt",
    ).to(next(model.parameters()).device)
    output = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
        pad_token_id=tokenizer.pad_token_id,
    )
    new_tokens = output[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def answer(model, tokenizer, prompt):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    return generate(model, tokenizer, messages, SMART_HOME_TOOLS)
''')

notebook.cells[15].source += '''

Результат функции — наблюдение из внешней среды, которое модель получает, а не генерирует.
В нативном шаблоне Qwen сообщения `tool` упакованы в `user` с тегами `<tool_response>`.
Роль `tool` при этом остаётся в исходной структуре диалога. На следующем шаге ассистент
использует результат как контекст: например, прочитанную температуру — для расчёта новой.

Если обучать и на содержимом `<tool_response>`, градиенты будут тратиться на предсказание
внешних наблюдений, хотя на инференсе их сообщает симулятор. Это создаёт лишнюю задачу и
может поощрять имитацию результатов. Само по себе включение этих токенов в loss не означает,
что агент обязательно обойдёт симулятор: исполнение задаёт программа. Правильная граница
супервизии здесь — только решения и реплики assistant.

| Тип | Навык | Что теряется без него |
|---|---|---|
| `single_call` | Выбор одной функции и заполнение аргументов | Базовое соответствие запроса схеме инструмента |
| `parallel_calls` | Несколько вызовов в одной реплике | Разложение составной команды, в том числе по всем комнатам |
| `multi_turn` | Использование результата функции и истории | Зависимые действия, относительные изменения, продолжение диалога |
| `no_call` | Обычный ответ, уточнение или отказ без инструмента | Умение воздержаться от неподдерживаемого или преждевременного действия |

Эти навыки частично пересекаются; удаление типа не доказывает полную потерю навыка,
но убирает прямую супервизию соответствующего поведения.
'''
notebook.cells[15].source = notebook.cells[15].source.replace("\n…\n", "\n")

notebook.cells[10].source += '''

def context(messages):
    return messages[1:] if messages and messages[0]["role"] == "system" else messages

def message_key(messages):
    return json.dumps(messages, sort_keys=True, ensure_ascii=False, separators=(",", ":"))

data_overlap = []
audit_cases = read_jsonl("data/test_cases.jsonl")
for split, rows in [("train", train_rows), ("validation", validation_rows)]:
    prefixes = {}
    for row in rows:
        ms = context(row["messages"])
        for stop, message in enumerate(ms, 1):
            if message["role"] in ("user", "tool"):
                prefixes.setdefault(message_key(ms[:stop]), []).append(row["id"])
    for case in audit_cases:
        matches = prefixes.get(message_key(context(case["messages"])), [])
        if matches:
            data_overlap.append({"split": split, "test_id": case["id"], "source_ids": matches})
scenario_overlap = {r["scenario_id"] for r in train_rows} & {r["scenario_id"] for r in validation_rows}
assert not scenario_overlap
print("Пересечение train/validation по scenario_id:", len(scenario_overlap))
print("Совпадение контекстов официального теста с train/validation:")
display(pd.DataFrame(data_overlap))
provenance["test_context_overlap"] = data_overlap
'''

code(19, r'''
# Разделяем обычные assistant-реплики и assistant с tool_calls.
# Генерационные теги не добавляют символов в отрендеренный диалог.
old_plain = r"""    {%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}"""
new_plain = r"""    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" and not message.tool_calls %}
        {{- '<|im_start|>' + message.role + '\n' }}{% generation %}{{- message.content + '<|im_end|>' + '\n' }}{% endgeneration %}
    {%- elif message.role == "assistant" %}"""
assert NATIVE_TEMPLATE.count(old_plain) == 1, "Изменился нативный шаблон: требуется проверка"
TRAIN_TEMPLATE = NATIVE_TEMPLATE.replace(old_plain, new_plain)
tool_start = "{{- '<|im_start|>' + message.role }}"
tool_end = r"{{- '<|im_end|>\n' }}" + '\n    {%- elif message.role == "tool" %}'
assert TRAIN_TEMPLATE.count(tool_start) == 1 and TRAIN_TEMPLATE.count(tool_end) == 1
TRAIN_TEMPLATE = TRAIN_TEMPLATE.replace(tool_start, tool_start + "{% generation %}")
TRAIN_TEMPLATE = TRAIN_TEMPLATE.replace(
    tool_end, r"{{- '<|im_end|>\n' }}{% endgeneration %}" + '\n    {%- elif message.role == "tool" %}'
)
tokenizer.chat_template = TRAIN_TEMPLATE
''')

notebook.cells[20].source += '''

# Проверяем все диалоги, включая обычный текст и несколько tool-ответов подряд.
lengths = []
supervised = []
for row in train_rows + validation_rows:
    native = tokenizer.apply_chat_template(row["messages"], tools=row["tools"], tokenize=False,
                                           chat_template=NATIVE_TEMPLATE)
    patched = tokenizer.apply_chat_template(row["messages"], tools=row["tools"], tokenize=False)
    assert native == patched, row["id"]
    enc = tokenizer.apply_chat_template(row["messages"], tools=row["tools"],
                                       return_dict=True, return_assistant_tokens_mask=True)
    trained_text = tokenizer.decode([t for t, flag in zip(enc["input_ids"], enc["assistant_masks"]) if flag])
    assert sum(enc["assistant_masks"]) > 0, row["id"]
    assert "<tool_response>" not in trained_text and "<tools>" not in trained_text, row["id"]
    assert "<|im_start|>" not in trained_text, row["id"]
    n_assistant = sum(m["role"] == "assistant" for m in row["messages"])
    assert trained_text.count("<|im_end|>") == n_assistant, row["id"]
    lengths.append(len(enc["input_ids"]))
    supervised.append(sum(enc["assistant_masks"]))
assert max(lengths) <= MAX_LENGTH, "Нельзя молча обрезать диалоги"
print(f"Проверены {len(lengths)} диалога: native == train; max_length={max(lengths)}")
print(pd.Series(lengths).describe())
print(f"Доля обучаемых токенов: {sum(supervised) / sum(lengths):.2%}")
'''

code(24, '''
# Обучение на двух GPU запускается в train_distributed.py.
worker_path = Path("train_distributed.py")
assert worker_path.is_file(), "Запускайте notebook из папки hw3"
tokenizer.save_pretrained("outputs/tokenizer_for_training")
training_config = {
    "model_name": MODEL_NAME, "model_revision": provenance["model_revision"],
    "output_dir": str(OUTPUT_DIR), "seed": SEED, "max_length": MAX_LENGTH,
}
Path("outputs/training_config.json").write_text(json.dumps(training_config, indent=2))
del model
gc.collect()
torch.cuda.empty_cache()
subprocess.run([
    sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
    str(worker_path), "--config", "outputs/training_config.json",
], check=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0,1", "OMP_NUM_THREADS": "8"})
training_state = json.loads((OUTPUT_DIR / "training_state.json").read_text())
assert training_state["world_size"] == 2 and training_state["status"] == "completed"
model = AutoModelForCausalLM.from_pretrained(
    OUTPUT_DIR, dtype=torch.bfloat16, attn_implementation="sdpa"
).to("cuda:0").eval()
saved_tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
assert saved_tokenizer.chat_template == tokenizer.chat_template
provenance.update({k: training_state[k] for k in [
    "trainable_parameters", "training_seconds", "training_metrics", "peak_allocated_gib",
]})
provenance["config"] = {
    "epochs": 3, "learning_rate": 5e-5, "global_batch_size": 8,
    "assistant_only_loss": True, "max_length": MAX_LENGTH, "parallelism": "DistributedDataParallel",
}
Path("outputs/run_manifest.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2))
Path("outputs/training_state.json").write_text(json.dumps(training_state, ensure_ascii=False, indent=2))
print(f"Обучение: {training_state['training_seconds']:.1f} секунд; "
      f"peak GiB: {training_state['peak_allocated_gib']}")
''')

notebook.cells[25].source = notebook.cells[25].source.replace(
    "pd.DataFrame(trainer.state.log_history)", 'pd.DataFrame(training_state["log_history"])'
)

notebook.cells[25].source = notebook.cells[25].source.replace(
    "plt.show()", 'plt.tight_layout()\nplt.savefig("outputs/loss.png", dpi=160)\nplt.show()\nhistory.to_csv("outputs/training_history.csv", index=False)'
)

# Keep the instructor's metric definitions; reject unmatched closing tags too.
notebook.cells[27].source = notebook.cells[27].source.replace(
    'if len(blocks) != text.count("<tool_call>"):',
    'if len(blocks) != text.count("<tool_call>") or len(blocks) != text.count("</tool_call>"):'
)
notebook.cells[27].source = notebook.cells[27].source.replace(
    '    try:\n        calls = [json.loads(block) for block in blocks]',
    '    if re.search(r"</?tool_call\\b", TOOL_CALL_PATTERN.sub("", text)):\n        return None\n    try:\n        calls = [json.loads(block) for block in blocks]'
)

code(28, '''
gc.collect()
torch.cuda.empty_cache()
model.eval()
cases = read_jsonl("data/test_cases.jsonl")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, revision=provenance["model_revision"], dtype=torch.bfloat16, attn_implementation="sdpa"
).to("cuda:1")
base_results = evaluate(base_model, tokenizer, cases)
sft_results = evaluate(model, tokenizer, cases)
scores = pd.concat([
    summarize(base_results).assign(model="base"),
    summarize(sft_results).assign(model="sft"),
]).reset_index()[["model", "lang", "tool_set_accuracy", "exact_match"]]
all_predictions = pd.concat([base_results.assign(model="base"), sft_results.assign(model="sft")])
all_predictions.to_csv("outputs/eval_predictions.csv", index=False)
scores.to_csv("outputs/eval_results.csv", index=False)
overall = all_predictions.groupby("model").agg(
    n=("id", "size"), tool_set_accuracy=("tools_ok", "mean"), exact_match=("args_ok", "mean")
)
display(scores)
display(overall)
overlap_ids = {x["test_id"] for x in data_overlap}
nonoverlap = all_predictions.loc[~all_predictions["id"].isin(overlap_ids)].groupby("model").agg(
    n=("id", "size"), tool_set_accuracy=("tools_ok", "mean"), exact_match=("args_ok", "mean")
)
nonoverlap.to_csv("outputs/eval_nonoverlap.csv")
print("Дополнительно: 56 случаев без точного совпадения контекста с train/validation:")
display(nonoverlap)
print("Ошибки SFT:")
display(sft_results.loc[~sft_results["args_ok"], ["id", "output", "tools_ok", "args_ok"]])
del base_model
gc.collect()
torch.cuda.empty_cache()
''')
notebook.cells[30].source += '\ncomparison.to_csv("outputs/comparison.csv", index=False)'

code(32, '''
def run_agent(model, tokenizer, home, messages, max_steps=4):
    for _ in range(max_steps):
        text = generate(model, tokenizer, messages, SMART_HOME_TOOLS)
        calls = parse_tool_calls(text)
        if calls is None:
            # Битый ответ сохраняем для диагностики, ничего из него не исполняем.
            messages.append({"role": "assistant", "content": text})
            raise ValueError("Невалидный формат tool_call")
        if not calls:
            messages.append({"role": "assistant", "content": text})
            return {"status": "completed"}
        messages.append({
            "role": "assistant",
            "content": TOOL_CALL_PATTERN.sub("", text).strip(),
            "tool_calls": [{"type": "function", "function": call} for call in calls],
        })
        for call in calls:
            result = home.call(call["name"], call["arguments"])
            messages.append({"role": "tool", "name": call["name"],
                             "content": json.dumps(result, ensure_ascii=False)})
    return {"status": "max_steps_reached"}
''')

code(33, '''
scenarios = [
    ["Turn on the kitchen light."],
    ["Turn on the kitchen light and set it to 23 degrees."],
    ["Make it two degrees warmer in the bedroom.", "Two more degrees."],
    ["Thanks!"],
    ["Open the garage."],
]
agent_results = []
for number, prompts in enumerate(scenarios, 1):
    home = SmartHome()
    initial = home.snapshot()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    statuses = []
    turn_snapshots = []
    for prompt in prompts:
        messages.append({"role": "user", "content": prompt})
        statuses.append(run_agent(model, tokenizer, home, messages))
        turn_snapshots.append(home.snapshot())
    calls = [c["function"] for m in messages for c in m.get("tool_calls", [])]
    results = [json.loads(m["content"]) for m in messages if m["role"] == "tool"]
    snapshot = home.snapshot()
    completed = all(s["status"] == "completed" for s in statuses)
    mechanical_ok = len(calls) == len(results) and not any("error" in x for x in results) and completed
    if number == 1:
        goal_ok = snapshot["rooms"]["kitchen"]["light"] == "on"
    elif number == 2:
        goal_ok = snapshot["rooms"]["kitchen"] == {"light": "on", "temperature": 23}
        goal_ok = goal_ok and any(len(m.get("tool_calls", [])) == 2 for m in messages)
    elif number == 3:
        goal_ok = [s["rooms"]["bedroom"]["temperature"] for s in turn_snapshots] == [23, 25]
        goal_ok = goal_ok and bool(calls) and calls[0]["name"] == "get_room_state"
    else:
        goal_ok = not calls and snapshot == initial
    agent_results.append({"scenario": number, "prompts": prompts, "statuses": statuses,
                          "mechanical_ok": mechanical_ok, "goal_ok": goal_ok,
                          "messages": messages, "snapshot": snapshot, "turn_snapshots": turn_snapshots})
    print("=" * 80)
    show_dialog(messages[1:])
    print(f"дом: {json.dumps(snapshot, ensure_ascii=False)}")
    print(f"Цикл: {mechanical_ok}; цель сценария: {goal_ok}; statuses={statuses}")
Path("outputs/agent_results.json").write_text(json.dumps(agent_results, ensure_ascii=False, indent=2))
display(pd.DataFrame([{k: r[k] for k in ["scenario", "mechanical_ok", "goal_ok"]} for r in agent_results]))
''')

notebook.cells.extend([
    nbformat.v4.new_markdown_cell('''## Интерпретация результатов

Loss — средняя отрицательная log-вероятность правильного следующего токена на размеченных
assistant-позициях. Его снижение показывает улучшение предсказания этих реплик, но не доказывает
успех выполнения команд. Поэтому отдельно измеряются выбор функций и точность аргументов,
а затем проверяется выполнение в симуляторе. Train loss усредняется по обучающим шагам,
validation loss считается на конце эпохи: точки нельзя считать полностью синхронными измерениями.

`tool_set_accuracy` сравнивает множества имён; число повторных вызовов и аргументы в ней не учитываются.
`exact_match` сравнивает вызовы с аргументами, не учитывая порядок и подставляя `quantity=1`.
Проверка тест-сета оценивает следующий ответ на предоставленную историю, а не весь агентный цикл.
Пять сценариев агента — отдельная небольшая проверка исполнения и состояния симулятора.

Параметры выбраны до оценки теста; тест не используется для обучения, выбора эпохи или подбора
гиперпараметров. Один seed и 60 запросов дают учебную оценку на этих данных, но не доказывают
надёжность на произвольных формулировках. Ошибки сохраняются в `eval_predictions.csv`.

В предоставленных файлах 3 тестовых контекста точно совпадают с префиксами обучающих диалогов
(`en_17`, `en_22`, `ru_13`), ещё 1 — с валидационным (`en_27`). Train и validation по `scenario_id`
не пересекаются. Официальный тест сохранён целиком; отдельно показаны метрики на остальных 56
контекстах. Это исключает найденные точные совпадения, но само по себе не устанавливает
независимость всех формулировок или семантических шаблонов.
'''),
    nbformat.v4.new_code_cell('''
print("Итог по всем 60 случаям:")
display(overall)
print(f"Агент: цели {sum(r['goal_ok'] for r in agent_results)}/5; "
      f"механика {sum(r['mechanical_ok'] for r in agent_results)}/5")
provenance["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
provenance["overall_metrics"] = overall.reset_index().to_dict(orient="records")
provenance["agent_goal_passes"] = sum(r["goal_ok"] for r in agent_results)
Path("outputs/run_manifest.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2))
'''.strip()),
])
notebook.metadata.kernelspec = {"display_name": "Python (HW3)", "language": "python", "name": "python3"}
for cell in notebook.cells:
    if cell.cell_type == "code":
        cell.outputs = []
        cell.execution_count = None
        assert "NotImplementedError" not in cell.source
        compile(cell.source, "<cell>", "exec")
nbformat.validate(notebook)
target = ROOT / "sft_function_calling_hw.ipynb"
nbformat.write(notebook, target)
print(f"Created {target}: {len(notebook.cells)} cells")
