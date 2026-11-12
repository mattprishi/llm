"""Convert ToolACE dialogues to RL prompts with structured reference calls."""
import argparse
import ast
from collections import Counter
import hashlib
from io import StringIO
import json
import math
from pathlib import Path
import random
import re
import tokenize

DATASET = "Team-ACE/ToolACE"
DATA_REVISION = "6bda777c88d21e5a204703c1ee45597a8fa4f734"
MODEL = "Qwen/Qwen3.5-2B"
MODEL_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
SCHEMA_MARKERS = (
    "Here is a list of functions in JSON format that you can invoke:",
    "以下是可以调用的",
)
TYPE_MAP = {"dict": "object", "int": "integer", "float": "number",
            "bool": "boolean", "str": "string", "list": "array", "tuple": "array"}
SPECIAL_MARKUP = re.compile(r"<\|(?:im_start|im_end|endoftext)\|>")
KIND_ORDER = ("no_call", "single_call", "multi_call")


def extract_schema(system):
    for marker in SCHEMA_MARKERS:
        if marker in system:
            tail = system.split(marker, 1)[1]
            start = tail.find("[")
            if start >= 0:
                tools, _ = json.JSONDecoder().raw_decode(tail[start:])
                if isinstance(tools, list) and tools:
                    return tools
    raise ValueError("unsupported_schema_header")


def normalize_schema(schema):
    """Visit schema positions while leaving literal defaults and enum values intact."""
    result = dict(schema)
    if isinstance(result.get("type"), str):
        result["type"] = TYPE_MAP.get(result["type"], result["type"])
    if isinstance(result.get("properties"), dict):
        result["properties"] = {
            key: normalize_schema(value) if isinstance(value, dict) else value
            for key, value in result["properties"].items()
        }
    for key in ("items", "additionalProperties"):
        if isinstance(result.get(key), dict):
            result[key] = normalize_schema(result[key])
    return result


def convert_tools(schemas):
    name_map = {}
    tools = []
    names = {tool["name"] for tool in schemas}
    if len(names) != len(schemas) or any(not isinstance(name, str) for name in names):
        raise ValueError("ambiguous_tool_names")
    legal_names = {name for name in names if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name)}
    occupied = set(legal_names)
    for schema in schemas:
        original = schema["name"]
        if original in legal_names:
            name = original
        else:
            name = re.sub(r"[^A-Za-z0-9_-]+", "_", original).strip("_")[:64] or "tool"
            if name in occupied:
                suffix = hashlib.sha256(original.encode()).hexdigest()[:10]
                name = f"{name[:53]}_{suffix}"
            if name in occupied:
                raise ValueError("ambiguous_tool_names")
            occupied.add(name)
        name_map[original] = name
        params = normalize_schema(schema["parameters"])
        required = params.get("required") or schema.get("required") or []
        params["required"] = required
        params.setdefault("properties", {})
        if params.get("type") != "object" or not isinstance(required, list):
            raise ValueError("invalid_tool_schema")
        if not set(required) <= set(params["properties"]):
            raise ValueError("invalid_tool_schema")
        tools.append({"type": "function", "function": {
            "name": name, "description": schema.get("description") or "", "parameters": params,
        }})
    return tools, name_map


class JsonNames(ast.NodeTransformer):
    def visit_Name(self, node):
        literals = {"true": True, "false": False, "null": None}
        if node.id not in literals:
            raise ValueError("unknown_literal")
        return ast.copy_location(ast.Constant(literals[node.id]), node)


def read_value(text):
    tree = JsonNames().visit(ast.parse(text.strip(), mode="eval"))
    value = ast.literal_eval(tree)
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def consume_arguments(source):
    """Read (key=value, ...) with balanced brackets and quoted literals."""
    offsets = [0]
    for line in source.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    depth = 0
    segment_start = 1
    arguments = {}
    for tok in tokenize.generate_tokens(StringIO(source).readline):
        if tok.type != tokenize.OP:
            continue
        if tok.string in ("(", "[", "{"):
            depth += 1
            continue
        if tok.string in (")", "]", "}"):
            depth -= 1
        closes = depth == 0 and tok.string == ")"
        if not (closes or (depth == 1 and tok.string == ",")):
            continue
        segment_end = offsets[tok.start[0] - 1] + tok.start[1]
        segment = source[segment_start:segment_end].strip()
        if segment:
            key, equals, value = segment.partition("=")
            key = key.strip()
            if not equals or not key or key in arguments:
                raise ValueError("invalid_keyword_argument")
            arguments[key] = read_value(value)
        elif not closes:
            raise ValueError("empty_keyword_argument")
        if closes:
            return arguments, source[segment_end + 1:].strip()
        segment_start = segment_end + 1
    raise ValueError("unclosed_call")


def parse_calls(text, name_map):
    """Return None for plain text. Raise on malformed call syntax."""
    text = text.strip()
    if not text.startswith("["):
        return None
    if not text.endswith("]"):
        raise ValueError("unclosed_call_list")
    remaining = text[1:-1].strip()
    calls = []
    names = sorted(name_map, key=len, reverse=True)
    while remaining:
        for original in names:
            if remaining.startswith(original):
                suffix = remaining[len(original):].lstrip()
                if suffix.startswith("("):
                    args, remaining = consume_arguments(suffix)
                    calls.append({"name": name_map[original], "arguments": args})
                    break
        else:
            raise ValueError("unknown_call_name")
        if not remaining:
            break
        if not remaining.startswith(","):
            raise ValueError("invalid_call_separator")
        remaining = remaining[1:].lstrip()
    if not calls:
        raise ValueError("empty_call_list")
    return calls


def rewrite_tool_result(text, name_map):
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        return text
    if not isinstance(result, list):
        return text
    for entry in result:
        if isinstance(entry, dict) and isinstance(entry.get("name"), str) and entry["name"] in name_map:
            entry["name"] = name_map[entry["name"]]
    return json.dumps(result, ensure_ascii=False, separators=(",", ":"))


def references_match_schema(calls, tools_by_name):
    for call in calls:
        params = tools_by_name[call["name"]]["parameters"]
        keys = set(call["arguments"])
        if not set(params["required"]) <= keys <= set(params["properties"]):
            return False
    return True


def expand_dialogue(dialogue, dialogue_id, stats):
    try:
        tools, name_map = convert_tools(extract_schema(dialogue["system"]))
    except (ValueError, TypeError, KeyError) as exc:
        stats[str(exc) if isinstance(exc, ValueError) else "invalid_tool_schema"] += 1
        return

    history = []
    turns = dialogue["conversations"]
    tools_by_name = {tool["function"]["name"]: tool["function"] for tool in tools}
    for turn_index, turn in enumerate(turns):
        role, content = turn["from"], turn["value"].strip()
        if SPECIAL_MARKUP.search(content):
            stats["control_markup"] += 1
            break
        if role == "user":
            history.append({"role": role, "content": content})
        elif role == "tool":
            history.append({"role": role, "content": rewrite_tool_result(content, name_map)})
        elif role == "assistant" and history:
            next_role = turns[turn_index + 1]["from"] if turn_index + 1 < len(turns) else None
            try:
                calls = parse_calls(content, name_map)
            except (ValueError, SyntaxError, tokenize.TokenError, TypeError):
                stats["malformed_call"] += 1
                break
            if calls is None and next_role == "tool":
                stats["unparsed_call_before_tool"] += 1
                break
            if calls and not references_match_schema(calls, tools_by_name):
                stats["reference_schema_conflict"] += 1
                break
            if calls or (history[-1]["role"] == "user" and content):
                kind = "no_call" if not calls else "single_call" if len(calls) == 1 else "multi_call"
                yield {
                    "id": f"{dialogue_id}:{turn_index}",
                    "dialogue_id": dialogue_id,
                    "messages": list(history),
                    "tools": tools,
                    "expected": calls or [],
                    "kind": kind,
                    "source_answer": content,
                }
            elif history[-1]["role"] == "tool":
                stats["post_tool_text_excluded"] += 1
            else:
                stats["empty_or_unsupported_turn"] += 1
            message = {"role": role, "content": "" if calls else content}
            if calls:
                message["tool_calls"] = [{"type": "function", "function": call} for call in calls]
            history.append(message)
        else:
            stats["unsupported_role_or_order"] += 1
            break


def dialogue_group(dialogue):
    # A single-turn record can also appear as the first turn of a longer dialogue.
    first_user = next((m["value"] for m in dialogue["conversations"] if m["from"] == "user"), "")
    normalized = " ".join(first_user.casefold().split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def stratified_validation(rows, size, rng):
    """Equal coverage for the three validation strata, with spare capacity refilled."""
    strata = {kind: [row for row in rows if row["kind"] == kind] for kind in KIND_ORDER}
    target = min(size, len(rows))
    if not all(strata.values()):
        raise ValueError("Held-out data must contain each of the three reference types")
    result, spare = [], []
    for index, kind in enumerate(KIND_ORDER):
        group = strata[kind]
        rng.shuffle(group)
        quota = target // len(KIND_ORDER) + (index < target % len(KIND_ORDER))
        result.extend(group[:quota])
        spare.extend(group[quota:])
    rng.shuffle(spare)
    return result + spare[:target - len(result)]


def summarize_lengths(values):
    values = sorted(values)
    if not values:
        return {}
    return {
        "min": values[0],
        "p50": values[math.ceil(0.50 * len(values)) - 1],
        "p95": values[math.ceil(0.95 * len(values)) - 1],
        "max": values[-1],
    }


def summarize(rows):
    return {
        "rows": len(rows),
        "dialogues": len({row["dialogue_id"] for row in rows}),
        "kinds": dict(Counter(row["kind"] for row in rows)),
        "prompt_tokens": summarize_lengths([row["prompt_tokens"] for row in rows]),
        "source_answer_tokens": summarize_lengths([row["source_answer_tokens"] for row in rows]),
        "available_tools": summarize_lengths([len(row["tools"]) for row in rows]),
        "history_with_tool_response": sum(
            any(message["role"] == "tool" for message in row["messages"]) for row in rows
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Local copy of the pinned ToolACE data.json")
    parser.add_argument("--tokenizer", default=MODEL, help="Local copy of the pinned tokenizer or model ID")
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--max-prompt-tokens", type=int, default=1536)
    parser.add_argument("--val-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.val_size < 3 or args.max_prompt_tokens < 1:
        parser.error("--val-size must be at least 3 and --max-prompt-tokens must be positive")

    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    source = args.source or Path(hf_hub_download(
        DATASET, "data.json", repo_type="dataset", revision=DATA_REVISION,
    ))
    source_bytes = source.read_bytes()
    dialogues = json.loads(source_bytes)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, revision=MODEL_REVISION if args.tokenizer == MODEL else None,
    )
    rng = random.Random(args.seed)
    groups = sorted({dialogue_group(dialogue) for dialogue in dialogues})
    rng.shuffle(groups)
    val_groups = set(groups[:max(1, len(groups) // 10)])
    splits = {"train": [], "val": []}
    stats = Counter()
    lengths_before_filter = []
    unique_prompts = {"train": set(), "val": set()}
    for index, dialogue in enumerate(dialogues):
        split = "val" if dialogue_group(dialogue) in val_groups else "train"
        for row in expand_dialogue(dialogue, f"toolace:{index}", stats):
            prompt = tokenizer.apply_chat_template(
                row["messages"], tools=row["tools"], tokenize=False,
                add_generation_prompt=True, enable_thinking=False,
            )
            if prompt in unique_prompts[split]:
                stats["duplicate_prompt"] += 1
                continue
            unique_prompts[split].add(prompt)
            tokens = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
            lengths_before_filter.append(tokens)
            if tokens > args.max_prompt_tokens:
                stats["long_prompt"] += 1
                continue
            row["prompt"] = prompt
            row["prompt_tokens"] = tokens
            row["source_answer_tokens"] = len(tokenizer(
                row["source_answer"], add_special_tokens=False,
            )["input_ids"])
            splits[split].append(row)
    assert not unique_prompts["train"] & unique_prompts["val"], "Cross-split prompt leakage"
    splits["val"] = stratified_validation(splits["val"], args.val_size, rng)
    rng.shuffle(splits["train"])
    assert splits["train"] and splits["val"], "Empty split"
    assert not (
        {row["dialogue_id"] for row in splits["train"]}
        & {row["dialogue_id"] for row in splits["val"]}
    ), "Cross-split dialogue leakage"

    args.output.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        write_jsonl(args.output / f"{split}.jsonl", rows)
    examples = [
        next(row for row in splits["val"] if row["kind"] == kind) for kind in KIND_ORDER
    ]
    history_example = next((
        row for row in splits["train"] if any(msg["role"] == "tool" for msg in row["messages"])
    ), None)
    if history_example:
        examples.append(history_example)
    write_jsonl(args.output / "examples.jsonl", examples)
    eda = {
        "dataset": DATASET,
        "revision": DATA_REVISION,
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "tokenizer": MODEL,
        "tokenizer_revision": MODEL_REVISION,
        "seed": args.seed,
        "source_dialogues": len(dialogues),
        "unique_first_user_groups": len(groups),
        "reserved_validation_groups": len(val_groups),
        "max_prompt_tokens": args.max_prompt_tokens,
        "prompt_tokens_before_length_filter": summarize_lengths(lengths_before_filter),
        "filter_counts": dict(stats),
        "splits": {name: summarize(rows) for name, rows in splits.items()},
    }
    report = json.dumps(eda, ensure_ascii=False, indent=2)
    (args.output / "eda.json").write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
