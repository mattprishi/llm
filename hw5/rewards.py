"""Verifiable rewards for Qwen's tool-call format on the ToolACE split."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

_CALL = re.compile(r"<tool_call>\s*<function=([^<>\n]+)>\s*(.*?)</function>\s*</tool_call>", re.S)
_PARAMETER = re.compile(r"<parameter=([^<>\n]+)>(.*?)</parameter>\s*", re.S)
_TOOL_MARKUP = re.compile(r"<\s*/?\s*(?:tool|function|parameter)", re.I)
_STOP = re.compile(r"(?:\s*<\|(?:im_end|endoftext)\|>)+\s*$")
_THINK = re.compile(r"^\s*<think>\s*</think>\s*")


@dataclass
class Score:
    strict: float
    shaped: float
    reason: str
    calls: list[dict[str, Any]]


def _json_type(value: Any) -> str:
    return {
        str: "string",
        bool: "boolean",
        int: "integer",
        float: "number",
        list: "array",
        dict: "object",
        type(None): "null",
    }[type(value)]


def _schema_types(schema: Any) -> set[str]:
    if not isinstance(schema, dict):
        return {"string"}
    declared = schema.get("type")
    types = {declared} if isinstance(declared, str) else set()
    if isinstance(declared, list):
        types.update(value for value in declared if isinstance(value, str))
    enum = schema.get("enum")
    if isinstance(enum, list):
        types.update(_json_type(value) for value in enum)
    for key in ("oneOf", "anyOf", "allOf"):
        children = schema.get(key)
        if isinstance(children, list):
            for child in children:
                types.update(_schema_types(child))
    return types or {"string"}


def _coerce(raw: str, schema: Any) -> Any:
    # Qwen surrounds values with line breaks. Preserve other whitespace in strings.
    raw = raw.removeprefix("\n").removesuffix("\n")
    types = _schema_types(schema)
    stripped = raw.strip()
    if "null" in types and stripped.lower() == "null":
        return None
    if types & {"integer", "int"}:
        try:
            return int(stripped)
        except ValueError:
            pass
    if types & {"number", "float"}:
        try:
            number = float(stripped)
            if math.isfinite(number):
                return number
        except ValueError:
            pass
    if types & {"boolean", "bool"}:
        if stripped.lower() in {"true", "1"}:
            return True
        if stripped.lower() in {"false", "0"}:
            return False
    if types & {"array", "list", "object", "dict", "tuple"}:
        try:
            value = json.loads(stripped)
            if isinstance(value, (list, dict)):
                return value
        except (ValueError, RecursionError):
            pass
    return raw


def parse_calls(completion: str, tools: list[dict]) -> list[dict]:
    """Read all fully enclosed calls. Raise on orphaned or duplicate markup."""
    by_name = {tool["function"]["name"]: tool["function"] for tool in tools}
    calls = []
    position = 0
    for match in _CALL.finditer(completion):
        if _TOOL_MARKUP.search(completion[position : match.start()]):
            raise ValueError("unexpected tool markup")
        name, body = match.groups()
        if name != name.strip() or not name:
            raise ValueError("invalid function name")
        schema = by_name.get(name, {}).get("parameters") or {}
        properties = schema.get("properties", {})
        arguments = {}
        cursor = 0
        while cursor < len(body):
            parameter = _PARAMETER.match(body, cursor)
            if parameter is None:
                raise ValueError("invalid parameter markup")
            key, raw = parameter.groups()
            if key in arguments or key != key.strip() or _TOOL_MARKUP.search(raw):
                raise ValueError("invalid or duplicate parameter")
            arguments[key] = _coerce(raw, properties.get(key, {}))
            cursor = parameter.end()
        calls.append({"name": name, "arguments": arguments})
        position = match.end()
    if _TOOL_MARKUP.search(completion[position:]):
        raise ValueError("unexpected tool markup")
    return calls


def _equal(left: Any, right: Any) -> bool:
    # JSON booleans and numbers are distinct despite Python's True == 1.
    if type(left) in (int, float) and type(right) in (int, float):
        return left == right
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_equal(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right))
    return left == right


def _call_similarity(prediction: dict, target: dict) -> float:
    if prediction["name"] != target["name"]:
        return 0.0
    predicted = prediction["arguments"]
    expected = target["arguments"]
    keys = predicted.keys() | expected.keys()
    if not keys:
        return 1.0
    correct = sum(
        key in predicted and key in expected and _equal(predicted[key], expected[key])
        for key in keys
    )
    return 0.25 + 0.75 * correct / len(keys)


def score_completion(
    completion: str, expected: list[dict] | str, tools: list[dict] | str, truncated: bool = False
) -> Score:
    """Score one sampled answer. ToolACE references are lists of named calls."""
    if truncated:
        return Score(0.0, 0.0, "truncated", [])
    completion = _THINK.sub("", _STOP.sub("", completion)).strip()
    if not completion:
        return Score(0.0, 0.0, "empty", [])
    if re.search(r"</?think\b|<\|im_start\|>|<\|im_end\|>", completion):
        return Score(0.0, 0.0, "unexpected_turn_or_thinking", [])
    if isinstance(expected, str):
        expected = json.loads(expected)
    if isinstance(tools, str):
        tools = json.loads(tools)
    try:
        calls = parse_calls(completion, tools)
    except ValueError:
        return Score(0.0, 0.0, "malformed", [])
    if not expected:
        if not calls:
            return Score(1.0, 1.0, "no_call_correct", calls)
        return Score(0.0, 0.0, "unwanted_call", calls)
    if not calls:
        return Score(0.0, 0.0, "missing_call", calls)

    scores = [[_call_similarity(call, target) for target in expected] for call in calls]
    if len(calls) == len(expected) == 1:
        matched = scores[0][0]
    else:
        # Optimal one-to-one matching handles order and repeated function names.
        from scipy.optimize import linear_sum_assignment

        rows, columns = linear_sum_assignment(scores, maximize=True)
        matched = sum(scores[i][j] for i, j in zip(rows, columns))
    denominator = max(len(calls), len(expected))
    if matched == denominator:
        return Score(1.0, 1.0, "exact", calls)
    return Score(0.0, 0.9 * matched / denominator, "non_exact", calls)
