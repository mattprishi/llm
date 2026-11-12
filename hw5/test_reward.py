"""Regression checks for tool syntax, typed arguments and parallel matching."""

import json
import unittest

from rewards import score_completion


TOOLS = [
    {"type": "function", "function": {
        "name": "weather",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string"},
            "days": {"type": "integer"},
            "metric": {"type": "boolean"},
            "options": {"type": "object"},
        }},
    }},
    {"type": "function", "function": {"name": "clock"}},
]


def render(name, /, **arguments):
    parts = ["<tool_call>", f"<function={name}>"]
    for key, value in arguments.items():
        if not isinstance(value, str):
            value = json.dumps(value, ensure_ascii=False)
        parts.extend([f"<parameter={key}>", value, "</parameter>"])
    return "\n".join(parts + ["</function>", "</tool_call>"])


def target(name, /, **arguments):
    return {"name": name, "arguments": arguments}


class RewardTests(unittest.TestCase):
    def test_exact_argument_types_and_json_serialized_metadata(self):
        args = {"city": "001", "days": 1, "metric": True, "options": {"tags": ["α", 1]}}
        result = score_completion(
            render("weather", **args) + "<|im_end|>",
            json.dumps([target("weather", **args)]), json.dumps(TOOLS),
        )
        self.assertEqual((result.strict, result.shaped), (1.0, 1.0))
        self.assertEqual(result.calls[0]["arguments"], args)

    def test_argument_match_and_wrong_type(self):
        expected = [target("weather", city="Moscow", days=3)]
        result = score_completion(render("weather", city="Moscow", days=2), expected, TOOLS)
        self.assertEqual(result.strict, 0)
        self.assertAlmostEqual(result.shaped, 0.5625)
        result = score_completion(render("weather", options={"v": True}),
                                  [target("weather", options={"v": 1})], TOOLS)
        self.assertEqual(result.strict, 0)
        self.assertAlmostEqual(result.shaped, 0.225)

    def test_parallel_multiset_and_cardinality(self):
        expected = [target("weather", city="Moscow"), target("weather", city="Oslo")]
        a = render("weather", city="Moscow")
        b = render("weather", city="Oslo")
        self.assertEqual(score_completion(b + "\n" + a, expected, TOOLS).strict, 1)
        self.assertAlmostEqual(score_completion(a, expected, TOOLS).shaped, 0.45)
        self.assertAlmostEqual(score_completion(a + b + a, expected, TOOLS).shaped, 0.6)
        self.assertEqual(score_completion(a + a, expected, TOOLS).strict, 0)

    def test_no_call_vs_malformed_and_truncated(self):
        self.assertEqual(score_completion("Please specify the city.", [], TOOLS).strict, 1)
        self.assertEqual(score_completion(render("clock"), [], TOOLS).reason, "unwanted_call")
        self.assertEqual(score_completion("   <|im_end|>", [], TOOLS).reason, "empty")
        self.assertEqual(score_completion("Please specify", [], TOOLS, truncated=True).strict, 0)
        self.assertEqual(score_completion("Sure, <tool_cal", [], TOOLS).reason, "malformed")
        self.assertEqual(score_completion(render("clock") + "</tool_call>", [], TOOLS).reason, "malformed")
        self.assertEqual(
            score_completion(
                "<tool_call><function=weather><parameter=city>Oslo</parameter>"
                "<parameter=city>Moscow</parameter></function></tool_call>",
                [], TOOLS,
            ).reason,
            "malformed",
        )


if __name__ == "__main__":
    unittest.main()
