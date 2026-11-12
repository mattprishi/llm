"""Check prepared labels against the pinned Qwen rendering and strict reward."""
import argparse
import hashlib
import json
from pathlib import Path

from prepare_data import MODEL, MODEL_REVISION
from rewards import score_completion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--tokenizer", default=MODEL)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, revision=MODEL_REVISION if args.tokenizer == MODEL else None,
    )
    seen = {}
    for split in ("train", "val"):
        ids, dialogues, prompts = set(), set(), set()
        n = 0
        with (args.data / (split + ".jsonl")).open() as stream:
            for line in stream:
                row = json.loads(line)
                row_id = row["id"]
                prompt = tokenizer.apply_chat_template(
                    row["messages"], tools=row["tools"], add_generation_prompt=True,
                    enable_thinking=False, tokenize=False,
                )
                assert prompt == row["prompt"], row_id
                assert len(tokenizer(prompt, add_special_tokens=False)["input_ids"]) == row["prompt_tokens"], row_id
                calls = row["expected"]
                reference = {"role": "assistant", "content": "" if calls else row["source_answer"]}
                if calls:
                    reference["tool_calls"] = [{"type": "function", "function": call} for call in calls]
                full = tokenizer.apply_chat_template(
                    [*row["messages"], reference], tools=row["tools"],
                    add_generation_prompt=False, enable_thinking=False, tokenize=False,
                )
                assert full.startswith(prompt), row_id
                score = score_completion(full[len(prompt):], calls, row["tools"])
                assert score.strict == score.shaped == 1.0, (row_id, score)
                fingerprint = hashlib.sha256(prompt.encode()).digest()
                assert row_id not in ids and fingerprint not in prompts, row_id
                ids.add(row_id)
                dialogues.add(row["dialogue_id"])
                prompts.add(fingerprint)
                n += 1
        assert n > 0
        seen[split] = dialogues, prompts
        print(f"{split}: {n} references passed", flush=True)
    assert not (seen["train"][0] & seen["val"][0] or seen["train"][1] & seen["val"][1])
    print("No cross-split overlap by dialogue ID or rendered prompt.")


if __name__ == "__main__":
    main()
