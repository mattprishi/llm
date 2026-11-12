"""Merge a HW5 LoRA adapter into the full, pinned Qwen3.5 checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from shutil import copyfile

from train_rl import BASE_MODEL, BASE_REVISION

BASE_PROCESSOR_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "merges.txt",
    "vocab.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapter", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if not (args.adapter / "adapter_config.json").is_file():
        parser.error(f"Adapter is missing: {args.adapter}")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText
    from transformers.utils.hub import cached_file

    # Copy the original tokenizer and processor assets byte for byte.
    processor_files = {}
    for filename in BASE_PROCESSOR_FILES:
        source = cached_file(
            BASE_MODEL, filename, revision=BASE_REVISION,
            _raise_exceptions_for_missing_entries=False,
        )
        if source is not None:
            processor_files[filename] = source
    if not {"tokenizer.json", "tokenizer_config.json"} <= processor_files.keys():
        raise RuntimeError("Pinned tokenizer assets are missing")
    torch.set_num_threads(min(8, torch.get_num_threads()))
    base = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL, revision=BASE_REVISION, dtype=torch.bfloat16
    )
    model = PeftModel.from_pretrained(base, str(args.adapter))
    merged = model.merge_and_unload(safe_merge=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = args.output.with_name(args.output.name + ".merging")
    staging.mkdir(exist_ok=False)
    merged.save_pretrained(staging, safe_serialization=True, max_shard_size="2GB")
    for filename, source in processor_files.items():
        copyfile(source, staging / filename)
    staging.rename(args.output)
    print(f"Merged checkpoint: {args.output}", flush=True)


if __name__ == "__main__":
    main()
