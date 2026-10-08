#!/usr/bin/env python3
"""Print the two fastest successful screening configurations."""

import json
from pathlib import Path


rows = []
for path in Path("results").glob("screen_*/summary.json"):
    summary = json.loads(path.read_text())
    rows.append((summary["tokens_per_second"], summary["experiment"]))

seen = set()
for _, experiment in sorted(rows, reverse=True):
    if experiment not in seen:
        print(experiment)
        seen.add(experiment)
    if len(seen) == 2:
        break
