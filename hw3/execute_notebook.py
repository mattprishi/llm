#!/usr/bin/env python3
"""Execute the homework with this Python and checkpoint completed cells."""

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import nbformat
from nbclient import NotebookClient


class LocalPythonNotebookClient(NotebookClient):
    def create_kernel_manager(self):
        manager = super().create_kernel_manager()
        # Override only this in-memory spec; do not change shared Jupyter kernels.
        manager.kernel_spec.argv = [
            sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"
        ]
        return manager


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_write(path, text):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", default="sft_function_calling_hw.ipynb")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="Timeout in seconds for each code cell (default: 1800)")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    root = Path(__file__).resolve().parent
    notebook_path = Path(args.notebook)
    if not notebook_path.is_absolute():
        notebook_path = root / notebook_path
    notebook_path = notebook_path.resolve()
    output_dir = root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = output_dir / "executed_partial.ipynb"
    status_path = output_dir / "run_status.json"
    started = time.monotonic()
    cell_started = {}
    cell_finished = set()
    notebook = None
    client = None
    status = {
        "status": "running",
        "started_at": utc_now(),
        "notebook": str(notebook_path),
        "python": sys.executable,
        "kernel": "python3",
        "timeout_seconds": args.timeout,
        "completed_code_cells": 0,
    }

    def save_status():
        status["elapsed_seconds"] = round(time.monotonic() - started, 2)
        atomic_write(status_path, json.dumps(status, ensure_ascii=False, indent=2) + "\n")

    def save_partial():
        if client is not None:
            client.set_widgets_metadata()
        if notebook is not None:
            atomic_write(partial_path, nbformat.writes(notebook))

    def on_cell_start(cell, cell_index, **kwargs):
        if cell.cell_type != "code" or not cell.source.strip():
            return
        cell_started[cell_index] = time.monotonic()
        status["active_cell"] = cell_index + 1
        save_status()
        print(f"START cell {cell_index + 1}/{len(notebook.cells)}", flush=True)

    def on_cell_executed(cell, cell_index, execute_reply, **kwargs):
        elapsed = time.monotonic() - cell_started[cell_index]
        result = execute_reply.get("content", {}).get("status", "unknown")
        cell_finished.add(cell_index)
        if result == "ok":
            status["completed_code_cells"] += 1
            status["active_cell"] = None
        status["last_executed_cell"] = cell_index + 1
        status["last_cell_seconds"] = round(elapsed, 2)
        print(f"FINISH cell {cell_index + 1}: {result}, {elapsed:.2f}s", flush=True)
        save_partial()
        save_status()

    save_status()
    print(f"Notebook: {notebook_path}\nPython: {sys.executable}", flush=True)
    try:
        notebook = nbformat.read(notebook_path, as_version=4)
        nbformat.validate(notebook)
        # Partial artifacts must not retain outputs from an earlier execution.
        notebook.metadata.pop("widgets", None)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                cell.outputs = []
                cell.execution_count = None
                cell.metadata.pop("execution", None)
        status["total_code_cells"] = sum(
            cell.cell_type == "code" and bool(cell.source.strip())
            for cell in notebook.cells
        )
        client = LocalPythonNotebookClient(
            notebook,
            timeout=args.timeout,
            kernel_name="python3",
            allow_errors=False,
            force_raise_errors=True,
            record_timing=True,
            store_widget_state=True,
            on_cell_start=on_cell_start,
            on_cell_executed=on_cell_executed,
        )
        save_partial()
        client.execute(cwd=str(root))
        save_partial()
        nbformat.validate(notebook)
        atomic_write(notebook_path, nbformat.writes(notebook))
    except (Exception, KeyboardInterrupt) as exc:
        status["status"] = "failed"
        status["finished_at"] = utc_now()
        detail = getattr(exc, "evalue", str(exc)) or type(exc).__name__
        status["error"] = f"{type(exc).__name__}: {detail}"[:1000]
        active_cell = status.get("active_cell")
        if active_cell is not None and active_cell - 1 in cell_started:
            index = active_cell - 1
            elapsed = time.monotonic() - cell_started[index]
            status["failed_cell_seconds"] = round(elapsed, 2)
            if index not in cell_finished:
                print(f"FINISH cell {active_cell}: failed, {elapsed:.2f}s", flush=True)
        save_partial()
        save_status()
        print(f"FAILED: {status['error']}", file=sys.stderr, flush=True)
        traceback.print_exc()
        return 130 if isinstance(exc, KeyboardInterrupt) else 1

    status.update(status="completed", finished_at=utc_now(), active_cell=None)
    save_status()
    print(f"COMPLETED in {status['elapsed_seconds']:.2f}s: {notebook_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
