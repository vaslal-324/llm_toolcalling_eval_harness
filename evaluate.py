"""Evaluation harness: runs the benchmark in agent mode (with tools) and baseline
mode (same model, no tools), scores every answer and saves results.

Examples
  python evaluate.py --backend mock                 # offline smoke test, no LLM needed
  python evaluate.py                                # uses LLM_* settings from .env
  python evaluate.py --mode agent --limit 10        # quick partial run
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path

from agent.core import run_agent
from agent.llm import FatalLLMError, MockLLM, OpenAIChat

ROOT = Path(__file__).resolve().parent
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def load_env(path: Path) -> None:
    """Minimal .env loader (KEY=VALUE lines) so no extra dependency is needed."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def is_correct(pred: str | None, gold) -> bool:
    """Numeric answers: 0.1% relative tolerance (or 0.01 absolute). Text: case-insensitive."""
    if pred is None:
        return False
    if isinstance(gold, (int, float)):
        nums = NUM_RE.findall(pred.replace(",", "").replace("$", ""))
        if not nums:
            return False
        val = float(nums[0])
        return abs(val - gold) <= max(1e-3 * abs(gold), 0.01)
    return pred.strip().strip(".").lower() == str(gold).strip().lower()


def evaluate(llm, tasks, mode: str, max_steps: int) -> list[dict]:
    rows = []
    for i, task in enumerate(tasks, 1):
        if isinstance(llm, MockLLM):
            llm.reset(task)
        try:
            r = run_agent(llm, task["question"], use_tools=(mode == "agent"), max_steps=max_steps)
            err = None
        except FatalLLMError:
            raise  # bad key / model name: no point running the remaining tasks
        except Exception as exc:  # noqa: BLE001  keep going if one task blows up
            r, err = None, str(exc)
        ok = bool(r) and is_correct(r.answer, task["answer"])
        rows.append({
            "id": task["id"], "category": task["category"], "mode": mode,
            "question": task["question"], "gold": task["answer"],
            "pred": r.answer if r else None, "correct": ok,
            "steps": r.steps if r else 0, "tool_calls": r.tool_calls if r else 0,
            "tool_errors": r.tool_errors if r else 0, "malformed_calls": r.malformed_calls if r else 0,
            "tokens": (r.prompt_tokens + r.completion_tokens) if r else 0,
            "latency_s": r.latency_s if r else 0.0, "error": err,
            "trace": r.trace if r else [],
        })
        print(f"[{mode:8s}] {i:2d}/{len(tasks)} {task['id']} {'PASS' if ok else 'FAIL'}  "
              f"pred={rows[-1]['pred']!s:<22.22} gold={task['answer']}")
    return rows


def summarize(rows: list[dict]) -> dict:
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r["correct"])
    calls = sum(r["tool_calls"] for r in rows)
    return {
        "n": len(rows),
        "accuracy": round(100 * sum(r["correct"] for r in rows) / len(rows), 1),
        "by_category": {c: round(100 * sum(v) / len(v), 1) for c, v in sorted(by_cat.items())},
        "avg_steps": round(statistics.mean(r["steps"] for r in rows), 2),
        "avg_tool_calls": round(calls / len(rows), 2),
        "tool_error_rate": round(100 * sum(r["tool_errors"] for r in rows) / calls, 1) if calls else 0.0,
        "recovered_after_error": sum(1 for r in rows if r["tool_errors"] and r["correct"]),
        "runs_with_tool_error": sum(1 for r in rows if r["tool_errors"]),
        "avg_tokens": round(statistics.mean(r["tokens"] for r in rows)),
        "avg_latency_s": round(statistics.mean(r["latency_s"] for r in rows), 2),
        "p95_latency_s": round(sorted(r["latency_s"] for r in rows)[int(0.95 * (len(rows) - 1))], 2),
        "crashes": sum(1 for r in rows if r["error"]),
    }


def main():
    load_env(ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["openai", "mock"], default="openai")
    ap.add_argument("--mode", choices=["agent", "baseline", "both"], default="both")
    ap.add_argument("--limit", type=int, default=None, help="only run the first N tasks")
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()

    tasks = [json.loads(l) for l in open(ROOT / "data" / "benchmark.jsonl")][: args.limit]
    try:
        llm = MockLLM() if args.backend == "mock" else OpenAIChat()
    except FatalLLMError as exc:
        raise SystemExit(f"Error: {exc}")
    modes = ["agent", "baseline"] if args.mode == "both" else [args.mode]

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    summary = {"model": llm.model, "backend": args.backend,
               "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "modes": {}}
    for mode in modes:
        try:
            rows = evaluate(llm, tasks, mode, args.max_steps)
        except FatalLLMError as exc:
            raise SystemExit(f"\nError: {exc}")
        with open(out / f"runs_{mode}.jsonl", "w") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in rows)
        summary["modes"][mode] = summarize(rows)

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n" + json.dumps(summary, indent=2))
    print(f"\nSaved to {out}/. Next: python make_report.py")


if __name__ == "__main__":
    main()
