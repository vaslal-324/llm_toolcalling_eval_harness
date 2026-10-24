"""Turns results/summary.json + runs_*.jsonl into REPORT.md and charts in results/.

Run after evaluate.py:   python make_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"  # override: python make_report.py results_3b


def load_runs(mode):
    p = RES / f"runs_{mode}.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else []


def chart_by_category(summary):
    modes = summary["modes"]
    cats = sorted(next(iter(modes.values()))["by_category"])
    fig, ax = plt.subplots(figsize=(8, 4.2))
    width = 0.38
    colors = {"agent": "#2a6fdb", "baseline": "#b0b7c3"}
    for k, mode in enumerate(modes):
        vals = [modes[mode]["by_category"].get(c, 0) for c in cats]
        xs = [i + (k - (len(modes) - 1) / 2) * width for i in range(len(cats))]
        bars = ax.bar(xs, vals, width, label=f"{mode} ({modes[mode]['accuracy']}% overall)",
                      color=colors.get(mode, "#888"))
        ax.bar_label(bars, fmt="%.0f", fontsize=8, padding=2)
    ax.set_xticks(range(len(cats)), [c.replace("_", "-") for c in cats])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 110)
    ax.set_title(f"Task accuracy by category - {summary['model']}")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(RES / "accuracy_by_category.png", dpi=150)
    plt.close(fig)


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    global RES
    if len(sys.argv) > 1:
        RES = ROOT / sys.argv[1]
    summary = json.loads((RES / "summary.json").read_text())
    modes = summary["modes"]
    chart_by_category(summary)
    a, b = modes.get("agent"), modes.get("baseline")

    lines = [
        "# Evaluation Report: Tool-Using LLM Agent",
        "",
        f"*Model:* `{summary['model']}`  |  *Backend:* `{summary['backend']}`  |  *Run:* {summary['timestamp']}",
        "",
    ]
    if summary["backend"] == "mock":
        lines += ["> **Note:** this report was produced with the offline mock backend, which replays the "
                  "reference plans. It verifies the pipeline only; the numbers are not model results. "
                  "Re-run `python evaluate.py` with a real model before quoting any metric.", ""]

    lines += ["## 1. Headline results", ""]
    if a and b:
        gain = round(a["accuracy"] - b["accuracy"], 1)
        rel = f"{round(a['accuracy'] / b['accuracy'], 1)}x" if b["accuracy"] else "n/a (baseline 0%)"
        lines += [f"- Tool-using agent accuracy: **{a['accuracy']}%** vs **{b['accuracy']}%** for the same "
                  f"model without tools (**+{gain} percentage points**, {rel}).",
                  f"- Benchmark: {a['n']} tasks across {len(a['by_category'])} categories.",
                  f"- Tool calls per task: {a['avg_tool_calls']}; tool-error rate {a['tool_error_rate']}%; "
                  f"{a['recovered_after_error']}/{a['runs_with_tool_error']} runs that hit a tool error still "
                  f"reached the correct answer (self-correction).",
                  f"- Average cost: ~{a['avg_tokens']} tokens and {a['avg_latency_s']}s per task "
                  f"(p95 {a['p95_latency_s']}s). Crashes: {a['crashes']}.", ""]

    metric_rows = []
    keys = [("accuracy", "Accuracy (%)"), ("avg_steps", "Avg LLM steps"), ("avg_tool_calls", "Avg tool calls"),
            ("tool_error_rate", "Tool error rate (%)"), ("avg_tokens", "Avg tokens / task"),
            ("avg_latency_s", "Avg latency (s)"), ("p95_latency_s", "p95 latency (s)"), ("crashes", "Crashes")]
    for k, label in keys:
        metric_rows.append([label] + [modes[m][k] for m in modes])
    lines += ["## 2. Overall metrics", "", md_table(["Metric"] + list(modes), metric_rows), ""]

    cats = sorted(next(iter(modes.values()))["by_category"])
    lines += ["## 3. Accuracy by category", "",
              md_table(["Category"] + list(modes), [[c] + [modes[m]["by_category"][c] for m in modes] for c in cats]),
              "", f"![accuracy by category]({RES.name}/accuracy_by_category.png)", ""]

    lines += ["## 4. Failure analysis (agent mode)", ""]
    fails = [r for r in load_runs("agent") if not r["correct"]]
    if fails:
        lines += [md_table(["Task", "Category", "Predicted", "Expected", "Tool calls", "Tool errors"],
                           [[r["id"], r["category"], r["pred"], r["gold"], r["tool_calls"], r["tool_errors"]]
                            for r in fails]),
                  "", "Full step-by-step traces for every task are in `results/runs_agent.jsonl`.", ""]
    else:
        lines += ["The agent answered every task correctly.", ""]

    lines += [
        "## 5. Method",
        "",
        "- **Agent loop** (`agent/core.py`): the model receives JSON-schema tool definitions via native function "
        "calling, emits tool calls, observes results and iterates until it outputs `FINAL ANSWER:` or hits an "
        "8-step budget. Malformed JSON and tool exceptions are returned to the model as errors so it can retry.",
        "- **Tools** (`agent/tools.py`): AST-sandboxed calculator, date arithmetic, unit conversion, and a query "
        "tool over a private sales CSV (data the model cannot know).",
        "- **Baseline:** identical model and prompt format, but no tools.",
        "- **Scoring:** numeric answers correct within 0.1% relative (or 0.01 absolute) tolerance; text answers "
        "case-insensitive exact match. Ground truth is generated by executing reference tool plans, so labels are exact.",
        "",
        "## 6. Limitations",
        "",
        "- 40 tasks is a small benchmark; differences of a few points are within noise.",
        "- Results depend on the model; smaller local models make more malformed tool calls.",
        "- Data-lookup tasks are unanswerable without tools by design, which inflates the gap in that category.",
        "",
    ]
    name = "REPORT.md" if RES.name == "results" else f"REPORT_{RES.name}.md"
    (ROOT / name).write_text("\n".join(lines))
    print(f"wrote {name} and {RES.name}/accuracy_by_category.png")


if __name__ == "__main__":
    main()
