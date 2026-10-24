"""Builds data/benchmark.jsonl: 40 tasks in 5 categories.

Ground-truth answers are computed by executing each task's reference tool plan,
so every label is guaranteed correct. The plan is also what MockLLM replays.
Run:  python data/build_benchmark.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.tools import run_tool  # noqa: E402

C = lambda e: {"tool": "calculator", "args": {"expression": e}}  # noqa: E731
D = lambda **a: {"tool": "date_tool", "args": a}  # noqa: E731
U = lambda v, f, t: {"tool": "unit_convert", "args": {"value": v, "from_unit": f, "to_unit": t}}  # noqa: E731
S = lambda **a: {"tool": "sales_query", "args": a}  # noqa: E731

TASKS = [
    # ---- arithmetic: hard for LLMs to do "in their head"
    ("arithmetic", "What is 48217 multiplied by 9361?", [C("48217*9361")]),
    ("arithmetic", "Compute 7.5% of 1,284,650.", [C("1284650*0.075")]),
    ("arithmetic", "What is the square root of 98596?", [C("sqrt(98596)")]),
    ("arithmetic", "Compute (3847 + 9128) * 17 - 4521.", [C("(3847+9128)*17-4521")]),
    ("arithmetic", "What is 2 to the power 37?", [C("2**37")]),
    ("arithmetic", "Divide 987654321 by 12345 and round to 2 decimal places.", [C("round(987654321/12345, 2)")]),
    ("arithmetic", "What is 123456789 mod 9973?", [C("123456789 % 9973")]),
    ("arithmetic", "A loan of 250000 grows at 6.8% per year compounded annually. What is it worth after 12 years? Round to 2 decimals.",
     [C("round(250000*1.068**12, 2)")]),
    ("arithmetic", "What is the natural log of 5000, rounded to 4 decimals?", [C("round(log(5000), 4)")]),
    ("arithmetic", "Compute 76543 * 3289 / 41, rounded to 3 decimals.", [C("round(76543*3289/41, 3)")]),
    # ---- dates
    ("dates", "How many days are there from 2024-02-14 to 2026-11-03?", [D(operation="diff_days", date_a="2024-02-14", date_b="2026-11-03")]),
    ("dates", "What date is 1000 days after 2025-03-17? Answer as YYYY-MM-DD.", [D(operation="add_days", date_a="2025-03-17", days=1000)]),
    ("dates", "What day of the week was 1999-12-31?", [D(operation="weekday", date_a="1999-12-31")]),
    ("dates", "How many days between 2000-01-01 and 2026-10-19?", [D(operation="diff_days", date_a="2000-01-01", date_b="2026-10-19")]),
    ("dates", "What date is 365 days before 2028-03-01? Answer as YYYY-MM-DD.", [D(operation="add_days", date_a="2028-03-01", days=-365)]),
    ("dates", "What day of the week will 2031-07-04 be?", [D(operation="weekday", date_a="2031-07-04")]),
    ("dates", "How many days from 2026-01-01 to 2026-12-25?", [D(operation="diff_days", date_a="2026-01-01", date_b="2026-12-25")]),
    ("dates", "What date is 77 days after 2026-10-19? Answer as YYYY-MM-DD.", [D(operation="add_days", date_a="2026-10-19", days=77)]),
    # ---- unit conversion
    ("units", "Convert 26.2 miles to kilometers.", [U(26.2, "mi", "km")]),
    ("units", "Convert 98.6 degrees Fahrenheit to Celsius.", [U(98.6, "f", "c")]),
    ("units", "How many pounds is 73 kg?", [U(73, "kg", "lb")]),
    ("units", "Convert 5280 feet to meters.", [U(5280, "ft", "m")]),
    ("units", "How many liters are in 12 gallons?", [U(12, "gal", "l")]),
    ("units", "Convert 3 weeks to hours.", [U(3, "week", "h")]),
    ("units", "Convert 300 kelvin to Fahrenheit.", [U(300, "k", "f")]),
    ("units", "How many ounces are in 2.5 kg?", [U(2.5, "kg", "oz")]),
    # ---- data lookup (impossible without the tool: private data)
    ("data", "What was total revenue for the North region across all months in the sales database?", [S(metric="revenue", aggregate="sum", region="North")]),
    ("data", "How many Gadget units were sold in 2026-03 across all regions?", [S(metric="units", aggregate="sum", product="Gadget", month="2026-03")]),
    ("data", "What is the average monthly revenue of Gizmo in the West region?", [S(metric="revenue", aggregate="avg", product="Gizmo", region="West")]),
    ("data", "What is the maximum units sold in a single row for Widget?", [S(metric="units", aggregate="max", product="Widget")]),
    ("data", "What was total revenue in 2026-06 for all products and regions?", [S(metric="revenue", aggregate="sum", month="2026-06")]),
    ("data", "What was the minimum revenue of any row in the East region?", [S(metric="revenue", aggregate="min", region="East")]),
    ("data", "How many total units did the South region sell?", [S(metric="units", aggregate="sum", region="South")]),
    ("data", "What was total Widget revenue across the whole database?", [S(metric="revenue", aggregate="sum", product="Widget")]),
    # ---- multi-step: needs chaining 2+ tool calls
    ("multi_step", "Total revenue for North minus total revenue for South in the sales database?",
     [S(metric="revenue", aggregate="sum", region="North"), S(metric="revenue", aggregate="sum", region="South"), C("{0}-{1}")]),
    ("multi_step", "What percentage of all units sold were Gadgets? Round to 2 decimals.",
     [S(metric="units", aggregate="sum", product="Gadget"), S(metric="units", aggregate="sum"), C("round({0}/{1}*100, 2)")]),
    ("multi_step", "A runner covers 10 miles per day. How many kilometers do they cover in 23 days? Round to 2 decimals.",
     [U(10, "mi", "km"), C("round({0}*23, 2)")]),
    ("multi_step", "If Widget revenue in the East region grows by 12%, what would it be? Round to 2 decimals.",
     [S(metric="revenue", aggregate="sum", product="Widget", region="East"), C("round({0}*1.12, 2)")]),
    ("multi_step", "How many weeks (rounded to 2 decimals) are there between 2025-06-01 and 2026-10-19?",
     [D(operation="diff_days", date_a="2025-06-01", date_b="2026-10-19"), C("round({0}/7, 2)")]),
    ("multi_step", "What is the ratio of West total units to East total units, rounded to 3 decimals?",
     [S(metric="units", aggregate="sum", region="West"), S(metric="units", aggregate="sum", region="East"), C("round({0}/{1}, 3)")]),
]


def resolve(plan):
    """Execute the plan; '{i}' placeholders are filled with earlier results."""
    results, concrete = [], []
    for step in plan:
        args = {k: (v.format(*results) if isinstance(v, str) else v) for k, v in step["args"].items()}
        out = run_tool(step["tool"], args)
        assert "error" not in out, (step, out)
        results.append(out["result"])
        concrete.append({"tool": step["tool"], "args": args})
    return results[-1], concrete


def main():
    out = Path(__file__).with_name("benchmark.jsonl")
    with open(out, "w") as fh:
        for i, (cat, q, plan) in enumerate(TASKS, 1):
            answer, concrete = resolve(plan)
            fh.write(json.dumps({"id": f"T{i:02d}", "category": cat, "question": q,
                                 "answer": answer, "plan": concrete}) + "\n")
    print(f"wrote {len(TASKS)} tasks to {out}")


if __name__ == "__main__":
    main()
