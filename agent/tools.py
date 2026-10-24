"""Tool registry for the agent.

Each tool is a plain Python function plus a JSON-schema description that is
sent to the LLM (OpenAI "function calling" format). Tools never raise: they
return {"error": "..."} so the agent can read the error and self-correct.
"""
from __future__ import annotations

import ast
import csv
import math
import operator
import re
from datetime import date, datetime, timedelta
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ---------------------------------------------------------------------------
# 1. Calculator: safe arithmetic via AST (no eval of arbitrary code)
# ---------------------------------------------------------------------------
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv, ast.USub: operator.neg, ast.UAdd: operator.pos,
}
_FUNCS = {"sqrt": math.sqrt, "log": math.log, "log10": math.log10, "exp": math.exp,
          "abs": abs, "round": round, "min": min, "max": max,
          "sin": math.sin, "cos": math.cos, "floor": math.floor, "ceil": math.ceil}
_CONSTS = {"pi": math.pi, "e": math.e}


def _eval_node(node):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        return _FUNCS[node.func.id](*[_eval_node(a) for a in node.args])
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    raise ValueError(f"unsupported expression element: {ast.dump(node)[:60]}")


def calculator(expression: str) -> dict:
    try:
        # strip thousands separators like 1,284,650 but keep function-argument commas
        expr = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", expression.replace("^", "**"))
        value = _eval_node(ast.parse(expr, mode="eval"))
        if isinstance(value, float):
            value = round(value, 6)
        return {"result": value}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"could not evaluate '{expression}': {exc}"}


# ---------------------------------------------------------------------------
# 2. Date tool
# ---------------------------------------------------------------------------
def date_tool(operation: str, date_a: str, date_b: str | None = None, days: int | None = None) -> dict:
    """operation: 'diff_days' | 'add_days' | 'weekday'. Dates are YYYY-MM-DD."""
    try:
        a = datetime.strptime(date_a, "%Y-%m-%d").date()
        if operation == "diff_days":
            b = datetime.strptime(date_b, "%Y-%m-%d").date()
            return {"result": (b - a).days}
        if operation == "add_days":
            return {"result": (a + timedelta(days=int(days))).isoformat()}
        if operation == "weekday":
            return {"result": a.strftime("%A")}
        return {"error": f"unknown operation '{operation}'; use diff_days, add_days or weekday"}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"bad arguments: {exc}"}


# ---------------------------------------------------------------------------
# 3. Unit converter
# ---------------------------------------------------------------------------
_UNITS = {  # factor to SI base unit
    "length": {"m": 1, "km": 1000, "cm": 0.01, "mm": 0.001, "mi": 1609.344,
               "ft": 0.3048, "in": 0.0254, "yd": 0.9144},
    "mass": {"kg": 1, "g": 0.001, "lb": 0.45359237, "oz": 0.028349523125, "t": 1000},
    "volume": {"l": 1, "ml": 0.001, "gal": 3.785411784, "cup": 0.2365882365},
    "time": {"s": 1, "min": 60, "h": 3600, "day": 86400, "week": 604800},
}


def unit_convert(value: float, from_unit: str, to_unit: str) -> dict:
    f, t = from_unit.strip().lower(), to_unit.strip().lower()
    temps = {"c", "f", "k"}
    if f in temps and t in temps:
        c = {"c": value, "f": (value - 32) * 5 / 9, "k": value - 273.15}[f]
        out = {"c": c, "f": c * 9 / 5 + 32, "k": c + 273.15}[t]
        return {"result": round(out, 4)}
    for table in _UNITS.values():
        if f in table and t in table:
            return {"result": round(value * table[f] / table[t], 6)}
    return {"error": f"cannot convert '{from_unit}' to '{to_unit}'. Known units: "
                     + ", ".join(sorted({u for tb in _UNITS.values() for u in tb} | temps))}


# ---------------------------------------------------------------------------
# 4. Sales-data lookup over a local CSV (simulates an internal database)
# ---------------------------------------------------------------------------
def _load_sales():
    with open(DATA_DIR / "sales.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["units"] = int(r["units"])
        r["revenue"] = float(r["revenue"])
    return rows


def sales_query(metric: str, aggregate: str = "sum", region: str | None = None,
                product: str | None = None, month: str | None = None) -> dict:
    """metric: units|revenue; aggregate: sum|avg|max|min|count; optional filters."""
    try:
        rows = _load_sales()
        if region:
            rows = [r for r in rows if r["region"].lower() == region.lower()]
        if product:
            rows = [r for r in rows if r["product"].lower() == product.lower()]
        if month:
            rows = [r for r in rows if r["month"] == month]
        if not rows:
            return {"error": "no rows match these filters. Regions: North, South, East, West; "
                             "products: Widget, Gadget, Gizmo; months: 2026-01..2026-06"}
        if aggregate == "count":
            return {"result": len(rows)}
        if metric not in ("units", "revenue"):
            return {"error": "metric must be 'units' or 'revenue'"}
        vals = [r[metric] for r in rows]
        agg = {"sum": sum, "avg": lambda v: sum(v) / len(v), "max": max, "min": min}.get(aggregate)
        if agg is None:
            return {"error": "aggregate must be sum, avg, max, min or count"}
        return {"result": round(agg(vals), 2), "rows_used": len(rows)}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Registry + OpenAI-format schemas
# ---------------------------------------------------------------------------
TOOLS = {
    "calculator": calculator,
    "date_tool": date_tool,
    "unit_convert": unit_convert,
    "sales_query": sales_query,
}

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression exactly. Supports + - * / ** % and "
                       "sqrt, log, exp, abs, round, min, max, floor, ceil, pi, e.",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string", "description": "e.g. '(1234*5678)/7'"}},
            "required": ["expression"]}}},
    {"type": "function", "function": {
        "name": "date_tool",
        "description": "Date arithmetic. operation='diff_days' returns date_b - date_a in days; "
                       "'add_days' adds `days` to date_a; 'weekday' returns the weekday name of date_a.",
        "parameters": {"type": "object", "properties": {
            "operation": {"type": "string", "enum": ["diff_days", "add_days", "weekday"]},
            "date_a": {"type": "string", "description": "YYYY-MM-DD"},
            "date_b": {"type": "string", "description": "YYYY-MM-DD (diff_days only)"},
            "days": {"type": "integer", "description": "days to add (add_days only)"}},
            "required": ["operation", "date_a"]}}},
    {"type": "function", "function": {
        "name": "unit_convert",
        "description": "Convert a value between units of length (m, km, cm, mm, mi, ft, in, yd), "
                       "mass (kg, g, lb, oz, t), volume (l, ml, gal, cup), time (s, min, h, day, week) "
                       "or temperature (c, f, k).",
        "parameters": {"type": "object", "properties": {
            "value": {"type": "number"},
            "from_unit": {"type": "string"},
            "to_unit": {"type": "string"}},
            "required": ["value", "from_unit", "to_unit"]}}},
    {"type": "function", "function": {
        "name": "sales_query",
        "description": "Query the company sales database (Jan-Jun 2026). Regions: North, South, East, "
                       "West. Products: Widget, Gadget, Gizmo. Months formatted 'YYYY-MM'.",
        "parameters": {"type": "object", "properties": {
            "metric": {"type": "string", "enum": ["units", "revenue"]},
            "aggregate": {"type": "string", "enum": ["sum", "avg", "max", "min", "count"]},
            "region": {"type": "string"},
            "product": {"type": "string"},
            "month": {"type": "string"}},
            "required": ["metric"]}}},
]


def run_tool(name: str, args: dict) -> dict:
    fn = TOOLS.get(name)
    if fn is None:
        return {"error": f"unknown tool '{name}'. Available: {', '.join(TOOLS)}"}
    try:
        return fn(**args)
    except TypeError as exc:
        return {"error": f"bad arguments for {name}: {exc}"}
