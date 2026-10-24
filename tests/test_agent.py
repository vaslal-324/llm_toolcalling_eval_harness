"""Unit tests: tools, answer scoring, and the agent loop's self-correction (no LLM needed)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.core import extract_final, run_agent  # noqa: E402
from agent.llm import LLMResponse, ToolCall  # noqa: E402
from agent.tools import calculator, date_tool, run_tool, sales_query, unit_convert  # noqa: E402
from evaluate import is_correct  # noqa: E402


def test_calculator_basic_and_thousands_separator():
    assert calculator("48217*9361")["result"] == 451359337
    assert calculator("1,284,650*0.075")["result"] == 96348.75
    assert calculator("round(10/3, 2)")["result"] == 3.33


def test_calculator_rejects_code():
    assert "error" in calculator("__import__('os').system('ls')")
    assert "error" in calculator("2**100000")


def test_date_tool():
    assert date_tool("diff_days", "2026-01-01", "2026-12-25")["result"] == 358
    assert date_tool("weekday", "1999-12-31")["result"] == "Friday"
    assert "error" in date_tool("explode", "2026-01-01")


def test_unit_convert():
    assert abs(unit_convert(26.2, "mi", "km")["result"] - 42.16481) < 1e-3
    assert unit_convert(98.6, "F", "C")["result"] == 37.0
    assert "error" in unit_convert(1, "kg", "km")


def test_sales_query():
    assert sales_query("units", "count")["result"] == 72
    assert "error" in sales_query("revenue", region="Mars")


def test_unknown_tool():
    assert "error" in run_tool("rm_rf", {})


def test_scoring():
    assert is_correct("42.16", 42.16481)
    assert is_correct("$96,348.75", 96348.75)
    assert not is_correct("42.5", 42.16481)
    assert is_correct("friday", "Friday")
    assert not is_correct(None, 1)
    assert extract_final("blah\nFINAL ANSWER: 12 km") == "12 km"


class ScriptedLLM:
    """Sends broken JSON first, then a valid call, then the answer."""
    model = "scripted"

    def __init__(self):
        self.turn = 0

    def chat(self, messages, tools=None):
        self.turn += 1
        if self.turn == 1:
            return LLMResponse(None, [ToolCall("a", "calculator", "{not json")])
        if self.turn == 2:
            # the error from turn 1 must have been fed back to the model
            assert "invalid JSON" in messages[-1]["content"]
            return LLMResponse(None, [ToolCall("b", "calculator", json.dumps({"expression": "6*7"}))])
        return LLMResponse("FINAL ANSWER: 42")


def test_agent_recovers_from_malformed_tool_call():
    r = run_agent(ScriptedLLM(), "what is 6*7?")
    assert r.answer == "42"
    assert r.malformed_calls == 1 and r.tool_errors == 1 and r.tool_calls == 2


class LoopingLLM:
    model = "looper"

    def chat(self, messages, tools=None):
        return LLMResponse(None, [ToolCall("x", "calculator", '{"expression": "1+1"}')])


def test_step_budget_stops_infinite_loops():
    r = run_agent(LoopingLLM(), "loop", max_steps=3)
    assert r.answer is None and r.steps == 3
