"""The agent loop: reason -> call tools -> observe -> repeat -> final answer.

Guardrails that make it robust (and that the evaluation measures):
  * max_steps budget so a confused model can't loop forever
  * malformed tool-call JSON is caught and fed back as an error (self-correction)
  * tool errors are returned to the model instead of crashing the run
  * a full trace of every step is recorded for debugging / the report
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from .tools import TOOL_SCHEMAS, run_tool

AGENT_SYSTEM = (
    "You are a precise assistant that solves tasks using tools. "
    "Never do arithmetic, date math, unit conversion or data lookups in your head: "
    "always call the appropriate tool. If a tool returns an error, fix your arguments and retry. "
    "When you are done, reply with one line in the form 'FINAL ANSWER: <answer>' "
    "where <answer> is only the number or short text, without units."
)
BASELINE_SYSTEM = (
    "You are a precise assistant. Solve the task. You have no tools. "
    "Reply with one line in the form 'FINAL ANSWER: <answer>' where <answer> "
    "is only the number or short text, without units."
)
FINAL_RE = re.compile(r"FINAL ANSWER:\s*(.+)", re.IGNORECASE)


@dataclass
class RunResult:
    answer: str | None
    steps: int
    tool_calls: int
    tool_errors: int
    malformed_calls: int
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    trace: list[dict] = field(default_factory=list)


def extract_final(text: str | None) -> str | None:
    if not text:
        return None
    m = FINAL_RE.search(text)
    return m.group(1).strip().splitlines()[0].strip() if m else None


def run_agent(llm, task: str, use_tools: bool = True, max_steps: int = 8) -> RunResult:
    messages = [{"role": "system", "content": AGENT_SYSTEM if use_tools else BASELINE_SYSTEM},
                {"role": "user", "content": task}]
    trace: list[dict] = []
    stats = dict(tool_calls=0, tool_errors=0, malformed=0, pt=0, ct=0)
    t0 = time.perf_counter()

    for step in range(1, max_steps + 1):
        resp = llm.chat(messages, TOOL_SCHEMAS if use_tools else None)
        stats["pt"] += resp.prompt_tokens
        stats["ct"] += resp.completion_tokens

        if not resp.tool_calls:
            answer = extract_final(resp.content)
            trace.append({"step": step, "type": "answer", "content": resp.content})
            if answer is None and step < max_steps:
                # model forgot the required format: nudge it once more
                messages.append({"role": "assistant", "content": resp.content or ""})
                messages.append({"role": "user", "content": "Reply only with 'FINAL ANSWER: <answer>'."})
                continue
            return _result(answer, step, stats, t0, trace)

        messages.append({"role": "assistant", "content": resp.content or "",
                         "tool_calls": [{"id": c.id, "type": "function",
                                         "function": {"name": c.name, "arguments": c.arguments}}
                                        for c in resp.tool_calls]})
        for call in resp.tool_calls:
            stats["tool_calls"] += 1
            try:
                args = json.loads(call.arguments) if call.arguments else {}
                if not isinstance(args, dict):
                    raise ValueError("arguments must be a JSON object")
                output = run_tool(call.name, args)
            except (json.JSONDecodeError, ValueError) as exc:
                stats["malformed"] += 1
                args, output = call.arguments, {"error": f"invalid JSON arguments: {exc}"}
            if "error" in output:
                stats["tool_errors"] += 1
            trace.append({"step": step, "type": "tool", "tool": call.name, "args": args, "output": output})
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(output)})

    return _result(None, max_steps, stats, t0, trace)  # step budget exhausted


def _result(answer, steps, s, t0, trace) -> RunResult:
    return RunResult(answer, steps, s["tool_calls"], s["tool_errors"], s["malformed"],
                     s["pt"], s["ct"], round(time.perf_counter() - t0, 3), trace)
