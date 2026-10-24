# Tool-Using LLM Agent with an Evaluation Harness

A from-scratch LLM agent (no LangChain or other agent framework) that solves tasks by
calling tools through native function calling, plus a benchmark harness that measures
how much the tools actually help compared with the same model answering on its own.

```
question ──► LLM ──► tool call (JSON) ──► tool runs ──► result ──► LLM ──► ... ──► FINAL ANSWER
                         ▲                                  │
                         └──── errors fed back for retry ◄──┘
```

**What it demonstrates**
- An agent loop with tool schemas, a step budget, rate-limit-aware retries, and self-correction
  when the model sends malformed JSON or bad arguments.
- 4 tools: a restricted AST-based calculator (cannot run arbitrary code), date arithmetic,
  unit conversion, and a query tool over a private sales dataset.
- A 40-task benchmark in 5 categories with exact, auto-generated ground truth.
- An A/B evaluation (agent vs. no-tool baseline) that reports accuracy, tool-error rate,
  tokens, and latency, and writes a `REPORT.md` with a chart.
- Lightweight: runs on a free cloud model (Groq or Gemini), so nothing large is downloaded.
  The whole install is a few small Python packages. Also works with OpenAI or a local Ollama model.

## Project structure

```
agent-tool-eval/
├── agent/
│   ├── core.py            # the agent loop (reason → act → observe), guardrails, tracing
│   ├── tools.py           # tool implementations + JSON schemas
│   └── llm.py             # OpenAI-compatible client + offline mock
├── data/
│   ├── sales.csv          # private dataset used by the sales_query tool
│   ├── benchmark.jsonl    # 40 tasks with ground-truth answers
│   ├── build_benchmark.py # regenerates benchmark.jsonl (answers computed by running tools)
│   └── make_sales_csv.py  # regenerates sales.csv (seeded)
├── tests/test_agent.py    # 9 unit tests, no LLM needed
├── evaluate.py            # runs the benchmark, writes results/
├── make_report.py         # builds REPORT.md + results/accuracy_by_category.png
├── requirements.txt
└── .env.example
```

## Step-by-step: run it (Ubuntu / macOS / Windows)

You need Python 3.8+ and an internet connection. No GPU, and no model download.

### Step 1: Set up the project

**Ubuntu:**
```bash
sudo apt update && sudo apt install -y python3 python3-venv python3-pip unzip
cd agent-tool-eval
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell):** `python -m venv .venv`, then `.venv\Scripts\Activate.ps1`, then
`pip install -r requirements.txt`.

Each time you open a new terminal, `cd agent-tool-eval` and activate the venv again.

### Step 2: Check everything works offline (about 5 seconds)

```bash
python -m pytest -q                 # expect: 9 passed
python evaluate.py --backend mock   # dry run of the whole pipeline, no LLM
```

The mock backend replays the reference answers, so it shows 100% for the agent. That only
proves the pipeline works; it is **not** a result to report.

### Step 3: Get a free API key (about 2 minutes)

**Option A: Groq (default, fastest).** Sign in at https://console.groq.com/keys and click
**Create API Key**. Copy the key (it starts with `gsk_`); it is shown only once.

**Option B: Google Gemini.** Sign in at https://aistudio.google.com/apikey and click
**Create API key**.

### Step 4: Put the key in `.env`

```bash
cp .env.example .env        # Windows: copy .env.example .env
nano .env                   # or open it in any text editor
```

- **Groq:** replace `gsk_your_key_here` with your key. Save (in nano: Ctrl+O, Enter, Ctrl+X).
- **Gemini:** put `#` in front of the four Option A lines, remove `#` from the four Option B
  lines, and paste your key into `LLM_API_KEY`.

`.env` is in `.gitignore`, so your key never gets pushed to GitHub.

### Step 5: Run the evaluation

```bash
python evaluate.py --limit 3    # quick check on 3 questions (a few seconds)
python evaluate.py              # full run: 40 questions with tools, 40 without
```

The full run makes about 130 API calls and takes roughly 5 to 10 minutes on the free tier.
Free tiers allow only a limited number of requests per minute, so you will sometimes see
`(rate limit hit; waiting 8s, retry 1/8)`. That is normal: the program waits and continues on
its own. `LLM_MIN_INTERVAL_S` in `.env` spaces out calls to reduce this.

### Step 6: Generate the report

```bash
python make_report.py
```

This creates `REPORT.md` (headline numbers, metric tables, per-category accuracy,
failure analysis) and `results/accuracy_by_category.png`. Take your resume numbers from
the "Headline results" section, and mention which model you used.

### Step 7 (optional): Compare two models

Run with a second model, saving to a different folder:

```bash
LLM_MODEL=<another-model-name> python evaluate.py --out results_model2
python make_report.py results_model2     # writes REPORT_results_model2.md
```

### Using Hugging Face instead

Hugging Face's Inference Providers API also works, but free accounts get no monthly credits:
you need to buy credits or have a PRO plan. A full run uses roughly 100K tokens, which
should cost a few cents. Create a fine-grained token at https://huggingface.co/settings/tokens
with the "Make calls to Inference Providers" permission, then use Option D in `.env`.

### Prefer to run fully offline?

Use Option C in `.env` with [Ollama](https://ollama.com). It downloads a model of about
2 to 5 GB (`ollama pull qwen2.5:7b`, or `qwen2.5:3b` on a laptop with 8 GB RAM or less).

## Push to GitHub with the report

```bash
git init
git add .
git commit -m "Tool-using LLM agent with evaluation harness and report"
git branch -M main
git remote add origin https://github.com/<your-username>/agent-tool-eval.git
git push -u origin main
```

`.env` (your API key) is in `.gitignore` and will not be uploaded. `REPORT.md` and
`results/` are committed so the report shows up on GitHub.

## Command reference

| Command | What it does |
|---|---|
| `python evaluate.py` | Agent and baseline on all 40 tasks |
| `python evaluate.py --mode agent` | Agent only |
| `python evaluate.py --limit 10` | First 10 tasks only |
| `python evaluate.py --max-steps 4` | Tighter step budget |
| `python evaluate.py --backend mock` | Offline pipeline check |
| `python make_report.py` | Build `REPORT.md` from `results/` |
| `python data/build_benchmark.py` | Rebuild the benchmark after editing tasks |

## How it works

1. `evaluate.py` loads each task and calls `run_agent()` in `agent/core.py`.
2. The agent sends the conversation plus the tool JSON schemas to the model.
3. If the model returns tool calls, each one is parsed and run by `run_tool()`. Invalid JSON,
   unknown tools and bad arguments come back as `{"error": ...}` messages so the model can fix
   its call instead of the program crashing.
4. The loop stops when the model replies `FINAL ANSWER: <x>` or after 8 steps.
5. Answers are scored against ground truth (numbers within 0.1% tolerance), and every step
   is saved to `results/runs_agent.jsonl` for debugging.

## Extending it

- **Add a tool:** write a function in `agent/tools.py`, register it in `TOOLS`, and add its schema
  to `TOOL_SCHEMAS`.
- **Add tasks:** append to `TASKS` in `data/build_benchmark.py` and run it; answers are computed for you.

## Troubleshooting

| Message | Fix |
|---|---|
| `LLM settings missing` | You haven't created `.env`. Run `cp .env.example .env` (Step 4). |
| `LLM_API_KEY in .env is still the placeholder` | Paste your real key into `.env`. |
| `API key rejected` | The key was copied wrongly or deleted. Create a new one. |
| `Model or URL not found` | The provider renamed or retired the model. Pick a current model that supports tool use / function calling from the provider's model list, and set `LLM_MODEL` to it. |
| Many `rate limit hit` lines | Normal on free tiers. If the run fails, raise `LLM_MIN_INTERVAL_S` (for example to 5) or run in two halves with `--limit 20`. |
| Daily limit reached | Free tiers also have daily caps. Wait until tomorrow or switch to the other provider. |
| `externally-managed-environment` (Ubuntu) | Activate the venv first: `source .venv/bin/activate`. |
| `ensurepip is not available` (Ubuntu) | `sudo apt install python3-venv`, delete `.venv`, and redo Step 1. |
| The agent never calls tools | The model lacks tool-calling support. Choose one that lists "tool use" or "function calling". |
