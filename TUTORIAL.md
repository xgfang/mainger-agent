# mainger-agent — Complete Tutorial

This document walks you through installing and using **mainger-agent**, an
LLM-driven assistant for the `mainger` framework for privacy-constrained
transfer learning.

The tool comes in two interchangeable forms:

- A **command-line tool** (`agent.py`): single-shot analysis from a terminal.
- A **web UI** (`server.py` + `web/index.html`): multi-turn chat in your browser.

Both share the same R bridge to the `mainger` package, the same vendor support,
and the same output format. The only difference is the interface.

---

## Table of contents

1. [What you'll need](#1-what-youll-need)
2. [Install](#2-install)
3. [Get an API key](#3-get-an-api-key)
4. [Configure](#4-configure)
5. [Use the command-line tool](#5-use-the-command-line-tool)
6. [Use the web UI](#6-use-the-web-ui)
7. [Output files](#7-output-files)
8. [Choosing a model](#8-choosing-a-model)
9. [Troubleshooting](#9-troubleshooting)
10. [Privacy and security](#10-privacy-and-security)
11. [What the agent does, step by step](#11-what-the-agent-does-step-by-step)

Plus an [Appendix](#appendix-extension-points) with extension points.

## Quick reading guide

Pick the path below that matches what you are trying to do.

- **Just want to try it?** (~10 minutes) Read Sections 1 through 4, then either Section 5 (command line) or Section 6 (web UI).
- **Deploying for collaborators?** (~30 minutes) Sections 1 through 4 and 6, then 7 and 10. Refer to Section 9 if something breaks.
- **Extending the agent or adding a tool?** (~1 hour) Sections 1 through 4, then the Appendix. Section 8 helps you pick a development model.
- **Just want a conceptual overview?** Section 11 alone is enough.

---

## 1. What you'll need

Three pieces of software and one API key:

| Tool   | Minimum version | What it's for                           |
|--------|-----------------|-----------------------------------------|
| Python | 3.10+           | The agent orchestration code            |
| R      | 4.2+            | The `mainger` package and the bridge    |
| `mainger` R package | 0.2.0+ | The actual statistical computation |
| An LLM API key | — | At least one (see Section 3)           |

### Verify what's already installed

```bash
python --version          # should print 3.10 or higher
Rscript --version         # should print 4.2 or higher
```

- **macOS / Linux**: open Terminal.
- **Windows**: open PowerShell. If `Rscript` isn't found, add R's `bin` folder
  (typically `C:\Program Files\R\R-x.y.z\bin\`) to your PATH and restart the shell.

If Python is missing, install from [python.org](https://www.python.org/downloads/)
or via your package manager. For R, download from [r-project.org](https://cran.r-project.org/).

---

## 2. Install

### 2.1 Get the code

```bash
git clone https://github.com/<your-org>/mainger-agent.git
cd mainger-agent
```

On Windows, paths with spaces or parentheses must be quoted:

```powershell
cd "C:\path\with spaces\mainger-agent"
```

### 2.2 Install Python dependencies

```bash
pip install -r requirements.txt
```

> **Recommended:** use a virtual environment first.
> ```bash
> python -m venv .venv
> source .venv/bin/activate          # macOS/Linux
> .venv\Scripts\activate             # Windows
> pip install -r requirements.txt
> ```

### 2.3 Install the `mainger` R package

```bash
Rscript -e "install.packages('jsonlite', repos='https://cloud.r-project.org')"
Rscript -e "install.packages('path/to/mainger_0.2.0.tar.gz', repos=NULL, type='source')"
```

### 2.4 Verify everything

```bash
python -c "from agent import run_agent; print('agent OK')"
Rscript -e "library(mainger); cat('mainger', as.character(packageVersion('mainger')), '\n')"
```

You should see `agent OK` and `mainger 0.2.0` (or higher).

---

## 3. Get an API key

You need at least one LLM provider's API key. The list below covers all
supported vendors. **For first-time users, OpenRouter is the easiest path**:
sign up, grab a key, and use one of their free-tier models.

### Closed-source hosted models

#### Anthropic (Claude family)

1. Sign up at <https://console.anthropic.com>.
2. Add a payment method under **Plans & Billing**.
3. Navigate to **API Keys**, click **Create Key**, copy the value (`sk-ant-...`).

Models: `claude-opus-4-7`, `claude-sonnet-4-6`, `claude-haiku-4-5-20251001`.
Env var: `ANTHROPIC_API_KEY`.

#### OpenAI (GPT family)

1. Sign up at <https://platform.openai.com>.
2. Add a payment method under **Settings → Billing**.
3. Go to **API Keys**, click **Create new secret key**, copy the value (`sk-...`).

Models: `gpt-4o`, `gpt-4o-mini`. Env var: `OPENAI_API_KEY`.

#### Google (Gemini family)

The lowest-friction signup of the closed providers.

1. Visit <https://aistudio.google.com/app/apikey>.
2. Click **Create API key**. AI Studio creates a project automatically.
3. Copy the key (`AIza...`).

Free tier exists for development. Models: `gemini-2.5-pro`, `gemini-2.5-flash`,
`gemini-1.5-pro`. Env var: `GOOGLE_API_KEY` or `GEMINI_API_KEY`.

#### xAI (Grok family)

1. Sign up at <https://console.x.ai>.
2. Add a payment method.
3. Go to **API Keys**, click **Create API Key**.

Models: `grok-4`, `grok-3`, `grok-3-mini`. Env var: `XAI_API_KEY`.

### Open-source models via inference providers

#### OpenRouter (recommended starter)

One key works across many providers. Some models are free.

1. Sign up at <https://openrouter.ai/keys>.
2. Click **Create Key**, copy the value (`sk-or-...`).
3. Optionally add credits in **Settings → Credits** for paid models.

Models: `qwen/qwen-2.5-72b-instruct`, `meta-llama/llama-3.3-70b-instruct`,
`deepseek/deepseek-chat`, plus many more at <https://openrouter.ai/models>.
Env var: `OPENROUTER_API_KEY`.

#### Together AI

1. Sign up at <https://api.together.xyz>.
2. Click avatar → **API Keys** → **Create API Key**.

Models: `Qwen/Qwen2.5-72B-Instruct-Turbo`, `meta-llama/Llama-3.3-70B-Instruct-Turbo`,
`mistralai/Mixtral-8x7B-Instruct-v0.1`. Env var: `TOGETHER_API_KEY`.

#### Fireworks AI

1. Sign up at <https://fireworks.ai>.
2. Go to dashboard → **API Keys** → create one.

Models: `accounts/fireworks/models/qwen2p5-72b-instruct`,
`accounts/fireworks/models/llama-v3p3-70b-instruct`. Env var: `FIREWORKS_API_KEY`.

#### Groq

Generous free tier, very fast inference.

1. Sign up at <https://console.groq.com>.
2. Go to **API Keys** → **Create API Key**.

Models: `llama-3.3-70b-versatile`, `qwen-2.5-32b`, `mixtral-8x7b-32768`.
Env var: `GROQ_API_KEY`.

#### HuggingFace Inference Providers

**Important:** must create a fine-grained token with the "Make calls to
Inference Providers" permission, or requests fail with 401.

1. Sign up at <https://huggingface.co>.
2. Go to **Settings → Access Tokens**.
3. Click **Create new token**, choose **Fine-grained**.
4. **Enable "Make calls to Inference Providers"**.
5. Click **Create token**, copy the value (`hf_...`).

Models: `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen2.5-72B-Instruct`,
`meta-llama/Llama-3.3-70B-Instruct`. Env var: `HUGGINGFACE_API_KEY` or `HF_TOKEN`.

#### Custom (your own OpenAI-compatible endpoint)

For self-hosted models (vLLM, Ollama, TGI) or HuggingFace Inference Endpoints.

You'll need: a base URL ending in `/v1`, a model name, and an API key
(any string for Ollama).

This is what you'll use for a fine-tuned `mainger-qwen` model. No code
changes required.

### Quick comparison

| Provider     | Free tier     | Best for                       |
|--------------|---------------|--------------------------------|
| Anthropic    | No            | Final/paper-quality runs        |
| OpenAI       | No            | Default closed model            |
| Google       | Yes           | Cost-sensitive closed model     |
| xAI          | Limited       | Long-context tasks              |
| OpenRouter   | Some models   | Trying many open-source models  |
| Together     | Free credits  | Cheap reliable open-source      |
| Fireworks    | Free credits  | Fast open-source                |
| Groq         | Generous      | Speed                           |
| HuggingFace  | Yes (limited) | Multi-provider routing          |
| Custom       | N/A           | Self-hosted or fine-tuned       |

---

## 4. Configure

### 4.1 `.env` (your API keys)

```bash
cp .env.example .env                # macOS/Linux
copy .env.example .env              # Windows
```

Edit `.env` and uncomment the line for your chosen vendor:

```
ANTHROPIC_API_KEY=sk-ant-api03-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AIza...
XAI_API_KEY=xai-...
OPENROUTER_API_KEY=sk-or-v1-...
TOGETHER_API_KEY=...
FIREWORKS_API_KEY=...
GROQ_API_KEY=...
HUGGINGFACE_API_KEY=hf_...
```

`.env` is in `.gitignore` by default; never commit it.

### 4.2 `config.yaml` (default vendor and model)

```yaml
vendor: anthropic
model: claude-opus-4-7
max_tokens: 4096
temperature: 0.0          # keep low for reproducibility
max_tool_iterations: 8
```

Vendor names: `anthropic`, `openai`, `gemini`, `xai`, `together`,
`fireworks`, `openrouter`, `groq`, `huggingface`, `custom`.

These are defaults; both the CLI flags and the web UI dropdowns let you
override per run.

---

## 5. Use the command-line tool

### 5.1 Basic invocation

```bash
python agent.py \
    --input examples/partial_example.csv \
    --external-coef examples/external_coef.csv \
    --regime partial \
    --out-dir runs/demo
```

Windows PowerShell:
```powershell
python agent.py --input examples\partial_example.csv --external-coef examples\external_coef.csv --regime partial --out-dir runs\demo
```

### 5.2 All flags

| Flag | Required | Description |
|------|----------|-------------|
| `--input PATH`            | yes  | Internal individual-data file (CSV or Parquet). First column is Y. |
| `--external-coef PATH`    | yes  | External coefficients file. Two columns: `variable, estimate`. |
| `--external-sigma PATH`   | no   | External Σ₂ matrix; full regime only. |
| `--reference-sigma PATH`  | no   | Reference Σ panel; restricted regime only. |
| `--sigma2-int FLOAT`      | no   | Internal error variance estimate. |
| `--sigma2-ext FLOAT`      | no   | External error variance estimate. |
| `--n-ext INT`             | no   | External sample size. |
| `--regime {full,partial,restricted}` | no | Optional hint; the agent calls `detect_regime` regardless. |
| `--vendor NAME`           | no   | Override `config.yaml` vendor. |
| `--model NAME`            | no   | Override `config.yaml` model. |
| `--base-url URL`          | no   | Custom OpenAI-compatible endpoint URL. |
| `--config FILE`           | no   | Use a different config file. |
| `--out-dir DIR`           | no   | Where to write outputs (default: `runs/latest`). |
| `--message TEXT`          | no   | Override the default user message. |

### 5.3 Examples

**Full regime:**
```bash
python agent.py \
    --input my_data/internal.csv \
    --external-coef my_data/external_coefs.csv \
    --external-sigma my_data/external_sigma.csv \
    --sigma2-ext 0.85 --n-ext 8000 \
    --regime full \
    --out-dir runs/full_run
```

**Restricted regime:**
```bash
python agent.py \
    --input my_data/marginal_only.csv \
    --external-coef my_data/external_coefs.csv \
    --reference-sigma my_data/reference_panel.csv \
    --regime restricted \
    --out-dir runs/restricted_run
```

**Different vendor without editing config.yaml:**
```bash
python agent.py \
    --vendor openrouter \
    --model qwen/qwen-2.5-72b-instruct \
    --input examples/partial_example.csv \
    --external-coef examples/external_coef.csv \
    --regime partial \
    --out-dir runs/qwen_test
```

**Self-hosted model:**
```bash
python agent.py \
    --vendor custom \
    --base-url http://localhost:8000/v1 \
    --model my-finetuned-mainger-qwen \
    --input examples/partial_example.csv \
    --external-coef examples/external_coef.csv \
    --regime partial \
    --out-dir runs/local_test
```

---

## 6. Use the web UI

> **Single-user local use only.** The server stores session state in memory and accepts API keys via the form. Do not expose it beyond `127.0.0.1` without changing the auth model.

Better experience for interactive analysis: multi-turn conversation,
follow-up questions, mid-session parameter updates.

### 6.1 Start the server

```bash
python server.py
```

Open <http://localhost:8000>. Server runs locally; nothing is hosted externally.

### 6.2 The setup form (first analysis)

Five sections:

1. **Required inputs**: internal data file, external coefficients file.
2. **Optional — regime-specific**: external Σ₂ (full) or reference Σ (restricted).
3. **Optional — hints & parameters**: regime hint, `n_ext`, `sigma2_int`, `sigma2_ext`, **CV seed** (default 548).
4. **LLM**: vendor dropdown (closed and open-source families), model name (typeable + autocomplete), API key, optional Base URL.
5. **Initial message**: defaults to a generic analysis request.

The vendor dropdown groups closed and open-source. Click **Start session** to run.

### 6.3 The chat thread (after setup)

After submission the form is replaced by:

- **Session banner** at the top showing current metadata (`n_int`, `n_ext`, `p`, etc.) and the CV seed.
- **Message thread** with user bubbles (right) and agent bubbles (left).
- **Agent bubble contents**: text response, collapsible tool-call trace, three artifact tabs (Report / Code / Explanation), download buttons.

Math notation renders via KaTeX. R code is syntax-highlighted.

### 6.4 Follow-up questions

The composer at the bottom has:

- A textarea for the message
- **⚙ gear**: toggles a panel with `n_ext`, `sigma2_int`, `sigma2_ext`, `cv_seed` inputs
- **📎 paperclip**: attach files mid-conversation, with a role dropdown (Internal / External coefs / External Σ / Reference Σ)
- **Send** (or Ctrl/Cmd+Enter)

Examples:
- "Why did you pick that value of eta?" → agent answers in plain prose, no tools called.
- "Re-run with eta=0.05" → agent re-fits and produces new artifacts.
- Click 📎, attach a new external coefs file → agent updates the session and re-runs.

### 6.5 Updating mid-session

Numeric parameters (`n_ext`, `sigma2_int`, `sigma2_ext`) and the CV seed:
gear panel → enter values → send. File replacements: paperclip → pick file →
set role → send.

### 6.6 Switching vendors mid-experiment

Vendor and model are locked at session start. Click **New session** in the
banner to start over with a different vendor.

---

## 7. Output files

Every run (CLI or web) writes the same five files to its session directory:

| File | Contents |
|------|----------|
| `report.md` | Final integration report (markdown) |
| `analysis.R` | Runnable R script reproducing the analysis |
| `explanation.md` | Plain-language summary |
| `trace.json` | Full audit trail of every tool call |
| `final.json` | LLM's raw final response before rendering |

Plus internal files (`session.rds`, `session.json`, `_persist_session.R`,
and `chat_log.json` for web sessions).

You can verify the analysis externally:

```bash
Rscript runs/<session-id>/analysis.R
```

The coefficients should match those in `report.md`. If they don't, the
LLM hallucinated; check `trace.json` to see what actually came back from
the tools.

---

## 8. Choosing a model

**For development:** cheap and fast. `claude-haiku-4-5-20251001`,
`gpt-4o-mini`, `gemini-2.5-flash`, or anything on Groq's free tier.

**For final / paper-quality runs:** `claude-opus-4-7`, `gpt-4o`, or
`gemini-2.5-pro`. Stronger models follow the structured-output format
much more reliably; you'll see fewer parser failures and more consistent
artifacts.

**Cost** (approximate, verify at provider sites):

- Cheapest: Groq free tier, OpenRouter free models, Gemini Flash (~free to a few cents per million tokens)
- Mid: GPT-4o-mini, Claude Haiku, hosted Qwen/Llama (~10–30 cents)
- High: GPT-4o, Claude Sonnet (~$3–5)
- Highest: Claude Opus, Gemini 2.5 Pro (~$15)

A typical run is 5,000–20,000 tokens, so even Claude Opus is well under
$0.50 per analysis.

---

## 9. Troubleshooting

### "ANTHROPIC_API_KEY not provided" (or equivalent)

Your `.env` isn't being read or the wrong key is set for your vendor. Check:
1. `.env` is in the project root (same folder as `agent.py`).
2. The line for your vendor is uncommented.
3. The vendor in `config.yaml` matches the env var you set.

### "Failed to persist session as RDS"

The R bridge couldn't convert your data. Read the `stdout` and `stderr` in
the error message. Most common cause: a column expected to be numeric
contains non-numeric values (empty strings, dates, "N/A").

### Output appears as raw text instead of artifact tabs

The LLM emitted artifacts in a format the parser doesn't recognize. Two fixes:
1. **Switch to a stronger model** (most common solution).
2. Open `final.json` to see what the LLM emitted.

### HuggingFace 401: "Invalid username or password"

Your token doesn't have the right permissions. HuggingFace tokens must be
**fine-grained** with **"Make calls to Inference Providers"** explicitly
enabled. The default "Read" token is not enough.

### Server starts but page is blank / 404

The frontend isn't where the server expects it. Verify `web/index.html`
exists relative to the project root. If not, place it there and restart.

### Greek letters appear as `Î·` instead of `η`

A file is being read or written with the wrong encoding. Open `skill.md`
in VS Code; if the bottom-right shows "Windows 1252," click it and
re-save as UTF-8.

### "Hit max_tool_iterations without final answer"

The LLM is looping. Either bump `max_tool_iterations` in `config.yaml` to
~16, or switch to a stronger model.

### Caveat says "external sample size was not provided" when it was

Skill prompt issue with older versions; the current version (described
in this tutorial) explicitly tells the LLM to check session metadata
before claiming a field is missing.

### CV picks η ≈ 0 and integrated equals internal

Genuine math, not a bug. CV picked the internal-only solution, which can
happen when external bias is large relative to internal noise. The agent
should report this honestly. If you don't expect it, examine your
external coefficients for systematic bias.

---

## 10. Privacy and security

This section is short by design. For the longer treatment, see the
companion document `reproducibility_privacy.md`.

**What stays local under all configurations:**
- Individual-level data ($\bm{X}_1$, $\bm{Y}_1$).
- API keys (held in process memory; not logged or written to disk).

**What may flow to your chosen LLM provider:**
- Session metadata (sample sizes, predictor names, regime).
- User messages.
- Tool results returned to the agent (coefficients, η values, MSE estimates).

**To keep everything local:** use the **Custom** vendor pointing at a
self-hosted model (vLLM, Ollama, or the distributed fine-tuned Qwen-7B).
No information transits any external service in this configuration.

**The server binds to 127.0.0.1.** Do not expose it beyond localhost
without changing the auth model.

---

## 11. What the agent does, step by step

For transparency, here is the workflow the agent executes for every run:

1. Read your data, write a session file (`session.rds`).
2. Send the user message and session metadata to the LLM with the skill
   prompt and tool specifications.
3. **The LLM calls `detect_regime`** with metadata about which inputs
   you provided. The R bridge classifies the regime (full / partial / restricted).
4. **The LLM calls `compute_eta_bound`** with the detected regime. The R
   bridge uses your installed `mainger` package to compute $\eta^\star$.
5. **(Full regime only)** **The LLM calls `check_concordance`** to evaluate
   whether full sharing is theoretically expected to outperform partial.
6. **The LLM calls `fit_integrated_estimator`** with a chosen $\eta$. The
   R bridge returns coefficients, MSE estimates, and diagnostics.
7. **The LLM produces three strings** (report, code, explanation) drawing
   only on the numbers from the tool output.
8. **The agent writes** the three artifacts plus the audit trail.

Every numerical claim in the report comes from a tool call; the LLM does
not perform calculations itself. This is what lets you trust the report
numbers and verify them against `trace.json`.

---

## Appendix: extension points

- **Edit `skill.md`** to tune the agent's tone or add domain-specific
  guidance.
- **Edit `templates/*.j2`** to adjust report / code / explanation
  structure (referenced from the prompt as the expected structure).
- **Add a new tool**: add an entry to `TOOL_SPECS` in `tools.py` and
  implement it in `r_helpers/run_mainger.R`.
- **Add a new vendor**: add a new client class in `llm_client.py` (only
  needed for vendors with a non-OpenAI-compatible schema; anything
  OpenAI-compatible is already supported via the `Custom` vendor).
- **Deploy your own fine-tuned model**: upload to HuggingFace, deploy as
  an Inference Endpoint, point the `Custom` vendor at the endpoint URL.
