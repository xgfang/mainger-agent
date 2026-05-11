"""
sft/collect_teacher_traces.py
=============================
Phase 2 of the SFT pipeline: collect teacher traces from a frontier model.

Updates from previous version:
  - Adds retry-on-429 with exponential backoff. Token-per-minute rate
    limits no longer kill problems; the script just waits and retries.
  - Logs retry events so you can see when throttling is happening.
  - Other behavior unchanged: resumable, cost-tracked, failure-tolerant.

Run from the agent project root:
  python sft/collect_teacher_traces.py \
      --problems sft/data/problems_pilot.jsonl \
      --output   sft/data/teacher_traces_pilot.jsonl \
      --model    gpt-4o \
      --rps      0.5 \
      --max_retries 5
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# Make the agent's modules importable when running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIError, APITimeoutError

from data_io import build_session, persist_session
from tools import TOOL_SPECS, call_tool


# --------------------------------------------------------------------------- #
# Cost tracking (approximate; verify on OpenAI's pricing page)                  #
# --------------------------------------------------------------------------- #
PRICING_PER_1M_TOKENS = {
    "gpt-4o":      {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
}


def estimate_cost(model: str, n_input_tokens: int, n_output_tokens: int) -> float:
    p = PRICING_PER_1M_TOKENS.get(model)
    if p is None:
        return 0.0
    return (n_input_tokens / 1e6) * p["input"] + (n_output_tokens / 1e6) * p["output"]


# --------------------------------------------------------------------------- #
# Retry wrapper for OpenAI calls                                               #
# --------------------------------------------------------------------------- #
_RETRY_AFTER_RX = re.compile(r"try again in (\d+(?:\.\d+)?)\s*(s|ms)", re.IGNORECASE)


def _parse_retry_after(err_msg: str) -> float | None:
    """Extract OpenAI's suggested wait time from a 429 message, if present."""
    m = _RETRY_AFTER_RX.search(err_msg)
    if not m:
        return None
    val, unit = float(m.group(1)), m.group(2).lower()
    return val / 1000 if unit == "ms" else val


def call_openai_with_retry(
    client: OpenAI,
    *,
    model: str,
    messages: list,
    tools: list,
    max_retries: int = 5,
    base_backoff: float = 2.0,
    log_prefix: str = "",
):
    """Call client.chat.completions.create with retry on 429 / 5xx / timeout."""
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.0,
            )
        except RateLimitError as e:
            last_err = e
            err_msg = str(e)
            suggested = _parse_retry_after(err_msg)
            wait = suggested if suggested is not None else (base_backoff * (2 ** attempt))
            wait = wait + 0.5  # safety margin
            if attempt < max_retries:
                print(f"  {log_prefix}429 hit, attempt {attempt+1}/{max_retries+1}, "
                      f"sleeping {wait:.1f}s...")
                time.sleep(wait)
                continue
            raise
        except (APIError, APITimeoutError) as e:
            last_err = e
            wait = base_backoff * (2 ** attempt)
            if attempt < max_retries:
                print(f"  {log_prefix}{type(e).__name__}: {e}; "
                      f"attempt {attempt+1}/{max_retries+1}, sleeping {wait:.1f}s...")
                time.sleep(wait)
                continue
            raise
    raise last_err if last_err else RuntimeError("retry wrapper exited unexpectedly")


# --------------------------------------------------------------------------- #
# Convert a problem dict into a session                                        #
# --------------------------------------------------------------------------- #
def problem_to_session(problem: dict, work_dir: Path) -> dict:
    work_dir.mkdir(parents=True, exist_ok=True)
    pred_names = problem["predictor_names"]
    paths = {}

    if "X_int" in problem and "Y_int" in problem:
        import csv
        internal_path = work_dir / "internal.csv"
        with internal_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["y"] + pred_names)
            for y, row in zip(problem["Y_int"], problem["X_int"]):
                w.writerow([y] + list(row))
        paths["internal_path"] = str(internal_path)

    import csv
    ext_path = work_dir / "external_coef.csv"
    with ext_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variable", "estimate"])
        for nm, b in zip(pred_names, problem["beta_ext"]):
            w.writerow([nm, b])
    paths["external_coef_path"] = str(ext_path)

    if "Sigma_ext" in problem:
        sigma_path = work_dir / "external_sigma.csv"
        with sigma_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for row in problem["Sigma_ext"]:
                w.writerow(row)
        paths["external_sigma_path"] = str(sigma_path)

    if "Sigma_ref" in problem:
        ref_path = work_dir / "reference_sigma.csv"
        with ref_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for row in problem["Sigma_ref"]:
                w.writerow(row)
        paths["reference_sigma_path"] = str(ref_path)

    if "r_int" in problem:
        manual = {
            "r_int":           problem["r_int"],
            "predictor_names": problem.get("predictor_names"),
            "n_int":           problem.get("n_int"),
        }
    else:
        manual = None

    session = build_session(
        **paths,
        sigma2_ext=problem.get("sigma2_ext"),
        n_ext=problem.get("n_ext"),
        manual=manual,
    )
    session = persist_session(session, work_dir)
    return session


SYSTEM_PROMPT_PATH = Path(__file__).parent.parent / "skill.md"


def openai_tools_format(specs: list[dict]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": s["name"],
                "description": s["description"],
                "parameters": s["input_schema"],
            },
        }
        for s in specs
    ]


def run_one_problem(
    client: OpenAI,
    model: str,
    problem: dict,
    work_dir: Path,
    max_tool_iters: int = 8,
    max_retries: int = 5,
) -> dict:
    session = problem_to_session(problem, work_dir)
    skill = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")

    system_content = (
        f"{skill}\n\n"
        f"Session metadata (read-only):\n"
        f"{json.dumps(session['_metadata'], indent=2)}"
    )

    messages: list[dict] = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": problem["user_message"]},
    ]
    tools = openai_tools_format(TOOL_SPECS)

    n_input_tokens = 0
    n_output_tokens = 0
    log_prefix = f"[{problem['problem_id']}] "

    for step in range(max_tool_iters):
        resp = call_openai_with_retry(
            client, model=model, messages=messages, tools=tools,
            max_retries=max_retries, log_prefix=log_prefix,
        )

        if resp.usage is not None:
            n_input_tokens  += resp.usage.prompt_tokens
            n_output_tokens += resp.usage.completion_tokens

        msg = resp.choices[0].message
        assistant_turn: dict[str, Any] = {"role": "assistant"}
        if msg.content:
            assistant_turn["content"] = msg.content
        if msg.tool_calls:
            assistant_turn["tool_calls"] = [
                {
                    "id":   tc.id,
                    "type": "function",
                    "function": {
                        "name":      tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_turn)

        if not msg.tool_calls:
            break

        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            result = call_tool(tc.function.name, args, session)
            content_obj = result.get("result") if result.get("ok") else {"error": result.get("error")}
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(content_obj, default=str),
            })

    return {
        "problem_id": problem["problem_id"],
        "regime":     problem["regime"],
        "model":      model,
        "messages":   messages,
        "n_input_tokens":  n_input_tokens,
        "n_output_tokens": n_output_tokens,
        "estimated_cost":  estimate_cost(model, n_input_tokens, n_output_tokens),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="gpt-4o-mini",
                    choices=list(PRICING_PER_1M_TOKENS.keys()))
    ap.add_argument("--rps", type=float, default=2.0)
    ap.add_argument("--max_retries", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--work-dir", default="sft/data/_session_workdir")
    args = ap.parse_args()

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set in environment or .env file.")
        sys.exit(1)

    client = OpenAI()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: count only successful prior traces
    completed_ids: set[str] = set()
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if "error" not in rec:
                        completed_ids.add(rec["problem_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"Found {len(completed_ids)} previously-successful traces; "
              f"errors and missing problems will be (re)attempted.")

    problems = []
    with Path(args.problems).open("r", encoding="utf-8") as f:
        for line in f:
            problems.append(json.loads(line))
    if args.limit is not None:
        problems = problems[:args.limit]

    pending = [p for p in problems if p["problem_id"] not in completed_ids]
    print(f"Processing {len(pending)} problems with {args.model} at "
          f"{args.rps} rps (max_retries={args.max_retries}).")

    work_root = Path(args.work_dir)
    delay = 1.0 / args.rps if args.rps > 0 else 0
    total_cost = 0.0
    n_ok = 0
    n_fail = 0

    with out_path.open("a", encoding="utf-8") as out_f:
        for i, problem in enumerate(pending):
            t_start = time.time()
            problem_workdir = work_root / problem["problem_id"]
            try:
                trace = run_one_problem(
                    client, args.model, problem, problem_workdir,
                    max_retries=args.max_retries,
                )
                out_f.write(json.dumps(trace) + "\n")
                out_f.flush()
                total_cost += trace["estimated_cost"]
                n_ok += 1
                tag = "ok"
            except Exception as e:
                err_trace = {
                    "problem_id": problem["problem_id"],
                    "regime":     problem["regime"],
                    "model":      args.model,
                    "error":      str(e),
                    "traceback":  traceback.format_exc(),
                }
                out_f.write(json.dumps(err_trace) + "\n")
                out_f.flush()
                n_fail += 1
                tag = "ERROR"

            elapsed = time.time() - t_start
            print(f"  [{i+1}/{len(pending)}] {problem['problem_id']} "
                  f"({problem['regime']}) {tag} {elapsed:.1f}s "
                  f"cost ${total_cost:.3f}")

            sleep_for = delay - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

    print(f"\nDone. ok={n_ok}, errors={n_fail}, total estimated cost=${total_cost:.3f}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
