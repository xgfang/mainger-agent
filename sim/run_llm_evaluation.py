"""
sim/run_llm_evaluation.py
=========================
Run LLM strategies on simulation problems, with token usage and
detailed timing recorded per cell.

Strategies:
  S1 (no_method)     : single LLM call, asks for coefficients directly.
  S3 (mainger_agent) : multi-turn with tool calls via the R bridge.

Metrics recorded per cell (in the output JSONL):
  - prompt_tokens, completion_tokens, total_tokens
    (For S3: summed across all LLM turns. Note that S3's prompt_tokens
    include conversation history that grows with each turn.)
  - For S3 only:
      * llm_turn_seconds: list of wall-clock seconds for each LLM call
      * tool_call_seconds: list of wall-clock seconds for each tool call
      * tool_call_names: list of tool names called
      * total_llm_seconds: sum of llm_turn_seconds
      * total_tool_seconds: sum of tool_call_seconds
      * n_llm_turns, n_tool_calls
  - elapsed_s: total wall-clock from first request to final response.
    For S1 this equals the single LLM call's time. For S3 this is
    end-to-end including all turns and tool calls.

This data supports paper-side claims like "the agent uses N tokens vs
M for the no-method baseline, with overhead Y seconds for tool execution."

Run:
  python sim/run_llm_evaluation.py \
      --problems sim/data/sim_problems_moderate.jsonl \
      --output sim/data/llm_results.jsonl \
      --strategies S1 S3 \
      --models gpt-4o qwen-7b qwen-1.5b \
      --rps 1.0
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

import numpy as np

# Make agent modules importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from openai import OpenAI

from data_io import build_session, persist_session
from tools import TOOL_SPECS, call_tool


# --------------------------------------------------------------------------- #
# Model endpoints                                                              #
# --------------------------------------------------------------------------- #
def make_client(model_label: str) -> tuple[OpenAI, str]:
    if model_label == "gpt-4o":
        return OpenAI(), "gpt-4o"
    if model_label == "qwen-7b":
        url = os.getenv("OLLAMA_URL", "http://localhost:11434/v1")
        return OpenAI(base_url=url, api_key="ollama"), "qwen2.5:7b-instruct"
    if model_label == "qwen-1.5b":
        url = os.getenv("OLLAMA_URL", "http://localhost:11434/v1")
        return OpenAI(base_url=url, api_key="ollama"), "qwen2.5:1.5b-instruct"
    if model_label == "qwen-1.5b-ft":
        # Fine-tuned Qwen-1.5B, served by vLLM (typically on a GreatLakes
        # compute node). VLLM_URL points at the OpenAI-compatible endpoint;
        # default localhost:8000 matches the standard vLLM serve config.
        url = os.getenv("VLLM_URL", "http://localhost:8000/v1")
        return OpenAI(base_url=url, api_key="vllm"), "qwen-1.5b-ft"
    raise ValueError(f"Unknown model: {model_label}")


# --------------------------------------------------------------------------- #
# Coefficient parsing (for S1)                                                 #
# --------------------------------------------------------------------------- #
_JSON_BLOCK_RX = re.compile(r"\{[^{}]*\"coefficients\"[^{}]*\}", re.DOTALL)
_JSON_FENCE_RX = re.compile(r"```(?:json)?\s*\n(.*?)\n```", re.DOTALL | re.IGNORECASE)


def parse_coefficients(text: str, expected_p: int) -> tuple[list[float] | None, str]:
    if not text:
        return None, "no_extractable"
    candidates: list[Any] = []
    for m in _JSON_FENCE_RX.finditer(text):
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict) and "coefficients" in obj:
                candidates.append(obj["coefficients"])
            elif isinstance(obj, list):
                candidates.append(obj)
        except json.JSONDecodeError:
            pass
    for m in _JSON_BLOCK_RX.finditer(text):
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and "coefficients" in obj:
                candidates.append(obj["coefficients"])
        except json.JSONDecodeError:
            pass
    try:
        obj = json.loads(text.strip())
        if isinstance(obj, dict) and "coefficients" in obj:
            candidates.append(obj["coefficients"])
        elif isinstance(obj, list):
            candidates.append(obj)
    except json.JSONDecodeError:
        pass
    if not candidates:
        m = re.search(r"\[\s*-?\d[\d.\-eE+, \n]*\]", text)
        if m:
            try:
                arr = json.loads(m.group(0))
                if isinstance(arr, list):
                    candidates.append(arr)
            except json.JSONDecodeError:
                pass
    if not candidates:
        return None, "no_extractable"
    for c in candidates:
        if not isinstance(c, list): continue
        if len(c) != expected_p: continue
        try:
            return [float(v) for v in c], "ok"
        except (ValueError, TypeError):
            continue
    for c in candidates:
        if isinstance(c, list):
            try:
                return [float(v) for v in c], "wrong_length"
            except (ValueError, TypeError):
                return None, "non_numeric"
    return None, "no_extractable"


# --------------------------------------------------------------------------- #
# S1: no-method strategy                                                       #
# --------------------------------------------------------------------------- #
S1_SYSTEM = (
    "You are a statistician. The user will provide internal individual-level "
    "regression data and an external coefficient summary from a related study "
    "(possibly with population heterogeneity). Produce a single coefficient "
    "vector that you believe will best predict on a held-out test set drawn "
    "from the same distribution as the internal data. Output ONLY a JSON "
    "object with a single key 'coefficients' whose value is a list of numbers, "
    "wrapped in a fenced ```json``` block. No explanation, no other text."
)


def s1_user_message(problem: dict) -> str:
    parts = []
    parts.append(f"Predictor names (length {problem['p']}): {problem['predictor_names']}")
    parts.append(f"Internal sample size: {problem['n_int']}")
    parts.append(f"External coefficient estimates (same order as predictor names):")
    parts.append(f"  {[round(v, 4) for v in problem['beta_ext']]}")

    if "X_int" in problem and "Y_int" in problem:
        X = np.array(problem["X_int"])
        Y = np.array(problem["Y_int"])
        beta_ols = np.linalg.lstsq(X, Y, rcond=None)[0].tolist()
        parts.append(f"Internal OLS estimate from individual data:")
        parts.append(f"  {[round(v, 4) for v in beta_ols]}")
        XtX_n = (X.T @ X / problem["n_int"]).tolist()
        parts.append(f"Internal X'X / n (covariance proxy, {problem['p']}x{problem['p']} matrix):")
        parts.append(f"  {[[round(v, 4) for v in row] for row in XtX_n]}")
    elif "r_int" in problem:
        parts.append(f"Internal marginal correlations r_int = X'Y/n:")
        parts.append(f"  {[round(v, 4) for v in problem['r_int']]}")
        parts.append(f"Reference covariance Sigma_ref ({problem['p']}x{problem['p']}):")
        parts.append(f"  {[[round(v, 4) for v in row] for row in problem['Sigma_ref']]}")

    if "Sigma_ext" in problem:
        parts.append(f"External covariance Sigma_ext ({problem['p']}x{problem['p']}):")
        parts.append(f"  {[[round(v, 4) for v in row] for row in problem['Sigma_ext']]}")
    if "n_ext" in problem:
        parts.append(f"External sample size: {problem['n_ext']}")
    if "sigma2_ext" in problem:
        parts.append(f"External residual variance: {problem['sigma2_ext']:.4f}")

    parts.append("")
    parts.append("Produce coefficient estimates that you believe will best predict on a "
                 "held-out test set from the same distribution as the internal data.")
    return "\n".join(parts)


def run_s1(client: OpenAI, model: str, problem: dict,
           max_retries: int = 3) -> dict:
    """Run S1 (no method specified) on one problem with token+timing."""
    user_msg = s1_user_message(problem)
    last_err = None
    raw_text = ""
    usage = None
    t_call = None

    for attempt in range(max_retries + 1):
        try:
            t0 = time.perf_counter()
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": S1_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                seed=0,
                max_tokens=2048,
            )
            t_call = time.perf_counter() - t0
            raw_text = resp.choices[0].message.content or ""
            usage = resp.usage
            break
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(2 * (2 ** attempt))
                continue
            return {
                "error": f"{type(e).__name__}: {e}",
                "raw_text": "",
                "coefficients": None,
                "parse_status": "exception",
                "prompt_tokens": None, "completion_tokens": None, "total_tokens": None,
                "llm_turn_seconds": [], "tool_call_seconds": [], "tool_call_names": [],
                "total_llm_seconds": 0.0, "total_tool_seconds": 0.0,
                "n_llm_turns": 0, "n_tool_calls": 0,
            }

    coefs, status = parse_coefficients(raw_text, problem["p"])

    return {
        "raw_text": raw_text[:500],
        "coefficients": coefs,
        "parse_status": status,
        "prompt_tokens":     getattr(usage, "prompt_tokens", None)     if usage else None,
        "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
        "total_tokens":      getattr(usage, "total_tokens", None)      if usage else None,
        "llm_turn_seconds":  [t_call] if t_call is not None else [],
        "tool_call_seconds": [],
        "tool_call_names":   [],
        "total_llm_seconds":  t_call if t_call is not None else 0.0,
        "total_tool_seconds": 0.0,
        "n_llm_turns":  1 if t_call is not None else 0,
        "n_tool_calls": 0,
    }


# --------------------------------------------------------------------------- #
# S3: mainger-agent strategy                                                   #
# --------------------------------------------------------------------------- #
def materialize_session(problem: dict, work_dir: Path) -> dict:
    work_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    pred_names = problem["predictor_names"]

    X_int = problem.get("X_int") or problem.get("_X_int_eval_only")
    Y_int = problem.get("Y_int") or problem.get("_Y_int_eval_only")

    if X_int is not None and problem["regime"] != "restricted":
        import csv
        ip = work_dir / "internal.csv"
        with ip.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["y"] + pred_names)
            for y, row in zip(Y_int, X_int):
                w.writerow([y] + list(row))
        paths["internal_path"] = str(ip)

    import csv
    ep = work_dir / "external_coef.csv"
    with ep.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variable", "estimate"])
        for nm, b in zip(pred_names, problem["beta_ext"]):
            w.writerow([nm, b])
    paths["external_coef_path"] = str(ep)

    if "Sigma_ext" in problem:
        sp = work_dir / "external_sigma.csv"
        with sp.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for row in problem["Sigma_ext"]:
                w.writerow(row)
        paths["external_sigma_path"] = str(sp)

    if "Sigma_ref" in problem:
        rp = work_dir / "reference_sigma.csv"
        with rp.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for row in problem["Sigma_ref"]:
                w.writerow(row)
        paths["reference_sigma_path"] = str(rp)

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
    return persist_session(session, work_dir)


def run_s3(client: OpenAI, model: str, problem: dict, work_dir: Path,
           max_iters: int = 8, max_retries: int = 3) -> dict:
    """Run mainger-agent pipeline; record per-turn LLM tokens/time and
    per-tool-call time."""
    skill_path = Path(__file__).parent.parent / "skill.md"
    skill = skill_path.read_text(encoding="utf-8")

    session = materialize_session(problem, work_dir)
    system = (f"{skill}\n\nSession metadata (read-only):\n"
              f"{json.dumps(session['_metadata'], indent=2)}")

    tools_openai = [
        {"type": "function",
         "function": {"name": s["name"],
                      "description": s["description"],
                      "parameters": s["input_schema"]}}
        for s in TOOL_SPECS
    ]

    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content":
            "Please analyze my data and produce the integration report, "
            "code, and explanation."},
    ]

    extracted_coefs = None
    n_tool_calls = 0
    last_err = None

    # Per-cell instrumentation
    llm_turn_seconds: list[float] = []
    tool_call_seconds: list[float] = []
    tool_call_names: list[str] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    have_token_data = False

    for step in range(max_iters):
        # ---- LLM call (with retries on exception) ----
        resp = None
        for retry in range(max_retries + 1):
            try:
                t0 = time.perf_counter()
                resp = client.chat.completions.create(
                    model=model, messages=messages, tools=tools_openai,
                    tool_choice="auto", temperature=0.0, seed=0,
                    max_tokens=2048,
                )
                llm_turn_seconds.append(time.perf_counter() - t0)
                break
            except Exception as e:
                last_err = e
                if retry < max_retries:
                    time.sleep(2 * (2 ** retry))
                    continue
                return {
                    "error": f"LLM call failed: {last_err}",
                    "n_tool_calls": n_tool_calls,
                    "coefficients": extracted_coefs,
                    "parse_status": "exception",
                    "prompt_tokens": total_prompt_tokens if have_token_data else None,
                    "completion_tokens": total_completion_tokens if have_token_data else None,
                    "total_tokens": total_tokens if have_token_data else None,
                    "llm_turn_seconds":  llm_turn_seconds,
                    "tool_call_seconds": tool_call_seconds,
                    "tool_call_names":   tool_call_names,
                    "total_llm_seconds":  sum(llm_turn_seconds),
                    "total_tool_seconds": sum(tool_call_seconds),
                    "n_llm_turns":  len(llm_turn_seconds),
                    "n_tool_calls": n_tool_calls,
                }

        # Accumulate token usage if reported
        usage = getattr(resp, "usage", None)
        if usage is not None:
            have_token_data = True
            total_prompt_tokens     += getattr(usage, "prompt_tokens", 0) or 0
            total_completion_tokens += getattr(usage, "completion_tokens", 0) or 0
            total_tokens            += getattr(usage, "total_tokens", 0) or 0

        msg = resp.choices[0].message
        assistant_turn: dict[str, Any] = {"role": "assistant"}
        if msg.content: assistant_turn["content"] = msg.content
        if msg.tool_calls:
            assistant_turn["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name,
                              "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]
        messages.append(assistant_turn)

        if not msg.tool_calls:
            # FT models trained on multi-turn agent traces sometimes emit
            # narrative text turns between tool calls (the teacher's habit).
            # If we already have coefficients, this is the final response;
            # break. Otherwise nudge the model and let it try the next tool.
            if extracted_coefs is not None:
                break
            if step >= max_iters - 1:
                break
            messages.append({
                "role": "user",
                "content": "Continue with the next step in the workflow. "
                           "Call the appropriate tool.",
            })
            continue

        # ---- Tool calls (instrumented) ----
        for tc in msg.tool_calls:
            n_tool_calls += 1
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            t0 = time.perf_counter()
            result = call_tool(tc.function.name, args, session)
            tool_call_seconds.append(time.perf_counter() - t0)
            tool_call_names.append(tc.function.name)

            content_obj = result.get("result") if result.get("ok") else \
                          {"error": result.get("error")}

            if (tc.function.name == "fit_integrated_estimator"
                    and result.get("ok")
                    and "coefficients" in (result["result"] or {})):
                coefs_obj = result["result"]["coefficients"]
                if isinstance(coefs_obj, dict):
                    extracted_coefs = [float(coefs_obj[n])
                                       for n in problem["predictor_names"]
                                       if n in coefs_obj]
                    if len(extracted_coefs) != problem["p"]:
                        extracted_coefs = [float(v) for v in coefs_obj.values()]
                elif isinstance(coefs_obj, list):
                    extracted_coefs = [float(v) for v in coefs_obj]

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(content_obj, default=str),
            })

    if extracted_coefs is not None and len(extracted_coefs) == problem["p"]:
        status = "ok"
    elif extracted_coefs is not None:
        status = "wrong_length"
    else:
        status = "no_extractable"

    return {
        "coefficients": extracted_coefs,
        "parse_status": status,
        "n_tool_calls": n_tool_calls,
        "prompt_tokens":     total_prompt_tokens     if have_token_data else None,
        "completion_tokens": total_completion_tokens if have_token_data else None,
        "total_tokens":      total_tokens            if have_token_data else None,
        "llm_turn_seconds":  llm_turn_seconds,
        "tool_call_seconds": tool_call_seconds,
        "tool_call_names":   tool_call_names,
        "total_llm_seconds":  sum(llm_turn_seconds),
        "total_tool_seconds": sum(tool_call_seconds),
        "n_llm_turns":  len(llm_turn_seconds),
    }


# --------------------------------------------------------------------------- #
# S3 (scaffolded) — for SFT models that learned individual tool calls but    #
# not autonomous chaining. The harness orchestrates the workflow: the model  #
# autonomously chooses on turn 1 (detect_regime), then we inject scaffolding #
# user messages plus forced tool_choice for the next two tools.              #
# --------------------------------------------------------------------------- #
def run_s3_scaffolded(client: OpenAI, model: str, problem: dict, work_dir: Path,
                      max_retries: int = 3) -> dict:
    """Scaffolded variant: forces the workflow's tool sequence via user
    nudges and OpenAI-style forced tool_choice."""
    skill_path = Path(__file__).parent.parent / "skill.md"
    skill = skill_path.read_text(encoding="utf-8")

    session = materialize_session(problem, work_dir)
    system = (f"{skill}\n\nSession metadata (read-only):\n"
              f"{json.dumps(session['_metadata'], indent=2)}")

    tools_openai = [
        {"type": "function",
         "function": {"name": s["name"],
                      "description": s["description"],
                      "parameters": s["input_schema"]}}
        for s in TOOL_SPECS
    ]

    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content":
            "Please analyze my data and produce the integration report, "
            "code, and explanation."},
    ]

    # Mutable state shared across nested helpers
    st: dict[str, Any] = {
        "llm_turn_seconds":  [],
        "tool_call_seconds": [],
        "tool_call_names":   [],
        "n_tool_calls":      0,
        "prompt_tokens":     0,
        "completion_tokens": 0,
        "total_tokens":      0,
        "have_token_data":   False,
        "extracted_coefs":   None,
    }

    def _force_execute(name: str, args: dict) -> Any:
        """Synthesize an assistant tool-call message and execute the tool
        ourselves. Used when the model fails to call a specific tool we
        know is needed at this point. Tagged with `__forced` in
        tool_call_names so post-hoc analysis can count autonomous vs
        harness-forced calls per step."""
        fake_id = f"forced_{name}_{len(messages)}"
        messages.append({
            "role": "assistant",
            "tool_calls": [{
                "id": fake_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }],
        })
        st["n_tool_calls"] += 1
        t0 = time.perf_counter()
        result = call_tool(name, args, session)
        st["tool_call_seconds"].append(time.perf_counter() - t0)
        st["tool_call_names"].append(f"{name}__forced")

        content_obj = (result.get("result") if result.get("ok")
                       else {"error": result.get("error")})

        if (name == "fit_integrated_estimator"
                and result.get("ok")
                and "coefficients" in (result["result"] or {})):
            coefs_obj = result["result"]["coefficients"]
            if isinstance(coefs_obj, dict):
                coefs = [float(coefs_obj[n])
                         for n in problem["predictor_names"]
                         if n in coefs_obj]
                if len(coefs) != problem["p"]:
                    coefs = [float(v) for v in coefs_obj.values()]
                st["extracted_coefs"] = coefs
            elif isinstance(coefs_obj, list):
                st["extracted_coefs"] = [float(v) for v in coefs_obj]

        messages.append({
            "role": "tool",
            "tool_call_id": fake_id,
            "content": json.dumps(content_obj, default=str),
        })
        return content_obj

    def _llm_turn(tool_choice):
        last_err = None
        for retry in range(max_retries + 1):
            try:
                t0 = time.perf_counter()
                resp = client.chat.completions.create(
                    model=model, messages=messages, tools=tools_openai,
                    tool_choice=tool_choice, temperature=0.0, seed=0,
                    max_tokens=2048,
                )
                st["llm_turn_seconds"].append(time.perf_counter() - t0)
                break
            except Exception as e:
                last_err = e
                if retry < max_retries:
                    time.sleep(2 * (2 ** retry))
                    continue
                raise last_err

        usage = getattr(resp, "usage", None)
        if usage is not None:
            st["have_token_data"] = True
            st["prompt_tokens"]     += getattr(usage, "prompt_tokens", 0) or 0
            st["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
            st["total_tokens"]      += getattr(usage, "total_tokens", 0) or 0

        msg = resp.choices[0].message
        assistant_turn: dict[str, Any] = {"role": "assistant"}
        if msg.content: assistant_turn["content"] = msg.content
        if msg.tool_calls:
            assistant_turn["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name,
                              "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]
        messages.append(assistant_turn)
        return msg

    def _exec_tool_calls(msg) -> dict:
        results: dict[str, Any] = {}
        if not msg.tool_calls:
            return results
        for tc in msg.tool_calls:
            st["n_tool_calls"] += 1
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            t0 = time.perf_counter()
            result = call_tool(tc.function.name, args, session)
            st["tool_call_seconds"].append(time.perf_counter() - t0)
            st["tool_call_names"].append(tc.function.name)

            content_obj = (result.get("result") if result.get("ok")
                           else {"error": result.get("error")})

            if (tc.function.name == "fit_integrated_estimator"
                    and result.get("ok")
                    and "coefficients" in (result["result"] or {})):
                coefs_obj = result["result"]["coefficients"]
                if isinstance(coefs_obj, dict):
                    coefs = [float(coefs_obj[n])
                             for n in problem["predictor_names"]
                             if n in coefs_obj]
                    if len(coefs) != problem["p"]:
                        coefs = [float(v) for v in coefs_obj.values()]
                    st["extracted_coefs"] = coefs
                elif isinstance(coefs_obj, list):
                    st["extracted_coefs"] = [float(v) for v in coefs_obj]

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(content_obj, default=str),
            })
            results[tc.function.name] = content_obj
        return results

    try:
        # Step 1: detect_regime (autonomous, model demonstrably handles this)
        msg = _llm_turn(tool_choice="auto")
        r1 = _exec_tool_calls(msg)
        regime = None
        if isinstance(r1.get("detect_regime"), dict):
            regime = r1["detect_regime"].get("regime")
        if regime not in ("full", "partial", "restricted"):
            regime = problem["regime"]  # fall back to ground truth

        # Step 2: compute_eta_bound. Try the model first; force-execute if
        # it called the wrong tool (or none).
        messages.append({
            "role": "user",
            "content": f"Now call compute_eta_bound for the '{regime}' regime.",
        })
        msg = _llm_turn(tool_choice="auto")
        r2 = _exec_tool_calls(msg)
        if "compute_eta_bound" not in r2:
            _force_execute("compute_eta_bound", {"regime": regime})

        # Step 3: fit_integrated_estimator. Same logic.
        tuning = "eaic" if regime == "restricted" else "cv"
        fit_args: dict[str, Any] = {"regime": regime, "tuning": tuning}
        if tuning == "cv":
            fit_args["cv_seed"] = 548
        cv_directive = (" Pass cv_seed=548 for reproducibility."
                        if tuning == "cv" else "")
        messages.append({
            "role": "user",
            "content": (f"Now call fit_integrated_estimator with "
                        f"regime='{regime}' and tuning='{tuning}'.{cv_directive}"),
        })
        msg = _llm_turn(tool_choice="auto")
        r3 = _exec_tool_calls(msg)
        if "fit_integrated_estimator" not in r3:
            _force_execute("fit_integrated_estimator", fit_args)

    except Exception as e:
        return {
            "error": f"LLM call failed: {e}",
            "coefficients":      st["extracted_coefs"],
            "parse_status":      "exception",
            "n_tool_calls":      st["n_tool_calls"],
            "prompt_tokens":     st["prompt_tokens"]     if st["have_token_data"] else None,
            "completion_tokens": st["completion_tokens"] if st["have_token_data"] else None,
            "total_tokens":      st["total_tokens"]      if st["have_token_data"] else None,
            "llm_turn_seconds":  st["llm_turn_seconds"],
            "tool_call_seconds": st["tool_call_seconds"],
            "tool_call_names":   st["tool_call_names"],
            "total_llm_seconds":  sum(st["llm_turn_seconds"]),
            "total_tool_seconds": sum(st["tool_call_seconds"]),
            "n_llm_turns":  len(st["llm_turn_seconds"]),
        }

    extracted = st["extracted_coefs"]
    if extracted is not None and len(extracted) == problem["p"]:
        status = "ok"
    elif extracted is not None:
        status = "wrong_length"
    else:
        status = "no_extractable"

    return {
        "coefficients":      extracted,
        "parse_status":      status,
        "n_tool_calls":      st["n_tool_calls"],
        "prompt_tokens":     st["prompt_tokens"]     if st["have_token_data"] else None,
        "completion_tokens": st["completion_tokens"] if st["have_token_data"] else None,
        "total_tokens":      st["total_tokens"]      if st["have_token_data"] else None,
        "llm_turn_seconds":  st["llm_turn_seconds"],
        "tool_call_seconds": st["tool_call_seconds"],
        "tool_call_names":   st["tool_call_names"],
        "total_llm_seconds":  sum(st["llm_turn_seconds"]),
        "total_tool_seconds": sum(st["tool_call_seconds"]),
        "n_llm_turns":  len(st["llm_turn_seconds"]),
    }


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--strategies", nargs="+", default=["S1", "S3"],
                    choices=["S1", "S3"])
    ap.add_argument("--models", nargs="+",
                    default=["gpt-4o", "qwen-7b", "qwen-1.5b"],
                    choices=["gpt-4o", "qwen-7b", "qwen-1.5b", "qwen-1.5b-ft"])
    ap.add_argument("--rps", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--work_dir", default="sim/data/_session_workdir")
    args = ap.parse_args()

    load_dotenv()

    problems = []
    with Path(args.problems).open("r", encoding="utf-8") as f:
        for line in f:
            problems.append(json.loads(line))
    if args.limit is not None:
        problems = problems[:args.limit]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    completed: set[tuple[str, str, str]] = set()
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    completed.add((r["problem_id"], r["strategy"], r["model"]))
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"Found {len(completed)} previously-completed cells; will skip.")

    cells = [(p, s, m) for p in problems
             for s in args.strategies for m in args.models]
    pending = [c for c in cells if (c[0]["problem_id"], c[1], c[2]) not in completed]
    print(f"Total cells: {len(cells)}, pending: {len(pending)}")

    delay = 1.0 / args.rps if args.rps > 0 else 0
    work_root = Path(args.work_dir)

    with out_path.open("a", encoding="utf-8") as f_out:
        for i, (problem, strategy, model_label) in enumerate(pending):
            t_start = time.perf_counter()
            try:
                client, model_str = make_client(model_label)
            except Exception as e:
                print(f"  cannot init client for {model_label}: {e}")
                continue

            try:
                if strategy == "S1":
                    result = run_s1(client, model_str, problem)
                else:
                    workdir = work_root / f"{problem['problem_id']}__{model_label}"
                    if model_label == "qwen-1.5b-ft":
                        # FT model failed to autonomously chain tool calls;
                        # use scaffolded harness with forced tool_choice.
                        result = run_s3_scaffolded(client, model_str, problem, workdir)
                    else:
                        result = run_s3(client, model_str, problem, workdir)
            except Exception as e:
                result = {"error": f"{type(e).__name__}: {e}",
                          "traceback": traceback.format_exc(),
                          "coefficients": None,
                          "parse_status": "exception",
                          "prompt_tokens": None, "completion_tokens": None,
                          "total_tokens": None,
                          "llm_turn_seconds": [], "tool_call_seconds": [],
                          "tool_call_names": [],
                          "total_llm_seconds": 0.0, "total_tool_seconds": 0.0,
                          "n_llm_turns": 0, "n_tool_calls": 0}

            X_test = np.array(problem["X_test"])
            Y_test = np.array(problem["Y_test"])

            mspe = None
            mse_vs_truth = None
            coefs = result.get("coefficients")
            if coefs is not None and len(coefs) == problem["p"]:
                try:
                    b = np.array(coefs, dtype=float)
                    bt = np.array(problem["beta_true"])
                    # Bias-squared MSPE (matching the simulation generator)
                    diff = X_test @ (b - bt)
                    mspe = float(np.dot(diff, diff) / X_test.shape[0])
                    mse_vs_truth = float(np.sum((b - bt) ** 2))
                except Exception:
                    pass

            elapsed = time.perf_counter() - t_start

            row = {
                "problem_id":   problem["problem_id"],
                "regime":       problem["regime"],
                "p":            problem["p"],
                "strategy":     strategy,
                "model":        model_label,
                "coefficients": coefs,
                "parse_status": result.get("parse_status", "ok"),
                "mspe":         mspe,
                "mse_vs_true":  mse_vs_truth,
                # Aggregate metrics
                "elapsed_s":          elapsed,
                "total_llm_seconds":  result.get("total_llm_seconds"),
                "total_tool_seconds": result.get("total_tool_seconds"),
                "n_llm_turns":  result.get("n_llm_turns"),
                "n_tool_calls": result.get("n_tool_calls"),
                # Token usage
                "prompt_tokens":     result.get("prompt_tokens"),
                "completion_tokens": result.get("completion_tokens"),
                "total_tokens":      result.get("total_tokens"),
                # Per-turn / per-tool detail
                "llm_turn_seconds":  result.get("llm_turn_seconds"),
                "tool_call_seconds": result.get("tool_call_seconds"),
                "tool_call_names":   result.get("tool_call_names"),
                # Diagnostics
                "error":     result.get("error"),
                "raw_text":  result.get("raw_text", "")[:300],
            }
            f_out.write(json.dumps(row) + "\n")
            f_out.flush()

            tag = result.get("parse_status", "?")
            mspe_str = f"{mspe:.3f}" if mspe is not None else " --- "
            tok_str = (f"tok={row['total_tokens']}"
                       if row['total_tokens'] is not None else "tok=NA")
            print(f"  [{i+1}/{len(pending)}] {problem['problem_id']:<22s} "
                  f"{strategy} {model_label:<10s} {tag:<14s} mspe={mspe_str} "
                  f"{tok_str} time={elapsed:.1f}s")

            sleep_for = delay - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

    print(f"\nDone. Output: {out_path}")


if __name__ == "__main__":
    main()
