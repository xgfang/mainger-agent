"""
agent.py
--------
Two entry points:
  - run_agent(...)            : synchronous; returns final dict. Used by CLI.
  - run_agent_streaming(...)  : generator; yields event dicts as they happen.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterator

import yaml
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader

from data_io import build_session, persist_session
from llm_client import LLMResponse, make_client
from tools import TOOL_SPECS, ToolError, call_tool

ROOT = Path(__file__).parent
SKILL_PATH    = ROOT / "skill.md"
TEMPLATE_DIR  = ROOT / "templates"


def load_skill() -> str:
    return SKILL_PATH.read_text(encoding="utf-8")


def render_one(env: Environment, template_name: str, payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        return env.get_template(template_name).render(**payload)
    return str(payload)


def render_artifacts(filled: dict[str, Any]) -> dict[str, str]:
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR), keep_trailing_newline=True)
    return {
        "report.md":      render_one(env, "report.md.j2",      filled.get("report", "")),
        "analysis.R":     render_one(env, "code.R.j2",         filled.get("code", "")),
        "explanation.md": render_one(env, "explanation.md.j2", filled.get("explanation", "")),
    }


def _balanced_json_at(s: str, start: int) -> str | None:
    """Return the substring s[start..end] containing a balanced JSON object
    starting with '{' at position `start`, or None if no balanced object."""
    if start >= len(s) or s[start] != "{":
        return None
    depth, in_string, escape = 0, False, False
    for i in range(start, len(s)):
        c = s[i]
        if escape:
            escape = False
            continue
        if in_string:
            if c == "\\":
                escape = True
            elif c == '"':
                in_string = False
            continue
        if c == '"':
            in_string = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return None


def _find_fenced_blocks(text: str, lang_pattern: str) -> list[str]:
    """Find all fenced blocks whose language tag matches lang_pattern."""
    rx = re.compile(rf"```\s*({lang_pattern})\s*\n(.*?)\n```", re.DOTALL | re.IGNORECASE)
    return [m.group(2) for m in rx.finditer(text)]


_ANY_FENCE_RX = re.compile(r"```\s*([A-Za-z0-9_+\-]*)\s*\n(.*?)\n```", re.DOTALL)


def _all_fenced_blocks(text: str) -> list[tuple[str, str]]:
    """Return [(lang, content), ...] for every fenced block."""
    return [(m.group(1).lower(), m.group(2)) for m in _ANY_FENCE_RX.finditer(text)]


def _find_all_json_objects(text: str) -> list[dict]:
    """Find every top-level balanced JSON object in `text` and parse it.

    This catches both fenced (```json ... ```) and bare ({ ... }) JSON.
    Returns the list of successfully-parsed dicts (skipping invalid ones)."""
    found: list[dict] = []
    seen_spans: list[tuple[int, int]] = []

    # First pass: ```json fenced blocks (most reliable)
    for m in re.finditer(r"```json\s*\n?", text, re.IGNORECASE):
        block = _balanced_json_at(text, m.end())
        if block is None:
            # Fenced ```json with content that doesn't start with {
            # (e.g., the LLM put a string or array inside). Try parsing
            # the chunk between the fence open and the next ```.
            close = text.find("```", m.end())
            if close > m.end():
                candidate = text[m.end():close].strip()
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict):
                        found.append(obj)
                except json.JSONDecodeError:
                    pass
            continue
        try:
            obj = json.loads(block)
            if isinstance(obj, dict):
                found.append(obj)
                # Record the span we already consumed so the bare-JSON pass
                # doesn't re-extract the same object.
                start_idx = text.rfind(block, 0, m.end() + len(block) + 10)
                if start_idx >= 0:
                    seen_spans.append((start_idx, start_idx + len(block)))
        except json.JSONDecodeError:
            pass

    # Second pass: bare top-level JSON objects (not inside any fence we already found)
    def _in_seen(pos: int) -> bool:
        return any(a <= pos < b for a, b in seen_spans)

    # Also skip JSON inside any fenced code block (json or otherwise) to
    # avoid double-counting; we already handled ```json above.
    fenced_spans: list[tuple[int, int]] = []
    for m in _ANY_FENCE_RX.finditer(text):
        fenced_spans.append((m.start(), m.end()))

    def _in_any_fence(pos: int) -> bool:
        return any(a <= pos < b for a, b in fenced_spans)

    for m in re.finditer(r"\{", text):
        pos = m.start()
        if _in_seen(pos) or _in_any_fence(pos):
            continue
        block = _balanced_json_at(text, pos)
        if block is None:
            continue
        try:
            obj = json.loads(block)
            if isinstance(obj, dict):
                found.append(obj)
                seen_spans.append((pos, pos + len(block)))
        except json.JSONDecodeError:
            continue

    return found


def extract_final(text: str) -> dict | None:
    """Pull report/code/explanation out of the LLM's final message.

    Strategies, tried in order:
      1. Any single JSON object (fenced or bare) with all three keys.
      2. Merge keys across multiple JSON objects (fenced or bare).
      3. Fenced ```r``` block fills missing `code`; ```markdown```/```text```/etc
         fenced blocks fill missing `explanation`.
      4. Last-resort: any leftover non-JSON, non-R fenced block becomes
         the explanation.

    Returns None if the message is purely conversational (no artifacts)."""
    if not text:
        return None

    REQUIRED = {"report", "code", "explanation"}

    # Collect every JSON object we can find, fenced or bare
    json_objects = _find_all_json_objects(text)

    # Strategy 1: a single JSON object with all three keys
    for obj in json_objects:
        if REQUIRED.issubset(obj.keys()):
            # Filter to just the required keys to avoid noise like
            # extra metadata fields the LLM sometimes adds.
            return {k: obj[k] for k in REQUIRED}

    # Strategy 2: merge keys across multiple JSON objects
    merged: dict[str, Any] = {}
    for obj in json_objects:
        for k in REQUIRED:
            if k in obj and k not in merged:
                merged[k] = obj[k]

    # Strategy 3: pull missing fields from raw fenced blocks
    if "code" not in merged:
        r_blocks = _find_fenced_blocks(text, r"r|R")
        if r_blocks:
            merged["code"] = r_blocks[0].strip()

    if "explanation" not in merged:
        md_blocks = _find_fenced_blocks(text, r"markdown|md|text|plaintext|plain|txt")
        if md_blocks:
            merged["explanation"] = md_blocks[0].strip()

    # Strategy 4: any leftover non-{json,r} fenced block as explanation
    if "explanation" not in merged:
        skip_langs = {"json", "r"}
        for lang, content in _all_fenced_blocks(text):
            if lang in skip_langs:
                continue
            stripped = content.strip()
            if stripped:
                merged["explanation"] = stripped
                break

    if REQUIRED.issubset(merged.keys()):
        return merged

    return None


# --------------------------------------------------------------------------- #
# Scaffolded first-turn workflow                                              #
# --------------------------------------------------------------------------- #
# For models that have learned individual tool calls but not autonomous
# chaining (specifically the SFT-tuned Qwen-1.5B), we orchestrate the
# initial workflow turn-by-turn: model autonomously picks detect_regime,
# then we prompt-nudge each subsequent step and force-execute if the model
# calls the wrong tool. After the scaffold, the main loop runs as usual to
# let the model produce the final report/code/explanation JSON.
#
# This path is OpenAI-compatible only (it builds OAI-style assistant + tool
# message dicts to inject the forced calls). Anthropic / Gemini models do
# not need scaffolding and should be run with cfg["scaffold"] = False.

def _force_oai_tool_call(messages: list[dict], name: str, args: dict,
                         session: dict, trace: list[dict]) -> tuple[str, dict]:
    """Synthesize an OpenAI-format assistant tool_call message, execute the
    tool, append the tool_result, and return (fake_id, result)."""
    fake_id = f"forced_{name}_{len(messages)}"
    try:
        bridge_out = call_tool(name, args, session)
        result = (bridge_out.get("result") if bridge_out.get("ok")
                  else {"error": bridge_out.get("error")})
    except ToolError as e:
        result = {"error": str(e)}
    messages.append({
        "role": "assistant",
        "tool_calls": [{
            "id": fake_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }],
    })
    messages.append({
        "role": "tool",
        "tool_call_id": fake_id,
        "content": json.dumps(result, ensure_ascii=False),
    })
    trace.append({"step": f"scaffold-forced-{name}",
                  "tool_result": {"name": name, "result": result}})
    return fake_id, result


def _scaffold_step(client, system, messages, trace, session, step_label,
                   target_tool: str, target_args: dict | None) -> Iterator[dict]:
    """Run one scaffolded step: LLM call, check whether the model called
    `target_tool`, force-execute with `target_args` if it did not. Yields
    the same event types as the main loop. Mutates `messages` and `trace`.
    If `target_args` is None, no force-fallback (used for the autonomous
    detect_regime step)."""
    yield {"type": "llm_call", "step": step_label}
    try:
        resp: LLMResponse = client.complete(messages, TOOL_SPECS, system)
    except Exception as e:
        import traceback as tb
        yield {"type": "error",
               "error": f"LLM call failed ({step_label}): {e}",
               "traceback": tb.format_exc()}
        return

    trace.append({"step": step_label, "text": resp.text,
                  "tool_calls": resp.tool_calls, "stop_reason": resp.stop_reason})

    if resp.text:
        yield {"type": "assistant_text", "step": step_label, "text": resp.text}
    messages.append(client.format_assistant_with_tools(resp))

    target_called = False
    for call in resp.tool_calls:
        yield {"type": "tool_call", "step": step_label,
               "name": call["name"], "args": call["args"], "id": call["id"]}
        try:
            bridge_out = call_tool(call["name"], call["args"], session)
            result = (bridge_out.get("result") if bridge_out.get("ok")
                      else {"error": bridge_out.get("error")})
        except ToolError as e:
            result = {"error": str(e)}
        trace.append({"step": step_label,
                      "tool_result": {"name": call["name"], "result": result}})
        yield {"type": "tool_result", "step": step_label,
               "name": call["name"], "result": result, "id": call["id"]}
        messages.append(client.format_tool_result(call["id"], call["name"], result))
        if call["name"] == target_tool:
            target_called = True

    if target_args is not None and not target_called:
        fake_id, result = _force_oai_tool_call(messages, target_tool,
                                                target_args, session, trace)
        forced_name = f"{target_tool}__forced"
        yield {"type": "tool_call", "step": step_label,
               "name": forced_name, "args": target_args, "id": fake_id}
        yield {"type": "tool_result", "step": step_label,
               "name": forced_name, "result": result, "id": fake_id}


def _run_scaffolded_workflow(client, session, system,
                              messages: list[dict],
                              trace: list[dict]) -> Iterator[dict]:
    """Run the 3-step scaffolded workflow with force-fallback. After this
    returns, `messages` ends with the fit_integrated_estimator result plus
    a user nudge asking for the final JSON artifacts; the caller should run
    one more LLM turn to elicit the artifacts."""
    md = session.get("_metadata", {})

    # Step 1: detect_regime (autonomous; no force-fallback target).
    yield from _scaffold_step(client, system, messages, trace, session,
                               step_label="scaffold-1",
                               target_tool="detect_regime",
                               target_args=None)

    # Determine the regime from the latest tool_result, or fall back to
    # metadata-based inference.
    detected_regime = None
    for entry in reversed(trace):
        tr = entry.get("tool_result")
        if tr and tr.get("name") == "detect_regime":
            res = tr.get("result")
            if isinstance(res, dict):
                detected_regime = res.get("regime")
            break
    if detected_regime not in ("full", "partial", "restricted"):
        if md.get("has_internal_individual_data") and md.get("has_external_sigma2"):
            detected_regime = "full"
        elif md.get("has_internal_individual_data"):
            detected_regime = "partial"
        elif md.get("has_internal_marginal_only") and md.get("has_reference_panel"):
            detected_regime = "restricted"
        else:
            yield {"type": "error",
                   "error": "Cannot determine regime from session metadata."}
            return

    # Step 2: compute_eta_bound (force-fallback).
    messages.append({
        "role": "user",
        "content": f"Now call compute_eta_bound for the '{detected_regime}' regime.",
    })
    yield from _scaffold_step(client, system, messages, trace, session,
                               step_label="scaffold-2",
                               target_tool="compute_eta_bound",
                               target_args={"regime": detected_regime})

    # Step 3: fit_integrated_estimator (force-fallback).
    tuning = "eaic" if detected_regime == "restricted" else "cv"
    fit_args: dict[str, Any] = {"regime": detected_regime, "tuning": tuning}
    if tuning == "cv":
        fit_args["cv_seed"] = 548
    cv_directive = (" Pass cv_seed=548 for reproducibility."
                    if tuning == "cv" else "")
    messages.append({
        "role": "user",
        "content": (f"Now call fit_integrated_estimator with "
                    f"regime='{detected_regime}' and tuning='{tuning}'.{cv_directive}"),
    })
    yield from _scaffold_step(client, system, messages, trace, session,
                               step_label="scaffold-3",
                               target_tool="fit_integrated_estimator",
                               target_args=fit_args)

    # Final nudge: ask the model to produce the JSON artifacts.
    messages.append({
        "role": "user",
        "content": ("All tool calls are complete. Now produce the final "
                    "integration report, runnable R code, and plain-language "
                    "explanation as a single JSON object with keys "
                    "`report`, `code`, `explanation`, wrapped in a fenced "
                    "```json``` code block."),
    })


def run_agent_streaming(
    session: dict,
    user_message: str,
    cfg: dict,
    api_key: str | None = None,
    base_url: str | None = None,
    prior_messages: list[dict] | None = None,
) -> Iterator[dict]:
    """Yield one event dict per step. Pass `base_url` to use a non-default
    endpoint for OpenAI-compatible vendors. Set `cfg['scaffold'] = True` to
    enable the scaffolded first-turn workflow (needed for the SFT FT model
    that does not autonomously chain tool calls)."""
    try:
        client = make_client(
            vendor=cfg["vendor"], model=cfg["model"],
            max_tokens=cfg.get("max_tokens", 4096),
            temperature=cfg.get("temperature", 0.0),
            api_key=api_key,
            base_url=base_url,
        )
    except Exception as e:
        import traceback as tb
        yield {"type": "error", "error": str(e), "traceback": tb.format_exc()}
        return

    yield {"type": "started", "vendor": cfg["vendor"], "model": cfg["model"]}

    skill = load_skill()
    system = (
        f"{skill}\n\n"
        f"Session metadata (read-only):\n{json.dumps(session['_metadata'], indent=2)}"
    )

    messages: list[dict] = list(prior_messages or [])
    messages.append({"role": "user", "content": user_message})

    trace: list[dict] = []
    max_iters = cfg.get("max_tool_iterations", 8)

    # First-turn scaffolding for FT models that do not autonomously chain
    # tool calls. The scaffold runs the workflow's three core steps with
    # prompt-nudge + force-fallback, then leaves a user message asking for
    # the final JSON artifacts; the main loop below produces them.
    if cfg.get("scaffold") and not prior_messages:
        for ev in _run_scaffolded_workflow(client, session, system,
                                            messages, trace):
            if ev.get("type") == "error":
                yield ev
                return
            yield ev

    for step in range(max_iters):
        yield {"type": "llm_call", "step": step}

        try:
            resp: LLMResponse = client.complete(messages, TOOL_SPECS, system)
        except Exception as e:
            import traceback as tb
            yield {"type": "error", "error": f"LLM call failed: {e}", "traceback": tb.format_exc()}
            return

        trace.append({
            "step": step, "text": resp.text,
            "tool_calls": resp.tool_calls, "stop_reason": resp.stop_reason,
        })

        if resp.text:
            yield {"type": "assistant_text", "step": step, "text": resp.text}

        messages.append(client.format_assistant_with_tools(resp))

        if not resp.tool_calls:
            final = extract_final(resp.text or "")
            if final is None:
                yield {"type": "final_text", "text": resp.text or "",
                       "trace": trace, "messages": messages}
            else:
                yield {"type": "final", "final": final,
                       "trace": trace, "messages": messages}
            return

        for call in resp.tool_calls:
            yield {"type": "tool_call", "step": step,
                   "name": call["name"], "args": call["args"], "id": call["id"]}
            try:
                bridge_out = call_tool(call["name"], call["args"], session)
                result = (bridge_out.get("result")
                          if bridge_out.get("ok")
                          else {"error": bridge_out.get("error")})
            except ToolError as e:
                result = {"error": str(e)}
            trace.append({"step": step, "tool_result": {"name": call["name"], "result": result}})
            yield {"type": "tool_result", "step": step,
                   "name": call["name"], "result": result, "id": call["id"]}
            messages.append(client.format_tool_result(call["id"], call["name"], result))

    yield {"type": "error",
           "error": f"Hit max_tool_iterations={max_iters} without final answer.",
           "traceback": ""}


def run_agent(
    session: dict, user_message: str, cfg: dict,
    api_key: str | None = None, base_url: str | None = None,
    prior_messages: list[dict] | None = None,
) -> dict:
    final, trace, messages, text = None, None, None, None
    err = None
    for event in run_agent_streaming(session, user_message, cfg,
                                     api_key=api_key, base_url=base_url,
                                     prior_messages=prior_messages):
        if event["type"] == "final":
            final = event["final"]; trace = event["trace"]; messages = event["messages"]
        elif event["type"] == "final_text":
            text = event["text"]; trace = event["trace"]; messages = event["messages"]
        elif event["type"] == "error":
            err = event
    if err:
        raise RuntimeError(f"{err.get('error')}\n{err.get('traceback', '')}")
    return {"final": final, "text": text, "trace": trace or [], "messages": messages or []}


def _safe_write_json(path: Path, obj: Any) -> None:
    try:
        path.write_text(json.dumps(obj, indent=2, default=str, ensure_ascii=False),
                        encoding="utf-8")
    except Exception as e:
        print(f"  (warning: could not write {path.name}: {e})")


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description="mainger-agent (online)")
    ap.add_argument("--input", required=True)
    ap.add_argument("--external-coef", required=True)
    ap.add_argument("--external-sigma")
    ap.add_argument("--reference-sigma")
    ap.add_argument("--sigma2-int", type=float)
    ap.add_argument("--sigma2-ext", type=float)
    ap.add_argument("--n-ext", type=int)
    ap.add_argument("--regime", choices=["full", "partial", "restricted"])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out-dir", default="runs/latest")
    ap.add_argument("--vendor", default=None, help="Override config.yaml vendor")
    ap.add_argument("--model",  default=None, help="Override config.yaml model")
    ap.add_argument("--base-url", default=None, help="Optional base URL for OpenAI-compatible vendors")
    ap.add_argument("--message", default="Please analyze my data and produce the integration report, code, and explanation.")
    ap.add_argument("--scaffold", action="store_true",
                    help="Enable scaffolded first-turn workflow. Required when "
                         "running the SFT-tuned Qwen-1.5B locally (the FT model "
                         "does not autonomously chain tool calls). Has no effect "
                         "for capable models like GPT-4o.")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.vendor: cfg["vendor"] = args.vendor
    if args.model:  cfg["model"]  = args.model
    if args.scaffold: cfg["scaffold"] = True

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Loading data into session ...")
    session = build_session(
        internal_path=args.input,
        external_coef_path=args.external_coef,
        external_sigma_path=args.external_sigma,
        reference_sigma_path=args.reference_sigma,
        sigma2_int=args.sigma2_int,
        sigma2_ext=args.sigma2_ext,
        n_ext=args.n_ext,
    )
    session = persist_session(session, out_dir)
    print(f"  metadata: {json.dumps(session['_metadata'])}")

    print(f"[2/4] Running agent (vendor={cfg['vendor']}, model={cfg['model']}) ...")
    user_msg = args.message
    if args.regime:
        user_msg += f"\n(I believe this is the {args.regime} regime.)"
    out = run_agent(session, user_msg, cfg, base_url=args.base_url)

    _safe_write_json(out_dir / "trace.json", out["trace"])
    if out["final"]:
        _safe_write_json(out_dir / "final.json", out["final"])

    print("[3/4] Rendering artifacts ...")
    if out["final"]:
        artifacts = render_artifacts(out["final"])
        print(f"[4/4] Writing outputs to {out_dir} ...")
        for name, content in artifacts.items():
            (out_dir / name).write_text(
                content if isinstance(content, str) else str(content),
                encoding="utf-8",
            )
    else:
        print(f"[4/4] No artifacts produced; conversational response only.")

    print("Done.")


if __name__ == "__main__":
    main()
