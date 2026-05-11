"""
server.py
---------
FastAPI server for multi-turn mainger-agent chat.
"""
from __future__ import annotations

import json
import shutil
import sys
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from agent import render_artifacts, run_agent_streaming
from data_io import build_session, persist_session

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

load_dotenv()

ROOT = Path(__file__).parent
RUNS_DIR = ROOT / "runs"
RUNS_DIR.mkdir(exist_ok=True)
WEB_DIR = ROOT / "web"

app = FastAPI(title="mainger-agent")


# --------------------------------------------------------------------------- #
# Vendor catalog                                                               #
# --------------------------------------------------------------------------- #
VENDOR_CATALOG: dict[str, dict[str, Any]] = {
    "anthropic": {
        "label": "Anthropic (Claude)", "kind": "closed",
        "models": ["claude-opus-4-7", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
        "base_url": None, "needs_url": False,
    },
    "openai": {
        "label": "OpenAI (GPT)", "kind": "closed",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
        "base_url": None, "needs_url": False,
    },
    "gemini": {
        "label": "Google (Gemini)", "kind": "closed",
        "models": ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-1.5-pro"],
        "base_url": None, "needs_url": False,
    },
    "xai": {
        "label": "xAI (Grok)", "kind": "closed",
        "models": ["grok-4", "grok-3", "grok-3-mini"],
        "base_url": "https://api.x.ai/v1", "needs_url": False,
    },
    "together": {
        "label": "Together AI (open-source hosted)", "kind": "open",
        "models": [
            "Qwen/Qwen2.5-72B-Instruct-Turbo",
            "Qwen/Qwen2.5-7B-Instruct-Turbo",
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
        ],
        "base_url": "https://api.together.xyz/v1", "needs_url": False,
    },
    "fireworks": {
        "label": "Fireworks AI (open-source hosted)", "kind": "open",
        "models": [
            "accounts/fireworks/models/qwen2p5-72b-instruct",
            "accounts/fireworks/models/llama-v3p3-70b-instruct",
            "accounts/fireworks/models/mixtral-8x22b-instruct",
        ],
        "base_url": "https://api.fireworks.ai/inference/v1", "needs_url": False,
    },
    "openrouter": {
        "label": "OpenRouter (multi-provider)", "kind": "open",
        "models": [
            "qwen/qwen-2.5-72b-instruct",
            "meta-llama/llama-3.3-70b-instruct",
            "deepseek/deepseek-chat",
            "google/gemini-flash-1.5",
        ],
        "base_url": "https://openrouter.ai/api/v1", "needs_url": False,
    },
    "groq": {
        "label": "Groq (fast inference)", "kind": "open",
        "models": ["llama-3.3-70b-versatile", "qwen-2.5-32b", "mixtral-8x7b-32768"],
        "base_url": "https://api.groq.com/openai/v1", "needs_url": False,
    },
    "huggingface": {
        "label": "HuggingFace Inference Endpoints", "kind": "open",
        "models": [
            "Qwen/Qwen2.5-7B-Instruct",
            "Qwen/Qwen2.5-72B-Instruct",
            "meta-llama/Llama-3.3-70B-Instruct",
        ],
        "base_url": "https://router.huggingface.co/v1", "needs_url": False,
    },
    "custom": {
        "label": "Custom (your own OpenAI-compatible endpoint)", "kind": "open",
        "models": [], "base_url": None, "needs_url": True,
    },
}


# --------------------------------------------------------------------------- #
# Session state                                                                #
# --------------------------------------------------------------------------- #
@dataclass
class SessionState:
    sid: str
    run_dir: Path
    session_data: dict
    messages: list[dict] = field(default_factory=list)
    cfg: dict = field(default_factory=dict)
    api_key: str | None = None
    base_url: str | None = None
    last_artifacts: dict[str, str] = field(default_factory=dict)
    chat_log: list[dict] = field(default_factory=list)


_SESSIONS: dict[str, SessionState] = {}


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
VALID_FILE_ROLES = {"internal", "external_coef", "external_sigma", "reference_sigma"}


def _save_upload(upload: UploadFile, dest: Path) -> None:
    with dest.open("wb") as f:
        shutil.copyfileobj(upload.file, f)


def _opt_float(s):
    if s is None or not str(s).strip(): return None
    return float(s)
def _opt_int(s):
    if s is None or not str(s).strip(): return None
    return int(s)
def _opt_str(s):
    if s is None or not str(s).strip(): return None
    return s


def _read_config() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


def _resolve_cfg(vendor_override, model_override) -> dict:
    cfg = _read_config()
    if vendor_override: cfg["vendor"] = vendor_override
    if model_override:  cfg["model"]  = model_override
    return cfg


def _resolve_base_url(vendor: str, base_url_override: str | None) -> str | None:
    if base_url_override:
        return base_url_override
    info = VENDOR_CATALOG.get((vendor or "").lower())
    return info["base_url"] if info else None


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, default=str, ensure_ascii=False)}\n\n"


def _write_text_utf8(path: Path, content: str) -> None:
    if not isinstance(content, str):
        content = str(content)
    path.write_text(content, encoding="utf-8")


def _save_uploaded_file(state: SessionState, upload: UploadFile, role: str) -> Path:
    if role not in VALID_FILE_ROLES:
        raise ValueError(f"unknown role '{role}'")
    uploads_dir = state.run_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(upload.filename or "").suffix.lower() or ".csv"
    dest = uploads_dir / f"{role}{suffix}"
    _save_upload(upload, dest)
    return dest


def _update_session_data(
    state: SessionState, *,
    internal_path=None, external_coef_path=None,
    external_sigma_path=None, reference_sigma_path=None,
    sigma2_int=None, sigma2_ext=None, n_ext=None,
):
    new_session = build_session(
        internal_path=internal_path,
        external_coef_path=external_coef_path,
        external_sigma_path=external_sigma_path,
        reference_sigma_path=reference_sigma_path,
        sigma2_int=sigma2_int, sigma2_ext=sigma2_ext, n_ext=n_ext,
        base_session=state.session_data,
    )
    state.session_data = persist_session(new_session, state.run_dir)


def _persist_chat_log(state: SessionState) -> None:
    _write_text_utf8(state.run_dir / "chat_log.json",
                     json.dumps(state.chat_log, indent=2, default=str, ensure_ascii=False))


def _seed_directive(seed: int | None) -> str:
    """Build a directive line appended to the user message when a seed is set.

    The directive is phrased so the LLM treats it as an instruction to
    include in the runnable R code (skill.md describes how)."""
    if seed is None:
        return ""
    return (
        f"\n\n[Reproducibility: use random seed {seed} for cross-validation. "
        f"Include set.seed({seed}) in the generated R code only if "
        f"tuning = \"cv\".]"
    )


# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #
@app.get("/api/config")
def get_config() -> dict:
    cfg = _read_config()
    catalog = {
        v: {"label": info["label"], "kind": info["kind"],
            "models": info["models"], "base_url": info["base_url"],
            "needs_url": info["needs_url"]}
        for v, info in VENDOR_CATALOG.items()
    }
    return {
        "vendor": cfg.get("vendor"),
        "model":  cfg.get("model"),
        "vendors": catalog,
    }


# --------------------------------------------------------------------------- #
# Init                                                                         #
# --------------------------------------------------------------------------- #
@app.post("/api/sessions/init")
async def session_init(
    internal: UploadFile = File(...),
    external_coef: UploadFile = File(...),
    external_sigma: UploadFile | None = File(None),
    reference_sigma: UploadFile | None = File(None),
    regime: str = Form(""),
    sigma2_int: str = Form(""),
    sigma2_ext: str = Form(""),
    n_ext: str = Form(""),
    cv_seed: str = Form(""),
    vendor: str = Form(""),
    model: str = Form(""),
    api_key: str = Form(""),
    base_url: str = Form(""),
    message: str = Form(
        "Please analyze my data and produce the integration report, code, and explanation."
    ),
):
    sid = uuid.uuid4().hex[:8]
    run_dir = RUNS_DIR / sid
    uploads_dir = run_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    try:
        def _stash(upload, role):
            suffix = Path(upload.filename or "").suffix.lower() or ".csv"
            dest = uploads_dir / f"{role}{suffix}"
            _save_upload(upload, dest)
            return dest

        internal_path = _stash(internal, "internal")
        external_coef_path = _stash(external_coef, "external_coef")
        external_sigma_path = _stash(external_sigma, "external_sigma") if (external_sigma and external_sigma.filename) else None
        reference_sigma_path = _stash(reference_sigma, "reference_sigma") if (reference_sigma and reference_sigma.filename) else None

        session = build_session(
            internal_path=internal_path,
            external_coef_path=external_coef_path,
            external_sigma_path=external_sigma_path,
            reference_sigma_path=reference_sigma_path,
            sigma2_int=_opt_float(sigma2_int),
            sigma2_ext=_opt_float(sigma2_ext),
            n_ext=_opt_int(n_ext),
        )
        session = persist_session(session, run_dir)

        cfg = _resolve_cfg(_opt_str(vendor), _opt_str(model))
        url = _resolve_base_url(cfg.get("vendor"), _opt_str(base_url))

        state = SessionState(
            sid=sid, run_dir=run_dir, session_data=session,
            cfg=cfg, api_key=_opt_str(api_key), base_url=url,
        )
        _SESSIONS[sid] = state
    except Exception as e:  # noqa: BLE001
        async def fail_gen():
            yield _sse({"type": "error", "error": str(e),
                        "traceback": traceback.format_exc()})
        return StreamingResponse(fail_gen(), media_type="text/event-stream")

    user_msg = message
    if _opt_str(regime):
        user_msg += f"\n(I believe this is the {regime} regime.)"
    user_msg += _seed_directive(_opt_int(cv_seed))

    return StreamingResponse(
        _run_turn_generator(state, user_msg, initial=True),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive",
                 "X-Accel-Buffering": "no"},
    )


# --------------------------------------------------------------------------- #
# Chat                                                                         #
# --------------------------------------------------------------------------- #
@app.post("/api/sessions/{sid}/chat")
async def session_chat(
    sid: str,
    message: str = Form(...),
    sigma2_int: str = Form(""),
    sigma2_ext: str = Form(""),
    n_ext: str = Form(""),
    cv_seed: str = Form(""),
    file_internal: UploadFile | None = File(None),
    file_external_coef: UploadFile | None = File(None),
    file_external_sigma: UploadFile | None = File(None),
    file_reference_sigma: UploadFile | None = File(None),
):
    if sid not in _SESSIONS:
        raise HTTPException(status_code=404, detail=f"Unknown session '{sid}'")
    state = _SESSIONS[sid]

    update_args: dict[str, Any] = {}
    try:
        if file_internal and file_internal.filename:
            update_args["internal_path"] = _save_uploaded_file(state, file_internal, "internal")
        if file_external_coef and file_external_coef.filename:
            update_args["external_coef_path"] = _save_uploaded_file(state, file_external_coef, "external_coef")
        if file_external_sigma and file_external_sigma.filename:
            update_args["external_sigma_path"] = _save_uploaded_file(state, file_external_sigma, "external_sigma")
        if file_reference_sigma and file_reference_sigma.filename:
            update_args["reference_sigma_path"] = _save_uploaded_file(state, file_reference_sigma, "reference_sigma")
        if _opt_float(sigma2_int) is not None: update_args["sigma2_int"] = _opt_float(sigma2_int)
        if _opt_float(sigma2_ext) is not None: update_args["sigma2_ext"] = _opt_float(sigma2_ext)
        if _opt_int(n_ext)        is not None: update_args["n_ext"]      = _opt_int(n_ext)

        if update_args:
            _update_session_data(state, **update_args)
    except Exception as e:  # noqa: BLE001
        async def fail_gen():
            yield _sse({"type": "error", "error": str(e),
                        "traceback": traceback.format_exc()})
        return StreamingResponse(fail_gen(), media_type="text/event-stream")

    user_msg = message
    if update_args:
        updated = sorted(update_args.keys())
        user_msg = f"{message}\n\n(Note: the user updated session data: {', '.join(updated)}.)"
    user_msg += _seed_directive(_opt_int(cv_seed))

    return StreamingResponse(
        _run_turn_generator(state, user_msg, initial=False),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive",
                 "X-Accel-Buffering": "no"},
    )


# --------------------------------------------------------------------------- #
# Get / delete                                                                 #
# --------------------------------------------------------------------------- #
@app.get("/api/sessions/{sid}")
def session_get(sid: str) -> dict:
    if sid not in _SESSIONS:
        raise HTTPException(status_code=404, detail=f"Unknown session '{sid}'")
    state = _SESSIONS[sid]
    return {
        "sid": sid,
        "vendor": state.cfg.get("vendor"),
        "model": state.cfg.get("model"),
        "metadata": state.session_data.get("_metadata", {}),
        "chat_log": state.chat_log,
        "last_artifacts": state.last_artifacts,
    }


@app.delete("/api/sessions/{sid}")
def session_delete(sid: str) -> dict:
    state = _SESSIONS.pop(sid, None)
    return {"closed": state is not None}


# --------------------------------------------------------------------------- #
# Turn generator                                                               #
# --------------------------------------------------------------------------- #
def _run_turn_generator(state: SessionState, user_msg: str, initial: bool):
    def gen():
        yield _sse({
            "type": "init",
            "sid": state.sid,
            "metadata": state.session_data.get("_metadata", {}),
            "vendor": state.cfg.get("vendor"),
            "model":  state.cfg.get("model"),
            "initial": initial,
        })

        user_turn = {"role": "user", "text": user_msg, "ts": _now_ms()}
        state.chat_log.append(user_turn)
        yield _sse({"type": "chat_turn", "turn": user_turn})

        cfg = dict(state.cfg)

        try:
            iterator = run_agent_streaming(
                state.session_data, user_msg, cfg,
                api_key=state.api_key, base_url=state.base_url,
                prior_messages=state.messages if not initial else None,
            )
        except Exception as e:  # noqa: BLE001
            yield _sse({"type": "error", "error": str(e),
                        "traceback": traceback.format_exc()})
            return

        final, text, trace, messages = None, None, None, None
        try:
            for event in iterator:
                if event["type"] == "final":
                    final = event["final"]; trace = event["trace"]; messages = event["messages"]
                elif event["type"] == "final_text":
                    text = event["text"]; trace = event["trace"]; messages = event["messages"]
                yield _sse(event)
        except Exception as e:  # noqa: BLE001
            yield _sse({"type": "error", "error": str(e),
                        "traceback": traceback.format_exc()})
            return

        if messages is not None:
            state.messages = messages

        assistant_turn: dict[str, Any] = {"role": "assistant", "ts": _now_ms()}
        if final is not None:
            try:
                artifacts = render_artifacts(final)
                state.last_artifacts = artifacts
                for name, content in artifacts.items():
                    _write_text_utf8(state.run_dir / name, content)
                _write_text_utf8(state.run_dir / "trace.json",
                                 json.dumps(trace, indent=2, default=str, ensure_ascii=False))
                _write_text_utf8(state.run_dir / "final.json",
                                 json.dumps(final, indent=2, default=str, ensure_ascii=False))
                assistant_turn["artifacts"] = artifacts
                assistant_turn["text"] = ""
                yield _sse({"type": "complete", "artifacts": artifacts})
            except Exception as e:  # noqa: BLE001
                yield _sse({"type": "error", "error": f"Render failed: {e}",
                            "traceback": traceback.format_exc()})
                return
        elif text is not None:
            assistant_turn["text"] = text
            yield _sse({"type": "complete_text", "text": text})

        if trace:
            assistant_turn["trace"] = trace
        state.chat_log.append(assistant_turn)
        _persist_chat_log(state)
        yield _sse({"type": "chat_turn", "turn": assistant_turn})

    return gen()


def _now_ms() -> int:
    import time
    return int(time.time() * 1000)


# --------------------------------------------------------------------------- #
# Static frontend                                                              #
# --------------------------------------------------------------------------- #
if WEB_DIR.exists() and (WEB_DIR / "index.html").exists():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
else:
    print(f"WARNING: {WEB_DIR / 'index.html'} not found.")

    @app.get("/")
    def _missing_index():
        return JSONResponse(
            status_code=503,
            content={"error": f"web/index.html not found at {WEB_DIR}."},
        )


if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("  mainger-agent web UI (multi-vendor)")
    print("=" * 60)
    print("  open: http://localhost:8000")
    print("  stop: Ctrl+C")
    print("=" * 60)
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
