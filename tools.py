"""
tools.py
--------
Tool specifications and dispatcher for the four tools the agent can call.
The R bridge (r_helpers/run_mainger.R) does the actual computation.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent
R_BRIDGE = ROOT / "r_helpers" / "run_mainger.R"


class ToolError(Exception):
    pass


# --------------------------------------------------------------------------- #
# Tool specifications (vendor-agnostic; LLM clients translate to native form) #
# --------------------------------------------------------------------------- #
TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "detect_regime",
        "description": (
            "Detect the data-sharing regime (full / partial / restricted) "
            "from the metadata of the user's session."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "has_internal_individual_data": {"type": "boolean"},
                "has_internal_marginal_only":   {"type": "boolean"},
                "has_external_theta":           {"type": "boolean"},
                "has_external_sigma2":          {"type": "boolean"},
                "has_reference_panel":          {"type": "boolean"},
            },
            "required": [
                "has_internal_individual_data",
                "has_external_theta",
            ],
        },
    },
    {
        "name": "compute_eta_bound",
        "description": (
            "Compute the upper limit eta_star of the beneficial range "
            "for the integration weight, given the detected regime."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "regime": {"type": "string", "enum": ["full", "partial", "restricted"]},
            },
            "required": ["regime"],
        },
    },
    {
        "name": "check_concordance",
        "description": (
            "Diagnose whether full sharing dominates partial. Returns a "
            "verdict in {concordant, discordant, indeterminate} along with "
            "the spectral advantage. Full regime only."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "eta": {"type": "number"},
            },
            "required": ["eta"],
        },
    },
    {
        "name": "fit_integrated_estimator",
        "description": (
            "Fit the integrated estimator. Pass tuning='fixed' with a specific "
            "eta, or tuning in {'cv', 'eaic'} to select eta from the grid. "
            "When tuning='cv', pass cv_seed to make the cross-validation "
            "reproducible; the seed has no effect for 'fixed' or 'eaic'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "regime": {"type": "string", "enum": ["full", "partial", "restricted"]},
                "tuning": {"type": "string", "enum": ["fixed", "cv", "eaic"]},
                "eta":    {"type": "number", "description": "Required when tuning='fixed'."},
                "cv_seed": {
                    "type": "integer",
                    "description": (
                        "Random seed for fold assignment in cross-validation. "
                        "Pass this whenever tuning='cv' and the user has "
                        "supplied a reproducibility directive. Ignored when "
                        "tuning is 'eaic' or 'fixed'."
                    ),
                },
            },
            "required": ["regime", "tuning"],
        },
    },
]


# --------------------------------------------------------------------------- #
# Dispatcher                                                                   #
# --------------------------------------------------------------------------- #
def call_tool(name: str, args: dict, session: dict) -> dict:
    """Dispatch a tool call. Returns {ok: bool, result: ..., error: ...}."""
    if name == "detect_regime":
        return _call_detect_regime(args, session)
    elif name == "compute_eta_bound":
        return _call_r_bridge("compute_eta_bound", args, session)
    elif name == "check_concordance":
        return _call_r_bridge("check_concordance", args, session)
    elif name == "fit_integrated_estimator":
        return _call_r_bridge("fit_integrated_estimator", args, session)
    else:
        return {"ok": False, "error": f"Unknown tool: {name}"}


def _call_detect_regime(args: dict, session: dict) -> dict:
    """Pure-Python regime detection from session metadata.

    Reads session["_metadata"] populated by data_io.persist_session().
    The metadata flag for external coefficients is `has_external_theta`.
    """
    md = session.get("_metadata", {})
    has_int_indiv  = bool(md.get("has_internal_individual_data", False))
    has_int_marg   = bool(md.get("has_internal_marginal_only", False))
    has_ext_theta  = bool(md.get("has_external_theta", False))
    has_ext_sigma  = bool(md.get("has_external_sigma2", False))
    has_ref_panel  = bool(md.get("has_reference_panel", False))

    if not has_ext_theta:
        return {"ok": False,
                "error": "External coefficients are required for any regime."}

    if has_int_indiv and has_ext_sigma:
        regime = "full"
    elif has_int_indiv:
        regime = "partial"
    elif has_int_marg and has_ref_panel:
        regime = "restricted"
    else:
        return {"ok": False, "error": "Cannot determine regime from inputs."}

    return {"ok": True, "result": {
        "regime": regime,
        "reason": (
            f"internal_indiv={has_int_indiv}, internal_marginal={has_int_marg}, "
            f"ext_theta={has_ext_theta}, ext_sigma2={has_ext_sigma}, "
            f"ref_panel={has_ref_panel}"
        ),
    }}


def _call_r_bridge(action: str, args: dict, session: dict) -> dict:
    """Invoke the R bridge with a JSON payload.

    The session's RDS path is stored at session["_path"] by
    data_io.persist_session().
    """
    session_path = session.get("_path")
    if not session_path:
        return {"ok": False, "error": "Session not persisted to RDS (missing _path)."}

    payload = {
        "action": action,
        "session_path": str(session_path),
        "args": args,
    }

    try:
        proc = subprocess.run(
            ["Rscript", "--vanilla", str(R_BRIDGE)],
            input=json.dumps(payload),
            capture_output=True, text=True, timeout=120,
            encoding="utf-8",
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "R bridge timed out (120s)."}
    except FileNotFoundError:
        return {"ok": False, "error": "Rscript not found in PATH."}

    if proc.returncode != 0:
        return {"ok": False,
                "error": f"R bridge failed: {proc.stderr.strip() or proc.stdout.strip()}"}

    try:
        out = json.loads(proc.stdout.strip().split("\n")[-1])
    except (json.JSONDecodeError, IndexError) as e:
        return {"ok": False,
                "error": f"R bridge returned non-JSON: {e}\nstdout:\n{proc.stdout}"}

    if isinstance(out, dict) and "error" in out:
        return {"ok": False, "error": out["error"]}
    return {"ok": True, "result": out}
