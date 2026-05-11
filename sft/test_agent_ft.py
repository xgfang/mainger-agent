"""
sft/test_agent_ft.py
--------------------
End-to-end smoke test of agent.py with the SFT-tuned Qwen-1.5B model and
the scaffolded first-turn workflow. Picks the first problem from the
simulation corpus, materializes CSVs, runs the agent against a locally-
running vLLM at 127.0.0.1:8000, and prints whether final JSON artifacts
were produced.

Run this from the project root after vLLM is serving the merged FT model.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from agent import run_agent
from data_io import build_session, persist_session

PROBLEMS = ROOT / "sim" / "data" / "sim_problems_moderate.jsonl"
OUT_DIR = ROOT / "runs" / "agent_ft_smoke"
VLLM_URL = "http://127.0.0.1:8000/v1"


def _materialize(problem: dict, work_dir: Path) -> dict:
    """Write the problem's data to CSVs and return build_session kwargs."""
    work_dir.mkdir(parents=True, exist_ok=True)
    pred_names = problem["predictor_names"]
    paths: dict = {}

    if "X_int" in problem and "Y_int" in problem:
        ip = work_dir / "internal.csv"
        with ip.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["y"] + pred_names)
            for y, row in zip(problem["Y_int"], problem["X_int"]):
                w.writerow([y] + list(row))
        paths["internal_path"] = str(ip)

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

    return paths


def main() -> int:
    with PROBLEMS.open("r", encoding="utf-8") as f:
        problem = json.loads(f.readline())

    print(f"Test problem: {problem['problem_id']} (regime={problem['regime']}, "
          f"p={problem['p']}, n_int={problem.get('n_int')})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = _materialize(problem, OUT_DIR / "uploads")

    session = build_session(
        **paths,
        sigma2_ext=problem.get("sigma2_ext"),
        n_ext=problem.get("n_ext"),
    )
    session = persist_session(session, OUT_DIR)
    print(f"Session metadata: {json.dumps(session['_metadata'], indent=2)}")

    cfg = {
        "vendor": "custom",
        "model": "qwen-1.5b-ft",
        "max_tokens": 2048,
        "temperature": 0.0,
        "max_tool_iterations": 8,
        "scaffold": True,
    }

    print(f"\nRunning agent.py with scaffold against {VLLM_URL} ...\n")
    out = run_agent(
        session=session,
        user_message="Please analyze my data and produce the integration "
                     "report, code, and explanation.",
        cfg=cfg,
        api_key="vllm",
        base_url=VLLM_URL,
    )

    # Persist outputs for inspection
    if out["final"] is not None:
        (OUT_DIR / "final.json").write_text(
            json.dumps(out["final"], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        for key, fname in [("report", "report.md"), ("code", "analysis.R"),
                            ("explanation", "explanation.md")]:
            content = out["final"].get(key, "")
            if isinstance(content, str):
                (OUT_DIR / fname).write_text(content, encoding="utf-8")

    (OUT_DIR / "trace.json").write_text(
        json.dumps(out["trace"], indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 70)
    print("RESULT")
    print("=" * 70)
    print(f"final JSON produced       : {out['final'] is not None}")
    if out["final"] is not None:
        print(f"  keys                    : {list(out['final'].keys())}")
        for k in ("report", "code", "explanation"):
            v = out["final"].get(k, "")
            n = len(v) if isinstance(v, str) else 0
            print(f"  len({k:<11s})         : {n} chars")
    else:
        print(f"final_text (first 500ch)  : {(out.get('text') or '')[:500]}")
    print(f"trace events              : {len(out['trace'])}")
    print(f"output dir                : {OUT_DIR}")
    print("=" * 70)

    return 0 if out["final"] is not None else 1


if __name__ == "__main__":
    sys.exit(main())
