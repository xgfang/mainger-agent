"""
sim/aggregate_results.py
========================
Aggregate LLM evaluation results into paper-ready tables.

Reads:
  - One simulation problems file (sim_problems_moderate.jsonl)
  - One or more LLM results files (llm_results_*.jsonl)

Produces:
  TABLE 1 (HEADLINE): success rate, median MSPE among successes,
    imputed MSPE (failures penalized as OLS-internal). Per (strategy,
    model, regime) cell.
  TABLE 2: mean MSPE comparison vs median (shows heavy-tail distortion)
  TABLE 3: compute cost — average tokens, average time, S3/S1 ratio
  TABLE 4: paired ratios — median LLM/baseline ratio across problems
    where the LLM succeeded, vs OLS, vs mainger-CV, vs mainger-oracle
  TABLE 5: non-LLM baselines (OLS, ext-only, mainger-CV, mainger-oracle)
    median and mean MSPE per regime, for context

Run:
  python sim/aggregate_results.py \
      --problems sim/data/sim_problems_moderate.jsonl \
      --results  sim/data/llm_results_gpt.jsonl \
                 sim/data/llm_results_qwen7b.jsonl \
                 sim/data/llm_results_qwen15b.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    p = Path(path)
    if not p.exists():
        return rows
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def safe_mean(xs: list[float]) -> float:
    valid = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(valid)) if valid else float("nan")


def safe_median(xs: list[float]) -> float:
    valid = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.median(valid)) if valid else float("nan")


def safe_std(xs: list[float]) -> float:
    valid = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.std(valid, ddof=1)) if len(valid) > 1 else float("nan")


def quantiles(xs: list[float], qs: list[float]) -> list[float]:
    valid = [x for x in xs if x is not None and np.isfinite(x)]
    if not valid:
        return [float("nan")] * len(qs)
    return [float(np.quantile(valid, q)) for q in qs]


def median_iqr(xs: list[float]) -> tuple[float, float, float]:
    """Return (median, q25, q75); each NaN if no valid values."""
    valid = [x for x in xs if x is not None and np.isfinite(x)]
    if not valid:
        return float("nan"), float("nan"), float("nan")
    return (float(np.median(valid)),
            float(np.quantile(valid, 0.25)),
            float(np.quantile(valid, 0.75)))


def fmt(x: float | None, digits: int = 4) -> str:
    if x is None or not np.isfinite(x):
        return "  -- "
    return f"{x:>{digits + 4}.{digits}f}"


# --------------------------------------------------------------------------- #
# Table 1: HEADLINE                                                            #
# --------------------------------------------------------------------------- #
def table1_headline(results: list[dict],
                    problems_by_id: dict[str, dict]) -> None:
    print("=" * 86)
    print("TABLE 1 (HEADLINE): Success rate, median MSPE, imputed MSPE")
    print("=" * 86)
    print("Imputation: when LLM fails, MSPE = OLS-internal-only MSPE for that problem")
    print()

    success_buckets = defaultdict(list)  # (strategy, model, regime) -> list of mspe (ok only)
    imputed_buckets = defaultdict(list)  # same key -> list of mspe (ok or imputed)
    success_counts = defaultdict(lambda: [0, 0])  # key -> [n_ok, n_total]

    for r in results:
        prob = problems_by_id.get(r["problem_id"])
        if prob is None:
            continue
        key = (r["strategy"], r["model"], r["regime"])
        success_counts[key][1] += 1

        ols_mspe = prob.get("mspe", {}).get("ols_internal")
        if r.get("parse_status") == "ok" and r.get("mspe") is not None:
            success_buckets[key].append(r["mspe"])
            imputed_buckets[key].append(r["mspe"])
            success_counts[key][0] += 1
        elif ols_mspe is not None:
            imputed_buckets[key].append(ols_mspe)

    strategies = sorted({r["strategy"] for r in results})
    models = sorted({r["model"] for r in results})
    regimes = ["full", "partial", "restricted"]
    regimes = [r for r in regimes if any(rr["regime"] == r for rr in results)]

    print(f"{'strategy':<8}{'model':<14}{'regime':<14}"
          f"{'success':>10}{'median MSPE':>14}{'IQR [q25,q75]':>22}"
          f"{'SD':>10}{'imputed MSPE':>16}")
    for s in strategies:
        for m in models:
            for rg in regimes:
                key = (s, m, rg)
                ok_n, tot_n = success_counts[key]
                rate = f"{ok_n}/{tot_n} ({100 * ok_n / max(1, tot_n):.0f}%)"
                med, q25, q75 = median_iqr(success_buckets[key])
                sd  = safe_std(success_buckets[key])
                imp = safe_mean(imputed_buckets[key])
                iqr_s = (f"[{q25:.4f}, {q75:.4f}]"
                         if np.isfinite(q25) else "    --")
                print(f"{s:<8}{m:<14}{rg:<14}"
                      f"{rate:>10}{fmt(med):>14}{iqr_s:>22}"
                      f"{fmt(sd, 4):>10}{fmt(imp):>16}")
    print()


# --------------------------------------------------------------------------- #
# Table 2: Mean vs median (shows heavy-tail issue)                             #
# --------------------------------------------------------------------------- #
def table2_mean_vs_median(results: list[dict]) -> None:
    print("=" * 86)
    print("TABLE 2: Mean vs median MSPE among successes (shows heavy-tail distortion)")
    print("=" * 86)
    print("Large mean/median ratio indicates heavy-tailed distribution; median is")
    print("the more reliable summary for cells where this is large.")
    print()

    bucket = defaultdict(list)
    for r in results:
        if r.get("parse_status") == "ok" and r.get("mspe") is not None:
            key = (r["strategy"], r["model"], r["regime"])
            bucket[key].append(r["mspe"])

    print(f"{'strategy':<8}{'model':<14}{'regime':<14}"
          f"{'n_ok':>6}{'mean':>10}{'median':>10}{'mean/median':>14}")
    for key in sorted(bucket.keys()):
        s, m, rg = key
        xs = bucket[key]
        if not xs:
            continue
        mn = safe_mean(xs)
        md = safe_median(xs)
        ratio = mn / md if md and md > 0 else float("nan")
        print(f"{s:<8}{m:<14}{rg:<14}{len(xs):>6}"
              f"{fmt(mn):>10}{fmt(md):>10}{fmt(ratio, 2):>14}")
    print()


# --------------------------------------------------------------------------- #
# Table 3: Compute cost                                                        #
# --------------------------------------------------------------------------- #
def table3_compute_cost(results: list[dict]) -> None:
    print("=" * 86)
    print("TABLE 3: Compute cost — average tokens and wall-clock per problem")
    print("=" * 86)
    print()

    bucket = defaultdict(lambda: {"tokens": [], "elapsed": [], "llm_s": [],
                                    "tool_s": [], "n_tools": []})
    for r in results:
        key = (r["strategy"], r["model"])
        if r.get("total_tokens") is not None:
            bucket[key]["tokens"].append(r["total_tokens"])
        if r.get("elapsed_s") is not None:
            bucket[key]["elapsed"].append(r["elapsed_s"])
        if r.get("total_llm_seconds") is not None:
            bucket[key]["llm_s"].append(r["total_llm_seconds"])
        if r.get("total_tool_seconds") is not None:
            bucket[key]["tool_s"].append(r["total_tool_seconds"])
        if r.get("n_tool_calls") is not None:
            bucket[key]["n_tools"].append(r["n_tool_calls"])

    print(f"{'strategy':<8}{'model':<14}"
          f"{'tokens (mean)':>15}{'time (s)':>10}"
          f"{'LLM s':>10}{'tool s':>10}{'n_tools':>10}")
    for key in sorted(bucket.keys()):
        s, m = key
        b = bucket[key]
        toks   = safe_mean(b["tokens"])
        elap   = safe_mean(b["elapsed"])
        llm_s  = safe_mean(b["llm_s"])
        tool_s = safe_mean(b["tool_s"])
        nt     = safe_mean(b["n_tools"])
        print(f"{s:<8}{m:<14}"
              f"{toks:>15.0f}{elap:>10.1f}"
              f"{llm_s:>10.1f}{tool_s:>10.1f}{nt:>10.1f}")

    # Compute S3/S1 ratio per model
    print()
    print("Token and time overhead of S3 vs S1 (multiplier per model):")
    print(f"  {'model':<14}{'token ratio':>14}{'time ratio':>14}")
    s1_tokens = {}
    s3_tokens = {}
    s1_time = {}
    s3_time = {}
    for (s, m), b in bucket.items():
        if s == "S1":
            s1_tokens[m] = safe_mean(b["tokens"])
            s1_time[m] = safe_mean(b["elapsed"])
        elif s == "S3":
            s3_tokens[m] = safe_mean(b["tokens"])
            s3_time[m] = safe_mean(b["elapsed"])
    for m in sorted(set(s1_tokens.keys()) | set(s3_tokens.keys())):
        tok_r = s3_tokens.get(m, float("nan")) / s1_tokens.get(m, float("nan")) \
                if s1_tokens.get(m, 0) else float("nan")
        time_r = s3_time.get(m, float("nan")) / s1_time.get(m, float("nan")) \
                 if s1_time.get(m, 0) else float("nan")
        print(f"  {m:<14}{tok_r:>14.2f}{time_r:>14.2f}")
    print()


# --------------------------------------------------------------------------- #
# Table 4: Paired ratios vs baselines                                          #
# --------------------------------------------------------------------------- #
def table4_paired_ratios(results: list[dict],
                          problems_by_id: dict[str, dict]) -> None:
    print("=" * 86)
    print("TABLE 4: Paired ratio of MSPE (LLM / baseline), median across "
          "successful runs")
    print("=" * 86)
    print("Ratio < 1 means LLM beats baseline. Computed only where LLM produced")
    print("a valid coefficient vector (parse_status = ok).")
    print()

    bucket = defaultdict(lambda: {"vs_int_only": [], "vs_cv": [], "vs_oracle": []})
    for r in results:
        if r.get("parse_status") != "ok" or r.get("mspe") is None:
            continue
        prob = problems_by_id.get(r["problem_id"])
        if prob is None:
            continue
        m_dict = prob.get("mspe", {})
        regime = r["regime"]

        # Regime-appropriate internal-only baseline
        int_only_key = "restricted_internal" if regime == "restricted" else "ols_internal"
        int_only = m_dict.get(int_only_key)
        cv = m_dict.get("mainger_cv")
        oracle = m_dict.get("mainger_oracle")

        key = (r["strategy"], r["model"], r["regime"])
        if int_only and int_only > 0:
            bucket[key]["vs_int_only"].append(r["mspe"] / int_only)
        if cv and cv > 0:
            bucket[key]["vs_cv"].append(r["mspe"] / cv)
        if oracle and oracle > 0:
            bucket[key]["vs_oracle"].append(r["mspe"] / oracle)

    print(f"{'strategy':<8}{'model':<14}{'regime':<14}"
          f"{'n_ok':>6}{'vs int-only':>14}{'vs CV':>10}{'vs oracle':>12}")
    for key in sorted(bucket.keys()):
        s, m, rg = key
        b = bucket[key]
        nok = len(b["vs_int_only"])
        if nok == 0:
            continue
        print(f"{s:<8}{m:<14}{rg:<14}{nok:>6}"
              f"{fmt(safe_median(b['vs_int_only']), 3):>14}"
              f"{fmt(safe_median(b['vs_cv']), 3):>10}"
              f"{fmt(safe_median(b['vs_oracle']), 3):>12}")
    print()


# --------------------------------------------------------------------------- #
# Table 5: Non-LLM baselines for context                                       #
# --------------------------------------------------------------------------- #
def table5_baselines(problems: list[dict]) -> None:
    print("=" * 86)
    print("TABLE 5 (REFERENCE): Non-LLM baselines, MSPE on test set")
    print("=" * 86)
    print("Internal-only is regime-appropriate: OLS for full/partial, "
          "Sigma_ref^-1 r_int for restricted.")
    print()

    by_regime = defaultdict(lambda: {"int": [], "ext": [], "cv": [], "oracle": [],
                                       "t_ols": [], "t_cv": []})
    for p in problems:
        regime = p["regime"]
        m = p.get("mspe", {})
        t = p.get("baseline_time_s", {})
        int_key = "restricted_internal" if regime == "restricted" else "ols_internal"
        by_regime[regime]["int"].append(m.get(int_key))
        by_regime[regime]["ext"].append(m.get("external_only"))
        by_regime[regime]["cv"].append(m.get("mainger_cv"))
        by_regime[regime]["oracle"].append(m.get("mainger_oracle"))
        by_regime[regime]["t_ols"].append(t.get("ols_internal"))
        by_regime[regime]["t_cv"].append(t.get("mainger_cv"))

    print(f"{'regime':<14}{'stat':<8}"
          f"{'int-only':>10}{'ext-only':>10}{'main-CV':>10}{'main-oracle':>13}")
    for regime in by_regime:
        b = by_regime[regime]
        for stat_name, stat_fn in [("median", safe_median),
                                    ("mean",   safe_mean),
                                    ("SD",     safe_std)]:
            print(f"{regime:<14}{stat_name:<8}"
                  f"{fmt(stat_fn(b['int'])):>10}"
                  f"{fmt(stat_fn(b['ext'])):>10}"
                  f"{fmt(stat_fn(b['cv'])):>10}"
                  f"{fmt(stat_fn(b['oracle'])):>13}")
    print()

    # Baseline wall-clock per problem (recorded in
    # generate_simulation_problems.py via time.perf_counter()).
    print("Baseline wall-clock per problem (seconds):")
    print(f"  {'regime':<14}{'OLS (median)':>14}{'OLS (mean)':>14}"
          f"{'mainger-CV (median)':>22}{'mainger-CV (mean)':>20}")
    for regime in by_regime:
        b = by_regime[regime]
        print(f"  {regime:<14}"
              f"{fmt(safe_median(b['t_ols']), 4):>14}"
              f"{fmt(safe_mean(b['t_ols']),   4):>14}"
              f"{fmt(safe_median(b['t_cv']),  3):>22}"
              f"{fmt(safe_mean(b['t_cv']),    3):>20}")
    print()
    print("Note: mainger-CV time includes Rscript subprocess startup (~0.4 s)")
    print("on typical systems; pure mainger() computation is sub-100ms.")
    print()


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", required=True,
                    help="Path to sim_problems_*.jsonl")
    ap.add_argument("--results", nargs="+", required=True,
                    help="One or more llm_results_*.jsonl files")
    args = ap.parse_args()

    problems = load_jsonl(args.problems)
    print(f"Loaded {len(problems)} problems from {args.problems}\n")

    all_results = []
    for path in args.results:
        rows = load_jsonl(path)
        all_results.extend(rows)
        print(f"  Loaded {len(rows)} rows from {path}")
    print(f"Total result rows: {len(all_results)}\n")

    if not all_results:
        print("No results to aggregate. Run sim/run_llm_evaluation.py first.")
        return

    problems_by_id = {p["problem_id"]: p for p in problems}

    table1_headline(all_results, problems_by_id)
    table5_baselines(problems)
    table2_mean_vs_median(all_results)
    table4_paired_ratios(all_results, problems_by_id)
    table3_compute_cost(all_results)


if __name__ == "__main__":
    main()
