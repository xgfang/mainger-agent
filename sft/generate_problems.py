"""
sft/generate_problems.py
========================
Phase 1 of the SFT pipeline: generate synthetic problem specifications.

UPDATE: full-regime problems now ALWAYS include sigma2_ext and n_ext.
The mainger package's eta_bound_full() requires both for the formula to
evaluate correctly; if either is missing, mainger() silently falls back
to a wrong bound (grid_max=5). To prevent training the small model on
those broken traces, we ensure full-regime problems always supply them.

For partial and restricted regimes, sigma2_ext and n_ext are not used
by the bound formula, so we keep the existing 70%/50% provisioning.

Run locally:
  python sft/generate_problems.py --n_problems 50 --output sft/data/problems_pilot.jsonl
  python sft/generate_problems.py --n_problems 10000 --output sft/data/problems_full.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import numpy as np


# --------------------------------------------------------------------------- #
# Distribution of problem characteristics                                       #
# --------------------------------------------------------------------------- #
REGIMES = ["partial", "full", "restricted"]
REGIME_WEIGHTS = [0.5, 0.3, 0.2]

P_VALUES = [3, 5, 8, 10, 15, 20, 30, 50]
P_WEIGHTS = [0.10, 0.25, 0.20, 0.15, 0.10, 0.10, 0.05, 0.05]

NP_RATIOS = [4, 8, 15, 25, 50, 100, 200]
NP_RATIO_WEIGHTS = [0.05, 0.15, 0.25, 0.25, 0.15, 0.10, 0.05]

BNR_CATEGORIES = ["low", "moderate", "high"]
BNR_WEIGHTS = [0.4, 0.4, 0.2]
BNR_RANGES = {"low": (0.5, 5.0), "moderate": (5.0, 20.0), "high": (20.0, 80.0)}

# For partial/restricted regimes only. Full regime ALWAYS provides them.
PROVIDE_N_EXT_PROB_NONFULL = 0.7
PROVIDE_SIGMA2_EXT_PROB_NONFULL = 0.5

USER_MESSAGE_TEMPLATES = [
    "Please analyze my data and produce the integration report, code, and explanation.",
    "I'd like to integrate external information with my internal data. Can you help?",
    "Run the mainger framework on this dataset and tell me what the recommended estimator is.",
    "Give me a full analysis: detect the regime, pick eta, fit the integrated estimator.",
    "What does mainger say about integrating my data with these external coefficients?",
    "Please produce the integration analysis with the appropriate tuning method.",
]

PREDICTOR_POOL = [
    "age", "bmi", "glucose", "sbp", "dbp", "ldl", "hdl", "tg", "hba1c",
    "creatinine", "egfr", "insulin", "weight", "height", "waist",
    "smoker", "alcohol", "exercise", "stress",
    "male", "education", "income", "married",
    "snp1", "snp2", "snp3", "snp4", "snp5", "snp6", "snp7", "snp8",
    "pc1", "pc2", "pc3", "pc4", "pc5",
    "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10",
]


def generate_one_problem(rng: np.random.Generator, problem_id: str) -> dict:
    regime = rng.choice(REGIMES, p=REGIME_WEIGHTS)
    p = int(rng.choice(P_VALUES, p=P_WEIGHTS))
    np_ratio = float(rng.choice(NP_RATIOS, p=NP_RATIO_WEIGHTS))
    n_int = max(int(np_ratio * p), p + 2)

    if p <= len(PREDICTOR_POOL):
        names = list(rng.choice(PREDICTOR_POOL, size=p, replace=False))
    else:
        names = [f"x{i+1}" for i in range(p)]

    beta_int_true = rng.normal(0, 0.7, size=p)

    bnr_cat = rng.choice(BNR_CATEGORIES, p=BNR_WEIGHTS)
    target_bnr = float(rng.uniform(*BNR_RANGES[bnr_cat]))

    sigma_noise = 1.0
    target_bias_quad = target_bnr * (sigma_noise ** 2) / n_int
    perturb_std = float(np.sqrt(target_bias_quad / p))
    delta = rng.normal(0, perturb_std, size=p)
    beta_ext = beta_int_true + delta

    out: dict = {
        "problem_id": problem_id,
        "regime": str(regime),
        "n_int": int(n_int),
        "p": int(p),
        "predictor_names": names,
        "beta_ext": [float(b) for b in beta_ext],
        "bias_to_noise_ratio": target_bnr,
        "user_message": str(rng.choice(USER_MESSAGE_TEMPLATES)),
    }

    if regime in ("full", "partial"):
        Sigma_int_design = _random_correlation(p, rng)
        L = np.linalg.cholesky(Sigma_int_design)
        Z = rng.normal(0, 1, size=(n_int, p))
        X = Z @ L.T

        for j, nm in enumerate(names):
            if nm in ("smoker", "alcohol", "male", "married") and rng.random() < 0.7:
                threshold = float(rng.uniform(-0.5, 0.5))
                X[:, j] = (X[:, j] > threshold).astype(float)

        Y = X @ beta_int_true + rng.normal(0, sigma_noise, size=n_int)

        out["X_int"] = [[float(v) for v in row] for row in X]
        out["Y_int"] = [float(v) for v in Y]

    if regime == "full":
        # External Sigma + REQUIRED sigma2_ext + REQUIRED n_ext.
        # mainger::eta_bound_full() needs all three; missing any of
        # sigma2_ext or n_ext silently produces a wrong bound.
        Sigma_ext_design = _random_correlation(p, rng, drift=0.15)
        out["Sigma_ext"] = [[float(v) for v in row] for row in Sigma_ext_design]
        # Always include for full regime, span typical published-study sizes
        out["n_ext"] = int(np.exp(rng.uniform(np.log(1000), np.log(100000))))
        # External residual variance, log-uniform around the internal scale
        out["sigma2_ext"] = float(np.exp(rng.uniform(np.log(0.5), np.log(2.0))))

    if regime == "restricted":
        Sigma_int_design = _random_correlation(p, rng)
        L = np.linalg.cholesky(Sigma_int_design)
        Z = rng.normal(0, 1, size=(n_int, p))
        X = Z @ L.T
        Y = X @ beta_int_true + rng.normal(0, sigma_noise, size=n_int)
        r_int = X.T @ Y / n_int
        out["r_int"] = [float(v) for v in r_int]

        Sigma_ref_design = _random_correlation(p, rng, drift=0.10)
        out["Sigma_ref"] = [[float(v) for v in row] for row in Sigma_ref_design]

    # Partial and restricted: keep optional behavior (the bound for these
    # regimes doesn't depend on sigma2_ext or n_ext).
    if regime in ("partial", "restricted"):
        if rng.random() < PROVIDE_N_EXT_PROB_NONFULL:
            out["n_ext"] = int(np.exp(rng.uniform(np.log(1000), np.log(100000))))
        if rng.random() < PROVIDE_SIGMA2_EXT_PROB_NONFULL:
            out["sigma2_ext"] = float(rng.uniform(0.5, 2.0))

    return out


def _random_correlation(p: int, rng: np.random.Generator,
                        drift: float = 0.0) -> np.ndarray:
    A = rng.normal(0, 0.3 + drift, size=(p, p))
    A = (A + A.T) / 2
    eig = np.linalg.eigvalsh(A)
    A = A + (abs(eig.min()) + 0.5) * np.eye(p)
    d = np.sqrt(np.diag(A))
    A = A / np.outer(d, d)
    return A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_problems", type=int, default=50)
    ap.add_argument("--output", type=str, default="sft/data/problems.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    summary = {"partial": 0, "full": 0, "restricted": 0}
    full_with_sigma_n = 0
    total_size_bytes = 0

    print(f"Generating {args.n_problems} problems with seed {args.seed} ...")
    with out_path.open("w", encoding="utf-8") as f:
        for i in range(args.n_problems):
            pid = f"p_{i+1:05d}"
            problem = generate_one_problem(rng, pid)
            line = json.dumps(problem)
            f.write(line + "\n")
            total_size_bytes += len(line)
            summary[problem["regime"]] += 1
            if problem["regime"] == "full" \
                    and "sigma2_ext" in problem and "n_ext" in problem:
                full_with_sigma_n += 1

    print(f"\nWrote {args.n_problems} problems to {out_path}")
    print(f"  Size: {total_size_bytes / 1024 / 1024:.1f} MB")
    print(f"  Regime distribution:")
    for r, n in summary.items():
        pct = 100 * n / args.n_problems
        print(f"    {r}: {n} ({pct:.1f}%)")
    if summary["full"] > 0:
        print(f"  Full-regime problems with both sigma2_ext and n_ext: "
              f"{full_with_sigma_n}/{summary['full']} (should be 100%)")


if __name__ == "__main__":
    main()
