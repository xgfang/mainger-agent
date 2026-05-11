"""
sft/generate_restricted_problems.py
====================================
Generate restricted-regime-only problems to complete the SFT corpus.

The original SFT collection produced 206 restricted problems but all
of them errored at trace-collection time due to a bug in the R bridge
(now fixed). This script generates fresh restricted problems using the
same broader distribution as the original generator (varying n, p,
BNR), so the corpus stays consistent.

Distribution (matching original sft/generate_problems.py):
  - p in {3, 5, 8, 10, 15, 20, 30, 50}
  - n/p in {4, 8, 15, 25, 50, 100, 200}
  - BNR low / moderate / high (50/30/20 weights)
  - All restricted regime

Outputs JSONL with the schema collect_teacher_traces.py expects.

Run:
  python sft/generate_restricted_problems.py \
      --n 270 \
      --output sft/data/restricted_problems.jsonl \
      --seed 7777
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


# Predictor pool, same as the original SFT generator (slightly expanded
# to cover p=50 with replace=False sampling)
PREDICTOR_POOL = [
    # Demographics & anthropometrics
    "age", "bmi", "glucose", "sbp", "dbp", "ldl", "hdl", "tg", "hba1c",
    "creatinine", "egfr", "insulin", "weight", "height", "waist",
    # Lifestyle
    "smoker", "alcohol", "exercise", "stress",
    # Demographic indicators
    "male", "education", "income", "married",
    # Genetic markers
    "snp1", "snp2", "snp3", "snp4", "snp5", "snp6", "snp7", "snp8",
    "snp9", "snp10", "snp11", "snp12",
    # Principal components
    "pc1", "pc2", "pc3", "pc4", "pc5", "pc6", "pc7",
    # Generic predictors
    "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10",
    # Additional clinical / lab values (added to ensure pool >= 50)
    "alt", "ast", "bun", "albumin", "vitd",
]

# Natural-language prompts matching the style of the existing SFT corpus.
# These are sampled per-problem to add the same variety the existing
# 756 clean traces have.
USER_MESSAGE_TEMPLATES = [
    "Run the mainger framework on this dataset and tell me what the recommended estimator is.",
    "Please produce the integration analysis with the appropriate tuning method.",
    "Help me integrate this external information using mainger.",
    "I have internal marginal correlations and a reference panel. What does mainger recommend?",
    "Please analyze my data and produce the integration report, code, and explanation.",
    "Apply the mainger framework and report the integrated estimator.",
    "Run the integration analysis and provide your recommendation.",
    "Use mainger to combine my internal summary with the external estimates.",
]

# Problem-distribution choices (broader than the simulation distribution)
P_VALUES = [3, 5, 8, 10, 15, 20, 30, 50]
NP_RATIOS = [4, 8, 15, 25, 50, 100, 200]
BNR_RANGES = [(1.0, 5.0), (5.0, 15.0), (15.0, 40.0)]
BNR_WEIGHTS = [0.5, 0.3, 0.2]


def random_correlation(p: int, rng: np.random.Generator,
                       drift: float = 0.0) -> np.ndarray:
    A = rng.normal(0, 0.3 + drift, size=(p, p))
    A = (A + A.T) / 2
    eig = np.linalg.eigvalsh(A)
    A = A + (abs(eig.min()) + 0.5) * np.eye(p)
    d = np.sqrt(np.diag(A))
    return A / np.outer(d, d)


def generate_one_restricted_problem(rng: np.random.Generator,
                                     problem_id: str) -> dict:
    """Generate one restricted-regime problem.

    Restricted regime exposes:
      - r_int = X' Y / n  (marginal correlations)
      - Sigma_ref         (reference panel covariance, uncentered X'X/n
                           from a separate independent draw of the
                           internal population)
      - beta_ext          (external coefficients)

    Internal X, Y are generated but NOT exposed to the user (only used
    for r_int).
    """
    p = int(rng.choice(P_VALUES))
    np_ratio = float(rng.choice(NP_RATIOS))
    n_int = max(int(np_ratio * p), p + 2)

    names = [str(n) for n in rng.choice(PREDICTOR_POOL, size=p, replace=False)]
    beta_true = rng.normal(0, 0.7, size=p)

    # Bias-to-noise: how far the external estimate is from internal truth
    bnr_idx = int(rng.choice(len(BNR_RANGES), p=BNR_WEIGHTS))
    target_bnr = float(rng.uniform(*BNR_RANGES[bnr_idx]))
    sigma_noise = 1.0
    perturb_std = float(np.sqrt(target_bnr * sigma_noise ** 2 / (n_int * p)))
    delta = rng.normal(0, perturb_std, size=p)
    beta_ext = beta_true + delta

    # Predictor covariance for internal population
    Sigma_design = random_correlation(p, rng)
    L = np.linalg.cholesky(Sigma_design)

    # Internal data (used only to compute r_int; not exposed)
    Z = rng.normal(0, 1, size=(n_int, p))
    X = Z @ L.T
    for j, nm in enumerate(names):
        if nm in ("smoker", "alcohol", "male", "married") and rng.random() < 0.6:
            X[:, j] = (X[:, j] > rng.uniform(-0.5, 0.5)).astype(float)
    Y = X @ beta_true + rng.normal(0, sigma_noise, size=n_int)
    r_int = X.T @ Y / n_int

    # Reference panel: separate independent draw of size 1000 from the
    # internal population. Uncentered X'X/n.
    n_ref = 1000
    Z_ref = rng.normal(0, 1, size=(n_ref, p))
    X_ref = Z_ref @ L.T
    for j, nm in enumerate(names):
        if nm in ("smoker", "alcohol", "male", "married") and \
                np.unique(X[:, j]).size <= 2:
            X_ref[:, j] = (X_ref[:, j] > rng.uniform(-0.5, 0.5)).astype(float)
    Sigma_ref = X_ref.T @ X_ref / n_ref

    return {
        "problem_id":          problem_id,
        "regime":              "restricted",
        "n_int":               n_int,
        "p":                   p,
        "predictor_names":     names,
        "beta_true":           [float(b) for b in beta_true],
        "beta_ext":            [float(b) for b in beta_ext],
        "bias_to_noise_ratio": target_bnr,
        "sigma_noise_true":    sigma_noise,
        "r_int":               [float(v) for v in r_int],
        "Sigma_ref":           [[float(v) for v in row] for row in Sigma_ref],
        "user_message":        str(rng.choice(USER_MESSAGE_TEMPLATES)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=270,
                    help="Number of restricted problems to generate")
    ap.add_argument("--output", type=str,
                    default="sft/data/restricted_problems.jsonl")
    ap.add_argument("--seed", type=int, default=7777)
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    with out_path.open("w", encoding="utf-8") as f_out:
        for i in range(1, args.n + 1):
            pid = f"r_{i:05d}"
            p = generate_one_restricted_problem(rng, pid)
            f_out.write(json.dumps(p) + "\n")
            if i % 30 == 0 or i == args.n:
                print(f"[{i}/{args.n}] generated {pid} (p={p['p']}, n={p['n_int']}, "
                      f"BNR={p['bias_to_noise_ratio']:.1f})")

    print(f"\nWrote {args.n} restricted-regime problems to {out_path}")
    print(f"Size: {out_path.stat().st_size / 1024:.1f} KB")
    print()
    print("Next step: run trace collection on these problems:")
    print(f"  python sft/collect_teacher_traces.py \\")
    print(f"      --problems {args.output} \\")
    print(f"      --output sft/data/teacher_traces_restricted.jsonl \\")
    print(f"      --rps 0.5")


if __name__ == "__main__":
    main()
