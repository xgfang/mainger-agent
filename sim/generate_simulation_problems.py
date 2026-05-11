"""
sim/generate_simulation_problems.py
====================================
Phase 1: simulation problems with attached test sets and pre-computed
baselines.

Key design decisions (revised):
  - For each replication (outer loop), draw ONE (X_int, Y_int, X_test,
    Y_test, beta_true). Then create three problems (full / partial /
    restricted) that share this internal data but expose different
    subsets of it. This makes within-replication comparisons across
    regimes fair.
  - The "internal-only baseline" is regime-appropriate:
      full / partial: OLS on (X_int, Y_int)
      restricted: Sigma_ref^{-1} r_int (the moment-based estimator the
        user actually has access to)
  - Oracle eta selection minimizes test-set MSPE directly (NOT plain L2
    in coefficient space), so it is a true upper bound on what mainger
    can achieve.
  - MSPE = bias-squared (matching the user's reference scripts):
      MSPE(beta_hat) = || X_test (beta_hat - beta_true) ||^2 / n_test

Run:
  python sim/generate_simulation_problems.py \
      --n_per_regime 100 \
      --output sim/data/sim_problems_moderate.jsonl \
      --seed 2025
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np


# Predictor names matching the reference R scripts' x_z structure.
PREDICTOR_NAMES = ["intercept", "x1", "x2", "x3", "x4", "x5", "x1_x3_interaction"]
P_TOTAL = 7

# Sample sizes (matching the reference scripts)
INTERNAL_N    = 50
TEST_N        = 1000
EXTERNAL_N    = 1000
SIGMA_NOISE   = 1.5

# Internal population
INTERNAL_MU      = 0.0
INTERNAL_BETA_C  = 0.5
INTERNAL_BETA_X  = [0.5] * 6

# Moderate Heterogeneity setting
MODERATE_MU_EXT     = 0.5
MODERATE_BETA_C_EXT = 0.5
MODERATE_BETA_X_EXT = [0.6] * 6
MODERATE_EXT_INDEX  = list(range(0, 6))   # 0-indexed; equals 1:6 in R


# --------------------------------------------------------------------------- #
# Data generator (mirrors data_generator() in the reference R scripts)         #
# --------------------------------------------------------------------------- #
def data_generator(rng: np.random.Generator, n: int, mu: float,
                   beta_c: float, beta_x: list[float]) -> dict:
    x1 = rng.exponential(scale=1.0, size=n)

    cov_mat = np.full((4, 4), 0.3)
    np.fill_diagonal(cov_mat, 1.0)
    L = np.linalg.cholesky(cov_mat)
    Z = rng.normal(0, 1, size=(n, 4))
    x2_5 = Z @ L.T + mu

    x2_5[:, 1] = (x2_5[:, 1] > 0.7 * x2_5[:, 0]).astype(float)

    intercept = np.ones(n)
    interaction = x1 * x2_5[:, 1]
    x_z = np.column_stack([intercept, x1, x2_5, interaction])

    beta = np.array([beta_c] + list(beta_x))
    y = x_z @ beta + rng.normal(0, SIGMA_NOISE, size=n)

    return {"X": x_z, "Y": y, "beta_true": beta}


def compute_external_beta(rng: np.random.Generator,
                          mu_ext: float, beta_c_ext: float,
                          beta_x_ext: list[float],
                          external_index: list[int]) -> np.ndarray:
    ext = data_generator(rng, EXTERNAL_N, mu_ext, beta_c_ext, beta_x_ext)
    X_sub = ext["X"][:, external_index]
    Y     = ext["Y"]
    beta_sub = np.linalg.lstsq(X_sub, Y, rcond=None)[0]
    beta_full = np.zeros(P_TOTAL)
    beta_full[external_index] = beta_sub
    return beta_full


def compute_reference_panel(rng: np.random.Generator) -> np.ndarray:
    """Independent draw from the INTERNAL population, used as Sigma_ref
    for restricted regime. Returns X'X/n (uncentered)."""
    ref = data_generator(rng, EXTERNAL_N,
                         INTERNAL_MU, INTERNAL_BETA_C, INTERNAL_BETA_X)
    return ref["X"].T @ ref["X"] / EXTERNAL_N


# --------------------------------------------------------------------------- #
# MSPE/MSE conventions matching the reference scripts                          #
# --------------------------------------------------------------------------- #
def eval_mspe(beta: list[float] | None, X_test: np.ndarray,
              Y_test: np.ndarray, beta_true: list[float]) -> float | None:
    if beta is None: return None
    try:
        b  = np.array(beta, dtype=float)
        bt = np.array(beta_true, dtype=float)
        if b.shape != bt.shape: return None
        Xd = X_test @ (b - bt)
        return float(np.dot(Xd, Xd) / X_test.shape[0])
    except Exception:
        return None


def eval_mse(beta: list[float] | None,
             beta_true: list[float]) -> float | None:
    if beta is None: return None
    try:
        b  = np.array(beta, dtype=float)
        bt = np.array(beta_true, dtype=float)
        if b.shape != bt.shape: return None
        return float(np.sum((b - bt) ** 2))
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# R bridge call: mainger-CV (or eAIC) plus test-MSPE-based oracle              #
# --------------------------------------------------------------------------- #
R_MAINGER_BASELINE = r"""
suppressPackageStartupMessages({
  library(jsonlite); library(mainger)
})
args <- commandArgs(trailingOnly = TRUE)
inp <- fromJSON(args[1], simplifyVector = TRUE, simplifyMatrix = TRUE)

to_matrix <- function(x) {
  if (is.matrix(x)) return(matrix(as.numeric(x), nrow = nrow(x), ncol = ncol(x)))
  if (is.data.frame(x)) return(as.matrix(x))
  if (is.list(x)) return(do.call(rbind, lapply(x, as.numeric)))
  stop("Cannot coerce to matrix")
}

regime    <- inp$regime
beta_ext  <- as.numeric(inp$beta_ext)
beta_true <- as.numeric(inp$beta_true)
X_test    <- to_matrix(inp$X_test)
Y_test    <- as.numeric(inp$Y_test)
n_test    <- nrow(X_test)

# ---- Build call_args appropriate for the regime ----
call_args <- list(beta_ext = beta_ext)

if (regime == "restricted") {
  call_args$r_int     <- as.numeric(inp$r_int)
  call_args$Sigma_ref <- to_matrix(inp$Sigma_ref)
  if (!is.null(inp$n_int))      call_args$n_int      <- as.integer(inp$n_int)
  if (!is.null(inp$sigma2_int)) call_args$sigma2_int <- as.numeric(inp$sigma2_int)
} else {
  if (!is.null(inp$X_int) && !is.null(inp$Y_int)) {
    call_args$X_int <- to_matrix(inp$X_int)
    call_args$Y_int <- as.numeric(inp$Y_int)
  }
  if (!is.null(inp$Sigma_ext))  call_args$Sigma_ext  <- to_matrix(inp$Sigma_ext)
  if (!is.null(inp$sigma2_ext)) call_args$sigma2_ext <- as.numeric(inp$sigma2_ext)
  if (!is.null(inp$n_ext))      call_args$n_ext      <- as.integer(inp$n_ext)
  if (!is.null(inp$n_int))      call_args$n_int      <- as.integer(inp$n_int)
}

default_tuning <- if (regime == "restricted") "eaic" else "cv"
if (!is.null(inp$cv_seed) && default_tuning == "cv") set.seed(as.integer(inp$cv_seed))

# ----------------------------------------------------------------
# Compute the search-range basis. Each regime uses its own
# theoretical eta_bound:
#   partial    : eta_bound_partial(beta_int, beta_ext, Sigma_int, sigma2_int)
#   full       : eta_bound_full(Sigma_int, Sigma_ext, delta,
#                               sigma2_int, sigma2_ext, n_int, n_ext)
#   restricted : eta_bound_partial(beta_int_imputed, beta_ext,
#                                  Sigma_ref, sigma2_int)
# search_max = max(5, 1.5 * basis_bound) in every case (build_grid below).
# ----------------------------------------------------------------
basis_bound <- NA_real_
basis_source <- "package_default"

if (regime == "partial") {
  # Internal X, Y are present; mainger() will compute the bound itself.
  # We replicate by calling eta_bound_partial directly.
  X_int <- call_args$X_int
  Y_int <- call_args$Y_int
  beta_int <- as.numeric(solve(crossprod(X_int)) %*% crossprod(X_int, Y_int))
  Sigma_int <- crossprod(X_int) / nrow(X_int)
  resid <- Y_int - X_int %*% beta_int
  sigma2_int <- as.numeric(crossprod(resid)) / max(1, nrow(X_int) - ncol(X_int))
  basis_bound <- tryCatch(
    mainger::eta_bound_partial(beta_int = beta_int, beta_ext = beta_ext,
                                Sigma_int = Sigma_int, sigma2_int = sigma2_int),
    error = function(e) NA_real_
  )
  basis_source <- "partial_bound"
} else if (regime == "full") {
  # Use the partial-bound formula even for the full regime, to match
  # the patched bridge (r_helpers/run_mainger.R, May 2026). The
  # full-regime bound eta_bound_full(.) is derived through worst-case
  # operator-norm arguments and is empirically much looser than the
  # partial bound; on the OPTN eGFR demo eta_bound_full = 0.10 sat 8x
  # below the test minimum (eta ~ 0.7) while eta_bound_partial = 0.85
  # sat at the test minimum. The partial bound is therefore used as
  # the search-grid basis in every regime.
  X_int <- call_args$X_int
  Y_int <- call_args$Y_int
  beta_int <- as.numeric(solve(crossprod(X_int)) %*% crossprod(X_int, Y_int))
  Sigma_int <- crossprod(X_int) / nrow(X_int)
  resid <- Y_int - X_int %*% beta_int
  sigma2_int <- as.numeric(crossprod(resid)) / max(1, nrow(X_int) - ncol(X_int))
  basis_bound <- tryCatch(
    mainger::eta_bound_partial(beta_int = beta_int, beta_ext = beta_ext,
                                Sigma_int = Sigma_int, sigma2_int = sigma2_int),
    error = function(e) NA_real_
  )
  basis_source <- "partial_bound_for_full"
} else if (regime == "restricted") {
  # Use partial-style bound with imputed beta_int = Sigma_ref^{-1} r_int.
  # We compute the imputation inline rather than relying on a possibly
  # private helper from the package.
  beta_int_imputed <- tryCatch(
    as.numeric(solve(call_args$Sigma_ref) %*% call_args$r_int),
    error = function(e) NULL
  )
  sigma2_int_used <- if (!is.null(call_args$sigma2_int))
                       call_args$sigma2_int else 1
  if (!is.null(beta_int_imputed)) {
    basis_bound <- tryCatch(
      mainger::eta_bound_partial(beta_int = beta_int_imputed,
                                  beta_ext = beta_ext,
                                  Sigma_int = call_args$Sigma_ref,
                                  sigma2_int = sigma2_int_used),
      error = function(e) NA_real_
    )
  }
  basis_source <- "partial_bound_restricted"
}

# Construct the soft-bound search grid: [0, 1.5*basis_bound] when the
# bound is finite/positive, else fall back to [0, 5]. This mirrors the
# patched bridge in r_helpers/run_mainger.R (May 2026). The earlier
# `max(5, 1.5*bound)` floor let CV chase eta well past the empirical
# beneficial range on small-n problems; the partial bound is now used
# as the basis even in full regime (see eta_bound_partial branch above)
# and is empirically tight, so the 1.5x soft margin is sufficient.
build_grid <- function(bound, total = 100) {
  if (is.finite(bound) && bound > 0) {
    search_max <- 1.5 * bound
  } else {
    search_max <- 5
  }
  if (!is.finite(bound) || bound <= 0) {
    g <- exp(seq(log(0.001), log(search_max), length.out = total - 1))
    return(sort(unique(c(0, g))))
  }
  z1 <- exp(seq(log(bound / 1000), log(bound), length.out = 59))
  z1 <- c(0, z1)
  z2 <- seq(bound * 1.001, search_max, length.out = total - length(z1))
  sort(unique(c(z1, z2)))
}
eta_grid_search <- build_grid(basis_bound, total = 100)
search_max_used <- max(eta_grid_search)

# ----------------------------------------------------------------
# Regime-aware CV (or eAIC for restricted).
#
# The mainger package's tune_cv ALWAYS refits using est_partial during
# fold loops, regardless of regime. That makes CV-selected eta identical
# across regimes when given the same internal data, which is not what we
# want -- we want each regime to select eta that's optimal for its OWN
# coefficient formula.
#
# We implement custom CV here that calls the regime-appropriate
# est_full / est_partial / est_restricted at each fold's per-eta refit.
# eAIC for restricted is unchanged from the package's implementation
# since restricted regime has no individual data to do CV on.
# ----------------------------------------------------------------

regime_aware_cv <- function(X_int, Y_int, beta_ext, eta_grid, regime,
                            Sigma_ext = NULL, sigma2_ext = NULL,
                            n_ext = NULL, nfolds = 5, seed = NULL) {
  if (!is.null(seed)) set.seed(as.integer(seed))
  n <- nrow(X_int); p <- ncol(X_int)
  folds <- sample(rep(1:nfolds, length.out = n))
  cv_errs <- matrix(0, nrow = length(eta_grid), ncol = nfolds)

  for (k in 1:nfolds) {
    X_tr <- X_int[folds != k, , drop = FALSE]
    Y_tr <- Y_int[folds != k]
    X_te <- X_int[folds == k, , drop = FALSE]
    Y_te <- Y_int[folds == k]

    n_tr <- nrow(X_tr)
    XtX_tr <- crossprod(X_tr)
    beta_int_tr <- as.numeric(MASS::ginv(XtX_tr) %*% crossprod(X_tr, Y_tr))
    Sigma_int_tr <- XtX_tr / n_tr
    resid_tr <- Y_tr - X_tr %*% beta_int_tr
    sigma2_int_tr <- sum(resid_tr^2) / max(n_tr - p, 1)

    for (i in seq_along(eta_grid)) {
      e <- eta_grid[i]
      bp <- if (regime == "full") {
        mainger::est_full(beta_int_tr, Sigma_int_tr, beta_ext,
                          Sigma_ext, e)
      } else {
        mainger::est_partial(beta_int_tr, beta_ext, e)
      }
      cv_errs[i, k] <- mean((Y_te - X_te %*% bp)^2)
    }
  }
  mean_cv <- rowMeans(cv_errs)
  best_idx <- which.min(mean_cv)
  list(eta = eta_grid[best_idx], cv_errors = mean_cv)
}

# Run the appropriate selection routine for this regime
if (regime %in% c("full", "partial")) {
  cv_result <- tryCatch(
    regime_aware_cv(call_args$X_int, call_args$Y_int, beta_ext,
                    eta_grid_search, regime,
                    Sigma_ext = call_args$Sigma_ext,
                    sigma2_ext = call_args$sigma2_ext,
                    n_ext = call_args$n_ext,
                    seed = inp$cv_seed),
    error = function(e) list(error = conditionMessage(e))
  )
  if (is.null(cv_result$error)) {
    selected_eta <- cv_result$eta
    # Refit on full internal data at selected eta with regime-specific formula
    if (regime == "full") {
      X_int <- call_args$X_int; Y_int <- call_args$Y_int
      n <- nrow(X_int); p <- ncol(X_int)
      beta_int_full <- as.numeric(MASS::ginv(crossprod(X_int)) %*%
                                   crossprod(X_int, Y_int))
      Sigma_int_full <- crossprod(X_int) / n
      cv_coefs_vec <- mainger::est_full(beta_int_full, Sigma_int_full,
                                          beta_ext, call_args$Sigma_ext,
                                          selected_eta)
    } else {
      X_int <- call_args$X_int; Y_int <- call_args$Y_int
      beta_int_full <- as.numeric(MASS::ginv(crossprod(X_int)) %*%
                                   crossprod(X_int, Y_int))
      cv_coefs_vec <- mainger::est_partial(beta_int_full, beta_ext, selected_eta)
    }
    fit_cv <- list(eta = selected_eta, coefficients = cv_coefs_vec,
                    eta_bound = basis_bound)
  } else {
    fit_cv <- cv_result
  }
} else {
  # Restricted: use package's eAIC (operates on r_int + Sigma_ref)
  cv_args <- call_args
  cv_args$tuning <- "eaic"
  cv_args$eta_grid <- eta_grid_search
  fit_cv <- tryCatch(do.call(mainger::mainger, cv_args),
                     error = function(e) list(error = conditionMessage(e)))
}

# ---- ORACLE: pick eta on the same grid that minimizes TEST-MSPE ----
mspe_for_eta <- function(eta) {
  fa <- call_args; fa$tuning <- "fixed"; fa$eta <- eta
  f <- tryCatch(do.call(mainger::mainger, fa), error = function(e) NULL)
  if (is.null(f)) return(NA)
  beta_hat <- as.numeric(f$coefficients)
  if (length(beta_hat) != length(beta_true)) return(NA)
  diff <- X_test %*% (beta_hat - beta_true)
  sum(diff^2) / n_test
}

oracle_eta <- NA; oracle_beta <- NULL; oracle_mspe <- Inf
for (eta in eta_grid_search) {
  m <- mspe_for_eta(eta)
  if (!is.na(m) && m < oracle_mspe) {
    oracle_mspe <- m
    oracle_eta  <- eta
    fa <- call_args; fa$tuning <- "fixed"; fa$eta <- eta
    f <- tryCatch(do.call(mainger::mainger, fa), error = function(e) NULL)
    if (!is.null(f)) oracle_beta <- as.numeric(f$coefficients)
  }
}

result <- list(
  cv_eta            = if (!is.null(fit_cv$eta))       unname(fit_cv$eta)       else NA,
  cv_eta_bound      = if (!is.null(fit_cv$eta_bound)) unname(fit_cv$eta_bound) else NA,
  cv_coefs          = if (!is.null(fit_cv$coefficients)) as.numeric(fit_cv$coefficients) else NULL,
  basis_bound       = basis_bound,
  basis_source      = basis_source,
  search_max        = search_max_used,
  oracle_eta        = oracle_eta,
  oracle_coefs      = if (!is.null(oracle_beta)) oracle_beta else NULL,
  oracle_mspe       = if (is.finite(oracle_mspe)) oracle_mspe else NA,
  cv_error          = if (!is.null(fit_cv$error)) fit_cv$error else NULL
)
cat(toJSON(result, auto_unbox = TRUE, na = "null", null = "null", digits = 12), "\n")
"""


def compute_mainger_baselines(problem: dict, cv_seed: int) -> dict[str, Any]:
    """Run mainger with CV/eAIC and the test-MSPE-based oracle."""
    payload: dict[str, Any] = {
        "regime":     problem["regime"],
        "beta_ext":   problem["beta_ext"],
        "beta_true":  problem["beta_true"],
        "X_test":     problem["X_test"],
        "Y_test":     problem["Y_test"],
        "n_int":      problem["n_int"],
        "cv_seed":    cv_seed,
    }
    if problem["regime"] == "restricted":
        payload["r_int"]     = problem.get("r_int")
        payload["Sigma_ref"] = problem.get("Sigma_ref")
    else:
        payload["X_int"] = problem.get("X_int")
        payload["Y_int"] = problem.get("Y_int")
    for key in ("Sigma_ext", "sigma2_ext", "n_ext"):
        if key in problem:
            payload[key] = problem[key]

    with tempfile.TemporaryDirectory() as tmp:
        in_path = Path(tmp) / "in.json"
        sc_path = Path(tmp) / "_baseline.R"
        in_path.write_text(json.dumps(payload), encoding="utf-8")
        sc_path.write_text(R_MAINGER_BASELINE, encoding="utf-8")
        proc = subprocess.run(
            ["Rscript", "--vanilla", str(sc_path), str(in_path)],
            capture_output=True, text=True, timeout=180,
        )
        if proc.returncode != 0:
            return {"error": proc.stderr.strip() or proc.stdout.strip()}
        try:
            return json.loads(proc.stdout.strip().split("\n")[-1])
        except json.JSONDecodeError as e:
            return {"error": f"JSON parse: {e}\nstdout: {proc.stdout}"}


# --------------------------------------------------------------------------- #
# Per-replication assembly: shared internal data across regimes                #
# --------------------------------------------------------------------------- #
def assemble_problems_for_replication(rep_idx: int,
                                       internal_data: dict,
                                       test_data: dict,
                                       beta_ext_full: np.ndarray,
                                       Sigma_ref: np.ndarray,
                                       Sigma_ext_for_full: np.ndarray,
                                       regimes: list[str]) -> list[dict]:
    """For one replication, build one problem per regime, all sharing the
    same internal/test data and external coefs. Each problem exposes only
    what its regime makes available."""

    base = {
        "n_int":            INTERNAL_N,
        "n_test":            TEST_N,
        "p":                P_TOTAL,
        "predictor_names":  PREDICTOR_NAMES,
        "beta_true":        [float(b) for b in internal_data["beta_true"]],
        "beta_ext":         [float(b) for b in beta_ext_full],
        "sigma_noise_true": SIGMA_NOISE,
        "X_test":           [[float(v) for v in row] for row in test_data["X"]],
        "Y_test":           [float(v) for v in test_data["Y"]],
        "heterogeneity":    "moderate",
        "replication":      rep_idx,
    }

    problems = []
    for regime in regimes:
        p = dict(base)
        p["regime"]     = regime
        p["problem_id"] = f"sim_{regime[:3]}_{rep_idx:04d}"

        if regime in ("full", "partial"):
            p["X_int"] = [[float(v) for v in row] for row in internal_data["X"]]
            p["Y_int"] = [float(v) for v in internal_data["Y"]]

        if regime == "full":
            p["Sigma_ext"]  = [[float(v) for v in row] for row in Sigma_ext_for_full]
            p["n_ext"]      = EXTERNAL_N
            p["sigma2_ext"] = float(SIGMA_NOISE ** 2)

        if regime == "restricted":
            r_int = internal_data["X"].T @ internal_data["Y"] / INTERNAL_N
            p["r_int"]     = [float(v) for v in r_int]
            p["Sigma_ref"] = [[float(v) for v in row] for row in Sigma_ref]
            # Eval-only: not exposed to the agent; used only by baseline
            # bookkeeping for cross-method consistency
            p["_X_int_eval_only"] = [[float(v) for v in row] for row in internal_data["X"]]
            p["_Y_int_eval_only"] = [float(v) for v in internal_data["Y"]]

        problems.append(p)
    return problems


# --------------------------------------------------------------------------- #
# Baselines                                                                    #
# --------------------------------------------------------------------------- #
def compute_simple_baselines(problem: dict,
                             Sigma_ref: np.ndarray | None) -> dict[str, Any]:
    """Compute analytical baselines:
      - ols_internal: OLS on (X_int, Y_int) — full/partial regimes only
      - external_only: beta_ext as-is
      - restricted_internal: Sigma_ref^{-1} r_int — restricted regime only
        (this is the moment-based estimator the user actually has access
        to, not OLS on individual data)
    """
    out: dict[str, Any] = {"external_only": list(problem["beta_ext"])}

    regime = problem["regime"]

    if regime in ("full", "partial"):
        X = np.array(problem["X_int"])
        Y = np.array(problem["Y_int"])
        beta_ols = np.linalg.lstsq(X, Y, rcond=None)[0]
        out["ols_internal"] = [float(b) for b in beta_ols]
        # No restricted_internal baseline for full/partial
        out["restricted_internal"] = None

    if regime == "restricted":
        r_int = np.array(problem["r_int"])
        S_ref = Sigma_ref if Sigma_ref is not None \
                else np.array(problem["Sigma_ref"])
        try:
            beta_restricted_internal = np.linalg.solve(S_ref, r_int)
            out["restricted_internal"] = [float(b) for b in beta_restricted_internal]
        except np.linalg.LinAlgError:
            out["restricted_internal"] = None
        # No ols_internal baseline for restricted (no individual data!)
        out["ols_internal"] = None

    return out


def regime_internal_baseline_key(regime: str) -> str:
    """Which baseline serves as the regime-appropriate 'internal-only'
    reference."""
    return "restricted_internal" if regime == "restricted" else "ols_internal"


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per_regime", type=int, default=100)
    ap.add_argument("--output", type=str,
                    default="sim/data/sim_problems_moderate.jsonl")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--regimes", nargs="+",
                    default=["full", "partial", "restricted"])
    ap.add_argument("--cv_seed", type=int, default=42)
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Fixed external coef (matches the reference script's setting)
    rng_ext = np.random.default_rng(args.seed + 999)
    beta_ext_full = compute_external_beta(
        rng_ext,
        mu_ext=MODERATE_MU_EXT,
        beta_c_ext=MODERATE_BETA_C_EXT,
        beta_x_ext=MODERATE_BETA_X_EXT,
        external_index=MODERATE_EXT_INDEX,
    )
    print(f"Fixed external beta (moderate setting):")
    for nm, v in zip(PREDICTOR_NAMES, beta_ext_full):
        print(f"  {nm:>22s}: {v:+.4f}")

    # Fixed reference panel (separate independent draw from internal pop)
    rng_ref = np.random.default_rng(args.seed + 7777)
    Sigma_ref = compute_reference_panel(rng_ref)
    print(f"Reference panel computed (uncentered, {Sigma_ref.shape}).\n")

    # Fixed Sigma_ext for full regime (one draw, used across replications)
    rng_sext = np.random.default_rng(args.seed + 5555)
    sext_data = data_generator(rng_sext, EXTERNAL_N,
                                MODERATE_MU_EXT, MODERATE_BETA_C_EXT,
                                MODERATE_BETA_X_EXT)
    Sigma_ext_for_full = sext_data["X"].T @ sext_data["X"] / EXTERNAL_N

    summary = {r: {"int": [], "ext": [], "cv": [], "oracle": []}
               for r in args.regimes}
    eta_summary = {r: {"cv_eta": [], "oracle_eta": [], "eta_bound": [],
                       "basis_bound": [], "search_max": []}
                   for r in args.regimes}

    n_total = args.n_per_regime
    total_problems = n_total * len(args.regimes)
    counter = 0

    # Outer loop is REPLICATION; inner loop is REGIME.
    # Each replication produces three problems sharing the same internal data.
    rng_inner = np.random.default_rng(args.seed)

    with out_path.open("w", encoding="utf-8") as f_out:
        for rep_idx in range(1, n_total + 1):
            internal = data_generator(rng_inner, INTERNAL_N,
                                       INTERNAL_MU, INTERNAL_BETA_C, INTERNAL_BETA_X)
            test     = data_generator(rng_inner, TEST_N,
                                       INTERNAL_MU, INTERNAL_BETA_C, INTERNAL_BETA_X)

            problems = assemble_problems_for_replication(
                rep_idx, internal, test,
                beta_ext_full, Sigma_ref, Sigma_ext_for_full,
                args.regimes,
            )

            for p in problems:
                counter += 1
                if counter % 30 == 1 or counter == total_problems:
                    print(f"[{counter}/{total_problems}] {p['problem_id']} ({p['regime']}) ...")

                # Wall-clock timing of the non-LLM baselines. The OLS time
                # measures pure numpy.lstsq; the mainger-CV time includes
                # Rscript subprocess startup (~0.4s on typical systems),
                # which mirrors the deployment overhead the LLM cells
                # also incur (API roundtrip / vLLM startup is similarly
                # included on the LLM side).
                t_simple = time.perf_counter()
                simple = compute_simple_baselines(p, Sigma_ref)
                t_simple = time.perf_counter() - t_simple

                t_mainger = time.perf_counter()
                mainger = compute_mainger_baselines(p, args.cv_seed)
                t_mainger = time.perf_counter() - t_mainger

                # If the R bridge errored, print a one-time diagnostic so
                # we don't silently fill NaN for an entire regime.
                if mainger.get("error") and counter % 30 == 1:
                    err_msg = str(mainger.get("error"))[:300]
                    print(f"  [WARN] R bridge error for {p['problem_id']}: {err_msg}")
                if mainger.get("cv_error") and counter % 30 == 1:
                    print(f"  [WARN] mainger() error for {p['problem_id']}: "
                          f"{mainger['cv_error'][:300]}")

                p["baselines"] = {
                    "ols_internal":         simple.get("ols_internal"),
                    "external_only":        simple.get("external_only"),
                    "restricted_internal":  simple.get("restricted_internal"),
                    "mainger_cv":           mainger.get("cv_coefs"),
                    "mainger_oracle":       mainger.get("oracle_coefs"),
                    "mainger_cv_eta":       mainger.get("cv_eta"),
                    "mainger_cv_bound":     mainger.get("cv_eta_bound"),
                    "mainger_oracle_eta":   mainger.get("oracle_eta"),
                    "mainger_error":        mainger.get("cv_error"),
                }
                p["baseline_time_s"] = {
                    # `simple` computes OLS + external + restricted-internal
                    # in one call; the analytical baselines are dominated by
                    # the OLS / linalg.solve invocation.
                    "ols_internal": t_simple,
                    "mainger_cv":   t_mainger,
                }

                X_test = np.array(p["X_test"])
                Y_test = np.array(p["Y_test"])

                p["mspe"] = {
                    k: eval_mspe(p["baselines"].get(k), X_test, Y_test, p["beta_true"])
                    for k in ("ols_internal", "external_only", "restricted_internal",
                              "mainger_cv", "mainger_oracle")
                }
                p["mse_vs_true"] = {
                    k: eval_mse(p["baselines"].get(k), p["beta_true"])
                    for k in ("ols_internal", "external_only", "restricted_internal",
                              "mainger_cv", "mainger_oracle")
                }

                # Track summary for the headline table
                int_key = regime_internal_baseline_key(p["regime"])
                int_val = p["mspe"].get(int_key)
                if int_val is not None and np.isfinite(int_val):
                    summary[p["regime"]]["int"].append(int_val)
                for short, full_key in [("ext", "external_only"),
                                        ("cv",  "mainger_cv"),
                                        ("oracle", "mainger_oracle")]:
                    v = p["mspe"].get(full_key)
                    if v is not None and np.isfinite(v):
                        summary[p["regime"]][short].append(v)

                # Track eta diagnostics
                cv_eta = mainger.get("cv_eta")
                or_eta = mainger.get("oracle_eta")
                bd     = mainger.get("cv_eta_bound")
                bb     = mainger.get("basis_bound")
                sm     = mainger.get("search_max")
                if cv_eta is not None: eta_summary[p["regime"]]["cv_eta"].append(cv_eta)
                if or_eta is not None: eta_summary[p["regime"]]["oracle_eta"].append(or_eta)
                if bd     is not None: eta_summary[p["regime"]]["eta_bound"].append(bd)
                if bb     is not None: eta_summary[p["regime"]]["basis_bound"].append(bb)
                if sm     is not None: eta_summary[p["regime"]]["search_max"].append(sm)

                f_out.write(json.dumps(p) + "\n")
                f_out.flush()

    print(f"\nWrote {total_problems} problems to {out_path}")
    sz_mb = out_path.stat().st_size / 1024 / 1024
    print(f"Size: {sz_mb:.1f} MB")

    # ----------- Summary tables -----------
    print("\nMean MSPE on test set (bias-squared, regime-appropriate baselines):")
    print(f"  {'regime':<12} {'int-only':>10} {'ext-only':>10} {'main-CV':>10} {'main-oracle':>13}")
    for r in args.regimes:
        s = summary[r]
        ib = float(np.mean(s["int"]))    if s["int"]    else float("nan")
        ext= float(np.mean(s["ext"]))    if s["ext"]    else float("nan")
        cv = float(np.mean(s["cv"]))     if s["cv"]     else float("nan")
        orc= float(np.mean(s["oracle"])) if s["oracle"] else float("nan")
        print(f"  {r:<12} {ib:>10.4f} {ext:>10.4f} {cv:>10.4f} {orc:>13.4f}")
    print("\nNote: 'int-only' uses regime-appropriate internal baseline")
    print("  full/partial: OLS on (X_int, Y_int)")
    print("  restricted:   Sigma_ref^{-1} r_int (moment-based, no individual data)")

    # Eta diagnostics — useful for understanding tuning behavior
    print("\nEta diagnostics by regime:")
    print(f"  {'regime':<12} {'cv_eta':>10} {'oracle_eta':>12} {'pkg_bound':>11} {'basis_bnd':>11} {'search_max':>11}")
    print(f"  {'(means)':<12}")
    for r in args.regimes:
        e = eta_summary[r]
        cve = float(np.mean(e["cv_eta"]))     if e["cv_eta"]     else float("nan")
        ore = float(np.mean(e["oracle_eta"])) if e["oracle_eta"] else float("nan")
        bd  = float(np.mean(e["eta_bound"]))  if e["eta_bound"]  else float("nan")
        bb  = float(np.mean(e["basis_bound"])) if e["basis_bound"] else float("nan")
        sm  = float(np.mean(e["search_max"])) if e["search_max"] else float("nan")
        print(f"  {r:<12} {cve:>10.4f} {ore:>12.4f} {bd:>11.4f} {bb:>11.4f} {sm:>11.4f}")
    print()
    print("  pkg_bound  : eta_bound returned by mainger() (its internal estimate)")
    print("  basis_bnd  : the bound we use to construct the search grid")
    print("               (partial-style bound, used for all regimes)")
    print("  search_max : max(5, 1.5 * basis_bnd) — top of the search range")

    print("\nSanity checks:")
    print("  - main-oracle MSPE should be <= main-CV MSPE in every regime")
    print("    (since oracle picks eta to minimize TEST-set MSPE directly)")
    print("  - For restricted, if ext-only is much better than int-only,")
    print("    we expect main-CV to land closer to ext-only than to int-only")
    print("  - If oracle_eta >> cv_eta, the tuning method is selecting too small")


if __name__ == "__main__":
    main()
