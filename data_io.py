"""
data_io.py
----------
Load user inputs into a canonical session dict and persist it as an RDS file
that the R bridge reads.

Supported formats per input
---------------------------
  Internal individual data:  CSV or Parquet, with header. First column = Y.
  External coefficients:     CSV or Parquet, two columns (variable, estimate).
                             Header optional; auto-detected.
  Sigma matrices:            CSV or Parquet, square. Header optional; auto-
                             detected.

Reduced-space behavior
----------------------
  When the external coefficient file uses variable names, alignment is by
  name. Internal predictors that have no external coefficient receive 0
  (zero-padded). External coefficients for variables not in the internal
  data are dropped with a warning. Sigma matrices, by contrast, must already
  be in the same dimension and order as the internal predictors; we do not
  attempt to subset Sigma matrices automatically.
"""
from __future__ import annotations

import json
import subprocess
import warnings
from pathlib import Path
from typing import Any

import pandas as pd


# --------------------------------------------------------------------------- #
# Format detection                                                             #
# --------------------------------------------------------------------------- #
def _read_table_with_header(path: str | Path) -> pd.DataFrame:
    """Load a tabular file expected to have a header row."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(p)
    sep = "\t" if suffix == ".tsv" else ","
    return pd.read_csv(p, sep=sep)


def _looks_like_header(row_values) -> bool:
    """Heuristic: a row is a header if any cell is non-numeric."""
    for v in row_values:
        if v is None or (isinstance(v, str) and not _is_numeric_string(v)):
            return True
    return False


def _is_numeric_string(s: str) -> bool:
    s = s.strip()
    if not s:
        return False
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def _read_table_auto_header(path: str | Path) -> tuple[pd.DataFrame, bool]:
    """Load a tabular file that may or may not have a header.

    Detects by reading the first row as data, then checking whether any
    cell is non-numeric. Returns (DataFrame, had_header).
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(p), True

    sep = "\t" if suffix == ".tsv" else ","
    raw = pd.read_csv(p, sep=sep, header=None, nrows=1)
    has_header = _looks_like_header(raw.iloc[0].tolist())
    df = pd.read_csv(p, sep=sep, header=0 if has_header else None)
    return df, has_header


def _read_matrix(path: str | Path) -> list[list[float]]:
    """Load a numeric matrix. Auto-detects header. Validates shape."""
    df, _ = _read_table_auto_header(path)
    arr = df.to_numpy()
    try:
        arr = arr.astype(float)
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Matrix file {path} contains non-numeric values: {e}"
        ) from e
    if arr.shape[0] != arr.shape[1]:
        raise ValueError(
            f"Matrix file {path} must be square. Got shape "
            f"{arr.shape[0]} x {arr.shape[1]}."
        )
    return arr.tolist()


# --------------------------------------------------------------------------- #
# Loaders                                                                      #
# --------------------------------------------------------------------------- #
def load_individual(path: str | Path) -> dict[str, Any]:
    """Load individual-level (X, Y) data.

    The file MUST have a header row. The first column is treated as the
    response Y and the remaining columns as predictors X.
    """
    df = _read_table_with_header(path)
    if df.shape[0] == 0:
        raise ValueError(f"Internal data file {path} is empty.")
    if df.shape[1] < 2:
        raise ValueError(
            f"Internal data file {path} must have at least two columns "
            f"(response Y followed by predictors). Got {df.shape[1]}."
        )

    # Validate numeric content
    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(
            f"Internal data file {path} contains non-numeric columns: "
            f"{non_numeric}. All columns must be numeric (Y followed by X)."
        )
    if df.isna().any().any():
        raise ValueError(
            f"Internal data file {path} contains missing values. "
            f"Drop or impute them before running the agent."
        )

    y = df.iloc[:, 0].to_numpy()
    x = df.iloc[:, 1:].to_numpy()
    names = df.columns[1:].tolist()
    return {
        "X_int": x.tolist(),
        "Y_int": y.tolist(),
        "predictor_names": names,
        "n_int": int(len(y)),
    }


def load_external_coef(path: str | Path) -> dict[str, Any]:
    """External coefficients as a 2-column file: variable, estimate.

    Header is auto-detected. Both columns are required even if the file
    has no header (in which case columns are taken positionally).
    """
    df, _ = _read_table_auto_header(path)
    if df.shape[1] < 2:
        raise ValueError(
            f"External coefficient file {path} must have two columns: "
            f"variable name and estimate. Got {df.shape[1]} column(s)."
        )

    try:
        estimates = df.iloc[:, 1].astype(float)
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"External coefficient file {path}: column 2 must be numeric. "
            f"({e})"
        ) from e

    coef = dict(zip(df.iloc[:, 0].astype(str), estimates))
    return {"beta_ext_named": coef}


def align_external(
    beta_ext_named: dict[str, float],
    predictor_names: list[str],
) -> tuple[list[float], list[str], list[str]]:
    """Order external coefs to match internal predictor order.

    Returns (aligned_vector, missing_externally, dropped_externally) where
      - missing_externally are internal predictors with no external coef
        (zero-padded in the output vector)
      - dropped_externally are external variables not present internally
    """
    aligned = [float(beta_ext_named.get(n, 0.0)) for n in predictor_names]
    missing = [n for n in predictor_names if n not in beta_ext_named]
    dropped = [n for n in beta_ext_named if n not in predictor_names]
    return aligned, missing, dropped


# --------------------------------------------------------------------------- #
# Session assembly                                                             #
# --------------------------------------------------------------------------- #
def build_session(
    *,
    internal_path: str | Path | None = None,
    internal_format: str | None = None,           # accepted for backward compat
    external_coef_path: str | Path | None = None,
    external_sigma_path: str | Path | None = None,
    reference_sigma_path: str | Path | None = None,
    sigma2_int: float | None = None,
    sigma2_ext: float | None = None,
    n_ext: int | None = None,
    manual: dict[str, Any] | None = None,
    base_session: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a session dict from whatever the user has.

    If `base_session` is provided, fields from new uploads OVERRIDE existing
    fields, but other keys carry forward.

    Validation messages are stored in `s["_warnings"]` for the agent and UI
    to surface to the user.
    """
    s: dict[str, Any] = {}
    warnings_list: list[str] = []
    if base_session:
        s = {k: v for k, v in base_session.items() if not k.startswith("_")}
        warnings_list = list(base_session.get("_warnings", []))

    if manual:
        s.update(manual)

    if internal_path:
        s.update(load_individual(internal_path))

    if external_coef_path:
        ext = load_external_coef(external_coef_path)
        names = s.get("predictor_names")
        if names:
            aligned, missing, dropped = align_external(
                ext["beta_ext_named"], names
            )
            s["beta_ext"] = aligned
            if missing:
                warnings_list.append(
                    f"{len(missing)} internal predictor(s) had no matching "
                    f"external coefficient (zero-padded): {missing[:5]}"
                    + ("..." if len(missing) > 5 else "")
                )
            if dropped:
                warnings_list.append(
                    f"{len(dropped)} external coefficient(s) were dropped "
                    f"because they are not in the internal data: "
                    f"{dropped[:5]}" + ("..." if len(dropped) > 5 else "")
                )
        else:
            s["beta_ext"] = list(ext["beta_ext_named"].values())
            warnings_list.append(
                "External coefficients used without name-based alignment "
                "because internal predictor names are unavailable."
            )

    if external_sigma_path:
        Sigma = _read_matrix(external_sigma_path)
        names = s.get("predictor_names")
        if names and len(Sigma) != len(names):
            raise ValueError(
                f"External Sigma matrix has dimension {len(Sigma)} but "
                f"internal data has {len(names)} predictors. The matrix "
                f"must already be aligned to the internal predictor order."
            )
        s["Sigma_ext"] = Sigma

    if reference_sigma_path:
        Sigma = _read_matrix(reference_sigma_path)
        names = s.get("predictor_names")
        if names and len(Sigma) != len(names):
            raise ValueError(
                f"Reference Sigma matrix has dimension {len(Sigma)} but "
                f"internal data has {len(names)} predictors. The matrix "
                f"must already be aligned to the internal predictor order."
            )
        s["Sigma_ref"] = Sigma

    if sigma2_int is not None: s["sigma2_int"] = sigma2_int
    if sigma2_ext is not None: s["sigma2_ext"] = sigma2_ext
    if n_ext      is not None: s["n_ext"]      = int(n_ext)

    if warnings_list:
        s["_warnings"] = warnings_list
    return s


# --------------------------------------------------------------------------- #
# Persistence (JSON -> RDS via R)                                              #
# --------------------------------------------------------------------------- #
R_PERSIST_SCRIPT = r"""suppressPackageStartupMessages(library(jsonlite))

args <- commandArgs(trailingOnly = TRUE)
in_path  <- args[1]
out_path <- args[2]

s <- fromJSON(in_path, simplifyVector = TRUE, simplifyMatrix = TRUE)

coerce_matrix <- function(x) {
  if (is.null(x)) return(NULL)
  if (is.matrix(x)) return(matrix(as.numeric(x), nrow = nrow(x), ncol = ncol(x)))
  if (is.data.frame(x)) return(as.matrix(x))
  if (is.list(x)) return(do.call(rbind, lapply(x, as.numeric)))
  if (is.numeric(x)) return(matrix(x, nrow = 1))
  stop("Cannot coerce field to matrix")
}
for (nm in c("X_int", "Sigma_int", "Sigma_ext", "Sigma_ref")) {
  if (!is.null(s[[nm]])) s[[nm]] <- coerce_matrix(s[[nm]])
}
for (nm in c("Y_int", "beta_int", "beta_ext", "r_int")) {
  if (!is.null(s[[nm]])) s[[nm]] <- as.numeric(s[[nm]])
}
for (nm in c("sigma2_int", "sigma2_ext")) {
  if (!is.null(s[[nm]])) s[[nm]] <- as.numeric(s[[nm]])
}
for (nm in c("n_int", "n_ext")) {
  if (!is.null(s[[nm]])) s[[nm]] <- as.integer(s[[nm]])
}

saveRDS(s, out_path)
cat("OK\n")
"""


def persist_session(session: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """Write the session as an RDS file the R bridge can read."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "session.json"
    rds_path  = out_dir / "session.rds"
    r_script  = out_dir / "_persist_session.R"

    payload = {k: v for k, v in session.items() if not k.startswith("_")}
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    r_script.write_text(R_PERSIST_SCRIPT, encoding="utf-8")

    proc = subprocess.run(
        ["Rscript", "--vanilla", str(r_script), str(json_path), str(rds_path)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 or not rds_path.exists():
        raise RuntimeError(
            "Failed to persist session as RDS.\n"
            f"  returncode: {proc.returncode}\n"
            f"  stdout:\n{proc.stdout}\n"
            f"  stderr:\n{proc.stderr}\n"
        )

    session["_path"] = str(rds_path)
    session["_metadata"] = {
        "n_int": session.get("n_int"),
        "n_ext": session.get("n_ext"),
        "p":     len(session.get("predictor_names", [])) or None,
        "predictor_names": session.get("predictor_names"),
        "has_internal_individual_data": "X_int" in session and "Y_int" in session,
        "has_internal_marginal_only":   "r_int" in session and "X_int" not in session,
        "has_external_theta":           "beta_ext" in session,
        "has_external_sigma2":          "Sigma_ext" in session,
        "has_reference_panel":          "Sigma_ref" in session,
        "warnings":                     session.get("_warnings", []),
    }
    return session
