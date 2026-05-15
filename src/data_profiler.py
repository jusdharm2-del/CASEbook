"""
data_profiler.py
Computes a rich statistical profile of a pandas DataFrame.
No external dependencies beyond pandas and numpy.
"""

import pandas as pd
import numpy as np
from typing import Any


def profile_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """
    Returns a structured profile dict with:
      - overview  : row/col counts, memory, null rate
      - columns   : per-column stats
      - sample    : first 5 rows as list of dicts
      - dtypes    : raw dtype info
    """
    total_cells = df.shape[0] * df.shape[1]
    null_cells = df.isnull().sum().sum()

    overview = {
        "total_rows": int(df.shape[0]),
        "total_cols": int(df.shape[1]),
        "null_rate_pct": round(null_cells / total_cells * 100, 1) if total_cells else 0,
        "memory_kb": round(df.memory_usage(deep=True).sum() / 1024, 1),
        "duplicate_rows": int(df.duplicated().sum()),
    }

    columns = []
    for col in df.columns:
        series = df[col]
        null_count = int(series.isnull().sum())
        null_pct = round(null_count / len(series) * 100, 1) if len(series) else 0
        unique_count = int(series.nunique(dropna=True))
        non_null = series.dropna()

        # Infer semantic type
        col_type = _infer_type(series, unique_count)

        # Sample values (up to 5 unique non-null)
        sample_vals = [str(v) for v in non_null.unique()[:5].tolist()]

        col_info: dict[str, Any] = {
            "name": col,
            "dtype": str(series.dtype),
            "type": col_type,
            "null_count": null_count,
            "null_pct": null_pct,
            "unique": unique_count,
            "sample": sample_vals,
        }

        # Numeric stats
        if col_type == "numeric" and len(non_null) > 0:
            col_info.update(
                {
                    "mean": _safe_round(non_null.mean()),
                    "std": _safe_round(non_null.std()),
                    "min": _safe_round(non_null.min()),
                    "max": _safe_round(non_null.max()),
                    "median": _safe_round(non_null.median()),
                    "skew": _safe_round(non_null.skew()),
                }
            )
        # Categorical stats
        elif col_type == "categorical" and len(non_null) > 0:
            top = non_null.value_counts().head(3)
            col_info["top_values"] = {str(k): int(v) for k, v in top.items()}

        columns.append(col_info)

    # Sample rows (first 5, serialise-safe)
    sample_rows = df.head(5).replace({np.nan: None}).to_dict(orient="records")
    sample_rows = [{k: _serialise(v) for k, v in row.items()} for row in sample_rows]

    return {
        "overview": overview,
        "columns": columns,
        "sample": sample_rows,
        "column_names": list(df.columns),
    }


def _infer_type(series: pd.Series, unique_count: int) -> str:
    """Classify a column as numeric, categorical, datetime, or text."""
    dtype = series.dtype

    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"

    if pd.api.types.is_numeric_dtype(dtype):
        # Binary → treat as categorical
        if unique_count <= 2:
            return "categorical"
        return "numeric"

    # Object / string columns
    non_null = series.dropna()
    if len(non_null) == 0:
        return "text"

    # Try to parse as datetime
    if _looks_like_datetime(non_null):
        return "datetime"

    # Try to coerce to numeric
    coerced = pd.to_numeric(non_null, errors="coerce")
    numeric_ratio = coerced.notna().sum() / len(non_null)
    if numeric_ratio > 0.8:
        return "numeric"

    # Categorical vs free text
    if unique_count <= 30 or (unique_count / max(len(series), 1)) < 0.05:
        return "categorical"

    return "text"


def _looks_like_datetime(series: pd.Series) -> bool:
    sample = series.head(20).astype(str)
    dt_count = 0
    for val in sample:
        try:
            pd.to_datetime(val, dayfirst=False)
            dt_count += 1
        except (ValueError, TypeError):
            pass
    return dt_count / len(sample) > 0.7


def _safe_round(val: Any, decimals: int = 4) -> Any:
    try:
        return round(float(val), decimals)
    except Exception:
        return None


def _serialise(val: Any) -> Any:
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    return val


def quality_report(df: pd.DataFrame, profile: dict) -> dict:
    """
    Generates a structured data quality report with warnings across 5 categories:
      - missing       : high null rates
      - imbalance     : skewed class distributions in categorical columns
      - skewness      : highly skewed numeric distributions
      - leakage_risk  : columns that may leak target information
      - cardinality   : columns too unique or too constant to be useful
    Returns a dict ready to serialise as JSON.
    """
    warnings_list = []
    score = 100  # starts perfect, deducted per issue

    col_map = {c["name"]: c for c in profile["columns"]}

    # ── 1. Missing values ─────────────────────────────────────────────────────
    for c in profile["columns"]:
        pct = c["null_pct"]
        if pct >= 50:
            warnings_list.append(
                {
                    "category": "missing",
                    "severity": "high",
                    "column": c["name"],
                    "message": f"{pct}% of values are missing — consider dropping or imputing carefully.",
                    "fix": "Drop column if >70%; otherwise use median/mode imputation or a missingness indicator.",
                }
            )
            score -= 10
        elif pct >= 20:
            warnings_list.append(
                {
                    "category": "missing",
                    "severity": "medium",
                    "column": c["name"],
                    "message": f"{pct}% missing values detected.",
                    "fix": "Impute with median (numeric) or most-frequent (categorical). Add a binary flag column.",
                }
            )
            score -= 5

    # ── 2. Class imbalance (categorical columns) ──────────────────────────────
    for c in profile["columns"]:
        if c["type"] != "categorical" or c["unique"] < 2:
            continue
        top_vals = c.get("top_values", {})
        if not top_vals:
            continue
        total_non_null = df[c["name"]].count()
        top_count = list(top_vals.values())[0]
        top_pct = round(top_count / total_non_null * 100, 1) if total_non_null else 0
        if top_pct >= 90:
            warnings_list.append(
                {
                    "category": "imbalance",
                    "severity": "high",
                    "column": c["name"],
                    "message": f"Severely imbalanced — dominant class covers {top_pct}% of rows.",
                    "fix": "Use SMOTE/oversampling, class_weight='balanced', or treat rare classes as 'Other'.",
                }
            )
            score -= 10
        elif top_pct >= 75:
            warnings_list.append(
                {
                    "category": "imbalance",
                    "severity": "medium",
                    "column": c["name"],
                    "message": f"Moderately imbalanced — dominant class covers {top_pct}% of rows.",
                    "fix": "Monitor precision/recall per class. Consider stratified sampling.",
                }
            )
            score -= 5

    # ── 3. Numeric skewness ───────────────────────────────────────────────────
    for c in profile["columns"]:
        if c["type"] != "numeric":
            continue
        skew = c.get("skew")
        if skew is None:
            continue
        abs_skew = abs(skew)
        if abs_skew >= 3:
            warnings_list.append(
                {
                    "category": "skewness",
                    "severity": "high",
                    "column": c["name"],
                    "message": f"Highly skewed (skew={skew:.2f}) — likely contains outliers or a power-law distribution.",
                    "fix": "Apply log1p or Box-Cox transform. Cap outliers at the 99th percentile.",
                }
            )
            score -= 8
        elif abs_skew >= 1:
            warnings_list.append(
                {
                    "category": "skewness",
                    "severity": "medium",
                    "column": c["name"],
                    "message": f"Moderately skewed (skew={skew:.2f}).",
                    "fix": "Consider sqrt or log transform before modelling.",
                }
            )
            score -= 3

    # ── 4. Leakage risk ───────────────────────────────────────────────────────
    # Only flag columns whose names strongly suggest they are derived FROM or
    # ARE the target variable — not ID columns (those are obvious to any user).
    _LEAKAGE_KEYWORDS = [
        "target",
        "label",
        "outcome",
        "result",
        "response",
        "flag",
        "indicator",
        "final",
        "actual",
        "ground_truth",
        "y_true",
        "answer",
    ]
    # Keywords that imply post-event derivation — dangerous when predicting future
    _POST_EVENT_KEYWORDS = [
        "days_overdue",
        "overdue",
        "days_late",
        "write_off",
        "written_off",
        "default_flag",
        "loss_given",
        "recovery",
        "resolved",
        "closed",
        "post_",
        "_post",
        "after_",
        "_after",
    ]

    for c in profile["columns"]:
        name_lower = c["name"].lower()

        # Column name directly matches a target-like keyword
        if any(
            name_lower == kw
            or name_lower.startswith(kw + "_")
            or name_lower.endswith("_" + kw)
            for kw in _LEAKAGE_KEYWORDS
        ):
            warnings_list.append(
                {
                    "category": "leakage_risk",
                    "severity": "high",
                    "column": c["name"],
                    "message": f"'{c['name']}' appears to be a target or label column. Including it as a feature will cause target leakage — the model learns to copy the answer.",
                    "fix": "If this IS your target, set it as TARGET_COL and exclude it from FEATURE_COLS. If it's derived from the target, drop it entirely.",
                }
            )
            score -= 10

        # Column name implies it was computed after the event being predicted
        elif any(kw in name_lower for kw in _POST_EVENT_KEYWORDS):
            warnings_list.append(
                {
                    "category": "leakage_risk",
                    "severity": "high",
                    "column": c["name"],
                    "message": f"'{c['name']}' suggests a post-event derived metric — this value likely doesn't exist at the time you need to make predictions.",
                    "fix": "Verify this column is available before the event occurs. If it's computed after, exclude it from FEATURE_COLS.",
                }
            )
            score -= 8

        # Column name loosely matches — softer warning
        elif any(kw in name_lower for kw in _LEAKAGE_KEYWORDS):
            warnings_list.append(
                {
                    "category": "leakage_risk",
                    "severity": "medium",
                    "column": c["name"],
                    "message": f"'{c['name']}' may be correlated with or derived from the target variable.",
                    "fix": "Confirm this column is available at prediction time and is not computed from the outcome you're trying to predict.",
                }
            )
            score -= 4

    # ── 5. Cardinality issues ─────────────────────────────────────────────────
    for c in profile["columns"]:
        if c["type"] == "categorical":
            if c["unique"] > 50:
                warnings_list.append(
                    {
                        "category": "cardinality",
                        "severity": "medium",
                        "column": c["name"],
                        "message": f"High cardinality — {c['unique']} unique categories. One-hot encoding will create {c['unique']} new columns.",
                        "fix": "Use target encoding, frequency encoding, or group rare categories into 'Other' (threshold <1%).",
                    }
                )
                score -= 4
            elif c["unique"] == 1:
                warnings_list.append(
                    {
                        "category": "cardinality",
                        "severity": "high",
                        "column": c["name"],
                        "message": "Constant column — only 1 unique value. Carries zero information.",
                        "fix": "Drop this column entirely before modelling.",
                    }
                )
                score -= 6

    # ── Summary counts ────────────────────────────────────────────────────────
    severity_counts = {"high": 0, "medium": 0, "low": 0}
    category_counts: dict[str, int] = {}
    for w in warnings_list:
        severity_counts[w["severity"]] = severity_counts.get(w["severity"], 0) + 1
        category_counts[w["category"]] = category_counts.get(w["category"], 0) + 1

    overall_score = max(0, min(100, score))
    if overall_score >= 80:
        grade, grade_label = "A", "Good"
    elif overall_score >= 60:
        grade, grade_label = "B", "Fair"
    elif overall_score >= 40:
        grade, grade_label = "C", "Poor"
    else:
        grade, grade_label = "D", "Critical"

    return {
        "score": overall_score,
        "grade": grade,
        "grade_label": grade_label,
        "total_warnings": len(warnings_list),
        "severity_counts": severity_counts,
        "category_counts": category_counts,
        "warnings": warnings_list,
    }


def profile_to_text(profile: dict) -> str:
    """Convert profile to a compact text representation for the AI prompt."""
    ov = profile["overview"]
    lines = [
        f"Rows: {ov['total_rows']:,}  |  Columns: {ov['total_cols']}  |  "
        f"Null rate: {ov['null_rate_pct']}%  |  Duplicates: {ov['duplicate_rows']}",
        "",
        "Column details:",
    ]
    for c in profile["columns"]:
        stat_str = ""
        if c["type"] == "numeric":
            stat_str = (
                f"min={c.get('min')} max={c.get('max')} "
                f"mean={c.get('mean')} skew={c.get('skew')}"
            )
        elif c["type"] == "categorical":
            top = c.get("top_values", {})
            stat_str = "top: " + ", ".join(
                f"{k}({v})" for k, v in list(top.items())[:3]
            )
        lines.append(
            f"  {c['name']}: {c['type']}, nulls={c['null_pct']}%, "
            f"unique={c['unique']}  {stat_str}"
        )

    lines += ["", "Sample rows (first 5):"]
    for row in profile.get("sample", [])[:5]:
        lines.append("  " + str(row))

    return "\n".join(lines)
