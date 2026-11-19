"""Summarize out-of-fold SHAP values by feature.

The input may be a matrix whose columns are feature names, or a table whose
SHAP columns are named shap__<feature>. The script writes mean absolute SHAP,
mean signed SHAP, relative importance and rank.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def summarize(input_csv: Path, output_csv: Path) -> None:
    values = pd.read_csv(input_csv, low_memory=False)
    shap_columns = [c for c in values.columns if c.startswith("shap__")]
    if shap_columns:
        feature_columns = {c: c[len("shap__"):] for c in shap_columns}
    else:
        numeric = values.select_dtypes(include="number")
        if numeric.empty:
            raise ValueError("No numeric SHAP columns found in the input file.")
        feature_columns = {c: c for c in numeric.columns}

    rows = []
    for column, feature in feature_columns.items():
        series = pd.to_numeric(values[column], errors="coerce").dropna()
        if series.empty:
            continue
        rows.append({
            "feature": feature,
            "mean_abs_shap": float(series.abs().mean()),
            "mean_signed_shap": float(series.mean()),
        })

    result = pd.DataFrame(rows)
    total = result["mean_abs_shap"].sum()
    result["relative_importance"] = result["mean_abs_shap"] / total if total else 0.0
    result = result.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    result["rank"] = result.index + 1
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    summarize(args.input, args.output)


if __name__ == "__main__":
    main()
