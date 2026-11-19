"""Public machine-learning method-testing entry point.

The workflow uses city-grouped nested cross-validation and CatBoost candidates.
The external project modules imported below must be supplied when the full
event-level analysis is rerun.
"""
from __future__ import annotations

"""Recompute the CatBoost-derived data needed by the 2026-08-12 figures.

The script deliberately keeps tuning, held-out explanation, and packaging in one
auditable workflow.  Long nested-CV jobs are resumable at the outer-fold level.
"""

import argparse
import hashlib
import json
import platform
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import yaml
from catboost import Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from ml_shap.group_increment_analysis import (
    paired_group_bootstrap,
    resolve_ablation_definitions,
    resolve_conditional_addition_definitions,
    resolve_group_membership,
    resolve_stage_definitions,
)
from run_nested_group_tuning_31 import TARGETS, load_prepared_snapshot, run_target_model


CODE_ROOT = Path(__file__).resolve().parent
SOURCE_31 = Path("data/ml_inputs/model_comparison_31drivers_loss_group5fold")
SOURCE_38 = Path("data/ml_inputs/Day_warm_v2_all_event_environment_compact_no_ntl_collinearity")
TUNED_31 = Path("outputs/nested_group_tuning_31drivers_loss_seed42")
OUTPUT_ROOT = Path("outputs/nested_catboost_figure_updates_31drivers_loss_seed42")
PACKAGE_ROOT = Path("outputs/ml_figure_update_data")
GROUP_CONFIG = CODE_ROOT / "config_day_warm_compact_cross_group.yaml"
SEED = 42
OUTER_FOLDS = 5
INNER_FOLDS = 3
N_BOOTSTRAP = 5000


@dataclass(frozen=True)
class Variant:
    name: str
    label: str
    kind: str
    order: int
    features: tuple[str, ...]
    groups: tuple[str, ...] = ()
    alias_of: str | None = None


def write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def read_features(path: Path) -> list[str]:
    return list(json.loads(path.read_text(encoding="utf-8"))["feature_names"])


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "bias": float(np.mean(y_pred - y_true)),
    }


def load_exact_snapshot(source: Path, target: str, expected_features: int):
    feature_names = read_features(source / target / "feature_names.json")
    if len(feature_names) != expected_features:
        raise ValueError(f"{target}: expected {expected_features} features, found {len(feature_names)}")
    frame = pd.read_csv(source / target / "prepared_features.csv", low_memory=False)
    X = frame.iloc[:, -expected_features:].copy()
    stripped = [c.rsplit(".", 1)[0] if c.rsplit(".", 1)[-1].isdigit() else c for c in X.columns]
    if stripped != feature_names:
        raise ValueError(f"{target}: final feature block does not match feature_names.json")
    X.columns = feature_names
    X = X.astype(float)
    y = pd.to_numeric(frame["target"], errors="raise").astype(float)
    groups = frame["uid"].astype(str)
    meta = frame[[c for c in ["uid", "year", "event_id", "mode"] if c in frame]].copy()
    meta.insert(0, "row_index", np.arange(len(frame), dtype=int))
    if len(frame) != 9673 or groups.nunique() != 417:
        raise ValueError(f"Unexpected snapshot: rows={len(frame)}, cities={groups.nunique()}")
    return X, y, groups, meta


def audit_31_38_alignment(target: str) -> dict[str, Any]:
    X31, y31, g31, m31 = load_exact_snapshot(SOURCE_31, target, 31)
    X38, y38, g38, m38 = load_exact_snapshot(SOURCE_38, target, 38)
    if not m31.astype(str).equals(m38.astype(str)):
        raise ValueError(f"{target}: 31- and 38-feature event order differs")
    if not np.allclose(y31, y38, rtol=0, atol=0, equal_nan=True):
        raise ValueError(f"{target}: 31- and 38-feature targets differ")
    if not g31.equals(g38):
        raise ValueError(f"{target}: 31- and 38-feature city groups differ")
    common = [c for c in X31 if c in X38]
    if common != list(X31.columns):
        raise ValueError(f"{target}: the 31-feature set is not nested in the 38-feature set")
    max_common_abs_difference = 0.0
    for feature in common:
        finite = np.isfinite(X31[feature].to_numpy(float)) & np.isfinite(X38[feature].to_numpy(float))
        if finite.any():
            max_common_abs_difference = max(
                max_common_abs_difference,
                float(np.max(np.abs(X31.loc[finite, feature] - X38.loc[finite, feature]))),
            )
        # CSV round trips can differ in the final binary digit.  The tolerance
        # is far below the measurement precision and is recorded in the audit.
        if not np.allclose(X31[feature], X38[feature], rtol=0, atol=1e-12, equal_nan=True):
            raise ValueError(f"{target}: common feature differs: {feature}")
    return {
        "target": target,
        "n_rows": len(X31),
        "n_cities": g31.nunique(),
        "n_features_full": len(X38.columns),
        "n_features_reduced": len(X31.columns),
        "removed_features": [c for c in X38 if c not in X31],
        "events_targets_groups_and_common_features_identical": True,
        "common_feature_absolute_tolerance": 1e-12,
        "maximum_observed_common_feature_absolute_difference": max_common_abs_difference,
    }


def load_catboost_candidates() -> list[dict[str, Any]]:
    path = TUNED_31 / "all_candidate_configurations.json"
    candidates = json.loads(path.read_text(encoding="utf-8"))["catboost"]
    if len(candidates) != 32:
        raise ValueError(f"Expected 32 CatBoost candidates, found {len(candidates)}")
    return candidates


def resolve_variants(X: pd.DataFrame) -> tuple[pd.DataFrame, list[Variant]]:
    cfg = yaml.safe_load(GROUP_CONFIG.read_text(encoding="utf-8"))
    membership, group_to_features = resolve_group_membership(X, cfg)
    stages = resolve_stage_definitions(cfg, group_to_features)
    baseline, conditionals = resolve_conditional_addition_definitions(cfg, stages, group_to_features)
    ablations = resolve_ablation_definitions(cfg, stages[-1]["features"], group_to_features)

    variants: list[Variant] = []
    canonical = list(X.columns)

    def ordered(features: Iterable[str]) -> tuple[str, ...]:
        selected = set(features)
        return tuple(feature for feature in canonical if feature in selected)

    for stage in stages:
        variants.append(Variant(stage["name"], stage["label"], "progressive", int(stage["order"]),
                                ordered(stage["features"]), tuple(stage["groups"])))
    for definition in conditionals:
        alias = "M4_urban_environment" if definition["variant_name"] == "conditional_add_all_urban_environment" else None
        variants.append(Variant(definition["variant_name"], definition["label"], "conditional",
                                int(definition["order"]), ordered(definition["features"]),
                                tuple(definition["groups"]), alias))
    for order, definition in enumerate(ablations, start=1):
        alias = "M3_background_climate" if definition["name"] == "urban_environment" else None
        variants.append(Variant(f"without_{definition['name']}", definition["label"], "ablation", order,
                                ordered(definition["features"]), tuple(definition["groups"]), alias))
    if baseline is None or baseline["name"] != "M3_background_climate":
        raise ValueError("The conditional baseline must be M3_background_climate")
    return membership, variants


def run_subset_nested(n_jobs: int, overwrite: bool) -> None:
    candidates = load_catboost_candidates()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    audits: list[dict[str, Any]] = []
    for target in TARGETS:
        X, y, groups, meta, audit = load_prepared_snapshot(SOURCE_31, target)
        membership, variants = resolve_variants(X)
        target_root = OUTPUT_ROOT / "feature_subsets" / target
        target_root.mkdir(parents=True, exist_ok=True)
        membership.to_csv(target_root / "feature_group_membership.csv", index=False)
        variant_rows = []
        for variant in variants:
            variant_rows.append({
                "variant": variant.name, "label": variant.label, "kind": variant.kind,
                "order": variant.order, "n_features": len(variant.features),
                "groups": "|".join(variant.groups), "alias_of": variant.alias_of or "",
                "features": "|".join(variant.features),
            })
        pd.DataFrame(variant_rows).to_csv(target_root / "variant_definitions.csv", index=False)
        outer = GroupKFold(n_splits=OUTER_FOLDS)
        outer_splits = list(outer.split(X, y, groups))
        for variant in variants:
            if variant.alias_of or variant.name == "M4_urban_environment":
                continue
            print(f"\n[{target}] nested tuning {variant.name}: {len(variant.features)} features", flush=True)
            run_target_model(
                output_dir=target_root / "nested_runs", target=variant.name, model_name="catboost",
                X=X.loc[:, list(variant.features)], y=y, groups=groups, meta=meta,
                candidates=candidates, outer_splits=outer_splits, inner_folds=INNER_FOLDS,
                n_jobs=n_jobs, seed=SEED, overwrite=overwrite,
            )
        audits.append({**audit, "variants": len(variants),
                       "feature_order": "canonical order from the formal 31-feature snapshot",
                       "formal_M4_reused": True,
                       "independently_tuned_unique_variants":
                       sum(v.alias_of is None and v.name != "M4_urban_environment" for v in variants)})
    write_json(audits, OUTPUT_ROOT / "feature_subsets" / "data_audit.json")


def run_full38_nested(n_jobs: int, overwrite: bool) -> None:
    candidates = load_catboost_candidates()
    audits = []
    for target in TARGETS:
        alignment = audit_31_38_alignment(target)
        X, y, groups, meta = load_exact_snapshot(SOURCE_38, target, 38)
        outer_splits = list(GroupKFold(n_splits=OUTER_FOLDS).split(X, y, groups))
        run_target_model(
            output_dir=OUTPUT_ROOT / "full38", target=target, model_name="catboost",
            X=X, y=y, groups=groups, meta=meta, candidates=candidates,
            outer_splits=outer_splits, inner_folds=INNER_FOLDS, n_jobs=n_jobs,
            seed=SEED, overwrite=overwrite,
        )
        audits.append(alignment)
    write_json(audits, OUTPUT_ROOT / "full38" / "data_audit.json")


def _variant_prediction_path(target: str, variant: Variant) -> Path:
    actual = variant.alias_of or variant.name
    # M4 is the manuscript's formal 31-feature CatBoost model.  Reuse its exact
    # saved OOF predictions so every main/supplementary panel reports one model.
    if actual == "M4_urban_environment":
        return TUNED_31 / target / "catboost" / "fold_predictions.csv"
    return (OUTPUT_ROOT / "feature_subsets" / target / "nested_runs" / actual /
            "catboost" / "fold_predictions.csv")


def _wide_predictions(target: str, variants: Sequence[Variant]) -> pd.DataFrame:
    base: pd.DataFrame | None = None
    identity_cols = ["row_index", "uid", "y_true"]
    for variant in variants:
        pred = pd.read_csv(_variant_prediction_path(target, variant))
        if base is None:
            base = pred[identity_cols].copy()
        else:
            check = pred[identity_cols].copy()
            if not base[identity_cols].equals(check):
                raise ValueError(f"{target}/{variant.name}: OOF rows do not align")
        base[f"pred__{variant.name}"] = pred["y_pred"].to_numpy(float)
    assert base is not None
    return base


def _ci(values: pd.Series) -> tuple[float, float, float]:
    clean = values.dropna()
    return float(clean.mean()), float(clean.quantile(0.025)), float(clean.quantile(0.975))


def summarize_subsets(target: str, variants: Sequence[Variant], out: Path) -> None:
    wide = _wide_predictions(target, variants)
    wide.to_csv(out / f"{target}_all_variant_oof_predictions.csv", index=False)
    boot = paired_group_bootstrap(wide, [v.name for v in variants], "uid", N_BOOTSTRAP, SEED, "fast_r2")
    boot.to_csv(out / f"{target}_city_block_bootstrap_r2.csv", index=False)
    observed = {v.name: metric_dict(wide["y_true"].to_numpy(float), wide[f"pred__{v.name}"].to_numpy(float))["r2"]
                for v in variants}
    pivot = boot.pivot(index="bootstrap", columns="variant", values="r2")

    stages = sorted([v for v in variants if v.kind == "progressive"], key=lambda v: v.order)
    rows = []
    for v in stages:
        mean, lo, hi = _ci(pivot[v.name])
        rows.append({"paper_outcome": target.replace("_loss_log", ""), "target": target,
                     "stage_order": v.order, "stage": v.name, "stage_label": v.label,
                     "n_features": len(v.features), "r2": observed[v.name],
                     "r2_bootstrap_mean": mean, "ci_lower": lo, "ci_upper": hi,
                     "n_bootstrap": N_BOOTSTRAP})
    pd.DataFrame(rows).to_csv(out / f"{target}_progressive_stage_r2.csv", index=False)

    rows = []
    for before, after in zip(stages[:-1], stages[1:]):
        delta = pivot[after.name] - pivot[before.name]
        mean, lo, hi = _ci(delta)
        rows.append({"paper_outcome": target.replace("_loss_log", ""), "target": target,
                     "increment_order": after.order - 1, "from_stage": before.name,
                     "to_stage": after.name, "increment_label": after.label,
                     "delta_r2": observed[after.name] - observed[before.name],
                     "delta_r2_bootstrap_mean": mean, "ci_lower": lo, "ci_upper": hi,
                     "n_bootstrap": N_BOOTSTRAP})
    pd.DataFrame(rows).to_csv(out / f"{target}_progressive_incremental_delta_r2.csv", index=False)

    baseline = "M3_background_climate"
    rows = []
    for v in sorted([v for v in variants if v.kind == "conditional"], key=lambda v: v.order):
        delta = pivot[v.name] - pivot[baseline]
        mean, lo, hi = _ci(delta)
        rows.append({"paper_outcome": target.replace("_loss_log", ""), "target": target,
                     "addition_order": v.order, "baseline_stage": baseline,
                     "addition": v.name.removeprefix("conditional_add_"),
                     "addition_label": "Urban morphology" if v.label == "Urban form and scale" else v.label,
                     "added_groups": "|".join(v.groups),
                     "n_added_features": len(set(v.features) - set(next(x.features for x in stages if x.name == baseline))),
                     "r2_baseline": observed[baseline], "r2_with_addition": observed[v.name],
                     "delta_r2": observed[v.name] - observed[baseline],
                     "delta_r2_bootstrap_mean": mean, "ci_lower": lo, "ci_upper": hi,
                     "n_bootstrap": N_BOOTSTRAP})
    pd.DataFrame(rows).to_csv(out / f"{target}_conditional_addition_summary.csv", index=False)

    full = "M4_urban_environment"
    rows = []
    for v in sorted([v for v in variants if v.kind == "ablation"], key=lambda v: v.order):
        drop = pivot[full] - pivot[v.name]
        mean, lo, hi = _ci(drop)
        rows.append({"paper_outcome": target.replace("_loss_log", ""), "target": target,
                     "ablation_order": v.order, "ablation": v.name.removeprefix("without_"),
                     "ablation_label": v.label, "removed_groups": "|".join(v.groups),
                     "n_removed_features": 31 - len(v.features), "r2_full": observed[full],
                     "r2_without_group": observed[v.name], "drop_r2": observed[full] - observed[v.name],
                     "drop_r2_bootstrap_mean": mean, "ci_lower": lo, "ci_upper": hi,
                     "n_bootstrap": N_BOOTSTRAP})
    pd.DataFrame(rows).to_csv(out / f"{target}_leave_one_group_out_summary.csv", index=False)


def summarize_full38(target: str, out: Path) -> None:
    p31 = pd.read_csv(TUNED_31 / target / "catboost" / "fold_predictions.csv")
    p38 = pd.read_csv(OUTPUT_ROOT / "full38" / target / "catboost" / "fold_predictions.csv")
    base_cols = ["row_index", "uid", "y_true"]
    if not p31[base_cols].equals(p38[base_cols]):
        raise ValueError(f"{target}: full/reduced predictions do not align")
    wide = p31[base_cols].copy()
    wide["pred__reduced31"] = p31["y_pred"].to_numpy(float)
    wide["pred__full38"] = p38["y_pred"].to_numpy(float)
    wide.to_csv(out / f"{target}_full38_vs_reduced31_oof_predictions.csv", index=False)
    boot = paired_group_bootstrap(wide, ["full38", "reduced31"], "uid", N_BOOTSTRAP, SEED, "fast_r2")
    boot.to_csv(out / f"{target}_full38_vs_reduced31_city_block_bootstrap.csv", index=False)
    pivot = boot.pivot(index="bootstrap", columns="variant", values="r2")
    delta = pivot["reduced31"] - pivot["full38"]
    mean, lo, hi = _ci(delta)
    m31 = metric_dict(wide.y_true.to_numpy(float), wide.pred__reduced31.to_numpy(float))
    m38 = metric_dict(wide.y_true.to_numpy(float), wide.pred__full38.to_numpy(float))
    rows = []
    for name, n, metrics in [("full38", 38, m38), ("reduced31", 31, m31)]:
        vals = pivot[name]
        bmean, blo, bhi = _ci(vals)
        rows.append({"paper_outcome": target.replace("_loss_log", ""), "target": target,
                     "feature_set": name, "n_features": n, **metrics,
                     "r2_bootstrap_mean": bmean, "r2_ci_lower": blo, "r2_ci_upper": bhi,
                     "n_bootstrap": N_BOOTSTRAP})
    pd.DataFrame(rows).to_csv(out / f"{target}_full38_vs_reduced31_performance.csv", index=False)
    pd.DataFrame([{"paper_outcome": target.replace("_loss_log", ""), "target": target,
                   "contrast": "reduced31_minus_full38", "delta_r2": m31["r2"] - m38["r2"],
                   "delta_r2_bootstrap_mean": mean, "ci_lower": lo, "ci_upper": hi,
                   "n_bootstrap": N_BOOTSTRAP}]).to_csv(
        out / f"{target}_full38_vs_reduced31_delta_r2.csv", index=False)


def _group_importance(importance: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    merged = membership[["feature", "feature_group"]].merge(importance, on="feature", how="left")
    out = (merged.groupby("feature_group", as_index=False)
           .agg(mean_abs_shap_sum=("mean_abs_shap", "sum"),
                mean_abs_shap_mean=("mean_abs_shap", "mean"), n_features=("feature", "size"))
           .sort_values("mean_abs_shap_sum", ascending=False).reset_index(drop=True))
    out["relative_importance"] = out.mean_abs_shap_sum / out.mean_abs_shap_sum.sum()
    return out


def compute_oof_shap(target: str, out: Path) -> None:
    X, y, groups, meta, _ = load_prepared_snapshot(SOURCE_31, target)
    membership, _ = resolve_variants(X)
    model_root = TUNED_31 / target / "catboost"
    pred_all = pd.read_csv(model_root / "fold_predictions.csv").sort_values("row_index").reset_index(drop=True)
    shap_values = np.full(X.shape, np.nan, dtype=float)
    expected = np.full(len(X), np.nan, dtype=float)
    additivity_rows = []
    for fold in range(1, OUTER_FOLDS + 1):
        pred = pd.read_csv(model_root / f"outer_fold_{fold}" / "outer_predictions.csv")
        idx = pred["row_index"].to_numpy(int)
        model = joblib.load(model_root / f"outer_fold_{fold}" / "outer_model.joblib")
        raw = np.asarray(model.get_feature_importance(Pool(X.iloc[idx]), type="ShapValues"), dtype=float)
        shap_values[idx] = raw[:, :-1]
        expected[idx] = raw[:, -1]
        reconstructed = raw[:, :-1].sum(axis=1) + raw[:, -1]
        model_pred = np.asarray(model.predict(X.iloc[idx]), dtype=float)
        err = np.abs(reconstructed - model_pred)
        additivity_rows.append({"target": target, "fold": fold, "n_test": len(idx),
                                "mean_abs_additivity_error": err.mean(),
                                "max_abs_additivity_error": err.max()})
    if not np.isfinite(shap_values).all() or not np.isfinite(expected).all():
        raise ValueError(f"{target}: incomplete/non-finite OOF SHAP values")

    target_out = out / target
    target_out.mkdir(parents=True, exist_ok=True)
    identifiers = pred_all[[c for c in ["row_index", "uid", "year", "event_id", "mode", "fold",
                                        "y_true", "y_pred"] if c in pred_all]].copy()
    identifiers["expected_value"] = expected
    identifiers.to_csv(target_out / "oof_shap_row_metadata.csv", index=False)
    pd.DataFrame(shap_values, columns=X.columns).to_csv(target_out / "oof_shap_values_matrix.csv", index=False)
    X.to_csv(target_out / "oof_feature_values_matrix.csv", index=False)
    pd.DataFrame(additivity_rows).to_csv(target_out / "oof_shap_additivity_check.csv", index=False)

    importance = pd.DataFrame({"feature": X.columns, "mean_abs_shap": np.abs(shap_values).mean(axis=0),
                               "mean_signed_shap": shap_values.mean(axis=0)})
    importance = importance.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    importance["relative_importance"] = importance.mean_abs_shap / importance.mean_abs_shap.sum()
    importance["rank"] = np.arange(1, len(importance) + 1)
    importance.to_csv(target_out / "oof_shap_importance.csv", index=False)
    membership.merge(importance, on="feature", how="left").to_csv(
        target_out / "oof_shap_feature_group_membership.csv", index=False)
    _group_importance(importance, membership).to_csv(target_out / "oof_shap_group_importance.csv", index=False)

    rng = np.random.default_rng(SEED)
    sample_idx = np.sort(rng.choice(len(X), size=min(1500, len(X)), replace=False))
    sample = identifiers.iloc[sample_idx].reset_index(drop=True)
    for j, feature in enumerate(X.columns):
        sample[f"value__{feature}"] = X.iloc[sample_idx, j].to_numpy(float)
        q05, q95 = X[feature].quantile([0.05, 0.95])
        if q95 > q05:
            color = ((X.iloc[sample_idx, j] - q05) / (q95 - q05)).clip(0, 1)
        else:
            color = pd.Series(0.5, index=np.arange(len(sample_idx)))
        sample[f"color_05_95__{feature}"] = np.asarray(color, dtype=float)
        sample[f"shap__{feature}"] = shap_values[sample_idx, j]
    sample.to_csv(target_out / "oof_shap_beeswarm_sample1500_wide.csv", index=False)

    binned_rows = []
    for j, feature in enumerate(X.columns):
        values = X[feature].to_numpy(float)
        lo, hi = np.quantile(values, [0.005, 0.995])
        keep = (values >= lo) & (values <= hi)
        part = pd.DataFrame({"value": values[keep], "shap": shap_values[keep, j]})
        part["bin"] = pd.qcut(part["value"], q=18, labels=False, duplicates="drop")
        for bin_id, d in part.groupby("bin", sort=True):
            binned_rows.append({"target": target, "feature": feature, "bin": int(bin_id) + 1,
                                "n": len(d), "value_min": d.value.min(), "value_q25": d.value.quantile(.25),
                                "value_median": d.value.median(), "value_q75": d.value.quantile(.75),
                                "value_max": d.value.max(), "shap_q25": d.shap.quantile(.25),
                                "shap_median": d.shap.median(), "shap_q75": d.shap.quantile(.75),
                                "tail_exclusion": "values below 0.5th or above 99.5th percentile",
                                "requested_quantile_bins": 18})
    pd.DataFrame(binned_rows).to_csv(target_out / "oof_shap_binned_nonlinear_responses.csv", index=False)
    write_json({"target": target, "method": "native CatBoost SHAP on each outer-fold held-out set",
                "n_rows": len(X), "n_cities": groups.nunique(), "n_features": X.shape[1],
                "beeswarm_sample_size": len(sample_idx), "beeswarm_sampling_seed": SEED,
                "beeswarm_color_scaling": "per-feature 5th-95th percentile clipped to [0,1]",
                "dependence_tail_exclusion": "0.5% in each tail", "dependence_quantile_bins": 18,
                "fold_predictions_sha256": hashlib.sha256((model_root / "fold_predictions.csv").read_bytes()).hexdigest()},
               target_out / "oof_shap_method.json")


def concat_target_files(folder: Path, suffix: str, output_name: str) -> None:
    frames = [pd.read_csv(folder / f"{target}_{suffix}") for target in TARGETS]
    pd.concat(frames, ignore_index=True).to_csv(folder / output_name, index=False)


def summarize_and_package() -> None:
    fig3f = PACKAGE_ROOT / "Figure_3f_ready"
    s11 = PACKAGE_ROOT / "Supplementary_Fig_S11_ready"
    shap_out = PACKAGE_ROOT / "Figure_3a_e_and_S12_S13_ready"
    for folder in [fig3f, s11, shap_out]:
        folder.mkdir(parents=True, exist_ok=True)

    for target in TARGETS:
        X, *_ = load_prepared_snapshot(SOURCE_31, target)
        _, variants = resolve_variants(X)
        summarize_subsets(target, variants, fig3f)
        summarize_full38(target, s11)
        compute_oof_shap(target, shap_out)

    concat_target_files(fig3f, "progressive_stage_r2.csv", "progressive_M1_M4_stage_r2.csv")
    concat_target_files(fig3f, "progressive_incremental_delta_r2.csv", "progressive_M1_M4_incremental_delta_r2.csv")
    concat_target_files(fig3f, "conditional_addition_summary.csv", "figure3f_conditional_additions.csv")
    concat_target_files(fig3f, "leave_one_group_out_summary.csv", "leave_one_group_out_ablation.csv")
    concat_target_files(s11, "full38_vs_reduced31_performance.csv", "full38_vs_reduced31_performance.csv")
    concat_target_files(s11, "full38_vs_reduced31_delta_r2.csv", "full38_vs_reduced31_delta_r2.csv")
    shutil.copy2(OUTPUT_ROOT / "full38" / "data_audit.json", s11 / "full38_vs_reduced31_data_audit.json")
    shutil.copy2(OUTPUT_ROOT / "feature_subsets" / "data_audit.json", fig3f / "nested_subset_data_audit.json")

    run_manifest = {
        "completed_at": pd.Timestamp.now().isoformat(), "python": sys.version,
        "platform": platform.platform(), "seed": SEED, "outer_cv": "5-fold GroupKFold by uid (city)",
        "inner_tuning": "3-fold GroupKFold by uid within every outer-training set",
        "selection_metric": "pooled inner out-of-fold R2", "catboost_candidates": 32,
        "candidate_design": "the same pre-specified Latin-hypercube set used in the formal model comparison",
        "bootstrap": f"paired city-block bootstrap, {N_BOOTSTRAP} resamples, seed {SEED}",
        "outcomes": "negative-log resistance loss and negative-log recovery loss",
    }
    write_json(run_manifest, PACKAGE_ROOT / "RECOMPUTATION_MANIFEST.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=["subsets", "full38", "package"],
                        default=["subsets", "full38", "package"])
    parser.add_argument("--n-jobs", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_json({"started_at": pd.Timestamp.now().isoformat(), "tasks": args.tasks,
                "n_jobs": args.n_jobs, "seed": SEED}, OUTPUT_ROOT / "run_config.json")
    if "subsets" in args.tasks:
        run_subset_nested(args.n_jobs, args.overwrite)
    if "full38" in args.tasks:
        run_full38_nested(args.n_jobs, args.overwrite)
    if "package" in args.tasks:
        summarize_and_package()


if __name__ == "__main__":
    main()

