# -*- coding: utf-8 -*-
"""
Multi-city thermal resistance and thermal recovery capacity calculation from:
1) daily city-level MODIS LST CSVs, e.g.
   City_00_GF_MODIS_LST_AllDays_DailyStats.csv
2) ERA5-derived heatwave event CSVs, e.g.
   City_00_HeatwaveEvents_2000_2025_Baseline1991_2020_Win15d.csv

Main updates compared with the single-city version:
- Merge adjacent heatwave events into a compound heatwave event when the
  non-heatwave gap between two events is <= --merge_gap_days.
- Loop over multiple cities using fixed filename patterns.
- By default, write one event-metric CSV for each city.
- Optionally truncate the recovery search window before the next merged heatwave
  event, preventing a subsequent heatwave from contaminating recovery metrics.

Recommended indicators:
- Convert LST to positive anomaly against a multi-year same-season baseline.
- Thermal resistance: inverse of mean positive LST anomaly during heatwave event.
- Thermal recovery capacity: inverse of mean positive LST anomaly during post-event recovery period.
- Recovery time: number of days from event end to the first stable recovered day.

Author: ChatGPT
"""

from __future__ import annotations

import argparse
import calendar
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# =========================
# Default configuration
# =========================
DEFAULT_LST_DIR = "data/modis_lst_by_city"
DEFAULT_HW_DIR = "data/heatwave_events"
DEFAULT_OUT_DIR = "outputs/thermal_resilience"

# Use Python format syntax. {city:02d} will become 00, 01, 02, ...
DEFAULT_LST_PATTERN = "City_{city:02d}_GF_MODIS_LST_AllDays_DailyStats.csv"
DEFAULT_HW_PATTERN = "City_{city:02d}_HeatwaveEvents_2000_2025_Baseline1991_2020_Win15d.csv"

# LST column used for calculation. Change to LST_C_median if needed.
DEFAULT_LST_VALUE_COL = "LST_C_mean"

# Use all available MODIS modes by default: normally Day and Night.
DEFAULT_MODES = "all"  # "all", "Day", "Night", or "Day,Night"

# Baseline setting
DEFAULT_BASELINE_METHOD = "median"  # "median" or "mean"
DEFAULT_BASELINE_WINDOW_DAYS = 5    # centered moving window; 5 means +/-2 days
DEFAULT_EXCLUDE_TARGET_YEAR = True  # avoid using the event year itself in the baseline

# Heatwave-event merging setting
# If the number of non-heatwave days between two events is <= this value, merge them.
# Example: event1 ends on July 10, event2 starts on July 14 -> gap = 3 days: July 11-13.
DEFAULT_MERGE_GAP_DAYS = 5

# Data quality setting. For gap-filled data, event coverage can usually be 1.0.
DEFAULT_MIN_EVENT_COVERAGE = 1.0
DEFAULT_MIN_RECOVERY_COVERAGE = 0.5

# Recovery-period setting
DEFAULT_RECOVERY_MAX_DAYS = 30
DEFAULT_RECOVERY_THRESHOLD_C = 0.5  # recovered when positive anomaly <= this threshold
DEFAULT_RECOVERY_CONSECUTIVE_DAYS = 3

# If True, use "recovery_days_required" column in the heatwave CSV as max search days
# when this column exists. Usually I recommend False, because 10 days may be too short.
DEFAULT_USE_EVENT_RECOVERY_DAYS_REQUIRED = False

# If True, the recovery search period is cut off at the day before the next merged heatwave.
DEFAULT_TRUNCATE_RECOVERY_AT_NEXT_EVENT = True

# Output field setting
# The default profile is used for normal analysis; "minimal" is for quick overview;
# "full" preserves every diagnostic and original heatwave field.
DEFAULT_OUTPUT_COLUMNS = "compact"

MINIMAL_OUTPUT_COLUMNS = [
    "city_label", "event_id", "mode",
    "start_date", "end_date", "duration_days",
    "availability_flag",
    "mean_event_anomaly_C", "thermal_resistance_C",
    "mean_recovery_anomaly_C", "thermal_recovery_capacity_C",
    "thermal_resilience_combined_C",
    "recovered", "recovery_time_days",
]

COMPACT_OUTPUT_COLUMNS = [
    # ID and event window
    "city_label", "city_id_from_loop", "event_id", "event_seq", "mode",
    "start_date", "end_date", "duration_days",

    # Compound heatwave-event information
    "is_composite_event", "source_event_count", "source_event_ids", "max_internal_gap_days",

    # ERA5 heatwave intensity
    "mean_Tmax_C", "max_Tmax_C", "mean_threshold_C", "mean_excess_C", "max_excess_C",

    # MODIS availability and usability
    "event_lst_coverage", "recovery_lst_coverage", "availability_flag",

    # LST anomaly and main thermal-resilience metrics in Celsius units
    "mean_event_anomaly_C", "max_event_anomaly_C",
    "mean_recovery_anomaly_C", "max_recovery_anomaly_C",
    "thermal_resistance_C", "thermal_recovery_capacity_C", "thermal_resilience_combined_C",

    # Recovery-process information
    "recovered", "recovery_time_days", "recovery_end_date",
    "recovery_truncated_by_next_event", "next_event_start_date",

    # Key method parameters needed for traceability
    "merged_gap_threshold_days", "recovery_threshold_C", "recovery_consecutive_days",
]


# =========================
# Utility functions
# =========================
def noleap_doy(ts: pd.Timestamp) -> int:
    """
    Convert a date to climatological day-of-year in a no-leap 365-day calendar.
    Feb 29 is mapped to Feb 28's climatological day.
    Dates after Feb 29 in leap years are shifted by -1.
    """
    ts = pd.Timestamp(ts)
    doy = int(ts.dayofyear)
    if calendar.isleap(ts.year):
        if ts.month == 2 and ts.day == 29:
            return 59
        if ts.month > 2:
            return doy - 1
    return doy


def cyclic_window_days(center: int, window_days: int, n_days: int = 365) -> List[int]:
    """
    Return climatological days in a centered cyclic window.
    Example: center=1, window=5 -> [364, 365, 1, 2, 3]
    """
    if window_days < 1:
        raise ValueError("window_days must be >= 1")
    half_left = (window_days - 1) // 2
    half_right = window_days - 1 - half_left
    days = []
    for d in range(center - half_left, center + half_right + 1):
        dd = ((d - 1) % n_days) + 1
        days.append(dd)
    return days


def parse_modes(modes_str: str, available_modes: Iterable[str]) -> List[str]:
    available_modes = list(available_modes)
    if modes_str is None or str(modes_str).lower() == "all":
        return available_modes
    requested = [m.strip() for m in str(modes_str).split(",") if m.strip()]
    missing = [m for m in requested if m not in available_modes]
    if missing:
        raise ValueError(
            f"Requested mode(s) not found in LST data: {missing}. "
            f"Available modes: {available_modes}"
        )
    return requested


def safe_mean(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.mean()) if len(values) > 0 else np.nan


def safe_max(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.max()) if len(values) > 0 else np.nan


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce")
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return np.nan
    return float(np.average(values[mask], weights=weights[mask]))


def parse_city_ids(city_ids: str, lst_dir: Path, lst_pattern: str) -> List[int]:
    """
    Parse city id string.
    Supported examples:
        all
        0-9
        0,1,2,7
        00,01,02
        0-3,7,9
    """
    s = str(city_ids).strip()
    if s.lower() == "all":
        # Convert the LST pattern to a regex by replacing {city:02d} with (\d+).
        # This follows the project filename conventions.
        escaped = re.escape(lst_pattern)
        escaped = escaped.replace(re.escape("{city:02d}"), r"(?P<city>\d+)")
        regex = re.compile("^" + escaped + "$", re.IGNORECASE)
        found = []
        for p in lst_dir.iterdir():
            m = regex.match(p.name)
            if m:
                found.append(int(m.group("city")))
        return sorted(set(found))

    ids: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a_int, b_int = int(a), int(b)
            if b_int < a_int:
                raise ValueError(f"Invalid city id range: {part}")
            ids.extend(list(range(a_int, b_int + 1)))
        else:
            ids.append(int(part))
    return sorted(set(ids))


def format_city_id(city_id: int) -> str:
    return f"City_{city_id:02d}"


# =========================
# Data loading
# =========================
def load_lst(lst_csv: str | Path, lst_value_col: str) -> pd.DataFrame:
    lst = pd.read_csv(lst_csv)

    required = {"date", "mode", lst_value_col}
    missing = required - set(lst.columns)
    if missing:
        raise ValueError(
            f"LST CSV is missing required column(s): {missing}. "
            f"Available columns: {list(lst.columns)}"
        )

    lst = lst.copy()
    lst["date"] = pd.to_datetime(lst["date"], errors="coerce")
    lst = lst.dropna(subset=["date"])
    lst[lst_value_col] = pd.to_numeric(lst[lst_value_col], errors="coerce")

    # Make sure there is only one record per mode/date.
    # If duplicates exist, aggregate numeric columns by mean and keep the first non-numeric value.
    if lst.duplicated(subset=["mode", "date"]).any():
        agg = {}
        for c in lst.columns:
            if c in ["mode", "date"]:
                continue
            if pd.api.types.is_numeric_dtype(lst[c]):
                agg[c] = "mean"
            else:
                agg[c] = "first"
        lst = lst.groupby(["mode", "date"], as_index=False).agg(agg)

    lst["year"] = lst["date"].dt.year.astype(int)
    lst["month"] = lst["date"].dt.month.astype(int)
    lst["day"] = lst["date"].dt.day.astype(int)
    lst["clim_doy"] = lst["date"].apply(noleap_doy).astype(int)
    return lst.reset_index(drop=True)


def load_events(hw_csv: str | Path) -> pd.DataFrame:
    events = pd.read_csv(hw_csv)

    required = {"event_id", "start_date", "end_date"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(
            f"Heatwave CSV is missing required column(s): {missing}. "
            f"Available columns: {list(events.columns)}"
        )

    events = events.copy()
    events["start_date"] = pd.to_datetime(events["start_date"], errors="coerce")
    events["end_date"] = pd.to_datetime(events["end_date"], errors="coerce")
    events = events.dropna(subset=["start_date", "end_date"])
    events = events[events["end_date"] >= events["start_date"]].copy()

    if "duration_days" not in events.columns:
        events["duration_days"] = (events["end_date"] - events["start_date"]).dt.days + 1
    else:
        events["duration_days"] = pd.to_numeric(events["duration_days"], errors="coerce")
        missing_duration = events["duration_days"].isna()
        events.loc[missing_duration, "duration_days"] = (
            events.loc[missing_duration, "end_date"] - events.loc[missing_duration, "start_date"]
        ).dt.days + 1

    return events.sort_values(["start_date", "end_date"]).reset_index(drop=True)


# =========================
# Heatwave-event merging
# =========================
def _aggregate_event_group(
    group: pd.DataFrame,
    city_id: int,
    merged_seq: int,
    merge_gap_days: int,
) -> dict:
    """Aggregate one group of original events into one merged/compound event."""
    group = group.sort_values(["start_date", "end_date"]).copy()
    start = pd.Timestamp(group["start_date"].min())
    end = pd.Timestamp(group["end_date"].max())
    duration = int((end - start).days + 1)

    original_duration = pd.to_numeric(group.get("duration_days", duration), errors="coerce")
    original_duration = original_duration.fillna((group["end_date"] - group["start_date"]).dt.days + 1)

    row: dict = {}

    # Keep city fields if present, otherwise use the loop city_id.
    row["city_id"] = group["city_id"].iloc[0] if "city_id" in group.columns else city_id
    if "city_index" in group.columns:
        row["city_index"] = group["city_index"].iloc[0]
    else:
        row["city_index"] = city_id

    row["event_id"] = f"C{city_id:02d}_HW_M{merged_seq:04d}"
    row["event_seq"] = merged_seq
    row["start_date"] = start
    row["end_date"] = end
    row["duration_days"] = duration

    row["is_composite_event"] = bool(len(group) > 1)
    row["source_event_count"] = int(len(group))
    row["source_event_ids"] = ";".join(group["event_id"].astype(str).tolist())
    if "event_seq" in group.columns:
        row["source_event_seqs"] = ";".join(group["event_seq"].astype(str).tolist())
    else:
        row["source_event_seqs"] = ""
    row["source_start_dates"] = ";".join(group["start_date"].dt.strftime("%Y-%m-%d").tolist())
    row["source_end_dates"] = ";".join(group["end_date"].dt.strftime("%Y-%m-%d").tolist())
    row["source_duration_days_sum"] = int(original_duration.sum())
    row["merged_gap_threshold_days"] = int(merge_gap_days)

    if len(group) > 1:
        gaps = []
        group_sorted = group.reset_index(drop=True)
        for i in range(len(group_sorted) - 1):
            gap = int((group_sorted.loc[i + 1, "start_date"] - group_sorted.loc[i, "end_date"]).days - 1)
            gaps.append(gap)
        row["internal_gap_days"] = ";".join(map(str, gaps))
        row["max_internal_gap_days"] = int(max(gaps)) if gaps else 0
    else:
        row["internal_gap_days"] = ""
        row["max_internal_gap_days"] = 0

    # Aggregate other columns.
    skip_cols = {
        "city_id", "city_index", "event_id", "event_seq", "start_date", "end_date", "duration_days"
    }
    for c in group.columns:
        if c in skip_cols:
            continue
        if c in row:
            continue

        s = group[c]
        if pd.api.types.is_numeric_dtype(s):
            c_lower = c.lower()
            if c_lower.startswith("max_") or c_lower.endswith("_max"):
                row[c] = float(pd.to_numeric(s, errors="coerce").max())
            elif c_lower.startswith("min_") or c_lower.endswith("_min"):
                row[c] = float(pd.to_numeric(s, errors="coerce").min())
            elif c_lower.startswith("mean_") or c_lower.endswith("_mean"):
                row[c] = weighted_mean(s, original_duration)
            elif c_lower.startswith("sum_") or c_lower.endswith("_sum"):
                row[c] = float(pd.to_numeric(s, errors="coerce").sum())
            else:
                # For generic numeric columns, use duration-weighted mean.
                row[c] = weighted_mean(s, original_duration)
        else:
            vals = [str(v) for v in s.dropna().unique().tolist()]
            row[c] = vals[0] if len(vals) <= 1 else ";".join(vals)

    return row


def merge_heatwave_events(events: pd.DataFrame, city_id: int, merge_gap_days: int = 5) -> pd.DataFrame:
    """
    Merge adjacent heatwave events into compound events.

    Gap definition:
        gap_days = next_event_start_date - current_event_end_date - 1
    That is the number of non-heatwave days between two events.

    If gap_days <= merge_gap_days, the two events are merged.
    Example: event1 ends on July 10 and event2 starts on July 14.
             Non-heatwave days are July 11, 12, 13, so gap_days = 3.
             If --merge_gap_days >= 3, they are merged.
    """
    if events.empty:
        return events.copy()
    if merge_gap_days < 0:
        # Negative threshold means no merging, but still standardize IDs.
        merge_gap_days = -1

    events = events.sort_values(["start_date", "end_date"]).reset_index(drop=True).copy()
    groups: List[pd.DataFrame] = []
    current_indices: List[int] = [0]
    current_end = pd.Timestamp(events.loc[0, "end_date"])

    for i in range(1, len(events)):
        next_start = pd.Timestamp(events.loc[i, "start_date"])
        next_end = pd.Timestamp(events.loc[i, "end_date"])
        gap_days = int((next_start - current_end).days - 1)

        if gap_days <= merge_gap_days:
            current_indices.append(i)
            if next_end > current_end:
                current_end = next_end
        else:
            groups.append(events.loc[current_indices].copy())
            current_indices = [i]
            current_end = next_end

    groups.append(events.loc[current_indices].copy())

    merged_rows = [
        _aggregate_event_group(g, city_id=city_id, merged_seq=i + 1, merge_gap_days=merge_gap_days)
        for i, g in enumerate(groups)
    ]
    merged = pd.DataFrame(merged_rows)
    merged = merged.sort_values(["start_date", "end_date"]).reset_index(drop=True)
    return merged


# =========================
# Baseline calculation
# =========================
def build_baseline_lookup(
    lst: pd.DataFrame,
    lst_value_col: str,
    modes: List[str],
    method: str = "median",
    window_days: int = 5,
    exclude_target_year: bool = True,
) -> Dict[Tuple[str, Optional[int], int], Tuple[float, float, int]]:
    """
    Build lookup:
        (mode, target_year or None, clim_doy) -> (baseline, sigma, n)

    baseline is calculated from records in the centered climatological window.
    sigma is the std in that window, used for standardized anomalies.
    """
    if method not in {"mean", "median"}:
        raise ValueError("method must be 'mean' or 'median'")

    lookup: Dict[Tuple[str, Optional[int], int], Tuple[float, float, int]] = {}

    years = sorted(lst["year"].dropna().astype(int).unique().tolist())
    target_years: List[Optional[int]] = years if exclude_target_year else [None]

    for mode in modes:
        df_mode = lst[lst["mode"] == mode].copy()
        if df_mode.empty:
            continue

        for target_year in target_years:
            if exclude_target_year:
                df_base = df_mode[df_mode["year"] != target_year].copy()
            else:
                df_base = df_mode.copy()

            for d in range(1, 366):
                wdays = cyclic_window_days(d, window_days, 365)
                vals = pd.to_numeric(
                    df_base.loc[df_base["clim_doy"].isin(wdays), lst_value_col],
                    errors="coerce"
                ).dropna()

                if len(vals) == 0:
                    base = np.nan
                    sigma = np.nan
                    n = 0
                else:
                    base = float(vals.mean() if method == "mean" else vals.median())
                    sigma = float(vals.std(ddof=0))
                    n = int(len(vals))

                lookup[(mode, target_year, d)] = (base, sigma, n)

    return lookup


def attach_baseline(
    daily: pd.DataFrame,
    mode: str,
    baseline_lookup: Dict[Tuple[str, Optional[int], int], Tuple[float, float, int]],
    exclude_target_year: bool,
) -> pd.DataFrame:
    daily = daily.copy()
    bases, sigmas, ns = [], [], []
    for _, row in daily.iterrows():
        target_year = int(pd.Timestamp(row["date"]).year) if exclude_target_year else None
        clim_doy = int(noleap_doy(row["date"]))
        base, sigma, n = baseline_lookup.get((mode, target_year, clim_doy), (np.nan, np.nan, 0))
        bases.append(base)
        sigmas.append(sigma)
        ns.append(n)
    daily["baseline_LST_C"] = bases
    daily["baseline_sigma_C"] = sigmas
    daily["baseline_n"] = ns
    return daily


# =========================
# Event and recovery calculation
# =========================
def get_daily_lst_for_dates(
    lst: pd.DataFrame,
    mode: str,
    dates: pd.DatetimeIndex,
    lst_value_col: str,
) -> pd.DataFrame:
    base = pd.DataFrame({"date": pd.to_datetime(dates)})
    sub = lst[lst["mode"] == mode].copy()
    keep_cols = ["date", lst_value_col]
    for c in [
        "city_id", "city_index", "dataset",
        "spatial_coverage_fraction", "spatial_coverage_30", "spatial_coverage_50", "spatial_coverage_70"
    ]:
        if c in sub.columns:
            keep_cols.append(c)
    sub = sub[keep_cols].copy()
    out = base.merge(sub, on="date", how="left")
    out["has_lst"] = out[lst_value_col].notna()
    return out


def compute_anomalies(
    daily: pd.DataFrame,
    lst_value_col: str,
    positive_only: bool = True,
) -> pd.DataFrame:
    daily = daily.copy()
    if daily.empty:
        daily["lst_minus_baseline_C"] = pd.Series(dtype="float64")
        daily["anomaly_C"] = pd.Series(dtype="float64")
        daily["anomaly_z"] = pd.Series(dtype="float64")
        return daily

    daily["lst_minus_baseline_C"] = daily[lst_value_col] - daily["baseline_LST_C"]

    if positive_only:
        daily["anomaly_C"] = daily["lst_minus_baseline_C"].clip(lower=0)
    else:
        daily["anomaly_C"] = daily["lst_minus_baseline_C"]

    # Standardized positive anomaly.
    with np.errstate(divide="ignore", invalid="ignore"):
        z = daily["lst_minus_baseline_C"] / daily["baseline_sigma_C"]
    z = z.replace([np.inf, -np.inf], np.nan)
    daily["anomaly_z"] = z.clip(lower=0) if positive_only else z
    return daily


def find_recovery_end(
    rec_daily: pd.DataFrame,
    anomaly_col: str = "anomaly_C",
    threshold: float = 0.0,
    consecutive_days: int = 3,
) -> Tuple[bool, Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """
    Find first recovered day verified by a consecutive-day condition.
    Returns:
        recovered,
        recovery_end_date: first day of the stable recovered run,
        recovery_confirm_end_date: last day of the stable recovered run
    """
    if rec_daily.empty or consecutive_days < 1:
        return False, None, None

    cond = rec_daily["has_lst"] & rec_daily[anomaly_col].notna() & (rec_daily[anomaly_col] <= threshold)
    dates = rec_daily["date"].reset_index(drop=True)
    cond = cond.reset_index(drop=True)

    run = 0
    for i, ok in enumerate(cond):
        if bool(ok):
            run += 1
        else:
            run = 0

        if run >= consecutive_days:
            start_idx = i - consecutive_days + 1
            return True, pd.Timestamp(dates.iloc[start_idx]), pd.Timestamp(dates.iloc[i])

    return False, None, None


def summarize_event_period(daily: pd.DataFrame, anomaly_col: str) -> Tuple[float, float, int, int, float]:
    """
    Return mean anomaly, max anomaly, number of available days,
    total days, and coverage ratio.
    """
    total_days = int(len(daily))
    if daily.empty:
        return np.nan, np.nan, 0, 0, np.nan
    available = daily[daily["has_lst"] & daily[anomaly_col].notna()].copy()
    available_days = int(len(available))
    coverage = available_days / total_days if total_days > 0 else np.nan
    mean_anom = safe_mean(available[anomaly_col]) if available_days > 0 else np.nan
    max_anom = safe_max(available[anomaly_col]) if available_days > 0 else np.nan
    return mean_anom, max_anom, available_days, total_days, coverage


def inverse_index(disturbance: float) -> float:
    """
    Convert mean positive anomaly to an index where larger means better.
    Disturbance should be >= 0. If disturbance is 0, index is 1.
    """
    if pd.isna(disturbance):
        return np.nan
    if disturbance < 0:
        return np.nan
    return float(1.0 / (1.0 + disturbance))


def calculate_metrics(
    lst: pd.DataFrame,
    events: pd.DataFrame,
    lst_value_col: str,
    modes: List[str],
    baseline_lookup: Dict[Tuple[str, Optional[int], int], Tuple[float, float, int]],
    exclude_target_year: bool,
    baseline_method: str,
    baseline_window_days: int,
    min_event_coverage: float = 1.0,
    min_recovery_coverage: float = 0.5,
    recovery_max_days: int = 30,
    recovery_threshold_c: float = 0.0,
    recovery_consecutive_days: int = 3,
    use_event_recovery_days_required: bool = False,
    truncate_recovery_at_next_event: bool = True,
    return_daily_status: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    daily_rows = []

    if events.empty:
        return pd.DataFrame(), pd.DataFrame()

    events = events.sort_values(["start_date", "end_date"]).reset_index(drop=True).copy()

    lst_min_date = pd.Timestamp(lst["date"].min())
    lst_max_date = pd.Timestamp(lst["date"].max())

    for ev_idx, ev in events.iterrows():
        event_start = pd.Timestamp(ev["start_date"])
        event_end = pd.Timestamp(ev["end_date"])
        event_dates = pd.date_range(event_start, event_end, freq="D")

        if use_event_recovery_days_required and "recovery_days_required" in events.columns:
            try:
                max_days = int(ev["recovery_days_required"])
            except Exception:
                max_days = int(recovery_max_days)
        else:
            max_days = int(recovery_max_days)

        recovery_start = event_end + pd.Timedelta(days=1)
        nominal_recovery_search_end = event_end + pd.Timedelta(days=max_days)
        recovery_search_end = nominal_recovery_search_end
        next_event_start = pd.NaT
        recovery_truncated_by_next_event = False

        if truncate_recovery_at_next_event and ev_idx < len(events) - 1:
            next_event_start = pd.Timestamp(events.loc[ev_idx + 1, "start_date"])
            cutoff = next_event_start - pd.Timedelta(days=1)
            if cutoff < recovery_search_end:
                recovery_search_end = cutoff
                recovery_truncated_by_next_event = True

        if recovery_start <= recovery_search_end:
            recovery_dates = pd.date_range(recovery_start, recovery_search_end, freq="D")
        else:
            recovery_dates = pd.DatetimeIndex([])

        for mode in modes:
            # Event period
            event_daily = get_daily_lst_for_dates(lst, mode, event_dates, lst_value_col)
            event_daily = attach_baseline(event_daily, mode, baseline_lookup, exclude_target_year)
            event_daily = compute_anomalies(event_daily, lst_value_col, positive_only=True)
            event_daily["period"] = "event"

            # Recovery search period
            rec_daily_full = get_daily_lst_for_dates(lst, mode, recovery_dates, lst_value_col)
            rec_daily_full = attach_baseline(rec_daily_full, mode, baseline_lookup, exclude_target_year)
            rec_daily_full = compute_anomalies(rec_daily_full, lst_value_col, positive_only=True)
            rec_daily_full["period"] = "recovery_search"

            recovered, rec_end, rec_confirm_end = find_recovery_end(
                rec_daily_full,
                anomaly_col="anomaly_C",
                threshold=recovery_threshold_c,
                consecutive_days=recovery_consecutive_days,
            )

            if recovered and rec_end is not None:
                rec_daily_used = rec_daily_full[rec_daily_full["date"] <= rec_end].copy()
                recovery_end_date = rec_end
                recovery_confirm_end_date = rec_confirm_end
                rt_days = int((recovery_end_date - event_end).days)
            else:
                rec_daily_used = rec_daily_full.copy()
                recovery_end_date = pd.NaT
                recovery_confirm_end_date = pd.NaT
                rt_days = np.nan

            rec_daily_used["period"] = "recovery"

            # Summaries: raw Celsius anomaly
            event_mean_c, event_max_c, event_avail, event_total, event_cov = summarize_event_period(
                event_daily, "anomaly_C"
            )
            rec_mean_c, rec_max_c, rec_avail, rec_total, rec_cov = summarize_event_period(
                rec_daily_used, "anomaly_C"
            )

            # Summaries: standardized anomaly
            event_mean_z, event_max_z, _, _, _ = summarize_event_period(event_daily, "anomaly_z")
            rec_mean_z, rec_max_z, _, _, _ = summarize_event_period(rec_daily_used, "anomaly_z")

            event_enough = event_cov >= min_event_coverage if pd.notna(event_cov) else False
            rec_enough = rec_cov >= min_recovery_coverage if pd.notna(rec_cov) else False

            resistance_c = inverse_index(event_mean_c) if event_enough else np.nan
            recovery_c = inverse_index(rec_mean_c) if rec_enough else np.nan
            resistance_z = inverse_index(event_mean_z) if event_enough else np.nan
            recovery_z = inverse_index(rec_mean_z) if rec_enough else np.nan

            # Optional combined index. Keep it as a secondary metric.
            combined_c = np.nan
            combined_z = np.nan
            if pd.notna(resistance_c) and pd.notna(recovery_c):
                combined_c = float((resistance_c + recovery_c) / 2.0)
            if pd.notna(resistance_z) and pd.notna(recovery_z):
                combined_z = float((resistance_z + recovery_z) / 2.0)

            # Diagnose why an event may be unavailable
            if event_end < lst_min_date or event_start > lst_max_date:
                availability_flag = "outside_lst_period"
            elif not event_enough:
                availability_flag = "insufficient_event_lst"
            elif recovery_dates.empty:
                availability_flag = "no_recovery_window_before_next_event"
            elif not rec_enough:
                availability_flag = "insufficient_recovery_lst"
            elif not recovered:
                availability_flag = "not_recovered_within_search_window"
            else:
                availability_flag = "ok"

            metric = {
                # IDs
                "city_id": ev.get("city_id", np.nan),
                "city_index": ev.get("city_index", np.nan),
                "event_id": ev.get("event_id", np.nan),
                "event_seq": ev.get("event_seq", np.nan),
                "mode": mode,

                # Merged-event information
                "is_composite_event": ev.get("is_composite_event", False),
                "source_event_count": ev.get("source_event_count", 1),
                "source_event_ids": ev.get("source_event_ids", ev.get("event_id", np.nan)),
                "source_event_seqs": ev.get("source_event_seqs", np.nan),
                "source_start_dates": ev.get("source_start_dates", np.nan),
                "source_end_dates": ev.get("source_end_dates", np.nan),
                "source_duration_days_sum": ev.get("source_duration_days_sum", np.nan),
                "merged_gap_threshold_days": ev.get("merged_gap_threshold_days", np.nan),
                "internal_gap_days": ev.get("internal_gap_days", np.nan),
                "max_internal_gap_days": ev.get("max_internal_gap_days", np.nan),

                # Original or merged heatwave information
                "start_date": event_start.date().isoformat(),
                "end_date": event_end.date().isoformat(),
                "duration_days": ev.get("duration_days", len(event_dates)),
                "mean_Tmax_C": ev.get("mean_Tmax_C", np.nan),
                "max_Tmax_C": ev.get("max_Tmax_C", np.nan),
                "mean_threshold_C": ev.get("mean_threshold_C", np.nan),
                "mean_excess_C": ev.get("mean_excess_C", np.nan),
                "max_excess_C": ev.get("max_excess_C", np.nan),

                # MODIS availability
                "lst_value_col": lst_value_col,
                "event_lst_available_days": event_avail,
                "event_total_days": event_total,
                "event_lst_coverage": event_cov,
                "recovery_lst_available_days": rec_avail,
                "recovery_total_days_used": rec_total,
                "recovery_lst_coverage": rec_cov,
                "availability_flag": availability_flag,

                # Baseline and anomaly settings
                "baseline_method": baseline_method,
                "baseline_window_days": baseline_window_days,
                "exclude_target_year_from_baseline": exclude_target_year,

                # Disturbance intensity
                "mean_event_anomaly_C": event_mean_c,
                "max_event_anomaly_C": event_max_c,
                "mean_recovery_anomaly_C": rec_mean_c,
                "max_recovery_anomaly_C": rec_max_c,
                "mean_event_anomaly_z": event_mean_z,
                "max_event_anomaly_z": event_max_z,
                "mean_recovery_anomaly_z": rec_mean_z,
                "max_recovery_anomaly_z": rec_max_z,

                # Main indices
                "thermal_resistance_C": resistance_c,
                "thermal_recovery_capacity_C": recovery_c,
                "thermal_resilience_combined_C": combined_c,
                "thermal_resistance_z": resistance_z,
                "thermal_recovery_capacity_z": recovery_z,
                "thermal_resilience_combined_z": combined_z,

                # Recovery process
                "recovered": bool(recovered),
                "recovery_end_date": None if pd.isna(recovery_end_date) else pd.Timestamp(recovery_end_date).date().isoformat(),
                "recovery_confirm_end_date": None if pd.isna(recovery_confirm_end_date) else pd.Timestamp(recovery_confirm_end_date).date().isoformat(),
                "recovery_time_days": rt_days,
                "recovery_search_days_nominal": max_days,
                "recovery_search_start_date": None if recovery_dates.empty else pd.Timestamp(recovery_dates.min()).date().isoformat(),
                "recovery_search_end_date": None if recovery_dates.empty else pd.Timestamp(recovery_dates.max()).date().isoformat(),
                "nominal_recovery_search_end_date": nominal_recovery_search_end.date().isoformat(),
                "truncate_recovery_at_next_event": bool(truncate_recovery_at_next_event),
                "recovery_truncated_by_next_event": bool(recovery_truncated_by_next_event),
                "next_event_start_date": None if pd.isna(next_event_start) else pd.Timestamp(next_event_start).date().isoformat(),
                "recovery_threshold_C": recovery_threshold_c,
                "recovery_consecutive_days": recovery_consecutive_days,
            }

            # Add all merged/original event columns not already included.
            for c in ev.index:
                c2 = f"hw_{c}"
                if c2 not in metric:
                    v = ev[c]
                    if isinstance(v, pd.Timestamp):
                        v = v.date().isoformat()
                    metric[c2] = v

            metric_rows.append(metric)

            # Daily output for checking. This is optional because it can be large.
            if return_daily_status:
                daily_for_output = pd.concat([event_daily, rec_daily_used], ignore_index=True)
                daily_for_output.insert(0, "event_id", ev.get("event_id", np.nan))
                daily_for_output.insert(1, "mode", mode)
                daily_for_output.insert(2, "event_start_date", event_start.date().isoformat())
                daily_for_output.insert(3, "event_end_date", event_end.date().isoformat())
                daily_for_output["is_composite_event"] = ev.get("is_composite_event", False)
                daily_for_output["source_event_count"] = ev.get("source_event_count", 1)
                daily_for_output["source_event_ids"] = ev.get("source_event_ids", ev.get("event_id", np.nan))
                daily_for_output["recovered"] = bool(recovered)
                daily_for_output["recovery_end_date"] = (
                    None if pd.isna(recovery_end_date) else pd.Timestamp(recovery_end_date).date().isoformat()
                )
                daily_for_output["recovery_truncated_by_next_event"] = bool(recovery_truncated_by_next_event)
                daily_rows.append(daily_for_output)

    metrics_df = pd.DataFrame(metric_rows)
    daily_df = pd.concat(daily_rows, ignore_index=True) if (return_daily_status and daily_rows) else pd.DataFrame()
    return metrics_df, daily_df



def select_output_columns(metrics_df: pd.DataFrame, output_columns: str = "compact") -> pd.DataFrame:
    """
    Reduce event-metric output columns.

    Profiles:
    - minimal: only IDs, dates, main Celsius metrics, recovered flag, and recovery time.
    - default profile: keeps compound-event info, ERA5 intensity,
      availability, main metrics, and key method parameters.
    - full: keeps all diagnostic columns, including duplicated hw_* fields.
    """
    if metrics_df.empty:
        return metrics_df

    profile = str(output_columns).lower().strip()
    if profile == "full":
        return metrics_df
    if profile == "minimal":
        wanted = MINIMAL_OUTPUT_COLUMNS
    elif profile == "compact":
        wanted = COMPACT_OUTPUT_COLUMNS
    else:
        raise ValueError("output_columns must be one of: minimal, compact, full")

    keep = [c for c in wanted if c in metrics_df.columns]
    return metrics_df.loc[:, keep].copy()



# =========================
# One-city and multi-city workflows
# =========================
def process_one_city(
    city_id: int,
    lst_path: Path,
    hw_path: Path,
    out_dir: Path,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Process one city and save the default output.

    Default output is exactly one CSV per city:
        City_XX_thermal_resilience_event_metrics.csv

    The merged-event metadata are already included as columns in this metrics file
    (e.g., is_composite_event, source_event_count, source_event_ids), so an extra
    merged-event CSV is unnecessary unless --save_merged_events is specified.
    """
    city_label = format_city_id(city_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    lst = load_lst(lst_path, args.lst_value_col)
    events_raw = load_events(hw_path)
    events_merged = merge_heatwave_events(events_raw, city_id=city_id, merge_gap_days=args.merge_gap_days)

    modes = parse_modes(args.modes, sorted(lst["mode"].dropna().unique().tolist()))
    exclude_target_year = not args.include_target_year_in_baseline

    baseline_lookup = build_baseline_lookup(
        lst=lst,
        lst_value_col=args.lst_value_col,
        modes=modes,
        method=args.baseline_method,
        window_days=args.baseline_window_days,
        exclude_target_year=exclude_target_year,
    )

    need_daily = bool(args.save_daily_status or args.save_all_city_daily)

    metrics_df, daily_df = calculate_metrics(
        lst=lst,
        events=events_merged,
        lst_value_col=args.lst_value_col,
        modes=modes,
        baseline_lookup=baseline_lookup,
        exclude_target_year=exclude_target_year,
        baseline_method=args.baseline_method,
        baseline_window_days=args.baseline_window_days,
        min_event_coverage=args.min_event_coverage,
        min_recovery_coverage=args.min_recovery_coverage,
        recovery_max_days=args.recovery_max_days,
        recovery_threshold_c=args.recovery_threshold_c,
        recovery_consecutive_days=args.recovery_consecutive_days,
        use_event_recovery_days_required=args.use_event_recovery_days_required,
        truncate_recovery_at_next_event=(not args.allow_recovery_overlap_next_event),
        return_daily_status=need_daily,
    )

    # Ensure city_id columns are stable in outputs.
    if not metrics_df.empty:
        metrics_df.insert(0, "city_label", city_label)
        metrics_df.insert(1, "city_id_from_loop", city_id)
    if not daily_df.empty:
        daily_df.insert(0, "city_label", city_label)
        daily_df.insert(1, "city_id_from_loop", city_id)
    if not events_merged.empty:
        events_merged.insert(0, "city_label", city_label)
        events_merged.insert(1, "city_id_from_loop", city_id)

    # Default profile: one file per city.
    # Reduce columns before writing/returning so downstream all-city outputs use the same profile.
    metrics_df = select_output_columns(metrics_df, args.output_columns)

    metrics_path = out_dir / f"{city_label}_thermal_resilience_event_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    merged_path = None
    if args.save_merged_events:
        merged_path = out_dir / f"{city_label}_heatwave_events_merged.csv"
        events_merged.to_csv(merged_path, index=False, encoding="utf-8-sig")

    daily_path = None
    if args.save_daily_status:
        daily_path = out_dir / f"{city_label}_thermal_resilience_daily_status.csv"
        daily_df.to_csv(daily_path, index=False, encoding="utf-8-sig")

    summary = {
        "city_id": city_id,
        "city_label": city_label,
        "lst_path": str(lst_path),
        "hw_path": str(hw_path),
        "metrics_path": str(metrics_path),
        "merged_path": None if merged_path is None else str(merged_path),
        "daily_path": None if daily_path is None else str(daily_path),
        "lst_rows": int(len(lst)),
        "lst_min_date": None if lst.empty else pd.Timestamp(lst["date"].min()).date().isoformat(),
        "lst_max_date": None if lst.empty else pd.Timestamp(lst["date"].max()).date().isoformat(),
        "modes": ";".join(modes),
        "raw_event_count": int(len(events_raw)),
        "merged_event_count": int(len(events_merged)),
        "composite_event_count": int(events_merged["is_composite_event"].sum()) if not events_merged.empty else 0,
        "merge_gap_days": int(args.merge_gap_days),
        "metrics_rows": int(len(metrics_df)),
        "daily_rows": int(len(daily_df)),
        "availability_ok_count": int((metrics_df["availability_flag"] == "ok").sum()) if not metrics_df.empty else 0,
    }

    return metrics_df, daily_df, events_merged, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate thermal resistance and thermal recovery capacity for multiple cities "
            "from MODIS LST and ERA5 heatwave events. By default, output one CSV per city."
        )
    )

    # Multi-city IO
    parser.add_argument("--lst_dir", default=DEFAULT_LST_DIR, help="Folder containing city-level MODIS LST CSVs.")
    parser.add_argument("--hw_dir", default=DEFAULT_HW_DIR, help="Folder containing city-level heatwave event CSVs.")
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR, help="Output root folder.")
    parser.add_argument("--lst_pattern", default=DEFAULT_LST_PATTERN,
                        help="Filename pattern for LST CSVs. Use {city:02d} for city ID.")
    parser.add_argument("--hw_pattern", default=DEFAULT_HW_PATTERN,
                        help="Filename pattern for heatwave CSVs. Use {city:02d} for city ID.")
    parser.add_argument("--city_ids", default="all",
                        help='City IDs to process: "all", "0-9", "0,1,2", or "0-3,7,9".')
    parser.add_argument("--skip_missing", action="store_true",
                        help="Skip cities whose LST or heatwave file is missing. Otherwise raise an error.")

    # Output-profile controls. The default keeps one output file per city.
    parser.add_argument("--output_columns", default=DEFAULT_OUTPUT_COLUMNS,
                        choices=["minimal", "compact", "full"],
                        help=(
                            "Column profile for per-city metric outputs. "
                            "minimal = main metrics only; compact = recommended default; "
                            "full = keep all diagnostic and duplicated hw_* columns."
                        ))
    parser.add_argument("--save_daily_status", action="store_true",
                        help="Also save per-city daily event/recovery status. This adds one extra CSV per city.")
    parser.add_argument("--save_merged_events", action="store_true",
                        help="Also save per-city merged heatwave event tables. Usually unnecessary because merged metadata are in the metrics file.")
    parser.add_argument("--save_processing_summary", action="store_true",
                        help="Also save processing_summary.csv.")
    parser.add_argument("--save_all_city_metrics", action="store_true",
                        help="Also save one combined all-city event-metrics CSV.")
    parser.add_argument("--save_all_city_merged_events", action="store_true",
                        help="Also save one combined all-city merged-event CSV.")
    parser.add_argument("--save_all_city_daily", action="store_true",
                        help="Also save one combined all-city daily-status CSV. This can be very large.")

    # LST and modes
    parser.add_argument("--lst_value_col", default=DEFAULT_LST_VALUE_COL,
                        help="LST column to use, e.g., LST_C_mean or LST_C_median.")
    parser.add_argument("--modes", default=DEFAULT_MODES,
                        help='Modes to calculate: "all", "Day", "Night", or "Day,Night".')

    # Baseline
    parser.add_argument("--baseline_method", default=DEFAULT_BASELINE_METHOD,
                        choices=["mean", "median"], help="Baseline aggregation method.")
    parser.add_argument("--baseline_window_days", type=int, default=DEFAULT_BASELINE_WINDOW_DAYS,
                        help="Centered climatological window size. 5 means +/-2 days.")
    parser.add_argument("--include_target_year_in_baseline", action="store_true",
                        help="Use this flag to include the target year in baseline calculation.")

    # Event merging
    parser.add_argument("--merge_gap_days", type=int, default=DEFAULT_MERGE_GAP_DAYS,
                        help=(
                            "Merge two adjacent heatwave events when the number of non-heatwave days "
                            "between them is <= this threshold. Use 3, 5, or 7 for sensitivity analysis. "
                            "Use -1 to disable merging."
                        ))

    # Coverage and recovery
    parser.add_argument("--min_event_coverage", type=float, default=DEFAULT_MIN_EVENT_COVERAGE,
                        help="Minimum proportion of event days with available MODIS LST.")
    parser.add_argument("--min_recovery_coverage", type=float, default=DEFAULT_MIN_RECOVERY_COVERAGE,
                        help="Minimum proportion of recovery days with available MODIS LST.")
    parser.add_argument("--recovery_max_days", type=int, default=DEFAULT_RECOVERY_MAX_DAYS,
                        help="Maximum days after heatwave end to search for recovery.")
    parser.add_argument("--recovery_threshold_c", type=float, default=DEFAULT_RECOVERY_THRESHOLD_C,
                        help="Positive anomaly threshold for recovered state, in Celsius.")
    parser.add_argument("--recovery_consecutive_days", type=int, default=DEFAULT_RECOVERY_CONSECUTIVE_DAYS,
                        help="Number of consecutive days required to confirm recovery.")
    parser.add_argument("--use_event_recovery_days_required", action="store_true",
                        help="Use recovery_days_required column in heatwave CSV as max search days if present.")
    parser.add_argument("--allow_recovery_overlap_next_event", action="store_true",
                        help=(
                            "Allow recovery search to include days after the next merged heatwave starts. "
                            "By default, recovery search is truncated before the next merged heatwave."
                        ))

    args = parser.parse_args()

    lst_dir = Path(args.lst_dir)
    hw_dir = Path(args.hw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    city_ids = parse_city_ids(args.city_ids, lst_dir=lst_dir, lst_pattern=args.lst_pattern)
    if not city_ids:
        raise RuntimeError(f"No cities found. Check --lst_dir and --lst_pattern. lst_dir={lst_dir}")

    print("Multi-city thermal resilience calculation")
    print(f"  LST folder: {lst_dir}")
    print(f"  Heatwave folder: {hw_dir}")
    print(f"  Output folder: {out_dir}")
    print(f"  City IDs: {city_ids}")
    print(f"  Merge gap days: {args.merge_gap_days}")
    print(f"  Output columns: {args.output_columns}")
    print("  Default output: one event-metrics CSV per city")
    print()

    all_metrics: List[pd.DataFrame] = []
    all_daily: List[pd.DataFrame] = []
    all_merged_events: List[pd.DataFrame] = []
    summaries: List[dict] = []

    for city_id in city_ids:
        city_label = format_city_id(city_id)
        lst_path = lst_dir / args.lst_pattern.format(city=city_id)
        hw_path = hw_dir / args.hw_pattern.format(city=city_id)

        if not lst_path.exists() or not hw_path.exists():
            msg = f"[{city_label}] Missing file(s):"
            if not lst_path.exists():
                msg += f"\n  LST not found: {lst_path}"
            if not hw_path.exists():
                msg += f"\n  Heatwave not found: {hw_path}"
            if args.skip_missing:
                print(msg + "\n  -> skipped")
                continue
            raise FileNotFoundError(msg)

        print(f"[{city_label}] Processing")
        print(f"  LST: {lst_path.name}")
        print(f"  HW : {hw_path.name}")

        metrics_df, daily_df, merged_events_df, summary = process_one_city(
            city_id=city_id,
            lst_path=lst_path,
            hw_path=hw_path,
            out_dir=out_dir,
            args=args,
        )

        summaries.append(summary)
        if args.save_all_city_metrics:
            all_metrics.append(metrics_df)
        if args.save_all_city_merged_events:
            all_merged_events.append(merged_events_df)
        if args.save_all_city_daily:
            all_daily.append(daily_df)

        print(
            f"  raw events={summary['raw_event_count']}, "
            f"merged events={summary['merged_event_count']}, "
            f"composite events={summary['composite_event_count']}, "
            f"metric rows={summary['metrics_rows']}, ok rows={summary['availability_ok_count']}"
        )
        print(f"  output: {Path(summary['metrics_path']).name}")
        if not metrics_df.empty:
            print("  Availability summary:")
            print(metrics_df["availability_flag"].value_counts(dropna=False).to_string().replace("\n", "\n    "))
        print()

    if args.save_processing_summary:
        summary_df = pd.DataFrame(summaries)
        summary_path = out_dir / "processing_summary.csv"
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    else:
        summary_path = None

    if args.save_all_city_metrics:
        all_metrics_df = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
        all_metrics_path = out_dir / "thermal_resilience_event_metrics_all_cities.csv"
        all_metrics_df.to_csv(all_metrics_path, index=False, encoding="utf-8-sig")
    else:
        all_metrics_path = None

    if args.save_all_city_merged_events:
        all_merged_df = pd.concat(all_merged_events, ignore_index=True) if all_merged_events else pd.DataFrame()
        all_merged_path = out_dir / "heatwave_events_merged_all_cities.csv"
        all_merged_df.to_csv(all_merged_path, index=False, encoding="utf-8-sig")
    else:
        all_merged_path = None

    if args.save_all_city_daily:
        all_daily_df = pd.concat(all_daily, ignore_index=True) if all_daily else pd.DataFrame()
        all_daily_path = out_dir / "thermal_resilience_daily_status_all_cities.csv"
        all_daily_df.to_csv(all_daily_path, index=False, encoding="utf-8-sig")
    else:
        all_daily_path = None

    print("All done.")
    print(f"  City-level metrics files are saved in: {out_dir}")
    if summary_path is not None:
        print(f"  Summary: {summary_path}")
    if all_metrics_path is not None:
        print(f"  All-city metrics: {all_metrics_path}")
    if all_merged_path is not None:
        print(f"  All-city merged events: {all_merged_path}")
    if all_daily_path is not None:
        print(f"  All-city daily status: {all_daily_path}")


if __name__ == "__main__":
    main()


