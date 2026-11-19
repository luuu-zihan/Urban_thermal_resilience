# -*- coding: utf-8 -*-
"""
Process ERA5-Land daily Tmax CSV exported from GEE and identify heatwave events.

Output design in this version:
    1. main_results/ contains only per-city heatwave event CSV files.
       Example:
           main_results/City_00_HeatwaveEvents_2000_2025_Baseline1991_2020_Win15d.csv

    2. auxiliary/ contains all non-primary outputs, but every auxiliary output is also
       split by city and stored in its own subfolder.
       Example:
           auxiliary/city_summary/City_00_CityHeatwaveSummary_*.csv
           auxiliary/annual_summary/City_00_AnnualHeatwaveSummary_*.csv
           auxiliary/thresholds/City_00_DOY_P90_Thresholds_*.csv          optional
           auxiliary/daily_hotday/City_00_DailyHotday_*.csv               optional

Main logic:
    1. Use 1991-2020 as fixed climatological baseline.
    2. Use DOY +/- 7 days moving window, i.e. 15-day window.
    3. Compute P90 threshold for each city and each no-leap DOY.
    4. Identify hot days during 2000-2025.
    5. Consecutive hot days >= 3 are heatwave events.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


# =========================
# 0. User configuration
# =========================

# GEE 导出的每日 Tmax CSV 文件夹
INPUT_DIR = Path("data/daily_tmax")

# 输出根目录
OUTPUT_ROOT = Path("outputs/heatwaves")

# 主结果目录：只放每个城市单独的热浪事件文件
MAIN_OUTPUT_DIR = OUTPUT_ROOT

# 辅助结果目录：所有非主结果均放在这里，并按城市拆分到不同子文件夹
AUX_OUTPUT_DIR = OUTPUT_ROOT / "auxiliary"
AUX_CITY_SUMMARY_DIR = AUX_OUTPUT_DIR / "city_summary"
AUX_ANNUAL_SUMMARY_DIR = AUX_OUTPUT_DIR / "annual_summary"
AUX_THRESHOLD_DIR = AUX_OUTPUT_DIR / "thresholds"
AUX_DAILY_HOTDAY_DIR = AUX_OUTPUT_DIR / "daily_hotday"

# GEE 导出的文件匹配模式
FILE_PATTERN = "City_*_ERA5Land_DailyTmax_1991_2025.csv"

# 阈值基线期
BASELINE_START_YEAR = 1991
BASELINE_END_YEAR = 2020

# 事件识别期
EVENT_START_YEAR = 2000
EVENT_END_YEAR = 2025

# 热浪定义
PERCENTILE = 90
WINDOW_RADIUS = 7       # DOY ±7 天，总窗口 15 天
MIN_DURATION = 3        # 连续 >=3 天

# MODIS LST 可用期，仅做日期层面标记
MODIS_START_DATE = pd.Timestamp("2000-02-24")
MODIS_END_DATE = pd.Timestamp("2026-01-01")  # 开区间

# 恢复力分析需要事件结束后多少天
RECOVERY_DAYS = 10

# -------------------------
# 输出开关
# -------------------------

# 主结果：每个城市单独的热浪事件文件。建议保持 True。
SAVE_PER_CITY_HEATWAVE_EVENTS = True

# 辅助结果：也都按城市拆分保存。
SAVE_PER_CITY_SUMMARY = True
SAVE_PER_CITY_ANNUAL_SUMMARY = True

# 较大的辅助表默认关闭，需要检查阈值或逐日匹配 MODIS 时再打开。
SAVE_PER_CITY_THRESHOLD_TABLE = True
SAVE_PER_CITY_DAILY_HOTDAY_TABLE = True


# =========================
# 0.1 Fixed output columns
# =========================

# 事件表固定列。即使某个城市没有热浪，也会输出带表头的空 CSV。
EVENT_COLUMNS = [
    "city_id", "city_index", "event_id", "event_seq", "raw_event_id",
    "start_date", "end_date", "start_year", "end_year", "duration_days",
    "mean_Tmax_C", "max_Tmax_C", "mean_threshold_C", "mean_excess_C", "max_excess_C",
    "baseline_period", "event_period", "percentile", "min_duration_days",
    "window_radius_days", "window_total_days",
    "modis_usable_for_resistance_by_date_int",
    "modis_usable_for_recovery_by_date_int",
    "recovery_days_required",
]

CITY_SUMMARY_COLUMNS = [
    "city_id", "city_index", "n_heatwave_events",
    "n_modis_date_usable_resistance_events", "n_modis_date_usable_recovery_events",
    "modis_date_usable_resistance_fraction", "modis_date_usable_recovery_fraction",
    "mean_duration_days", "max_duration_days",
    "mean_excess_C", "max_excess_C", "max_Tmax_C",
    "baseline_period", "event_period", "percentile",
    "window_radius_days", "window_total_days", "min_duration_days",
]

ANNUAL_SUMMARY_COLUMNS = [
    "city_id", "city_index", "year",
    "n_heatwave_events", "total_heatwave_days",
    "mean_duration_days", "max_duration_days",
    "mean_excess_C", "max_excess_C", "max_Tmax_C",
    "baseline_period", "event_period", "percentile",
    "window_radius_days", "window_total_days", "min_duration_days",
]


# =========================
# 1. Date / DOY utilities
# =========================

def compute_noleap_doy0(date_series: pd.Series) -> pd.Series:
    """
    Compute no-leap 0-based DOY.

    Jan 1 = 0, Dec 31 = 364.
    Feb 29 is marked as NaN and should be removed.
    For leap years after Feb 29, DOY is shifted by -1.
    """
    dates = pd.to_datetime(date_series)

    raw_doy0 = dates.dt.dayofyear - 1
    is_leap = dates.dt.is_leap_year
    is_feb29 = (dates.dt.month == 2) & (dates.dt.day == 29)
    after_feb = dates.dt.month > 2

    noleap_doy0 = raw_doy0.copy().astype("float")
    noleap_doy0.loc[is_leap & after_feb] = noleap_doy0.loc[is_leap & after_feb] - 1
    noleap_doy0.loc[is_feb29] = np.nan

    return noleap_doy0


def circular_doy_distance(doy_values: np.ndarray, target_doy: int, year_length: int = 365) -> np.ndarray:
    """Circular distance on a no-leap 365-day calendar."""
    diff = np.abs(doy_values - target_doy)
    return np.minimum(diff, year_length - diff)


# =========================
# 2. Load GEE daily Tmax CSV
# =========================

def load_one_city_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load one city CSV exported from GEE.

    Expected fields:
        city_id, city_index, date, year, month, day, doy_noleap0, Tmax_C
    """
    df = pd.read_csv(csv_path)

    # 兼容旧版 GEE 导出结果
    df = df.drop(columns=[".geo", "system:index"], errors="ignore")

    if "date" not in df.columns:
        raise ValueError(f"{csv_path.name} 缺少 date 字段。")

    if "Tmax_C" not in df.columns:
        raise ValueError(f"{csv_path.name} 缺少 Tmax_C 字段。")

    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day

    if "city_id" not in df.columns:
        df["city_id"] = csv_path.stem.split("_ERA5Land")[0]

    if "city_index" not in df.columns:
        try:
            df["city_index"] = int(str(df["city_id"].iloc[0]).split("_")[-1])
        except Exception:
            df["city_index"] = -1

    if "doy_noleap0" not in df.columns:
        df["doy_noleap0"] = compute_noleap_doy0(df["date"])
    else:
        # 仍然检查 Feb29，把它设为 NaN 并删除
        is_feb29 = (df["month"] == 2) & (df["day"] == 29)
        df.loc[is_feb29, "doy_noleap0"] = np.nan

    df["Tmax_C"] = pd.to_numeric(df["Tmax_C"], errors="coerce")
    df["doy_noleap0"] = pd.to_numeric(df["doy_noleap0"], errors="coerce")

    df = df.dropna(subset=["Tmax_C", "doy_noleap0"]).copy()
    df["doy_noleap0"] = df["doy_noleap0"].astype(int)
    df["city_index"] = pd.to_numeric(df["city_index"], errors="coerce").fillna(-1).astype(int)

    return df.sort_values("date").reset_index(drop=True)


def load_all_city_csvs(input_dir: Path, pattern: str) -> pd.DataFrame:
    csv_paths = sorted(input_dir.glob(pattern))

    if not csv_paths:
        raise FileNotFoundError(f"没有找到 CSV 文件: {input_dir / pattern}")

    dfs = []
    for path in tqdm(csv_paths, desc="Loading city CSVs"):
        dfs.append(load_one_city_csv(path))

    all_df = pd.concat(dfs, ignore_index=True)
    return all_df.sort_values(["city_id", "date"]).reset_index(drop=True)


# =========================
# 3. Threshold calculation
# =========================

def compute_city_thresholds(
    city_df: pd.DataFrame,
    baseline_start_year: int,
    baseline_end_year: int,
    percentile: float,
    window_radius: int,
) -> pd.DataFrame:
    """Compute DOY P90 threshold for one city based on baseline years."""
    city_id = city_df["city_id"].iloc[0]
    city_index = int(city_df["city_index"].iloc[0])

    baseline = city_df[
        (city_df["year"] >= baseline_start_year) &
        (city_df["year"] <= baseline_end_year)
    ].copy()

    if baseline.empty:
        raise ValueError(f"{city_id} 没有基线期数据。")

    doy_values = baseline["doy_noleap0"].to_numpy()
    tmax_values = baseline["Tmax_C"].to_numpy()

    records = []
    for target_doy in range(365):
        dist = circular_doy_distance(doy_values, target_doy, year_length=365)
        mask = dist <= window_radius
        samples = tmax_values[mask]
        samples = samples[np.isfinite(samples)]

        threshold = np.nan if len(samples) == 0 else np.nanpercentile(samples, percentile)

        records.append({
            "city_id": city_id,
            "city_index": city_index,
            "doy_noleap0": target_doy,
            "Tmax_P90_C": threshold,
            "n_baseline_samples": int(len(samples)),
            "baseline_start_year": baseline_start_year,
            "baseline_end_year": baseline_end_year,
            "percentile": percentile,
            "window_radius_days": window_radius,
            "window_total_days": window_radius * 2 + 1,
        })

    return pd.DataFrame(records)


def apply_thresholds(city_df: pd.DataFrame, threshold_df: pd.DataFrame) -> pd.DataFrame:
    """Merge daily Tmax with DOY threshold and compute hot-day flag."""
    out = city_df.merge(
        threshold_df[["city_id", "doy_noleap0", "Tmax_P90_C"]],
        on=["city_id", "doy_noleap0"],
        how="left",
    )

    out["excess_C"] = out["Tmax_C"] - out["Tmax_P90_C"]
    out["hot_day"] = out["Tmax_C"] > out["Tmax_P90_C"]
    out["hot_day_int"] = out["hot_day"].astype(int)
    return out


# =========================
# 4. Heatwave event detection
# =========================

def identify_heatwave_events(
    daily_df: pd.DataFrame,
    event_start_year: int,
    event_end_year: int,
    min_duration: int,
    modis_start_date: pd.Timestamp,
    modis_end_date: pd.Timestamp,
    recovery_days: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identify consecutive hot-day events for one city.

    Returns:
        daily_event_df: daily table within event period with raw_event_id and is_heatwave_event_day
        event_df: heatwave event summary table
    """
    city_id = daily_df["city_id"].iloc[0]
    city_index = int(daily_df["city_index"].iloc[0])

    df = daily_df[
        (daily_df["year"] >= event_start_year) &
        (daily_df["year"] <= event_end_year)
    ].copy()

    df = df.sort_values("date").reset_index(drop=True)

    raw_event_ids = []
    run_lens = []

    current_event_id = 0
    current_run_len = 0
    prev_hot = False
    prev_date = None

    for _, row in df.iterrows():
        hot = bool(row["hot_day"])
        date = row["date"]

        if prev_date is None:
            is_consecutive_date = True
        else:
            # 删除 Feb29 后，闰年 Feb28 -> Mar1 的真实日期间隔是 2 天，所以允许 <=2。
            gap_days = (date - prev_date).days
            is_consecutive_date = gap_days <= 2

        if hot:
            if (not prev_hot) or (not is_consecutive_date):
                current_event_id += 1
                current_run_len = 1
            else:
                current_run_len += 1

            raw_event_ids.append(current_event_id)
            run_lens.append(current_run_len)
        else:
            current_run_len = 0
            raw_event_ids.append(-9999)
            run_lens.append(0)

        prev_hot = hot
        prev_date = date

    df["raw_event_id"] = raw_event_ids
    df["current_run_len"] = run_lens

    event_records = []
    candidate_ids = sorted([eid for eid in df["raw_event_id"].unique() if eid != -9999])

    for eid in candidate_ids:
        event_days = df[df["raw_event_id"] == eid].copy()
        duration = len(event_days)

        if duration < min_duration:
            continue

        start_date = event_days["date"].min()
        end_date = event_days["date"].max()
        recovery_end_date = end_date + pd.Timedelta(days=recovery_days)

        modis_usable_for_resistance = (
            (start_date >= modis_start_date) and
            (end_date < modis_end_date)
        )

        modis_usable_for_recovery = (
            (start_date >= modis_start_date) and
            (recovery_end_date < modis_end_date)
        )

        event_records.append({
            "city_id": city_id,
            "city_index": city_index,
            "raw_event_id": eid,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "start_year": int(start_date.year),
            "end_year": int(end_date.year),
            "duration_days": int(duration),
            "mean_Tmax_C": float(event_days["Tmax_C"].mean()),
            "max_Tmax_C": float(event_days["Tmax_C"].max()),
            "mean_threshold_C": float(event_days["Tmax_P90_C"].mean()),
            "mean_excess_C": float(event_days["excess_C"].mean()),
            "max_excess_C": float(event_days["excess_C"].max()),
            "baseline_period": f"{BASELINE_START_YEAR}-{BASELINE_END_YEAR}",
            "event_period": f"{EVENT_START_YEAR}-{EVENT_END_YEAR}",
            "percentile": PERCENTILE,
            "min_duration_days": MIN_DURATION,
            "window_radius_days": WINDOW_RADIUS,
            "window_total_days": WINDOW_RADIUS * 2 + 1,
            # 这里只是日期层面能否和 MODIS 对接，不代表 MODIS LST QC 合格。
            "modis_usable_for_resistance_by_date_int": int(modis_usable_for_resistance),
            "modis_usable_for_recovery_by_date_int": int(modis_usable_for_recovery),
            "recovery_days_required": recovery_days,
        })

    event_df = pd.DataFrame(event_records)

    # 让正式事件编号连续；raw_event_id 保留为内部候选段编号。
    if not event_df.empty:
        event_df = event_df.sort_values(["city_id", "start_date"]).reset_index(drop=True)
        event_df["event_seq"] = event_df.groupby("city_id").cumcount() + 1
        event_df["event_id"] = event_df.apply(
            lambda r: f"{r['city_id']}_HW_{int(r['event_seq']):04d}",
            axis=1,
        )
        event_df = event_df[EVENT_COLUMNS]
    else:
        event_df = pd.DataFrame(columns=EVENT_COLUMNS)

    valid_event_ids = set(event_df["raw_event_id"].tolist()) if not event_df.empty else set()
    df["is_heatwave_event_day"] = df["raw_event_id"].isin(valid_event_ids)
    df["is_heatwave_event_day_int"] = df["is_heatwave_event_day"].astype(int)

    return df, event_df


# =========================
# 5. Summary tables
# =========================

def build_city_summary_for_city(city_df: pd.DataFrame, event_df: pd.DataFrame) -> pd.DataFrame:
    """Build a one-row city summary for one city."""
    city_id = city_df["city_id"].iloc[0]
    city_index = int(city_df["city_index"].iloc[0])

    if event_df.empty:
        row = {
            "city_id": city_id,
            "city_index": city_index,
            "n_heatwave_events": 0,
            "n_modis_date_usable_resistance_events": 0,
            "n_modis_date_usable_recovery_events": 0,
            "modis_date_usable_resistance_fraction": np.nan,
            "modis_date_usable_recovery_fraction": np.nan,
            "mean_duration_days": np.nan,
            "max_duration_days": np.nan,
            "mean_excess_C": np.nan,
            "max_excess_C": np.nan,
            "max_Tmax_C": np.nan,
        }
    else:
        total = len(event_df)
        n_res = int(event_df["modis_usable_for_resistance_by_date_int"].sum())
        n_rec = int(event_df["modis_usable_for_recovery_by_date_int"].sum())
        row = {
            "city_id": city_id,
            "city_index": city_index,
            "n_heatwave_events": total,
            "n_modis_date_usable_resistance_events": n_res,
            "n_modis_date_usable_recovery_events": n_rec,
            "modis_date_usable_resistance_fraction": n_res / total if total > 0 else np.nan,
            "modis_date_usable_recovery_fraction": n_rec / total if total > 0 else np.nan,
            "mean_duration_days": event_df["duration_days"].mean(),
            "max_duration_days": event_df["duration_days"].max(),
            "mean_excess_C": event_df["mean_excess_C"].mean(),
            "max_excess_C": event_df["max_excess_C"].max(),
            "max_Tmax_C": event_df["max_Tmax_C"].max(),
        }

    row.update({
        "baseline_period": f"{BASELINE_START_YEAR}-{BASELINE_END_YEAR}",
        "event_period": f"{EVENT_START_YEAR}-{EVENT_END_YEAR}",
        "percentile": PERCENTILE,
        "window_radius_days": WINDOW_RADIUS,
        "window_total_days": WINDOW_RADIUS * 2 + 1,
        "min_duration_days": MIN_DURATION,
    })

    return pd.DataFrame([row], columns=CITY_SUMMARY_COLUMNS)


def build_annual_summary_for_city(city_df: pd.DataFrame, event_df: pd.DataFrame) -> pd.DataFrame:
    """Build annual heatwave summary for one city."""
    city_id = city_df["city_id"].iloc[0]
    city_index = int(city_df["city_index"].iloc[0])
    years = list(range(EVENT_START_YEAR, EVENT_END_YEAR + 1))

    if event_df.empty:
        annual = pd.DataFrame({
            "city_id": city_id,
            "city_index": city_index,
            "year": years,
            "n_heatwave_events": 0,
            "total_heatwave_days": 0,
            "mean_duration_days": np.nan,
            "max_duration_days": np.nan,
            "mean_excess_C": np.nan,
            "max_excess_C": np.nan,
            "max_Tmax_C": np.nan,
        })
    else:
        grouped = (
            event_df
            .groupby("start_year", as_index=False)
            .agg(
                n_heatwave_events=("event_id", "count"),
                total_heatwave_days=("duration_days", "sum"),
                mean_duration_days=("duration_days", "mean"),
                max_duration_days=("duration_days", "max"),
                mean_excess_C=("mean_excess_C", "mean"),
                max_excess_C=("max_excess_C", "max"),
                max_Tmax_C=("max_Tmax_C", "max"),
            )
            .rename(columns={"start_year": "year"})
        )
        template = pd.DataFrame({
            "city_id": city_id,
            "city_index": city_index,
            "year": years,
        })
        annual = template.merge(grouped, on="year", how="left")
        annual["n_heatwave_events"] = annual["n_heatwave_events"].fillna(0).astype(int)
        annual["total_heatwave_days"] = annual["total_heatwave_days"].fillna(0).astype(int)

    annual["baseline_period"] = f"{BASELINE_START_YEAR}-{BASELINE_END_YEAR}"
    annual["event_period"] = f"{EVENT_START_YEAR}-{EVENT_END_YEAR}"
    annual["percentile"] = PERCENTILE
    annual["window_radius_days"] = WINDOW_RADIUS
    annual["window_total_days"] = WINDOW_RADIUS * 2 + 1
    annual["min_duration_days"] = MIN_DURATION
    return annual[ANNUAL_SUMMARY_COLUMNS]


# =========================
# 6. Output helpers
# =========================

def make_output_dirs() -> None:
    """Create output directories based on enabled switches."""
    MAIN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    AUX_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if SAVE_PER_CITY_SUMMARY:
        AUX_CITY_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    if SAVE_PER_CITY_ANNUAL_SUMMARY:
        AUX_ANNUAL_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    if SAVE_PER_CITY_THRESHOLD_TABLE:
        AUX_THRESHOLD_DIR.mkdir(parents=True, exist_ok=True)
    if SAVE_PER_CITY_DAILY_HOTDAY_TABLE:
        AUX_DAILY_HOTDAY_DIR.mkdir(parents=True, exist_ok=True)


def output_tag() -> str:
    return (
        f"{EVENT_START_YEAR}_{EVENT_END_YEAR}_"
        f"Baseline{BASELINE_START_YEAR}_{BASELINE_END_YEAR}_"
        f"Win{WINDOW_RADIUS * 2 + 1}d"
    )


# =========================
# 7. Main workflow
# =========================

def main() -> None:
    make_output_dirs()
    tag = output_tag()

    print("========== Loading GEE CSVs ==========")
    all_daily_raw = load_all_city_csvs(INPUT_DIR, FILE_PATTERN)

    print(f"Loaded rows: {len(all_daily_raw):,}")
    print(f"Cities: {all_daily_raw['city_id'].nunique()}")
    print(all_daily_raw.head())

    city_ids = sorted(all_daily_raw["city_id"].unique())

    print("\n========== Processing cities ==========")
    for city_id in tqdm(city_ids, desc="Processing cities"):
        city_df = all_daily_raw[all_daily_raw["city_id"] == city_id].copy()

        threshold_df = compute_city_thresholds(
            city_df=city_df,
            baseline_start_year=BASELINE_START_YEAR,
            baseline_end_year=BASELINE_END_YEAR,
            percentile=PERCENTILE,
            window_radius=WINDOW_RADIUS,
        )

        daily_with_threshold = apply_thresholds(city_df, threshold_df)

        daily_event_df, event_df = identify_heatwave_events(
            daily_df=daily_with_threshold,
            event_start_year=EVENT_START_YEAR,
            event_end_year=EVENT_END_YEAR,
            min_duration=MIN_DURATION,
            modis_start_date=MODIS_START_DATE,
            modis_end_date=MODIS_END_DATE,
            recovery_days=RECOVERY_DAYS,
        )

        city_summary_df = build_city_summary_for_city(city_df, event_df)
        annual_summary_df = build_annual_summary_for_city(city_df, event_df)

        if SAVE_PER_CITY_HEATWAVE_EVENTS:
            path = MAIN_OUTPUT_DIR / f"{city_id}_HeatwaveEvents_{tag}.csv"
            event_df.to_csv(path, index=False, encoding="utf-8-sig")

        if SAVE_PER_CITY_SUMMARY:
            path = AUX_CITY_SUMMARY_DIR / f"{city_id}_CityHeatwaveSummary_{tag}.csv"
            city_summary_df.to_csv(path, index=False, encoding="utf-8-sig")

        if SAVE_PER_CITY_ANNUAL_SUMMARY:
            path = AUX_ANNUAL_SUMMARY_DIR / f"{city_id}_AnnualHeatwaveSummary_{tag}.csv"
            annual_summary_df.to_csv(path, index=False, encoding="utf-8-sig")

        if SAVE_PER_CITY_THRESHOLD_TABLE:
            path = AUX_THRESHOLD_DIR / f"{city_id}_DOY_P90_Thresholds_Baseline{BASELINE_START_YEAR}_{BASELINE_END_YEAR}_Win{WINDOW_RADIUS * 2 + 1}d.csv"
            threshold_df.to_csv(path, index=False, encoding="utf-8-sig")

        if SAVE_PER_CITY_DAILY_HOTDAY_TABLE:
            path = AUX_DAILY_HOTDAY_DIR / f"{city_id}_DailyHotday_{tag}.csv"
            daily_event_df.to_csv(path, index=False, encoding="utf-8-sig")

    print("\nDone.")
    print(f"Main heatwave event files: {MAIN_OUTPUT_DIR}")
    print(f"Auxiliary files root:      {AUX_OUTPUT_DIR}")
    if SAVE_PER_CITY_SUMMARY:
        print(f"  - City summaries:        {AUX_CITY_SUMMARY_DIR}")
    if SAVE_PER_CITY_ANNUAL_SUMMARY:
        print(f"  - Annual summaries:      {AUX_ANNUAL_SUMMARY_DIR}")
    if SAVE_PER_CITY_THRESHOLD_TABLE:
        print(f"  - Threshold tables:      {AUX_THRESHOLD_DIR}")
    if SAVE_PER_CITY_DAILY_HOTDAY_TABLE:
        print(f"  - Daily hot-day tables:  {AUX_DAILY_HOTDAY_DIR}")


if __name__ == "__main__":
    main()

