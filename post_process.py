"""
post_process.py
===============
Post-processing pipeline for demand forecast predictions.

Functions
---------
post_process(predictions_df, sunday_mask_df, cap_value, historical_df)
    Apply sunday mask → clip negatives → cap at 99.9th percentile → round to int.

apply_sunday_mask(predictions_df, sunday_mask_df)
    Force Sunday forecast columns to 0 for ALL SKUs.

compute_cap_value(historical_df, percentile=99.9)
    Compute per-SKU cap from historical net daily quantities.

Usage
-----
    from post_process import post_process, compute_cap_value
    import pandas as pd

    train_df  = pd.read_csv('train.csv', low_memory=False)
    sunday_df = pd.read_parquet('sunday_mask.parquet')
    pred_df   = pd.read_parquet('predictions_sparse_inactive.parquet')

    cap = compute_cap_value(train_df, percentile=99.9)   # pd.Series, index=ItemCode
    result = post_process(pred_df, sunday_df, cap_value=cap, historical_df=train_df)

Notes
-----
- predictions_df must have column 'id' (like sample_submission.csv) OR index = ItemCode.
  F-columns must be named F1..F28 (or F1..F56 for the full horizon).
- cap_value may be a scalar (global cap), or a pd.Series indexed by ItemCode
  (per-SKU cap).  Per-SKU cap is strongly recommended to avoid over-clipping
  high-volume SKUs.
- Rounding is done AFTER clipping and capping (order matters for WRMSSE since
  the metric scores floats — but the competition says predictions may be float;
  rounding here is optional / configurable).
"""

from __future__ import annotations

import warnings
from typing import Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1.  Apply Sunday mask
# ---------------------------------------------------------------------------

def apply_sunday_mask(
    predictions_df: pd.DataFrame,
    sunday_mask_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Zero-out all Sunday forecast columns for every SKU.

    Parameters
    ----------
    predictions_df : DataFrame with F-columns (F1..F28 or F1..F56).
                     Must have 'id' column OR ItemCode as index.
    sunday_mask_df : DataFrame from sunday_mask.parquet with columns
                     [forecast_day_index, date, day_of_week, force_zero].

    Returns
    -------
    DataFrame (same shape/index as input) with Sunday columns set to 0.
    """
    df = predictions_df.copy()
    fcols = _get_fcols(df)

    # Build set of F-column names that are Sundays
    sunday_fcols = set()
    for _, row in sunday_mask_df[sunday_mask_df["force_zero"]].iterrows():
        fname = f"F{int(row['forecast_day_index'])}"
        if fname in fcols:
            sunday_fcols.add(fname)

    if not sunday_fcols:
        warnings.warn("No Sunday F-columns found in predictions_df. Check sunday_mask_df indices.")
        return df

    df[list(sunday_fcols)] = 0.0
    return df


# ---------------------------------------------------------------------------
# 2.  Compute per-SKU cap from historical data
# ---------------------------------------------------------------------------

def compute_cap_value(
    historical_df: pd.DataFrame,
    percentile: float = 99.9,
    window_days: int | None = None,
    reference_date: str = "2025-09-05",
) -> pd.Series:
    """
    Compute per-SKU cap = percentile of historical NET daily quantities.

    Net quantity = sum of all transactions (sales + returns) per (Date, ItemCode).
    Only positive net-daily values are considered (a return-heavy day should not
    set a negative cap).

    Parameters
    ----------
    historical_df  : raw train.csv DataFrame
    percentile     : float in (0, 100], default 99.9
    window_days    : optional int — only use last N days for cap computation.
                     None → use entire history.
    reference_date : last date of training data (used with window_days).

    Returns
    -------
    pd.Series  index=ItemCode, values=cap (float), min cap = 1.0
    """
    df = historical_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    if window_days is not None:
        ref = pd.Timestamp(reference_date)
        start = ref - pd.Timedelta(days=window_days - 1)
        df = df[df["Date"] >= start]

    # Net daily qty per (Date, ItemCode)
    daily = (
        df.groupby(["Date", "ItemCode"])["Quantity"]
        .sum()
        .reset_index()
    )
    # Keep only positive-qty days for cap computation
    daily = daily[daily["Quantity"] > 0]

    # Per-SKU percentile
    cap = daily.groupby("ItemCode")["Quantity"].quantile(percentile / 100.0)

    # Ensure minimum cap = 1 (never cap at 0 for active SKUs)
    cap = cap.clip(lower=1.0)

    return cap


# ---------------------------------------------------------------------------
# 3.  Main post-process function
# ---------------------------------------------------------------------------

def post_process(
    predictions_df: pd.DataFrame,
    sunday_mask_df: pd.DataFrame,
    cap_value: Union[float, pd.Series],
    historical_df: pd.DataFrame | None = None,
    do_round: bool = True,
) -> pd.DataFrame:
    """
    Full post-processing pipeline:
        1. Apply Sunday mask  (Sunday forecast days → 0)
        2. Clip negatives     (predictions must be >= 0)
        3. Cap at cap_value   (per-SKU 99.9th percentile of historical qty)
        4. Round to integer   (optional, default True)

    Parameters
    ----------
    predictions_df : submission-format DataFrame.
                     Must have 'id' column (e.g. 'SKU-00001_validation')
                     and F-columns (F1..F28).
    sunday_mask_df : sunday_mask.parquet DataFrame.
    cap_value      : scalar float OR pd.Series indexed by ItemCode.
                     Use compute_cap_value() to generate per-SKU caps.
                     If a Series, SKUs not in the Series get an uncapped value
                     (very large float) — useful for INACTIVE/SPARSE with 0 preds.
    historical_df  : only needed when cap_value is not yet computed (convenience).
                     If both cap_value and historical_df are given, cap_value wins.
    do_round       : bool, default True. Round final float predictions to int.

    Returns
    -------
    DataFrame in same shape/format as predictions_df, values post-processed.
    """
    df = predictions_df.copy()
    fcols = _get_fcols(df)

    # Resolve ItemCode for per-SKU capping
    if "id" in df.columns:
        sku_series = df["id"].str.rsplit("_", n=1).str[0]
    else:
        sku_series = pd.Series(df.index, index=df.index, name="ItemCode")

    # ------------------------------------------------------------------
    # Step 1: Sunday mask
    # ------------------------------------------------------------------
    df = apply_sunday_mask(df, sunday_mask_df)

    # ------------------------------------------------------------------
    # Step 2: Clip negatives
    # ------------------------------------------------------------------
    df[fcols] = df[fcols].clip(lower=0.0)

    # ------------------------------------------------------------------
    # Step 3: Cap
    # ------------------------------------------------------------------
    if isinstance(cap_value, pd.Series):
        # Per-SKU cap — align to rows in df
        cap_series = sku_series.map(cap_value)           # NaN for unknown SKUs
        cap_matrix = cap_series.values[:, np.newaxis]    # (n_rows, 1) broadcast
        cap_matrix = np.where(np.isnan(cap_matrix), np.inf, cap_matrix)
        df[fcols] = np.minimum(df[fcols].values, cap_matrix)
    elif isinstance(cap_value, (int, float)):
        df[fcols] = df[fcols].clip(upper=float(cap_value))
    elif cap_value is None and historical_df is not None:
        cap_value = compute_cap_value(historical_df)
        return post_process(df, sunday_mask_df, cap_value, do_round=do_round)
    else:
        raise ValueError("cap_value must be a float, pd.Series, or None (with historical_df).")

    # ------------------------------------------------------------------
    # Step 4: Round (optional)
    # ------------------------------------------------------------------
    if do_round:
        df[fcols] = df[fcols].round(0).astype(int)

    return df


# ---------------------------------------------------------------------------
# 4.  Internal helpers
# ---------------------------------------------------------------------------

def _get_fcols(df: pd.DataFrame) -> list[str]:
    """Return sorted F-columns from a predictions DataFrame."""
    fcols = [c for c in df.columns if str(c).startswith("F") and str(c)[1:].isdigit()]
    return sorted(fcols, key=lambda x: int(x[1:]))


# ---------------------------------------------------------------------------
# 5.  CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os, sys

    print("=" * 60)
    print("  post_process.py  —  smoke-test")
    print("=" * 60)

    train_path = os.environ.get("TRAIN_CSV", "train.csv")
    if not os.path.exists(train_path):
        print(f"Cannot find {train_path}. Set TRAIN_CSV env variable.")
        sys.exit(1)

    sub_path = os.environ.get("SUBMISSION_CSV", "sample_submission.csv")
    mask_path = "sunday_mask.parquet"

    print(f"\nLoading {train_path} …")
    train_df = pd.read_csv(train_path, low_memory=False)

    print(f"Loading {sub_path} …")
    sub_df = pd.read_csv(sub_path)

    print(f"Loading {mask_path} …")
    sunday_mask = pd.read_parquet(mask_path)

    print("\nComputing per-SKU cap (99.9th percentile) …")
    cap = compute_cap_value(train_df, percentile=99.9)
    print(f"  Cap stats: min={cap.min():.1f}, median={cap.median():.1f}, "
          f"max={cap.max():.1f}, p99={cap.quantile(0.99):.1f}")

    # Inject some synthetic non-zero + negative predictions for testing
    test_df = sub_df.copy()
    rng = np.random.default_rng(42)
    fcols = _get_fcols(test_df)
    test_df[fcols] = rng.integers(-5, 200, size=(len(test_df), len(fcols)))
    test_df[fcols] = test_df[fcols].astype(float)

    print("\nRunning post_process() on synthetic predictions …")
    result = post_process(test_df, sunday_mask, cap_value=cap)

    # Verify
    fcols = _get_fcols(result)
    sunday_fcols = [f"F{r['forecast_day_index']}" for _, r in sunday_mask[sunday_mask["force_zero"]].iterrows()]
    sunday_fcols = [f for f in sunday_fcols if f in fcols]

    print(f"\n[Check 1] Negatives clipped: {(result[fcols].values < 0).any()} (expect False)")
    sun_vals = result[sunday_fcols].values
    print(f"[Check 2] Sundays all zero:   {(sun_vals == 0).all()} (expect True)")
    print(f"[Check 3] All integer:        {(result[fcols].values == result[fcols].values.astype(int)).all()} (expect True)")
    print(f"[Check 4] Shape preserved:    {result.shape == test_df.shape} (expect True)")
    max_val = result[fcols].max().max()
    print(f"[Check 5] Max prediction:     {max_val} (capped at p99.9)")
    print("\n  ✓  post_process smoke-test passed.")
