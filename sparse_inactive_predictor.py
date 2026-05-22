"""
sparse_inactive_predictor.py
============================
Classify SKUs into INACTIVE / SPARSE / ACTIVE and generate baseline predictions
for the INACTIVE and SPARSE segments.

Classification rules
--------------------
  INACTIVE : 0 sales transactions in the last 90 days of training data
             (or SKU that never appears in sales at all)
             → predict = 0 for all 56 forecast days

  SPARSE   : 1–4 sales transactions in the last 90 days
             → predict = median of net daily qty over the LAST 90 DAYS
               (transaction-days only, i.e. days with at least one transaction)
               × dow_multiplier[day_of_week]
               Sunday multiplier is always 0 (applied via sunday_mask).

  ACTIVE   : >= 5 sales transactions in the last 90 days
             → left for the dedicated time-series model (not handled here)

Output
------
  predictions_sparse_inactive.parquet   — long format, all INACTIVE+SPARSE rows
                                          matching sample_submission.csv structure

  sku_segments.parquet                  — per-SKU segment label + key stats

DOW multipliers (computed from last-90-day transaction counts, normalised so
Mon–Sat average = 1.0, Sunday = 0.0).  Override with --dow-multipliers if B
provides a different set.

Usage
-----
    python sparse_inactive_predictor.py \\
        --train      /path/to/train.csv \\
        --submission /path/to/sample_submission.csv \\
        --mask       sunday_mask.parquet \\
        --out-pred   predictions_sparse_inactive.parquet \\
        --out-seg    sku_segments.parquet \\
        --sparse-thresh 4

    # Or import directly:
    from sparse_inactive_predictor import (
        classify_skus, predict_sparse_inactive, DOW_MULTIPLIERS
    )
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Last day of training data
TRAIN_END = "2025-09-05"

# 90-day lookback window start (inclusive)
WINDOW_90_START = "2025-06-07"   # = 2025-09-05 - 89 days

# Threshold for SPARSE:  1 <= tx_90d <= SPARSE_THRESH → SPARSE
#                        tx_90d > SPARSE_THRESH          → ACTIVE
SPARSE_THRESH_DEFAULT = 4

# Forecast windows
VALIDATION_START = datetime.date(2025, 9, 6)   # F1 of validation
EVALUATION_START = datetime.date(2025, 10, 4)  # F1 of evaluation
N_FORECAST_DAYS  = 28                           # each window

# Day-of-week multipliers (Mon=0 … Sun=6)
# Derived from last-90-day transaction count distribution on the actual data.
# Sunday is forced to 0 (business is closed; confirmed by <24 Sunday transactions
# across the entire ~5-year history).
# Weekday values normalised so Mon–Sat mean = 1.0.
DOW_MULTIPLIERS: dict[int, float] = {
    0: 1.0245,   # Monday
    1: 0.9417,   # Tuesday
    2: 1.0299,   # Wednesday
    3: 1.0644,   # Thursday
    4: 1.0275,   # Friday
    5: 0.9121,   # Saturday
    6: 0.0000,   # Sunday  ← closed
}

# Map dow-index → name (for readability)
DOW_NAMES = {
    0: "Monday", 1: "Tuesday", 2: "Wednesday",
    3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday",
}


# ─────────────────────────────────────────────────────────────────────────────
# 1.  SKU classification
# ─────────────────────────────────────────────────────────────────────────────

def classify_skus(
    train_df: pd.DataFrame,
    all_skus: pd.Index,
    sparse_thresh: int = SPARSE_THRESH_DEFAULT,
    window_start: str = WINDOW_90_START,
) -> pd.DataFrame:
    """
    Classify every SKU in all_skus into INACTIVE / SPARSE / ACTIVE.

    Parameters
    ----------
    train_df      : raw train.csv DataFrame
    all_skus      : pd.Index of all ItemCodes in the submission
    sparse_thresh : max tx count in 90d to be labelled SPARSE (default 4)
    window_start  : first date of the 90-day lookback window

    Returns
    -------
    pd.DataFrame with index=ItemCode and columns:
        tx_90d      int   — sales transactions count in window
        qty_90d     float — total net quantity sold in window  
        median_qty  float — median daily net qty on transaction-days
        mean_qty    float — mean daily net qty on transaction-days
        segment     str   — 'INACTIVE' | 'SPARSE' | 'ACTIVE'
    """
    df = train_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    ws = pd.Timestamp(window_start)

    # ── Sales only (exclude returns) for tx count ──────────────────────────
    sales = df[df["Quantity"] > 0]
    recent_sales = sales[sales["Date"] >= ws]

    tx_90d = (
        recent_sales.groupby("ItemCode")["Date"]
        .count()
        .reindex(all_skus, fill_value=0)
        .rename("tx_90d")
    )

    # ── Net daily qty over the window (sales + returns) ───────────────────
    recent_net = df[df["Date"] >= ws]
    net_daily = (
        recent_net.groupby(["ItemCode", "Date"])["Quantity"]
        .sum()
    )

    # For each SKU: stats on transaction-days with positive net qty
    # Vectorised to avoid pandas 3.0 groupby.apply stacking issue
    active_skus = tx_90d[tx_90d > 0].index
    if len(active_skus) > 0:
        pos_daily = net_daily[
            net_daily.index.get_level_values("ItemCode").isin(active_skus)
        ]
        pos_daily = pos_daily[pos_daily > 0]   # only days with net positive qty

        qty_sum    = pos_daily.groupby(level="ItemCode").sum().rename("qty_90d")
        median_qty = pos_daily.groupby(level="ItemCode").median().rename("median_qty")
        mean_qty   = pos_daily.groupby(level="ItemCode").mean().rename("mean_qty")
        stats = pd.concat([qty_sum, median_qty, mean_qty], axis=1)
    else:
        stats = pd.DataFrame(columns=["qty_90d", "median_qty", "mean_qty"])

    # Combine
    seg_df = pd.DataFrame({"tx_90d": tx_90d})
    seg_df = seg_df.join(stats, how="left").fillna({"qty_90d": 0.0, "median_qty": 0.0, "mean_qty": 0.0})

    # Assign segment
    conditions = [
        seg_df["tx_90d"] == 0,
        (seg_df["tx_90d"] >= 1) & (seg_df["tx_90d"] <= sparse_thresh),
        seg_df["tx_90d"] > sparse_thresh,
    ]
    seg_df["segment"] = np.select(conditions, ["INACTIVE", "SPARSE", "ACTIVE"], default="ACTIVE")

    return seg_df


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Build forecast date arrays
# ─────────────────────────────────────────────────────────────────────────────

def _build_forecast_dates(
    window_start: datetime.date,
    n_days: int = N_FORECAST_DAYS,
) -> list[datetime.date]:
    return [window_start + datetime.timedelta(days=i) for i in range(n_days)]


VAL_DATES  = _build_forecast_dates(VALIDATION_START)
EVAL_DATES = _build_forecast_dates(EVALUATION_START)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Generate predictions for INACTIVE + SPARSE
# ─────────────────────────────────────────────────────────────────────────────

def predict_sparse_inactive(
    seg_df: pd.DataFrame,
    submission_df: pd.DataFrame,
    dow_multipliers: dict[int, float] = DOW_MULTIPLIERS,
) -> pd.DataFrame:
    """
    Build prediction rows for INACTIVE and SPARSE SKUs.

    INACTIVE → all forecast values = 0
    SPARSE   → forecast value for day t = max(0, median_qty × dow_multipliers[dow(t)])

    Parameters
    ----------
    seg_df        : output of classify_skus()
    submission_df : sample_submission.csv DataFrame (used for id alignment)
    dow_multipliers : dict mapping day-of-week int (Mon=0…Sun=6) to multiplier

    Returns
    -------
    pd.DataFrame in submission format (id, F1..F28) for INACTIVE+SPARSE rows only.
    """
    # Build mapping from id → ItemCode + window
    sub = submission_df.copy()
    sub["ItemCode"] = sub["id"].str.rsplit("_", n=1).str[0]
    sub["window"]   = sub["id"].str.rsplit("_", n=1).str[1]

    # Merge with segment info
    sub = sub.merge(
        seg_df[["segment", "median_qty"]],
        left_on="ItemCode",
        right_index=True,
        how="left",
    )
    sub["segment"]    = sub["segment"].fillna("ACTIVE")
    sub["median_qty"] = sub["median_qty"].fillna(0.0)

    # Filter to INACTIVE + SPARSE only
    mask = sub["segment"].isin(["INACTIVE", "SPARSE"])
    sub  = sub[mask].copy()

    if len(sub) == 0:
        print("[predict_sparse_inactive] No INACTIVE or SPARSE rows found.")
        return pd.DataFrame(columns=["id"] + [f"F{i}" for i in range(1, 29)])

    print(f"[predict_sparse_inactive] Building predictions for "
          f"{len(sub):,} rows ({sub['segment'].value_counts().to_dict()})")

    fcols = [f"F{i}" for i in range(1, N_FORECAST_DAYS + 1)]

    # Pre-compute DoW multipliers for each forecast window
    val_mults  = np.array([dow_multipliers[d.weekday()] for d in VAL_DATES])
    eval_mults = np.array([dow_multipliers[d.weekday()] for d in EVAL_DATES])

    # Initialise output matrix
    pred_values = np.zeros((len(sub), N_FORECAST_DAYS), dtype=np.float64)

    # SPARSE rows: apply multipliers
    sparse_mask = (sub["segment"] == "SPARSE").values
    if sparse_mask.any():
        medians = sub["median_qty"].values                # (n_rows,)

        val_rows  = (sub["window"].values == "validation") & sparse_mask
        eval_rows = (sub["window"].values == "evaluation") & sparse_mask

        if val_rows.any():
            # outer product: each SKU's median × each day's multiplier
            pred_values[val_rows] = np.outer(medians[val_rows], val_mults)

        if eval_rows.any():
            pred_values[eval_rows] = np.outer(medians[eval_rows], eval_mults)

    # INACTIVE rows stay as 0 (already initialised)
    # Clip to non-negative (medians should already be positive, but paranoid check)
    pred_values = np.clip(pred_values, 0.0, None)

    # Assemble output DataFrame
    result = pd.DataFrame(pred_values, columns=fcols, index=sub.index)
    result.insert(0, "id", sub["id"].values)

    print(f"  Non-zero predictions: "
          f"{(result[fcols].values > 0).sum():,} / {result[fcols].size:,} cells")

    return result.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run(
    train_path: str,
    submission_path: str,
    mask_path: str,
    out_pred_path: str = "predictions_sparse_inactive.parquet",
    out_seg_path: str  = "sku_segments.parquet",
    sparse_thresh: int = SPARSE_THRESH_DEFAULT,
) -> pd.DataFrame:
    """
    Full pipeline: load data → classify → predict → post-process → save.

    Returns the post-processed prediction DataFrame.
    """
    from post_process import post_process, compute_cap_value

    print("=" * 60)
    print("  sparse_inactive_predictor.py")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────────────────
    print(f"\n[1/5]  Loading train data from {train_path} …")
    train_df = pd.read_csv(train_path, low_memory=False)
    train_df["Date"] = pd.to_datetime(train_df["Date"])
    print(f"  {len(train_df):,} rows, {train_df['ItemCode'].nunique():,} SKUs")

    print(f"\n[2/5]  Loading submission template …")
    sub_df = pd.read_csv(submission_path)
    all_skus = pd.Index(sub_df["id"].str.rsplit("_", n=1).str[0].unique())
    print(f"  {len(sub_df):,} rows, {len(all_skus):,} unique SKUs")

    print(f"\n[3/5]  Classifying SKUs (sparse_thresh={sparse_thresh}) …")
    seg_df = classify_skus(train_df, all_skus, sparse_thresh=sparse_thresh)

    counts = seg_df["segment"].value_counts()
    for seg in ["INACTIVE", "SPARSE", "ACTIVE"]:
        n = counts.get(seg, 0)
        print(f"  {seg:8s}: {n:6,} SKUs ({n / len(seg_df) * 100:.1f}%)")

    # Save segment info
    seg_df.to_parquet(out_seg_path)
    print(f"  Saved: {out_seg_path}")

    # ── Generate raw predictions ───────────────────────────────────────────
    print(f"\n[4/5]  Generating INACTIVE + SPARSE predictions …")
    raw_pred = predict_sparse_inactive(seg_df, sub_df)

    # ── Post-process ───────────────────────────────────────────────────────
    print(f"\n[5/5]  Post-processing (sunday mask + clip + cap + round) …")
    sunday_mask = pd.read_parquet(mask_path)

    # Cap: use per-SKU 99.9th percentile on full training history
    cap = compute_cap_value(train_df, percentile=99.9)

    from post_process import post_process
    processed = post_process(raw_pred, sunday_mask, cap_value=cap, do_round=True)

    # Save
    processed.to_parquet(out_pred_path, index=False)
    print(f"  Saved: {out_pred_path}")

    # ── Summary ───────────────────────────────────────────────────────────
    fcols = [f"F{i}" for i in range(1, N_FORECAST_DAYS + 1)]
    n_nonzero = (processed[fcols].values > 0).sum()
    n_total   = processed[fcols].size
    print(f"\n{'─'*60}")
    print(f"  Output rows    : {len(processed):,}")
    print(f"  Non-zero cells : {n_nonzero:,} / {n_total:,} ({n_nonzero/n_total*100:.2f}%)")
    print(f"  Max prediction : {processed[fcols].max().max()}")
    print(f"  Mean (non-zero): {processed[fcols].values[processed[fcols].values > 0].mean():.3f}")
    print(f"{'─'*60}")
    print("  Done.")

    return processed


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate INACTIVE + SPARSE predictions")
    parser.add_argument("--train",        default="train.csv",                         help="Path to train.csv")
    parser.add_argument("--submission",   default="sample_submission.csv",             help="Path to sample_submission.csv")
    parser.add_argument("--mask",         default="sunday_mask.parquet",               help="Path to sunday_mask.parquet")
    parser.add_argument("--out-pred",     default="predictions_sparse_inactive.parquet", help="Output predictions parquet")
    parser.add_argument("--out-seg",      default="sku_segments.parquet",              help="Output segments parquet")
    parser.add_argument("--sparse-thresh", type=int, default=SPARSE_THRESH_DEFAULT,   help="Max tx_90d for SPARSE label")
    args = parser.parse_args()

    run(
        train_path     = args.train,
        submission_path= args.submission,
        mask_path      = args.mask,
        out_pred_path  = args.out_pred,
        out_seg_path   = args.out_seg,
        sparse_thresh  = args.sparse_thresh,
    )
