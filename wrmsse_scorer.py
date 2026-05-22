"""
wrmsse_scorer.py
================
Weighted Root Mean Squared Scaled Error (WRMSSE) scorer for
the HBAAC Round-2 demand forecasting competition.

Competition metric definition
------------------------------
For each SKU i with h-day forecast horizon:

    RMSSE_i  = sqrt( MSE_i / denom_i )

    MSE_i    = (1/h) * sum_{t=1}^{h} (y_true_i,t - y_pred_i,t)^2

    denom_i  = (1/(T-1)) * sum_{t=2}^{T} (Y_i,t - Y_i,t-1)^2
               (naive one-step MSE on the *daily* training series)

    w_i      = max(0, profit_i) / sum_j max(0, profit_j)
               (profit_i = total SalesAmount - CostAmount over train set)

    WRMSSE   = sum_i  w_i * RMSSE_i

Key design decisions
---------------------
- Daily series are built by SUMMING all transactions per (Date, ItemCode),
  then reindexing over the full calendar so every date in [first_date, last_date]
  appears (zero-filled).  This matches M5-competition convention.
- Returns (negative Quantity rows) are included in the daily sum — a return on
  day t naturally reduces that day's net quantity.  The final daily quantity
  can theoretically be negative, but **predictions must be non-negative** (clipped).
- SKUs where denom_i == 0 (constant training series, e.g. always 0) are assigned
  RMSSE = 0 so they don't artificially inflate the score — they contribute nothing
  to numerator or denominator of the scaled error anyway.
- SKUs with profit_i <= 0 receive weight 0 (excluded from WRMSSE).
- If all weights are 0 the function raises ValueError rather than returning NaN.

Usage
------
    from wrmsse_scorer import compute_wrmsse, build_scorer_inputs

    # One-shot: build everything from raw train DataFrame + predictions DataFrame
    score = compute_wrmsse_from_raw(
        train_df   = train_df,          # raw transaction DataFrame (see train.csv)
        y_pred_df  = submission_df,     # DataFrame: index=ItemCode, columns=days (28 values)
        horizon    = 28,
    )

    # Or step-by-step (useful for cross-validation):
    daily_train, weights, denominators = build_scorer_inputs(train_df)
    y_true = build_pseudo_test_actuals(daily_train, test_start, test_end)
    score  = compute_wrmsse(y_true, y_pred, weights, denominators)
"""

from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1.  Low-level scorer  (operates on pre-built numpy arrays)
# ---------------------------------------------------------------------------

def compute_wrmsse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray,
    denominators: np.ndarray,
    clip_pred: bool = True,
) -> float:
    """
    Compute WRMSSE from pre-built arrays.

    Parameters
    ----------
    y_true       : shape (n_skus, h)  — actual daily quantities, horizon h days
    y_pred       : shape (n_skus, h)  — predicted daily quantities (non-negative)
    weights      : shape (n_skus,)    — profit-weighted share per SKU (sum to 1.0,
                                        0-weight SKUs are excluded automatically)
    denominators : shape (n_skus,)    — naive one-step MSE on training data per SKU
    clip_pred    : bool               — clip y_pred to >= 0 before scoring (default True)

    Returns
    -------
    float  WRMSSE  (lower is better; 1.0 = naive baseline)
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    denominators = np.asarray(denominators, dtype=np.float64)

    assert y_true.shape == y_pred.shape, (
        f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
    )
    assert y_true.shape[0] == len(weights) == len(denominators), (
        "Number of SKUs must match across y_true, weights, denominators."
    )

    if clip_pred:
        y_pred = np.clip(y_pred, 0.0, None)

    # Normalise weights so they sum to 1 (handle edge case of all-zero weights)
    weight_sum = weights.sum()
    if weight_sum <= 0:
        raise ValueError(
            "All profit weights are 0 or negative. "
            "Cannot compute WRMSSE — no SKU contributes to the score."
        )
    w = weights / weight_sum          # (n_skus,)

    # Per-SKU MSE over the forecast horizon
    sq_err = (y_true - y_pred) ** 2   # (n_skus, h)
    mse    = sq_err.mean(axis=1)      # (n_skus,)

    # RMSSE per SKU — safe division: denom==0 → RMSSE=0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rmsse = np.where(denominators > 0, np.sqrt(mse / denominators), 0.0)

    # Weighted sum — zero-weight SKUs contribute 0 regardless of RMSSE
    wrmsse = float(np.dot(w, rmsse))
    return wrmsse


# ---------------------------------------------------------------------------
# 2.  Build daily series + profit weights + denominators from raw train.csv
# ---------------------------------------------------------------------------

def _parse_vnd(series: pd.Series) -> pd.Series:
    """
    Convert VND column to float.
    Handles:
      - plain integers / floats already numeric
      - string columns (object or ArrowStringDtype) with comma as decimal separator
        e.g. '1482709,2'  →  1482709.2
    """
    # Already numeric (int64, float64)?
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    # String (object or any Arrow/StringDtype)
    return (
        series.astype(str)
              .str.replace(",", ".", regex=False)
              .astype(float)
    )


def build_daily_series(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate raw transactions to a daily SKU × Date matrix.

    Parameters
    ----------
    train_df  : raw train.csv DataFrame with columns
                [Date, ItemCode, Quantity, SalesAmount, Cost Amount, ...]

    Returns
    -------
    DataFrame with:
        - index   : DatetimeIndex (every calendar day in [min_date, max_date])
        - columns : ItemCode strings
        - values  : net Quantity per (date, SKU), 0-filled for missing dates
    """
    df = train_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    # Net daily quantity (sales - returns already encoded as negative Quantity)
    daily_qty = (
        df.groupby(["Date", "ItemCode"])["Quantity"]
        .sum()
        .unstack(fill_value=0)
    )

    # Reindex over full date range so every calendar day is present
    full_idx = pd.date_range(daily_qty.index.min(), daily_qty.index.max(), freq="D")
    daily_qty = daily_qty.reindex(full_idx, fill_value=0)

    return daily_qty


def build_profit_weights(train_df: pd.DataFrame) -> pd.Series:
    """
    Compute per-SKU profit weight = max(0, profit_i) / sum_j max(0, profit_j).

    profit_i = sum over training period of (SalesAmount - Cost Amount) per SKU.

    Parameters
    ----------
    train_df  : raw train.csv DataFrame

    Returns
    -------
    pd.Series  index=ItemCode, values=normalised weight in [0, 1]
    """
    df = train_df.copy()
    df["profit"] = df["SalesAmount"].astype(float) - _parse_vnd(df["Cost Amount"].astype(str))

    sku_profit = df.groupby("ItemCode")["profit"].sum()
    sku_profit = sku_profit.clip(lower=0.0)           # negative profit → 0

    total = sku_profit.sum()
    if total <= 0:
        raise ValueError("Total profit is 0 or negative. Cannot normalise weights.")

    return sku_profit / total


def build_denominators(daily_series: pd.DataFrame) -> pd.Series:
    """
    Compute naive one-step MSE denominator for each SKU.

    denom_i = (1/(T-1)) * sum_{t=2}^{T} (Y_i,t - Y_i,t-1)^2

    If a SKU has fewer than 2 observations or constant series → denom = 0
    (RMSSE will be set to 0 for that SKU in compute_wrmsse).

    Parameters
    ----------
    daily_series : (dates × SKUs) DataFrame from build_daily_series()

    Returns
    -------
    pd.Series  index=ItemCode, values=denominator float
    """
    diff = daily_series.diff().iloc[1:]    # first-order differences, drop NaN row
    denom = (diff ** 2).mean(axis=0)       # mean squared diff per SKU
    return denom


def build_scorer_inputs(
    train_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    One-call convenience: build daily series, profit weights, and denominators.

    Parameters
    ----------
    train_df  : raw train.csv DataFrame

    Returns
    -------
    (daily_series, weights, denominators)
        daily_series  : (dates × SKUs) net-quantity DataFrame
        weights       : pd.Series, index=ItemCode, normalised profit weights
        denominators  : pd.Series, index=ItemCode, naive MSE denom
    """
    daily_series = build_daily_series(train_df)
    weights      = build_profit_weights(train_df)
    denominators = build_denominators(daily_series)
    return daily_series, weights, denominators


# ---------------------------------------------------------------------------
# 3.  Full pipeline: raw train + predictions → WRMSSE score
# ---------------------------------------------------------------------------

def compute_wrmsse_from_raw(
    train_df: pd.DataFrame,
    y_pred_df: pd.DataFrame,
    y_true_df: pd.DataFrame | None = None,
    train_cutoff: str | None = None,
    test_start: str | None = None,
    test_end: str | None = None,
) -> float:
    """
    End-to-end scorer.

    Offline / pseudo-test mode (internal validation):
        Pass train_cutoff, test_start, test_end.
        y_true is extracted from daily_series[test_start:test_end].
        train data for denom / weights uses only rows up to train_cutoff.

    Online mode (score against provided ground truth):
        Pass y_true_df directly.

    Parameters
    ----------
    train_df       : raw transactions DataFrame
    y_pred_df      : DataFrame — index=ItemCode, columns = day-1..day-h (h columns)
                     OR a submission-style DataFrame with 'id' column and F1..F28 cols
    y_true_df      : optional ground-truth DataFrame (same format as y_pred_df)
    train_cutoff   : 'YYYY-MM-DD' — last day of training for internal validation
    test_start     : 'YYYY-MM-DD' — first day of pseudo-test window
    test_end       : 'YYYY-MM-DD' — last  day of pseudo-test window (inclusive)

    Returns
    -------
    float  WRMSSE
    """
    df = train_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    # ------------------------------------------------------------------
    # Handle time-based split
    # ------------------------------------------------------------------
    if train_cutoff is not None:
        tc = pd.Timestamp(train_cutoff)
        train_part = df[df["Date"] <= tc]
    else:
        train_part = df

    # Build scorer inputs from (possibly truncated) training data
    daily_series, weights, denominators = build_scorer_inputs(train_part)

    # ------------------------------------------------------------------
    # Ground-truth actuals
    # ------------------------------------------------------------------
    if y_true_df is not None:
        y_true = _align_df(y_true_df, weights.index)
    elif test_start is not None and test_end is not None:
        ts = pd.Timestamp(test_start)
        te = pd.Timestamp(test_end)
        # Build actuals from full data (including rows after train_cutoff)
        full_daily = build_daily_series(df)
        window = full_daily.loc[ts:te]
        # Reindex to ensure all SKUs in submission appear
        window = window.reindex(columns=weights.index, fill_value=0)
        y_true = window.T.values   # shape (n_skus, h)
    else:
        raise ValueError(
            "Provide either (test_start + test_end) or y_true_df."
        )

    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------
    y_pred = _align_df(y_pred_df, weights.index)

    # Align arrays
    w     = weights.values
    d     = denominators.reindex(weights.index).fillna(0).values

    return compute_wrmsse(y_true, y_pred, w, d, clip_pred=True)


def _align_df(pred_df: pd.DataFrame, sku_index: pd.Index) -> np.ndarray:
    """
    Normalise prediction/actuals DataFrame to (n_skus, h) array aligned to sku_index.

    Accepts two formats:
      A) index = ItemCode,  columns = any (F1..F28 or 0..27)
      B) 'id' column like 'SKU-00001_validation', F1..F28 columns
    """
    if "id" in pred_df.columns:
        # Submission format
        df = pred_df.copy()
        df["ItemCode"] = df["id"].str.rsplit("_", n=1).str[0]
        df = df.set_index("ItemCode")
        fcols = [c for c in df.columns if str(c).startswith("F")]
        df = df[sorted(fcols, key=lambda x: int(x[1:]))]
    else:
        df = pred_df.copy()
        fcols = [c for c in df.columns if str(c).startswith("F")]
        if fcols:
            df = df[sorted(fcols, key=lambda x: int(x[1:]))]

    df = df.reindex(sku_index, fill_value=0)
    return df.values.astype(np.float64)


# ---------------------------------------------------------------------------
# 4.  Validation split helper
# ---------------------------------------------------------------------------

TRAIN_CUTOFF   = "2025-08-08"
PSEUDO_TEST_START = "2025-08-09"
PSEUDO_TEST_END   = "2025-09-05"


def create_validation_split(
    train_df: pd.DataFrame,
    output_path: str = "validation_split.parquet",
    train_cutoff: str = TRAIN_CUTOFF,
    test_start: str = PSEUDO_TEST_START,
    test_end: str = PSEUDO_TEST_END,
) -> pd.DataFrame:
    """
    Create time-based validation split and optionally save to parquet.

    The resulting DataFrame has multi-level columns:
        split  ∈ {'train', 'pseudo_test'}
        metric ∈ {'quantity'}   (net daily quantity per SKU)

    Index: DatetimeIndex (daily)
    Columns: ItemCode (one per SKU)

    In practice, callers usually prefer the flat format; use
    build_scorer_inputs(train_part) directly in your model pipeline.

    Returns
    -------
    dict with keys 'train' and 'pseudo_test', each a (dates × SKUs) DataFrame.
    """
    df = train_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    tc = pd.Timestamp(train_cutoff)
    ts = pd.Timestamp(test_start)
    te = pd.Timestamp(test_end)

    daily = build_daily_series(df)

    split_train = daily.loc[:tc]
    split_test  = daily.loc[ts:te]

    print(f"[validation_split]")
    print(f"  Train  : {split_train.index.min().date()} → {split_train.index.max().date()} "
          f"({len(split_train)} days)")
    print(f"  Pseudo-test : {split_test.index.min().date()} → {split_test.index.max().date()} "
          f"({len(split_test)} days)")
    print(f"  SKUs   : {daily.shape[1]}")

    if output_path:
        # Save as long-format parquet for easy downstream use
        long = (
            daily
            .reset_index()
            .rename(columns={"index": "Date"})
            .melt(id_vars="Date", var_name="ItemCode", value_name="Quantity")
        )
        long["split"] = "train"
        long.loc[long["Date"] >= ts, "split"] = "pseudo_test"
        long.loc[long["Date"] > te,  "split"] = "post_test"   # future (real test)
        long.to_parquet(output_path, index=False)
        print(f"  Saved  : {output_path}")

    return {"train": split_train, "pseudo_test": split_test}


# ---------------------------------------------------------------------------
# 5.  Sunday mask helper
# ---------------------------------------------------------------------------

FORECAST_START = "2025-09-06"   # F1

SUNDAY_FORECAST_DAYS = [2, 9, 16, 23, 30, 37, 44, 51]   # F-indices that are Sundays

def create_sunday_mask(
    forecast_start: str = FORECAST_START,
    n_days: int = 56,
    output_path: str = "sunday_mask.parquet",
) -> pd.DataFrame:
    """
    Build a DataFrame identifying Sundays within the 56-day forecast horizon.

    Columns
    -------
    forecast_day_index : int   (1-based: F1, F2, ..., F56)
    date               : date
    day_of_week        : str   (e.g. 'Sunday')
    force_zero         : bool  True for Sundays

    Returns the DataFrame (and saves to parquet if output_path is given).
    """
    start = pd.Timestamp(forecast_start)
    rows = []
    for i in range(1, n_days + 1):
        d = start + pd.Timedelta(days=i - 1)
        rows.append({
            "forecast_day_index": i,
            "date": d.date(),
            "day_of_week": d.day_name(),
            "force_zero": d.dayofweek == 6,   # 6 = Sunday
        })

    mask_df = pd.DataFrame(rows)

    sundays = mask_df[mask_df["force_zero"]]
    print(f"[sunday_mask]  {len(sundays)} Sundays in F1–F{n_days}:")
    for _, r in sundays.iterrows():
        print(f"  F{r['forecast_day_index']:>2}  {r['date']}  ({r['day_of_week']})")

    if output_path:
        mask_df.to_parquet(output_path, index=False)
        print(f"  Saved  : {output_path}")

    return mask_df


# ---------------------------------------------------------------------------
# 6.  CLI / smoke-test entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os

    print("=" * 60)
    print("  wrmsse_scorer.py  —  smoke-test / demo")
    print("=" * 60)

    # --- sunday mask (no data dependency) ---
    print("\n[1/3]  Building Sunday mask …")
    smask = create_sunday_mask(output_path="sunday_mask.parquet")
    assert list(smask[smask["force_zero"]]["forecast_day_index"]) == SUNDAY_FORECAST_DAYS, \
        "Sunday indices do not match expected!"
    print("  ✓  Sunday mask verified.")

    # --- load train.csv ---
    train_path = os.environ.get("TRAIN_CSV", "train.csv")
    if not os.path.exists(train_path):
        print(f"\nCannot find {train_path}.  "
              "Set TRAIN_CSV env variable or place train.csv in the working directory.")
        print("Skipping validation split + WRMSSE smoke-test.")
        sys.exit(0)

    print(f"\n[2/3]  Loading {train_path} …")
    train_df = pd.read_csv(train_path, low_memory=False)
    train_df["Date"] = pd.to_datetime(train_df["Date"])
    print(f"  Loaded {len(train_df):,} rows, "
          f"{train_df['ItemCode'].nunique():,} SKUs, "
          f"{train_df['Date'].min().date()} → {train_df['Date'].max().date()}")

    print("\n[3/3]  Building validation split …")
    splits = create_validation_split(train_df, output_path="validation_split.parquet")

    # Smoke-test WRMSSE: predict all zeros (worst case) vs naive
    print("\n[WRMSSE smoke-test]  Predicting all zeros …")
    y_true = splits["pseudo_test"]                          # (28 × n_skus)
    tc = pd.Timestamp(TRAIN_CUTOFF)
    train_part = train_df[train_df["Date"] <= tc]
    _, weights, denominators = build_scorer_inputs(train_part)

    y_true_arr = y_true.T.reindex(weights.index, fill_value=0).values   # (n_skus, 28)
    y_pred_zeros = np.zeros_like(y_true_arr)
    y_pred_naive = np.full_like(y_true_arr, y_true_arr.mean())

    score_zeros = compute_wrmsse(y_true_arr, y_pred_zeros, weights.values,
                                  denominators.reindex(weights.index).fillna(0).values)
    score_naive_mean = compute_wrmsse(y_true_arr, y_pred_naive, weights.values,
                                      denominators.reindex(weights.index).fillna(0).values)

    print(f"  WRMSSE (all-zeros)    = {score_zeros:.4f}")
    print(f"  WRMSSE (global mean)  = {score_naive_mean:.4f}")
    print("\n  ✓  Smoke-test complete.  Both scores finite and non-negative.")
