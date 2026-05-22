"""
baseline_wma.py
================
Weighted Moving Average (WMA) cho nhóm MODERATE / SPARSE SKU.
Safety net: chạy độc lập, không cần feature engineering, không cần LightGBM.

Luồng sử dụng:
    1. Truyền Series lịch sử net_qty của 1 SKU vào baseline_wma()
    2. Nhận lại mảng giá trị dự báo (28 hoặc 56 ngày)
    3. Gọi batch_wma_forecast() để xử lý toàn bộ danh sách SKU cùng lúc
    4. Gọi build_submission_block() để đóng gói thành format nộp bài/đánh giá
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ════════════════════════════════════════════════════════════
# 1. TRỌNG SỐ NGÀY TRONG TUẦN (DoW Multipliers)
# ════════════════════════════════════════════════════════════

DEFAULT_DOW_MULTIPLIERS = {
    0: 1.15,   # Thứ 2 — đầu tuần, đại lý hay đặt hàng bù
    1: 1.05,   # Thứ 3
    2: 1.00,   # Thứ 4 — baseline
    3: 1.00,   # Thứ 5
    4: 1.05,   # Thứ 6
    5: 0.75,   # Thứ 7 — nửa ngày làm việc
    6: 0.00,   # Chủ Nhật — không bán
}


# ════════════════════════════════════════════════════════════
# 2. DAILY RATE — Tốc độ bán hàng tổng hợp đa cửa sổ
# ════════════════════════════════════════════════════════════

def compute_daily_rate(
    values          : np.ndarray,
    windows         : tuple[int, ...] = (28, 56, 90),
    sparse_penalty  : float           = 0.5,
    sparse_threshold: float           = 0.15,
) -> float:
    """Tính Daily Rate tổng hợp từ nhiều khung cửa sổ thời gian."""
    weight_map = {28: 0.5, 56: 0.3, 90: 0.2}
    default_w  = 1.0 / len(windows)

    n        = len(values)
    rates    = []
    weights  = []

    for w in windows:
        if n < w:
            continue

        window_vals = values[-w:]
        nonzero     = window_vals[window_vals > 0]
        active_days = len(nonzero)

        if active_days == 0:
            rate = 0.0
        else:
            rate = float(window_vals.sum()) / w

        # Sparse penalty
        activity_ratio = active_days / w
        if activity_ratio < sparse_threshold and rate > 0:
            rate *= sparse_penalty

        rates.append(rate)
        weights.append(weight_map.get(w, default_w))

    if not rates:
        nonzero_all = values[values > 0]
        return float(nonzero_all.mean()) if len(nonzero_all) > 0 else 0.0

    weights = np.array(weights)
    weights = weights / weights.sum()

    return float(np.dot(weights, rates))


# ════════════════════════════════════════════════════════════
# 3. HÀM CHÍNH
# ════════════════════════════════════════════════════════════

def baseline_wma(
    series          : pd.Series | np.ndarray,
    window          : int   = 28,
    horizon         : int   = 28,  # Sửa mặc định thành 28 cho Pseudo-test
    dow_multipliers : dict  = None,
    future_dates    : pd.DatetimeIndex | None = None,
    decay           : float = 1.0,
    use_daily_rate  : bool  = True,
) -> np.ndarray:
    """Dự báo chuỗi thời gian bằng WMA có điều chỉnh DoW."""
    if dow_multipliers is None:
        dow_multipliers = DEFAULT_DOW_MULTIPLIERS

    if future_dates is None:
        # Hỗ trợ mặc định an toàn nếu không truyền future_dates
        future_dates = pd.date_range("2025-08-09", periods=horizon, freq="D")

    values = np.array(series, dtype=float)
    
    raw_weights = np.array([decay ** i for i in range(window - 1, -1, -1)])
    raw_weights = raw_weights / raw_weights.sum()

    if use_daily_rate:
        static_daily_rate = compute_daily_rate(values)

    forecasts = []
    history   = list(values)

    for i, date in enumerate(future_dates):
        dow = date.dayofweek

        if dow == 6:
            forecasts.append(0.0)
            history.append(0.0)
            continue

        if use_daily_rate:
            base = static_daily_rate
        else:
            window_vals = _get_non_sunday_window(history, window)
            if len(window_vals) == 0:
                base = 0.0
            elif len(window_vals) < window:
                base = float(np.mean(window_vals))
            else:
                base = float(np.dot(raw_weights, window_vals))

        multiplier = dow_multipliers.get(dow, 1.0)
        pred       = max(0.0, base * multiplier)

        forecasts.append(pred)
        history.append(pred)

    return np.array(forecasts)


def _get_non_sunday_window(history: list, window: int) -> list:
    """Lấy `window` giá trị gần nhất trong history, bỏ qua ngày CN."""
    tail = history[-(window * 2):]
    non_zero_or_valid = [v for v in tail if v > 0]
    return non_zero_or_valid[-window:]


# ════════════════════════════════════════════════════════════
# 3. BATCH FORECAST
# ════════════════════════════════════════════════════════════

def batch_wma_forecast(
    daily_df        : pd.DataFrame,
    sku_list        : list[str],
    start_date      : str   = "2025-08-09",  # Tham số hóa cho Validation/Test
    horizon         : int   = 28,            # 28 cho validation, 56 cho test
    date_col        : str   = "Date",
    qty_col         : str   = "net_qty",
    sku_col         : str   = "ItemCode",
    window          : int   = 28,
    decay           : float = 0.95,
    dow_multipliers : dict  = None,
) -> pd.DataFrame:
    """Chạy baseline_wma cho toàn bộ danh sách SKU."""
    future_dates = pd.date_range(start_date, periods=horizon, freq="D")

    daily_df = daily_df.copy()
    daily_df[date_col] = pd.to_datetime(daily_df[date_col])

    df_filtered = (
        daily_df[daily_df[sku_col].isin(sku_list)]
        .sort_values([sku_col, date_col])
    )

    results = []
    skus_done = 0

    for sku, grp in df_filtered.groupby(sku_col, observed=True):
        series = grp.set_index(date_col)[qty_col]

        preds = baseline_wma(
            series          = series.values,
            window          = window,
            horizon         = horizon,
            dow_multipliers = dow_multipliers,
            future_dates    = future_dates,
            decay           = decay,
        )

        for date, pred in zip(future_dates, preds):
            results.append({
                sku_col       : sku,
                "Date"        : date,
                "forecast_qty": round(pred, 4),
            })

        skus_done += 1
        if skus_done % 1000 == 0:
            print(f"   ... {skus_done:,}/{len(sku_list):,} SKU done")

    df_out = pd.DataFrame(results)
    print(f"✅ WMA forecast xong: {df_out[sku_col].nunique():,} SKU × {horizon} ngày = {len(df_out):,} dòng")
    return df_out


# ════════════════════════════════════════════════════════════
# 4. ĐÓNG GÓI OUTPUT
# ════════════════════════════════════════════════════════════

def build_submission_block(
    df_forecast: pd.DataFrame,
    sku_col    : str = "ItemCode",
    date_col   : str = "Date",
    qty_col    : str = "forecast_qty",
) -> pd.DataFrame:
    """Đóng gói output. Tự động nội suy ngày bắt đầu để chống lỗi."""
    df = df_forecast.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Lấy min date từ chính DataFrame để tái sử dụng cho cả Val và Test
    start_ts = df[date_col].min() 
    df["day_num"] = (df[date_col] - start_ts).dt.days + 1

    df["id"] = np.where(
        df["day_num"] <= 28,
        df[sku_col] + "_validation",
        df[sku_col] + "_evaluation",
    )
    df["f_index"] = np.where(
        df["day_num"] <= 28,
        df["day_num"],
        df["day_num"] - 28,
    )
    df["f_col"] = "F" + df["f_index"].astype(str)

    pivot = (
        df.pivot_table(index="id", columns="f_col",
                       values=qty_col, aggfunc="sum")
        .reindex(columns=[f"F{i}" for i in range(1, 29)])
        .clip(lower=0)
        .reset_index()
    )

    print(f"✅ WMA submission block: {pivot.shape}")
    return pivot


if __name__ == "__main__":
    print("baseline_wma.py — smoke test\n")

    np.random.seed(42)
    fake_history = np.maximum(0, np.random.poisson(lam=5, size=365).astype(float))
    for i in range(len(fake_history)):
        if i % 7 == 6:
            fake_history[i] = 0.0

    future = pd.date_range("2025-08-09", periods=28, freq="D")

    preds_dr = baseline_wma(fake_history, window=28, decay=0.95,
                            future_dates=future, use_daily_rate=True)
    preds_wma = baseline_wma(fake_history, window=28, decay=0.95,
                             future_dates=future, use_daily_rate=False)

    print(f"[Daily Rate mode] mean={preds_dr.mean():.2f}")
    print(f"[WMA mode       ] mean={preds_wma.mean():.2f}")
    print("\n✅ baseline_wma.py hoạt động hoàn hảo.")
