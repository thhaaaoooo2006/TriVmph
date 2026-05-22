"""
Vietnamese Auto Parts Demand Forecasting
Strategy:
  - Aggregate daily net sales (clip returns at 0 per SKU per day)
  - Segment SKUs by activity level
  - Inactive SKUs (no sales in last 90 days) → predict 0
  - Sparse SKUs (< 5 active days in last 90 days) → small constant based on recent history
  - Active SKUs → weighted moving average + DoW seasonal adjustment
  - Top high-volume SKUs → exponential smoothing with trend
  - Zero out Sundays (virtually no sales historically)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
train = pd.read_csv('train.csv', low_memory=False)
sub = pd.read_csv('sample_submission.csv')
train['Date'] = pd.to_datetime(train['Date'])

# ==============================================================================
# 1. Build daily SKU-level net sales
# ==============================================================================
print("Building daily panel...")
daily = (
    train
    .groupby(['Date', 'ItemCode'])['Quantity']
    .sum()
    .reset_index()
    .rename(columns={'Date': 'date', 'ItemCode': 'sku', 'Quantity': 'qty'})
)
daily['qty'] = daily['qty'].clip(lower=0)  # net returns: negative daily net → 0

# All SKUs from submission
all_skus = np.sort(sub['id'].str.rsplit('_', n=1).str[0].unique())
print(f"Total SKUs: {len(all_skus)}")

# ==============================================================================
# 2. Compute DoW multipliers from historical data (exclude Sundays)
# ==============================================================================
daily_agg = train.groupby('Date')['Quantity'].sum().reset_index()
daily_agg.columns = ['date', 'total_qty']
daily_agg['dow'] = pd.to_datetime(daily_agg['date']).dt.dayofweek

# Use only Mon-Sat (exclude the 4 Sundays which are outliers)
working_days = daily_agg[daily_agg['dow'] != 6]
global_mean = working_days['total_qty'].mean()
dow_means = working_days.groupby('dow')['total_qty'].mean()
dow_mult = (dow_means / global_mean).reindex(range(7), fill_value=0.0)
dow_mult[6] = 0.0  # Sunday → 0
print("DoW multipliers:")
for d, m in dow_mult.items():
    days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    print(f"  {days[d]}: {m:.3f}")

# ==============================================================================
# 3. Define forecast horizon
# ==============================================================================
last_train_date = train['Date'].max()
forecast_start = pd.Timestamp('2025-09-06')  # F1
forecast_end = pd.Timestamp('2025-10-31')    # F56
forecast_dates = pd.date_range(forecast_start, forecast_end, freq='D')
assert len(forecast_dates) == 56, f"Expected 56 forecast days, got {len(forecast_dates)}"
print(f"\nForecast horizon: {forecast_start.date()} to {forecast_end.date()} ({len(forecast_dates)} days)")

# ==============================================================================
# 4. For each SKU, compute forecasts
# ==============================================================================
# Pivot the sparse daily data into a dict for fast lookup
sku_history = {}
for sku, grp in daily.groupby('sku'):
    sku_history[sku] = grp.set_index('date')['qty']

# Reference windows for activity detection and average computation
ref_90 = last_train_date - pd.Timedelta(days=89)   # last 90 days
ref_56 = last_train_date - pd.Timedelta(days=55)   # last 56 days
ref_28 = last_train_date - pd.Timedelta(days=27)   # last 28 days
ref_14 = last_train_date - pd.Timedelta(days=13)   # last 14 days

# Working day count in each window (Mon-Sat only, no Sun)
def count_working_days(start, end):
    dates = pd.date_range(start, end, freq='D')
    return (dates.dayofweek < 6).sum()

wdays_90 = count_working_days(ref_90, last_train_date)
wdays_56 = count_working_days(ref_56, last_train_date)
wdays_28 = count_working_days(ref_28, last_train_date)

print(f"\nWorking days in windows: 90d={wdays_90}, 56d={wdays_56}, 28d={wdays_28}")

def get_window_stats(series, start_date, end_date):
    """Get sum and active day count for a date window."""
    mask = (series.index >= start_date) & (series.index <= end_date)
    s = series[mask]
    total = s.sum()
    active_days = (s > 0).sum()
    return total, active_days

def sku_daily_rate(sku):
    """
    Returns a scalar 'baseline' daily rate for Mon (DoW=0 ~ 1.0 multiplier).
    For non-working days, the forecast will be scaled by dow_mult.
    """
    if sku not in sku_history:
        return 0.0
    
    series = sku_history[sku]
    
    # Activity check
    total_90, active_90 = get_window_stats(series, ref_90, last_train_date)
    total_56, active_56 = get_window_stats(series, ref_56, last_train_date)
    total_28, active_28 = get_window_stats(series, ref_28, last_train_date)
    
    # Inactive: no sales in last 90 days → 0
    if total_90 == 0:
        return 0.0
    
    # Average daily rate in each window (per working day)
    # Scale by global mean dow to get "per weekday" rate
    global_dow_factor = dow_mult.drop(6).mean()  # ~0.883 (avg Mon-Sat mult)
    
    rate_90 = total_90 / wdays_90
    rate_56 = total_56 / wdays_56 if active_56 > 0 else 0.0
    rate_28 = total_28 / wdays_28 if active_28 > 0 else 0.0
    
    # Weighted blend: more weight on recent windows
    # If SKU has very sparse sales, use longer window to reduce noise
    if active_28 >= 5:
        # Enough recent activity: blend 28d, 56d, 90d
        rate = 0.5 * rate_28 + 0.3 * rate_56 + 0.2 * rate_90
    elif active_56 >= 5:
        # Moderate activity: blend 56d and 90d
        rate = 0.6 * rate_56 + 0.4 * rate_90
    elif active_90 >= 3:
        # Sparse but some activity: use 90d
        rate = rate_90
    else:
        # Very sparse: use 90d but apply heavy discount to avoid overprediction
        rate = rate_90 * 0.5
    
    return max(0.0, rate)

print("\nComputing per-SKU daily rates...")
sku_rates = {}
for sku in all_skus:
    sku_rates[sku] = sku_daily_rate(sku)

rates = np.array([sku_rates[s] for s in all_skus])
print(f"  SKUs with rate > 0: {(rates > 0).sum()}")
print(f"  SKUs with rate = 0: {(rates == 0).sum()}")
print(f"  Rate distribution (nonzero):")
nz = rates[rates > 0]
for p in [10, 25, 50, 75, 90, 99]:
    print(f"    p{p}: {np.percentile(nz, p):.3f}")

# ==============================================================================
# 5. Build exponential smoothing for top-volume SKUs
# ==============================================================================
# For high-volume SKUs, use Holt-Winters style smoothing for better trend capture
print("\nComputing ETS rates for top-volume SKUs...")

# Identify top SKUs by total sales
sku_totals = daily.groupby('sku')['qty'].sum()
top_n = 500
top_skus = set(sku_totals.nlargest(top_n).index)

def exponential_smoothing_forecast(series, alpha=0.3, h=56):
    """
    Simple exponential smoothing on non-zero working-day series.
    Returns a scalar smoothed level.
    """
    if len(series) == 0:
        return 0.0
    vals = series.values
    level = vals[0]
    for v in vals[1:]:
        level = alpha * v + (1 - alpha) * level
    return max(0.0, level)

top_sku_rates = {}
for sku in top_skus:
    if sku not in sku_history:
        continue
    series = sku_history[sku]
    # Use last 180 days, working days only
    ref_180 = last_train_date - pd.Timedelta(days=179)
    mask = (series.index >= ref_180) & (series.index.dayofweek < 6)
    recent = series[mask]
    if len(recent) >= 10:
        # Resample to daily (fill missing working days with 0)
        wd_range = pd.date_range(ref_180, last_train_date, freq='B')
        resampled = recent.reindex(wd_range, fill_value=0)
        smoothed = exponential_smoothing_forecast(resampled, alpha=0.2)
        top_sku_rates[sku] = smoothed

# Override rates for top SKUs with ETS rates
# Blend with MA rate for stability
for sku in top_sku_rates:
    ma_rate = sku_rates.get(sku, 0.0)
    ets_rate = top_sku_rates[sku]
    # Blend: 60% ETS, 40% MA
    sku_rates[sku] = 0.6 * ets_rate + 0.4 * ma_rate

print(f"  Updated {len(top_sku_rates)} top-volume SKU rates with ETS")

# ==============================================================================
# 6. Generate forecast matrix (SKUs × 56 days)
# ==============================================================================
print("\nGenerating forecast matrix...")

# For each forecast date, apply DoW multiplier
forecast_dow = pd.Series(forecast_dates.dayofweek, index=forecast_dates)
dow_mults_forecast = forecast_dow.map(dow_mult).values  # shape: (56,)

# Build forecast: outer product of rates × DoW multipliers
sku_rate_arr = np.array([sku_rates[s] for s in all_skus])  # (15972,)
forecast_matrix = np.outer(sku_rate_arr, dow_mults_forecast)  # (15972, 56)

# Round to non-negative integers
forecast_matrix = np.round(np.maximum(0, forecast_matrix)).astype(int)

print(f"  Forecast matrix shape: {forecast_matrix.shape}")
print(f"  Total forecast qty: {forecast_matrix.sum():,}")
print(f"  Non-zero forecasts: {(forecast_matrix > 0).sum():,}")

# Sanity check: what does a recent average day look like?
recent_daily_total = daily[daily['date'] >= ref_28]['qty'].sum() / wdays_28
print(f"\n  Recent 28d avg working-day total: {recent_daily_total:.0f}")
non_sun_days = [d for d in forecast_dates if d.dayofweek != 6]
avg_forecast_day = forecast_matrix.sum(axis=0)
non_sun_forecast_avg = forecast_matrix[:, [i for i, d in enumerate(forecast_dates) if d.dayofweek != 6]].sum(axis=0).mean()
print(f"  Avg forecast working-day total: {non_sun_forecast_avg:.0f}")

# ==============================================================================
# 7. Build submission
# ==============================================================================
print("\nBuilding submission file...")

# Create lookup: sku → row index in forecast_matrix
sku_to_idx = {sku: i for i, sku in enumerate(all_skus)}

# Process submission rows
rows = []
forecast_cols = [f'F{i}' for i in range(1, 57)]

for _, row in sub.iterrows():
    row_id = row['id']
    sku = row_id.rsplit('_', 1)[0]
    row_type = row_id.rsplit('_', 1)[1]  # 'validation' or 'evaluation'
    
    sku_idx = sku_to_idx.get(sku, None)
    if sku_idx is None:
        preds = [0] * 56
    else:
        preds = forecast_matrix[sku_idx].tolist()
    
    if row_type == 'validation':
        # F1..F28
        out = {f'F{i}': preds[i-1] for i in range(1, 29)}
    else:
        # F1..F28 in submission but maps to F29..F56
        # Check submission columns
        out = {f'F{i}': preds[i+28-1] for i in range(1, 29)}
    
    rows.append({'id': row_id, **out})

result = pd.DataFrame(rows)
# Reorder columns
cols = ['id'] + [f'F{i}' for i in range(1, 29)]
result = result[cols]

print(f"  Output shape: {result.shape}")
print(f"  Sample rows:")
print(result.head(3).to_string())

result.to_csv('submission.csv', index=False)
print(f"\n✓ Submission saved to submission.csv")

# ==============================================================================
# 8. Quality checks
# ==============================================================================
print("\n=== QUALITY CHECKS ===")
print(f"Total rows: {len(result)}")
print(f"Validation rows: {result['id'].str.endswith('validation').sum()}")
print(f"Evaluation rows: {result['id'].str.endswith('evaluation').sum()}")
val = result[result['id'].str.endswith('validation')]
print(f"\nValidation forecast stats:")
print(val.iloc[:, 1:].describe().loc[['mean','50%','max']].to_string())
print(f"\nSKUs with all-zero forecasts: {(val.iloc[:, 1:].sum(axis=1) == 0).sum()}")
print(f"SKUs with nonzero forecasts: {(val.iloc[:, 1:].sum(axis=1) > 0).sum()}")

# Sunday verification (F2=Sep7, F9=Sep14, F16=Sep21, F23=Sep28, F30=Oct5, F37=Oct12, F44=Oct19, F51=Oct26 are Sundays)
# F2 is Sep 7 2025 (Sunday in validation)
print(f"\nF2 (Sep 7, Sunday) total: {val['F2'].sum()} (should be 0)")
print(f"F9 (Sep 14, Sunday) total: {val['F9'].sum()} (should be 0)")
print(f"F1 (Sep 6, Saturday) total: {val['F1'].sum()}")
print(f"F3 (Sep 8, Monday) total: {val['F3'].sum()}")
