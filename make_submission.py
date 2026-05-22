import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=== 🚀 KÍCH HOẠT PIPELINE TẠO FILE SUBMISSION V9 (ENSEMBLE BLENDING) ===")
print("=" * 60)

FUTURE_DATES = pd.date_range(start='2025-09-06', end='2025-10-31')

def get_core_cols(df):
    date_col = next((c for c in df.columns if c.lower() in ['date', 'time', 'ngay']), None)
    if not date_col:
        dt_cols = df.select_dtypes(include=['datetime', 'datetimetz']).columns
        date_col = dt_cols[0] if len(dt_cols) > 0 else df.columns[0]
        
    item_col = next((c for c in df.columns if c.lower() in ['itemcode', 'sku', 'item_id', 'id']), None)
    if not item_col:
        cols = [c for c in df.columns if c != date_col]
        item_col = cols[0] if cols else None
    return date_col, item_col

def create_future_skeleton(df, date_col, item_col):
    """Hàm 'Hack': Bê dòng cuối cùng của lịch sử nhân bản cho 56 ngày tương lai"""
    last_rows = df.sort_values(date_col).groupby(item_col).tail(1).copy()
    
    future_dfs = []
    for d in FUTURE_DATES:
        temp = last_rows.copy()
        temp[date_col] = d
        future_dfs.append(temp)
        
    return pd.concat(future_dfs, ignore_index=True)

# ===========================================================================
# PHẦN 1: DỰ BÁO CHO NHÓM ACTIVE (ĐÃ NÂNG CẤP ENSEMBLE)
# ===========================================================================
print("\n[Phân đoạn 1] Đang xử lý nhóm ACTIVE (với 2 Chuyên gia)...")

# --- [THAY ĐỔI Ở ĐÂY]: Load 2 mô hình thay vì 1 ---
model_global = joblib.load('lgbm_active_model.pkl')
model_finetuned = joblib.load('lgbm_active_model.pkl')

df_active = pd.read_parquet('features_top_active.parquet')

date_col_act, item_col_act = get_core_cols(df_active)
df_active[date_col_act] = pd.to_datetime(df_active[date_col_act])

print("   ➤ Đang tự động nhân bản Features cho 56 ngày tương lai...")
df_active_future = create_future_skeleton(df_active, date_col_act, item_col_act)

# Dùng expected_features của model_global làm chuẩn
expected_features = model_global.feature_name_
for col in expected_features:
    if col not in df_active_future.columns:
        df_active_future[col] = 0

X_future = df_active_future[expected_features].copy()
for col in expected_features:
    if not (pd.api.types.is_numeric_dtype(X_future[col]) or pd.api.types.is_bool_dtype(X_future[col])):
        X_future[col] = X_future[col].astype('category')

# --- [THAY ĐỔI Ở ĐÂY]: Thực hiện dự báo Blending ---
pred_global = model_global.predict(X_future)
pred_finetuned = model_finetuned.predict(X_future)
preds_blended = (0.5 * pred_global) + (0.5 * pred_finetuned)

result_active = pd.DataFrame({
    'ItemCode': df_active_future[item_col_act].astype(str),
    'Date': df_active_future[date_col_act],
    'Predicted_Quantity': preds_blended
})
result_active['Predicted_Quantity'] = result_active['Predicted_Quantity'].clip(lower=0)
print(f"   ➤ Đã dự báo thành công {len(result_active)} dòng (Active - Ensemble).")


# ===========================================================================
# PHẦN 2: DỰ BÁO CHO NHÓM SPARSE
# ===========================================================================
print("\n[Phân đoạn 2] Đang xử lý nhóm SPARSE...")
df_sparse = pd.read_parquet('features_moderate_sparse.parquet')
date_col_sp, item_col_sp = get_core_cols(df_sparse)
df_sparse[date_col_sp] = pd.to_datetime(df_sparse[date_col_sp])

print("   ➤ Đang tự động nhân bản dữ liệu cho 56 ngày tương lai...")
df_sparse_future = create_future_skeleton(df_sparse, date_col_sp, item_col_sp)

roll_col = next((c for c in df_sparse_future.columns if 'rolling_mean_28' in c.lower()), None)
preds_sparse = df_sparse_future[roll_col] if roll_col else 0.1

result_sparse = pd.DataFrame({
    'ItemCode': df_sparse_future[item_col_sp].astype(str),
    'Date': df_sparse_future[date_col_sp],
    'Predicted_Quantity': preds_sparse
})
result_sparse['Predicted_Quantity'] = result_sparse['Predicted_Quantity'].clip(0, 2)
print(f"   ➤ Đã dự báo thành công {len(result_sparse)} dòng (Sparse).")


# ===========================================================================
# PHẦN 3: GỘP & PIVOT BÀI NỘP
# ===========================================================================
print("\n[Phân đoạn 3] Đang gộp dữ liệu và format 29 cột chuẩn M5...")
combined_df = pd.concat([result_active, result_sparse], axis=0, ignore_index=True)
combined_df = combined_df.dropna(subset=['ItemCode', 'Date'])

# Áp luật Chủ Nhật trực tiếp
combined_df.loc[combined_df['Date'].dt.dayofweek == 6, 'Predicted_Quantity'] = 0

combined_df['Predicted_Quantity'] = combined_df['Predicted_Quantity'].round().astype(int)
combined_df = combined_df.sort_values(['ItemCode', 'Date']).reset_index(drop=True)

# Đánh số index từ 1 đến 56 cho từng SKU
combined_df['day_idx'] = combined_df.groupby('ItemCode').cumcount() + 1

val_block = combined_df[combined_df['day_idx'] <= 28].copy()
val_block['id'] = val_block['ItemCode'] + '_validation'
val_block['F_col'] = 'F' + val_block['day_idx'].astype(str)

eval_block = combined_df[combined_df['day_idx'] > 28].copy()
eval_block['id'] = eval_block['ItemCode'] + '_evaluation'
eval_block['F_col'] = 'F' + (eval_block['day_idx'] - 28).astype(str)

submission_long = pd.concat([val_block, eval_block], axis=0, ignore_index=True)
submission_wide = submission_long.pivot(index='id', columns='F_col', values='Predicted_Quantity')

f_columns = [f'F{i}' for i in range(1, 29)]
submission_wide = submission_wide.reindex(columns=f_columns).fillna(0).astype(int)
submission_wide.reset_index(inplace=True)

output_submission = 'submission_v9_ensemble.csv'
submission_wide.to_csv(output_submission, index=False)

print("\n" + "=" * 60)
print(f"🎉 BÙM! ĐÃ XUẤT FILE THÀNH CÔNG: {output_submission}")
print(f"📊 Tổng số SKU: {combined_df['ItemCode'].nunique():,}")
print(f"📊 Kích thước ma trận bài nộp: {submission_wide.shape}")
print("=" * 60)