import pandas as pd
import numpy as np
import os

print("🚀 [TASK D] BẮT ĐẦU QUY TRÌNH MERGE PREDICTIONS & POST-PROCESSING...")

def safe_extract(df, file_name):
    """Trích xuất chính xác 3 cột, tạo bảng mới để cắt đứt lỗi trùng lặp cột (InvalidIndexError)"""
    # Xóa các cột bị trùng tên (nếu có)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    
    date_col = next((c for c in df.columns if c.lower() == 'date'), None)
    item_col = next((c for c in df.columns if c.lower() in ['itemcode', 'sku', 'item_id', 'id']), None)
    pred_col = next((c for c in df.columns if c.lower() in ['predicted_quantity', 'forecast_qty', 'prediction', 'net_qty']), None)
    
    if not pred_col:
        leftover = [c for c in df.columns if c not in [date_col, item_col]]
        if leftover: pred_col = leftover[-1]
        
    if not (date_col and item_col and pred_col):
        return None
        
    # Bốc đúng 3 cột tạo thành DataFrame mới tinh
    df_clean = pd.DataFrame({
        'Date': pd.to_datetime(df[date_col]),
        'ItemCode': df[item_col].astype(str),
        'Predicted_Quantity': df[pred_col]
    })
    return df_clean

df_list = []

# 1. Đọc TOP
if os.path.exists('predictions_top_internal.parquet'):
    print("  -> Đang nạp dự báo nhóm TOP...")
    df_top = pd.read_parquet('predictions_top_internal.parquet')
    clean_top = safe_extract(df_top, 'TOP')
    if clean_top is not None: df_list.append(clean_top)

# 2. Đọc ACTIVE
if os.path.exists('predictions_active_internal.parquet'):
    print("  -> Đang nạp dự báo nhóm ACTIVE...")
    df_act = pd.read_parquet('predictions_active_internal.parquet')
    clean_act = safe_extract(df_act, 'ACTIVE')
    if clean_act is not None: df_list.append(clean_act)

# 3. Đọc SPARSE/INACTIVE (Xử lý dứt điểm định dạng ngang F1-F56)
if os.path.exists('predictions_sparse_inactive.parquet'):
    df_si = pd.read_parquet('predictions_sparse_inactive.parquet')
    
    # Kiểm tra xem có phải định dạng ngang (Wide format: F1, F2...) không
    if 'F1' in df_si.columns:
        print("  -> Phát hiện SPARSE/INACTIVE ở dạng ngang (F1-F56). Đang xoay dọc (Melt)...")
        
        # Chỉ lấy 28 ngày đầu tiên (F1 đến F28) cho đợt pseudo-test
        f_cols = [f'F{i}' for i in range(1, 29)] 
        f_cols = [c for c in f_cols if c in df_si.columns]
        
        item_col_si = 'sku' if 'sku' in df_si.columns else ('ItemCode' if 'ItemCode' in df_si.columns else df_si.columns[0])
        
        # Xoay dữ liệu từ ngang sang dọc
        df_si_long = pd.melt(df_si, id_vars=[item_col_si], value_vars=f_cols, 
                             var_name='F_day', value_name='Predicted_Quantity')
        
        # Chuyển F1, F2... thành ngày thực tế (09/08/2025)
        date_map = {f'F{i}': pd.to_datetime('2025-08-09') + pd.Timedelta(days=i-1) for i in range(1, 57)}
        df_si_long['Date'] = df_si_long['F_day'].map(date_map)
        
        df_si_clean = pd.DataFrame({
            'Date': df_si_long['Date'],
            'ItemCode': df_si_long[item_col_si].astype(str),
            'Predicted_Quantity': df_si_long['Predicted_Quantity']
        })
        df_list.append(df_si_clean)
        print("  ✅ Đã xử lý và nạp xong SPARSE/INACTIVE.")
    else:
        clean_si = safe_extract(df_si, 'SPARSE/INACTIVE')
        if clean_si is not None: df_list.append(clean_si)

# =====================================================================
# MERGE VÀ POST-PROCESSING
# =====================================================================
if len(df_list) == 0:
    print("❌ LỖI: Không có file dữ liệu hợp lệ.")
    exit()

all_preds = pd.concat(df_list, ignore_index=True)

print("⚙️ Đang áp dụng quy trình Post-processing...")
# Ép Chủ Nhật về 0
all_preds['day_of_week'] = all_preds['Date'].dt.dayofweek
all_preds.loc[all_preds['day_of_week'] == 6, 'Predicted_Quantity'] = 0.0

# Chặn âm và làm tròn
all_preds['Predicted_Quantity'] = all_preds['Predicted_Quantity'].clip(lower=0).round().astype(int)

# Xuất file
print("💾 Đang xuất bản các file...")
all_preds[['Date', 'ItemCode', 'Predicted_Quantity']].to_parquet('all_predictions_v1.parquet', index=False)
with open('internal_wrmsse_v1.txt', 'w', encoding='utf-8') as f:
    f.write(f"Total Rows Merged: {len(all_preds)}\nStatus: Merge and melt complete.\n")
all_preds[['ItemCode', 'Date', 'Predicted_Quantity']].to_csv('submission_v1.csv', index=False)

print("\n🎉 --- HOÀN THÀNH XUẤT SẮC TOÀN BỘ NHIỆM VỤ TASK D ---")