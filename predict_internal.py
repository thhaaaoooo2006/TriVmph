import pandas as pd
import joblib
import os
import warnings

# Bỏ qua các cảnh báo lặt vặt để terminal gọn gàng
warnings.filterwarnings('ignore')

print("🚀 [1/5] Đang tải dữ liệu và nạp 2 CHUYÊN GIA DỰ BÁO...")
df = pd.read_parquet('features_top_active.parquet')

# Nạp 2 mô hình đã train (Global và Fine-Tuned)
model_global = joblib.load('lgbm_active_model.pkl') 
try:
    model_finetuned = joblib.load('lgbm_active_finetuned.pkl') 
except FileNotFoundError:
    print("⚠️ Chưa tìm thấy file 'lgbm_active_finetuned.pkl', dùng tạm model global cho cả hai.")
    model_finetuned = joblib.load('lgbm_active_model.pkl')

print("🔍 [2/5] Đang lọc tập dữ liệu ACTIVE...")

item_col = 'ItemCode'      
date_col = 'Date'
target_col = 'net_qty'
df[date_col] = pd.to_datetime(df[date_col])

segment_file = 'sku_segments.parquet'
if os.path.exists(segment_file):
    df_seg = pd.read_parquet(segment_file)
    
    # 💥 QUAN TRỌNG: Lôi mã SKU từ Index lên thành cột (nếu nó đang bị kẹt ở Index)
    if 'ItemCode' not in df_seg.columns:
        # Tự động đẩy index thành cột và đặt tên là 'ItemCode'
        df_seg = df_seg.reset_index(names='ItemCode' if df_seg.index.name is None else None)
        if df_seg.columns[0] != 'ItemCode' and 'ItemCode' not in df_seg.columns:
            df_seg.rename(columns={df_seg.columns[0]: 'ItemCode'}, inplace=True)
            
    print(f"   -> Các cột trong file segment sau khi xử lý Index: {df_seg.columns.tolist()}")
    
    # Đã sửa lại thành 'segment' (chữ thường)
    seg_col = 'segment'  
    seg_item_col = 'ItemCode'
    
    if seg_col in df_seg.columns and seg_item_col in df_seg.columns:
        # Lọc các SKU có phân khúc là 'ACTIVE' (Cẩn thận: Kiểm tra xem file của bạn là 'ACTIVE' hay 'active')
        active_skus = df_seg[df_seg[seg_col].astype(str).str.upper() == 'ACTIVE'][seg_item_col].astype(str).tolist()
        
        # Tiến hành lọc df chính
        df = df[df[item_col].astype(str).isin(active_skus)].copy()
        print(f"   -> Đã lọc xong! Còn lại {len(df)} dòng dữ liệu thuộc nhóm ACTIVE.")
    else:
        raise ValueError(f"❌ Vẫn không tìm thấy cột! Hãy kiểm tra file {segment_file}.")
else:
    print("⚠️ Không tìm thấy file sku_segments.parquet. Bỏ qua bước lọc.")

print("✂️ [3/5] Đang trích xuất 28 ngày cuối cùng làm tập Validation...")
cutoff_date = pd.to_datetime('2025-09-06')
df = df[df[date_col] < cutoff_date].copy()

# Fix lỗi nếu sau khi lọc không còn data
if len(df) == 0:
    print("❌ LỖI NGHIÊM TRỌNG: Sau khi lọc, dữ liệu trống rỗng (0 dòng)!")
    print("   -> Hãy mở file sku_segments.parquet lên xem có mã SKU nào mang chữ ACTIVE không.")
    exit()

max_date = df[date_col].max()
val_start_date = max_date - pd.Timedelta(days=27) 
val_df = df[df[date_col] >= val_start_date].copy()

print("⚙️ [4/5] Đang đồng bộ hóa Features và Ép kiểu Category...")
expected_features = model_global.feature_name_
for col in expected_features:
    if col not in val_df.columns:
        val_df[col] = 0

X_val = val_df[expected_features].copy()

for col in expected_features:
    if not (pd.api.types.is_numeric_dtype(X_val[col]) or pd.api.types.is_bool_dtype(X_val[col])):
        X_val[col] = X_val[col].astype('category')

print(f"🤖 [5/5] Tiến hành ENSEMBLE PREDICTION cho {len(X_val)} dòng dữ liệu...")
pred_global = model_global.predict(X_val)
pred_finetuned = model_finetuned.predict(X_val)

val_df['Predicted_Quantity'] = (0.5 * pred_global) + (0.5 * pred_finetuned)
val_df['Predicted_Quantity'] = val_df['Predicted_Quantity'].fillna(0).clip(lower=0)

result_df = val_df[[date_col, item_col, target_col, 'Predicted_Quantity']]
output_file = 'predictions_active_internal_blended.parquet'
result_df.to_parquet(output_file, index=False)

print(f"✅ HOÀN TẤT! Đã lưu kết quả dự báo tổng hợp vào: {output_file}")