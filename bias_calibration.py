import glob
import os
import pandas as pd

# ==============================================================================
# 1. NẠP DỮ LIỆU DỰ BÁO VÀ THỰC TẾ (RAW DATA - ĐÃ SỬA LỖI KEYERROR)
# ==============================================================================
print("🔄 Bước 1: Đang nạp dữ liệu panel và kết quả dự báo...")

# Đọc file phân khúc
df_skus = pd.read_parquet("sku_segments.parquet")
df_skus.columns = [col.upper() for col in df_skus.columns]

# Tự động dò tìm cột định danh sản phẩm TRONG BẢNG SEGMENT (Thường là 'SKU')
sku_col_skus = "SKU" if "SKU" in df_skus.columns else df_skus.columns[0]
# Tự động dò tìm tên cột phân khúc thực tế
seg_col = [col for col in df_skus.columns if "SEGMENT" in col][0]

# Đọc các file chuỗi thời gian chứa đặc trưng trễ và lượng bán
prediction_files = sorted(glob.glob("lag_features/lag_features_chunk*.parquet"))
df_raw = pd.read_parquet(prediction_files)
df_raw.columns = [col.upper() for col in df_raw.columns]

# Tự động dò tìm cột định danh sản phẩm TRONG BẢNG CHUỖI THỜI GIAN (Là 'ITEMCODE')
sku_col_ts = "ITEMCODE" if "ITEMCODE" in df_raw.columns else ("SKU" if "SKU" in df_raw.columns else df_raw.columns[0])
qty_col = "NET_QTY" if "NET_QTY" in df_raw.columns else "QUANTITY"

# Xác định cột dự báo của mô hình
pred_col = "PREDICTED" if "PREDICTED" in df_raw.columns else ([col for col in df_raw.columns if "PRED" in col] + [qty_col])[0]

print(f"📊 Khóa bảng TS: '{sku_col_ts}' | Khóa bảng Segment: '{sku_col_skus}' | Phân khúc: '{seg_col}'")

# THỰC HIỆN MERGE LINH HOẠT THEO HAI KHÓA KHÁC NHAU
df_main = pd.merge(
    df_raw, 
    df_skus[[sku_col_skus, seg_col]], 
    left_on=sku_col_ts, 
    right_on=sku_col_skus, 
    how="inner"
)

# Đồng nhất tên cột định danh về 'SKU' để các bước gom nhóm phía sau chạy mượt mà
df_main["SKU"] = df_main[sku_col_ts]

# Xử lý thời gian để trích xuất Thứ (DoW) và Tháng (Month)
df_main["DATE"] = pd.to_datetime(df_main["DATE"])
df_main["DOW"] = df_main["DATE"].dt.day_name()  # Trích xuất thứ tiếng Anh (Monday, Tuesday...)
df_main["MONTH"] = df_main["DATE"].dt.month  # Trích xuất tháng (1, 2, 3...)

# ==============================================================================
# 2. TÍNH TOÁN BIAS RATIO & CORRECTION FACTOR THEO NHÓM
# ==============================================================================
print("🧮 Bước 2: Đang tính toán Systematic Bias cho từng nhóm...")


# Hàm hỗ trợ tính toán bias và hệ số hiệu chỉnh an toàn
def calculate_calibration(df_group, groupby_cols):
    summary = (
        df_group.groupby(groupby_cols)
        .agg(total_actual=(qty_col, "sum"), total_predicted=(pred_col, "sum"))
        .reset_index()
    )

    # Tránh lỗi chia cho 0 nếu có nhóm nào đó tổng lượng bán thực tế bằng 0
    summary["bias_ratio"] = summary["total_predicted"] / summary[
        "total_actual"
    ].replace(0, 1)

    # Xác định trạng thái lệch
    summary["status"] = summary["bias_ratio"].apply(
        lambda x: "OVER-PREDICT"
        if x > 1.05
        else ("UNDER-PREDICT" if x < 0.95 else "NORMAL")
    )

    # Tính Correction Factor (Hệ số sửa sai) = 1 / Bias Ratio
    # Nếu thực tế bằng 0 hoặc dự báo bằng 0 thì giữ hệ số an toàn là 1.0
    summary["correction_factor"] = 1 / summary["bias_ratio"]
    summary.loc[
        (summary["total_actual"] == 0) | (summary["total_predicted"] == 0),
        "correction_factor",
    ] = 1.0

    return summary


# A. Tính toán theo Thứ (DoW) + Phân khúc (Segment) để phục vụ cho Task chốt chặn
df_calibration_dow = calculate_calibration(df_main, [seg_col, "DOW"])

# B. Tính toán theo Tháng + Phân khúc để bạn mở rộng phân tích
df_calibration_month = calculate_calibration(df_main, [seg_col, "MONTH"])

# ==============================================================================
# 3. KIỂM TRA CHỐT CHẶN KHẨN CẤP (MÔ HÌNH TOP VÀO THỨ BẢY OVER-PREDICT 20%)
# ==============================================================================
print("🚨 Bước 3: Đang rà soát chốt chặn an toàn cho nhóm TOP vào thứ Bảy...")

# Lọc tìm bản ghi của nhóm TOP vào ngày Thứ Bảy (Saturday)
top_sat_record = df_calibration_dow[
    (df_calibration_dow[seg_col].str.upper() == "TOP")
    & (df_calibration_dow["DOW"].str.upper() == "SATURDAY")
]

if not top_sat_record.empty:
    actual_bias = top_sat_record["bias_ratio"].values[0]
    # Over-predict 20% nghĩa là bias_ratio tương ứng >= 1.20
    if actual_bias >= 1.20:
        over_pct = (actual_bias - 1) * 100
        print(f"\n[🛑 CẢNH BÁO BÁO ĐỘNG ĐỎ GỬI D]")
        print(
            f"-> Phát hiện mô hình TOP đang OVER-PREDICT nghiêm trọng vào ngày thứ Bảy!"
        )
        print(
            f"-> Tỷ lệ lệch thực tế: {over_pct:.2f}% (Vượt ngưỡng chốt chặn 20%)."
        )
        print(
            f"-> HÀNH ĐỘNG: Đề xuất D áp thêm hệ số nhân multiplier = {top_sat_record['correction_factor'].values[0]:.4f} ngay lập tức!"
        )
    else:
        print(
            f"-> An toàn: Nhóm TOP vào thứ Bảy có mức bias là {actual_bias:.2f} (Không bị vượt ngưỡng 20%)."
        )
else:
    print(
        "⚠️ Không tìm thấy bản ghi kết hợp giữa nhóm TOP và ngày Saturday trong dữ liệu thô."
    )

# ==============================================================================
# 4. KẾT XUẤT FILE BIAS_CALIBRATION.PARQUET ĐỂ BÀN GIAO LÚC 12:00
# ==============================================================================
print("\n💾 Bước 4: Đang đóng gói file bàn giao...")

# Chuẩn hóa cấu trúc đầu ra theo đúng mô tả: (segment, dow_or_month, correction_factor)
# Gom hai bảng phân tích lại thành một cấu trúc đồng nhất để bàn giao
df_calibration_dow = df_calibration_dow.rename(columns={"DOW": "TIME_WINDOW"})
df_calibration_month = df_calibration_month.rename(
    columns={"MONTH": "TIME_WINDOW"}
)
df_calibration_month["TIME_WINDOW"] = (
    "Month_" + df_calibration_month["TIME_WINDOW"].astype(str)
)

final_delivery_df = pd.concat(
    [
        df_calibration_dow[[seg_col, "TIME_WINDOW", "correction_factor"]],
        df_calibration_month[[seg_col, "TIME_WINDOW", "correction_factor"]],
    ],
    axis=0,
).reset_index(drop=True)

# Ghi dữ liệu ra file parquet
output_filename = "bias_calibration.parquet"
final_delivery_df.to_parquet(output_filename, index=False)

print(
    f"✨ HOÀN THÀNH BIÊN TẬP! File '{output_filename}' đã được kết xuất thành công."
)
print(f"👉 Sẵn sàng bàn giao đúng hạn lúc 12:00!")

# In thử 10 dòng đầu của file bàn giao để xem trước cấu trúc
print("\n--- BẢNG XEM TRƯỚC DỮ LIỆU BÀN GIAO ---")
print(final_delivery_df.head(10).to_string(index=False))