import numpy as np
import pandas as pd
import glob

# ==========================================
# 1. ĐỌC VÀ CHUẨN BỊ DỮ LIỆU
# ==========================================
# Đọc dữ liệu từ file parquet của bạn
# 1. Tìm tất cả các file chunk
files = sorted(glob.glob('filtered_data/daily_panel_filtered_chunk*.parquet'))
print(f"👉 Tìm thấy {len(files)} file chunks để xử lý.")

# Đọc từng file chunk và nối vào DataFrame chính
df = pd.concat([pd.read_parquet(file) for file in files], ignore_index=True)

DATE_COL = 'Date'       # Chữ D viết hoa theo đúng metadata của bạn
QTY_COL = 'net_qty'

# Đảm bảo cột date ở dạng datetime để trích xuất thứ trong tuần
df[DATE_COL] = pd.to_datetime(df[DATE_COL])

# Trích xuất thứ trong tuần: dt.dayofweek trả về từ 0 (Thứ 2) đến 6 (Chủ Nhật)
df["dow"] = df[DATE_COL].dt.dayofweek

# ==========================================
# 2. TÍNH TRUNG BÌNH QUANTITY THEO DOW
# ==========================================
# Gom nhóm theo dow và tính trung bình quantity của toàn bộ hệ thống
dow_analysis = (
    df.groupby("dow")[QTY_COL].mean().reset_index(name="mean_quantity")
)

# Tạo một cột tên thứ tiếng Anh để dễ đọc khi print kiểm tra
dow_names = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}
dow_analysis["dow_name"] = dow_analysis["dow"].map(dow_names)

print("--- Kết quả phân tích lượng bán trung bình theo Thứ ---")
print(dow_analysis[["dow", "dow_name", "mean_quantity"]])
print("-" * 50)

# ==========================================
# 3. XÁC NHẬN SUNDAY ≈ 0 & TÍNH MULTIPLIER
# ==========================================
# Lấy giá trị trung bình của ngày Chủ Nhật (dow = 6)
sunday_mean = dow_analysis.loc[dow_analysis["dow"] == 6, "mean_quantity"].values[
    0
]

print(f"Kiểm tra doanh số Chủ Nhật (Sunday Mean): {sunday_mean:.4f}")
if sunday_mean < 0.1:  # Ngưỡng xấp xỉ 0 (tùy thuộc vào scale dữ liệu của bạn)
    print("=> Xác nhận: Doanh số Chủ Nhật xấp xỉ bằng 0 (Sunday ≈ 0).")
else:
    print(
        "=> Lưu ý: Doanh số Chủ Nhật lớn hơn kỳ vọng, hãy kiểm tra lại dữ liệu."
    )
print("-" * 50)

# Lọc bỏ ngày Chủ Nhật để tính toán hệ số multiplier cho Mon-Sat
# (Vì nếu Sunday ≈ 0, đưa vào tính trung bình chung sẽ làm lệch hệ số của các ngày khác)
active_days = dow_analysis[dow_analysis["dow"] != 6].copy()

# Tính lượng bán trung bình tổng thể của các ngày hoạt động (Mon-Sat)
overall_active_mean = active_days["mean_quantity"].mean()

# Multiplier = Trung bình của thứ đó / Trung bình chung của các ngày hoạt động
active_days["multiplier"] = (
    active_days["mean_quantity"] / overall_active_mean
)

# ==========================================
# 4. TẠO FILE DOW_MULTIPLIERS.PARQUET
# ==========================================
# Chuẩn bị bảng kết quả cuối cùng bao gồm cả Chủ Nhật (gán multiplier của Sunday = 0.0)
final_multipliers = pd.DataFrame({"dow": range(7)})

# Map hệ số multiplier từ bảng active_days sang, ngày nào thiếu (Sunday) sẽ điền 0.0
final_multipliers = final_multipliers.merge(
    active_days[["dow", "multiplier"]], on="dow", how="left"
).fillna(0.0)

# Ép kiểu dữ liệu chuẩn theo yêu cầu
final_multipliers["dow"] = final_multipliers["dow"].astype(int)
final_multipliers["multiplier"] = final_multipliers["multiplier"].astype(float)

# Xuất ra file parquet
final_multipliers.to_parquet("dow_multipliers.parquet", index=False)

print("--- Bảng hệ số Multiplier cuối cùng đã lưu ---")
print(final_multipliers)