import os
import pandas as pd

# Cấu hình chu kỳ Test thực tế (Dự báo 56 ngày tiếp theo sau train)
START_TEST_DATE = "2025-09-06"
N_DAYS = 56
OUTPUT_MASK_FILE = "sunday_mask.parquet"


def generate_sunday_mask():
    print(f"⚡ Đang khởi tạo chuỗi ngày test từ: {START_TEST_DATE}")

    # Tạo dải chuỗi thời gian liên tục cho 56 ngày tiếp theo
    date_range = pd.date_range(start=START_TEST_DATE, periods=N_DAYS, freq="D")

    mask_records = []

    # Quét qua từng ngày để tìm Chủ Nhật (dayofweek == 6 trong pandas)
    for idx, dt in enumerate(date_range):
        # Thiết lập vị trí cột tương ứng: idx = 0 -> F1, idx = 1 -> F2...
        forecast_day_index = idx + 1
        forecast_column_name = f"F{forecast_day_index}"

        # Kiểm tra nếu là Chủ Nhật
        if dt.dayofweek == 6:
            mask_records.append(
                {
                    "forecast_day_index": forecast_day_index,
                    "forecast_column": forecast_column_name,
                    "date": dt.strftime("%Y-%m-%d"),
                    "force_zero": True,
                }
            )

    # Chuyển thành Dataframe
    mask_df = pd.DataFrame(mask_records)

    # Xuất ra định dạng Parquet
    mask_df.to_parquet(OUTPUT_MASK_FILE, index=False)

    print(
        f"✅ Đã tạo thành công file: {OUTPUT_MASK_FILE} với {len(mask_df)} ngày Chủ Nhật."
    )
    print("\n📋 DANH SÁCH CÁC NGÀY CHỦ NHẬT BỊ ÉP ĐỂ VỀ 0 (FORCE ZERO):")
    print("-" * 55)
    print(mask_df.to_string(index=False))
    print("-" * 55)


if __name__ == "__main__":
    generate_sunday_mask()


#Khi bạn chuẩn bị xuất file nộp bài (submission), bạn chỉ cần dùng đoạn code sau để ép giá trị về 0:
"""sub = pd.read_csv("submission_model.csv")  # File kết quả từ model
mask = pd.read_parquet("sunday_mask.parquet")

# Duyệt qua các cột Chủ Nhật và gán thẳng bằng 0
for col in mask["forecast_column"]:
    if col in sub.columns:
        sub[col] = 0.0

sub.to_csv("final_submission.csv", index=False)""""