import glob
import pandas as pd

# 1. Tìm tất cả các file chunk
files = sorted(glob.glob('filtered_data/daily_panel_filtered_chunk*.parquet'))
print(f"👉 Tìm thấy {len(files)} file chunks để xử lý.")

DATE_COL = 'Date'       # Chữ D viết hoa theo đúng metadata của bạn
QTY_COL = 'net_qty'

chunk_aggregates = []

print("\n🔄 Bước 1: Đọc dữ liệu thực tế và gom cụm theo tháng...")
for file in files:
    # Chỉ đọc 2 cột cần thiết để tối ưu RAM
    df_chunk = pd.read_parquet(file, columns=[DATE_COL, QTY_COL])
    
    # Ép kiểu Date về Datetime chuẩn, tự động xử lý múi giờ
    df_chunk[DATE_COL] = pd.to_datetime(df_chunk[DATE_COL], utc=True)
    
    # KHÔNG dùng bộ lọc năm cũ nữa, trích xuất thẳng 'YYYY-MM' từ dữ liệu hiện tại
    df_chunk['year_month'] = df_chunk[DATE_COL].dt.to_period('M')
    
    # Tính Tổng (sum) và Số dòng (count) của net_qty theo từng tháng trong chunk này
    agg = df_chunk.groupby('year_month')[QTY_COL].agg(['sum', 'count']).reset_index()
    chunk_aggregates.append(agg)

# 2. Gộp kết quả của tất cả các chunk một cách chính xác
print("📊 Bước 2: Tổng hợp dữ liệu trên toàn bộ các chunks...")
total_agg = pd.concat(chunk_aggregates, ignore_index=True)
total_monthly = total_agg.groupby('year_month').sum().reset_index()

# Tính toán lượng bán trung bình thực tế cho từng tháng (mean = tổng_qty / tổng_số_dòng)
total_monthly['mean_qty'] = total_monthly['sum'] / total_monthly['count']

# Sắp xếp theo thứ tự thời gian tăng dần
total_monthly = total_monthly.sort_values(by='year_month').reset_index(drop=True)

# 3. In bảng kết quả hiển thị dữ liệu thực tế của năm 2026
print("\n📋 BẢNG THỐNG KÊ LƯỢNG BÁN TRUNG BÌNH THEO TỪNG THÁNG THỰC TẾ")
print("-" * 65)
print(f"{'Tháng (Year-Month)':<20} | {'Tổng dòng (count)':<15} | {'Lượng bán Trung bình':<20}")
print("-" * 65)

for _, row in total_monthly.iterrows():
    print(f"{str(row['year_month']):<20} | {int(row['count']):<15} | {row['mean_qty']:<20.4f}")

print("-" * 65)