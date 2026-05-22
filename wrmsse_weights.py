import pandas as pd
import numpy as np

# 1. Đọc và làm sạch dữ liệu số chứa dấu phẩy
df = pd.read_csv("train.csv")

def sanitize_numeric(col):
    # Ép toàn bộ về chuỗi, thay thế dấu phẩy (,) thành dấu chấm (.), sau đó ép về số thực
    # Cách này xử lý an toàn cho cả số nguyên, số thập phân có dấu phẩy và các đơn hàng bị âm
    return df[col].astype(str).str.replace(',', '.', regex=False).astype(float)

# Cập nhật lại tên cột chứa dấu cách cho chính xác
df['SalesAmount'] = sanitize_numeric('SalesAmount')
df['Cost Amount'] = sanitize_numeric('Cost Amount')

# 2. TÍNH PROFIT THEO YÊU CẦU CỦA BTC
# Công thức: Profit_i = SalesAmount - Cost Amount
df['Profit'] = df['SalesAmount'] - df['Cost Amount']

# Lọc các dòng bị âm lợi nhuận (nếu có chính sách cắt giảm, hoặc cứ để nguyên để cấn trừ)
# Tuỳ theo dữ liệu, nhưng ta cộng dồn toàn bộ lịch sử
profit_by_sku = df.groupby('ItemCode')['Profit'].sum().reset_index()

# Đảm bảo không có Profit < 0 làm sai lệch tỷ trọng, thông thường ép về 0
profit_by_sku['Profit'] = profit_by_sku['Profit'].clip(lower=0) 

# 3. TÍNH WEIGHT
total_system_profit = profit_by_sku['Profit'].sum()
profit_by_sku['wrmsse_weight'] = profit_by_sku['Profit'] / total_system_profit

# Đổi tên cột và lưu kết quả
final_weights = profit_by_sku.sort_values(by='wrmsse_weight', ascending=False)
final_weights.to_csv("wrmsse_weights.csv", index=False)

print(f"Top 3 SKU có Lợi Nhuận (và Trọng số) cao nhất hệ thống:")
print(final_weights.head(3))