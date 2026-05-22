import pandas as pd
import numpy as np

# 1. Tạo khoảng thời gian mở rộng để tránh hiệu ứng biên (boundary effect) khi tính pre/post holiday
start_date = pd.to_datetime('2019-12-20')
end_date = pd.to_datetime('2026-01-10')

dates = pd.date_range(start=start_date, end=end_date)
df = pd.DataFrame({'date': dates})

# 2. Định nghĩa các ngày lễ dương lịch cố định
def check_fixed_holiday(dt):
    return (dt.month == 1 and dt.day == 1) or \
           (dt.month == 4 and dt.day == 30) or \
           (dt.month == 5 and dt.day == 1) or \
           (dt.month == 9 and dt.day == 2)

# 3. Hard-code lịch nghỉ Tết Nguyên Đán từng năm cụ thể (bao gồm cả các ngày nghỉ bù/cuối tuần liền kề)
tet_ranges = [
    ('2020-01-23', '2020-01-29'),
    ('2021-02-10', '2021-02-16'),
    ('2022-01-31', '2022-02-04'),
    ('2023-01-20', '2023-01-26'),
    ('2024-02-08', '2024-02-14'),
    ('2025-01-25', '2025-02-02')
]

holiday_dates = set()
for start, end in tet_ranges:
    for d in pd.date_range(start, end):
        holiday_dates.add(d.date())

# Cập nhật cột is_holiday
df['is_holiday'] = df['date'].apply(lambda x: 1 if check_fixed_holiday(x) or x.date() in holiday_dates else 0)

# 4. Hard-code ngày Mùng 1 Tết qua các năm để tính khoảng cách đếm ngược/đếm tiến
tet_mung1 = [
    '2019-02-05', # Thêm mốc 2019 để tính cho những ngày đầu năm 2020
    '2020-01-25',
    '2021-02-12',
    '2022-02-01',
    '2023-01-22',
    '2024-02-10',
    '2025-01-29',
    '2026-02-17'  # Thêm mốc 2026 để tính cho những ngày cuối năm 2025
]
tet_mung1_dt = [pd.to_datetime(d) for d in tet_mung1]

def get_days_to_tet(dt):
    # Số ngày cho tới ngày Mùng 1 Tết tiếp theo
    upcoming = [t for t in tet_mung1_dt if t >= dt]
    if upcoming:
        return (upcoming[0] - dt).days
    return np.nan

def get_days_after_tet(dt):
    # Số ngày kể từ ngày Mùng 1 Tết gần nhất trước đó
    past = [t for t in tet_mung1_dt if t <= dt]
    if past:
        return (dt - past[-1]).days
    return np.nan

df['days_to_tet'] = df['date'].apply(get_days_to_tet)
df['days_after_tet'] = df['date'].apply(get_days_after_tet)

# 5. Tạo đặc trưng trước lễ (3 ngày) và sau lễ (5 ngày) đối với các ngày bình thường
df['is_pre_holiday'] = 0
df['is_post_holiday'] = 0

for i in range(len(df)):
    if df.loc[i, 'is_holiday'] == 0:
        # Nếu bất kỳ ngày nào trong 3 ngày tiếp theo là ngày lễ thương mại/Tết
        if (df.loc[i+1 : i+3, 'is_holiday'] == 1).any():
            df.loc[i, 'is_pre_holiday'] = 1
        # Nếu bất kỳ ngày nào trong 5 ngày trước đó là ngày lễ thương mại/Tết
        if (df.loc[max(0, i-5) : i-1, 'is_holiday'] == 1).any():
            df.loc[i, 'is_post_holiday'] = 1

# 6. Lọc lại chính xác khoảng thời gian yêu cầu 2020–2025
df_filtered = df[(df['date'] >= '2020-01-01') & (df['date'] <= '2025-12-31')].copy().reset_index(drop=True)

# 7. Xuất dữ liệu ra file Parquet (Yêu cầu cài đặt thư viện pyarrow: pip install pyarrow)
df_filtered.to_parquet('calendar_features.parquet', index=False)
print("Đã xuất file calendar_features.parquet thành công!")