import pandas as pd

def clean_and_normalize_data(file_path, output_path):
    print("--- Bắt đầu quá trình làm sạch dữ liệu ---")
    
    # 1. Đọc dữ liệu từ file CSV
    # Sử dụng low_memory=False vì cột 'Stt' chứa dữ liệu hỗn hợp (cả số và mã chữ)
    df = pd.read_csv(file_path, low_memory=False)
    initial_rows = len(df)
    print(f"Số lượng dòng dữ liệu ban đầu: {initial_rows:,}")
    
    # 2. Chuẩn hóa tên các cột thành định dạng snake_case (chữ thường, dấu gạch dưới)
    column_mapping = {
        'Date': 'date',
        'Stt': 'stt',
        'ItemCode': 'item_code',
        'Quantity': 'quantity',
        'UnitPrice': 'unit_price',
        'SalesAmount': 'sales_amount',
        'Unit Cost': 'unit_cost',
        'Cost Amount': 'cost_amount'
    }
    df = df.rename(columns=column_mapping)
    print("-> Đã chuẩn hóa tên các cột thành chữ thường và snake_case.")
    
    # 3. Xử lý và loại bỏ các dòng trùng lặp hoàn toàn
    df = df.drop_duplicates()
    dropped_duplicates = initial_rows - len(df)
    print(f"-> Đã loại bỏ {dropped_duplicates:,} dòng trùng lặp hoàn toàn.")
    
    # 4. Chuẩn hóa định dạng Ngày tháng
    df['date'] = pd.to_datetime(df['date'])
    print("-> Đã chuyển đổi cột 'date' sang định dạng datetime64.")
    
    # 5. Làm sạch và xử lý các cột dạng chuỗi (Text/Categorical)
    # Loại bỏ khoảng trắng thừa ở đầu/cuối của mã đơn hàng và mã sản phẩm
    df['stt'] = df['stt'].astype(str).str.strip()
    df['item_code'] = df['item_code'].astype(str).str.strip()
    print("-> Đã định dạng chuỗi và xóa khoảng trắng thừa cho 'stt' và 'item_code'.")
    
    # 6. Chuẩn hóa các cột dữ liệu số (Numerical)
    # Thay thế dấu phẩy ',' thành dấu chấm '.' làm dấu thập phân và xóa dấu nháy đôi thừa
    numeric_cols_to_fix = ['unit_price', 'unit_cost', 'cost_amount']
    for col in numeric_cols_to_fix:
        df[col] = df[col].astype(str).str.replace(',', '.').str.replace('"', '').astype(float)
    
    # Đảm bảo các cột số lượng và doanh thu đúng kiểu dữ liệu số
    df['quantity'] = df['quantity'].astype(int)
    df['sales_amount'] = df['sales_amount'].astype(float)
    print("-> Đã chuẩn hóa định dạng số thực/số nguyên cho các cột đơn giá, chi phí và doanh thu.")
    
    # 7. Xuất dữ liệu đã làm sạch ra file mới
    df.to_csv(output_path, index=False)
    print(f"--- Hoàn thành! Dữ liệu sạch đã được lưu vào file: '{output_path}' ---")
    print(f"Số lượng dòng dữ liệu sau khi làm sạch: {len(df):,}")
    
    return df

# Thực thi hàm làm sạch dữ liệu
df_cleaned = clean_and_normalize_data('train.csv', 'train_cleaned.csv')