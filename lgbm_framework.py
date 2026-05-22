import pandas as pd
import numpy as np
import lightgbm as lgb
import os
from typing import List, Dict, Any

def get_lgbm_hyperparameters() -> Dict[str, Any]:
    """
    Thiết lập tham số cấu hình LightGBM chống overfitting theo đúng yêu cầu của nhóm
    và định hướng tối ưu từ buổi Workshop.
    """
    return {
        'objective': 'regression_l1',  # Tối ưu theo MAE (L1 Loss) - phù hợp nhất với cấu trúc WRMSSE
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'n_estimators': 500,           # Số lượng cây quyết định tối đa
        'learning_rate': 0.05,         # Tốc độ học (co shrinkage)
        'max_depth': 6,                # Giới hạn độ sâu của cây để chặn học vẹt
        'num_leaves': 63,              # Số nút lá tối đa trong một cây (2^max_depth - 1)
        'random_state': 42,
        'n_jobs': -1,                  # Sử dụng tối đa số nhân CPU để tăng tốc độ tính toán
        'verbose': -1
    }

def train_lgbm_pipeline(
    data_path: str, 
    feature_cols: List[str], 
    target_col: str, 
    params: Dict[str, Any]
) -> Tuple[lgb.LGBMRegressor, pd.DataFrame]:
    """
    Hàm vận hành pipeline đọc dữ liệu, phân cắt chuỗi thời gian nghiêm ngặt,
    huấn luyện mô hình tích hợp Early Stopping và trả về model tối ưu.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ Không tìm thấy file dữ liệu tại: {data_path}. Đợi Thành viên A bàn giao!")
        
    print(f"📦 Đang nạp dữ liệu đặc trưng từ: {data_path}...")
    df = pd.read_parquet(data_path)
    
    # Ép kiểu dữ liệu ngày tháng để đảm bảo màng lọc thời gian hoạt động chuẩn xác
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])
        
    # Nguyên tắc bất biến: Luôn sắp xếp tuần tự chuỗi thời gian trước khi xử lý
    df = df.sort_values(by=['ItemCode', 'Date']).reset_index(drop=True)
    
    print("=== PHÂN HOẠCH TIME-BASED SPLIT NỘI BỘ (CHỐNG RÒ RỈ) ===")
    # Chỉ train đến ngày 08/08/2025 theo đúng yêu cầu nghiêm ngặt của nhóm
    train_mask = df['Date'] <= '2025-08-08'
    val_mask = (df['Date'] >= '2025-08-09') & (df['Date'] <= '2025-09-05')
    
    df_train = df[train_mask].copy()
    df_val = df[val_mask].copy()
    
    X_train, y_train = df_train[feature_cols], df_train[target_col]
    X_val, y_val = df_val[feature_cols], df_val[target_col]
    
    print(f"📊 Kích thước tập huấn luyện (Train): {len(df_train):,} dòng")
    print(f"📊 Kích thước tập kiểm định (Val): {len(df_val):,} dòng")
    print("🚀 Bắt đầu khởi chạy tiến trình huấn luyện LightGBM...")
    
    model = lgb.LGBMRegressor(**params)
    
    # Tích hợp cơ chế Early Stopping: Nếu sau 30 cây liên tiếp sai số trên tập Val không giảm, 
    # máy sẽ tự động ngắt huấn luyện để chặn đứng Overfitting.
    callbacks = [lgb.early_stopping(stopping_rounds=30, verbose=True)]
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=callbacks
    )
    
    # Tạo dự báo thử nghiệm trên tập Val để chiều nay Thành viên B làm phân tích sai số (Residuals)
    df_val['preds_val'] = model.predict(X_val)
    
    return model, df_val[['ItemCode', 'Date', target_col, 'preds_val']]

if __name__ == '__main__':
    print("--- KHỞI ĐỘNG HỆ THỐNG CODE KHUNG LIGHTGBM (THÀNH VIÊN C) ---")
    
    # Định nghĩa sẵn danh sách các đặc trưng mà Thành viên A và B sẽ gộp vào file chiều nay
    FEATURES_TO_USE = [
        'lag_7d', 'lag_14d', 'lag_28d', 
        'rolling_mean_7', 'rolling_mean_28', 'rolling_std_28',
        'Day_of_Week', 'Week_of_Year', 'is_holiday', 'days_to_tet'
    ]
    TARGET = 'Quantity' # Cột sản lượng thực tế đang chạy dưới local của bạn
    
    # Đường dẫn giả định file dữ liệu chiều nay bạn nhận từ A
    DATA_INPUT_PATH = "filtered_data/features_top_active.parquet"
    
    LGBM_PARAMS = get_lgbm_hyperparameters()
    
    print("✔ Hạ tầng cấu hình tham số LightGBM đã sẵn sàng!")
    print(f"✔ Danh sách đặc trưng dự kiến kiểm soát: {FEATURES_TO_USE}")
    print("\n👉 Chiều nay khi có file từ A, bạn chỉ cần chạy lệnh sau để kích hoạt model:")
    print("   model, val_predictions = train_lgbm_pipeline(DATA_INPUT_PATH, FEATURES_TO_USE, TARGET, LGBM_PARAMS)")
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import timedelta

def train_time_series_cv(data_path: str, feature_cols: list, target_col: str, params: dict, n_folds: int = 3):
    """
    HỆ THỐNG KIỂM ĐỊNH CHÉO CHUỖI THỜI GIAN (WALK-FORWARD VALIDATION)
    Chạy n_folds đợt tịnh tiến lùi về quá khứ, mỗi đợt chặn đúng 28 ngày làm đề thi thử.
    """
    print(f"📦 Đang nạp Master Data để chuẩn bị chạy {n_folds} đợt kiểm thử...")
    df = pd.read_parquet(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['ItemCode', 'Date']).reset_index(drop=True)
    
    # Mốc thời gian chốt chặn cuối cùng của cuộc thi (ngày cuối cùng trong tập lịch sử)
    anchor_date = pd.to_datetime('2025-09-05')
    
    fold_maes = []
    
    print("\n=== BẮT ĐẦU VẬN HÀNH MẠCH VÒNG KIỂM THỬ ĐA ĐỢT ===")
    
    for fold in range(n_folds):
        print(f"\n--- 🔄 ĐANG CHẠY ĐỢT {fold + 1}/{n_folds} ---")
        
        # Tính toán động ranh giới thời gian cho từng đợt (mỗi đợt dịch lùi 28 ngày)
        val_end_date = anchor_date - timedelta(days=fold * 28)
        val_start_date = val_end_date - timedelta(days=27)
        train_end_date = val_start_date - timedelta(days=1)
        
        print(f"📅 Tập Train: Khởi đầu ──> {train_end_date.strftime('%Y-%m-%d')}")
        print(f"📅 Tập Validation: {val_start_date.strftime('%Y-%m-%d')} ──> {val_end_date.strftime('%Y-%m-%d')} (Đúng 28 ngày)")
        
        # Tạo màng lọc Mask logic an toàn cho từng đợt
        train_mask = df['Date'] <= train_end_date
        val_mask = (df['Date'] >= '2020-11-17') & (df['Date'] >= val_start_date) & (df['Date'] <= val_end_date)
        
        df_train = df[train_mask]
        df_val = df[val_mask]
        
        # Bọc lót nếu tập dữ liệu quá khứ bị cạn kiệt (dành cho các đợt lùi quá sâu)
        if len(df_train) == 0 or len(df_val) == 0:
            print(f"⚠️ Không đủ dữ liệu cho Đợt {fold + 1}, tiến hành ngắt mạch vòng.")
            break
            
        X_train, y_train = df_train[feature_cols], df_train[target_col]
        X_val, y_val = df_val[feature_cols], df_val[target_col]
        
        # Khởi tạo mô hình riêng biệt cho từng đợt để tránh học vẹt chồng chéo
        model = lgb.LGBMRegressor(**params)
        callbacks = [lgb.early_stopping(stopping_rounds=30, verbose=False)]
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=callbacks
        )
        
        # Chấm điểm sai số cho đợt hiện tại
        preds = model.predict(X_val)
        fold_mae = np.mean(np.abs(y_val - preds))
        fold_maes.append(fold_mae)
        
        print(f"📉 Điểm MAE của Đợt {fold + 1}: {fold_mae:.4f}")
        
    # Tổng kết hiệu suất của toàn bộ hệ thống huấn luyện
    mean_cv_mae = np.mean(fold_maes)
    print("\n==================================================")
    print(f"🏆 KẾT LUẬN HIỆU SUẤT KIỂM ĐỊNH CHÉO ({len(fold_maes)} ĐỢT):")
    for i, mae in enumerate(fold_maes):
        print(f"   • Đợt {i+1}: MAE = {mae:.4f}")
    print(f"🚀 ĐIỂM MAE CV TRUNG BÌNH CHỐT HẠ: {mean_cv_mae:.4f}")
    print("==================================================")
    
    return mean_cv_mae