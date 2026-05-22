import xgboost as xgb
import pyarrow.parquet as pq
import gc
import warnings

# Tắt các cảnh báo hệ thống không cần thiết
warnings.filterwarnings('ignore')

class ParquetDataIter(xgb.DataIter):
    """
    Iterator đọc file Parquet theo lô nhỏ (Batch) giúp gộp logic 
    nhiều file và tự động giải phóng RAM để chống lỗi bad_malloc.
    """
    def __init__(self, file_paths, batch_size=50000):
        self.file_paths = file_paths
        self.batch_size = batch_size
        
        # Danh sách cột KHÔNG ĐƯỢC PHÉP đưa vào làm Feature huấn luyện
        self.drop_cols = [
            "ItemCode",           # Định danh sản phẩm
            "Date",               # Thời gian (timestamp)
            "net_qty",            # Biến mục tiêu (Target Variable)
            "wrmsse_weight",      # Trọng số tĩnh cuộc thi
            "xgb_sample_weight"   # Trọng số động (Dùng làm tham số weight riêng)
        ]
        super().__init__()
        self.reset()

    def reset(self):
        self.current_file_idx = 0
        self.pf = None
        self.batch_iter = None
        if self.file_paths:
            self._load_next_file()

    def _load_next_file(self):
        if self.current_file_idx < len(self.file_paths):
            print(f"-> Đang nạp phân đoạn dữ liệu: {self.file_paths[self.current_file_idx]}")
            self.pf = pq.ParquetFile(self.file_paths[self.current_file_idx])
            self.batch_iter = self.pf.iter_batches(batch_size=self.batch_size)
        else:
            self.batch_iter = None

    def next(self, input_data):
        if self.batch_iter is None:
            return 0  # Tín hiệu dừng khi đã duyệt qua hết danh sách file

        try:
            batch = next(self.batch_iter)
            df = batch.to_pandas()
        except StopIteration:
            self.current_file_idx += 1
            self._load_next_file()
            return self.next(input_data)

        # Trích xuất Features (X), Target (y), và Trọng số động (w)
        features = [c for c in df.columns if c not in self.drop_cols]
        X = df[features]
        y = df["net_qty"]
        w = df["xgb_sample_weight"]

        # Nạp trực tiếp khối dữ liệu nhỏ vào cấu trúc của XGBoost
        input_data(data=X, label=y, weight=w)

        # =====================================================================
        # CƠ CHẾ SỬA LỖI TRÀN RAM (CRITICAL FIX FOR BAD_MALLOC):
        # Thu hồi và ép giải phóng bộ nhớ đệm ngay lập tức sau mỗi Batch
        # =====================================================================
        del df, X, y, w, batch
        gc.collect()

        return 1

# =============================================================================
# CẤU HÌNH ĐƯỜNG DẪN FILE CHUẨN XÁC TỪ MÁY CỦA BẠN
# =============================================================================
train_files = [
    r"train_phase1.parquet",
    r"train_phase2.parquet"
]

# Sử dụng chính xác file giả lập mà hệ thống của bạn đã nhận diện thành công
val_files = [
    r"val_phase.parquet" 
]

print("⚡ BƯỚC 1: Khởi tạo luồng xử lý dữ liệu tuần tự...")
train_iter = ParquetDataIter(train_files, batch_size=50000)
val_iter = ParquetDataIter(val_files, batch_size=50000)

print("⚡ BƯỚC 2: Thiết lập cấu trúc QuantileDMatrix (Đang tính toán phân phối)...")
# Hàm này sẽ kích hoạt luồng đọc lặp để gom phân vị mà không giữ toàn bộ data trên RAM
dtrain = xgb.QuantileDMatrix(train_iter)
dval = xgb.QuantileDMatrix(val_iter, ref=dtrain)

# =============================================================================
# THIẾT LẬP THAM SỐ KIỂM SOÁT OVERFITTING VÀ HIỆU NĂNG BỘ NHỚ
# =============================================================================
params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",     # Bắt buộc để tối ưu thuật toán histograms lưu trữ nhẹ
    "learning_rate": 0.05,
    "max_depth": 7,            # Giới hạn độ sâu vừa phải để tránh overfit mẫu panel data
    "subsample": 0.8,          # Lấy mẫu ngẫu nhiên 80% dòng để tăng tính tổng quát
    "colsample_bytree": 0.8,   # Lấy mẫu ngẫu nhiên 80% cột cho mỗi cây
    "min_child_weight": 20     # Ép các cụm SKU thưa phải gộp nhóm, giảm nhiễu mẫu
}

print("⚡ BƯỚC 3: Kích hoạt tiến trình huấn luyện Global Model...")
evals = [(dtrain, "train"), (dval, "val")]

model_global = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=2000,
    evals=evals,
    early_stopping_rounds=50,   # Tự động ngắt nếu tập Val giả lập không giảm RMSE sau 50 cây
    verbose_eval=50
)

# BƯỚC 4: Lưu trữ thành phẩm mô hình cốt lõi
model_global.save_model("Global_Model.json")
print("\n✅ THÀNH CÔNG! File mô hình đã được xuất ra ổ cứng: 'Global_Model.json'")