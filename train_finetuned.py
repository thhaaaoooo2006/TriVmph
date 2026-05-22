import xgboost as xgb
import pyarrow.parquet as pq

class ParquetDataIter(xgb.DataIter):
    """
    Iterator đọc file Parquet theo lô nhỏ (Batch).
    Cấu trúc giữ nguyên hoàn toàn như Bước A.
    """
    def __init__(self, file_paths, batch_size=250000):
        self.file_paths = file_paths
        self.batch_size = batch_size
        
        # Cột loại bỏ (Giữ nguyên tuyệt đối như Bước A để không sai Schema)
        self.drop_cols = [
            "ItemCode",           
            "Date",               
            "net_qty",            
            "wrmsse_weight",      
            "xgb_sample_weight"   
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
            print(f"Đang đọc dữ liệu từ: {self.file_paths[self.current_file_idx]}")
            self.pf = pq.ParquetFile(self.file_paths[self.current_file_idx])
            self.batch_iter = self.pf.iter_batches(batch_size=self.batch_size)
        else:
            self.batch_iter = None

    def next(self, input_data):
        if self.batch_iter is None:
            return 0 

        try:
            batch = next(self.batch_iter)
            df = batch.to_pandas()
        except StopIteration:
            self.current_file_idx += 1
            self._load_next_file()
            return self.next(input_data)

        features = [c for c in df.columns if c not in self.drop_cols]
        X = df[features]
        y = df["net_qty"]
        w = df["xgb_sample_weight"]

        input_data(data=X, label=y, weight=w)
        return 1

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN FILE (CHỈ DÙNG PHASE 2)
# ==========================================
# CHÚ Ý: Điểm khác biệt lớn nhất nằm ở đây
train_files = [
    r"train_phase2.parquet"
]

val_files = [
    r"val_phase.parquet" 
]

print("1. Khởi tạo Iterator cho Fine-tuned Model...")
train_iter = ParquetDataIter(train_files, batch_size=250000)
val_iter = ParquetDataIter(val_files, batch_size=250000)

print("2. Đóng gói dữ liệu vào QuantileDMatrix...")
dtrain = xgb.QuantileDMatrix(train_iter)
dval = xgb.QuantileDMatrix(val_iter, ref=dtrain)

# ==========================================
# THIẾT LẬP THAM SỐ FINE-TUNED MODEL
# ==========================================
params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",     
    "learning_rate": 0.03,     # Giảm learning rate để mô hình học "từ tốn" hơn trên dữ liệu mới
    "max_depth": 5,            # GIẢM XUỐNG 5 (Thay vì 8 như Global) để tránh học vẹt
    "subsample": 0.7,          # Lấy ngẫu nhiên ít dòng hơn mỗi cây
    "colsample_bytree": 0.7,   
    "min_child_weight": 15     
}

print("3. Bắt đầu huấn luyện Fine-tuned Model...")
evals = [(dtrain, "train"), (dval, "val")]

model_finetuned = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=1500,      # Giảm số lượng vòng lặp tối đa
    evals=evals,
    early_stopping_rounds=50,
    verbose_eval=50
)

# 4. Lưu mô hình
model_finetuned.save_model("FineTuned_Model.json")
print("HOÀN TẤT! Mô hình Fine-Tuned đã được lưu thành 'FineTuned_Model.json'.")