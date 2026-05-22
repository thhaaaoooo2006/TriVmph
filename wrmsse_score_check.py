import numpy as np
import pandas as pd


def compute_wrmsse(
    df_true: pd.DataFrame, df_pred: pd.DataFrame, df_weights: pd.DataFrame
) -> float:
    """Hàm tính điểm WRMSSE chuẩn cuộc thi tại máy Local.

    Điểm càng thấp càng tốt.
    """
    # 1. CHUẨN HÓA TÊN CỘT ĐỂ TRÁNH LỖI KEYERROR
    df_true = df_true.copy()
    df_pred = df_pred.copy()
    df_weights = df_weights.copy()

    df_true.columns = [col.upper() for col in df_true.columns]
    df_pred.columns = [col.upper() for col in df_pred.columns]
    df_weights.columns = [col.upper() for col in df_weights.columns]

    # Dò tìm cột định danh sản phẩm (SKU / ITEMCODE)
    sku_true = (
        "ITEMCODE"
        if "ITEMCODE" in df_true.columns
        else ("SKU" if "SKU" in df_true.columns else df_true.columns[0])
    )
    sku_pred = (
        "ITEMCODE"
        if "ITEMCODE" in df_pred.columns
        else ("SKU" if "SKU" in df_pred.columns else df_pred.columns[0])
    )
    sku_weight = (
        "SKU"
        if "SKU" in df_weights.columns
        else ("ID" if "ID" in df_weights.columns else df_weights.columns[0])
    )

    # Dò tìm cột lượng bán thực tế và lượng bán dự báo
    qty_true = "NET_QTY" if "NET_QTY" in df_true.columns else "QUANTITY"
    qty_pred = (
        "PREDICTED"
        if "PREDICTED" in df_pred.columns
        else ([col for col in df_pred.columns if "PRED" in col] + [qty_true])[0]
    )

    # Dò tìm cột trọng số kinh doanh
    weight_col = [col for col in df_weights.columns if "WEIGHT" in col][0]

    # Đồng bộ tên cột định danh về tiêu chuẩn chung để xử lý nội bộ
    df_true = df_true.rename(columns={sku_true: "SKU", qty_true: "ACTUAL"})
    df_pred = df_pred.rename(columns={sku_pred: "SKU", qty_pred: "PREDICTED"})
    df_weights = df_weights.rename(
        columns={sku_weight: "SKU", weight_col: "WEIGHT"}
    )

    # 2. CHỐT CHẶN AN TOÀN: XỬ LÝ TRÙNG LẶP VÀ SẮP XẾP THỜI GIAN
    df_true["DATE"] = pd.to_datetime(df_true["DATE"])
    df_pred["DATE"] = pd.to_datetime(df_pred["DATE"])

    df_true = df_true.drop_duplicates(subset=["SKU", "DATE"]).reset_index(
        drop=True
    )
    df_pred = df_pred.drop_duplicates(subset=["SKU", "DATE"]).reset_index(
        drop=True
    )

    df_true = df_true.sort_values(by=["SKU", "DATE"]).reset_index(drop=True)
    df_pred = df_pred.sort_values(by=["SKU", "DATE"]).reset_index(drop=True)

    # Merge bảng Thực tế và bảng Dự báo lại làm một
    df_eval = pd.merge(
        df_true[["SKU", "DATE", "ACTUAL"]],
        df_pred[["SKU", "DATE", "PREDICTED"]],
        on=["SKU", "DATE"],
        how="inner",
    )

    if df_eval.empty:
        raise ValueError(
            "❌ LỖI: Không tìm thấy sự trùng khớp nào về ngày tháng và SKU giữa bảng thực tế và dự báo!"
        )

    # 3. TÍNH RMSSE CHO TỪNG SKU (SCALE THEO GIÁ TRỊ MẢNG NUMPY)
    df_eval["SQUARED_ERROR"] = (
        df_eval["ACTUAL"] - df_eval["PREDICTED"]
    ) ** 2

    sku_mse = (
        df_eval.groupby("SKU", as_index=False)["SQUARED_ERROR"]
        .mean()
        .reset_index(drop=True)
    )

    scale_dict = {}
    for sku, group in df_eval.groupby("SKU"):
        actual_series = group["ACTUAL"].values
        if len(actual_series) > 1:
            diff_sq = np.mean(np.diff(actual_series) ** 2)
            scale_dict[sku] = diff_sq if diff_sq != 0 else 1.0
        else:
            scale_dict[sku] = 1.0

    sku_scale = pd.DataFrame(
        list(scale_dict.items()), columns=["SKU", "SCALE_FACTOR"]
    )

    sku_metrics = pd.merge(sku_mse, sku_scale, on="SKU", how="inner")
    sku_metrics["RMSSE"] = np.sqrt(
        sku_metrics["SQUARED_ERROR"] / sku_metrics["SCALE_FACTOR"]
    )

    # 4. ÁP TRỌNG SỐ WRMSSE WEIGHTS
    sku_final = pd.merge(sku_metrics, df_weights, on="SKU", how="inner")

    total_w = sku_final["WEIGHT"].sum()
    if total_w == 0:
        total_w = 1.0
    sku_final["NORMALIZED_WEIGHT"] = sku_final["WEIGHT"] / total_w

    wrmsse_score = np.sum(
        sku_final["RMSSE"] * sku_final["NORMALIZED_WEIGHT"]
    )

    return float(wrmsse_score)


# ==============================================================================
# KHU VỰC THỬ NGHIỆM: ĐỂ TRỐNG NGOẶC KÉP ĐỂ INPUT FILE CHÍNH THỨC
# ==============================================================================
if __name__ == "__main__":
    print("⏳ Đang chuẩn bị nạp các file dữ liệu chính thức để chấm điểm...")

    # 💡 Mẹo: Nếu file là đuôi .csv thì dùng pd.read_csv(""), nếu là đuôi .parquet thì sửa thành pd.read_parquet("")
    try:
        # 1. ĐIỀN TÊN FILE CHỨA DOANH SỐ THỰC TẾ (TẬP PSEUDO-TEST) VÀO ĐÂY:
        file_actual = "train.csv"
        df_true_official = (
            pd.read_parquet(file_actual) if file_actual else pd.DataFrame()
        )

        # 2. ĐIỀN TÊN FILE CHỨA KẾT QUẢ MÔ HÌNH DỰ BÁO (PREDICTIONS) VÀO ĐÂY:
        file_predicted = "submission_v9_ensemble.csv"
        df_pred_official = (
            pd.read_parquet(file_predicted)
            if file_predicted
            else pd.DataFrame()
        )

        # 3. ĐIỀN TÊN FILE CHỨA TRỌNG SỐ (WRMSSE_WEIGHTS) VÀO ĐÂY:
        file_weights = "wrmsse_weights.csv"
        df_weights_official = (
            pd.read_csv(file_weights) if file_weights else pd.DataFrame()
        )

        # Kiểm tra xem bạn đã điền file chưa, nếu chưa điền sẽ chạy chế độ Demo tự động
        if (
            df_true_official.empty
            or df_pred_official.empty
            or df_weights_official.empty
        ):
            print(
                "\n📢 [CHẾ ĐỘ DEMO]: Bạn chưa điền tên file chính thức vào ngoặc kép."
            )
            print(
                "-> Máy đang tự động tạo dữ liệu mẫu ngẫu nhiên để test kiểm tra hàm..."
            )

            demo_skus = ["SKU-08605", "SKU-15015", "SKU-11413"]
            demo_dates = pd.date_range(start="2025-08-09", periods=5)
            data_true = [
                {"SKU": sku, "Date": d, "Quantity": np.random.randint(1, 10)}
                for sku in demo_skus
                for d in demo_dates
            ]
            df_true_official = pd.DataFrame(data_true)

            data_pred = [
                {
                    "SKU": sku,
                    "Date": d,
                    "Predicted": max(
                        0.0, np.random.randint(1, 10) + np.random.uniform(-1, 1)
                    ),
                }
                for sku in demo_skus
                for d in demo_dates
            ]
            df_pred_official = pd.DataFrame(data_pred)
            df_weights_official = pd.DataFraimport pandas as pd
from wrmsse_scorer import compute_wrmsse_from_raw

# 1. Đọc dữ liệu đầu vào
print("Đang đọc dữ liệu...")
train_df = pd.read_csv("train.csv", low_memory=False)
submission_df = pd.read_csv("submisson_v9_ensemble.csv")

# 2. Tính điểm WRMSSE
# KỊCH BẢN: Đánh giá cục bộ (Local Validation / Pseudo-test)
# Thay đổi train_cutoff, test_start, test_end tương ứng với khoảng thời gian bạn hold-out
print("Đang tính toán WRMSSE...")
score = compute_wrmsse_from_raw(
    train_df=train_df,
    y_pred_df=submission_df,
    train_cutoff="2025-08-08",     # Ngày cuối cùng của tập train
    test_start="2025-08-09",       # Ngày bắt đầu tập validation
    test_end="2025-09-05"          # Ngày kết thúc tập validation (28 ngày sau)
)

print(f"\n=> Điểm WRMSSE của model ensemble là: {score:.4f}")me(
                {"SKU": demo_skus, "wrmsse_weight": [0.5, 0.3, 0.2]}
            )

        # Tiến hành chấm điểm WRMSSE
        score = compute_wrmsse(
            df_true_official, df_pred_official, df_weights_official
        )
        print(f"\n🎉 KẾT QUẢ CHẤM ĐIỂM THÀNH CÔNG!")
        print(f"👉 Điểm số WRMSSE tính được: {score:.5f}")

    except Exception as e:
        print(f"\n❌ Chạy chấm điểm thất bại. Vui lòng kiểm tra lại lỗi:")
        print(f"Lỗi cụ thể: {str(e)}")