# TriVmph
# Mô hình sử dụng phương pháp phân khúc sản phẩm theo mức độ hoạt động, kết hợp giữa **Trung bình trượt có trọng số (Weighted Moving Average)**, **San phẳng mũ (Exponential Smoothing - ETS)** và **Điều chỉnh mùa vụ theo ngày trong tuần (Day of Week - DoW)**.

Hệ thống được thiết kế tối ưu để dự báo chuỗi thời gian thưa (sparse time series) và xử lý hiệu quả hành vi mua hàng đặc thù của thị trường (ví dụ: không phát sinh doanh số vào Chủ Nhật).

---

##  Chiến lược & Kiến trúc Mô hình

Quy trình dự báo được thực thi xuyên suốt qua các giai đoạn cốt lõi sau:

### 1. Tiền xử lý dữ liệu (Data Preprocessing)
* **Tính tổng doanh số ròng hàng ngày (Daily Net Sales):** Gom nhóm dữ liệu theo ngày và mã sản phẩm (`ItemCode`).
* **Xử lý hàng trả lại (Returns Processing):** Giới hạn giá trị dưới tại `0` (`clip(lower=0)`) cho lượng sản phẩm bán ra mỗi ngày trên từng SKU, loại bỏ nhiễu do lượng hàng hoàn trả gây ra âm doanh số ròng.

### 2. Phân khúc SKU & Chiến lược Dự báo (SKU Segmentation)
Mô hình phân loại toàn bộ hệ thống $15,972$ SKUs dựa trên lịch sử hoạt động trong vòng 90 ngày gần nhất để áp dụng các thuật toán tương ứng:

| Phân khúc SKU | Điều kiện định nghĩa | Chiến lược dự báo |
| :--- | :--- | :--- |
| **Inactive (Không hoạt động)** | Không có doanh số trong 90 ngày qua | Dự báo cố định bằng `0`. |
| **Very Sparse (Cực kỳ thưa)** | Có dưới 3 ngày phát sinh doanh số trong 90 ngày | Lấy tỷ lệ trung bình 90 ngày và áp dụng mức chiết khấu mạnh (`0.5`) để tránh dự báo quá mức (overprediction). |
| **Sparse (Thưa)** | Có từ 3 đến dưới 5 ngày hoạt động trong 90 ngày | Sử dụng tỷ lệ trung bình của cửa sổ 90 ngày (`rate_90`). |
| **Moderate (Hoạt động vừa)** | Có dưới 5 ngày hoạt động trong 28 ngày, nhưng $\ge 5$ ngày trong 56 ngày | Phối hợp trọng số giữa cửa sổ trung hạn và dài hạn: <br> $0.6 \times \text{rate\_56} + 0.4 \times \text{rate\_90}$ |
| **Active (Hoạt động mạnh)** | Có từ 5 ngày phát sinh doanh số trở lên trong 28 ngày | Phối hợp trọng số ưu tiên dữ liệu gần nhất: <br> $0.5 \times \text{rate\_28} + 0.3 \times \text{rate\_56} + 0.2 \times \text{rate\_90}$ |
| **Top High-Volume** | Thuộc nhóm 500 SKUs có tổng doanh số lớn nhất | Áp dụng **San phẳng mũ (ETS - Holt-Winters style)** trên chuỗi ngày làm việc (180 ngày gần nhất) để bắt kịp xu hướng (trend). Kết quả cuối được blend với tỷ lệ: <br> $0.6 \times \text{rate\_ETS} + 0.4 \times \text{rate\_MA}$ |

### 3. Điều chỉnh Hệ số Mùa vụ (DoW Multipliers)
* Tính toán hệ số tác động của từng ngày trong tuần (Từ Thứ 2 đến Thứ 7) dựa trên tỷ lệ đóng góp của ngày đó so với mức trung bình của toàn bộ các ngày làm việc lịch sử.
* **Xử lý ngày Chủ Nhật:** Gán cứng hệ số bằng `0.0` do dữ liệu lịch sử cho thấy hầu như không có hoạt động kinh doanh vào ngày này.

### 4. Thiết lập Ma trận Dự báo (Forecast Horizon)
* **Khung thời gian dự báo:** 56 ngày liên tiếp (từ `2025-09-06` đến `2025-10-31`).
* **Tích toán tử ngoài (Outer Product):** Dự báo thô được thiết lập bằng tích toán tử ngoài giữa Vector tỷ lệ baseline của SKU và Vector hệ số ngày trong tuần (DoW). Kết quả được làm tròn về số nguyên không âm.

---

##  Cấu trúc thư mục yêu cầu

Để script chạy xuyên suốt không lỗi, thư mục làm việc cần đảm bảo cấu trúc tối thiểu sau:

```text
├── train.csv               # Dữ liệu bán hàng lịch sử (chứa các cột Date, ItemCode, Quantity)
├── sample_submission.csv   # File mẫu nộp bài (chứa cột id định dạng SKU_validation hoặc SKU_evaluation)
├── forecast_script.py      # File mã nguồn chính chứa đoạn code trên
└── README.md               # File hướng dẫn này
