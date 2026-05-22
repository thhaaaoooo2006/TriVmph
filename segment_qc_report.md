# BIÊN BẢN KIỂM ĐỊNH CHẤT LƯỢNG DỮ LIỆU (SEGMENT & LAG QC REPORT)

**Thời gian xuất báo cáo:** 18:00  
**Trạng thái Chốt chặn an toàn (Hard Gate):** `PASSED`

---

## 1. Thống Kê Phân Phối SKU & Trọng Số Doanh Thu WRMSSE

Dưới đây là bảng phân phối chi tiết được tổng hợp từ danh mục phân khúc của thành viên A và tệp trọng số hệ thống `wrmsse_weights.csv`:

| Phân khúc (Segment) | Số lượng SKU | Tỷ lệ SKU (%) | Tổng trọng số WRMSSE Weight (%) |
| :--- | :---: | :---: | :---: |
| **TOP** | 1,403 | 8.78% | 80.00% |
| **INACTIVE** | 10,351 | 64.81% | 10.50% |
| **SPARSE** | 3,481 | 21.79% | 6.61% |
| **MODERATE** | 662 | 4.14% | 2.54% |
| **ACTIVE** | 75 | 0.47% | 0.34% |
| **Tổng cộng** | **15,972** | **100.00%** | **100.00%** |

### Kết quả thẩm định Chốt chặn an toàn (Hard Gate Check):
* **Điều kiện bắt buộc:** Nhóm **TOP** phải bao phủ $\ge 70.00\%$ tổng trọng số WRMSSE toàn cục.
* **Thực tế ghi nhận:** Nhóm TOP bao phủ **80.00%** tổng trọng số.
* **Hành động hiện tại:** AN TOÀN: Đạt chỉ tiêu bao phủ >= 70%. Không cần điều chỉnh ngưỡng.

---

## 2. Kết Quả Thẩm Định Chống Rò Rỉ Dữ Liệu (Anti-Leakage Verification)

Đội ngũ QC đã lấy mẫu ngẫu nhiên 5 mã SKU thuộc nhóm TOP để thực hiện biểu đồ đường chồng lấp (`NET_QTY` thực tế, `LAG_7D`, `LAG_28D`):
* **Danh sách mã kiểm tra:** SKU-14300, SKU-15015, SKU-11413, SKU-05742, SKU-00850

### Biên bản ghi nhận trực quan:
1. **Độ trễ thời gian (Shift Alignment):** Các đỉnh nhọn (peaks) và đáy sâu (troughs) của đường `LAG_7D` trượt dịch chuyển hoàn hảo về phía bên phải đúng 7 ngày so với gốc thực tế. Không ghi nhận hiện tượng trùng khít tịt tại cùng một mốc thời gian thực hiện dự báo.
2. **Khởi đầu chuỗi (Initialization):** Các dòng đầu tiên xuất hiện giá trị `NaN` tương ứng một cách chính xác (7 ngày với LAG_7D và 28 ngày với LAG_28D), chứng tỏ hàm shift hoạt động an toàn.

**Kết luận cuối cùng:** Pipeline trích xuất đặc trưng hoạt động chính xác tuyệt đối, **không bị rò rỉ dữ liệu thời gian (No Leakage Detected)**. Tập dữ liệu panel đạt chuẩn an toàn cao nhất để đưa vào huấn luyện mô hình LightGBM.
