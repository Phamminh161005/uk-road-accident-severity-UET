import streamlit as st

st.title("Mức độ nghiêm trọng của tai nạn giao thông đường bộ tại Vương quốc Anh")

st.markdown(
    """
Trang web trực quan hóa dữ liệu tai nạn giao thông có người bị thương tại Great Britain
(England, Wales, Scotland) giai đoạn 2021–2025, và trình bày kết quả phân tích các yếu tố
liên quan đến mức độ nghiêm trọng của tai nạn.

### Mức độ nghiêm trọng được đo thế nào

Mỗi vụ tai nạn được xếp theo người bị thương nặng nhất: tử vong (Fatal), bị thương nặng
(Serious) hoặc bị thương nhẹ (Slight). Chỉ số chính của trang là tỷ lệ KSI (Killed or
Seriously Injured): tỷ lệ số vụ có người tử vong hoặc bị thương nặng trên tổng số vụ.

Trong những năm gần đây, nhiều lực lượng cảnh sát đã chuyển sang hệ thống ghi nhận CRASH, phân loại mức độ
theo loại thương tích thay vì đánh giá của cảnh sát tại hiện trường. Cách ghi mới xếp nhiều
vụ vào mức nặng hơn, nên số liệu giữa các khu vực và các năm không so sánh trực tiếp được.
Trang này dùng mức độ nghiêm trọng đã điều chỉnh do Bộ Giao thông Anh (DfT) cung cấp để đưa
mọi bản ghi về cùng một chuẩn.

### Nguồn dữ liệu

- Road Safety Open Data (STATS19), Department for Transport: bảng vụ tai nạn, phương tiện
  và người bị thương.
- Dữ liệu chỉ gồm các vụ có người bị thương được cảnh sát ghi nhận, nên trang này phân tích
  mức độ nghiêm trọng khi tai nạn đã xảy ra, không phải khả năng xảy ra tai nạn.

### Các trang

- Tổng quan: số vụ, số người thương vong, tỷ lệ KSI và xu hướng theo tháng.
"""
)
