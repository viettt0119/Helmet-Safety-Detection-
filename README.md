# Phát Hiện Mũ Bảo Hộ (Helmet Safety Detection) 🚧🪖

## Tổng Quan
Helmet Safety Detection là một dự án **thị giác máy tính (Computer Vision)** nhằm phát hiện xem công nhân tại công trường có đang đội **mũ bảo hộ** hay không bằng cách sử dụng mô hình **YOLOv10**.

Hệ thống có thể phân tích hình ảnh và phát hiện hai trường hợp:

- Công nhân **có đội mũ bảo hộ**
- Công nhân **không đội mũ bảo hộ**

Dự án nhằm hỗ trợ **giám sát an toàn lao động** tại công trường và giảm thiểu nguy cơ tai nạn.

---

# Mục Tiêu Dự Án
- Phát hiện công nhân **đội mũ bảo hộ**
- Phát hiện công nhân **không đội mũ bảo hộ**
- Đánh giá hiệu suất mô hình bằng dữ liệu validation
- Xây dựng nền tảng cho hệ thống **giám sát an toàn tự động**

---

# Cấu Trúc Dự Án

```text
Helmet-Safety-Detection
│
├── data/                     # Dataset theo định dạng YOLO
│   ├── train/                # Dữ liệu train
│   ├── valid/                # Dữ liệu validation
│   ├── test/                 # Dữ liệu test
│   ├── data.yaml             # File cấu hình dataset
│   ├── README.dataset.txt
│   └── README.roboflow.txt
│
├── models/                   # Lưu trọng số mô hình
│   └── yolov10n.pt           # Pretrained YOLOv10 model
│
├── runs/                     # Kết quả train và detect
│   ├── detect/
│   ├── helmet_yolov10n/
│   ├── helmet_yolov10n2/
│   ├── helmet_yolov10n3/
│   └── ...
│
├── inference.py              # Script chạy dự đoán (inference)
├── train.py                  # Script train mô hình
├── val.py                    # Script đánh giá mô hình
├── requirements.txt          # Thư viện cần cài
│
└── README.md                 # Tài liệu dự án
```

Lưu ý: Cấu trúc thư mục có thể thay đổi tùy theo cách bạn tổ chức project.

---

# Công Nghệ Sử Dụng

- Python
- YOLOv10
- PyTorch
- OpenCV
- NumPy

---

# Dataset

Dự án sử dụng **dataset phát hiện mũ bảo hộ** với dữ liệu đã được gán nhãn (bounding box).

Các lớp chính thường bao gồm:

- `helmet`
- `no_helmet`

Dataset được tổ chức theo định dạng YOLO gồm:

- images
- labels
- train / validation / test

---

# Mô Hình

Dự án sử dụng **YOLOv10**, một mô hình object detection hiện đại với khả năng:

- phát hiện vật thể **nhanh**
- **độ chính xác cao**
- phù hợp cho **real-time detection**

---

# Cài Đặt

Clone repository:

```bash
git clone https://github.com/viettt0119/Helmet-Safety-Detection-.git
cd Helmet-Safety-Detection-
```

Cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

---

# Validation Mô Hình

Chạy script đánh giá mô hình:

```bash
python val.py
```

Script này dùng để:

- đánh giá hiệu suất mô hình
- tính toán các chỉ số như precision, recall, mAP
- kiểm tra kết quả dự đoán trên tập validation

---

# Kết Quả

Sau khi chạy validation, hệ thống có thể xuất ra:

- Precision
- mAP
- Confusion Matrix

---

# Ứng Dụng

Dự án có thể áp dụng trong:

- Giám sát an toàn tại công trường
- Kiểm tra tuân thủ quy định bảo hộ lao động
- Hệ thống camera thông minh
- Nghiên cứu về thị giác máy tính

---

# Hướng Phát Triển

Một số hướng cải tiến trong tương lai:

- Phát hiện **real-time từ webcam**
- Tích hợp với **camera CCTV**
- Phát hiện nhiều loại **PPE** (áo phản quang, găng tay, khẩu trang)
- Xây dựng giao diện web bằng **Streamlit hoặc Flask**
- Cải thiện dataset và tuning mô hình

---

# Tác Giả

**Viet Tran**

GitHub:  
https://github.com/viettt0119

---

# License

Dự án được phát triển cho **mục đích học tập, nghiên cứu và portfolio**.
