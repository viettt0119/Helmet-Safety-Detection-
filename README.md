# Helmet Safety Detection 🚧🪖

Hệ thống **phát hiện công nhân không đội mũ bảo hộ** dựa trên **YOLOv10n**, đóng gói theo
Clean Architecture: thư viện Python + **CLI** + **REST API (FastAPI)** + **Docker**, có bộ kiểm thử
tự động đầy đủ.

> Phiên bản 2.0 — xem [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) cho báo cáo kiểm toán,
> sơ đồ kiến trúc và luồng dữ liệu.

## Tính năng

* Phát hiện 3 lớp `person`, `helmet`, `head` (đầu trần) và **suy ra trạng thái từng công nhân**:
  `compliant` (có mũ) · `violation` (không mũ) · `unknown` (không thấy đầu), kèm trạng thái tổng
  `safe / unsafe / inconclusive / empty` và tỉ lệ tuân thủ.
* Đầu vào: ảnh, thư mục ảnh, video, webcam, RTSP/HTTP stream.
* Đầu ra: JSON, ảnh/video đã vẽ khung, exit code `2` khi có vi phạm (tích hợp cron/CI/pipeline).
* REST API: `/v1/detections` (JSON), `/v1/detections/annotated` (JPEG), health/readiness,
  OpenAPI tại `/docs`.
* Train / evaluate lại model bằng một lệnh, tự tải pretrained weights (retry + timeout).
* Cấu hình 100% qua biến môi trường `HELMET_*` / `.env`, được validate bằng pydantic.

## Kết quả model (`models/helmet_yolov10n_best.pt`, YOLOv10n, 50 epoch, imgsz 640)

Đo lại bằng `helmet-safety evaluate` trên CPU:

| Tập | Precision | Recall | mAP50 | mAP50-95 |
|-----|-----------|--------|-------|----------|
| test (109 ảnh, 320 đối tượng) | 0.711 | 0.869 | **0.854** | 0.459 |

mAP50-95 theo lớp (test): head 0.389 · helmet 0.499 · person 0.489.

## Dataset

[Safety Helmet Dataset (Roboflow, CC BY 4.0)](https://universe.roboflow.com/dataperson/safety-helmet-dataset-uvh1t/dataset/1)
— 1088 ảnh 416×416, định dạng YOLO, 3 lớp: `head` (0), `helmet` (1), `person` (2).
Chia: train 761 · valid 218 · test 109.

## Cài đặt

Yêu cầu: Python ≥ 3.12 và [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/viettt0119/Helmet-Safety-Detection-.git
cd Helmet-Safety-Detection-
uv sync                 # tạo .venv và cài toàn bộ dependency theo uv.lock
cp .env.example .env    # (tuỳ chọn) chỉnh cấu hình
```

> Không dùng uv? `python -m venv .venv && . .venv/bin/activate && pip install -e .`
> (Windows: `.venv\Scripts\activate`).

## Sử dụng — CLI

```bash
uv run helmet-safety --help

# Ảnh / thư mục ảnh -> in kết quả, lưu ảnh annotate + JSON vào outputs/
uv run helmet-safety predict data/test/images -o outputs
uv run helmet-safety predict path/to/image.jpg --json
uv run helmet-safety predict path/to/image.jpg --show            # mở cửa sổ

# Video / webcam / RTSP
uv run helmet-safety predict video.mp4 -o outputs                # ghi outputs/video_annotated.mp4
uv run helmet-safety predict 0 --show                            # webcam, nhấn q/ESC để thoát
uv run helmet-safety predict rtsp://user:pass@cam/stream --max-frames 500 --json

# Dùng trong pipeline: exit code 2 nếu có người không đội mũ
uv run helmet-safety predict snapshot.jpg --fail-on-violation

# Đánh giá model
uv run helmet-safety evaluate --split test

# Train lại (tự tải yolov10n.pt nếu thiếu)
uv run helmet-safety train --epochs 50 --batch 8 --name my_run
# -> runs/my_run/weights/best.pt ; dùng bằng HELMET_MODEL__WEIGHTS_PATH=runs/my_run/weights/best.pt

uv run helmet-safety info               # weights, device, class đang dùng
```

Các script cũ vẫn chạy được (shim gọi CLI mới): `python train.py`, `python val.py`,
`python inference.py` (mặc định mở webcam).

## Sử dụng — REST API

```bash
uv run helmet-safety serve              # http://localhost:8000/docs
```

```bash
curl -F "file=@data/test/images/person-535-_jpg.rf.629cf0b56b737d6b9833694eb7a50542.jpg" \
     http://localhost:8000/v1/detections
```

```json
{
  "status": "unsafe",
  "image_width": 416,
  "image_height": 416,
  "inference_ms": 44.11,
  "summary": {"workers": 1, "compliant": 0, "violations": 1, "unknown": 0,
              "compliance_rate": 0.0, "counts": {"head": 1, "helmet": 0, "person": 1}},
  "workers": [{
    "status": "violation",
    "box": {"x1": 152.93, "y1": 106.96, "x2": 256.07, "y2": 358.29},
    "person": {"label": "person", "confidence": 0.8405, "box": {"x1": 152.93, "y1": 106.96, "x2": 256.07, "y2": 358.29}},
    "helmet": null,
    "head": {"label": "head", "confidence": 0.6134, "box": {"x1": 190.58, "y1": 116.62, "x2": 237.64, "y2": 168.03}}
  }],
  "detections": [
    {"label": "person", "confidence": 0.8405, "box": {"x1": 152.93, "y1": 106.96, "x2": 256.07, "y2": 358.29}},
    {"label": "head", "confidence": 0.6134, "box": {"x1": 190.58, "y1": 116.62, "x2": 237.64, "y2": 168.03}}
  ]
}
```

| Endpoint | Mô tả |
|----------|-------|
| `GET /health/live` | Liveness |
| `GET /health/ready` | Readiness (503 cho tới khi model đã nạp) |
| `GET /v1/model` | Thông tin model |
| `POST /v1/detections` | multipart `file` → JSON báo cáo |
| `POST /v1/detections/annotated?quality=90` | multipart `file` → ảnh JPEG; header `X-Safety-Status`, `X-Safety-Violations`… |

Lỗi luôn có dạng `{"error": {"code", "message", "request_id"}}` với mã HTTP 413 (file quá lớn),
415 (sai định dạng), 422 (dữ liệu không hợp lệ), 503 (thiếu model), 504 (quá thời gian suy luận).

## Docker

```bash
docker compose up --build -d            # API: http://localhost:8000/docs
docker compose run --rm predict         # batch predict data/test/images -> outputs/
# hoặc
docker build -t helmet-safety .                                   # CPU (mặc định)
docker build --build-arg TORCH_VARIANT=cuda -t helmet-safety:cuda .   # giữ torch CUDA
docker run --rm -p 8000:8000 helmet-safety
docker run --rm -v "$PWD/data/test/images:/in:ro" helmet-safety predict /in --json
```

Image chạy non-root, filesystem read-only (compose), có `HEALTHCHECK`, dùng OpenCV headless.

## Cấu hình

Mọi tham số nằm trong [.env.example](.env.example) (tiền tố `HELMET_`, `__` cho mục lồng nhau).
Quan trọng nhất:

| Biến | Mặc định | Ý nghĩa |
|------|----------|---------|
| `HELMET_MODEL__WEIGHTS_PATH` | `models/helmet_yolov10n_best.pt` | Model dùng cho suy luận |
| `HELMET_MODEL__DEVICE` | `auto` | `auto` / `cpu` / `mps` / `cuda:0` / `0,1` (tự lùi về CPU nếu không có GPU) |
| `HELMET_POLICY__HEAD_CONFIDENCE` | `0.35` | Ngưỡng tin cậy cho đầu trần (tương tự `PERSON_`, `HELMET_`) |
| `HELMET_POLICY__HEAD_REGION_RATIO` | `0.5` | Phần trên của khung người dùng để tìm đầu/mũ |
| `HELMET_API__MAX_UPLOAD_MB` | `10` | Giới hạn upload |
| `HELMET_API__INFERENCE_TIMEOUT_S` | `30` | Timeout suy luận (504) |
| `HELMET_LOG_JSON` | `false` | Log JSON (bật trong production) |

## Phát triển & kiểm thử

```bash
make install        # uv sync --frozen
make check          # ruff + mypy --strict + pytest (194 test)
make test-fast      # bỏ qua test dùng model thật (@slow)
make cov            # coverage (~95%)
```

* `tests/unit` — domain (hình học box, policy, luật ghép mũ/đầu), use case (fake detector),
  config, device, weights (mock HTTP bằng respx: retry/timeout/atomic), dataset, OpenCV, adapter YOLO.
* `tests/integration` — API end-to-end (TestClient), CLI (Typer runner), và E2E với model thật trên
  ảnh test (kể cả `evaluate` đo mAP).

## Cấu trúc dự án

```text
src/helmet_safety/
├── domain/          # entity, policy, luật nghiệp vụ — không phụ thuộc framework
├── application/     # use case + ports (Protocol)
├── infrastructure/  # YOLO, OpenCV, config, logging, tải weights, DI container
└── interfaces/      # FastAPI + Typer CLI
tests/{unit,integration}/
```

Chi tiết: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Tác giả

**Viet Tran** — https://github.com/viettt0119

## License

Dự án được phát triển cho **mục đích học tập, nghiên cứu và portfolio** (chưa gắn license mã nguồn).
Dataset: CC BY 4.0 (Roboflow). Ultralytics YOLO được phát hành theo AGPL-3.0 — cần tuân thủ khi
phân phối/triển khai thương mại.
