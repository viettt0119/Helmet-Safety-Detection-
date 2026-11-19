# Kiến trúc & Báo cáo kiểm toán (v2.0)

## 1. Kiểm toán hiện trạng (v1 — 3 script `train.py`, `val.py`, `inference.py`)

| # | Vấn đề | Mức độ | Chi tiết | Xử lý ở v2 |
|---|--------|--------|----------|------------|
| 1 | **Đường dẫn weights mặc định sai** | Cao | `runs/helmet_yolov10n/weights/best.pt` không tồn tại (run đó hỏng, chỉ có `args.yaml`). Chương trình rơi vào fallback "best.pt mới nhất theo mtime" — sau khi `git clone`, mtime ngẫu nhiên nên có thể chọn **nhầm model của run lỗi** (`helmet_yolov10n8`, chỉ train 1 epoch, mAP50 = 0.24). `val.py` còn có thể rơi xuống `yolov10n.pt` COCO (không có class helmet/head) mà chỉ in cảnh báo. | Copy model tốt nhất (`helmet_yolov10n11`, 50 epoch) thành `models/helmet_yolov10n_best.pt`; `WeightsResolver` (Chain of Responsibility) có thứ tự rõ ràng, báo lỗi `ModelNotFoundError` thay vì âm thầm dùng model sai. |
| 2 | **`data.yaml` dùng `path: data` tương đối** | Cao | Ultralytics phân giải đường dẫn tương đối theo `datasets_dir` toàn cục → hay lỗi "dataset not found" khi chạy từ thư mục khác / máy khác. | `infrastructure/dataset.py` validate `data.yaml` (nc ↔ names, split tồn tại) và sinh bản sao đường dẫn tuyệt đối. |
| 3 | **Code trùng lặp (DRY)** | Trung bình | `resolve_device`, `find_latest_best`, logic fallback weights bị copy ở cả 3 file. | Gom về `infrastructure/device.py`, `infrastructure/weights.py`. |
| 4 | **Không có logic nghiệp vụ "tuân thủ"** | Cao | Script chỉ đếm số box `person/helmet/head`; không trả lời câu hỏi chính: *công nhân nào không đội mũ?* | `domain/services.py::ComplianceAnalyzer` ghép mũ/đầu trần vào từng người (vùng đầu của box người, gán tham lam theo độ tin cậy, mỗi mũ chỉ gán 1 lần) → trạng thái `compliant / violation / unknown`, tổng hợp `safe / unsafe / inconclusive / empty`. |
| 5 | **Tight coupling** | Trung bình | Logic vẽ, lọc ngưỡng, I/O OpenCV, YOLO trộn trong một hàm (`draw_detections`). Không thể test mà không có model + GUI. | Clean Architecture: domain thuần Python, use case phụ thuộc *ports* (Protocol), adapter YOLO/OpenCV ở infrastructure, DI qua `Container`. |
| 6 | **Thiếu xử lý lỗi / validate đầu vào** | Cao | Không kiểm tra ảnh hỏng, file sai định dạng, ngưỡng ngoài [0,1], imgsz không chia hết 32, webcam mất kết nối… | `pydantic-settings` validate cấu hình; `validate_image`, giới hạn kích thước upload, chống decompression-bomb (40 MP), reconnect stream, hệ thống lỗi có kiểu + global error handler. |
| 7 | **Tải weights không an toàn** | Trung bình | `urlretrieve` không timeout, không retry, ghi trực tiếp → file hỏng nếu đứt mạng giữa chừng. | `WeightsDownloader`: httpx streaming, timeout, retry exponential back-off + jitter (tenacity), chỉ HTTPS, ghi atomic (`.part` → rename), kiểm tra kích thước tối thiểu. |
| 8 | **Logging bằng `print`** | Thấp | Không có mức log, không có ngữ cảnh. | `structlog`: console đẹp khi dev, JSON khi production, `request_id` cho mọi request. |
| 9 | **Không có test, không CI, không đóng gói** | Cao | — | 194 test (unit + integration + E2E với model thật), coverage 96%, ruff + mypy strict, GitHub Actions, Dockerfile multi-stage, docker-compose. |
| 10 | **Repo rác** | Thấp | `.idea/`, `labels.cache` (chứa đường dẫn Windows) bị commit; file có BOM UTF-8. | Xoá khỏi git, thêm `.gitignore`, bỏ BOM. |
| 11 | **README sai lệch dataset** | Thấp | README ghi class `helmet / no_helmet`, thực tế là `head / helmet / person`. | README viết lại theo đúng dữ liệu. |
| 12 | **Không có cách tích hợp hệ thống** | Trung bình | Chỉ chạy được bằng cửa sổ OpenCV. | REST API FastAPI (JSON + ảnh annotate), CLI có `--json`, exit code `2` khi có vi phạm (dùng cho cron/CI/camera pipeline). |

### Cập nhật công nghệ

| Hạng mục | Trước | Sau |
|----------|-------|-----|
| Quản lý package | `requirements.txt` (chỉ `>=`) | `pyproject.toml` (PEP 621) + **uv** + `uv.lock` (tái lập 100%) |
| Python | không ràng buộc | `>=3.12` (đã kiểm thử 3.13), cú pháp `type X = ...`, `StrEnum`, `Self` |
| Cấu hình | argparse rải rác | **pydantic v2 / pydantic-settings**, biến môi trường `HELMET_*`, `.env` |
| API | — | **FastAPI** + lifespan + anyio thread offloading + CapacityLimiter + timeout |
| CLI | argparse | **Typer** (sub-command `predict/train/evaluate/serve/info/download-weights`) |
| Logging | `print` | **structlog** (JSON) |
| Retry/timeout | — | **tenacity** + **httpx** |
| Chất lượng | — | **ruff** (lint+format), **mypy --strict**, **pytest 9** + pytest-cov + respx |
| Đóng gói | — | **Docker multi-stage**, non-root, read-only FS, HEALTHCHECK, OpenCV headless, torch CPU |

## 2. Kiến trúc phân tầng

```
            ┌──────────────────────── interfaces ────────────────────────┐
            │  api/ (FastAPI: routes, schemas, errors, middleware)       │
            │  cli/ (Typer)            serialization.py (DTO pydantic)   │
            └───────────────┬────────────────────────────────────────────┘
                            │ gọi use case (qua Container - DI)
            ┌───────────────▼──────────── application ───────────────────┐
            │  AnalyzeImageUseCase, AnalyzeStreamUseCase                 │
            │  TrainModelUseCase, EvaluateModelUseCase                   │
            │  ports.py  (Protocol: ObjectDetector, ImageAnnotator,      │
            │             FrameSource, FrameSink, ModelTrainer, ...)     │
            └───────┬───────────────────────────────────▲────────────────┘
                    │ dùng                               │ implement
            ┌───────▼────── domain ──────────┐  ┌────────┴──── infrastructure ─────────────┐
            │ entities (BoundingBox,         │  │ yolo.py      (Ultralytics adapters)      │
            │   Detection, SafetyReport…)    │  │ imaging.py   (OpenCV codec/annotator/    │
            │ policies (ConfidencePolicy)    │  │               video source & sinks)      │
            │ services (ComplianceAnalyzer)  │  │ weights.py   (resolver + downloader)     │
            │ errors                         │  │ dataset.py   config.py  logging.py       │
            │ — KHÔNG phụ thuộc framework —  │  │ device.py    container.py (composition)  │
            └────────────────────────────────┘  └──────────────────────────────────────────┘
```

Quy tắc phụ thuộc: `interfaces → application → domain`; `infrastructure` implement các port của
`application`. Domain không import numpy/cv2/torch/fastapi.

### Design patterns

* **Ports & Adapters / Dependency Inversion** — use case chỉ biết `ObjectDetector` (Protocol);
  `UltralyticsDetector` là adapter; test dùng `FakeDetector`.
* **Dependency Injection / Composition Root** — `infrastructure/container.py`.
* **Strategy** — `ConfidencePolicy` (ngưỡng theo class), `ComplianceAnalyzer` (luật ghép).
* **Chain of Responsibility** — `WeightsResolver` (weights chính → best.pt mới nhất → fallback).
* **Factory** — `create_app()`, `model_factory` trong adapter YOLO.
* **Value Object** — các entity `frozen dataclass(slots=True)` bất biến, tự validate.

## 3. Luồng dữ liệu (Data Flow)

```
 ảnh (upload / file / webcam / RTSP / video)
        │
        ▼
 decode_image / OpenCVFrameSource ──► validate_image (uint8, HxWx3, ≥16px, ≤40MP)
        │
        ▼
 ObjectDetector.detect (YOLOv10n, conf=0.10, NMS-free)  ── lock, chạy trong thread pool
        │  list[Detection]   (bỏ class lạ, clamp toạ độ)
        ▼
 ConfidencePolicy.apply  (person/helmet/head ≥ 0.35 mặc định)
        │
        ▼
 ComplianceAnalyzer.analyze
   ├─ với mỗi person (theo conf giảm dần): tìm helmet & head trong 50% trên của box
   │    (≥50% diện tích nằm trong vùng) → COMPLIANT / VIOLATION / UNKNOWN
   ├─ head lẻ (không thuộc ai)   → VIOLATION   (ảnh cận mặt)
   └─ helmet lẻ                  → COMPLIANT   (trừ khi trùng head có conf cao hơn)
        │
        ▼
 SafetyReport (status: SAFE / UNSAFE / INCONCLUSIVE / EMPTY, compliance_rate)
        │
        ├──► JSON (SafetyReportDTO)               – API /v1/detections, CLI --json
        └──► OpenCVAnnotator → JPEG / MP4 / cửa sổ – API /v1/detections/annotated, CLI -o / --show
```

## 4. Cây thư mục

```
Helmet-Safety-Detection-/
├── pyproject.toml            # metadata, dependencies, ruff/mypy/pytest config
├── uv.lock                   # lockfile tái lập
├── .env.example              # toàn bộ biến cấu hình HELMET_*
├── Dockerfile                # multi-stage, non-root, CPU torch, OpenCV headless
├── docker-compose.yml
├── Makefile
├── .github/workflows/ci.yml  # lint + mypy + test (3.12, 3.13) + docker smoke test
├── data/                     # dataset YOLO (Roboflow, CC BY 4.0): train/valid/test + data.yaml
├── models/
│   ├── yolov10n.pt           # pretrained COCO (dùng để fine-tune)
│   └── helmet_yolov10n_best.pt  # model đã train (50 epoch) dùng cho inference
├── runs/                     # lịch sử train cũ (giữ lại để tham khảo)
├── docs/ARCHITECTURE.md
├── train.py / val.py / inference.py   # shim tương thích ngược → CLI mới
├── src/helmet_safety/
│   ├── __init__.py  __main__.py
│   ├── domain/        entities.py  policies.py  services.py  errors.py
│   ├── application/   ports.py  analyze.py  training.py
│   ├── infrastructure/config.py  logging.py  device.py  weights.py  dataset.py
│   │                  yolo.py  imaging.py  container.py
│   └── interfaces/    serialization.py
│       ├── api/       app.py  routes.py  schemas.py  errors.py  dependencies.py
│       └── cli/       main.py
└── tests/
    ├── conftest.py               # fakes: FakeDetector, ListFrameSource, RecordingSink
    ├── unit/                     # domain, use case, config, device, weights, dataset, imaging, yolo adapter
    └── integration/              # API (FastAPI TestClient), CLI (Typer), E2E model thật (@slow)
```

## 5. Ghi chú vận hành

* **Đồng thời**: Ultralytics predictor không thread-safe → adapter khoá `predict`; API giới hạn
  `max_concurrency` và `inference_timeout_s` (trả 504). Muốn tăng throughput → tăng
  `HELMET_API__WORKERS` hoặc số replica (mỗi process tải một bản model riêng).
* **Timeout**: khi hết hạn, request trả 504 ngay; luồng suy luận đang chạy sẽ hoàn tất ở nền
  (Python không huỷ được thread); slot của `CapacityLimiter` được trả lại ngay, nhưng request kế tiếp
  vẫn có thể phải chờ lock của model cho tới khi lượt suy luận cũ xong.
* **Exit code CLI**: `0` OK, `1` lỗi, `2` có vi phạm (khi dùng `--fail-on-violation`).
