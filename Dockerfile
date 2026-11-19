# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------------------------
# Helmet Safety Detection — multi-stage image (uv + Python 3.13, non-root, CPU torch by default)
#   docker build -t helmet-safety .                          # CPU (small image)
#   docker build --build-arg TORCH_VARIANT=cuda -t helmet-safety:cuda .   # keep CUDA wheels
# ---------------------------------------------------------------------------------------------
ARG PYTHON_VERSION=3.13
ARG UV_VERSION=0.11.32

# ---------- builder: resolve & install dependencies into /app/.venv ---------------------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS builder
ARG TORCH_VARIANT=cpu
ARG UV_VERSION
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv
RUN pip install --no-cache-dir "uv==${UV_VERSION}"
WORKDIR /app

# 1) dependencies only (cached layer as long as the lockfile does not change)
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# 2) the project itself (non-editable, no dependency changes)
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /app/.venv/bin/python --no-deps .

# 3) servers have no display: swap OpenCV for the headless build (no libGL / X11 system packages)
RUN --mount=type=cache,target=/root/.cache/uv \
    set -eu; PY=/app/.venv/bin/python; \
    CV_V=$($PY -c "import importlib.metadata as m; print(m.version('opencv-python'))"); \
    uv pip uninstall --python $PY opencv-python; \
    uv pip install --python $PY --no-deps "opencv-python-headless==${CV_V}"; \
    $PY -c "import cv2; print('opencv', cv2.__version__)"

# 4) optional: replace the CUDA torch wheels (~4 GB) with the CPU build (~0.8 GB)
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$TORCH_VARIANT" = "cpu" ]; then \
      set -eu; \
      PY=/app/.venv/bin/python; \
      TORCH_V=$($PY -c "import importlib.metadata as m; print(m.version('torch').split('+')[0])"); \
      TV_V=$($PY -c "import importlib.metadata as m; print(m.version('torchvision').split('+')[0])"); \
      GPU_PKGS=$(uv pip list --python $PY --format freeze | grep -E '^(nvidia-|triton=)' | cut -d= -f1 || true); \
      if [ -n "$GPU_PKGS" ]; then uv pip uninstall --python $PY $GPU_PKGS; fi; \
      uv pip install --python $PY --reinstall --no-deps \
        --index-url https://download.pytorch.org/whl/cpu "torch==${TORCH_V}" "torchvision==${TV_V}"; \
      $PY -c "import torch, torchvision; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"; \
    fi

# ---------- runtime -------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime
LABEL org.opencontainers.image.title="helmet-safety" \
      org.opencontainers.image.description="Helmet (PPE) compliance detection API with YOLOv10"

RUN groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid app --home-dir /app --shell /usr/sbin/nologin app

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    YOLO_OFFLINE=1 \
    HELMET_ENVIRONMENT=production \
    HELMET_LOG_JSON=true \
    HELMET_MODEL__DEVICE=cpu \
    HELMET_MODEL__WEIGHTS_PATH=/app/models/helmet_yolov10n_best.pt \
    HELMET_MODEL__RUNS_DIR=/app/runs

WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app models/helmet_yolov10n_best.pt /app/models/helmet_yolov10n_best.pt

USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=4).status == 200 else 1)"]

ENTRYPOINT ["helmet-safety"]
CMD ["serve"]
