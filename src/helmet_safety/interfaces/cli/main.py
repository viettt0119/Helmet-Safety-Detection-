"""``helmet-safety`` command line: predict / train / evaluate / serve / info / download-weights."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

import typer

from helmet_safety import __version__
from helmet_safety.application.ports import EvaluationRequest, FrameSink, TrainingRequest
from helmet_safety.domain.entities import OverallStatus
from helmet_safety.domain.errors import HelmetSafetyError, InvalidInputError
from helmet_safety.infrastructure.config import Settings, get_settings
from helmet_safety.infrastructure.container import Container
from helmet_safety.infrastructure.dataset import resolve_dataset
from helmet_safety.infrastructure.imaging import (
    IMAGE_SUFFIXES,
    OpenCVFrameSource,
    VideoFileSink,
    WindowSink,
    is_image_path,
    parse_source,
    read_image,
    write_image,
)
from helmet_safety.infrastructure.logging import configure_logging, get_logger
from helmet_safety.interfaces.serialization import SafetyReportDTO

app = typer.Typer(
    name="helmet-safety",
    help="Helmet (PPE) compliance detection with YOLOv10.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)
logger = get_logger(__name__)

EXIT_OK = 0
EXIT_UNSAFE = 2  # predict --fail-on-violation found a worker without helmet
EXIT_ERROR = 1


class _State:
    container: Container | None = None


state = _State()


def _container() -> Container:
    if state.container is None:
        state.container = Container(get_settings())
    return state.container


def _settings() -> Settings:
    return _container().settings


def _fail(exc: HelmetSafetyError) -> typer.Exit:
    typer.secho(f"Error [{exc.code}]: {exc.message}", fg=typer.colors.RED, err=True)
    return typer.Exit(code=EXIT_ERROR)


@app.callback()
def main(
    log_level: Annotated[str | None, typer.Option(help="Override HELMET_LOG_LEVEL")] = None,
    log_json: Annotated[bool | None, typer.Option("--log-json/--no-log-json", help="JSON logs")] = None,
) -> None:
    settings = _settings()
    configure_logging(
        log_level or settings.log_level, json=settings.log_json if log_json is None else log_json
    )


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command()
def info() -> None:
    """Show resolved configuration and model information."""
    container = _container()
    try:
        detector = container.detector
        detector.load()
        model = detector.info
    except HelmetSafetyError as exc:
        raise _fail(exc) from exc
    typer.echo(
        json.dumps(
            {
                "version": __version__,
                "weights": str(model.weights_path),
                "device": model.device,
                "image_size": model.image_size,
                "classes": model.class_names,
            },
            indent=2,
        )
    )


def _iter_images(source: Path) -> list[Path]:
    if source.is_dir():
        files = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if not files:
            raise InvalidInputError(f"No images found in directory: {source}")
        return files
    return [source]


@app.command()
def predict(
    source: Annotated[
        str, typer.Argument(help="Image, folder of images, video file, RTSP/HTTP URL or webcam index")
    ] = "0",
    output_dir: Annotated[
        Path | None, typer.Option("--output-dir", "-o", help="Save annotated results here")
    ] = None,
    show: Annotated[bool, typer.Option("--show/--no-show", help="Display results in a window")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Print JSON reports to stdout")] = False,
    max_frames: Annotated[int | None, typer.Option(min=1, help="Stop a stream after N frames")] = None,
    fail_on_violation: Annotated[
        bool, typer.Option(help=f"Exit with code {EXIT_UNSAFE} if a violation is found")
    ] = False,
) -> None:
    """Run detection on images, videos, streams or a webcam."""
    container = _container()
    parsed = parse_source(source)
    try:
        if isinstance(parsed, str) and (Path(parsed).is_dir() or is_image_path(parsed)):
            unsafe = _predict_images(container, Path(parsed), output_dir, show=show, as_json=as_json)
        else:
            unsafe = _predict_stream(
                container, parsed, output_dir, show=show, max_frames=max_frames, as_json=as_json
            )
    except HelmetSafetyError as exc:
        raise _fail(exc) from exc
    if fail_on_violation and unsafe:
        raise typer.Exit(code=EXIT_UNSAFE)


def _predict_images(
    container: Container, source: Path, output_dir: Path | None, *, show: bool, as_json: bool
) -> bool:
    use_case = container.analyze_image
    annotate = output_dir is not None or show
    any_unsafe = False
    reports: list[dict[str, object]] = []
    for path in _iter_images(source):
        result = use_case.execute(read_image(path), annotate=annotate)
        report = result.report
        any_unsafe |= report.status is OverallStatus.UNSAFE
        dto = SafetyReportDTO.from_domain(report, result.inference_ms)
        if as_json:
            reports.append({"source": str(path), **dto.model_dump(mode="json")})
        else:
            typer.echo(
                f"{path.name}: {report.status.value.upper()} workers={len(report.workers)} "
                f"compliant={report.compliant_count} violations={report.violation_count} "
                f"unknown={report.unknown_count} ({result.inference_ms:.1f} ms)"
            )
        if output_dir is not None and result.annotated is not None:
            write_image(output_dir / f"{path.stem}_annotated.jpg", result.annotated)
            (output_dir / f"{path.stem}.json").write_text(dto.model_dump_json(indent=2), encoding="utf-8")
        if show and result.annotated is not None:
            sink = WindowSink()
            sink.write(result.annotated)
            import cv2

            cv2.waitKey(0)
            sink.close()
    if as_json:
        typer.echo(json.dumps(reports if len(reports) != 1 else reports[0], indent=2))
    return any_unsafe


def _predict_stream(
    container: Container,
    source: int | str,
    output_dir: Path | None,
    *,
    show: bool,
    max_frames: int | None,
    as_json: bool,
) -> bool:
    frame_source = OpenCVFrameSource(source)
    sinks: list[FrameSink] = []
    if show:
        sinks.append(WindowSink())
    if output_dir is not None:
        name = "webcam" if isinstance(source, int) else Path(str(source)).stem or "stream"
        sinks.append(VideoFileSink(output_dir / f"{name}_annotated.mp4", fps=frame_source.fps))
    if not sinks and max_frames is None and isinstance(source, int):
        typer.secho("Webcam without --show/--output-dir/--max-frames would run forever; add one.", err=True)
        raise typer.Exit(code=EXIT_ERROR)
    if show:
        typer.echo("Press 'q' or ESC to stop.", err=True)
    summary = container.analyze_stream.run(frame_source, sinks, max_frames=max_frames)
    payload = {
        "source": str(source),
        "frames": summary.frames,
        "unsafe_frames": summary.unsafe_frames,
        "max_violations_in_frame": summary.max_violations,
        "avg_inference_ms": round(summary.avg_inference_ms, 2),
    }
    typer.echo(json.dumps(payload, indent=2) if as_json else " ".join(f"{k}={v}" for k, v in payload.items()))
    return summary.unsafe_frames > 0


@app.command()
def train(
    epochs: Annotated[int | None, typer.Option(min=1)] = None,
    batch: Annotated[int | None, typer.Option(help="Batch size, -1 = auto")] = None,
    image_size: Annotated[int | None, typer.Option("--imgsz", min=32)] = None,
    workers: Annotated[int | None, typer.Option(min=0)] = None,
    name: Annotated[str | None, typer.Option(help="Run name under runs/")] = None,
    data: Annotated[Path | None, typer.Option(help="Path to data.yaml")] = None,
    base_weights: Annotated[Path | None, typer.Option(help="Pretrained checkpoint")] = None,
) -> None:
    """Fine-tune YOLOv10 on the helmet dataset (downloads pretrained weights if needed)."""
    container = _container()
    s = container.settings
    try:
        weights = base_weights or s.model.pretrained_weights_path
        if base_weights is None:
            container.downloader.ensure(str(s.model.pretrained_weights_url), weights)
        dataset = resolve_dataset(data or s.training.dataset_yaml)
        result = container.train_model.execute(
            TrainingRequest(
                dataset_yaml=dataset.resolved_yaml,
                base_weights=weights,
                epochs=epochs or s.training.epochs,
                image_size=image_size or s.model.image_size,
                batch=batch or s.training.batch,
                workers=s.training.workers if workers is None else workers,
                device=container.device,
                project_dir=s.model.runs_dir,
                run_name=name or s.training.run_name,
                patience=s.training.patience,
                seed=s.training.seed,
            )
        )
    except HelmetSafetyError as exc:
        raise _fail(exc) from exc
    typer.echo(
        json.dumps(
            {
                "run_dir": str(result.run_dir),
                "best_weights": str(result.best_weights),
                "metrics": result.metrics,
            },
            indent=2,
        )
    )


@app.command()
def evaluate(
    split: Annotated[str, typer.Option(help="train | val | test")] = "test",
    weights: Annotated[
        Path | None, typer.Option(help="Checkpoint to evaluate (default: configured weights)")
    ] = None,
    data: Annotated[Path | None, typer.Option(help="Path to data.yaml")] = None,
    batch: Annotated[int, typer.Option(min=1)] = 16,
) -> None:
    """Evaluate a checkpoint (precision, recall, mAP) on a dataset split."""
    container = _container()
    s = container.settings
    try:
        resolved_weights = weights or container.weights_resolver.resolve()
        dataset = resolve_dataset(data or s.training.dataset_yaml)
        result = container.evaluate_model.execute(
            EvaluationRequest(
                dataset_yaml=dataset.resolved_yaml,
                weights=resolved_weights,
                split=split,
                image_size=s.model.image_size,
                batch=batch,
                device=container.device,
                project_dir=s.model.runs_dir,
            )
        )
    except HelmetSafetyError as exc:
        raise _fail(exc) from exc
    typer.echo(
        json.dumps(
            {
                "weights": str(resolved_weights),
                "split": split,
                "precision": round(result.precision, 4),
                "recall": round(result.recall, 4),
                "mAP50": round(result.map50, 4),
                "mAP50-95": round(result.map50_95, 4),
                "per_class_mAP50-95": {k: round(v, 4) for k, v in result.per_class_map50_95.items()},
                "save_dir": str(result.save_dir) if result.save_dir else None,
            },
            indent=2,
        )
    )


@app.command("download-weights")
def download_weights(
    url: Annotated[str | None, typer.Option(help="HTTPS URL (default: configured pretrained URL)")] = None,
    destination: Annotated[Path | None, typer.Option(help="Target file")] = None,
    force: Annotated[bool, typer.Option(help="Re-download even if the file exists")] = False,
) -> None:
    """Download pretrained YOLOv10n weights (retry with exponential back-off)."""
    container = _container()
    s = container.settings
    target = destination or s.model.pretrained_weights_path
    source = url or str(s.model.pretrained_weights_url)
    try:
        path = (
            container.downloader.download(source, target)
            if force
            else container.downloader.ensure(source, target)
        )
    except HelmetSafetyError as exc:
        raise _fail(exc) from exc
    typer.echo(str(path))


@app.command()
def serve(
    host: Annotated[str | None, typer.Option()] = None,
    port: Annotated[int | None, typer.Option(min=1, max=65535)] = None,
    workers: Annotated[int | None, typer.Option(min=1)] = None,
    reload: Annotated[bool, typer.Option(help="Auto-reload (development only)")] = False,
) -> None:
    """Start the REST API (uvicorn)."""
    import uvicorn

    s = _settings()
    uvicorn.run(
        "helmet_safety.interfaces.api.app:app_factory",
        factory=True,
        host=host or s.api.host,
        port=port or s.api.port,
        workers=None if reload else (workers or s.api.workers),
        reload=reload,
        log_config=None,
        proxy_headers=True,
    )


def run() -> None:  # pragma: no cover - console-script entry
    try:
        app()
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":  # pragma: no cover
    run()
