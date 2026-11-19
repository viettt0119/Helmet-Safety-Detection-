# Graph Report - Helmet-Safety-Detection-  (2026-11-19)

## Corpus Check
- Corpus is ~36,019 words - fits in a single context window. You may not need a graph.

## Summary
- 841 nodes · 2294 edges · 33 communities (19 shown, 14 thin omitted)
- Extraction: 84% EXTRACTED · 16% INFERRED · 0% AMBIGUOUS · INFERRED: 367 edges (avg confidence: 0.93)
- Token cost: 97,568 input · 0 output

## Community Hubs (Navigation)
- YOLO Adapters & Ports
- Analysis Use Cases & Architecture
- API App & Logging
- CLI Commands
- Weights Resolve & Download
- API Routes & Errors
- CLI & E2E Tests
- YOLO Adapter Tests
- CI Workflow
- Docker Compose
- Settings & Config
- Project Docs Overview
- BoundingBox Value Object
- Dataset Validation
- Detection Classes & Policy
- OpenCV Video I/O
- Compliance Status Enums
- Worker DTO Serialization
- Analyze Module Imports
- Compliance Matching Rule
- SafetyReport Aggregation
- Device Resolution
- SafetyReport Tests
- Project Metadata

## God Nodes (most connected - your core abstractions)
1. `Image` - 68 edges
2. `Detection` - 51 edges
3. `Container` - 51 edges
4. `FakeDetector` - 45 edges
5. `BoundingBox` - 33 edges
6. `InvalidInputError` - 33 edges
7. `DetectionClass` - 32 edges
8. `SafetyReport` - 31 edges
9. `ComplianceAnalyzer` - 31 edges
10. `Settings` - 30 edges

## Surprising Connections (you probably didn't know these)
- `Overall Status (safe/unsafe/inconclusive/empty)` --references--> `SafetyReport`  [INFERRED]
  README.md → src/helmet_safety/domain/entities.py
- `Strategy Pattern` --references--> `ConfidencePolicy`  [EXTRACTED]
  docs/ARCHITECTURE.md → src/helmet_safety/domain/policies.py
- `Dependency Injection / Composition Root` --references--> `Container`  [EXTRACTED]
  docs/ARCHITECTURE.md → src/helmet_safety/infrastructure/container.py
- `Safe Weights Download (retry, timeout, atomic write)` --references--> `WeightsDownloader`  [EXTRACTED]
  docs/ARCHITECTURE.md → src/helmet_safety/infrastructure/weights.py
- `Inference Concurrency Model (lock + CapacityLimiter)` --references--> `UltralyticsDetector`  [INFERRED]
  docs/ARCHITECTURE.md → src/helmet_safety/infrastructure/yolo.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Image-to-SafetyReport Pipeline** — src_helmet_safety_infrastructure_imaging_opencvframesource, src_helmet_safety_infrastructure_yolo_ultralyticsdetector, src_helmet_safety_domain_policies_confidencepolicy, src_helmet_safety_domain_services_complianceanalyzer, src_helmet_safety_domain_entities_safetyreport, src_helmet_safety_interfaces_serialization_safetyreportdto [EXTRACTED 1.00]
- **Clean Architecture Design Patterns** — docs_architecture_ports_and_adapters, docs_architecture_composition_root, docs_architecture_strategy_pattern, docs_architecture_chain_of_responsibility, docs_architecture_factory_pattern, docs_architecture_value_object [EXTRACTED 1.00]
- **Model Weights Resolution & Download** — src_helmet_safety_infrastructure_weights_weightsresolver, src_helmet_safety_infrastructure_weights_weightsdownloader, src_helmet_safety_domain_errors_modelnotfounderror, docs_architecture_weights_resolution, docs_architecture_safe_weights_download [INFERRED 0.85]

## Communities (33 total, 14 thin omitted)

### Community 0 - "YOLO Adapters & Ports"
Cohesion: 0.05
Nodes (33): EvaluationRequest, EvaluationResult, ModelEvaluator, ModelTrainer, TrainingRequest, TrainingResult, EvaluateModelUseCase, TrainModelUseCase (+25 more)

### Community 1 - "Analysis Use Cases & Architecture"
Cohesion: 0.05
Nodes (25): AnalysisResult, AnalyzeImageUseCase, AnalyzeStreamUseCase, validate_image(), FrameSink, FrameSource, Image, ImageAnnotator (+17 more)

### Community 2 - "API App & Logging"
Cohesion: 0.05
Nodes (37): get_settings(), Settings, configure_logging(), get_logger(), _identity(), app_factory(), create_app(), lifespan() (+29 more)

### Community 3 - "CLI Commands"
Cohesion: 0.07
Nodes (21): command, Container, is_image_path(), parse_source(), WindowSink, _container(), download_weights(), evaluate() (+13 more)

### Community 4 - "Weights Resolve & Download"
Cohesion: 0.08
Nodes (10): DownloadError, ModelNotFoundError, find_latest_best(), _RetryableHTTPError, WeightsDownloader, WeightsResolver, downloader(), make_best() (+2 more)

### Community 5 - "API Routes & Errors"
Cohesion: 0.06
Nodes (20): ConfigurationError, HelmetSafetyError, ModelLoadError, _envelope(), register_exception_handlers(), _domain_error(), _http_error(), _unhandled() (+12 more)

### Community 6 - "CLI & E2E Tests"
Cohesion: 0.11
Nodes (25): read_image(), _dataset(), images_dir(), install_container(), test_download_weights_existing(), test_download_weights_rejects_http(), test_evaluate_invalid_split(), test_evaluate_with_mock() (+17 more)

### Community 7 - "YOLO Adapter Tests"
Cohesion: 0.11
Nodes (20): test_invalid_parameters(), test_detection_confidence_bounds(), test_image_size_must_be_positive(), FakeBoxes, FakeTensor, FakeYOLO, make_detector(), test_corrupt_weights() (+12 more)

### Community 8 - "CI Workflow"
Cohesion: 0.06
Nodes (34): concurrency, cancel-in-progress, group, needs, runs-on, steps, jobs, docker (+26 more)

### Community 9 - "Docker Compose"
Cohesion: 0.06
Nodes (30): build, container_name, deploy, environment, image, ports, read_only, restart (+22 more)

### Community 10 - "Settings & Config"
Cohesion: 0.10
Nodes (10): ApiSettings, DownloadSettings, ModelSettings, TrainingSettings, image(), _isolate_settings(), settings(), worker_with_helmet() (+2 more)

### Community 11 - "Project Docs Overview"
Cohesion: 0.10
Nodes (19): Architecture & Audit Report (v2.0), Backward-compatible Script Shims, CI Pipeline (GitHub Actions), CLI Exit Codes (0/1/2), Structured Logging (structlog), Technology Stack Update, v1 Audit of Existing Scripts, helmet-safety CLI (+11 more)

### Community 13 - "Dataset Validation"
Cohesion: 0.22
Nodes (11): DatasetError, _class_names(), DatasetSpec, _load_yaml(), resolve_dataset(), make_dataset(), test_invalid(), test_missing_file() (+3 more)

### Community 14 - "Detection Classes & Policy"
Cohesion: 0.16
Nodes (8): DetectionClass, ConfidencePolicy, PolicySettings, TestDetectionClass, test_per_class_thresholds(), test_rejects_out_of_range(), test_threshold_boundary_is_inclusive(), test_thresholds_are_immutable()

### Community 15 - "OpenCV Video I/O"
Cohesion: 0.19
Nodes (4): VideoSourceError, OpenCVFrameSource, VideoFileSink, TestVideo

### Community 16 - "Compliance Status Enums"
Cohesion: 0.24
Nodes (16): ComplianceStatus, OverallStatus, det(), analyze(), test_conflicting_helmet_and_head_uses_confidence(), test_duplicate_unmatched_head_and_helmet_counted_once(), test_each_helmet_is_assigned_once(), test_helmet_in_lower_body_is_not_matched() (+8 more)

### Community 17 - "Worker DTO Serialization"
Cohesion: 0.20
Nodes (8): WorkerAssessment, BoxDTO, DetectionDTO, _Frozen, SafetyReportDTO, SummaryDTO, WorkerDTO, conv()

### Community 19 - "Compliance Matching Rule"
Cohesion: 0.18
Nodes (4): Data Flow (image -> SafetyReport), ComplianceAnalyzer, _is_dominated(), _sorted_by_conf()

### Community 21 - "Device Resolution"
Cohesion: 0.20
Nodes (3): resolve_device(), test_default_probes_do_not_crash(), test_resolve_device()

## Knowledge Gaps
- **8 isolated node(s):** `helmet-safety`, `Docker Deployment`, `Test Suite (194 tests, ~96% coverage)`, `Licensing (CC BY 4.0 dataset, AGPL-3.0 Ultralytics)`, `uv Package Manager` (+3 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 247 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **14 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Container` connect `CLI Commands` to `YOLO Adapters & Ports`, `Analysis Use Cases & Architecture`, `API App & Logging`, `Weights Resolve & Download`, `CLI & E2E Tests`, `Settings & Config`?**
  _High betweenness centrality (0.082) - this node is a cross-community bridge._
- **Are the 34 inferred relationships involving `Image` (e.g. with `AnalysisResult` and `AnalyzeImageUseCase`) actually correct?**
  _`Image` has 34 INFERRED edges - model-reasoned connections that need verification._
- **What connects `helmet-safety`, `Docker Deployment`, `Test Suite (194 tests, ~96% coverage)` to the rest of the system?**
  _8 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `YOLO Adapters & Ports` be split into smaller, more focused modules?**
  _Cohesion score 0.05183459522422831 - nodes in this community are weakly interconnected._
- **Why does `Image` connect `Analysis Use Cases & Architecture` to `YOLO Adapters & Ports`, `API App & Logging`, `CLI Commands`, `CLI & E2E Tests`, `YOLO Adapter Tests`, `Settings & Config`, `OpenCV Video I/O`, `Compliance Status Enums`, `Analyze Module Imports`, `SafetyReport Aggregation`?**
  _High betweenness centrality (0.068) - this node is a cross-community bridge._
- **Are the 25 inferred relationships involving `Detection` (e.g. with `ObjectDetector` and `ConfidencePolicy`) actually correct?**
  _`Detection` has 25 INFERRED edges - model-reasoned connections that need verification._
- **Should `Analysis Use Cases & Architecture` be split into smaller, more focused modules?**
  _Cohesion score 0.050101010101010104 - nodes in this community are weakly interconnected._