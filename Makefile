# Developer shortcuts. Requires uv (https://docs.astral.sh/uv/).
.DEFAULT_GOAL := help
UV ?= uv
IMAGE ?= helmet-safety:latest

.PHONY: help install lint format typecheck test test-fast cov check serve predict evaluate train docker-build docker-run clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Create .venv and install all dependencies (incl. dev)
	$(UV) sync --frozen

lint: ## Ruff lint + format check
	$(UV) run ruff check src tests
	$(UV) run ruff format --check src tests

format: ## Auto-format and fix lint issues
	$(UV) run ruff format src tests
	$(UV) run ruff check --fix src tests

typecheck: ## mypy --strict
	$(UV) run mypy

test: ## Full test suite (includes slow real-model tests)
	$(UV) run pytest

test-fast: ## Unit + integration tests without the real model
	$(UV) run pytest -m "not slow"

cov: ## Tests with coverage report
	$(UV) run pytest --cov --cov-report=term-missing --cov-report=xml

check: lint typecheck test ## Everything CI runs

serve: ## Run the API locally (http://localhost:8000/docs)
	$(UV) run helmet-safety serve --reload

predict: ## Predict on the test images, results in outputs/
	$(UV) run helmet-safety predict data/test/images --output-dir outputs

evaluate: ## Evaluate the trained model on the test split
	$(UV) run helmet-safety evaluate --split test

train: ## Fine-tune YOLOv10n (override: make train ARGS="--epochs 100")
	$(UV) run helmet-safety train $(ARGS)

docker-build: ## Build the CPU Docker image
	docker build -t $(IMAGE) .

docker-run: ## Run the API container
	docker run --rm -p 8000:8000 $(IMAGE)

clean: ## Remove caches and generated outputs
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml htmlcov outputs data/.resolved
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
