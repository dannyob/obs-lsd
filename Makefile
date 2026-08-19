.PHONY: help venv install-dev test lint format models smoke run clean
.DEFAULT_GOAL := help
UV := uv
MODELS := src/obs_lsd/models
SEG_URL := https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite
FL_URL := https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
PORTRAIT_URL := https://storage.googleapis.com/mediapipe-assets/portrait.jpg

help: ## Show targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

venv: ## Create .venv with Python 3.12
	$(UV) venv -p 3.12

install-dev: venv ## Install package + dev deps
	$(UV) pip install -e ".[dev]"

models: $(MODELS)/selfie_multiclass_256x256.tflite $(MODELS)/face_landmarker.task ## Download MediaPipe models

$(MODELS)/selfie_multiclass_256x256.tflite:
	mkdir -p $(MODELS) && curl -sSL -o $@ $(SEG_URL)

$(MODELS)/face_landmarker.task:
	mkdir -p $(MODELS) && curl -sSL -o $@ $(FL_URL)

tests/data/portrait.jpg: ## Public MediaPipe sample portrait for tests
	mkdir -p tests/data && curl -sSL -o $@ $(PORTRAIT_URL)

testdata: tests/data/portrait.jpg ## Fetch test assets

test: ## Run tests
	$(UV) run pytest -q

lint: ## ruff check
	$(UV) run ruff check .

format: ## ruff format
	$(UV) run ruff format .

smoke: models testdata ## End-to-end headless run on the still portrait
	$(UV) run obs-lsd --input tests/data/portrait.jpg --seconds 2 --no-vcam --no-preview --dump-frame /tmp/obs-lsd-smoke.png
	@test -s /tmp/obs-lsd-smoke.png && echo "smoke OK: /tmp/obs-lsd-smoke.png"

run: models ## Run live with preview + virtual camera
	$(UV) run obs-lsd

clean: ## Remove caches
	\rm -rf .pytest_cache .ruff_cache dist build
	find . -name __pycache__ -type d -exec \rm -rf {} +
