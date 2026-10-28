.PHONY: help venv install-dev test lint format models smoke run clean baseline eval judge-smoke evolve bench
.DEFAULT_GOAL := help
UV := uv
MODELS := src/obs_lsd/models
SEG_URL := https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite
FL_URL := https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
PORTRAIT_URL := https://storage.googleapis.com/mediapipe-assets/portrait.jpg
BENCH_HOST ?= pequod
BENCH_DIR ?= Public/src/dob/obs-lsd
BENCH_CLIP ?= clips/bench-10s.mov

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

baseline: models ## Render + judge main on CLIP, save runs/baseline.json (CLIP=clips/x.mov)
	$(UV) run obs-lsd-rig baseline $(CLIP)

eval: models ## Render + judge the current tree on CLIP
	$(UV) run obs-lsd-rig render $(CLIP) --out runs/$$(git rev-parse --abbrev-ref HEAD | tr / -)-$$(git rev-parse --short HEAD) && \
	$(UV) run obs-lsd-rig judge runs/$$(git rev-parse --abbrev-ref HEAD | tr / -)-$$(git rev-parse --short HEAD)

judge-smoke: ## One real judge call against the latest run dir
	$(UV) run obs-lsd-rig judge $$(ls -dt runs/*/ | head -1) --samples 1

evolve: models ## Run GENS generations of the loop on CLIP
	$(UV) run obs-lsd-rig evolve $(CLIP) --gens $(GENS)

bench: ## Sync src + BENCH_CLIP to BENCH_HOST, render there, print per-stage ms
	@rsync -az --exclude models --exclude __pycache__ src/obs_lsd/ $(BENCH_HOST):$(BENCH_DIR)/src/obs_lsd/
	@rsync -az pyproject.toml uv.lock $(BENCH_HOST):$(BENCH_DIR)/ && rsync -az $(BENCH_CLIP) $(BENCH_HOST):$(BENCH_DIR)/clips/
	@ssh $(BENCH_HOST) 'cd $(BENCH_DIR) && export PATH=/opt/homebrew/bin:$$PATH && \rm -rf runs/bench && \
	  uv run --quiet obs-lsd-rig render clips/$(notdir $(BENCH_CLIP)) --out runs/bench >/dev/null 2>&1 && \
	  uv run --quiet python -c "import json; m=json.load(open(\"runs/bench/metrics.json\")); \
	  print(\"bench $(BENCH_HOST): total=%.1fms segment=%.1fms halo=%.1fms compose=%.1fms  fps=%.1f  no_head=%.3f frames=%d\" % \
	  (m[\"ms_total_mean\"], m[\"ms_segment_mean\"], m[\"ms_halo_mean\"], m[\"ms_compose_mean\"], 1000/m[\"ms_total_mean\"], m[\"no_head_frac\"], m[\"frames\"]))"'
