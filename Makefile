PYTHON ?= python3
HOST ?= 0.0.0.0
PORT ?= 8000

.PHONY: install test lint smoke check-assets serve demo benchmark benchmark-http package-model clean

install:
	$(PYTHON) -m pip install -e '.[serving,benchmark,dev]'

test:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src $(PYTHON) -m pytest -p no:cacheprovider -q

lint:
	$(PYTHON) -m ruff check src scripts tests

smoke:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src $(PYTHON) -m pytest -p no:cacheprovider -q tests/test_serving.py tests/test_models.py tests/test_catalog.py

check-assets:
	PYTHONPATH=src $(PYTHON) scripts/check_serving_assets.py

serve: check-assets
	PYTHONPATH=src $(PYTHON) -m uvicorn regenrec.api:app --host $(HOST) --port $(PORT)

demo: check-assets
	PYTHONPATH=src $(PYTHON) scripts/demo_serving.py --beam-size 20 --long-tail-mode off

benchmark: check-assets
	PYTHONPATH=src $(PYTHON) scripts/benchmark_serving.py

benchmark-http:
	PYTHONPATH=src $(PYTHON) scripts/benchmark_http.py

package-model: check-assets
	PYTHONPATH=src $(PYTHON) scripts/package_model_release.py

clean:
	find src scripts tests -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache build dist
