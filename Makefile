# Convenience targets for the ML drift + optimization system.
# Run `make help` for the list.

PYTHON ?= python3
PIP    ?= pip

.PHONY: help install data train detect optimize pipeline monitor app test lint clean

help:
	@echo "Targets:"
	@echo "  install    Install package + dependencies (editable)"
	@echo "  data       Generate reference + current (drifted) datasets"
	@echo "  train      Train the baseline champion model"
	@echo "  detect     Run drift detection + Evidently reports"
	@echo "  optimize   Run post-detection optimization (remediation)"
	@echo "  pipeline   Run the full end-to-end pipeline"
	@echo "  advanced   Demo the advanced features (schema, impact, pred-drift, cost gate)"
	@echo "  monitor    Append current run metrics to monitoring history"
	@echo "  app        Launch the Streamlit dashboard"
	@echo "  test       Run the pytest suite"
	@echo "  clean      Remove generated artifacts"

install:
	$(PIP) install -e ".[dev]"

data:
	$(PYTHON) scripts/generate_data.py

train:
	$(PYTHON) scripts/train_baseline.py

detect:
	$(PYTHON) scripts/detect_drift.py

optimize:
	$(PYTHON) scripts/optimize.py

pipeline:
	$(PYTHON) scripts/run_pipeline.py

advanced:
	$(PYTHON) examples/advanced_features.py

monitor:
	$(PYTHON) scripts/run_pipeline.py --log-monitoring

app:
	streamlit run app/streamlit_app.py

test:
	PYTHONPATH=src $(PYTHON) -m pytest

clean:
	rm -rf artifacts data/*.csv .pytest_cache **/__pycache__ *.html
