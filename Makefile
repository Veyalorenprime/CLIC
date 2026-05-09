.PHONY: install test format lint train exp1 exp2 clean help

help:
	@echo "Available commands:"
	@echo "  make install  - Install package and dependencies"
	@echo "  make test     - Run unit tests"
	@echo "  make format   - Format code with black"
	@echo "  make lint     - Lint code with flake8"
	@echo "  make train    - Train model (default config)"
	@echo "  make exp1     - Circuit analysis (summary + residual scatter)"
	@echo "  make exp2     - Causal vs. linear baseline comparison"
	@echo "  make clean    - Remove generated files"

install:
	pip install -e .
	pip install -r requirements.txt

test:
	pytest tests/ -v

format:
	black src/ scripts/ tests/

lint:
	flake8 src/ scripts/ tests/

train:
	python scripts/train.py --config configs/base_config.yaml

exp1:
	python scripts/plot_circuit_summary.py --checkpoint experiments/test/best_model.pt
	python scripts/plot_circuit_residuals.py --checkpoint experiments/test/best_model.pt

exp2:
	python scripts/compare_linear_baseline.py --checkpoint experiments/test/best_model.pt

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov
