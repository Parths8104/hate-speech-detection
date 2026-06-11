.PHONY: install test lint train sample serve docker clean

install:
	pip install -e ".[dev]" || pip install -r requirements-dev.txt && pip install -e .

test:
	pytest -q

lint:
	ruff check src tests

sample:
	python -m hsd.train --config config/sample.yaml

train:
	python -m hsd.train --config config/config.yaml

serve:
	uvicorn hsd.api:app --reload

docker:
	docker build -t hate-speech-detection .

clean:
	rm -rf models/*.joblib models/*.json models/reports .pytest_cache .ruff_cache
