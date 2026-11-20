.PHONY: install test lint serve demo eval bench

install:
	pip install -e ".[dev,tokens]"

test:
	pytest -q

lint:
	ruff check . && ruff format --check .

serve:            ## real providers from .env
	research-server

demo:             ## offline: no API keys needed
	LLM_PROVIDER=fake SEARCH_PROVIDER=fake FAKE_LATENCY_SCALE=0.5 research-server

eval:             ## needs real keys in .env; ~200 pipeline runs + judge calls
	python -m eval.run_eval --limit 100 --concurrency 3

bench:
	python -m eval.benchmark --limit 10 --concurrency 4
