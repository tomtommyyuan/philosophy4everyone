.PHONY: help setup library ingest test ask daily doctor clean

help:
	@echo "make setup    create a venv and install philo"
	@echo "make library  download the public-domain texts from Project Gutenberg"
	@echo "make ingest   chunk, embed and index the library"
	@echo "make test     run the test suite (offline, no API key needed)"
	@echo "make doctor   check configuration, index and connectivity"

setup:
	python3 -m venv .venv
	.venv/bin/python -m pip install -q --upgrade pip
	.venv/bin/python -m pip install -q -e ".[all]" pytest

library:
	.venv/bin/python scripts/fetch_library.py

ingest:
	.venv/bin/python -m philo ingest

test:
	.venv/bin/python -m pytest tests -q

ask:
	.venv/bin/python -m philo ask "why do we fear death?"

daily:
	.venv/bin/python -m philo daily

doctor:
	.venv/bin/python -m philo doctor

clean:
	rm -rf .philo .pytest_cache **/__pycache__
