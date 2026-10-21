.PHONY: install test smoke download reproduce verify clean

install:
	python -m pip install -e .

test:
	python -m pytest -q

smoke:
	python scripts/run_smoke.py

download:
	python -m scale_residual.cli download-data --config configs/paper_main.toml

reproduce:
	python scripts/reproduce_paper.py --config configs/paper_main.toml

verify:
	python -m scale_residual.cli verify --config configs/paper_main.toml

clean:
	rm -rf outputs/* data/processed/* .pytest_cache
