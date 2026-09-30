.PHONY: setup validate ingest eda test lint

setup:
	pip install -r requirements.txt

validate:
	python -m aegisflow validate-dataset --dataset cic_ids2017

ingest:
	python -m aegisflow ingest --dataset cic_ids2017

ingest-sample:
	python -m aegisflow ingest --dataset cic_ids2017 --sample-size 100000

eda:
	python -m aegisflow eda --dataset cic_ids2017

test:
	pytest tests/ -v

feature-registry:
	python -m aegisflow feature-registry
