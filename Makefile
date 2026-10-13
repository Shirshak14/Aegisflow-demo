.PHONY: smoke manifest check setup validate ingest eda preprocess train serve test feature-registry

setup:
	pip install -r requirements.txt

# lists missing data/model/report files and the command that creates each
check:
	python -m aegisflow doctor

validate:
	python -m aegisflow validate-dataset --dataset cic_ids2017

ingest:
	python -m aegisflow ingest --dataset cic_ids2017

ingest-sample:
	python -m aegisflow ingest --dataset cic_ids2017 --sample-size 100000

eda:
	python -m aegisflow eda --dataset cic_ids2017

preprocess:
	python -m aegisflow preprocess --dataset cic_ids2017

train:
	python -m aegisflow train --dataset cic_ids2017

serve:
	uvicorn backend.app.main:app --host 127.0.0.1 --port 8000

# plain pytest uses testpaths from pyproject.toml (tests/)
test:
	pytest

feature-registry:
	python -m aegisflow feature-registry

# synthetic end-to-end run in a temp directory; needs no dataset (prints a JSON report, exit 1 on failure)
smoke:
	python -m aegisflow.smoke

# write manifest.json (file hashes, feature schema, threshold, training config) for the demo model; add --verify to check it
manifest:
	python -m aegisflow manifest --dataset cic_ids2017
