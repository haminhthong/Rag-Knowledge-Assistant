setup:
	python -m pip install -r requirements.txt

download:
	python scripts/download_data.py

train:
	python -m src.train

evaluate:
	python -m src.evaluate

ablation:
	python scripts/ablation_experiments.py

serve:
	uvicorn src.api:app --host 0.0.0.0 --port 8000

test:
	pytest -q

