.PHONY: install run test lint format up down

install:
	python -m pip install -r requirements.txt

run:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest -q

lint:
	python -m compileall .

format:
	python -m pip install ruff black && black . && ruff check . --fix

up:
	docker compose up --build

down:
	docker compose down
