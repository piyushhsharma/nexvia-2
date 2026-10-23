.PHONY: up down build test lint

up:
	docker-compose up -d

down:
	docker-compose down

build:
	docker-compose build

test:
	py -m pytest tests/ -v > reports/tests.txt 2>&1 || true

lint:
	py -m pylint src/ || true
