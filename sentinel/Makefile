.PHONY: all install docker-up docker-down bootstrap backfill \
        terminal api test test-unit test-integration test-evals \
        lint format typecheck clean

# ── Setup ───────────────────────────────────────────────────────────────────
install:
	poetry install

docker-up:
	docker compose -f infra/docker-compose.yml up -d
	@echo "Waiting for postgres to be ready..."
	@sleep 3
	docker compose -f infra/docker-compose.yml exec postgres pg_isready -U sentinel

docker-down:
	docker compose -f infra/docker-compose.yml down

docker-logs:
	docker compose -f infra/docker-compose.yml logs -f

bootstrap: docker-up
	poetry run python scripts/bootstrap.py

backfill:
	poetry run python scripts/backfill.py --source edgar --years 5
	poetry run python scripts/backfill.py --source fred

# ── Run ─────────────────────────────────────────────────────────────────────
terminal:
	poetry run streamlit run stu/app.py --server.port 8501

api:
	poetry run uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

mcp:
	poetry run python -m sil.mcp_server

# ── Test ────────────────────────────────────────────────────────────────────
test-unit:
	poetry run pytest tests/unit/ -v --tb=short

test-integration:
	poetry run pytest tests/integration/ -v --tb=short -m "not slow"

test-evals:
	poetry run pytest tests/financial_evals/ -v --tb=long -s

test: test-unit test-integration

# ── Quality ─────────────────────────────────────────────────────────────────
lint:
	poetry run ruff check sentinel/ --fix

format:
	poetry run ruff format sentinel/

typecheck:
	poetry run mypy sentinel/ --ignore-missing-imports

# ── Data ────────────────────────────────────────────────────────────────────
bulk-edgar:
	poetry run python scripts/backfill.py --source edgar-bulk

ingest-cot:
	poetry run python -c "from sma.cot_report import COTParser; import asyncio; asyncio.run(COTParser().ingest_all())"

# ── Clean ───────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	find . -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
