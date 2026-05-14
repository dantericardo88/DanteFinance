# SENTINEL Makefile
.PHONY: install docker-up docker-down bootstrap db-init db-migrate db-status \
        check check-quick backfill backfill-resume data-status \
        backfill-congress backfill-cot backfill-insider backfill-institutional \
        backfill-news backfill-embeddings backfill-rag backfill-crypto \
        backfill-continuous terminal api mcp \
        test test-unit test-integration test-evals lint format typecheck \
        bulk-edgar ingest-cot clean

# ─── Setup ────────────────────────────────────────────────────────────────────
install:
	poetry install

docker-up:
	docker-compose up -d
	@echo "Waiting for PostgreSQL..."
	@sleep 5
	@echo "Services running. Check: http://localhost:8000/docs (API) | http://localhost:8501 (Terminal)"

docker-down:
	docker-compose down

# ─── Bootstrap ────────────────────────────────────────────────────────────────
bootstrap: docker-up
	@echo "Running bootstrap..."
	poetry run python scripts/bootstrap.py

# ─── Database Migrations ──────────────────────────────────────────────────────
db-init: docker-up
	@echo "Schema created via infra/postgres/init.sql at container startup."
	@echo "Run 'make db-migrate' to apply Alembic migrations on top."

db-migrate:
	@echo "Running Alembic migrations..."
	poetry run alembic upgrade head

db-status:
	@echo "Alembic migration state:"
	poetry run alembic current
	poetry run alembic history --verbose

check:
	poetry run python scripts/check.py

check-quick:
	poetry run python scripts/check.py --quick

# ─── Data Backfill ────────────────────────────────────────────────────────────
backfill:
	poetry run python scripts/backfill.py --mode all --years 10

backfill-resume:
	poetry run python scripts/backfill.py --mode all --years 10 --resume

data-status:
	poetry run python scripts/backfill.py --status

bulk-edgar:
	poetry run python scripts/backfill.py --mode edgar

backfill-congress:
	poetry run python scripts/backfill.py --mode congress

backfill-cot:
	poetry run python scripts/backfill.py --mode cot --years 5

ingest-cot:
	poetry run python scripts/backfill.py --mode cot --years 5

backfill-insider:
	poetry run python scripts/backfill.py --mode insider --years 3

backfill-institutional:
	poetry run python scripts/backfill.py --mode institutional --years 2

backfill-news:
	poetry run python scripts/backfill.py --mode news --days 30

backfill-embeddings:
	poetry run python scripts/backfill.py --mode embeddings

backfill-rag:
	poetry run python scripts/backfill.py --mode rag

backfill-crypto:
	poetry run python scripts/backfill.py --mode crypto

backfill-continuous:
	poetry run python scripts/backfill.py --mode continuous --years 20

# ─── Run Services ─────────────────────────────────────────────────────────────
terminal:
	poetry run streamlit run sentinel/stu/terminal.py \
		--server.port=8501 \
		--server.address=localhost \
		--theme.base=dark \
		--theme.primaryColor="#00ff88"

api:
	poetry run uvicorn sentinel.api.main:app \
		--host 0.0.0.0 --port 8000 --reload

mcp:
	poetry run python -m sentinel.sil.mcp_server

# ─── Testing ──────────────────────────────────────────────────────────────────
test:
	poetry run pytest tests/ -v --tb=short

test-unit:
	poetry run pytest tests/test_financial_evals.py -v --tb=short -k "not integration"

test-integration:
	poetry run pytest tests/ -v --tb=short -k "integration"

test-evals:
	poetry run pytest tests/test_financial_evals.py -v --tb=long

test-cov:
	poetry run pytest tests/ --cov=sentinel --cov-report=html -v

# ─── Code Quality ─────────────────────────────────────────────────────────────
lint:
	poetry run ruff check sentinel/ tests/
	poetry run ruff check scripts/

format:
	poetry run black sentinel/ tests/ scripts/
	poetry run ruff check --fix sentinel/ tests/

typecheck:
	poetry run mypy sentinel/ --ignore-missing-imports

# ─── Cleanup ──────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage dist/ build/

# ─── Docker Build ─────────────────────────────────────────────────────────────
build:
	docker-compose build

rebuild:
	docker-compose down && docker-compose build --no-cache && docker-compose up -d
