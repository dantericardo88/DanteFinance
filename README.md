# SENTINEL

Sovereign, AI-native financial terminal — a self-hostable alternative to a Bloomberg/Refinitiv terminal.
Market data, fundamentals, screening, backtesting, portfolio risk, and an AI research layer (RAG + MCP),
served through a web terminal and a REST/MCP API.

## Quick start (Docker)

Requires [Docker Desktop](https://docs.docker.com/get-docker/) (started and running).

```powershell
.\install.ps1
```

The installer creates `.env` (with a generated DB password) if missing, then builds and starts the full
stack. Database schema and migrations run automatically.

Once it's up:

| Service        | URL                          |
|----------------|------------------------------|
| Terminal (UI)  | http://localhost:8501        |
| API docs       | http://localhost:8000/docs   |
| MCP server     | http://localhost:8001        |

Manage the stack:

```powershell
docker compose logs -f      # view logs
docker compose down         # stop
docker compose up -d        # start again (fast, no rebuild)
```

## Configuration

All data/AI providers are optional — the stack boots without them. Add keys to `.env` and run
`docker compose restart` to enable live data and AI features (FRED, Finnhub, Alpha Vantage, Polygon,
Alpaca, Anthropic, Voyage AI). See `.env.example` for the full list.

## Architecture

- **Python 3.12**, FastAPI (API), Streamlit (terminal UI), Poetry for dependency management.
- **PostgreSQL** with TimescaleDB (time-series hypertables) + pgvector (embeddings) + pg_trgm.
- **Redis** for the event bus and caching.
- Service modules live under `sentinel/` (data services, fundamentals engine, backtesting, portfolio
  risk, macro/alt-data, and the AI/intelligence layer).

## Development

```bash
make install        # poetry install
make test           # run the test suite
make lint           # ruff
make typecheck      # mypy
```
