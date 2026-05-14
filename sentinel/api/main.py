"""FastAPI application — SENTINEL REST + WebSocket API."""
from __future__ import annotations
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sentinel.core.config import get_settings
from sentinel.core.logging import configure_logging, get_logger
from sentinel.sds import build_default_adapters

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Startup: init adapters + scheduler. Shutdown: stop scheduler."""
    configure_logging()
    s = get_settings()
    logger.info("SENTINEL API starting", version="1.0.0", env=s.environment)
    build_default_adapters()
    from sentinel.sds.scheduler import start_scheduler
    await start_scheduler()
    yield
    from sentinel.sds.scheduler import stop_scheduler
    await stop_scheduler()
    from sentinel.sds.db import dispose_engine
    await dispose_engine()
    logger.info("SENTINEL API shutdown")


app = FastAPI(
    title="SENTINEL Financial Terminal API",
    description="Sovereign AI-native financial terminal — replaces Bloomberg at $0/yr",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and register all route modules
from sentinel.api.routes import data, screen, backtest, portfolio, intelligence, orders, macro, cleaner, futures

app.include_router(data.router,         prefix="/api/v1/data",        tags=["Data"])
app.include_router(screen.router,       prefix="/api/v1/screen",      tags=["Screener"])
app.include_router(backtest.router,     prefix="/api/v1/backtest",     tags=["Backtesting"])
app.include_router(portfolio.router,    prefix="/api/v1/portfolio",    tags=["Portfolio"])
app.include_router(intelligence.router, prefix="/api/v1/intelligence", tags=["Intelligence"])
app.include_router(orders.router,       prefix="/api/v1/orders",       tags=["Orders"])
app.include_router(macro.router,        prefix="/api/v1/macro",        tags=["Macro"])
app.include_router(cleaner.router,      prefix="/api/v1/cleaner",      tags=["DataCleaner"])
app.include_router(futures.router,      prefix="/api/v1/futures",      tags=["Futures"])


@app.get("/health")
async def health():
    from sentinel.sds.normalizer import run_all_health_checks
    events = await run_all_health_checks()
    return {
        "status": "ok",
        "adapters": [
            {"name": e.adapter, "event": e.event_type, "severity": e.severity.value}
            for e in events
        ],
    }


@app.get("/")
async def root():
    return {"name": "SENTINEL", "version": "1.0.0", "docs": "/docs"}
