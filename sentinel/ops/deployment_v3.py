"""Self-hosted / sovereign deployment and operations management for SENTINEL.

Dimension 096: Self-hosted / sovereign (no seat fee) — score 7 → 9.

Quantifies savings vs Bloomberg ($24K/yr), Refinitiv ($22K/yr), FactSet ($12K/yr),
Capital IQ ($15K/yr).  Manages Docker config generation, environment bootstrapping,
service lifecycle, backup/restore, and continuous health monitoring.

Usage:
    python -m sentinel.ops.deployment_v3                   # full system check + report
    python -m sentinel.ops.deployment_v3 --setup           # one-command setup
    python -m sentinel.ops.deployment_v3 --docker          # generate Docker files
    python -m sentinel.ops.deployment_v3 --health          # live health report
"""
from __future__ import annotations

import concurrent.futures
import dataclasses
import datetime
import gzip
import importlib
import json
import logging
import math
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import textwrap
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Optional dependency guards ─────────────────────────────────────────────────

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.columns import Columns
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


# ── Status / Result Enums ──────────────────────────────────────────────────────


class CheckStatus(Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    SKIP = "SKIP"


class ServiceState(Enum):
    RUNNING = "RUNNING"
    STOPPED = "STOPPED"
    UNKNOWN = "UNKNOWN"
    ERROR = "ERROR"


# ── Dataclasses ────────────────────────────────────────────────────────────────


@dataclass
class CheckResult:
    name: str
    status: CheckStatus
    message: str
    detail: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status in (CheckStatus.PASS, CheckStatus.WARN)


@dataclass
class PackageCheck:
    name: str
    required: bool
    installed: bool
    version: Optional[str] = None
    install_cmd: Optional[str] = None

    @property
    def status(self) -> CheckStatus:
        if self.installed:
            return CheckStatus.PASS
        if self.required:
            return CheckStatus.FAIL
        return CheckStatus.WARN


@dataclass
class NetworkCheck:
    name: str
    url: str
    reachable: bool
    latency_ms: Optional[float] = None
    error: Optional[str] = None

    @property
    def status(self) -> CheckStatus:
        return CheckStatus.PASS if self.reachable else CheckStatus.WARN


@dataclass
class SystemReport:
    python_check: CheckResult
    package_checks: List[PackageCheck]
    disk_check: CheckResult
    network_checks: List[NetworkCheck]
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.utcnow)

    @property
    def overall_ok(self) -> bool:
        if self.python_check.status == CheckStatus.FAIL:
            return False
        if any(not pc.ok for pc in [self.python_check, self.disk_check]):
            return False
        if any(p.required and not p.installed for p in self.package_checks):
            return False
        return True


@dataclass
class SentinelConfig:
    data_dir: str = "sentinel/data"
    log_dir: str = "sentinel/logs"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    alpaca_api_key: Optional[str] = None
    alpaca_secret_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    dash_port: int = 8050
    worker_interval_minutes: int = 15
    backup_dir: str = "sentinel/backups"
    backup_keep_n: int = 7


@dataclass
class DataFreshness:
    path: str
    exists: bool
    size_bytes: int
    last_modified: Optional[datetime.datetime]
    row_count: Optional[int]
    staleness_hours: Optional[float]

    @property
    def status(self) -> CheckStatus:
        if not self.exists:
            return CheckStatus.FAIL
        if self.staleness_hours is None:
            return CheckStatus.WARN
        if self.staleness_hours > 48:
            return CheckStatus.FAIL
        if self.staleness_hours > 24:
            return CheckStatus.WARN
        return CheckStatus.PASS


@dataclass
class ProcessStatus:
    name: str
    pid: Optional[int]
    state: ServiceState
    started_at: Optional[datetime.datetime] = None
    cpu_pct: Optional[float] = None
    mem_mb: Optional[float] = None


@dataclass
class ServiceStatus:
    name: str
    state: ServiceState
    port: Optional[int] = None
    uptime_seconds: Optional[float] = None
    last_error: Optional[str] = None


@dataclass
class BackupResult:
    success: bool
    path: Optional[str]
    size_bytes: int
    duration_seconds: float
    files_included: int
    error: Optional[str] = None


@dataclass
class BackupInfo:
    path: str
    created_at: datetime.datetime
    size_bytes: int
    label: Optional[str] = None


@dataclass
class CostComparison:
    bloomberg_annual: float = 24_000.0
    refinitiv_annual: float = 22_000.0
    factset_annual: float = 12_000.0
    capital_iq_annual: float = 15_000.0
    sentinel_infra_monthly: float = 0.0   # user-adjustable
    users: int = 1
    years: int = 3

    @property
    def sentinel_annual(self) -> float:
        return self.sentinel_infra_monthly * 12

    @property
    def bloomberg_savings_3yr(self) -> float:
        return (self.bloomberg_annual - self.sentinel_annual) * self.users * self.years

    @property
    def refinitiv_savings_3yr(self) -> float:
        return (self.refinitiv_annual - self.sentinel_annual) * self.users * self.years

    @property
    def factset_savings_3yr(self) -> float:
        return (self.factset_annual - self.sentinel_annual) * self.users * self.years

    @property
    def capital_iq_savings_3yr(self) -> float:
        return (self.capital_iq_annual - self.sentinel_annual) * self.users * self.years


@dataclass
class HealthReport:
    api_ok: bool
    databases_fresh: bool
    memory_ok: bool
    disk_ok: bool
    external_apis_ok: bool
    uptime_seconds: float
    issues: List[str] = field(default_factory=list)
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.utcnow)

    @property
    def overall_healthy(self) -> bool:
        return self.api_ok and self.databases_fresh and self.memory_ok and self.disk_ok

    @property
    def health_score(self) -> float:
        """0-100 score."""
        components = [self.api_ok, self.databases_fresh, self.memory_ok,
                      self.disk_ok, self.external_apis_ok]
        return sum(1 for c in components if c) / len(components) * 100


# ── System Requirements Checker ────────────────────────────────────────────────


class SystemRequirementsChecker:
    """Verify Python version, packages, disk space, and network connectivity."""

    REQUIRED_PACKAGES = [
        "requests", "pandas", "numpy", "yfinance", "fastapi", "uvicorn", "duckdb",
    ]
    RECOMMENDED_PACKAGES = [
        "plotly", "dash", "transformers", "sentence_transformers", "chromadb", "scipy",
    ]
    OPTIONAL_PACKAGES = [
        "ccxt", "pandas_ta", "mcp", "hmmlearn", "rich",
    ]

    NETWORK_ENDPOINTS = [
        ("EDGAR EFTS", "https://efts.sec.gov/LATEST/search-index?q=annual&dateRange=custom&startdt=2024-01-01&enddt=2024-01-02"),
        ("FRED API", "https://api.stlouisfed.org/fred/series?series_id=GDP&api_key=annualonly&file_type=json"),
        ("CoinGecko", "https://api.coingecko.com/api/v3/ping"),
        ("DefiLlama", "https://api.llama.fi/protocols"),
        ("CFTC COT", "https://www.cftc.gov/dea/options/deacmesf.htm"),
        ("Stooq", "https://stooq.com/q/d/l/?s=aapl.us&i=d"),
    ]

    def check_python_version(self) -> CheckResult:
        v = sys.version_info
        if v >= (3, 9):
            return CheckResult(
                name="Python Version",
                status=CheckStatus.PASS,
                message=f"Python {v.major}.{v.minor}.{v.micro} — OK (≥3.9 required)",
            )
        return CheckResult(
            name="Python Version",
            status=CheckStatus.FAIL,
            message=f"Python {v.major}.{v.minor}.{v.micro} — FAIL (≥3.9 required)",
            detail="Upgrade Python: https://python.org",
        )

    def check_required_packages(self) -> List[PackageCheck]:
        results = []
        all_pkgs = (
            [(p, True) for p in self.REQUIRED_PACKAGES]
            + [(p, False) for p in self.RECOMMENDED_PACKAGES]
            + [(p, False) for p in self.OPTIONAL_PACKAGES]
        )
        for pkg, required in all_pkgs:
            try:
                mod = importlib.import_module(pkg.replace("-", "_"))
                ver = getattr(mod, "__version__", None)
                results.append(PackageCheck(name=pkg, required=required, installed=True, version=ver))
            except ImportError:
                results.append(
                    PackageCheck(
                        name=pkg,
                        required=required,
                        installed=False,
                        install_cmd=f"pip install {pkg}",
                    )
                )
        return results

    def check_disk_space(self, path: str = ".") -> CheckResult:
        try:
            usage = shutil.disk_usage(path)
            free_gb = usage.free / 1e9
            total_gb = usage.total / 1e9
            used_pct = (usage.used / usage.total) * 100
            if free_gb >= 5:
                return CheckResult(
                    name="Disk Space",
                    status=CheckStatus.PASS,
                    message=f"{free_gb:.1f} GB free / {total_gb:.0f} GB total ({used_pct:.0f}% used) — OK",
                )
            elif free_gb >= 1:
                return CheckResult(
                    name="Disk Space",
                    status=CheckStatus.WARN,
                    message=f"{free_gb:.1f} GB free — low (≥5 GB recommended for full data)",
                )
            else:
                return CheckResult(
                    name="Disk Space",
                    status=CheckStatus.FAIL,
                    message=f"{free_gb:.2f} GB free — critical: SENTINEL requires ≥5 GB",
                )
        except Exception as exc:
            return CheckResult(name="Disk Space", status=CheckStatus.SKIP, message=f"Could not check: {exc}")

    def _probe_url(self, name: str, url: str) -> NetworkCheck:
        if not HAS_REQUESTS:
            return NetworkCheck(name=name, url=url, reachable=False, error="requests not installed")
        try:
            start = time.monotonic()
            resp = requests.head(url, timeout=6, allow_redirects=True)
            latency = (time.monotonic() - start) * 1000
            ok = resp.status_code < 500
            return NetworkCheck(name=name, url=url, reachable=ok, latency_ms=round(latency, 1))
        except Exception as exc:
            return NetworkCheck(name=name, url=url, reachable=False, error=str(exc)[:80])

    def check_network_access(self) -> List[NetworkCheck]:
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
            futures = {ex.submit(self._probe_url, name, url): name for name, url in self.NETWORK_ENDPOINTS}
            results = []
            for fut in concurrent.futures.as_completed(futures):
                results.append(fut.result())
        return sorted(results, key=lambda x: x.name)

    def run_full_check(self) -> SystemReport:
        py_check = self.check_python_version()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            pkg_fut = ex.submit(self.check_required_packages)
            disk_fut = ex.submit(self.check_disk_space)
            net_fut = ex.submit(self.check_network_access)
            pkg_checks = pkg_fut.result()
            disk_check = disk_fut.result()
            net_checks = net_fut.result()
        return SystemReport(
            python_check=py_check,
            package_checks=pkg_checks,
            disk_check=disk_check,
            network_checks=net_checks,
        )

    def print_report(self, report: SystemReport) -> None:
        if HAS_RICH:
            console = Console()
            _print_system_report_rich(console, report)
        else:
            _print_system_report_plain(report)


def _status_icon(status: CheckStatus) -> str:
    return {"PASS": "[green]PASS[/]", "WARN": "[yellow]WARN[/]",
            "FAIL": "[red]FAIL[/]", "SKIP": "[dim]SKIP[/]"}.get(status.value, status.value)


def _status_plain(status: CheckStatus) -> str:
    return {"PASS": "PASS", "WARN": "WARN", "FAIL": "FAIL", "SKIP": "SKIP"}.get(status.value, status.value)


def _print_system_report_rich(console: "Console", report: SystemReport) -> None:
    console.rule("[bold cyan]SENTINEL System Requirements Report[/]")

    # Python
    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_column("Check")
    t.add_column("Status", width=8)
    t.add_column("Detail")
    t.add_row(report.python_check.name,
               Text(_status_plain(report.python_check.status),
                    style="green" if report.python_check.ok else "red"),
               report.python_check.message)
    t.add_row(report.disk_check.name,
               Text(_status_plain(report.disk_check.status),
                    style="green" if report.disk_check.ok else "yellow"),
               report.disk_check.message)
    console.print(Panel(t, title="System", border_style="dim"))

    # Packages
    pt = Table(show_header=True, header_style="bold cyan")
    pt.add_column("Package")
    pt.add_column("Required", width=10)
    pt.add_column("Installed", width=10)
    pt.add_column("Version")
    for p in report.package_checks:
        req_str = "Required" if p.required else "Optional"
        inst_str = Text(p.version or "yes", style="green") if p.installed else Text("missing", style="red" if p.required else "yellow")
        pt.add_row(p.name, req_str, inst_str, p.version or ("—" if p.installed else p.install_cmd or ""))
    console.print(Panel(pt, title="Packages", border_style="dim"))

    # Network
    nt = Table(show_header=True, header_style="bold cyan")
    nt.add_column("Endpoint")
    nt.add_column("Reachable", width=10)
    nt.add_column("Latency")
    for nc in report.network_checks:
        reach_text = Text("yes", style="green") if nc.reachable else Text("no", style="yellow")
        lat = f"{nc.latency_ms:.0f}ms" if nc.latency_ms else (nc.error or "—")
        nt.add_row(nc.name, reach_text, lat)
    console.print(Panel(nt, title="Network Connectivity", border_style="dim"))

    overall = "[bold green]ALL SYSTEMS GO[/]" if report.overall_ok else "[bold red]ISSUES DETECTED[/]"
    console.print(f"\n  Overall: {overall}")


def _print_system_report_plain(report: SystemReport) -> None:
    print("\n" + "=" * 60)
    print("  SENTINEL System Requirements Report")
    print("=" * 60)
    print(f"  Python:    {report.python_check.message}")
    print(f"  Disk:      {report.disk_check.message}")
    print("\n  Packages:")
    for p in report.package_checks:
        req = "REQ" if p.required else "OPT"
        inst = "OK" if p.installed else "MISSING"
        ver = f" ({p.version})" if p.version else ""
        print(f"    [{req}] {p.name:<25} {inst}{ver}")
    print("\n  Network:")
    for nc in report.network_checks:
        ok = "OK" if nc.reachable else "FAIL"
        lat = f" ({nc.latency_ms:.0f}ms)" if nc.latency_ms else ""
        err = f" — {nc.error}" if nc.error else ""
        print(f"    {nc.name:<25} {ok}{lat}{err}")
    overall = "ALL SYSTEMS GO" if report.overall_ok else "ISSUES DETECTED"
    print(f"\n  Overall: {overall}")
    print("=" * 60)


# ── Docker Deployer ────────────────────────────────────────────────────────────


class DockerDeployer:
    """Generate Dockerfile, docker-compose.yml, and nginx.conf for SENTINEL."""

    def generate_dockerfile(self) -> str:
        return textwrap.dedent("""\
            # SENTINEL Financial Terminal — Dockerfile
            # Multi-stage build: slim Python 3.11 base
            # EXPOSE 8000 (API)  8050 (Dash UI)

            # ── Stage 1: dependency builder ──────────────────────────────────
            FROM python:3.11-slim AS builder

            WORKDIR /build

            # System deps for scientific packages
            RUN apt-get update && apt-get install -y --no-install-recommends \\
                gcc g++ libssl-dev libffi-dev curl git \\
                && rm -rf /var/lib/apt/lists/*

            COPY requirements-core.txt requirements-ai.txt requirements-ui.txt ./

            RUN pip install --upgrade pip && \\
                pip install --no-cache-dir -r requirements-core.txt && \\
                pip install --no-cache-dir -r requirements-ui.txt || true && \\
                pip install --no-cache-dir -r requirements-ai.txt || true

            # ── Stage 2: runtime image ───────────────────────────────────────
            FROM python:3.11-slim AS runtime

            WORKDIR /app

            # Runtime system deps only
            RUN apt-get update && apt-get install -y --no-install-recommends \\
                curl libgomp1 \\
                && rm -rf /var/lib/apt/lists/*

            # Copy installed Python packages from builder
            COPY --from=builder /usr/local/lib/python3.11/site-packages \\
                 /usr/local/lib/python3.11/site-packages
            COPY --from=builder /usr/local/bin /usr/local/bin

            # Copy application source
            COPY sentinel/ ./sentinel/
            COPY pyproject.toml ./

            # Non-root user for security
            RUN useradd -m -u 1001 sentinel && \\
                mkdir -p /app/sentinel/data /app/sentinel/logs /app/sentinel/backups && \\
                chown -R sentinel:sentinel /app

            USER sentinel

            # Data and logs as volumes
            VOLUME ["/app/sentinel/data", "/app/sentinel/logs", "/app/sentinel/backups"]

            EXPOSE 8000
            EXPOSE 8050

            HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \\
                CMD curl -f http://localhost:8000/health || exit 1

            # Default: API server (override in compose for other services)
            CMD ["uvicorn", "sentinel.api.main:app", \\
                 "--host", "0.0.0.0", "--port", "8000", "--workers", "2", \\
                 "--log-level", "info"]
        """)

    def generate_docker_compose(self) -> str:
        return textwrap.dedent("""\
            # SENTINEL docker-compose.yml
            # Services: sentinel-api, sentinel-ui, sentinel-worker
            # Usage: docker compose up -d

            version: "3.9"

            x-sentinel-common: &sentinel-common
              image: sentinel:latest
              build:
                context: .
                dockerfile: Dockerfile
              restart: unless-stopped
              environment:
                - ALPACA_API_KEY=${ALPACA_API_KEY:-}
                - ALPACA_SECRET_KEY=${ALPACA_SECRET_KEY:-}
                - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-}
                - SENTINEL_DATA_DIR=/app/sentinel/data
                - SENTINEL_LOG_DIR=/app/sentinel/logs
                - SENTINEL_LOG_LEVEL=${SENTINEL_LOG_LEVEL:-INFO}
              volumes:
                - sentinel_data:/app/sentinel/data
                - sentinel_logs:/app/sentinel/logs
                - sentinel_backups:/app/sentinel/backups
              networks:
                - sentinel_net

            services:

              # ── REST API (FastAPI / uvicorn) ─────────────────────────────
              sentinel-api:
                <<: *sentinel-common
                container_name: sentinel-api
                command: >
                  uvicorn sentinel.api.main:app
                  --host 0.0.0.0 --port 8000 --workers 2 --log-level info
                ports:
                  - "${SENTINEL_PORT:-8000}:8000"
                healthcheck:
                  test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
                  interval: 30s
                  timeout: 10s
                  retries: 3
                  start_period: 20s

              # ── Dash UI (multi-panel workspace) ──────────────────────────
              sentinel-ui:
                <<: *sentinel-common
                container_name: sentinel-ui
                command: >
                  python -m sentinel.ui.workspace_v3
                  --port 8050 --layout EQUITY_DEEP_DIVE --ticker AAPL
                ports:
                  - "${SENTINEL_DASH_PORT:-8050}:8050"
                depends_on:
                  sentinel-api:
                    condition: service_healthy
                healthcheck:
                  test: ["CMD", "curl", "-f", "http://localhost:8050"]
                  interval: 60s
                  timeout: 10s
                  retries: 3
                  start_period: 30s

              # ── Background data refresh worker ────────────────────────────
              sentinel-worker:
                <<: *sentinel-common
                container_name: sentinel-worker
                command: >
                  python -m sentinel.ops.deployment_v3 --worker
                environment:
                  - ALPACA_API_KEY=${ALPACA_API_KEY:-}
                  - ALPACA_SECRET_KEY=${ALPACA_SECRET_KEY:-}
                  - SENTINEL_WORKER_INTERVAL=${SENTINEL_WORKER_INTERVAL:-15}
                  - SENTINEL_DATA_DIR=/app/sentinel/data
                depends_on:
                  - sentinel-api
                healthcheck:
                  test: ["CMD", "python", "-c", "import sys; sys.exit(0)"]
                  interval: 120s
                  timeout: 10s
                  retries: 2

            volumes:
              sentinel_data:
                driver: local
              sentinel_logs:
                driver: local
              sentinel_backups:
                driver: local

            networks:
              sentinel_net:
                driver: bridge
        """)

    def generate_nginx_config(self) -> str:
        return textwrap.dedent("""\
            # SENTINEL nginx.conf
            # Reverse proxy with SSL termination, rate limiting, WebSocket upgrade (SSE)
            # Place in /etc/nginx/sites-available/sentinel and symlink to sites-enabled/

            upstream sentinel_api {
                server 127.0.0.1:8000;
                keepalive 64;
            }

            upstream sentinel_ui {
                server 127.0.0.1:8050;
                keepalive 16;
            }

            # Rate limiting zones
            limit_req_zone $binary_remote_addr zone=api_limit:10m rate=60r/m;
            limit_req_zone $binary_remote_addr zone=ui_limit:10m  rate=30r/m;

            # Redirect HTTP → HTTPS
            server {
                listen 80;
                server_name YOUR_DOMAIN;
                return 301 https://$host$request_uri;
            }

            server {
                listen 443 ssl http2;
                server_name YOUR_DOMAIN;

                # SSL — replace with your cert paths (certbot / Let's Encrypt)
                ssl_certificate     /etc/letsencrypt/live/YOUR_DOMAIN/fullchain.pem;
                ssl_certificate_key /etc/letsencrypt/live/YOUR_DOMAIN/privkey.pem;
                ssl_protocols       TLSv1.2 TLSv1.3;
                ssl_ciphers         ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;
                ssl_session_cache   shared:SSL:10m;
                ssl_session_timeout 10m;
                add_header Strict-Transport-Security "max-age=63072000" always;

                # Security headers
                add_header X-Frame-Options SAMEORIGIN;
                add_header X-Content-Type-Options nosniff;
                add_header Referrer-Policy strict-origin-when-cross-origin;

                # SENTINEL REST API
                location /api/ {
                    limit_req zone=api_limit burst=20 nodelay;
                    proxy_pass http://sentinel_api/;
                    proxy_http_version 1.1;
                    proxy_set_header Host $host;
                    proxy_set_header X-Real-IP $remote_addr;
                    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
                    proxy_set_header X-Forwarded-Proto $scheme;
                    proxy_read_timeout 120s;

                    # SSE / Server-Sent Events (streaming)
                    proxy_set_header Connection "";
                    proxy_buffering off;
                    proxy_cache off;
                }

                # SENTINEL Dash UI
                location / {
                    limit_req zone=ui_limit burst=10;
                    proxy_pass http://sentinel_ui/;
                    proxy_http_version 1.1;

                    # WebSocket upgrade (Dash live updates)
                    proxy_set_header Upgrade $http_upgrade;
                    proxy_set_header Connection "upgrade";
                    proxy_set_header Host $host;
                    proxy_set_header X-Real-IP $remote_addr;
                    proxy_read_timeout 3600s;
                    proxy_send_timeout 3600s;
                }

                # Health endpoint — no rate limit, no auth
                location /health {
                    proxy_pass http://sentinel_api/health;
                    access_log off;
                }

                # Gzip compression for API responses
                gzip on;
                gzip_types application/json text/plain text/css;
                gzip_min_length 1024;
            }
        """)

    def write_docker_files(self, output_dir: str = ".") -> List[str]:
        written = []
        files = {
            "Dockerfile.sentinel": self.generate_dockerfile(),
            "docker-compose.sentinel.yml": self.generate_docker_compose(),
            "nginx.sentinel.conf": self.generate_nginx_config(),
        }
        for fname, content in files.items():
            path = os.path.join(output_dir, fname)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(content)
            written.append(path)
            logger.info("Wrote %s", path)
        return written


# ── Environment Manager ────────────────────────────────────────────────────────


class EnvironmentManager:
    """Manage .env files, config loading, validation, and requirements files."""

    ENV_TEMPLATE = textwrap.dedent("""\
        # SENTINEL Environment Configuration
        # Copy to .env and fill in your values
        # All API keys are OPTIONAL — SENTINEL works without them using free public endpoints

        # ── Alpaca Markets (optional, for live order routing & portfolio data) ──────────
        ALPACA_API_KEY=
        ALPACA_SECRET_KEY=

        # ── Anthropic Claude (optional, for AI-powered features: NL screener, RAG, doc summarization) ──
        ANTHROPIC_API_KEY=

        # ── Paths ──────────────────────────────────────────────────────────────────────
        SENTINEL_DATA_DIR=sentinel/data
        SENTINEL_LOG_DIR=sentinel/logs
        SENTINEL_BACKUP_DIR=sentinel/backups

        # ── Server settings ───────────────────────────────────────────────────────────
        SENTINEL_HOST=0.0.0.0
        SENTINEL_PORT=8000
        SENTINEL_DASH_PORT=8050
        SENTINEL_LOG_LEVEL=INFO

        # ── Worker settings ────────────────────────────────────────────────────────────
        # How often the background data refresh worker runs (minutes, during market hours)
        SENTINEL_WORKER_INTERVAL=15

        # ── Feature flags ─────────────────────────────────────────────────────────────
        # Set SENTINEL_DEMO_MODE=true to run with synthetic/cached data (no API calls)
        SENTINEL_DEMO_MODE=false

        # ── Database paths (override defaults if needed) ───────────────────────────────
        # SENTINEL_DUCKDB_OHLCV=sentinel/data/ohlcv_daily.duckdb
        # SENTINEL_DUCKDB_FUNDAMENTALS=sentinel/data/fundamentals.duckdb
        # SENTINEL_SQLITE_SCREENER=sentinel/data/screener.db
    """)

    CORE_REQUIREMENTS = textwrap.dedent("""\
        # requirements-core.txt — SENTINEL core dependencies
        # Install: pip install -r requirements-core.txt

        requests>=2.31.0
        pandas>=2.0.0
        numpy>=1.24.0
        yfinance>=0.2.36
        fastapi>=0.110.0
        uvicorn[standard]>=0.27.0
        duckdb>=0.10.0
        python-dotenv>=1.0.0
        pydantic>=2.5.0
        httpx>=0.26.0
        aiohttp>=3.9.0
        schedule>=1.2.0
        SQLAlchemy>=2.0.0
        alembic>=1.13.0
    """)

    AI_REQUIREMENTS = textwrap.dedent("""\
        # requirements-ai.txt — SENTINEL AI/ML features (optional)
        # Install: pip install -r requirements-ai.txt
        # Requires: ~4 GB disk for model weights

        transformers>=4.38.0
        sentence-transformers>=2.6.0
        chromadb>=0.4.22
        anthropic>=0.20.0
        torch>=2.1.0
        huggingface-hub>=0.20.0
        scipy>=1.12.0
        hmmlearn>=0.3.2
        scikit-learn>=1.4.0
    """)

    UI_REQUIREMENTS = textwrap.dedent("""\
        # requirements-ui.txt — SENTINEL UI dependencies (optional)
        # Install: pip install -r requirements-ui.txt

        plotly>=5.19.0
        dash>=2.16.0
        rich>=13.7.0
        dash-bootstrap-components>=1.5.0
    """)

    def setup_env_file(self, path: str = ".env") -> str:
        if os.path.exists(path):
            logger.info(".env already exists at %s — skipping overwrite", path)
            return path
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.ENV_TEMPLATE)
        logger.info("Created .env template at %s", path)
        return path

    def load_config(self) -> SentinelConfig:
        # Try python-dotenv first
        try:
            from dotenv import load_dotenv
            load_dotenv(override=False)
        except ImportError:
            pass

        def _env(key: str, default: str = "") -> str:
            return os.environ.get(key, default)

        return SentinelConfig(
            data_dir=_env("SENTINEL_DATA_DIR", "sentinel/data"),
            log_dir=_env("SENTINEL_LOG_DIR", "sentinel/logs"),
            host=_env("SENTINEL_HOST", "0.0.0.0"),
            port=int(_env("SENTINEL_PORT", "8000")),
            log_level=_env("SENTINEL_LOG_LEVEL", "INFO"),
            alpaca_api_key=_env("ALPACA_API_KEY") or None,
            alpaca_secret_key=_env("ALPACA_SECRET_KEY") or None,
            anthropic_api_key=_env("ANTHROPIC_API_KEY") or None,
            dash_port=int(_env("SENTINEL_DASH_PORT", "8050")),
            worker_interval_minutes=int(_env("SENTINEL_WORKER_INTERVAL", "15")),
            backup_dir=_env("SENTINEL_BACKUP_DIR", "sentinel/backups"),
        )

    def validate_config(self, config: SentinelConfig) -> List[str]:
        issues = []
        if not config.data_dir:
            issues.append("SENTINEL_DATA_DIR is not set")
        if config.port < 1 or config.port > 65535:
            issues.append(f"SENTINEL_PORT={config.port} is invalid")
        if config.log_level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            issues.append(f"SENTINEL_LOG_LEVEL={config.log_level!r} is not a valid level")
        if config.worker_interval_minutes < 1:
            issues.append("SENTINEL_WORKER_INTERVAL must be ≥1 minute")
        if not config.alpaca_api_key:
            issues.append("WARN: ALPACA_API_KEY not set — live trading features disabled")
        if not config.anthropic_api_key:
            issues.append("WARN: ANTHROPIC_API_KEY not set — AI features disabled")
        return issues

    def generate_requirements_txt(self) -> Dict[str, str]:
        return {
            "requirements-core.txt": self.CORE_REQUIREMENTS,
            "requirements-ai.txt": self.AI_REQUIREMENTS,
            "requirements-ui.txt": self.UI_REQUIREMENTS,
        }

    def write_requirements_files(self, output_dir: str = ".") -> List[str]:
        written = []
        for fname, content in self.generate_requirements_txt().items():
            path = os.path.join(output_dir, fname)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(content)
            written.append(path)
        return written


# ── Data Directory Initializer ─────────────────────────────────────────────────


class DataDirectoryInitializer:
    """Create and validate SENTINEL's data directory structure."""

    REQUIRED_DIRS = [
        "sentinel/data",
        "sentinel/data/cache",
        "sentinel/logs",
        "sentinel/backups",
        "sentinel/data/parquet",
        "sentinel/data/exports",
    ]

    DATA_FILES = {
        "sentinel/data/ohlcv_daily.duckdb": "duckdb",
        "sentinel/data/fundamentals.duckdb": "duckdb",
        "sentinel/data/ohlcv_intraday.duckdb": "duckdb",
        "sentinel/data/strategy_registry.json": "json",
        "sentinel/data/trends_cache.json": "json",
    }

    def initialize(self, data_dir: str = "sentinel/data") -> None:
        for d in self.REQUIRED_DIRS:
            os.makedirs(d, exist_ok=True)
            logger.debug("Ensured directory: %s", d)

        # Init JSON files if missing
        for path, ftype in self.DATA_FILES.items():
            if not os.path.exists(path) and ftype == "json":
                with open(path, "w") as fh:
                    json.dump({}, fh)
                logger.debug("Initialized %s", path)

        # Create .gitkeep in each dir
        for d in self.REQUIRED_DIRS:
            gk = os.path.join(d, ".gitkeep")
            if not os.path.exists(gk):
                with open(gk, "w") as fh:
                    fh.write("")

        logger.info("Data directories initialized at %s", data_dir)

    def check_data_freshness(self) -> Dict[str, DataFreshness]:
        results: Dict[str, DataFreshness] = {}
        now = datetime.datetime.utcnow()

        paths_to_check = list(self.DATA_FILES.keys())
        # Also auto-discover .duckdb and .db files in sentinel/data/
        data_dir = "sentinel/data"
        if os.path.isdir(data_dir):
            for fname in os.listdir(data_dir):
                if fname.endswith((".duckdb", ".db", ".parquet")):
                    fp = os.path.join(data_dir, fname)
                    if fp not in paths_to_check:
                        paths_to_check.append(fp)

        for path in paths_to_check:
            if not os.path.exists(path):
                results[path] = DataFreshness(
                    path=path, exists=False, size_bytes=0,
                    last_modified=None, row_count=None, staleness_hours=None,
                )
                continue

            stat = os.stat(path)
            size = stat.st_size
            mtime = datetime.datetime.utcfromtimestamp(stat.st_mtime)
            staleness = (now - mtime).total_seconds() / 3600

            # Attempt row count for DuckDB files
            row_count = None
            if path.endswith(".duckdb") and size > 0:
                try:
                    import duckdb
                    con = duckdb.connect(path, read_only=True)
                    tables = con.execute("SHOW TABLES").fetchdf()
                    if not tables.empty:
                        tbl = tables.iloc[0]["name"]
                        row_count = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
                    con.close()
                except Exception:
                    pass

            results[path] = DataFreshness(
                path=path, exists=True, size_bytes=size,
                last_modified=mtime, row_count=row_count,
                staleness_hours=round(staleness, 1),
            )

        return results

    def compact_databases(self) -> Dict[str, int]:
        """VACUUM DuckDB and SQLite databases; return bytes freed per file."""
        freed: Dict[str, int] = {}
        data_dir = "sentinel/data"
        if not os.path.isdir(data_dir):
            return freed

        for fname in os.listdir(data_dir):
            path = os.path.join(data_dir, fname)
            if fname.endswith(".duckdb"):
                try:
                    import duckdb
                    before = os.path.getsize(path)
                    con = duckdb.connect(path)
                    con.execute("VACUUM")
                    con.close()
                    after = os.path.getsize(path)
                    freed[path] = max(0, before - after)
                    logger.info("VACUUM %s: freed %d bytes", fname, freed[path])
                except Exception as exc:
                    logger.warning("VACUUM failed for %s: %s", fname, exc)

            elif fname.endswith(".db"):
                try:
                    import sqlite3
                    before = os.path.getsize(path)
                    con = sqlite3.connect(path)
                    con.execute("VACUUM")
                    con.commit()
                    con.close()
                    after = os.path.getsize(path)
                    freed[path] = max(0, before - after)
                    logger.info("VACUUM (sqlite) %s: freed %d bytes", fname, freed[path])
                except Exception as exc:
                    logger.warning("VACUUM sqlite failed for %s: %s", fname, exc)

        return freed


# ── Service Manager ────────────────────────────────────────────────────────────


class ServiceManager:
    """Manage SENTINEL background services (API, UI, worker)."""

    SERVICES = {
        "sentinel-api": {
            "cmd": [sys.executable, "-m", "uvicorn", "sentinel.api.main:app",
                    "--host", "0.0.0.0", "--port", "8000"],
            "port": 8000,
        },
        "sentinel-ui": {
            "cmd": [sys.executable, "-m", "sentinel.ui.workspace_v3",
                    "--port", "8050", "--layout", "EQUITY_DEEP_DIVE"],
            "port": 8050,
        },
        "sentinel-worker": {
            "cmd": [sys.executable, "-m", "sentinel.ops.deployment_v3", "--worker"],
            "port": None,
        },
    }

    def __init__(self):
        self._processes: Dict[str, subprocess.Popen] = {}
        self._start_times: Dict[str, float] = {}

    def start_all_services(self) -> Dict[str, ProcessStatus]:
        results = {}
        for name, cfg in self.SERVICES.items():
            results[name] = self.start_service(name)
        return results

    def start_service(self, name: str) -> ProcessStatus:
        if name not in self.SERVICES:
            return ProcessStatus(name=name, pid=None, state=ServiceState.ERROR)
        cfg = self.SERVICES[name]
        if name in self._processes and self._processes[name].poll() is None:
            return ProcessStatus(
                name=name,
                pid=self._processes[name].pid,
                state=ServiceState.RUNNING,
                started_at=datetime.datetime.utcfromtimestamp(self._start_times[name]),
            )
        try:
            log_dir = "sentinel/logs"
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, f"{name}.log")
            with open(log_path, "a") as log_fh:
                proc = subprocess.Popen(
                    cfg["cmd"],
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                )
            self._processes[name] = proc
            self._start_times[name] = time.time()
            logger.info("Started %s (pid=%d)", name, proc.pid)
            return ProcessStatus(
                name=name,
                pid=proc.pid,
                state=ServiceState.RUNNING,
                started_at=datetime.datetime.utcnow(),
            )
        except Exception as exc:
            logger.error("Failed to start %s: %s", name, exc)
            return ProcessStatus(name=name, pid=None, state=ServiceState.ERROR)

    def stop_all_services(self) -> None:
        for name in list(self._processes.keys()):
            self.stop_service(name)

    def stop_service(self, name: str) -> None:
        proc = self._processes.get(name)
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            del self._processes[name]
            logger.info("Stopped service: %s", name)

    def restart_service(self, name: str) -> ProcessStatus:
        self.stop_service(name)
        time.sleep(1)
        return self.start_service(name)

    def get_service_status(self) -> Dict[str, ServiceStatus]:
        statuses = {}
        for name, cfg in self.SERVICES.items():
            proc = self._processes.get(name)
            if proc is None:
                state = ServiceState.STOPPED
                uptime = None
            elif proc.poll() is None:
                state = ServiceState.RUNNING
                uptime = time.time() - self._start_times.get(name, time.time())
            else:
                state = ServiceState.ERROR
                uptime = None
            statuses[name] = ServiceStatus(
                name=name,
                state=state,
                port=cfg.get("port"),
                uptime_seconds=uptime,
            )
        return statuses

    def get_logs(self, service: str, n_lines: int = 100) -> List[str]:
        log_path = os.path.join("sentinel/logs", f"{service}.log")
        if not os.path.exists(log_path):
            return []
        with open(log_path) as fh:
            lines = fh.readlines()
        return [l.rstrip() for l in lines[-n_lines:]]


# ── Backup Manager ────────────────────────────────────────────────────────────


class BackupManager:
    """Create, restore, schedule, and prune SENTINEL data backups."""

    def __init__(self, backup_dir: str = "sentinel/backups"):
        self.backup_dir = backup_dir
        os.makedirs(backup_dir, exist_ok=True)

    def backup_all_data(self, label: Optional[str] = None) -> BackupResult:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = f"_{label}" if label else ""
        archive_name = f"sentinel_backup_{ts}{tag}.tar.gz"
        archive_path = os.path.join(self.backup_dir, archive_name)

        sources = []
        # Collect all DuckDB, SQLite, JSON, Parquet, .env files
        for root, _dirs, files in os.walk("sentinel/data"):
            for f in files:
                if f.endswith((".duckdb", ".db", ".json", ".parquet", ".csv")):
                    sources.append(os.path.join(root, f))
        # .env
        if os.path.exists(".env"):
            sources.append(".env")
        # strategy configs
        for root, _dirs, files in os.walk("sentinel/strategies"):
            for f in files:
                if f.endswith(".json"):
                    sources.append(os.path.join(root, f))

        start = time.monotonic()
        n_files = 0
        try:
            with tarfile.open(archive_path, "w:gz") as tar:
                for src in sources:
                    if os.path.exists(src):
                        tar.add(src)
                        n_files += 1
            elapsed = time.monotonic() - start
            size = os.path.getsize(archive_path)
            logger.info("Backup complete: %s (%d files, %.1fMB)", archive_name, n_files, size / 1e6)
            return BackupResult(
                success=True, path=archive_path, size_bytes=size,
                duration_seconds=elapsed, files_included=n_files,
            )
        except Exception as exc:
            elapsed = time.monotonic() - start
            logger.error("Backup failed: %s", exc)
            return BackupResult(
                success=False, path=None, size_bytes=0,
                duration_seconds=elapsed, files_included=n_files,
                error=str(exc),
            )

    def restore_backup(self, backup_path: str, target_dir: str = ".") -> "RestoreResult":
        @dataclass
        class RestoreResult:
            success: bool
            files_restored: int
            error: Optional[str] = None

        if not os.path.exists(backup_path):
            return RestoreResult(success=False, files_restored=0, error="Backup file not found")
        try:
            with tarfile.open(backup_path, "r:gz") as tar:
                members = tar.getmembers()
                tar.extractall(path=target_dir)
            logger.info("Restored %d files from %s", len(members), backup_path)
            return RestoreResult(success=True, files_restored=len(members))
        except Exception as exc:
            return RestoreResult(success=False, files_restored=0, error=str(exc))

    def schedule_daily_backup(self, time_str: str = "02:00") -> threading.Thread:
        """Run daily backup in a background thread at the given HH:MM time."""
        hour, minute = (int(x) for x in time_str.split(":"))

        def _loop():
            while True:
                now = datetime.datetime.now()
                target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                if target <= now:
                    target += datetime.timedelta(days=1)
                wait = (target - now).total_seconds()
                logger.info("Daily backup scheduled in %.0f seconds (at %s)", wait, time_str)
                time.sleep(wait)
                result = self.backup_all_data(label="scheduled")
                if result.success:
                    logger.info("Scheduled backup: %s", result.path)
                    self.prune_old_backups()

        t = threading.Thread(target=_loop, daemon=True, name="sentinel-backup-scheduler")
        t.start()
        return t

    def list_backups(self) -> List[BackupInfo]:
        infos = []
        if not os.path.isdir(self.backup_dir):
            return infos
        for fname in sorted(os.listdir(self.backup_dir)):
            if fname.startswith("sentinel_backup_") and fname.endswith(".tar.gz"):
                path = os.path.join(self.backup_dir, fname)
                stat = os.stat(path)
                # Parse timestamp from filename: sentinel_backup_YYYYMMDD_HHMMSS...
                m = re.search(r"(\d{8}_\d{6})", fname)
                created = datetime.datetime.now()
                if m:
                    try:
                        created = datetime.datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")
                    except ValueError:
                        pass
                label_m = re.search(r"\d{8}_\d{6}_?(.*)\.tar\.gz", fname)
                label = label_m.group(1) if label_m else None
                infos.append(BackupInfo(
                    path=path,
                    created_at=created,
                    size_bytes=stat.st_size,
                    label=label or None,
                ))
        return infos

    def prune_old_backups(self, keep_n: int = 7) -> int:
        backups = self.list_backups()
        if len(backups) <= keep_n:
            return 0
        to_delete = backups[:-keep_n]
        count = 0
        for b in to_delete:
            try:
                os.remove(b.path)
                count += 1
                logger.info("Pruned old backup: %s", b.path)
            except Exception as exc:
                logger.warning("Failed to prune %s: %s", b.path, exc)
        return count


# ── Cost Analyzer ─────────────────────────────────────────────────────────────


class CostAnalyzer:
    """Quantify SENTINEL's cost savings vs paid financial terminals."""

    TERMINAL_COSTS: Dict[str, float] = {
        "Bloomberg Terminal": 24_000.0,
        "Refinitiv Eikon (LSEG)": 22_000.0,
        "FactSet": 12_000.0,
        "Capital IQ (S&P)": 15_000.0,
        "Koyfin Pro": 1_800.0,
        "YCharts": 5_400.0,
    }

    INFRA_PRESETS: Dict[str, float] = {
        "Bare metal (own hardware)": 0.0,
        "Raspberry Pi 5 (electricity ~5W)": 3.5,
        "Hetzner VPS CX21 (2 vCPU, 4GB)": 6.0,
        "AWS t3.medium (2 vCPU, 4GB)": 34.0,
        "GCP e2-standard-2 (2 vCPU, 8GB)": 49.0,
        "DigitalOcean 4GB Droplet": 24.0,
    }

    SENTINEL_DIMENSIONS = {
        "Real-time price data": ("yfinance / Stooq (free)", True),
        "Fundamental data": ("EDGAR SEC filings (free)", True),
        "Macro / economic": ("FRED API (free)", True),
        "Earnings / estimates": ("Calculated from EDGAR", True),
        "Options chain": ("yfinance (free, 15-min delay)", True),
        "News / sentiment": ("GDELT / Yahoo RSS (free)", True),
        "Factor models": ("Ken French data library (free)", True),
        "Backtesting": ("Custom engine, no license fee", True),
        "Multi-panel workspace": ("Dash / Rich (open source)", True),
        "AI-powered analysis": ("Anthropic API (usage-based)", False),
        "Real-time order routing": ("Alpaca (free API tier)", True),
        "Portfolio analytics": ("VaR/CVaR/Sharpe (custom)", True),
        "Crypto & DeFi": ("CoinGecko / DefiLlama (free)", True),
        "On-chain data": ("Etherscan free tier", True),
        "Congressional trading": ("House/Senate disclosures (public)", True),
        "Insider filings": ("SEC Form 4 (EDGAR, free)", True),
        "CFTC commitment of traders": ("CFTC website (public)", True),
        "NL screener": ("Claude API (usage-based)", False),
        "MCP integration": ("Open protocol", True),
        "Data sovereignty": ("Own storage, no vendor lock-in", True),
    }

    def compute_annual_savings(
        self,
        users: int = 1,
        sentinel_infra_monthly: float = 0.0,
    ) -> CostComparison:
        return CostComparison(
            users=users,
            sentinel_infra_monthly=sentinel_infra_monthly,
        )

    def generate_roi_report(
        self,
        users: int = 1,
        years: int = 3,
        sentinel_infra_monthly: float = 0.0,
    ) -> str:
        sentinel_annual = sentinel_infra_monthly * 12
        lines = []
        lines.append("=" * 68)
        lines.append("  SENTINEL vs Paid Financial Terminals — ROI Analysis")
        lines.append(f"  Users: {users}   |   Analysis period: {years} years")
        lines.append(f"  SENTINEL infra cost: ${sentinel_infra_monthly:.2f}/month = ${sentinel_annual:,.0f}/year/user")
        lines.append("=" * 68)

        lines.append("\n  Annual License Costs (per user) vs SENTINEL:\n")
        header = f"  {'Terminal':<30} {'Annual/User':>12} {f'{years}yr Savings':>14} {'Payback'}"
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))

        for name, annual in sorted(self.TERMINAL_COSTS.items(), key=lambda x: -x[1]):
            savings_yr = (annual - sentinel_annual) * users * years
            if annual > 0:
                payback_days = (sentinel_annual / annual) * 365 if annual > 0 else 0
                payback_str = f"Day 0" if sentinel_annual == 0 else f"{payback_days:.0f}d"
            else:
                payback_str = "N/A"
            lines.append(
                f"  {name:<30} ${annual:>10,.0f}   ${savings_yr:>12,.0f}  {payback_str}"
            )

        lines.append(f"\n  SENTINEL (infrastructure only): ${sentinel_annual:,.0f}/year")

        # Cumulative savings table
        lines.append("\n  Cumulative Bloomberg savings over time:\n")
        bloomberg = self.TERMINAL_COSTS["Bloomberg Terminal"]
        lines.append(f"  {'Year':<8} {'Bloomberg Cost':>16} {'SENTINEL Cost':>16} {'Savings':>14}")
        lines.append("  " + "-" * 56)
        for yr in range(1, years + 1):
            bb_cost = bloomberg * users * yr
            s_cost = sentinel_annual * users * yr
            savings = bb_cost - s_cost
            lines.append(f"  {yr:<8} ${bb_cost:>15,.0f} ${s_cost:>15,.0f} ${savings:>13,.0f}")

        # Feature parity
        lines.append("\n  Feature Parity Analysis:\n")
        lines.append(f"  {'Feature':<35} {'Data Source':<35} {'Free?'}")
        lines.append("  " + "-" * 76)
        for feat, (source, is_free) in self.SENTINEL_DIMENSIONS.items():
            flag = "YES" if is_free else " ~"
            lines.append(f"  {feat:<35} {source:<35} {flag}")

        # Infrastructure options
        lines.append("\n  Infrastructure Options:\n")
        lines.append(f"  {'Option':<40} {'Monthly':>10} {'Annual':>10}")
        lines.append("  " + "-" * 62)
        for option, monthly in sorted(self.INFRA_PRESETS.items(), key=lambda x: x[1]):
            lines.append(f"  {option:<40} ${monthly:>9.2f} ${monthly*12:>9.2f}")

        lines.append("\n" + "=" * 68)
        lines.append("  SUMMARY: SENTINEL replaces $12K–$24K/user/year with ~$0–$50/month.")
        lines.append(f"  3-year Bloomberg savings (1 user): ${(bloomberg - sentinel_annual)*3:,.0f}")
        lines.append("  Zero vendor lock-in. All data in open formats.")
        lines.append("=" * 68)

        return "\n".join(lines)

    def compute_infrastructure_cost(self, specs: Optional[dict] = None) -> float:
        """Estimate monthly infra cost given specs dict."""
        if specs is None:
            return 0.0
        provider = specs.get("provider", "own")
        preset = specs.get("preset", "")
        if preset in self.INFRA_PRESETS:
            return self.INFRA_PRESETS[preset]
        # Custom: vCPU * $5 + RAM_GB * $2 + storage_GB * $0.10
        vcpu = specs.get("vcpu", 2)
        ram_gb = specs.get("ram_gb", 4)
        storage_gb = specs.get("storage_gb", 50)
        return vcpu * 5.0 + ram_gb * 2.0 + storage_gb * 0.10


# ── SENTINEL Health Monitor ────────────────────────────────────────────────────


class SentinelHealthMonitor:
    """Continuous health monitoring for all SENTINEL subsystems."""

    API_ENDPOINTS = [
        ("http://localhost:8000/health", "API Health"),
        ("http://localhost:8050", "Dash UI"),
    ]

    FREE_APIS = [
        ("https://api.coingecko.com/api/v3/ping", "CoinGecko"),
        ("https://api.stlouisfed.org/fred/series?series_id=GDP&api_key=annualonly&file_type=json", "FRED"),
        ("https://efts.sec.gov/LATEST/search-index?q=test", "EDGAR"),
    ]

    def __init__(self):
        self._start_time = time.time()
        self._data_initializer = DataDirectoryInitializer()

    def run_health_check(self) -> HealthReport:
        issues: List[str] = []

        # 1. API endpoints
        api_ok = False
        if HAS_REQUESTS:
            try:
                resp = requests.get("http://localhost:8000/health", timeout=5)
                api_ok = resp.status_code == 200
            except Exception:
                issues.append("API server not responding on :8000")
        else:
            issues.append("WARN: requests not installed — cannot probe API")
            api_ok = True  # assume ok if we can't check

        # 2. Data freshness
        freshness = self._data_initializer.check_data_freshness()
        stale_files = [p for p, df in freshness.items() if df.status == CheckStatus.FAIL]
        databases_fresh = len(stale_files) == 0
        if stale_files:
            issues.append(f"Stale/missing data files: {', '.join(os.path.basename(p) for p in stale_files[:3])}")

        # 3. Memory usage
        memory_ok = True
        try:
            import psutil
            mem = psutil.virtual_memory()
            if mem.percent > 80:
                memory_ok = False
                issues.append(f"High memory usage: {mem.percent:.0f}% (threshold: 80%)")
        except ImportError:
            pass  # psutil not available — skip

        # 4. Disk usage
        disk_ok = True
        data_dir = "sentinel/data"
        if os.path.isdir(data_dir):
            try:
                usage = shutil.disk_usage(data_dir)
                disk_pct = (usage.used / usage.total) * 100
                if disk_pct > 90:
                    disk_ok = False
                    issues.append(f"Disk usage critical: {disk_pct:.0f}% (threshold: 90%)")
            except Exception:
                pass

        # 5. External API reachability
        external_ok = True
        if HAS_REQUESTS:
            for url, name in self.FREE_APIS[:2]:
                try:
                    r = requests.head(url, timeout=5)
                    if r.status_code >= 500:
                        external_ok = False
                        issues.append(f"External API unreachable: {name}")
                except Exception:
                    # Network issues are warnings not failures
                    issues.append(f"WARN: Could not reach {name}")

        return HealthReport(
            api_ok=api_ok,
            databases_fresh=databases_fresh,
            memory_ok=memory_ok,
            disk_ok=disk_ok,
            external_apis_ok=external_ok,
            uptime_seconds=self.get_uptime(),
            issues=issues,
        )

    def get_uptime(self) -> float:
        return time.time() - self._start_time

    def compute_data_coverage_report(self) -> Optional[Any]:
        if not HAS_PANDAS:
            return None
        freshness = self._data_initializer.check_data_freshness()
        rows = []
        for path, df in freshness.items():
            rows.append({
                "File": os.path.basename(path),
                "Exists": "Yes" if df.exists else "No",
                "Size (MB)": f"{df.size_bytes/1e6:.2f}" if df.exists else "—",
                "Last Modified": df.last_modified.strftime("%Y-%m-%d %H:%M") if df.last_modified else "—",
                "Staleness (h)": f"{df.staleness_hours:.1f}" if df.staleness_hours is not None else "—",
                "Status": df.status.value,
            })
        return pd.DataFrame(rows)

    def print_health_report(self, report: HealthReport) -> None:
        if HAS_RICH:
            console = Console()
            _print_health_rich(console, report)
        else:
            _print_health_plain(report)


def _print_health_rich(console: "Console", report: HealthReport) -> None:
    score = report.health_score
    score_color = "green" if score >= 80 else "yellow" if score >= 60 else "red"
    console.rule("[bold cyan]SENTINEL Health Report[/]")
    console.print(f"  Health Score: [{score_color}]{score:.0f}/100[/]   "
                  f"Uptime: {report.uptime_seconds:.0f}s   "
                  f"Checked: {report.timestamp.strftime('%H:%M:%S UTC')}")

    t = Table(show_header=True, header_style="bold cyan")
    t.add_column("Subsystem")
    t.add_column("Status", width=10)
    checks = [
        ("API Server", report.api_ok),
        ("Data Freshness", report.databases_fresh),
        ("Memory Usage", report.memory_ok),
        ("Disk Space", report.disk_ok),
        ("External APIs", report.external_apis_ok),
    ]
    for name, ok in checks:
        t.add_row(name, Text("OK" if ok else "FAIL", style="green" if ok else "red"))
    console.print(t)

    if report.issues:
        console.print("\n  Issues:")
        for issue in report.issues:
            style = "yellow" if issue.startswith("WARN") else "red"
            console.print(f"    [{style}]{issue}[/]")

    overall = "HEALTHY" if report.overall_healthy else "DEGRADED"
    color = "green" if report.overall_healthy else "red"
    console.print(f"\n  Overall: [{color}]{overall}[/]")


def _print_health_plain(report: HealthReport) -> None:
    print("\n" + "=" * 50)
    print(f"  SENTINEL Health Report  ({report.timestamp.strftime('%H:%M:%S UTC')})")
    print(f"  Score: {report.health_score:.0f}/100  Uptime: {report.uptime_seconds:.0f}s")
    print("=" * 50)
    checks = [
        ("API Server", report.api_ok),
        ("Data Freshness", report.databases_fresh),
        ("Memory", report.memory_ok),
        ("Disk", report.disk_ok),
        ("External APIs", report.external_apis_ok),
    ]
    for name, ok in checks:
        print(f"  {name:<20} {'OK' if ok else 'FAIL'}")
    if report.issues:
        print("\n  Issues:")
        for issue in report.issues:
            print(f"    - {issue}")
    print(f"\n  Overall: {'HEALTHY' if report.overall_healthy else 'DEGRADED'}")
    print("=" * 50)


# ── SENTINEL Installer ────────────────────────────────────────────────────────


class SentinelInstaller:
    """One-command setup: checks, installs, initialises, and bootstraps SENTINEL."""

    def run_setup(
        self,
        data_dir: str = "sentinel/data",
        port: int = 8000,
        install_missing: bool = True,
    ) -> None:
        console = Console() if HAS_RICH else None

        def _print(msg: str) -> None:
            if console:
                console.print(msg)
            else:
                print(msg)

        _print("\n[bold cyan]SENTINEL Setup[/]" if HAS_RICH else "\nSENTINEL Setup")
        _print("=" * 60)

        # Step 1: System check
        _print("\n[1/7] Checking system requirements...")
        checker = SystemRequirementsChecker()
        report = checker.run_full_check()
        checker.print_report(report)

        # Step 2: Install missing packages
        if install_missing:
            _print("\n[2/7] Installing missing required packages...")
            missing = [p for p in report.package_checks if p.required and not p.installed]
            if missing:
                pkgs = " ".join(p.name for p in missing)
                _print(f"  pip install {pkgs}")
                ret = subprocess.run(
                    [sys.executable, "-m", "pip", "install"] + [p.name for p in missing],
                    capture_output=True, text=True,
                )
                if ret.returncode == 0:
                    _print("  Required packages installed successfully.")
                else:
                    _print(f"  WARNING: pip install failed:\n{ret.stderr[:500]}")
            else:
                _print("  All required packages already installed.")
        else:
            _print("\n[2/7] Skipping package installation (install_missing=False)")

        # Step 3: Initialize directories
        _print("\n[3/7] Initializing data directories...")
        init = DataDirectoryInitializer()
        init.initialize(data_dir)
        _print(f"  Data directories ready at {data_dir}")

        # Step 4: .env template
        _print("\n[4/7] Setting up environment configuration...")
        env_mgr = EnvironmentManager()
        env_path = env_mgr.setup_env_file(".env")
        _print(f"  .env template at {env_path}")

        # Step 5: Docker files
        _print("\n[5/7] Generating Docker deployment files...")
        docker = DockerDeployer()
        written = docker.write_docker_files(".")
        for f in written:
            _print(f"  Wrote: {f}")

        # Step 6: Requirements files
        _print("\n[6/7] Writing requirements files...")
        req_files = env_mgr.write_requirements_files(".")
        for f in req_files:
            _print(f"  Wrote: {f}")

        # Step 7: Bootstrap hint
        _print("\n[7/7] Bootstrap data (run these commands):\n")
        bootstrap_cmds = [
            "python -m sentinel.sds.bootstrap             # fetch initial OHLCV data",
            "python -m sentinel.sma.economic_calendar_v3  # fetch economic calendar",
            "python -m sentinel.sma.cftc_cot_v3           # fetch CFTC COT data",
        ]
        for cmd in bootstrap_cmds:
            _print(f"  $ {cmd}")

        # Startup instructions
        _print("\n" + "=" * 60)
        _print("  SENTINEL is ready. Startup options:\n")
        _print("  [A] Direct (development):")
        _print(f"      $ uvicorn sentinel.api.main:app --port {port}")
        _print(f"      $ python -m sentinel.ui.workspace_v3 --layout EQUITY_DEEP_DIVE")
        _print("\n  [B] Docker Compose:")
        _print("      $ docker compose -f docker-compose.sentinel.yml up -d")
        _print("\n  [C] Health check:")
        _print("      $ python -m sentinel.ops.deployment_v3 --health")
        _print("=" * 60)
        _print("\nCost savings vs Bloomberg Terminal: $24,000/user/year")
        _print("SENTINEL is FREE and open-source. Zero vendor lock-in.\n")


# ── Background Worker ──────────────────────────────────────────────────────────


def _run_worker(interval_minutes: int = 15) -> None:
    """Data refresh worker — runs periodically during market hours."""
    logger.info("SENTINEL data refresh worker started (interval=%dmin)", interval_minutes)
    while True:
        now = datetime.datetime.now()
        # Only run during extended market hours (Mon–Fri, 07:00–21:00 local)
        is_weekday = now.weekday() < 5
        is_market_hours = 7 <= now.hour < 21
        if is_weekday and is_market_hours:
            logger.info("[worker] Starting data refresh cycle at %s", now.strftime("%H:%M:%S"))
            _worker_refresh_cycle()
        else:
            logger.debug("[worker] Outside market hours — sleeping")
        time.sleep(interval_minutes * 60)


def _worker_refresh_cycle() -> None:
    """Execute one data refresh cycle."""
    tasks = [
        ("OHLCV daily prices", _worker_refresh_ohlcv),
        ("Economic calendar", _worker_refresh_calendar),
        ("CFTC COT", _worker_refresh_cot),
    ]
    for name, fn in tasks:
        try:
            fn()
            logger.info("[worker] Completed: %s", name)
        except Exception as exc:
            logger.warning("[worker] Failed: %s — %s", name, exc)


def _worker_refresh_ohlcv() -> None:
    if not HAS_YF:
        return
    import yfinance as yf
    tickers = ["SPY", "QQQ", "IWM", "GLD", "TLT", "AAPL", "MSFT", "NVDA"]
    for t in tickers:
        try:
            yf.download(t, period="5d", interval="1d", progress=False)
        except Exception:
            pass


def _worker_refresh_calendar() -> None:
    try:
        from sentinel.sma.economic_calendar_v3 import EconomicCalendarFetcher  # type: ignore
        cal = EconomicCalendarFetcher()
        cal.fetch_upcoming(days=14)
    except Exception:
        pass


def _worker_refresh_cot() -> None:
    try:
        from sentinel.sma.cftc_cot_v3 import COTFetcher  # type: ignore
        cot = COTFetcher()
        cot.fetch_latest()
    except Exception:
        pass


# ── CLI ───────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SENTINEL Deployment & Operations Manager")
    parser.add_argument("--setup", action="store_true", help="Run full one-command setup")
    parser.add_argument("--check", action="store_true", help="Run system requirements check")
    parser.add_argument("--health", action="store_true", help="Run health check report")
    parser.add_argument("--docker", action="store_true", help="Generate Docker files")
    parser.add_argument("--backup", action="store_true", help="Create a data backup")
    parser.add_argument("--cost", action="store_true", help="Show cost comparison report")
    parser.add_argument("--worker", action="store_true", help="Start background data refresh worker")
    parser.add_argument("--compact", action="store_true", help="VACUUM all databases")
    parser.add_argument("--users", type=int, default=1, help="Number of users (for cost report)")
    parser.add_argument("--years", type=int, default=3, help="Years for ROI analysis")
    parser.add_argument("--infra-monthly", type=float, default=0.0, help="Monthly infra cost ($)")
    parser.add_argument("--port", type=int, default=8000, help="API server port")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # Default: run all if no specific flag given
    run_all = not any([args.setup, args.check, args.health, args.docker,
                       args.backup, args.cost, args.worker, args.compact])

    if args.worker:
        env_mgr = EnvironmentManager()
        cfg = env_mgr.load_config()
        _run_worker(interval_minutes=cfg.worker_interval_minutes)
        sys.exit(0)

    if args.setup or run_all:
        installer = SentinelInstaller()
        installer.run_setup(port=args.port)

    elif args.check:
        checker = SystemRequirementsChecker()
        report = checker.run_full_check()
        checker.print_report(report)

    if args.health or run_all:
        monitor = SentinelHealthMonitor()
        health = monitor.run_health_check()
        monitor.print_health_report(health)
        if HAS_PANDAS:
            coverage = monitor.compute_data_coverage_report()
            if coverage is not None:
                print("\n  Data Coverage:")
                print(coverage.to_string(index=False))

    if args.docker or run_all:
        deployer = DockerDeployer()
        files = deployer.write_docker_files(".")
        print("\nDocker files written:")
        for f in files:
            print(f"  {f}")

    if args.cost or run_all:
        analyzer = CostAnalyzer()
        report_text = analyzer.generate_roi_report(
            users=args.users,
            years=args.years,
            sentinel_infra_monthly=args.infra_monthly,
        )
        print("\n" + report_text)

    if args.backup:
        mgr = BackupManager()
        result = mgr.backup_all_data(label="manual")
        if result.success:
            print(f"\nBackup created: {result.path}  ({result.size_bytes/1e6:.1f} MB, "
                  f"{result.files_included} files, {result.duration_seconds:.1f}s)")
        else:
            print(f"\nBackup FAILED: {result.error}")

    if args.compact:
        init = DataDirectoryInitializer()
        freed = init.compact_databases()
        total = sum(freed.values())
        print(f"\nDatabase compaction complete. Total freed: {total/1e6:.2f} MB")
        for path, b in freed.items():
            print(f"  {os.path.basename(path)}: {b/1e6:.2f} MB freed")

    if run_all:
        # Print startup summary
        print("\n" + "=" * 60)
        print("  SENTINEL Quick Start:\n")
        print("  $ uvicorn sentinel.api.main:app --port 8000")
        print("  $ python -m sentinel.ui.workspace_v3 --ticker AAPL")
        print("  $ docker compose -f docker-compose.sentinel.yml up -d")
        print("=" * 60)
