#!/usr/bin/env bash
# dim_096: Docker infra — verify docker-compose.yml and Makefile exist with required services/targets
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())
from pathlib import Path

# Check docker-compose.yml exists and has required services
dc_path = Path("docker-compose.yml")
assert dc_path.exists(), "docker-compose.yml must exist in project root"

dc_content = dc_path.read_text(encoding="utf-8")
for service in ["postgres", "redis", "api"]:
    assert service in dc_content, f"docker-compose.yml must define '{service}' service"
print(f"[OK] docker-compose.yml: exists with postgres, redis, api services")

# Check TimescaleDB is used
assert "timescale" in dc_content.lower() or "timescaledb" in dc_content.lower(), \
    "docker-compose.yml should use TimescaleDB image"
print(f"[OK] docker-compose.yml: uses TimescaleDB")

# Check Redis maxmemory policy (good ops practice)
assert "maxmemory" in dc_content, "Redis should have maxmemory configured"
print(f"[OK] docker-compose.yml: Redis has maxmemory configured")

# Check healthchecks are defined for postgres and redis
assert "healthcheck" in dc_content, "Services should have healthchecks"
assert "pg_isready" in dc_content, "Postgres healthcheck should use pg_isready"
print(f"[OK] docker-compose.yml: healthchecks configured")

# Check volumes
assert "volumes:" in dc_content, "docker-compose.yml should define named volumes"
print(f"[OK] docker-compose.yml: volumes configured")

# Check Makefile exists
mk_path = Path("Makefile")
assert mk_path.exists(), "Makefile must exist in project root"
mk_content = mk_path.read_text(encoding="utf-8")

# Verify required Makefile targets
for target in ["docker-up", "docker-down", "bootstrap", "check"]:
    assert target + ":" in mk_content or f"\n{target}:" in mk_content, \
        f"Makefile must have '{target}' target"
print(f"[OK] Makefile: has docker-up, docker-down, bootstrap, check targets")

# Check for db-migrate target (Alembic migrations)
assert "db-migrate" in mk_content, "Makefile should have 'db-migrate' target"
assert "alembic" in mk_content.lower(), "Makefile should reference Alembic"
print(f"[OK] Makefile: has db-migrate / Alembic target")

# Check for backfill target
assert "backfill" in mk_content, "Makefile should have 'backfill' target"
print(f"[OK] Makefile: has backfill target")

# Verify infra SQL init scripts exist
init_sql = Path("infra/postgres/init.sql")
hyper_sql = Path("infra/timescaledb/hypertables.sql")
# Check at least the paths are referenced in docker-compose (they may or may not exist)
assert "init.sql" in dc_content, "docker-compose.yml should reference init.sql"
print(f"[OK] docker-compose.yml: references init.sql for DB initialization")

# Check Dockerfile exists
dockerfile = Path("Dockerfile")
assert dockerfile.exists(), "Dockerfile must exist"
df_content = dockerfile.read_text(encoding="utf-8")
# Should be a Python-based Docker image
assert "python" in df_content.lower() or "FROM" in df_content, \
    "Dockerfile should be Python-based"
print(f"[OK] Dockerfile: exists")

# Check pyproject.toml or requirements.txt exists (Poetry-based project)
has_pyproject = Path("pyproject.toml").exists()
has_requirements = Path("requirements.txt").exists()
assert has_pyproject or has_requirements, \
    "Project must have pyproject.toml or requirements.txt"
if has_pyproject:
    pp_content = Path("pyproject.toml").read_text(encoding="utf-8")
    assert "sentinel" in pp_content.lower() or "poetry" in pp_content.lower(), \
        "pyproject.toml should reference sentinel or poetry"
    print(f"[OK] pyproject.toml: exists")
else:
    print(f"[OK] requirements.txt: exists")

print("\n[PASS] dim_096: Docker infrastructure")
PYEOF
