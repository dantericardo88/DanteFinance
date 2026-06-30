<#
  SENTINEL — simple one-command installer (Windows / PowerShell)

  Usage:   .\install.ps1

  What it does:
    1. Checks Docker is installed.
    2. Creates .env from .env.example (with a generated DB password) if missing.
    3. Builds and starts the whole stack with Docker Compose.
       Database schema + migrations run automatically (the 'migrate' service
       in docker-compose.yml applies Alembic migrations before the app starts).

  Manage afterwards:
    docker compose logs -f      # view logs
    docker compose down         # stop everything
    docker compose up -d        # start again
#>
$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot

Write-Host "SENTINEL installer" -ForegroundColor Cyan
Write-Host "------------------" -ForegroundColor Cyan

# 1. Docker present?
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
  Write-Host "Docker not found. Install Docker Desktop first: https://docs.docker.com/get-docker/" -ForegroundColor Red
  exit 1
}

# Detect 'docker compose' (plugin) vs legacy 'docker-compose'.
docker compose version *> $null
if ($?) { $compose = 'docker compose' }
elseif (Get-Command docker-compose -ErrorAction SilentlyContinue) { $compose = 'docker-compose' }
else {
  Write-Host "Docker Compose not found: https://docs.docker.com/compose/install/" -ForegroundColor Red
  exit 1
}

# 2. .env (create with a generated password if it doesn't exist).
if (-not (Test-Path .env)) {
  Write-Host "Creating .env with a generated database password..." -ForegroundColor Yellow
  Copy-Item .env.example .env
  $bytes = New-Object 'System.Byte[]' 24
  [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
  $pw = ([System.BitConverter]::ToString($bytes)).Replace('-', '').ToLower()
  $content = Get-Content .env -Raw
  $content = $content -replace '(?m)^POSTGRES_PASSWORD=.*', "POSTGRES_PASSWORD=$pw"
  $content = $content -replace 'sentinel_dev_password', $pw
  Set-Content .env -Value $content -Encoding utf8 -NoNewline
} else {
  Write-Host ".env already exists - leaving it untouched." -ForegroundColor Green
}

# 3. Build & start everything (migrations run automatically via the 'migrate' service).
Write-Host "Building and starting services (first run can take a few minutes)..." -ForegroundColor Yellow
& cmd /c "$compose up -d --build"
if (-not $?) { Write-Host "docker compose failed. See output above." -ForegroundColor Red; exit 1 }

Write-Host ""
Write-Host "SENTINEL is up:" -ForegroundColor Green
Write-Host "  Terminal (UI) : http://localhost:8501"
Write-Host "  API docs      : http://localhost:8000/docs"
Write-Host "  MCP server    : http://localhost:8001"
Write-Host ""
Write-Host "Logs: $compose logs -f    Stop: $compose down"
