FROM python:3.12-slim

WORKDIR /app

# System deps for QuantLib, psycopg2, numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev libssl-dev pkg-config curl git \
    && rm -rf /var/lib/apt/lists/*

# Poetry
ENV POETRY_HOME=/opt/poetry POETRY_VIRTUALENVS_CREATE=false
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /opt/poetry/bin/poetry /usr/local/bin/poetry

# Install dependencies
COPY pyproject.toml poetry.lock* ./
RUN poetry install --no-interaction --no-ansi --no-root

# Copy application
COPY . .
RUN pip install -e . --no-deps

EXPOSE 8000 8001 8501

ENV PYTHONPATH=/app PYTHONUNBUFFERED=1
