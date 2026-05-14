from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Database
    database_url: str = "postgresql+asyncpg://sentinel:sentinel@localhost:5432/sentinel"
    database_url_sync: str = "postgresql://sentinel:sentinel@localhost:5432/sentinel"
    postgres_password: str = "sentinel"

    # Redis
    redis_url: str = "redis://localhost:6379"

    # SEC EDGAR
    edgar_user_agent: str = "Sentinel sentinel@example.com"

    # Free API keys
    fred_api_key: str = ""
    finnhub_api_key: str = ""
    alphavantage_api_key: str = ""
    openfigi_api_key: str = ""
    polygon_api_key: str = ""
    etherscan_api_key: str = ""
    fmp_api_key: str = ""
    eodhd_api_key: str = ""
    glassnode_api_key: str = ""
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_user_agent: str = "sentinel:v0.1"

    # Broker
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    alpaca_base_url: str = "https://paper-api.alpaca.markets"
    ib_host: str = "127.0.0.1"
    ib_port: int = 7497
    ib_client_id: int = 1

    # AI
    anthropic_api_key: str = ""
    voyage_api_key: str = ""

    # Safety
    sentinel_live_trading: bool = False

    # Monitoring
    grafana_password: str = "sentinel"

    # App
    log_level: str = "INFO"
    environment: str = "development"

    @property
    def is_live_trading_enabled(self) -> bool:
        return self.sentinel_live_trading

    @property
    def has_anthropic(self) -> bool:
        return bool(self.anthropic_api_key)

    @property
    def has_finnhub(self) -> bool:
        return bool(self.finnhub_api_key)

    @property
    def has_polygon(self) -> bool:
        return bool(self.polygon_api_key)

    @property
    def has_alpaca(self) -> bool:
        return bool(self.alpaca_api_key and self.alpaca_secret_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
