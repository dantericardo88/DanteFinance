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

    # AI — tiered routing strategy
    # Tier 1 (free/cheap): Ollama local models for routine analysis
    # Tier 2 (cheap):      xAI Grok for mid-complexity reasoning
    # Tier 3 (paid):       OpenAI GPT-5 for complex multi-step workflows
    # Tier 4 (premium):    Anthropic Claude Opus 4.7 for complex trading strategies only
    anthropic_api_key: str = ""
    xai_api_key: str = ""                          # xAI Grok — primary cheap AI
    openai_api_key: str = ""                        # GPT-5 — complex workflows
    voyage_api_key: str = ""

    # Ollama — local, free, runs on-machine
    ollama_base_url: str = "http://localhost:11434"
    ollama_default_model: str = "gemma3:27b"        # pull: ollama pull gemma3:27b
    ollama_fast_model: str = "gemma3:4b"            # pull: ollama pull gemma3:4b

    # xAI model selection
    xai_model: str = "grok-3-mini"                  # cheapest xAI model
    xai_complex_model: str = "grok-3"               # for harder xAI tasks
    xai_base_url: str = "https://api.x.ai/v1"

    # OpenAI model selection
    openai_model: str = "gpt-5"                     # complex multi-step workflows
    openai_base_url: str = "https://api.openai.com/v1"

    # Anthropic model selection — only for complex trading strategies
    anthropic_model: str = "claude-opus-4-7"        # premium, use sparingly
    anthropic_fast_model: str = "claude-haiku-4-5-20251001"  # cheap Claude fallback

    # AI routing thresholds
    # "simple"   -> Ollama Gemma (free)
    # "standard" -> xAI Grok-3-mini (cheap)
    # "complex"  -> xAI Grok-3 or GPT-5
    # "trading"  -> Claude Opus 4.7 (premium, trading strategies only)
    ai_default_tier: str = "standard"              # default when tier not specified

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
    def has_xai(self) -> bool:
        return bool(self.xai_api_key)

    @property
    def has_openai(self) -> bool:
        return bool(self.openai_api_key)

    @property
    def has_ollama(self) -> bool:
        import httpx
        try:
            r = httpx.get(f"{self.ollama_base_url}/api/tags", timeout=2.0)
            return r.status_code == 200
        except Exception:
            return False

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
