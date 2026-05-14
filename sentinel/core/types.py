"""All shared Pydantic v2 schemas. Imported by every module — no circular deps."""
from __future__ import annotations
from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import Optional, Any
from pydantic import BaseModel, ConfigDict


# ── Enums ────────────────────────────────────────────────────────────────────

class AssetClass(str, Enum):
    EQUITY = "equity"
    BOND = "bond"
    FIXED_INCOME = "fixed_income"   # alias kept alongside BOND for OpenFIGI compat
    ETF = "etf"
    CRYPTO = "crypto"
    FX = "fx"
    COMMODITY = "commodity"
    OPTION = "option"
    FUTURE = "future"
    FUND = "fund"


class CorporateActionType(str, Enum):
    SPLIT = "split"
    REVERSE_SPLIT = "reverse_split"
    DIVIDEND = "dividend"
    SPIN_OFF = "spin_off"
    RIGHTS_ISSUE = "rights_issue"


class DelistReason(str, Enum):
    BANKRUPTCY = "bankruptcy"
    ACQUISITION = "acquisition"
    MERGER = "merger"
    VOLUNTARY_DELISTING = "voluntary_delisting"
    REGULATORY = "regulatory"
    UNKNOWN = "unknown"


class StrategyStatus(str, Enum):
    BACKTEST = "backtest"
    PAPER = "paper"
    CAPPED_LIVE = "capped_live"
    FULL_AUTONOMOUS = "full_autonomous"
    PAUSED = "paused"
    RETIRED = "retired"


class MacroRegime(str, Enum):
    GROWTH_INFLATION = "Growth/Inflation"
    GROWTH_DEFLATION = "Growth/Deflation"
    CONTRACTION_INFLATION = "Contraction/Inflation"
    CONTRACTION_DEFLATION = "Contraction/Deflation"
    UNKNOWN = "Unknown"


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MKT"
    LIMIT = "LMT"
    STOP = "STP"
    STOP_LIMIT = "STPLMT"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class DataSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class InsiderRole(str, Enum):
    OFFICER = "Officer"
    DIRECTOR = "Director"
    TEN_PCT = "10% Owner"
    OTHER = "Other"


class InsiderTxCode(str, Enum):
    PURCHASE = "P"
    SALE = "S"
    AWARD = "A"
    EXERCISE = "M"
    GIFT = "G"
    TAX_WITHHOLDING = "F"
    OTHER = "X"


# ── Market Data ───────────────────────────────────────────────────────────────

class OHLCVBar(BaseModel):
    model_config = ConfigDict(frozen=True)

    time: datetime
    figi: str
    ticker: Optional[str] = None
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal                        # RAW — never overwritten after ingestion
    volume: int
    vwap: Optional[Decimal] = None
    adj_factor: Decimal = Decimal("1.0")  # set by CA engine on read; 1.0 = unadjusted
    source: str

    @property
    def adjusted_close(self) -> Decimal:
        """Backward-adjusted close: raw close × cumulative split/dividend factor."""
        return self.close * self.adj_factor


class Quote(BaseModel):
    time: datetime
    figi: str
    bid: Optional[Decimal] = None
    ask: Optional[Decimal] = None
    last: Optional[Decimal] = None
    volume: Optional[int] = None
    source: str


class OptionGreeks(BaseModel):
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None
    iv: Optional[float] = None


class OptionContract(BaseModel):
    time: datetime
    underlying_figi: str
    expiry: date
    strike: Decimal
    option_type: str  # "C" or "P"
    bid: Optional[Decimal] = None
    ask: Optional[Decimal] = None
    last: Optional[Decimal] = None
    volume: Optional[int] = None
    open_interest: Optional[int] = None
    greeks: Optional[OptionGreeks] = None


# ── Instrument ────────────────────────────────────────────────────────────────

class Instrument(BaseModel):
    figi: str
    share_class_figi: Optional[str] = None
    composite_figi: Optional[str] = None
    ticker: Optional[str] = None
    name: Optional[str] = None
    isin: Optional[str] = None
    cusip: Optional[str] = None
    sedol: Optional[str] = None
    ric: Optional[str] = None
    asset_class: AssetClass = AssetClass.EQUITY
    market: Optional[str] = None
    exchange_code: Optional[str] = None
    currency: str = "USD"
    country: Optional[str] = None
    gics_sector: Optional[int] = None
    gics_group: Optional[int] = None
    gics_industry: Optional[int] = None
    gics_sub: Optional[int] = None
    sic_code: Optional[int] = None
    ipo_date: Optional[date] = None
    delist_date: Optional[date] = None
    is_active: bool = True


# ── Financials ────────────────────────────────────────────────────────────────

class FinancialFact(BaseModel):
    cik: str
    figi: Optional[str] = None
    concept: str
    taxonomy: str = "us-gaap"
    label: Optional[str] = None
    value: Optional[Decimal] = None
    unit: str = "USD"
    period_start: Optional[date] = None
    period_end: date
    instant: Optional[date] = None
    form_type: str = ""
    filed_date: Optional[date] = None  # immutable at ingest; enforced at API layer for PIT
    accession: Optional[str] = None
    frame: Optional[str] = None


# ── Ownership ─────────────────────────────────────────────────────────────────

class InsiderTransaction(BaseModel):
    accession: str
    figi: Optional[str] = None
    cik_issuer: str
    cik_owner: str
    owner_name: str
    owner_role: InsiderRole = InsiderRole.OTHER
    is_director: bool = False
    is_officer: bool = False
    is_ten_pct: bool = False
    transaction_date: Optional[date] = None
    transaction_code: InsiderTxCode = InsiderTxCode.OTHER
    shares: Optional[Decimal] = None
    price_per_share: Optional[Decimal] = None
    total_value: Optional[Decimal] = None
    shares_owned_after: Optional[Decimal] = None
    is_10b5_1: Optional[bool] = None
    filed_date: date


class InstitutionalHolding(BaseModel):
    filing_id: str
    manager_cik: str
    manager_name: Optional[str] = None
    period_of_report: date
    figi: Optional[str] = None
    cusip: Optional[str] = None
    security_name: Optional[str] = None
    value: Optional[int] = None  # $ thousands
    shares: Optional[int] = None
    option_type: Optional[str] = None  # Put/Call/None
    filed_date: Optional[date] = None


class CongressionalTrade(BaseModel):
    politician_name: str
    chamber: str  # "Senate" / "House"
    party: Optional[str] = None
    state: Optional[str] = None
    ticker: Optional[str] = None
    figi: Optional[str] = None
    asset_name: Optional[str] = None
    tx_date: Optional[date] = None
    filed_date: date
    tx_type: Optional[str] = None  # "buy" / "sell" / "other"
    amount_low: int = 0
    amount_high: int = 0
    filing_lag_days: int = 0
    late_filing: bool = False
    source: str = ""
    disclosure_url: Optional[str] = None


# ── Macro ─────────────────────────────────────────────────────────────────────

class MacroDataPoint(BaseModel):
    time: datetime
    series_id: str
    value: Optional[Decimal] = None
    vintage: Optional[datetime] = None


class RegimeResult(BaseModel):
    date: date
    regime: MacroRegime
    regime_id: int
    confidence: float
    probabilities: dict[str, float]


class COTReport(BaseModel):
    report_date: date
    market: str
    exchange: str
    commodity: str
    # Non-commercial (large speculators)
    nc_long: int = 0
    nc_short: int = 0
    nc_net: int = 0
    # Commercial (hedgers)
    comm_long: int = 0
    comm_short: int = 0
    comm_net: int = 0
    # Non-reportable (small specs)
    nr_long: int = 0
    nr_short: int = 0
    nr_net: int = 0
    # COT Index (0-100, 52-week percentile of nc_net)
    cot_index: Optional[float] = None
    open_interest: int = 0


# ── Backtesting ───────────────────────────────────────────────────────────────

class BacktestMetrics(BaseModel):
    strategy_id: str = ""
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    total_return: Decimal = Decimal("0")
    cagr: Decimal = Decimal("0")
    volatility: Decimal = Decimal("0")
    sharpe_ratio: Decimal = Decimal("0")
    sortino_ratio: Decimal = Decimal("0")
    calmar_ratio: Decimal = Decimal("0")
    max_drawdown: Decimal = Decimal("0")
    win_rate: Decimal = Decimal("0")
    profit_factor: Decimal = Decimal("0")
    avg_win: Decimal = Decimal("0")
    avg_loss: Decimal = Decimal("0")
    var_95: Decimal = Decimal("0")
    cvar_95: Decimal = Decimal("0")
    beta: Decimal = Decimal("0")
    alpha: Decimal = Decimal("0")
    information_ratio: Decimal = Decimal("0")
    skewness: Decimal = Decimal("0")
    kurtosis: Decimal = Decimal("0")
    deflated_sharpe_ratio: Decimal = Decimal("0")
    turnover: Decimal = Decimal("0")
    exposure: Decimal = Decimal("0")
    n_trades: int = 0
    n_trials: int = 1


class StrategySpec(BaseModel):
    name: str
    description: str
    universe: dict[str, Any]
    entry_signals: list[dict[str, Any]]
    exit_signals: list[dict[str, Any]]
    position_sizing: dict[str, Any]
    execution: dict[str, Any]


class PromotionGateResult(BaseModel):
    gate_name: str
    passed: bool
    value: Optional[float] = None
    threshold: Optional[float] = None
    message: str = ""


class PromotionResult(BaseModel):
    strategy_id: str
    from_status: StrategyStatus
    to_status: Optional[StrategyStatus] = None
    success: bool
    failed_gates: list[PromotionGateResult] = []
    requires_human_approval: bool = True


# ── Orders ────────────────────────────────────────────────────────────────────

class Order(BaseModel):
    id: Optional[str] = None
    strategy_id: Optional[str] = None
    broker: str
    broker_order_id: Optional[str] = None
    figi: Optional[str] = None
    ticker: Optional[str] = None
    side: OrderSide
    order_type: OrderType
    qty: Decimal
    limit_price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: Decimal = Decimal("0")
    avg_fill_price: Optional[Decimal] = None
    commission: Decimal = Decimal("0")
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    error_message: Optional[str] = None


# ── Health ────────────────────────────────────────────────────────────────────

class DataHealthEvent(BaseModel):
    time: datetime
    adapter: str
    event_type: str  # staleness/gap/schema_drift/throttle/ok
    severity: DataSeverity
    details: dict[str, Any] = {}


# ── News / Intelligence ───────────────────────────────────────────────────────

class NewsArticle(BaseModel):
    source: str
    external_id: Optional[str] = None
    headline: str
    summary: Optional[str] = None
    full_text: Optional[str] = None
    url: Optional[str] = None
    published_at: datetime
    entities: list[dict[str, Any]] = []
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[str] = None


class SentimentResult(BaseModel):
    text: str
    label: str  # positive/negative/neutral/uncertain
    score: float
    confidence: float


class ScreenResult(BaseModel):
    figi: str
    ticker: Optional[str] = None
    name: Optional[str] = None
    score: float = 0.0
    matched_criteria: list[str] = []
    fields: dict[str, Any] = {}


# ── Corporate Actions ─────────────────────────────────────────────────────────

class CorporateAction(BaseModel):
    id: Optional[int] = None
    figi: str
    ticker: str
    action_type: CorporateActionType
    ex_date: date
    ratio_new: Decimal = Decimal("1")
    ratio_old: Decimal = Decimal("1")
    factor: Decimal                       # backward-adjustment multiplier
    source: str
    created_at: Optional[datetime] = None


# ── Survivorship ──────────────────────────────────────────────────────────────

class SurvivorshipRecord(BaseModel):
    id: Optional[int] = None
    cik: str
    figi: Optional[str] = None
    ticker: str
    company_name: str
    delist_date: date
    delist_reason: DelistReason = DelistReason.UNKNOWN
    exchange: Optional[str] = None
    notes: Optional[str] = None
    created_at: Optional[datetime] = None


# ── Data Provenance ───────────────────────────────────────────────────────────

class DataProvenance(BaseModel):
    id: Optional[int] = None
    batch_id: str
    source: str
    ticker: str
    figi: Optional[str] = None
    interval: str
    start_time: datetime
    end_time: datetime
    bar_count: int
    sha256: str
    prev_hash: Optional[str] = None
    validated: bool = False
    validation_delta_pct: Optional[Decimal] = None
    ingested_at: datetime
