"""SDS Gen 0 evaluation tests.

These are correctness benchmarks, not unit tests. They verify that the data
quality guarantees defined in SENTINEL_SDS_PRD_v1.0.md are actually enforced:
  - Corporate action factors are mathematically correct
  - Survivorship registry contains the right securities
  - Validator hard-rejects impossible OHLCV values
  - Cross-source validation catches 1%+ deltas
  - Gap detector identifies silent API throttling
  - Provenance receipts are deterministic and chainable

Run: make test-evals
"""
from __future__ import annotations
from datetime import date, datetime
from decimal import Decimal
from typing import Optional

import pytest


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _bar(
    ticker: str = "AAPL",
    dt: datetime = datetime(2020, 1, 2),
    open: float = 100.0,
    high: float = 102.0,
    low: float = 99.0,
    close: float = 101.0,
    volume: int = 1_000_000,
    source: str = "yfinance",
    adj_factor: float = 1.0,
):
    from sentinel.core.types import OHLCVBar
    return OHLCVBar(
        time=dt,
        figi=ticker,
        ticker=ticker,
        open=Decimal(str(open)),
        high=Decimal(str(high)),
        low=Decimal(str(low)),
        close=Decimal(str(close)),
        volume=volume,
        adj_factor=Decimal(str(adj_factor)),
        source=source,
    )


def _action(
    ticker: str,
    ex_date: date,
    ratio: float,
    action_type: str = "split",
):
    from sentinel.core.types import CorporateAction, CorporateActionType
    return CorporateAction(
        figi=ticker,
        ticker=ticker,
        action_type=CorporateActionType(action_type),
        ex_date=ex_date,
        ratio_new=Decimal(str(ratio)),
        ratio_old=Decimal("1"),
        factor=Decimal("1") / Decimal(str(ratio)),
        source="test",
    )


# ─── Corporate Actions — Factor Math ──────────────────────────────────────────

class TestCorporateActionFactors:
    """Validate backward-adjustment factor computation for known historical splits."""

    def test_aapl_4for1_split_2020(self):
        """AAPL 4-for-1 split 2020-08-31: bar from 2020-01-01 must get factor=0.25."""
        from sentinel.sds.corporate_actions import compute_cumulative_factor
        action = _action("AAPL", date(2020, 8, 31), ratio=4.0)
        factor = compute_cumulative_factor(date(2020, 1, 1), [action])
        assert factor == Decimal("0.25"), f"Expected 0.25, got {factor}"

    def test_aapl_4for1_bar_after_split_unaffected(self):
        """Bar dated AFTER the split ex_date should have factor=1.0 (no adjustment needed)."""
        from sentinel.sds.corporate_actions import compute_cumulative_factor
        action = _action("AAPL", date(2020, 8, 31), ratio=4.0)
        factor = compute_cumulative_factor(date(2020, 9, 1), [action])
        assert factor == Decimal("1.0"), f"Post-split bar should be unadjusted, got {factor}"

    def test_tsla_5for1_split_2020(self):
        """TSLA 5-for-1 split 2020-08-31: factor for pre-split bar = 0.20."""
        from sentinel.sds.corporate_actions import compute_cumulative_factor
        action = _action("TSLA", date(2020, 8, 31), ratio=5.0)
        factor = compute_cumulative_factor(date(2019, 12, 31), [action])
        assert factor == Decimal("0.2"), f"Expected 0.2, got {factor}"

    def test_amzn_20for1_split_2022(self):
        """AMZN 20-for-1 split 2022-06-06: factor = 0.05."""
        from sentinel.sds.corporate_actions import compute_cumulative_factor
        action = _action("AMZN", date(2022, 6, 6), ratio=20.0)
        factor = compute_cumulative_factor(date(2022, 1, 1), [action])
        assert factor == Decimal("0.05"), f"Expected 0.05, got {factor}"

    def test_googl_20for1_split_2022(self):
        """GOOGL 20-for-1 split 2022-07-18: factor = 0.05 for pre-split bars."""
        from sentinel.sds.corporate_actions import compute_cumulative_factor
        action = _action("GOOGL", date(2022, 7, 18), ratio=20.0)
        factor = compute_cumulative_factor(date(2021, 1, 1), [action])
        assert factor == Decimal("0.05"), f"Expected 0.05, got {factor}"

    def test_nvda_10for1_split_2024(self):
        """NVDA 10-for-1 split 2024-06-10: factor = 0.1 for pre-split bars."""
        from sentinel.sds.corporate_actions import compute_cumulative_factor
        action = _action("NVDA", date(2024, 6, 10), ratio=10.0)
        factor = compute_cumulative_factor(date(2023, 1, 1), [action])
        assert factor == Decimal("0.1"), f"Expected 0.1, got {factor}"

    def test_cumulative_two_splits(self):
        """Two splits compound correctly: 4:1 then 5:1 → factor = 0.05 for oldest bars."""
        from sentinel.sds.corporate_actions import compute_cumulative_factor
        actions = [
            _action("AAPL", date(2014, 6, 9), ratio=7.0),   # 7-for-1 (historical)
            _action("AAPL", date(2020, 8, 31), ratio=4.0),  # 4-for-1
        ]
        # Bar from 2010: both splits apply → 1/7 × 1/4 = 1/28
        factor = compute_cumulative_factor(date(2010, 1, 1), actions)
        expected = Decimal("1") / Decimal("7") * (Decimal("1") / Decimal("4"))
        assert abs(factor - expected) < Decimal("1e-10"), f"Expected {expected}, got {factor}"

    def test_no_split_no_adjustment(self):
        """Ticker with no corporate actions: factor = 1.0 for any bar date."""
        from sentinel.sds.corporate_actions import compute_cumulative_factor
        factor = compute_cumulative_factor(date(2020, 1, 1), [])
        assert factor == Decimal("1.0")

    def test_apply_adjustments_preserves_raw_close(self):
        """apply_adjustments must not modify raw close — only adj_factor changes."""
        from sentinel.sds.corporate_actions import apply_adjustments
        bar = _bar(close=400.0)
        action = _action("AAPL", date(2020, 8, 31), ratio=4.0)
        [adjusted] = apply_adjustments([bar], [action])
        assert adjusted.close == Decimal("400.0"), "Raw close must be immutable"
        assert adjusted.adj_factor == Decimal("0.25")

    def test_adjusted_close_property(self):
        """adjusted_close property returns close × adj_factor correctly."""
        bar = _bar(close=400.0, adj_factor=0.25)
        assert bar.adjusted_close == Decimal("100.0"), (
            f"adjusted_close expected 100.0, got {bar.adjusted_close}"
        )

    def test_reverse_split_factor_greater_than_one(self):
        """Reverse split: 1-for-10 → ratio=0.1, factor=10. Pre-split bars adjust UP."""
        from sentinel.sds.corporate_actions import compute_cumulative_factor
        from sentinel.core.types import CorporateAction, CorporateActionType
        action = CorporateAction(
            figi="GME", ticker="GME",
            action_type=CorporateActionType.REVERSE_SPLIT,
            ex_date=date(2023, 1, 1),
            ratio_new=Decimal("0.1"),
            ratio_old=Decimal("1"),
            factor=Decimal("10"),   # 1/0.1 = 10
            source="test",
        )
        factor = compute_cumulative_factor(date(2022, 1, 1), [action])
        assert factor == Decimal("10"), f"Expected 10, got {factor}"


# ─── Survivorship Registry ────────────────────────────────────────────────────

class TestSurvivorshipRegistry:
    """Verify the survivorship registry correctly tracks famous failures."""

    def test_lehman_brothers_in_registry(self):
        from sentinel.sds.survivorship import is_delisted, lookup_by_ticker
        assert is_delisted("0000806157"), "Lehman Brothers must be in registry"
        matches = lookup_by_ticker("LEHMQ")
        assert len(matches) == 1
        assert matches[0].delist_date == date(2008, 9, 15)

    def test_enron_in_registry(self):
        from sentinel.sds.survivorship import is_delisted
        assert is_delisted("0000101830"), "Enron must be in registry"

    def test_bear_stearns_in_registry(self):
        from sentinel.sds.survivorship import is_delisted, lookup_by_ticker
        assert is_delisted("0000777819"), "Bear Stearns must be in registry"
        matches = lookup_by_ticker("BSC")
        from sentinel.core.types import DelistReason
        assert matches[0].delist_reason == DelistReason.ACQUISITION

    def test_active_ticker_not_in_registry(self):
        from sentinel.sds.survivorship import is_delisted, lookup_by_ticker
        assert not is_delisted("9999999999"), "Random CIK must not be delisted"
        assert lookup_by_ticker("AAPL") == []

    def test_pit_universe_includes_lehman_in_2007(self):
        """A 2007 universe must include Lehman (delisted 2008) to avoid survivorship bias."""
        from sentinel.sds.survivorship import build_point_in_time_universe
        active = ["AAPL", "MSFT", "JPM"]
        universe = build_point_in_time_universe(active, as_of_date=date(2007, 1, 1))
        assert "LEHMQ" in universe, "Lehman must be in 2007 PIT universe"

    def test_pit_universe_excludes_lehman_after_delist(self):
        """A universe as-of 2010 must NOT add Lehman (already delisted 2008)."""
        from sentinel.sds.survivorship import build_point_in_time_universe
        active = ["AAPL", "MSFT", "JPM"]
        universe = build_point_in_time_universe(active, as_of_date=date(2010, 1, 1))
        assert "LEHMQ" not in universe, "Lehman must not appear in post-delist 2010 universe"

    def test_range_query_2008_crisis(self):
        """get_delisted_in_range should capture Lehman, Bear Stearns, WaMu in 2008."""
        from sentinel.sds.survivorship import get_delisted_in_range
        failures = get_delisted_in_range(date(2008, 1, 1), date(2008, 12, 31))
        tickers = {r.ticker for r in failures}
        assert "LEHMQ" in tickers
        assert "BSC" in tickers
        assert "WAMUQ" in tickers

    def test_register_new_record(self):
        from sentinel.sds.survivorship import register, is_delisted, get_all_delisted
        from sentinel.core.types import SurvivorshipRecord, DelistReason
        record = SurvivorshipRecord(
            cik="9999000001", ticker="TESTQ", company_name="Test Corp",
            delist_date=date(2020, 6, 15), delist_reason=DelistReason.BANKRUPTCY,
        )
        count_before = len(get_all_delisted())
        register(record)
        assert is_delisted("9999000001")
        assert len(get_all_delisted()) == count_before + 1


# ─── OHLCV Validator ──────────────────────────────────────────────────────────

class TestOHLCVValidator:
    """Hard-reject impossible values; flag cross-source deltas above 1%."""

    def test_valid_bar_passes(self):
        from sentinel.sds.validator import validate_single_source
        bars = [_bar()]
        passed, report = validate_single_source(bars, "AAPL")
        assert report.is_clean
        assert len(passed) == 1

    def test_close_above_high_rejected(self):
        from sentinel.sds.validator import validate_single_source
        bars = [_bar(close=103.0, high=102.0)]  # close > high — impossible
        passed, report = validate_single_source(bars, "AAPL")
        assert len(passed) == 0
        assert report.rejected == 1
        assert report.errors[0].rule == "close_above_high"

    def test_close_below_low_rejected(self):
        from sentinel.sds.validator import validate_single_source
        bars = [_bar(close=98.0, low=99.0)]  # close < low — impossible
        passed, report = validate_single_source(bars, "AAPL")
        assert len(passed) == 0
        assert report.errors[0].rule == "close_below_low"

    def test_high_below_low_rejected(self):
        from sentinel.sds.validator import validate_single_source
        bars = [_bar(high=98.0, low=100.0)]
        passed, report = validate_single_source(bars, "AAPL")
        assert report.rejected == 1
        assert report.errors[0].rule == "high_below_low"

    def test_negative_price_rejected(self):
        from sentinel.sds.validator import validate_single_source
        bars = [_bar(open=-1.0, close=-0.5)]
        passed, report = validate_single_source(bars, "AAPL")
        assert report.rejected == 1

    def test_multiple_bars_partial_pass(self):
        from sentinel.sds.validator import validate_single_source
        bars = [
            _bar(dt=datetime(2024, 1, 2)),                          # valid
            _bar(dt=datetime(2024, 1, 3), close=103.0, high=102.0), # close > high
            _bar(dt=datetime(2024, 1, 4)),                          # valid
        ]
        passed, report = validate_single_source(bars, "AAPL")
        assert len(passed) == 2
        assert report.rejected == 1
        assert report.pass_rate == pytest.approx(2 / 3)

    def test_cross_validate_within_threshold_passes(self):
        """0.5% delta — within 1% tolerance — should pass."""
        from sentinel.sds.validator import cross_validate
        dt = datetime(2024, 1, 2)
        primary = [_bar(dt=dt, close=100.0, source="yfinance")]
        secondary = [_bar(dt=dt, close=100.5, source="polygon")]  # 0.5% delta
        passed, report = cross_validate(primary, secondary, "AAPL")
        assert len(passed) == 1
        assert report.is_clean

    def test_cross_validate_above_threshold_flagged(self):
        """1.5% delta — exceeds 1% tolerance — should be rejected."""
        from sentinel.sds.validator import cross_validate
        dt = datetime(2024, 1, 2)
        primary = [_bar(dt=dt, close=100.0, source="yfinance")]
        secondary = [_bar(dt=dt, close=101.6, source="polygon")]  # 1.6% delta
        passed, report = cross_validate(primary, secondary, "AAPL")
        assert len(passed) == 0
        assert report.rejected == 1
        assert report.errors[0].rule == "cross_source_delta"

    def test_cross_validate_no_secondary_bar_passes(self):
        """Bar with no secondary counterpart should pass through unconditionally."""
        from sentinel.sds.validator import cross_validate
        primary = [_bar(dt=datetime(2024, 1, 2), source="yfinance")]
        secondary = [_bar(dt=datetime(2024, 1, 3), source="polygon")]  # different date
        passed, report = cross_validate(primary, secondary, "AAPL")
        assert len(passed) == 1

    def test_custom_threshold(self):
        """Custom 0.1% threshold should flag 0.5% delta."""
        from sentinel.sds.validator import cross_validate
        dt = datetime(2024, 1, 2)
        primary = [_bar(dt=dt, close=100.0)]
        secondary = [_bar(dt=dt, close=100.5)]  # 0.5% delta
        passed, report = cross_validate(primary, secondary, "AAPL", threshold=Decimal("0.001"))
        assert report.rejected == 1


# ─── Gap Detector ─────────────────────────────────────────────────────────────

class TestGapDetector:
    """Verify silent throttle detection and gap counting."""

    def test_zero_bars_is_silent_throttle(self):
        """Empty result from an adapter must be classified as silent throttle."""
        from sentinel.sds.gap_detector import detect_gaps
        report = detect_gaps(
            bars=[], ticker="SPY",
            start=date(2024, 1, 2), end=date(2024, 1, 5),
            exchange="NYSE", interval="1d",
        )
        assert report.is_silent_throttle, "0 bars must flag as silent throttle"

    def test_intraday_zero_bars_is_silent_throttle(self):
        from sentinel.sds.gap_detector import detect_gaps
        report = detect_gaps(
            bars=[], ticker="AAPL",
            start=date(2024, 1, 2), end=date(2024, 1, 2),
            exchange="NYSE", interval="1m",
        )
        assert report.is_silent_throttle

    def test_complete_week_no_gaps(self):
        """5 bars for Mon-Fri with no holidays should show completeness=1.0."""
        from sentinel.sds.gap_detector import detect_gaps
        # 2024-01-02 to 2024-01-05: Tue-Fri (Jan 1 = NY holiday, Jan 2 = first trading day)
        bars = [
            _bar(dt=datetime(2024, 1, 2)),
            _bar(dt=datetime(2024, 1, 3)),
            _bar(dt=datetime(2024, 1, 4)),
            _bar(dt=datetime(2024, 1, 5)),
        ]
        report = detect_gaps(
            bars=bars, ticker="SPY",
            start=date(2024, 1, 2), end=date(2024, 1, 5),
            exchange="NYSE", interval="1d",
        )
        assert not report.is_silent_throttle
        assert report.completeness > 0.9, f"Completeness {report.completeness} too low"

    def test_gap_detected_for_missing_day(self):
        """Providing 4 bars for a 5-day week must detect 1 missing day."""
        from sentinel.sds.gap_detector import detect_gaps
        # Skip Wednesday 2024-01-03
        bars = [
            _bar(dt=datetime(2024, 1, 2)),
            _bar(dt=datetime(2024, 1, 4)),
            _bar(dt=datetime(2024, 1, 5)),
        ]
        report = detect_gaps(
            bars=bars, ticker="SPY",
            start=date(2024, 1, 2), end=date(2024, 1, 5),
            exchange="NYSE", interval="1d",
        )
        assert report.gap_count >= 1

    def test_intraday_bars_skip_calendar_check(self):
        """Intraday intervals should not trigger calendar-based gap detection."""
        from sentinel.sds.gap_detector import detect_gaps
        bars = [_bar(dt=datetime(2024, 1, 2, 10, 0))]
        report = detect_gaps(
            bars=bars, ticker="AAPL",
            start=date(2024, 1, 2), end=date(2024, 1, 2),
            exchange="NYSE", interval="5m",
        )
        # Should not be flagged as silent throttle (has bars)
        assert not report.is_silent_throttle


# ─── Data Provenance ──────────────────────────────────────────────────────────

class TestDataProvenance:
    """SHA-256 receipts must be deterministic and chainable."""

    def test_sha256_is_deterministic(self):
        """Same bar list → same hash, every time."""
        from sentinel.sds.provenance import compute_sha256
        bars = [_bar(dt=datetime(2024, 1, 2)), _bar(dt=datetime(2024, 1, 3))]
        h1 = compute_sha256(bars)
        h2 = compute_sha256(bars)
        assert h1 == h2

    def test_different_bars_different_hash(self):
        from sentinel.sds.provenance import compute_sha256
        bars_a = [_bar(close=100.0)]
        bars_b = [_bar(close=101.0)]
        assert compute_sha256(bars_a) != compute_sha256(bars_b)

    def test_receipt_has_correct_bar_count(self):
        from sentinel.sds.provenance import create_receipt
        bars = [_bar(dt=datetime(2024, 1, d + 2)) for d in range(5)]
        receipt = create_receipt(bars, source="yfinance", ticker="AAPL", interval="1d")
        assert receipt.bar_count == 5

    def test_receipt_time_range(self):
        from sentinel.sds.provenance import create_receipt
        bars = [
            _bar(dt=datetime(2024, 1, 2)),
            _bar(dt=datetime(2024, 1, 5)),
        ]
        receipt = create_receipt(bars, source="yfinance", ticker="MSFT", interval="1d")
        assert receipt.start_time == datetime(2024, 1, 2)
        assert receipt.end_time == datetime(2024, 1, 5)

    def test_chain_links_prev_hash(self):
        """Second receipt for same (ticker, source, interval) must link to first."""
        from sentinel.sds.provenance import create_receipt, compute_sha256, _chain
        # Use unique ticker to avoid state bleed from other tests
        ticker = "CHAIN_TEST_XYZ"
        chain_key = f"{ticker}:yfinance:1d"
        _chain.pop(chain_key, None)  # ensure clean slate

        bars1 = [_bar(ticker=ticker, dt=datetime(2024, 1, 2))]
        r1 = create_receipt(bars1, source="yfinance", ticker=ticker, interval="1d")
        sha1 = compute_sha256(bars1)

        bars2 = [_bar(ticker=ticker, dt=datetime(2024, 1, 3))]
        r2 = create_receipt(bars2, source="yfinance", ticker=ticker, interval="1d")

        assert r1.prev_hash is None, "First receipt must have no prev_hash"
        assert r2.prev_hash == sha1, "Second receipt must link to first"

    def test_empty_bars_raises(self):
        from sentinel.sds.provenance import create_receipt
        with pytest.raises(ValueError, match="empty bar list"):
            create_receipt([], source="yfinance", ticker="AAPL", interval="1d")

    def test_chain_integrity_clean(self):
        """Properly linked chain must return no errors."""
        from sentinel.sds.provenance import create_receipt, verify_chain_integrity, _chain
        ticker = "INTEG_TEST_ABC"
        _chain.pop(f"{ticker}:yfinance:1d", None)

        bars1 = [_bar(ticker=ticker, dt=datetime(2024, 2, 1))]
        bars2 = [_bar(ticker=ticker, dt=datetime(2024, 2, 2))]
        r1 = create_receipt(bars1, source="yfinance", ticker=ticker, interval="1d")
        r2 = create_receipt(bars2, source="yfinance", ticker=ticker, interval="1d")

        errors = verify_chain_integrity([r1, r2])
        assert errors == [], f"Clean chain should have no errors, got: {errors}"


# ─── OHLCVBar Schema ──────────────────────────────────────────────────────────

class TestOHLCVBarSchema:
    """Verify the updated OHLCVBar type fields and computed property."""

    def test_ticker_field_optional(self):
        """OHLCVBar.ticker should be optional — figi is the canonical ID."""
        from sentinel.core.types import OHLCVBar
        bar = OHLCVBar(
            time=datetime(2024, 1, 2),
            figi="BBG000B9XRY4",
            open=Decimal("185.0"),
            high=Decimal("186.0"),
            low=Decimal("184.5"),
            close=Decimal("185.5"),
            volume=50_000_000,
            source="polygon",
        )
        assert bar.ticker is None

    def test_adjusted_close_with_split_factor(self):
        """adjusted_close = close × adj_factor."""
        bar = _bar(close=400.0, adj_factor=0.25)
        assert bar.adjusted_close == Decimal("100.0")

    def test_adjusted_close_unadjusted(self):
        """With adj_factor=1.0, adjusted_close equals raw close."""
        bar = _bar(close=185.50)
        assert bar.adjusted_close == bar.close

    def test_frozen_model_immutable(self):
        """OHLCVBar is frozen — direct attribute assignment must raise."""
        bar = _bar()
        with pytest.raises(Exception):
            bar.close = Decimal("999")  # type: ignore

    def test_model_copy_updates_adj_factor(self):
        """model_copy(update=...) is the correct mutation pattern on frozen models."""
        bar = _bar(close=400.0, adj_factor=1.0)
        updated = bar.model_copy(update={"adj_factor": Decimal("0.25")})
        assert updated.adj_factor == Decimal("0.25")
        assert bar.adj_factor == Decimal("1.0")   # original unchanged
        assert updated.close == bar.close          # raw close preserved


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
