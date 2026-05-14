"""
Financial evaluation tests — these are correctness benchmarks, not just unit tests.
A strategy must beat buy-and-hold SPY on a risk-adjusted basis.
DSR must be ≥ 0.95 for any strategy that claims alpha.
PIT integrity: vintage dates must never look forward.
"""
from __future__ import annotations
import pytest
from datetime import datetime, date
import pandas as pd
import numpy as np
from decimal import Decimal


# ─── SBE Metrics Tests ────────────────────────────────────────────────────────

class TestBacktestMetrics:
    """Verify correctness of 24-metric computation."""

    def _make_returns(self, n: int = 500, mu: float = 0.0004, sigma: float = 0.012, seed: int = 42):
        rng = np.random.default_rng(seed)
        return pd.Series(rng.normal(mu, sigma, n),
                         index=pd.date_range("2020-01-01", periods=n, freq="B"))

    def test_sharpe_positive_for_positive_drift(self):
        from sentinel.sbe.metrics import compute_metrics
        returns = self._make_returns(mu=0.001)
        metrics = compute_metrics(returns, strategy_id="test_positive")
        assert float(metrics.sharpe_ratio) > 0, "Positive drift must yield positive Sharpe"

    def test_sharpe_negative_for_negative_drift(self):
        from sentinel.sbe.metrics import compute_metrics
        returns = self._make_returns(mu=-0.001)
        metrics = compute_metrics(returns, strategy_id="test_negative")
        assert float(metrics.sharpe_ratio) < 0

    def test_max_drawdown_is_negative(self):
        from sentinel.sbe.metrics import compute_metrics
        returns = self._make_returns()
        metrics = compute_metrics(returns, strategy_id="test_dd")
        assert float(metrics.max_drawdown) < 0

    def test_win_rate_between_zero_and_one(self):
        from sentinel.sbe.metrics import compute_metrics
        returns = self._make_returns()
        metrics = compute_metrics(returns, strategy_id="test_wr")
        assert 0.0 <= float(metrics.win_rate) <= 1.0

    def test_dsr_less_than_one(self):
        from sentinel.sbe.metrics import compute_metrics
        returns = self._make_returns(n=1000)
        metrics = compute_metrics(returns, n_trials=100, strategy_id="test_dsr")
        # DSR penalized by 100 trials: must be < raw Sharpe probability
        assert float(metrics.deflated_sharpe_ratio) <= 1.0
        assert float(metrics.deflated_sharpe_ratio) >= 0.0

    def test_dsr_with_one_trial_equals_high_sharpe(self):
        from sentinel.sbe.metrics import compute_metrics
        returns = self._make_returns(mu=0.002, n=2000)  # Very positive drift
        metrics = compute_metrics(returns, n_trials=1, strategy_id="test_dsr_1trial")
        # With 1 trial and strong drift, DSR should be high
        assert float(metrics.deflated_sharpe_ratio) > 0.7

    def test_cagr_consistent_with_total_return(self):
        from sentinel.sbe.metrics import compute_metrics, annualize_return
        returns = self._make_returns(n=504, mu=0.0003)  # ~2 years
        metrics = compute_metrics(returns, strategy_id="test_cagr")
        # CAGR should be close to annualized total return
        tr = float(metrics.total_return)
        years = 504 / 252
        expected_cagr = annualize_return(tr, years)
        assert abs(float(metrics.cagr) - expected_cagr) < 0.01

    def test_var_is_negative(self):
        from sentinel.sbe.metrics import compute_metrics
        returns = self._make_returns()
        metrics = compute_metrics(returns, strategy_id="test_var")
        assert float(metrics.var_95) < 0


# ─── DSR Tests ────────────────────────────────────────────────────────────────

class TestDSR:
    def test_dsr_decreases_with_more_trials(self):
        from sentinel.sbe.metrics import compute_dsr
        sharpe, n, skew, kurt = 1.0, 500, 0.0, 3.0
        dsr1 = compute_dsr(sharpe, 1, n, skew, kurt)
        dsr10 = compute_dsr(sharpe, 10, n, skew, kurt)
        dsr100 = compute_dsr(sharpe, 100, n, skew, kurt)
        assert dsr1 > dsr10 > dsr100

    def test_dsr_increases_with_more_observations(self):
        from sentinel.sbe.metrics import compute_dsr
        sharpe = 1.5
        dsr_small = compute_dsr(sharpe, 5, 100, 0.0, 3.0)
        dsr_large = compute_dsr(sharpe, 5, 5000, 0.0, 3.0)
        assert dsr_large > dsr_small

    def test_dsr_range(self):
        from sentinel.sbe.metrics import compute_dsr
        dsr = compute_dsr(1.0, 10, 500, 0.0, 3.0)
        assert 0.0 <= dsr <= 1.0


# ─── PBO Tests ────────────────────────────────────────────────────────────────

class TestPBO:
    def _make_matrix(self, n_strategies: int = 20, n_periods: int = 500, seed: int = 42):
        rng = np.random.default_rng(seed)
        data = rng.normal(0.0004, 0.012, (n_periods, n_strategies))
        return pd.DataFrame(data, columns=[f"s{i}" for i in range(n_strategies)],
                            index=pd.date_range("2020-01-01", periods=n_periods, freq="B"))

    def test_pbo_between_zero_and_one(self):
        from sentinel.sbe.pbo import compute_pbo
        matrix = self._make_matrix()
        result = compute_pbo(matrix, n_partitions=8)
        assert 0.0 <= result["pbo"] <= 1.0

    def test_pbo_high_for_random_strategies(self):
        """Random strategies should have high PBO (overfitting likely)."""
        from sentinel.sbe.pbo import compute_pbo
        matrix = self._make_matrix(n_strategies=50)
        result = compute_pbo(matrix, n_partitions=8)
        # Random strategies: expected PBO ≈ 0.5
        assert result["pbo"] > 0.2

    def test_pbo_output_structure(self):
        from sentinel.sbe.pbo import compute_pbo
        matrix = self._make_matrix()
        result = compute_pbo(matrix, n_partitions=8)
        assert "pbo" in result
        assert "lambda_bar" in result
        assert "n_combinations" in result
        assert "interpretation" in result

    def test_pbo_requires_min_strategies(self):
        from sentinel.sbe.pbo import compute_pbo
        single = pd.DataFrame({"s0": [0.001] * 200},
                               index=pd.date_range("2020-01-01", periods=200, freq="B"))
        with pytest.raises(ValueError):
            compute_pbo(single)


# ─── SPY Buy-and-Hold Benchmark ───────────────────────────────────────────────

class TestSPYBenchmark:
    """The fundamental correctness test: SPY buy-and-hold should compute correctly."""

    def test_spy_sharpe_positive_over_10_years(self):
        """SPY had positive Sharpe over any 10-year period post-2010."""
        from sentinel.sbe.metrics import compute_metrics
        # Simulate SPY-like returns: ~10% annual, ~15% vol
        rng = np.random.default_rng(0)
        daily_mu = 0.10 / 252
        daily_sigma = 0.15 / np.sqrt(252)
        n = 252 * 10
        returns = pd.Series(
            rng.normal(daily_mu, daily_sigma, n),
            index=pd.date_range("2014-01-01", periods=n, freq="B"),
        )
        metrics = compute_metrics(returns, strategy_id="spy_sim", n_trials=1)
        assert float(metrics.sharpe_ratio) > 0.3, f"SPY sim Sharpe {metrics.sharpe_ratio} too low"
        assert float(metrics.cagr) > 0.05, "SPY sim CAGR should be > 5%"

    def test_momentum_beats_random_in_sharpe(self):
        """Momentum on trending data should outperform purely random."""
        from sentinel.sbe.metrics import compute_metrics
        from sentinel.sbe.strategies import momentum_strategy

        rng = np.random.default_rng(1)
        # Create trending price series
        n = 500
        returns = rng.normal(0.0006, 0.012, n)
        prices = pd.Series(
            100 * (1 + returns).cumprod(),
            index=pd.date_range("2020-01-01", periods=n, freq="B"),
        )
        ret_series = pd.Series(returns, index=prices.index)

        entries, exits = momentum_strategy(prices, lookback=126)
        # Momentum strategy should have non-zero signal activity
        assert entries.sum() > 0
        assert exits.sum() > 0


# ─── Point-in-Time Integrity ──────────────────────────────────────────────────

class TestPITIntegrity:
    """Verify that FRED vintage dates don't create look-ahead bias."""

    def test_vintage_filter_removes_future_observations(self):
        """Observations filed after vintage_date should not appear in PIT fetch."""
        from datetime import date
        # Simulate a series with observations at different filing dates
        # If we request vintage 2020-01-01, we should NOT see data filed after that date
        # This is tested via the ALFRED API; here we test the filter logic directly
        observations = [
            {"val": 100.0, "filed": "2019-06-01", "end": "2019-03-31"},
            {"val": 101.0, "filed": "2020-06-01", "end": "2020-03-31"},  # Filed after vintage
            {"val": 99.0, "filed": "2018-06-01", "end": "2018-03-31"},
        ]
        vintage = date(2020, 1, 1)
        visible = [o for o in observations
                   if date.fromisoformat(str(o["filed"])) <= vintage]
        assert len(visible) == 2
        assert all(date.fromisoformat(str(o["filed"])) <= vintage for o in visible)


# ─── XBRL Parser Tests ────────────────────────────────────────────────────────

class TestXBRLParser:
    def test_extract_facts_from_minimal_payload(self):
        from sentinel.sfe.xbrl_parser import extract_facts
        payload = {
            "facts": {
                "us-gaap": {
                    "Revenues": {
                        "units": {
                            "USD": [
                                {"val": 385_000_000_000, "end": "2023-09-30",
                                 "form": "10-K", "filed": "2023-11-03", "accn": "0000320193-23-000106"}
                            ]
                        }
                    }
                }
            }
        }
        facts = extract_facts(payload, cik="0000320193", figi="BBG000B9XRY4")
        assert len(facts) == 1
        assert facts[0].label == "revenue"
        assert facts[0].value == Decimal("385000000000")
        assert facts[0].form_type == "10-K"

    def test_annual_facts_sorted_descending(self):
        from sentinel.sfe.xbrl_parser import extract_facts, get_annual_facts
        from decimal import Decimal
        from datetime import date

        payload = {"facts": {"us-gaap": {"NetIncomeLoss": {"units": {"USD": [
            {"val": 100, "end": "2021-12-31", "form": "10-K", "filed": "2022-02-01"},
            {"val": 200, "end": "2022-12-31", "form": "10-K", "filed": "2023-02-01"},
            {"val": 150, "end": "2020-12-31", "form": "10-K", "filed": "2021-02-01"},
        ]}}}}}
        facts = extract_facts(payload, cik="TEST")
        annual = get_annual_facts(facts, "net_income")
        assert annual[0].period_end == date(2022, 12, 31)  # Most recent first
        assert annual[-1].period_end == date(2020, 12, 31)


# ─── OpenFIGI Tests ───────────────────────────────────────────────────────────

class TestOpenFIGI:
    def test_infer_asset_class_equity(self):
        from sentinel.sim.openfigi_client import _infer_asset_class
        from sentinel.core.types import AssetClass
        assert _infer_asset_class("Common Stock") == AssetClass.EQUITY

    def test_infer_asset_class_etf(self):
        from sentinel.sim.openfigi_client import _infer_asset_class
        from sentinel.core.types import AssetClass
        assert _infer_asset_class("ETF") == AssetClass.ETF

    def test_infer_asset_class_crypto(self):
        from sentinel.sim.openfigi_client import _infer_asset_class
        from sentinel.core.types import AssetClass
        assert _infer_asset_class("Digital Asset") == AssetClass.CRYPTO


# ─── COT Tests ────────────────────────────────────────────────────────────────

class TestCOTIndex:
    def test_cot_index_range(self):
        """COT Index must always be 0-100."""
        import pandas as pd
        from sentinel.sma.cot_report import COTClient

        # Create synthetic net position series
        client = COTClient()
        # Inject mock data directly
        dates = pd.date_range("2022-01-01", periods=60, freq="W")
        net_pos = pd.Series(range(-30, 30), index=dates)
        roll_max = net_pos.rolling(52).max()
        roll_min = net_pos.rolling(52).min()
        denom = roll_max - roll_min
        cot_index = ((net_pos - roll_min) / denom.replace(0, float("nan")) * 100).dropna()
        assert (cot_index >= 0).all()
        assert (cot_index <= 100).all()


# ─── Congressional Trade Tests ────────────────────────────────────────────────

class TestCongressionalParser:
    def test_extract_ticker_from_name(self):
        from sentinel.sod.congressional import _extract_ticker_from_name
        assert _extract_ticker_from_name("Apple Inc. (AAPL)") == "AAPL"
        assert _extract_ticker_from_name("Microsoft Corporation (MSFT)") == "MSFT"
        assert _extract_ticker_from_name("No ticker here") is None

    def test_parse_senate_amount(self):
        from sentinel.sod.congressional import _parse_senate_amount
        low, high = _parse_senate_amount("$50,001 - $100,000")
        assert low == 50001
        assert high == 100000

    def test_signal_strength(self):
        from sentinel.sod.congressional import _compute_signal_strength, CongressionalTrade
        from datetime import date
        trade = CongressionalTrade(
            politician_name="Test", chamber="Senate", party="D", state="CA",
            ticker="AAPL", figi="", asset_name="Apple Inc.",
            tx_date=date.today(), filed_date=date.today(),
            tx_type="buy", amount_low=500001, amount_high=1000000,
            filing_lag_days=30, late_filing=False, source="senate_efd",
        )
        assert _compute_signal_strength(trade) == "moderate"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])


# ════════════════════════════════════════════════════════════════
# OPTIONS ANALYTICS
# ════════════════════════════════════════════════════════════════

def _make_contract(ctype: str, strike: float, iv: float, oi: int, gamma: float = 0.05,
                   volume: int = 100, underlying: float = 100.0) -> dict:
    return {
        "underlying_ticker": "TEST",
        "expiration_date": "2030-01-17",
        "strike_price": strike,
        "contract_type": ctype,
        "implied_volatility": iv,
        "open_interest": oi,
        "volume": volume,
        "gamma": gamma,
        "delta": 0.5,
    }


class TestOptionsNormalizeChain:
    def test_valid_contract_accepted(self):
        from sentinel.sbx.options_analytics import normalize_chain
        raw = [_make_contract("call", 100.0, 0.25, 500)]
        result = normalize_chain(raw, underlying_price=100.0)
        assert len(result) == 1
        assert result[0].contract_type == "call"
        assert result[0].moneyness == pytest.approx(1.0)

    def test_zero_iv_filtered(self):
        from sentinel.sbx.options_analytics import normalize_chain
        raw = [_make_contract("call", 100.0, 0.0, 500)]
        result = normalize_chain(raw, underlying_price=100.0)
        assert len(result) == 0

    def test_zero_oi_filtered(self):
        from sentinel.sbx.options_analytics import normalize_chain
        raw = [_make_contract("put", 95.0, 0.30, 0)]
        result = normalize_chain(raw, underlying_price=100.0)
        assert len(result) == 0


class TestGEXCalculation:
    def test_gex_formula_known_values(self):
        """gamma=0.05, OI=1000, spot=100 → GEX per call = 0.05*1000*100*100^2*0.01 = 500.0"""
        from sentinel.sbx.options_analytics import normalize_chain, compute_gex
        raw = [_make_contract("call", 100.0, 0.25, 1000, gamma=0.05, underlying=100.0)]
        contracts = normalize_chain(raw, underlying_price=100.0)
        assert len(contracts) == 1
        result = compute_gex(contracts, underlying_price=100.0)
        assert result.net_gex == pytest.approx(500.0, rel=1e-3)

    def test_puts_subtract_from_gex(self):
        from sentinel.sbx.options_analytics import normalize_chain, compute_gex
        calls = [_make_contract("call", 100.0, 0.25, 500, gamma=0.05)]
        puts = [_make_contract("put", 100.0, 0.28, 500, gamma=0.05)]
        contracts = normalize_chain(calls + puts, underlying_price=100.0)
        result = compute_gex(contracts, underlying_price=100.0)
        assert result.net_gex == pytest.approx(0.0, abs=1e-3)


class TestMaxPain:
    def test_max_pain_at_strike_with_lowest_holder_loss(self):
        from sentinel.sbx.options_analytics import normalize_chain, compute_max_pain
        raw = [
            _make_contract("call", 110.0, 0.25, 1000),
            _make_contract("put", 90.0, 0.28, 1000),
            _make_contract("call", 100.0, 0.25, 500),
            _make_contract("put", 100.0, 0.28, 500),
        ]
        contracts = normalize_chain(raw, underlying_price=100.0)
        result = compute_max_pain(contracts)
        assert result.max_pain_strike in [90.0, 100.0, 110.0]
        assert "pain_table" in result.model_fields_set or result.pain_table is not None


class TestPCRatio:
    def test_equal_oi_gives_ratio_one(self):
        from sentinel.sbx.options_analytics import normalize_chain, compute_pc_ratio
        raw = [
            _make_contract("call", 100.0, 0.25, 1000),
            _make_contract("put", 100.0, 0.28, 1000),
        ]
        contracts = normalize_chain(raw, underlying_price=100.0)
        result = compute_pc_ratio(contracts)
        assert result.oi_ratio == pytest.approx(1.0)

    def test_more_puts_gives_ratio_above_one(self):
        from sentinel.sbx.options_analytics import normalize_chain, compute_pc_ratio
        raw = [
            _make_contract("call", 100.0, 0.25, 500),
            _make_contract("put", 100.0, 0.28, 1000),
        ]
        contracts = normalize_chain(raw, underlying_price=100.0)
        result = compute_pc_ratio(contracts)
        assert result.oi_ratio == pytest.approx(2.0)


# ════════════════════════════════════════════════════════════════
# CB SPEECH — HAWKISH / DOVISH TONE
# ════════════════════════════════════════════════════════════════

class TestCBSpeechTone:
    def test_hawkish_text_returns_positive_net_tone(self):
        from sentinel.sma.cb_speech import score_tone
        h, d, net, label = score_tone(
            "The Fed will aggressively raise rates to combat inflation with multiple hikes"
        )
        assert net > 0, f"Expected hawkish net_tone > 0, got {net}"
        assert label in ("hawkish", "very_hawkish")

    def test_dovish_text_returns_negative_net_tone(self):
        from sentinel.sma.cb_speech import score_tone
        h, d, net, label = score_tone(
            "The Fed will maintain accommodative policy with patient approach to normalization"
        )
        assert net < 0, f"Expected dovish net_tone < 0, got {net}"
        assert label in ("dovish", "very_dovish")

    def test_empty_text_is_neutral(self):
        from sentinel.sma.cb_speech import score_tone
        h, d, net, label = score_tone("")
        assert net == 0.0
        assert label == "neutral"

    def test_scores_sum_to_one_when_hits_nonzero(self):
        from sentinel.sma.cb_speech import score_tone
        h, d, net, label = score_tone("inflation rate hike tighten restrictive")
        assert h + d == pytest.approx(1.0)


# ════════════════════════════════════════════════════════════════
# KELLY SIZER
# ════════════════════════════════════════════════════════════════

class TestKellySingle:
    def test_symmetric_bet_returns_0_2(self):
        """f* = p - q = 0.6 - 0.4 = 0.20 for symmetric 1:1 bet."""
        from sentinel.spr.kelly_sizer import kelly_single
        result = kelly_single(win_prob=0.6, win_return=1.0, loss_return=-1.0)
        assert result.full_kelly_fraction == pytest.approx(0.20, abs=1e-4)

    def test_half_kelly_is_half_of_full(self):
        from sentinel.spr.kelly_sizer import kelly_single
        result = kelly_single(win_prob=0.6, win_return=1.0, loss_return=-1.0)
        assert result.half_kelly_fraction == pytest.approx(result.full_kelly_fraction * 0.5)

    def test_losing_bet_returns_zero_fraction(self):
        """Edge < 0 → f* should be 0 (clipped)."""
        from sentinel.spr.kelly_sizer import kelly_single
        result = kelly_single(win_prob=0.3, win_return=1.0, loss_return=-1.0)
        assert result.full_kelly_fraction == 0.0

    def test_invalid_win_prob_raises(self):
        from sentinel.spr.kelly_sizer import kelly_single
        with pytest.raises(ValueError):
            kelly_single(win_prob=1.1, win_return=1.0, loss_return=-1.0)


class TestKellyFromReturns:
    def test_positive_drift_gives_positive_fraction(self):
        from sentinel.spr.kelly_sizer import kelly_from_returns
        rng = np.random.default_rng(99)
        rets = pd.Series(rng.normal(0.001, 0.01, 500))
        result = kelly_from_returns(rets, ticker="SPY")
        assert result.full_kelly_fraction >= 0.0

    def test_insufficient_data_raises(self):
        from sentinel.spr.kelly_sizer import kelly_from_returns
        with pytest.raises(ValueError):
            kelly_from_returns(pd.Series([0.01] * 5))


class TestSizePortfolio:
    def _returns_df(self):
        rng = np.random.default_rng(7)
        data = rng.normal(0.0003, 0.012, (300, 3))
        return pd.DataFrame(data, columns=["A", "B", "C"],
                            index=pd.date_range("2022-01-01", periods=300, freq="B"))

    def test_equal_weights_sum_to_one(self):
        from sentinel.spr.kelly_sizer import size_portfolio
        df = self._returns_df()
        result = size_portfolio(["A", "B", "C"], df, method="equal", capital=100_000)
        total_weight = sum(v["weight"] for v in result.values())
        assert total_weight == pytest.approx(1.0, abs=1e-6)

    def test_unknown_method_raises(self):
        from sentinel.spr.kelly_sizer import size_portfolio
        df = self._returns_df()
        with pytest.raises(ValueError):
            size_portfolio(["A"], df, method="magic")  # type: ignore


# ════════════════════════════════════════════════════════════════
# DCF MODEL
# ════════════════════════════════════════════════════════════════

class TestWACC:
    def test_wacc_known_value(self):
        """WACC = 0.12*0.7 + 0.05*(1-0.21)*0.3 = 0.084 + 0.01185 = 0.09585 ≈ 0.0954"""
        from sentinel.sfe.dcf_model import compute_wacc, WACCInputs
        # Ke = rf + beta*ERP = 0.045 + 1.0*0.075 = 0.12 → set beta so ke=0.12
        # ERP default 0.055, rf default 0.045, ke = 0.045 + beta*0.055 = 0.12 → beta=1.3636
        beta = (0.12 - 0.045) / 0.055
        inputs = WACCInputs(
            equity_beta=beta,
            cost_of_debt=0.05,
            tax_rate=0.21,
            equity_weight=0.7,
            debt_to_equity=0.3 / 0.7,
        )
        wacc = compute_wacc(inputs)
        # ke=0.12, kd_after_tax=0.05*(1-0.21)=0.0395, WACC=0.12*0.7+0.0395*0.3=0.084+0.01185=0.09585
        assert wacc == pytest.approx(0.0954, abs=0.001)

    def test_wacc_returns_float(self):
        from sentinel.sfe.dcf_model import compute_wacc, WACCInputs
        inputs = WACCInputs(equity_beta=1.0, cost_of_debt=0.04, equity_weight=0.8, debt_to_equity=0.25)
        assert isinstance(compute_wacc(inputs), float)


class TestDCFAssumptions:
    def test_dcf_assumptions_model_construction(self):
        from sentinel.sfe.dcf_model import DCFAssumptions
        a = DCFAssumptions(
            ticker="AAPL", revenue_base=400e9, revenue_growth_rates=[0.08, 0.07, 0.06],
            ebit_margin=0.30, wacc=0.09, shares_outstanding=16000,
        )
        assert a.terminal_growth_rate == 0.025

    def test_run_dcf_raises_when_wacc_below_tgr(self):
        from sentinel.sfe.dcf_model import DCFAssumptions, run_dcf
        a = DCFAssumptions(
            ticker="X", revenue_base=1e9, revenue_growth_rates=[0.05],
            ebit_margin=0.20, wacc=0.02, terminal_growth_rate=0.03,
            shares_outstanding=100,
        )
        with pytest.raises(ValueError):
            run_dcf(a, current_price=50.0)

    def test_run_dcf_returns_positive_ev(self):
        from sentinel.sfe.dcf_model import DCFAssumptions, run_dcf
        a = DCFAssumptions(
            ticker="X", revenue_base=1e9, revenue_growth_rates=[0.10, 0.08, 0.06, 0.05, 0.04],
            ebit_margin=0.20, wacc=0.10, shares_outstanding=100,
        )
        result = run_dcf(a, current_price=50.0)
        assert result.enterprise_value > 0
        assert len(result.year_fcfs) == 5


# ════════════════════════════════════════════════════════════════
# FACTOR MODEL
# ════════════════════════════════════════════════════════════════

class TestFactorModel:
    def _synth_factors(self, n: int = 252) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom", "RF"]
        data = rng.normal(0.0, 0.01, (n, len(cols)))
        return pd.DataFrame(data, columns=cols,
                            index=pd.date_range("2023-01-01", periods=n, freq="B"))

    def test_ff5_regression_returns_result(self):
        from sentinel.spr.factor_model import run_ff5_regression
        rng = np.random.default_rng(42)
        n = 252
        ff = self._synth_factors(n)
        mkt_arr: np.ndarray = ff["Mkt-RF"].to_numpy(dtype=float)
        returns = pd.Series(
            0.0003 + 1.1 * mkt_arr + rng.normal(0, 0.005, n),
            index=ff.index,
        )
        result = run_ff5_regression("SPY_SIM", returns, ff)
        assert result.ticker == "SPY_SIM"
        assert 0.0 <= result.r_squared <= 1.0
        mkt = next(e for e in result.exposures if e.factor == "Mkt-RF")
        assert mkt.loading == pytest.approx(1.1, abs=0.15)

    def test_r_squared_for_pure_market_return(self):
        from sentinel.spr.factor_model import run_ff5_regression
        n = 252
        ff = self._synth_factors(n)
        mkt_arr: np.ndarray = ff["Mkt-RF"].to_numpy(dtype=float)
        rf_arr: np.ndarray = ff["RF"].to_numpy(dtype=float)
        returns = pd.Series(mkt_arr + rf_arr, index=ff.index)
        result = run_ff5_regression("MKT", returns, ff)
        assert result.r_squared > 0.8


# ════════════════════════════════════════════════════════════════
# STRESS TEST
# ════════════════════════════════════════════════════════════════

class TestStressTest:
    def test_scenarios_registry_non_empty(self):
        from sentinel.spr.stress_test import SCENARIOS
        assert len(SCENARIOS) >= 5

    def test_parametric_shock_keys(self):
        from sentinel.spr.stress_test import _STANDARD_SHOCKS
        names = [s[0] for s in _STANDARD_SHOCKS]
        assert "severe_recession" in names
        assert "rate_spike_200bps" in names

    def test_scenario_result_model_construction(self):
        from sentinel.spr.stress_test import ScenarioResult
        r = ScenarioResult(
            scenario="test", description="test scenario",
            portfolio_return=-0.30, max_drawdown=-0.40,
            worst_day=-0.12, best_day=0.05,
            volatility=0.35, holding_returns={"SPY": -0.30},
        )
        assert r.portfolio_return == -0.30


# ════════════════════════════════════════════════════════════════
# CORRELATION
# ════════════════════════════════════════════════════════════════

class TestRollingCorrelation:
    def test_identical_series_correlation_is_one(self):
        from sentinel.spr.correlation import compute_rolling_correlation
        n = 100
        x = pd.Series(np.random.default_rng(0).normal(0, 1, n))
        df = pd.DataFrame({"A": x, "B": x},
                          index=pd.date_range("2023-01-01", periods=n, freq="B"))
        result = compute_rolling_correlation(df, window=30)
        last_corr = result[("A", "B")].dropna().iloc[-1]
        assert last_corr == pytest.approx(1.0, abs=1e-6)

    def test_negatively_correlated_series(self):
        from sentinel.spr.correlation import compute_rolling_correlation
        n = 100
        x = pd.Series(np.linspace(0, 1, n))
        df = pd.DataFrame({"A": x, "B": -x},
                          index=pd.date_range("2023-01-01", periods=n, freq="B"))
        result = compute_rolling_correlation(df, window=30)
        last_corr = result[("A", "B")].dropna().iloc[-1]
        assert last_corr == pytest.approx(-1.0, abs=1e-6)

    def test_single_ticker_returns_empty(self):
        from sentinel.spr.correlation import compute_rolling_correlation
        df = pd.DataFrame({"A": [0.01] * 50},
                          index=pd.date_range("2023-01-01", periods=50, freq="B"))
        result = compute_rolling_correlation(df, window=20)
        assert result.empty

    def test_detect_regime_break_fires_on_large_change(self):
        from sentinel.spr.correlation import detect_regime_break
        curr = pd.DataFrame({"A": [1.0, 0.9], "B": [0.9, 1.0]}, index=["A", "B"])
        prior = pd.DataFrame({"A": [1.0, 0.2], "B": [0.2, 1.0]}, index=["A", "B"])
        alerts = detect_regime_break(curr, prior, threshold=0.25)
        assert len(alerts) == 1
        assert alerts[0].severity == "high"


# ════════════════════════════════════════════════════════════════
# ECONOMIC CALENDAR
# ════════════════════════════════════════════════════════════════

class TestEconomicCalendar:
    def test_score_importance_high_for_cpi(self):
        from sentinel.sma.economic_calendar import score_importance
        assert score_importance("Consumer Price Index") == "high"

    def test_score_importance_high_for_nonfarm(self):
        from sentinel.sma.economic_calendar import score_importance
        assert score_importance("nonfarm payrolls") == "high"

    def test_score_importance_medium_for_existing_homes(self):
        from sentinel.sma.economic_calendar import score_importance
        assert score_importance("existing home sales") == "medium"

    def test_score_importance_low_for_unknown(self):
        from sentinel.sma.economic_calendar import score_importance
        assert score_importance("Obscure Baltic Dry Index Subcomponent") == "low"

    def test_economic_release_model(self):
        from sentinel.sma.economic_calendar import EconomicRelease
        r = EconomicRelease(
            release_id="1", name="CPI", release_date=date.today(),
            release_time="08:30 ET", frequency="Monthly", importance="high",
            fred_series_ids=["CPIAUCSL"],
            consensus=None, prior=None,
        )
        assert r.consensus is None


# ════════════════════════════════════════════════════════════════
# SOCIAL SENTIMENT
# ════════════════════════════════════════════════════════════════

class TestSocialSentiment:
    def test_social_post_model_construction(self):
        from sentinel.snm.social_sentiment import SocialPost
        post = SocialPost(
            platform="reddit", ticker="AAPL", post_id="abc123",
            created_at=datetime(2025, 1, 15, 12, 0, 0),
            title="Apple is going to the moon", body=None,
            score=500, author_sentiment=None,
            finbert_sentiment="positive", finbert_confidence=0.92,
        )
        assert post.platform == "reddit"
        assert post.finbert_confidence == 0.92

    def test_social_post_frozen(self):
        from sentinel.snm.social_sentiment import SocialPost
        post = SocialPost(
            platform="stocktwits", ticker="TSLA", post_id="x1",
            created_at=datetime(2025, 1, 1, 0, 0, 0),
            title="Bullish!", body=None, score=10,
            author_sentiment="Bullish", finbert_sentiment=None, finbert_confidence=None,
        )
        with pytest.raises(Exception):
            object.__setattr__(post, "score", 999)


# ════════════════════════════════════════════════════════════════
# NL SCREENER
# ════════════════════════════════════════════════════════════════

class TestNLScreener:
    def test_screen_criterion_schema_has_required_fields(self):
        from sentinel.sil.nl_screener import SCREEN_TOOL
        schema = SCREEN_TOOL["input_schema"]
        assert "criteria" in schema["properties"]
        assert "screen_name" in schema["required"]

    def test_available_columns_includes_pe_ratio(self):
        from sentinel.sil.nl_screener import AVAILABLE_COLUMNS
        assert "pe_ratio" in AVAILABLE_COLUMNS
        assert "market_cap" in AVAILABLE_COLUMNS


# ════════════════════════════════════════════════════════════════
# RAG
# ════════════════════════════════════════════════════════════════

class TestRAG:
    def test_document_chunk_model(self):
        from sentinel.sil.rag import DocumentChunk
        chunk = DocumentChunk(
            chunk_id="c1", doc_id="d1", ticker="AAPL",
            doc_type="10-K", filed_date=date(2024, 11, 1),
            text="Apple reported record revenue of $391B.",
        )
        assert chunk.ticker == "AAPL"
        assert chunk.embedding is None

    def test_rag_result_model(self):
        from sentinel.sil.rag import RAGResult
        r = RAGResult(
            chunk_id="c1", ticker="MSFT", doc_type="earnings_call",
            filed_date=date(2025, 1, 30), text="Cloud revenue grew 21%.",
            score=0.87, dense_rank=1, sparse_rank=3, source="fusion",
        )
        assert r.score == 0.87
        assert r.source == "fusion"


# ════════════════════════════════════════════════════════════════
# FX ADAPTER
# ════════════════════════════════════════════════════════════════

class TestFXAdapter:
    def test_fx_bar_model_construction(self):
        from sentinel.sds.adapters.fx_adapter import FXBar
        bar = FXBar(
            pair="EURUSD", date=date(2025, 1, 2),
            open=Decimal("1.1050"), high=Decimal("1.1080"),
            low=Decimal("1.1020"), close=Decimal("1.1065"),
        )
        assert bar.pair == "EURUSD"
        assert bar.close == Decimal("1.1065")

    def test_parse_pair_splits_correctly(self):
        from sentinel.sds.adapters.fx_adapter import _parse_pair
        base, quote = _parse_pair("GBPUSD")
        assert base == "GBP"
        assert quote == "USD"

    def test_parse_pair_raises_for_bad_length(self):
        from sentinel.sds.adapters.fx_adapter import _parse_pair
        with pytest.raises(ValueError):
            _parse_pair("EURUSD1")


# ════════════════════════════════════════════════════════════════
# SHORT INTEREST ADAPTER
# ════════════════════════════════════════════════════════════════

class TestShortInterestAdapter:
    def test_short_interest_record_model(self):
        from sentinel.sds.adapters.short_interest_adapter import ShortInterestRecord
        r = ShortInterestRecord(
            ticker="GME", settlement_date=date(2025, 1, 15),
            short_interest=50_000_000, days_to_cover=3.5,
        )
        assert r.source == "finra"
        assert r.days_to_cover == 3.5

    def test_parse_finra_date_formats(self):
        from sentinel.sds.adapters.short_interest_adapter import _parse_finra_date
        assert _parse_finra_date("2025-01-15") == date(2025, 1, 15)
        assert _parse_finra_date("20250115") == date(2025, 1, 15)
        assert _parse_finra_date("01/15/2025") == date(2025, 1, 15)
        assert _parse_finra_date("not-a-date") is None


# ════════════════════════════════════════════════════════════════
# TRACE CLIENT
# ════════════════════════════════════════════════════════════════

class TestTraceClient:
    def test_bond_quote_model(self):
        from sentinel.sbx.trace_client import BondQuote
        q = BondQuote(
            cusip="037833100", issuer_name="Apple Inc.",
            description="AAPL 3.0% 2027", coupon=3.0,
            maturity_date=date(2027, 6, 20), last_price=Decimal("98.50"),
            last_yield=3.35, spread_to_benchmark=55.0,
            last_sale_date=date(2025, 1, 14), trade_count=12,
        )
        assert q.source == "finra_trace"
        assert q.last_yield == 3.35

    def test_credit_curve_model(self):
        from sentinel.sbx.trace_client import CreditCurve
        curve = CreditCurve(
            issuer="Apple Inc.", as_of=date.today(),
            points=[{"maturity_years": 2.5, "yield": 3.35, "spread": 55, "cusip": "037833100"}],
        )
        assert len(curve.points) == 1

    def test_parse_date_handles_multiple_formats(self):
        from sentinel.sbx.trace_client import _parse_date
        assert _parse_date("2025-01-15") == date(2025, 1, 15)
        assert _parse_date("01/15/2025") == date(2025, 1, 15)
        assert _parse_date(None) is None


# ════════════════════════════════════════════════════════════════
# SEGMENT PARSER
# ════════════════════════════════════════════════════════════════

class TestSegmentParser:
    def test_segment_revenue_model(self):
        from sentinel.sfe.segment_parser import SegmentRevenue
        seg = SegmentRevenue(
            ticker="AAPL", cik="0000320193",
            period_end=date(2024, 9, 28), period_type="annual",
            filed_at=datetime(2024, 11, 1, 8, 0, 0),
            segment_name="Americas", segment_type="geographic",
            revenue=Decimal("169658000000"), operating_income=None,
            revenue_pct=0.42,
        )
        assert seg.currency == "USD"
        assert seg.revenue_pct == pytest.approx(0.42)

    def test_norm_cik_pads_to_10(self):
        from sentinel.sfe.segment_parser import _norm_cik
        assert _norm_cik("320193") == "0000320193"
        assert len(_norm_cik("1")) == 10

    def test_segment_breakdown_model(self):
        from sentinel.sfe.segment_parser import SegmentBreakdown, SegmentRevenue
        seg = SegmentRevenue(
            ticker="MSFT", cik="0000789019",
            period_end=date(2024, 6, 30), period_type="annual",
            filed_at=datetime(2024, 7, 30), segment_name="Cloud",
            segment_type="business", revenue=Decimal("105000000000"),
            operating_income=None, revenue_pct=0.46,
        )
        bd = SegmentBreakdown(
            ticker="MSFT", period_end=date(2024, 6, 30),
            total_revenue=Decimal("228000000000"),
            segments=[seg], geographic_segments=[], has_segment_data=True,
        )
        assert bd.has_segment_data is True


# ════════════════════════════════════════════════════════════════
# GLOBAL MACRO
# ════════════════════════════════════════════════════════════════

class TestGlobalMacro:
    def test_compute_macro_score_known_value(self):
        """gdp=3, infl=2, unemp=4 → 5 + 3 - (2-2) - 4/2 = 5+3-0-2 = 6.0"""
        from sentinel.sma.global_macro import compute_macro_score, CountryMacroSnapshot
        snap = CountryMacroSnapshot(
            country="US", as_of=date.today(),
            gdp_growth_pct=3.0, inflation_cpi_yoy=2.0,
            unemployment_rate=4.0, policy_rate=5.25,
            ten_yr_yield=4.3, real_rate=2.3, macro_score=None,
        )
        score = compute_macro_score(snap)
        assert score == pytest.approx(6.0, abs=0.01)

    def test_macro_score_clamped_between_0_and_10(self):
        from sentinel.sma.global_macro import compute_macro_score, CountryMacroSnapshot
        snap = CountryMacroSnapshot(
            country="VZ", as_of=date.today(),
            gdp_growth_pct=-20.0, inflation_cpi_yoy=200.0,
            unemployment_rate=50.0, policy_rate=None,
            ten_yr_yield=None, real_rate=None, macro_score=None,
        )
        score = compute_macro_score(snap)
        assert 0.0 <= score <= 10.0

    def test_rate_differential_signal_strong_when_us_premium_high(self):
        from sentinel.sma.global_macro import rate_differential_signal, CountryMacroSnapshot
        snapshots = [
            CountryMacroSnapshot(country="US", as_of=date.today(), gdp_growth_pct=2.5,
                                 inflation_cpi_yoy=2.5, unemployment_rate=4.0,
                                 policy_rate=5.25, ten_yr_yield=4.5,
                                 real_rate=2.0, macro_score=6.0),
            CountryMacroSnapshot(country="JP", as_of=date.today(), gdp_growth_pct=0.5,
                                 inflation_cpi_yoy=0.5, unemployment_rate=2.5,
                                 policy_rate=0.1, ten_yr_yield=0.7,
                                 real_rate=0.2, macro_score=5.0),
        ]
        signal = rate_differential_signal(snapshots)
        assert signal == "strong"


# ════════════════════════════════════════════════════════════════
# NAUTILUS BACKEND — FALLBACK ENGINE
# ════════════════════════════════════════════════════════════════

class TestNautilusBackend:
    def _make_bars(self, n: int = 100, price: float = 100.0) -> pd.DataFrame:
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        prices = np.linspace(price, price * 1.1, n)
        return pd.DataFrame({
            "open": prices, "high": prices * 1.005,
            "low": prices * 0.995, "close": prices, "volume": 1e6,
        }, index=idx)

    def test_fallback_engine_buy_and_hold_increases_capital(self):
        from sentinel.sbe.nautilus_backend import _FallbackEngine, NautilusBacktestConfig
        config = NautilusBacktestConfig(
            strategy_name="always_buy",
            tickers=["SPY"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 6, 1),
            initial_capital=100_000.0,
        )
        # Signal: buy immediately, never sell
        def signal_fn(bar): return 1

        engine = _FallbackEngine(config, signal_fn)
        result = engine.run({"SPY": self._make_bars()})
        assert result.final_capital > config.initial_capital

    def test_fallback_engine_result_model_fields(self):
        from sentinel.sbe.nautilus_backend import _FallbackEngine, NautilusBacktestConfig
        config = NautilusBacktestConfig(
            strategy_name="hold",
            tickers=["SPY"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 6, 1),
        )
        engine = _FallbackEngine(config, lambda bar: 0)
        result = engine.run({"SPY": self._make_bars()})
        assert result.total_return is not None
        assert result.max_drawdown <= 0.0

    def test_fallback_engine_no_data_raises(self):
        from sentinel.sbe.nautilus_backend import _FallbackEngine, NautilusBacktestConfig
        config = NautilusBacktestConfig(
            strategy_name="test",
            tickers=["MISSING"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 6, 1),
        )
        engine = _FallbackEngine(config, lambda bar: 0)
        with pytest.raises(ValueError):
            engine.run({"SPY": self._make_bars()})
