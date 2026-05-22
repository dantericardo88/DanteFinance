# SENTINEL — Nexus-Style Agentic Forecasting Framework
**Version:** 1.1 (adds full-text prompt templates, instrument-universe roster, SLOs, capacity planning, Anthropic SDK doctrine, reproducibility manifest, data-flow diagram, internal-consistency fixes)
**Date:** 2026-05-18 (v1.0 → v1.1 same-day revision)
**Status:** Ready for forge — Phase 0 through Phase 8 fully specified
**Owner:** Ricky Porras / Sentinel core (richard.porras@realempanada.com)
**Classification:** Substrate-defining capability PRD — extends `SENTINEL_PRD_v2.0`
**Precondition:** Sentinel ≥ 7.9/10, Wave 35 complete (commit `dd8799e`), Phase 0 prerequisites met
**Target ship:** 2026-06-15 (28 calendar days; 20 engineering days through Phase 8 + 1 ship day = 21 total; 3 buffer days)
**Cost envelope:** ≤ $300/month steady state at 48-instrument universe × 2 horizons × daily forecasts gated to 20% coverage at launch — see Appendix G
**Constitutional weight:** P0 — without this, Sentinel remains a data platform; with it, Sentinel becomes a research platform

---

## Table of contents

0. [Executive summary](#0-executive-summary)
1. [Architectural vision and prerequisites](#1-architectural-vision-and-prerequisites)
2. [Glossary and terminology](#2-glossary-and-terminology)
3. [Sentinel module catalog — integration points](#3-sentinel-module-catalog--integration-points)
4. [Constitutional invariants](#4-constitutional-invariants)
5. [Phase 0 — Preflight and environment readiness](#5-phase-0--preflight-and-environment-readiness-1-day)
6. [Phase 1 — Harvest discipline](#6-phase-1--harvest-discipline-12-days)
7. [Phase 2 — Four-stage decomposition core](#7-phase-2--four-stage-decomposition-core-34-days)
8. [Phase 3 — Domain adapters per asset class](#8-phase-3--domain-adapters-per-asset-class-45-days)
9. [Phase 4 — Reasoning trace and evidence chain](#9-phase-4--reasoning-trace-and-evidence-chain-integration-2-days)
10. [Phase 5 — Post-cutoff evaluation as substrate gate](#10-phase-5--post-cutoff-evaluation-as-substrate-gate-23-days)
11. [Phase 6 — Council of Investors integration path](#11-phase-6--council-of-investors-integration-path-1-day-deferred-work)
12. [Phase 7 — Cost gates and quality controls](#12-phase-7--cost-gates-and-quality-controls-12-days)
13. [Phase 8 — End-to-end validation](#13-phase-8--end-to-end-validation-23-days)
14. [Prompt template specifications per stage](#14-prompt-template-specifications-per-stage)
15. [Data contracts — full Pydantic schemas](#15-data-contracts--full-pydantic-schemas)
16. [Observability, telemetry, and dashboards](#16-observability-telemetry-and-dashboards)
17. [Security, privacy, and compliance](#17-security-privacy-and-compliance)
18. [Operational runbooks](#18-operational-runbooks)
19. [Rollout and launch plan](#19-rollout-and-launch-plan)
20. [Versioning and backwards compatibility](#20-versioning-and-backwards-compatibility)
21. [Timeline and roadmap with absolute dates](#21-timeline-and-roadmap-with-absolute-dates)
22. [Comprehensive command surface map](#22-comprehensive-command-surface-map)
23. [Configuration and storage surfaces](#23-configuration-and-storage-surfaces)
24. [Consolidated stop conditions](#24-consolidated-stop-conditions)
25. [Verification artifacts per phase](#25-verification-artifacts-per-phase)
26. [Risks and mitigations](#26-risks-and-mitigations)
27. [Open questions and decisions log](#27-open-questions-and-decisions-log)
28. [FAQ](#28-faq)
29. [References](#29-references)
30. [What this enables](#30-what-this-enables)
- [Appendix A — Worked reasoning trace](#appendix-a--worked-reasoning-trace-spy-30d-as-of-2026-04-01)
- [Appendix B — Worked forecast output JSON](#appendix-b--worked-forecast-output-json)
- [Appendix C — Sample post-cutoff test set construction](#appendix-c--sample-post-cutoff-test-set-construction)
- [Appendix D — Prompt evaluation rubric](#appendix-d--prompt-evaluation-rubric)
- [Appendix E — Capability test reference card](#appendix-e--capability-test-reference-card)
- [Appendix F — Internal Sentinel cross-references](#appendix-f--internal-sentinel-cross-references)
- [Appendix G — Instrument universe roster](#appendix-g--instrument-universe-roster)

---

## 0. Executive summary

### The deliverable

A native, four-stage agentic forecasting capability inside Sentinel that produces three artifacts as a single unit of output:

1. **Numerical forecast** across five asset classes — equities, FX, fixed income, commodities, crypto — at horizons from 1 day to 1 year.
2. **Structured reasoning trace** stored in the SoulSeal evidence chain, queryable by instrument, time range, confidence, dominant signal type, and semantic similarity.
3. **Post-cutoff out-of-sample evaluation evidence** that meets Sentinel's substrate-gate honesty standard — no claimed quality without test-set proof against data the underlying LLM could not have memorized.

The architectural pattern is harvested from the Nexus paper (Anthropic, 2025). The implementation is 100% native Sentinel composition over modules that already exist (`sma/`, `sai/`, `sfe/`, `spm/`). No upstream code, no upstream dependencies, no fine-tuning, no RL — pure inference-time orchestration.

### Why this and why now

Sentinel already has the components needed for forecasting:

| Component class | Existing Sentinel modules | Maps to Nexus stage |
|-----------------|----------------------------|---------------------|
| Macro signals | `sma/global_macro_v3`, `sma/cftc_cot_v3`, `sma/inflation_vix_analytics`, `sma/central_bank_nlp_v3` | Macro-Temporal Isolator |
| Numerical patterns | `sfe/historical_pit_v3`, `sfe/pit_integrity_v3`, technical screeners, vectorbt vol estimators | Micro-Temporal Isolator |
| Contextual integration | `sma/news_sentiment_pipeline_v3`, `sma/social_sentiment_v3`, `sma/economic_calendar_v3`, `sai/financial_rag_v3`, `sai/earnings_rag_v3` | Contextual Integrator |
| Synthesis substrate | `sai/research_agent_v3`, `sai/nl_strategy_generator_v3`, `sai/factor_research_v3` | Synthesis Agent |
| Evaluation rigor | PBO/DSR overfitting detection (Wave 34), walk-forward (existing), source lineage tracking | Post-cutoff substrate gate |
| Storage substrate | SoulSeal evidence chain, Postgres + pgvector, JSONL audit logs | Reasoning trace persistence |

What is missing is the **orchestration** that composes these into a coherent forecasting capability with structured handoffs and auditable reasoning. This PRD is the orchestration.

### The competitive bet

| Capability axis | Bloomberg / FactSet / AlphaSense | Sentinel after this PRD |
|------------------|----------------------------------|--------------------------|
| Data depth | $25K-$50K/yr seat | Free + open source |
| LLM copilot | Bolted on top of data | Native four-stage agentic core |
| Reasoning trace | None public | First-class evidence-chained artifact |
| Post-cutoff honesty | Not claimed | Substrate gate; required for claim |
| Council of Investors | Not on roadmap | Direct input to Council agents |
| Operator scale | Enterprise team | Solo operator on a Mac mini |

Bloomberg-class **data** is commoditizing — SEC EDGAR, FRED, OpenFIGI, CCXT, NautilusTrader, FinBERT + LlamaIndex + pgvector all replace what the incumbents charge for. **Auditable reasoning over data** is not commoditized. This PRD is what makes Sentinel a research platform in the way that matters: structured, traced, post-cutoff-validated forecasts at solo-operator cost.

### Phase summary

| Phase | Window | Eng days | Output |
|-------|--------|----------|--------|
| 0. Preflight | 2026-05-19 | 1 | Prerequisites confirmed, interfaces frozen |
| 1. Harvest discipline | 2026-05-20 → 2026-05-21 | 2 | Five harvest notes in `docs/harvest-notes/nexus/` |
| 2. Four-stage core | 2026-05-22, 05-25 → 05-26 | 3 | `nexus_forecaster_v3.py` + four stages + handoff dataclasses + e2e test |
| 3. Five adapters | 2026-05-27 → 2026-05-29 | 3 | Equities, FX, fixed income, commodities, crypto adapters with capability tests |
| 4. Trace + chain | 2026-06-01 → 2026-06-02 | 2 | `ReasoningTrace` schema, SoulSeal integration, pgvector indexing |
| 5. Post-cutoff eval | 2026-06-03 → 2026-06-05 | 3 | Test set builder, evaluator, per-adapter capability tests |
| 6. Council contract | 2026-06-06 (Sat) | 1 | `council_contract.py`, advisor consumption pattern |
| 7. Cost gates | 2026-06-07 (Sun) → 06-08 | 2 | Cost tracker, budget gates, quality controls |
| 8. E2E validation | 2026-06-09 → 2026-06-11 | 3 | Full validation suite, evidence files, honest assessment |
| Buffer | 2026-06-12 → 2026-06-14 | 0 (calendar buffer; 3 days) | Slack, rework, ship gate review |
| Ship | 2026-06-15 | 1 | Tag `nexus-v1.0`, merge to `main`, post-launch dashboards live |

**Total:** 21 engineering days (20 across Phase 0–8 + 1 ship day) inside a 28-day calendar window with 3 buffer days. Phase 6 and Phase 7 day-1 are deliberately scheduled on the weekend of 06-06/06-07 to land Phase 8 on a clean three-day weekday block; if weekend work is rejected, slide Phase 6 to Mon 06-08 and Phase 7 to Tue 06-09, compressing or eating into buffer.

### Success criteria (decided up front, measured at Phase 8)

| Criterion | Threshold | Source of truth |
|-----------|-----------|-----------------|
| Stage capability tests | 4/4 pass | `pytest sentinel/sai/nexus/stages/tests/` |
| Adapter post-cutoff tests | ≥ 4/5 pass | `sentinel forecast eval --all` |
| Decomposition beats monolithic baseline | ≥ 3/5 asset classes | Phase 8 head-to-head |
| Trace integrity | 100% (no broken chain entries on 1000 sampled forecasts) | `sentinel forecast trace verify --all` |
| Operational cost | ≤ $300/month at steady state | `sentinel forecast cost --month` |
| Strategy-lab integration | ≥ 1 end-to-end forecast → strategy → backtest example with PBO/DSR intact | Phase 8.4 evidence file |

**Acceptable failure mode:** post-cutoff evaluation shows monolithic prompting equals or beats decomposition (risk R1). Surface it honestly; the discipline of the test is the deliverable, not the result. The system that knows when it can't help is more valuable than one that pretends.

---

## 1. Architectural vision and prerequisites

### Section ↔ Phase numbering

Sections in this PRD are 1-indexed top-to-bottom. Build phases are 0-indexed (Phase 0 = preflight). The mapping is:

| Section | Phase | Window | Eng days |
|---------|-------|--------|----------|
| 5 | Phase 0 — Preflight | 2026-05-19 | 1 |
| 6 | Phase 1 — Harvest discipline | 2026-05-20 → 21 | 2 |
| 7 | Phase 2 — Four-stage core | 2026-05-22, 25, 26 | 3 |
| 8 | Phase 3 — Adapters | 2026-05-27 → 29 | 3 |
| 9 | Phase 4 — Trace + evidence chain | 2026-06-01 → 02 | 2 |
| 10 | Phase 5 — Post-cutoff eval | 2026-06-03 → 05 | 3 |
| 11 | Phase 6 — Council contract | 2026-06-06 | 1 |
| 12 | Phase 7 — Cost gates | 2026-06-07 → 08 | 2 |
| 13 | Phase 8 — E2E validation | 2026-06-09 → 11 | 3 |

`Section N = Phase (N − 5)` for `N ∈ [5, 13]`. Sections outside that range (1–4, 14–30, appendices A–G) are cross-cutting reference material that applies to every phase.

### What this builds

A native forecasting capability in Sentinel that produces (a) numerical forecasts across asset classes (equities, FX, fixed income, commodities, crypto), (b) structured reasoning traces explaining the fundamental drivers behind each forecast, and (c) honest evaluation evidence proving the forecasts hold up post-cutoff out-of-sample. The capability composes existing Sentinel modules into a four-stage pipeline: macro-temporal isolator, micro-temporal isolator, contextual integrator, synthesis agent.

After this PRD ships, Sentinel can produce forecasts that are both numerically competitive and explanatorily auditable, fed into the meta-allocator (`spm/`) and consumed by Council of Investors agents when that layer lands.

### Why this fits Sentinel structurally

The Nexus paper formalizes what Sentinel's architecture was already converging toward. Sentinel has the components but not the orchestration. The mapping is shown in the Executive summary table. What is missing is the explicit four-stage decomposition with structured handoffs, the reasoning trace artifact, and the post-cutoff capability tests as substrate gates. This PRD builds those.

### Prerequisites that must be true when this begins

Sentinel must be at the state achieved by the most recent push (Wave 35 / commit `dd8799e`):

- 24 dims at 9 (math-verified)
- 73 dims at 8
- 1 dim at 7 (Alpaca 2-year ceiling)
- All 98 capability tests passing
- Data lineage tracking in place per the `source_lineage` discussion

The Council of Investors layer does **not** need to be built first. This PRD's output (the Nexus forecaster + reasoning traces) is what the Council eventually consumes. Build forecasting infrastructure first, council layer on top later.

The DanteForge outcome-derived scoring should be available so Sentinel can declare T4+ outcomes that depend on post-cutoff evaluation evidence. If outcome-derived scoring is not yet live in Sentinel, Phase 5 of this PRD can still ship but the outcomes are stored as capability tests in the existing pattern and migrated to outcomes when the scoring system updates.

---

## 2. Glossary and terminology

Define once, reference everywhere.

### Metrics

| Term | Definition | Used where |
|------|------------|------------|
| **MASE** | Mean Absolute Scaled Error. `mean(|actual − forecast|) / mean(|actual_t − actual_{t-1}|)`. Compares forecast error to naive (random walk) baseline. < 1.0 means the model beats the naive baseline. **Edge case:** when the denominator (mean absolute first-difference of the actual series) is zero — degenerate constant series — the implementation returns `MASE = NaN` and the test point is dropped from aggregate metrics with a `mase_undefined` flag in the per-point log. The eval reporter refuses to publish an aggregate MASE if > 5% of points are dropped. | Phase 5, 8 |
| **sMAPE** | Symmetric Mean Absolute Percentage Error. `mean(2·|actual − forecast| / (|actual| + |forecast|))`. Scale-independent. Bounded [0, 2]. Lower is better. **Edge case:** when both `actual` and `forecast` are zero, the per-point contribution is defined as 0 (perfect agreement), not NaN. Implementation in `sentinel/sai/nexus/eval/metrics.py::smape()`. | Phase 5, 8 |
| **Directional accuracy** | Fraction of forecasts where sign of forecast change matches sign of actual change. Beating 0.50 means doing better than a coin flip. | Phase 5, 8 |
| **Calibration** | When the model says 70% confidence, are 70% of those predictions correct? Measured by Brier score or reliability diagram. | Phase 5 |
| **Coverage** | Fraction of true outcomes that fall within the forecast's stated confidence interval. For an 80% CI, coverage should be ≈ 0.80. | Phase 5 |
| **PBO** | Probability of Backtest Overfitting. From Bailey, Borwein, López de Prado (2014). Measures the chance that the in-sample-best strategy will underperform out-of-sample. Gate: ≤ 0.50. | Phase 8 (Wave 34 infra) |
| **DSR** | Deflated Sharpe Ratio. Adjusts Sharpe for multiple testing. From López de Prado. Gate: > 0 means likely real after adjustment. | Phase 8 (Wave 34 infra) |
| **Brier score** | Mean squared error of probabilistic forecasts. Lower is better. Decomposable into reliability + resolution − uncertainty. | Phase 5 |

### Architectural terms

| Term | Definition |
|------|------------|
| **SoulSeal** | Sentinel's cryptographic hash-chain artifact store. Each entry references the previous entry's hash; tampering breaks the chain. Used for evidence-chain integrity. |
| **Source lineage tier** | A classification of every data input: T1 (public domain — SEC EDGAR, FRED, public government data), T2 (commercial-with-attribution — vendor APIs with permissive ToS), T3 (ToS-restricted — scraped or licensed-internal-only), T4 (synthetic — derived locally and stamped). |
| **Capability test** | A bash/python script in `.danteforge/capability-tests/` that exits 0 only when a claimed capability empirically holds. The substrate honesty gate. |
| **Substrate gate** | A capability test that must pass before downstream consumers may rely on the capability. Failure marks the capability `degraded` and consumers are warned. |
| **Handoff invariants** | Constraints on the dataclasses passed between stages — no NaN, confidence ∈ [0, 1], `source_lineage` non-empty, etc. Enforced by Pydantic validators. |
| **Conservative cutoff** | An LLM's training cutoff *minus* a 60-day safety buffer to account for residual leakage through web crawls, RAG corpora, vendor caches. |
| **Post-cutoff data** | Data whose timestamp is strictly after the conservative cutoff. The only data on which an LLM-based forecaster can be honestly evaluated. |
| **Forecast horizon** | The lookahead window for the prediction. Standard set: `1d`, `5d`, `30d`, `90d`, `1y`. Encoded as `ForecastHorizon` enum. |
| **Forecast result** | The numerical + distributional output of the synthesis agent. Always paired 1:1 with a `ReasoningTrace`. |
| **Reasoning trace** | A structured, auditable record of which signals were dominant, how stages disagreed, what was considered and rejected, with cost and provenance metadata. First-class artifact. |
| **Adapter** | An asset-class-specific subclass of the four stages that customizes feature engineering and synthesis emphasis without modifying the orchestrator. |
| **Monolithic baseline** | A single LLM call with all relevant context dumped in at once, asked to forecast directly. The pre-Nexus way. The thing decomposition must beat. |

### Sentinel module-naming conventions

| Prefix | Meaning | Example |
|--------|---------|---------|
| `sma/` | Sentinel Market Analytics — macro, sentiment, calendar, central-bank | `sma/global_macro_v3` |
| `sfe/` | Sentinel Feature Engineering — quantitative features, point-in-time, vol estimators | `sfe/historical_pit_v3` |
| `sai/` | Sentinel AI — LLM-powered modules, RAG, research agents | `sai/research_agent_v3` |
| `spm/` | Sentinel Portfolio Management — allocation, risk, meta-allocator | `spm/meta_allocator_v3` |
| `_vN` suffix | Module API version. Breaking changes bump the suffix. Old version remains for one release cycle. | `_v3` |

---

## 3. Sentinel module catalog — integration points

Every Nexus stage consumes existing modules. This section enumerates the concrete touch points so Phase 2 implementation can wire directly without re-discovery.

### Data flow

```text
Caller (CLI / MCP / Strategy Lab / Council Advisor)
   |
   |  forecast(instrument, horizon, as_of, context)
   v
+-------------------------------------------------------+
|                NexusForecaster (orchestrator)         |
|         sentinel/sai/nexus/nexus_forecaster_v3.py     |
+----+---------------------+-----------------------+----+
     |                     |                       |
     v                     v                       v
+-----------+        +-----------+         +---------------+
|   Macro   |        |   Micro   |         |  Contextual   |
| Isolator  |        | Isolator  |         |  Integrator   |
+-----+-----+        +-----+-----+         +-------+-------+
      |                    |                       |
      v                    v                       v
[MacroSignal v1.0]   [MicroSignal v1.0]    [ContextualSignal v1.0]
      \                    |                       /
       \                   v                      /
        +---------> Synthesis Agent <------------+
                    sai/research_agent_v3
                    via sai/llm_router
                    (Claude Opus 4.7 default)
                            |
                            v
                  ForecastResult + ReasoningTrace
                  (reproducibility manifest stamp)
                            |
            +---------------+----------------+
            v               v                v
       SoulSeal chain   Postgres +       Caller response
       (jsonl + hash)   pgvector index   (small JSON; trace
       Section 9        Section 23       lazy-loadable)
```

Read across the diagram: the three stages run in parallel (no inter-dependencies between Macro / Micro / Contextual), and only the Synthesis stage joins them. The orchestrator enforces handoff invariants (Section 15) between every arrow.

Stage-to-module mapping follows.

### Macro-Temporal Isolator consumes

| Module path | What it provides | Stage 1 use |
|-------------|------------------|-------------|
| `sentinel/sma/global_macro_v3.py` | GDP nowcasts, PMI, unemployment, inflation prints | Regime detection inputs |
| `sentinel/sma/cftc_cot_v3.py` | Commitment-of-traders positioning by instrument | Speculator/commercial positioning signal |
| `sentinel/sma/inflation_vix_analytics.py` | Breakevens, real yields, VIX term structure | Inflation regime / vol regime |
| `sentinel/sma/central_bank_nlp_v3.py` | Hawkish-dovish score per central bank, speech parsing | Policy stance signal |
| `sentinel/sma/regime_detector_hmm.py` (new in this PRD) | HMM-based regime classifier wrapping `hmmlearn` | Discrete regime label |
| `sentinel/sma/yield_curve_v3.py` | Treasury curve shape, slope, butterfly | Curve regime |

### Micro-Temporal Isolator consumes

| Module path | What it provides | Stage 2 use |
|-------------|------------------|-------------|
| `sentinel/sfe/historical_pit_v3.py` | Point-in-time-correct OHLCV | Lookback windows without leakage |
| `sentinel/sfe/pit_integrity_v3.py` | Asserts no future bars leak into past windows | Handoff guard |
| `sentinel/sfe/vol_estimators_v3.py` | Garman-Klass, Rogers-Satchell, Yang-Zhang (Wave 35) | Realized vol panel |
| `sentinel/sfe/technical_screeners.py` | RSI, MACD, Bollinger, momentum factor library | Momentum / mean-reversion signals |
| `sentinel/sfe/microstructure_v3.py` (new) | Bid-ask proxy from OHLC, Amihud illiquidity | Liquidity state |
| `sentinel/sfe/vectorbt_runner.py` | Vectorized backtest runtime | Recent return distribution snapshots |

### Contextual Integrator consumes

| Module path | What it provides | Stage 3 use |
|-------------|------------------|-------------|
| `sentinel/sma/news_sentiment_pipeline_v3.py` | News article sentiment, entity linkage, freshness | News aggregate |
| `sentinel/sma/social_sentiment_v3.py` | Reddit, X / Twitter, StockTwits aggregates | Crowd sentiment |
| `sentinel/sma/economic_calendar_v3.py` | Forward calendar of releases | Pending event flags |
| `sentinel/sma/central_bank_nlp_v3.py` | Latest FOMC, ECB, BoE communications | Policy stance |
| `sentinel/sai/financial_rag_v3.py` | RAG over 10-K, 10-Q, S-1 filings | SEC filing context |
| `sentinel/sai/earnings_rag_v3.py` | RAG over earnings transcripts | Earnings-call context |
| `sentinel/sai/ma_intelligence_v3.py` | M&A pipeline tracker (Wave 34) | Corporate action context |

### Synthesis Agent consumes

| Module path | What it provides | Stage 4 use |
|-------------|------------------|-------------|
| `sentinel/sai/research_agent_v3.py` | Multi-step LLM reasoning scaffold | Synthesis runtime |
| `sentinel/sai/nl_strategy_generator_v3.py` | Natural-language strategy formulation | Optional downstream feed |
| `sentinel/sai/factor_research_v3.py` | Factor exposure context | Cross-section context |
| `sentinel/sai/llm_router.py` | Routes calls to Opus 4.7 / Sonnet 4.6 / Haiku 4.5 | Model selection |
| `sentinel/sai/prompt_cache_v3.py` | Anthropic prompt cache integration | Cost reduction |

### Cross-cutting

| Module path | What it provides | Used by |
|-------------|------------------|---------|
| `sentinel/core/soulseal_chain.py` | Hash-chain artifact writer | Phase 4 trace storage |
| `sentinel/core/source_lineage.py` | Lineage tier tagging utilities | Every stage |
| `sentinel/core/cost_tracker.py` (extended in Phase 7) | Token accounting | Synthesis agent + cost gates |
| `sentinel/core/staleness_guard.py` (new) | Asserts inputs not older than configured windows | Stage entry points |
| `sentinel/core/audit_log.py` | JSONL append-only audit log | Cost, capability tests, degradations |

If a module listed above does **not** exist or is older than the referenced `_v3` API, that is a Phase 0 blocker — see `Section 5`.

---

## 4. Constitutional invariants

These hold throughout the PRD. Stop and report if any would be violated.

**I1. No external paper-derived dependencies without sovereignty audit.** Nexus is not on PyPI as far as we know, and shouldn't be installed even if it were. The paper is teacher, not vendor. The four-stage decomposition pattern is harvested as architectural insight, not as code.

**I2. Existing Sentinel modules are composed, not rebuilt.** Every component the four-stage pipeline needs already exists in Sentinel (see Section 3). The Nexus harvest is orchestration work, not new construction. If a phase proposes building something that already exists, refactor to use the existing module instead.

**I3. Reasoning traces are first-class artifacts.** Every forecast produces a structured trace. Traces are stored in the SoulSeal evidence chain alongside the numerical prediction. A forecast without a trace is not a valid Nexus forecast and must not be consumed by downstream modules.

**I4. Post-cutoff evaluation is the only honest test.** Backtests on data the underlying LLM could have seen during training are data leakage. Every forecast model's capability test runs against data strictly post-cutoff. The cutoff date is the underlying LLM's training cutoff, conservatively estimated (cutoff − 60d buffer).

**I5. Forecast accuracy is not the same as trading profitability.** The PRD measures forecast quality (MASE, sMAPE, directional accuracy) separately from strategy returns (Sharpe, max drawdown, profitability after costs). Confusing the two is the canonical retail quant mistake. Sentinel reports both.

**I6. Cost gates apply.** A four-stage LLM pipeline is expensive. The system tracks token cost per forecast and refuses to run forecasts that exceed configured budgets. Cost-effectiveness is a first-class quality dimension.

**I7. Data lineage propagates.** Every input to every forecast carries its `source_lineage`. The reasoning trace records which lineage tier each input came from (T1 public, T2 commercial-with-attribution, T3 ToS-restricted, T4 synthetic). This is operationally necessary for any future commercialization path.

**I8. Decomposition must beat monolithic baseline, or be flagged.** Phase 8 runs a head-to-head between the four-stage Nexus pipeline and a single-shot monolithic prompt on identical post-cutoff test sets. If the four-stage approach does not win on at least 3 of 5 asset classes, the result is surfaced — not hidden — and the affected adapters are flagged `decomposition-no-lift`.

**I9. Reasoning must be grounded in signals, not generated freely.** The synthesis agent receives structured `MacroSignal`, `MicroSignal`, `ContextualSignal` dataclasses with explicit numerical fields. The prompt template forbids it from invoking drivers absent from those dataclasses. A trace that names a driver not present in the signal inputs fails validation.

**I10. Confidence must move under uncertainty.** Stage capability tests assert that confidence scores drop during known regime transitions (e.g., March 2020 onset, October 2008 crisis). A stage that returns constant high confidence regardless of market conditions fails the calibration test and may not ship.

**I11. The system must know when it cannot help.** A forecast generated with stale inputs, insufficient context, or under-cutoff data must be refused — not produced with a quiet quality reduction. Refusal is a first-class output (`ForecastResult.refused: bool` + `refusal_reason`).

**I12. No backwards-compat shims for Council layer.** The Council of Investors does not exist yet. Phase 6 defines the consumption contract once and stamps it as v1. When Council lands, contract v2 is permitted but no shims, deprecation warnings, or transitional types may pollute the v1 surface.

---

## 5. Phase 0 — Preflight and environment readiness (1 day)

A short discipline phase that catches "we thought we had X but X is `_v2`, not `_v3`" before any real work starts. Half a day on a clean repo; a full day if discrepancies surface.

### Goal

Prove that every prerequisite from Sections 1 and 3 is true *today*, not "as of Wave 35 according to the changelog." Freeze the public interface surface for downstream phases.

### Checklist

| # | Check | Pass condition | If fails |
|---|-------|----------------|----------|
| 0.1 | Repo state | `git status` clean, on `main`, at `dd8799e` or newer | Rebase / clean before proceeding |
| 0.2 | Test suite | `make check` green; all 98 capability tests pass | Halt; resolve regression before forge |
| 0.3 | Module availability | Each module in Section 3 imports without error | Halt; investigate missing or renamed module |
| 0.4 | LLM router live | `python -c "from sentinel.sai.llm_router import route; print(route('opus'))"` returns a client | Halt; restore router |
| 0.5 | SoulSeal chain healthy | `python -m sentinel.core.soulseal_chain verify` returns intact | Halt; do not write new entries until chain repaired |
| 0.6 | Source-lineage registry exists | `sentinel/core/source_lineage.py` defines `T1..T4` | Halt; resolve lineage module before stage 1 |
| 0.7 | Postgres + pgvector reachable | `make db-check` returns connection + `vector` extension | Halt; fix DB before Phase 4 |
| 0.8 | API key budget configured | `.danteforge/config/nexus-cost.json` present with daily / monthly caps | Author file from Phase 7 template before Phase 2 |
| 0.9 | Conservative LLM-cutoff registry | `sentinel/sai/nexus/eval/llm_cutoffs.json` exists with at least Opus 4.7, Sonnet 4.6, Haiku 4.5 entries | Author registry before Phase 5 |
| 0.10 | Interface freeze | Public surface of `nexus_forecaster_v3` documented in this PRD (Section 15) and merged into a stub module | Stub author commit before Phase 2 starts |

### Phase 0 deliverables

- `docs/harvest-notes/nexus/00-preflight.md` — checklist outcome, evidence, any waivers
- `sentinel/sai/nexus/__init__.py` — stub package, importable, exposing the public surface (functions raise `NotImplementedError` for now)
- `sentinel/sai/nexus/eval/llm_cutoffs.json` — populated with conservative cutoffs
- A green CI run on the stub branch

### Phase 0 first-day command sequence

For an operator starting cold on 2026-05-19, the exact command sequence:

```bash
# 1. Confirm we are on a clean main at the expected commit
git fetch origin && git checkout main && git pull --ff-only
git log -1 --pretty='%H %s'                    # expect ≥ dd8799e

# 2. Bring up the substrate (TimescaleDB + Postgres + pgvector)
make docker-up
make db-migrate
make db-check                                  # asserts pgvector extension live

# 3. Run the full pre-existing test suite (must be green before adding anything)
make check                                     # all 98 capability tests must pass

# 4. Confirm every Section 3 module imports
python - <<'PY'
import importlib
mods = [
  "sentinel.sma.global_macro_v3", "sentinel.sma.cftc_cot_v3",
  "sentinel.sma.inflation_vix_analytics", "sentinel.sma.central_bank_nlp_v3",
  "sentinel.sma.yield_curve_v3",
  "sentinel.sfe.historical_pit_v3", "sentinel.sfe.pit_integrity_v3",
  "sentinel.sfe.vol_estimators_v3", "sentinel.sfe.technical_screeners",
  "sentinel.sma.news_sentiment_pipeline_v3", "sentinel.sma.social_sentiment_v3",
  "sentinel.sma.economic_calendar_v3",
  "sentinel.sai.financial_rag_v3", "sentinel.sai.earnings_rag_v3",
  "sentinel.sai.research_agent_v3", "sentinel.sai.nl_strategy_generator_v3",
  "sentinel.sai.factor_research_v3", "sentinel.sai.llm_router",
  "sentinel.core.soulseal_chain", "sentinel.core.source_lineage",
  "sentinel.core.cost_tracker", "sentinel.core.audit_log",
]
missing = []
for m in mods:
    try: importlib.import_module(m)
    except Exception as e: missing.append((m, str(e)))
if missing:
    for m, e in missing: print(f"MISSING: {m}: {e}")
    raise SystemExit(1)
print("OK: all 22 prerequisite modules importable")
PY

# 5. Confirm SoulSeal chain integrity
python -m sentinel.core.soulseal_chain verify

# 6. Confirm Anthropic key is reachable and budget configured
python -c "from sentinel.sai.llm_router import route; c = route('opus'); print('opus client:', type(c).__name__)"
test -f .danteforge/config/nexus-cost.json    # if missing, author from Section 12 template

# 7. Confirm conservative-cutoff registry exists
test -f sentinel/sai/nexus/eval/llm_cutoffs.json || echo "AUTHOR from Section 10 template"

# 8. Create the stub package and merge a green PR
mkdir -p sentinel/sai/nexus/{stages,handoff,trace,prompts,cost,eval,adapters,integration,tests}
touch sentinel/sai/nexus/__init__.py
# author NotImplementedError stubs per Section 7 module structure
make check                                     # still green with stub added
git checkout -b nexus/phase-0-preflight
git add -A && git commit -m "Nexus Phase 0: preflight checklist + stub package"
git push -u origin nexus/phase-0-preflight
gh pr create --title "Nexus Phase 0 — Preflight" --body "see Docs/NexusAgenticForecasting.md §5"
```

If any of steps 1–7 fail, **do not proceed to Phase 1.** Halt, file a runbook entry, and resolve the blocker.

The author of `sentinel/sai/nexus/__init__.py` stub MUST expose exactly the public surface defined in Section 15 — every dataclass importable, every public function present but raising `NotImplementedError`. The CI on the Phase 0 PR asserts importability via `python -c "import sentinel.sai.nexus"` and the public-surface match via a small reflection test in `sentinel/sai/nexus/tests/test_public_surface.py`.

### Stop conditions

- Any check 0.1–0.7 fails → halt entire PRD; resolve before Phase 1
- Check 0.8–0.10 unmet → author the missing artifact within the same Phase 0 window; do not push the failure into Phase 2

---

## 6. Phase 1 — Harvest discipline (1–2 days)

### Goal

Read the Nexus paper carefully and produce harvest notes that capture the architectural insights without copying code or text. The harvest notes are the artifact of learning — they prove the patterns were understood, not just pattern-matched.

### Reading targets

Read the paper in full. For each of the following architectural elements, produce a harvest note in `docs/harvest-notes/nexus/<element>.md`:

#### 1.1 The four-stage decomposition (`decomposition-pipeline.md`)

- What does macro-temporal isolation specifically do? What signals does it produce?
- What does micro-temporal isolation specifically do? What signals does it produce?
- How does contextual integration weight unstructured information against numerical signals?
- What does the synthesis stage actually output beyond the final forecast?
- What handoff format passes information between stages?

#### 1.2 The reasoning trace format (`reasoning-trace-shape.md`)

- What structure does a reasoning trace take?
- Which decisions are explicit (recorded in the trace) vs implicit (made silently)?
- How are conflicting signals between stages represented?
- What makes a reasoning trace auditable vs decorative?

#### 1.3 Post-cutoff evaluation methodology (`post-cutoff-evaluation.md`)

- How did they construct test data strictly after LLM training cutoffs?
- What metrics did they use (MASE, sMAPE, directional accuracy, calibration)?
- How did they validate that data leakage didn't occur?
- What was the structure of their Zillow and equities testbed?

#### 1.4 The "stronger intrinsic forecasting" insight (`intrinsic-forecasting-claim.md`)

- What evidence did they present that frontier LLMs are better forecasters than typically credited?
- What specifically about agentic decomposition unlocked this capability?
- What was the monolithic-prompting baseline that decomposition beat?
- What did weak decomposition look like vs strong decomposition?

#### 1.5 What NOT to harvest (`product-decisions-not-to-copy.md`)

- The specific prompt templates they used
- The specific LLM they evaluated against (we use Claude/Opus)
- Their specific Zillow + equities testbed (we use FX, options, multi-asset)
- Their naming conventions for stages
- Any specific evaluator agent role-play (we use structured dataclasses, not character agents)

The harvest notes describe what was understood, the trade-offs, and how Sentinel will implement natively. **Do not copy paper text verbatim.** Cite the paper for attribution but write in our own words.

### Stop conditions for Phase 1

- A harvest note can't be written because the paper's description is too vague → stop, document what's unclear, decide whether the gap matters
- A harvest target turns out to require capability Sentinel doesn't have → stop, document the prerequisite gap, decide whether to build the prerequisite first or proceed without that pattern
- The harvest reveals the paper's methodology depends on training-time access (fine-tuning, RL) → re-scope to what's reproducible at inference time only

---

## 7. Phase 2 — Four-stage decomposition core (3–4 days)

### Goal

Build `sentinel/sai/nexus/nexus_forecaster_v3.py` as the core orchestration module. It composes the four stages explicitly, manages handoffs between stages, and produces both a numerical forecast and a reasoning trace artifact. This is the engine. Domain adapters in Phase 3 specialize it per asset class.

### Architectural shape

```python
class NexusForecaster:
    """Four-stage agentic forecaster following the Nexus pattern.

    Decomposes forecasting into specialized stages with explicit handoffs:
      1. macro_isolator: long-horizon trend, seasonality, regime
      2. micro_isolator: short-horizon dynamics, recent volatility, momentum
      3. contextual_integrator: news, events, sentiment, central bank signals
      4. synthesis_agent: combines above into forecast + reasoning trace
    """

    def forecast(
        self,
        instrument: Instrument,
        horizon: ForecastHorizon,
        as_of: datetime,
        context: ForecastContext,
    ) -> ForecastResult:
        macro_signal = self.macro_isolator.analyze(instrument, as_of, horizon)
        micro_signal = self.micro_isolator.analyze(instrument, as_of, horizon)
        contextual_signal = self.contextual_integrator.analyze(
            instrument, as_of, horizon, context,
        )
        result = self.synthesis_agent.synthesize(
            instrument, horizon, as_of,
            macro_signal, micro_signal, contextual_signal,
        )
        return result
```

### Module structure

```
sentinel/sai/nexus/
├── __init__.py
├── nexus_forecaster_v3.py        # NexusForecaster (orchestrator)
├── stages/
│   ├── __init__.py
│   ├── macro_isolator.py         # MacroTemporalIsolator
│   ├── micro_isolator.py         # MicroTemporalIsolator
│   ├── contextual_integrator.py  # ContextualIntegrator
│   └── synthesis_agent.py        # SynthesisAgent
├── handoff/
│   ├── __init__.py
│   ├── macro_signal.py           # MacroSignal dataclass
│   ├── micro_signal.py           # MicroSignal dataclass
│   ├── contextual_signal.py      # ContextualSignal dataclass
│   └── forecast_result.py        # ForecastResult + ReasoningTrace
├── trace/
│   ├── __init__.py
│   ├── reasoning_trace.py        # ReasoningTrace schema
│   └── trace_writer.py           # Writes traces to evidence chain
├── prompts/
│   ├── __init__.py
│   ├── macro_prompt.py           # MacroIsolator prompt builder
│   ├── micro_prompt.py
│   ├── contextual_prompt.py
│   └── synthesis_prompt.py
├── cost/
│   ├── __init__.py
│   └── cost_tracker.py           # Token + wall-clock accounting
├── eval/
│   ├── __init__.py
│   ├── llm_cutoffs.json
│   ├── test_set_builder.py
│   ├── post_cutoff_evaluator.py
│   └── run_capability_test.py
├── adapters/
│   ├── __init__.py
│   ├── equities.py
│   ├── fx.py
│   ├── fixed_income.py
│   ├── commodities.py
│   └── crypto.py
├── integration/
│   ├── __init__.py
│   ├── council_contract.py
│   └── advisor_consumes_forecast.py
└── tests/
    ├── test_macro_isolator.py
    ├── test_micro_isolator.py
    ├── test_contextual_integrator.py
    ├── test_synthesis_agent.py
    ├── test_handoff_invariants.py
    └── test_nexus_forecaster_e2e.py
```

### Stage 1 — Macro Temporal Isolator (`macro_isolator.py`)

Composes `sma/global_macro_v3`, `sma/cftc_cot_v3`, `sma/inflation_vix_analytics`, regime detection (HMM via `hmmlearn` wrapped in `sma/regime_detector_hmm.py`).

**Inputs:** `instrument`, `as_of` date, `horizon`

**Outputs (`MacroSignal` dataclass):**

- `regime`: detected market regime (e.g., risk-on, risk-off, stagflation, deflation)
- `trend_direction`: long-term trend (up, down, sideways) with strength score
- `seasonality`: detected seasonal patterns relevant to horizon
- `cycle_position`: where in the cycle we are (early, mid, late, recession)
- `macro_correlations`: which macro factors most influence this instrument currently
- `confidence`: self-assessed confidence in the macro read
- `source_lineage`: which data sources informed this signal

**Capability test:** `test_macro_isolator.py`

- Asserts known regime detection on historical data (e.g., 2008 Q4 → risk-off, 2020 Q1 → risk-off, 2021 Q2 → risk-on)
- Asserts `source_lineage` is populated for every output
- Asserts confidence scores are calibrated (high-confidence signals on stable regimes, low-confidence on transitions)

### Stage 2 — Micro Temporal Isolator (`micro_isolator.py`)

Composes technical screeners, volatility estimators (`sfe/vol_estimators_v3` — Garman-Klass, Rogers-Satchell, Yang-Zhang from Wave 35), momentum indicators, recent return distributions.

**Inputs:** `instrument`, `as_of` date, `horizon`

**Outputs (`MicroSignal` dataclass):**

- `recent_volatility`: realized vol estimates across multiple windows
- `volatility_regime`: low/normal/elevated/extreme
- `momentum`: short-horizon directional momentum with strength score
- `mean_reversion_signal`: whether the asset looks stretched
- `liquidity_state`: bid-ask spread proxy, volume relative to ADV
- `microstructure_anomalies`: detected unusual patterns
- `confidence`: self-assessed confidence
- `source_lineage`: data sources

**Capability test:** `test_micro_isolator.py`

- Asserts vol regime classification matches known periods (e.g., Feb 2018 → elevated, March 2020 → extreme)
- Asserts momentum direction matches known runs
- Asserts confidence drops during regime transitions

### Stage 3 — Contextual Integrator (`contextual_integrator.py`)

Composes `sma/news_sentiment_pipeline_v3`, `sma/social_sentiment_v3`, `sma/economic_calendar_v3`, `sma/central_bank_nlp_v3`, `sai/financial_rag_v3`, `sai/earnings_rag_v3`.

**Inputs:** `instrument`, `as_of` date, `horizon`, `ForecastContext` (which may include user-specified event focus)

**Outputs (`ContextualSignal` dataclass):**

- `news_summary`: structured summary of relevant recent news
- `news_sentiment`: aggregated sentiment with source attribution
- `pending_events`: upcoming scheduled events (earnings, Fed meetings, data releases)
- `central_bank_stance`: latest central bank communication parsed
- `relevant_filings`: SEC filings touching this instrument or sector
- `narrative_themes`: identified themes (e.g., "AI capex acceleration", "regional bank stress")
- `event_risk`: flagged risks in the horizon window
- `confidence`: self-assessed confidence in contextual completeness
- `source_lineage`: critical here — track which sources contributed each insight

**Capability test:** `test_contextual_integrator.py`

- Asserts narrative themes for known periods match obvious themes (e.g., March 2020 → pandemic, late 2022 → inflation/Fed)
- Asserts pending events are correctly extracted from economic calendar
- Asserts `source_lineage` covers every piece of contextual information

### Stage 4 — Synthesis Agent (`synthesis_agent.py`)

Composes `sai/research_agent_v3`, `sai/nl_strategy_generator_v3`, Claude (Opus 4.7 by default, Sonnet 4.6 under budget pressure, Haiku 4.5 for fast batch) routed via `sai/llm_router.py`.

**Inputs:** `instrument`, `horizon`, `as_of`, `MacroSignal`, `MicroSignal`, `ContextualSignal`

**Outputs (`ForecastResult` dataclass):**

- `prediction`: numerical forecast (price, return, distribution, scenario set depending on instrument type)
- `prediction_distribution`: full distribution where appropriate, not just point estimate
- `direction_probability`: directional probability (`P(up)`, `P(down)`, `P(sideways)`)
- `confidence_interval`: explicit uncertainty bounds
- `dominant_signals`: which of macro/micro/contextual most drove the conclusion
- `signal_disagreement`: where the three stages disagreed and how it was resolved
- `reasoning_trace`: full `ReasoningTrace` artifact (separate large object)
- `cost_metadata`: tokens consumed, wall-clock time, model used
- `evidence_chain_id`: SoulSeal artifact identifier for this forecast
- `refused`: bool — true when the system declines to forecast (stale data, insufficient context, budget exhausted)
- `refusal_reason`: structured enum when `refused == True`

**Capability test:** `test_synthesis_agent.py`

- Asserts forecast distributions are coherent (probabilities sum, confidence intervals contain median)
- Asserts reasoning trace references all three input signals
- Asserts `signal_disagreement` is populated when stages disagree
- Asserts cost metadata is recorded
- Asserts grounding (I9): every driver named in the trace appears in at least one of the three input signal dataclasses

### End-to-end capability test

`test_nexus_forecaster_e2e.py` runs the full pipeline on a known historical period and asserts:

- All four stages produced output
- Handoff invariants held (no NaN signals passed downstream, all confidence scores in [0,1])
- `ReasoningTrace` was stored in evidence chain
- Cost metadata was recorded
- Total wall-clock under configured budget
- `refused == False` when all inputs healthy; `refused == True` when stale-data fixture is injected

### Stop conditions for Phase 2

- Stage requires data source Sentinel doesn't have → halt that stage, document gap
- Handoff invariants can't be enforced cleanly → refactor handoff shape
- Capability tests for any stage cannot be satisfied → halt, document why

---

## 8. Phase 3 — Domain adapters per asset class (4–5 days)

### Goal

The four-stage decomposition is general; the specific signals and synthesis logic differ across asset classes. Build adapters that specialize the forecaster for each asset class Sentinel cares about. Each adapter is thin — it customizes the stages but reuses the orchestration.

### Adapters in order of priority

#### 3.1 Equities adapter (`sentinel/sai/nexus/adapters/equities.py`)

- Macro stage emphasizes: sector rotation, factor exposures, earnings season position
- Micro stage emphasizes: relative strength, sector momentum, vol surface
- Contextual stage emphasizes: earnings calls, SEC filings, analyst revisions
- Synthesis output specifies: target price, target return, factor decomposition

#### 3.2 FX adapter (`sentinel/sai/nexus/adapters/fx.py`)

- Macro stage emphasizes: rate differentials, central bank policy divergence, current account, real exchange rates
- Micro stage emphasizes: carry-vol relationship, momentum, term structure of vol
- Contextual stage emphasizes: central bank speak, geopolitical events, flow data
- Synthesis output specifies: direction, range, regime-conditional scenarios

#### 3.3 Fixed income adapter (`sentinel/sai/nexus/adapters/fixed_income.py`)

- Macro stage emphasizes: yield curve shape, credit spreads, Fed reaction function
- Micro stage emphasizes: liquidity premia, supply/demand technicals
- Contextual stage emphasizes: Treasury auction calendar, FOMC communication, credit events
- Synthesis output specifies: yield direction, curve shape forecast, spread direction

#### 3.4 Commodities adapter (`sentinel/sai/nexus/adapters/commodities.py`)

- Macro stage emphasizes: supply/demand fundamentals, inventory cycles, dollar strength
- Micro stage emphasizes: term structure (contango/backwardation), positioning (CFTC COT)
- Contextual stage emphasizes: weather, geopolitical supply disruption, industrial demand signals
- Synthesis output specifies: direction, term structure forecast, scenario set

#### 3.5 Crypto adapter (`sentinel/sai/nexus/adapters/crypto.py`)

- Macro stage emphasizes: on-chain flows, exchange balances, network metrics
- Micro stage emphasizes: leverage levels, funding rates, derivatives positioning
- Contextual stage emphasizes: regulatory developments, protocol upgrades, narrative themes
- Synthesis output specifies: direction, volatility forecast, regime scenarios

### Implementation pattern per adapter

```python
class EquitiesNexusAdapter(NexusForecaster):
    """Equities-specific Nexus forecaster."""

    def __init__(self, config: EquitiesNexusConfig):
        super().__init__(
            macro_isolator=EquitiesMacroIsolator(config),
            micro_isolator=EquitiesMicroIsolator(config),
            contextual_integrator=EquitiesContextualIntegrator(config),
            synthesis_agent=EquitiesSynthesisAgent(config),
        )
```

Each adapter has its own capability tests asserting domain-specific signal quality. Each adapter's stage subclass overrides the generic version with asset-specific feature engineering. **Adapters may not modify the orchestrator** — Invariant I12 enforcement.

### Adapter output guidance schema

The synthesis prompt (Section 14) references `adapter_output_guidance`. Every adapter ships a guidance object that the orchestrator injects into the synthesis prompt's cacheable system block.

```python
class AdapterOutputGuidance(BaseModel):
    schema_version: Literal["nexus-adapter-guidance/1.0"] = "nexus-adapter-guidance/1.0"
    asset_class: Literal["equity", "fx", "fixed_income", "commodity", "crypto"]
    adapter_version: str

    # Prediction shaping
    sideways_threshold_pct: float = Field(ge=0.0)   # |return| below this counts as "sideways"
    distribution_form_default: Literal["normal", "skew_normal", "mixture"]
    horizon_specific_priors: Dict[str, Dict[str, float]]  # e.g., {"5d": {"vol_floor": 0.05}, ...}

    # Weighting prior
    nexus_weights_prior: NexusStageWeights        # adapter's macro/micro/contextual lean
    weight_deviation_max: float = Field(ge=0.0, le=1.0)  # how far synthesis may deviate from prior

    # Domain emphasis
    must_consult_signal_fields: List[str]         # e.g., for FX: ["macro_signal.macro_correlations.rate_differential"]
    asset_specific_drivers: List[str]             # canonical driver names allowed in this adapter
    output_fields_required: List[str]             # e.g., equities require "target_price"

    # Refusal sensitivity
    confidence_floor_override: Optional[float] = None    # adapter can tighten the floor below 0.4
    disagreement_threshold_override: Optional[float] = None
```

Example for equities:

```json
{
  "schema_version": "nexus-adapter-guidance/1.0",
  "asset_class": "equity",
  "adapter_version": "v3.0.1",
  "sideways_threshold_pct": 0.005,
  "distribution_form_default": "skew_normal",
  "horizon_specific_priors": {
    "5d": {"vol_floor": 0.05, "vol_ceiling": 0.80},
    "30d": {"vol_floor": 0.08, "vol_ceiling": 1.20}
  },
  "nexus_weights_prior": {"macro": 0.30, "micro": 0.30, "contextual": 0.40},
  "weight_deviation_max": 0.25,
  "must_consult_signal_fields": [
    "contextual_signal.relevant_filings",
    "macro_signal.macro_correlations"
  ],
  "asset_specific_drivers": [
    "sector_rotation", "earnings_revision", "factor_exposure",
    "relative_strength", "vol_surface", "analyst_revision",
    "regime_risk_on_off", "macro_correlations"
  ],
  "output_fields_required": ["target_price", "target_return", "factor_decomposition"],
  "confidence_floor_override": null,
  "disagreement_threshold_override": null
}
```

The orchestrator injects the guidance object as a structured JSON block in the synthesis system prompt. The guidance is included in the cache breakpoints (Section 16), so changing a guidance value triggers a cache miss for that adapter — explicitly and intentionally.

### Per-adapter capability tests (Phase 3 acceptance)

| Adapter | Key historical assertion |
|---------|---------------------------|
| Equities | `as_of=2023-03-13` (SVB collapse) → regional bank exposure flagged in macro signal; financials underperformance assigned >0.6 probability |
| FX | `as_of=2022-09-26` (GBP mini-budget) → GBP/USD short flagged with high confidence; pending BoE intervention noted |
| Fixed income | `as_of=2023-10-19` (10y at 5%) → curve steepening forecast with auction-supply driver named in trace |
| Commodities | `as_of=2022-02-24` (Russia/Ukraine) → WTI bullish with geopolitical-supply driver named |
| Crypto | `as_of=2022-11-07` (FTX week) → BTC bearish with exchange-balance and contagion drivers named |

These are fixtures, not predictions — they assert that with the *as-of-date data only*, the adapter would have flagged the right drivers. This is consistency with historical reasoning, not prescience.

### Stop conditions for Phase 3

- Adapter requires data source Sentinel doesn't have → log gap, build adapter with available data, document limitations
- Asset class has fundamentally different forecasting structure that doesn't fit four-stage decomposition (e.g., binary event prediction) → halt that adapter, document why
- Adapter's capability tests can't be satisfied with current data freshness → log, defer

---

## 9. Phase 4 — Reasoning trace and evidence chain integration (2 days)

### Goal

Reasoning traces are not just documentation. They are first-class artifacts that:

- Get stored in SoulSeal evidence chain alongside the numerical prediction
- Are queryable (you can ask "what forecasts had high confidence in October 2025?")
- Feed Council of Investors agents when that layer arrives
- Provide regulatory auditability if Sentinel ever serves regulated buyers
- Enable retrospective analysis ("which signal types historically were dominant in correct forecasts?")

### Trace schema

```python
@dataclass
class ReasoningTrace:
    forecast_id: str                            # Links to ForecastResult
    instrument: Instrument
    as_of: datetime
    horizon: ForecastHorizon

    # Stage outputs preserved
    macro_signal: MacroSignal
    micro_signal: MicroSignal
    contextual_signal: ContextualSignal

    # Synthesis reasoning
    dominant_signal_type: Literal['macro', 'micro', 'contextual', 'balanced']
    signal_weights: Dict[str, float]            # how synthesis weighted each input

    # Reasoning narrative
    summary: str                                # 2-3 sentence summary of the call
    fundamental_drivers: List[Driver]           # explicit drivers with citations
    contradicting_evidence: List[Evidence]      # what we considered but dismissed
    key_uncertainties: List[Uncertainty]        # what could change this view

    # Auditability
    inputs_used: List[InputReference]           # every data source consulted
    inputs_excluded: List[InputReference]       # sources available but not used + why
    source_lineage_summary: Dict[str, int]      # count per lineage tier

    # Cost and provenance
    model_used: str
    tokens_consumed: int
    wall_clock_seconds: float
    soulseal_artifact_id: str
    generated_at: datetime

    # Validation
    grounding_check_passed: bool                # I9: every driver references a signal field
    schema_version: str                         # e.g., 'nexus-trace/1.0'
```

### Storage and indexing

**Storage:** traces are written to `data/forecasts/<year>/<month>/<forecast-id>.json` and registered in the evidence chain with the SoulSeal pattern (cryptographic hash chain). The numerical prediction is small; the trace is large; both are stored together but the trace can be lazy-loaded.

**Indexing:** a Postgres table `forecast_traces_index` (with pgvector for semantic search over summaries) provides fast queries:

- By instrument
- By time range
- By confidence level
- By dominant signal type
- By semantic similarity (find forecasts with similar reasoning)

### Capability tests for trace integration

`test_trace_evidence_chain.py`:

- Asserts every forecast produces a corresponding evidence chain entry
- Asserts trace can be retrieved from chain by `forecast_id`
- Asserts trace integrity (hash chain unbroken)

`test_trace_queryability.py`:

- Asserts queries by instrument return matching traces
- Asserts semantic search over summaries returns relevant traces
- Asserts confidence-based filtering works

`test_trace_grounding.py`:

- Asserts `grounding_check_passed == True` for every shipped trace
- Asserts every `Driver.citation` resolves to a field in one of the three signal dataclasses
- Asserts traces that fail grounding are written to a `quarantine/` directory, not the main evidence chain

### Stop conditions for Phase 4

- Reasoning traces are too large to store sustainably → reduce trace verbosity, retain critical fields
- Evidence chain integrity breaks under load → halt integration, fix chain
- Grounding check (I9) fails on > 5% of generated traces → halt; tighten synthesis prompt template

---

## 10. Phase 5 — Post-cutoff evaluation as substrate gate (2–3 days)

### Goal

Post-cutoff out-of-sample evaluation is the only honest test of a forecasting model that uses an LLM. Build evaluation infrastructure that conservatively estimates the underlying LLM's training cutoff, constructs test sets strictly after that cutoff, and produces evidence files that satisfy substrate capability tests.

This is the most important phase for substrate honesty. Without it, the system can self-deceive about forecast quality just like the matrix system was self-deceiving about capability quality before the substrate fix.

### Cutoff estimation

Conservative cutoff per model — each LLM has an estimated training cutoff. The system maintains a registry at `sentinel/sai/nexus/eval/llm_cutoffs.json` with conservative estimates (earlier than the official cutoff to allow for data leak through web cache, RAG corpora, etc.).

```json
{
  "schema_version": "1.0",
  "default_buffer_days": 60,
  "models": {
    "claude-opus-4-7": {
      "official_cutoff": "2026-01-31",
      "conservative_cutoff": "2025-12-01",
      "buffer_days": 60,
      "evaluation_window_start": "2026-01-30"
    },
    "claude-sonnet-4-6": {
      "official_cutoff": "2025-01-31",
      "conservative_cutoff": "2024-12-01",
      "buffer_days": 60,
      "evaluation_window_start": "2025-01-30"
    },
    "claude-haiku-4-5-20251001": {
      "official_cutoff": "2025-07-01",
      "conservative_cutoff": "2025-05-02",
      "buffer_days": 60,
      "evaluation_window_start": "2025-07-01"
    }
  }
}
```

For evaluation, the test set start date must be at least 60 days after the conservative cutoff. This buffer accounts for residual data leakage through web crawls.

### Test set construction

`sentinel/sai/nexus/eval/test_set_builder.py` constructs evaluation test sets:

- Pull historical data strictly after `cutoff_date + 60 days`
- For each instrument/horizon combination, generate N test points (default 50)
- Each test point: `(instrument, as_of_date, horizon, ground_truth_outcome)`
- Store test sets at `data/forecasts/eval/<model>/<asset-class>/<test-set-id>.json`
- Stratify test points across regimes (don't accidentally test only during calm periods)

### Evaluation runner

`sentinel/sai/nexus/eval/post_cutoff_evaluator.py` runs forecasts against test sets:

- For each test point, run the Nexus forecaster `as_of` that point's date
- Record the prediction, the reasoning trace, the cost
- Compare prediction to `ground_truth_outcome`
- Compute metrics across the test set:
  - **MASE** — accuracy relative to naive baseline
  - **sMAPE** — scale-independent accuracy
  - **Directional accuracy** — fraction of forecasts with correct direction
  - **Calibration** — when model says 70% confidence, are 70% correct?
  - **Coverage** — fraction of true outcomes within forecast confidence intervals
  - **Brier score** — for the probabilistic outputs

### Capability tests as substrate gates

Per dimension in Sentinel's matrix, add capability tests of the form:

```bash
# .danteforge/capability-tests/equities_forecasting.sh
#!/usr/bin/env bash
set -e
python -m sentinel.sai.nexus.eval.run_capability_test \
  --asset-class equities \
  --horizon 5d \
  --min-mase 0.85 \
  --min-directional-accuracy 0.55 \
  --min-calibration 0.8 \
  --post-cutoff-only
```

The test exits 0 only if the post-cutoff evaluation produces results meeting all thresholds. If the LLM has been updated (new cutoff), the test set is regenerated against the new cutoff date and re-run.

### Per-asset-class evaluation outcomes

Each adapter (equities, FX, fixed income, commodities, crypto) gets its own capability test with thresholds calibrated to that asset class's difficulty:

| Asset class | Directional accuracy | MASE | Calibration | Notes |
|-------------|----------------------|------|-------------|-------|
| Equities (5d) | > 0.55 | < 1.00 | > 0.80 | Reasonable on liquid SP500 names |
| Equities (30d) | > 0.55 | < 0.95 | > 0.80 | Longer horizon, slightly easier MASE |
| FX majors (5d) | > 0.53 | < 1.10 | > 0.75 | FX is structurally harder |
| Fixed income (30d) | > 0.58 | < 0.95 | > 0.80 | Yield-direction-specific |
| Commodities (30d) | > 0.55 | < 1.00 | > 0.75 | Term-structure-aware |
| Crypto (5d) | > 0.52 | < 1.20 | > 0.70 | High vol; loosest bar |

These thresholds beat naive baselines but are not so high as to be unachievable.

### Stop conditions for Phase 5

- Conservative cutoff estimate puts evaluation data start more than 90 days ago and we don't have enough post-cutoff data → halt that adapter's evaluation, document the data freshness gap
- Evaluation produces results worse than naive baseline → halt that adapter, don't ship forecasts for that asset class until model is improved
- Evaluation is too expensive to run regularly (each test set requires thousands of LLM calls) → reduce test set size to minimum statistical validity, sample rather than exhaustive

---

## 11. Phase 6 — Council of Investors integration path (1 day, deferred work)

### Goal

The Council of Investors layer will eventually consume Nexus forecasts as one input among many. This phase doesn't build the Council, but establishes the integration contract so the Council can land later without refactoring Nexus.

### Integration contract

Council agents receive `ForecastResult` + `ReasoningTrace`. They can:

- Disagree with Nexus's conclusion (Dalio-agent might reject a Nexus equity forecast because the macro stage missed a debt cycle signal)
- Reference specific stage outputs in their disagreement ("Nexus's contextual integrator weighted news sentiment heavily but central bank stance contradicts it")
- Add their own philosophical lens (Buffett-agent emphasizes quality + price, Druckenmiller-agent emphasizes macro divergence)
- Combine Nexus forecast with their own analysis to produce a recommendation

Council outputs feed the meta-allocator (`spm/` module) along with raw Nexus forecasts. The allocator decides risk budget allocation across the strategies advisors propose based on Nexus's forecasts.

### Per-advisor preferences for Nexus stages

Each advisor agent declares which Nexus stages they weight most heavily.

```python
class DalioAgent(CouncilAdvisor):
    nexus_weights = NexusStageWeights(
        macro=0.6,        # Dalio emphasizes macro regime above all
        micro=0.1,
        contextual=0.3,
    )

class BuffettAgent(CouncilAdvisor):
    nexus_weights = NexusStageWeights(
        macro=0.2,
        micro=0.1,
        contextual=0.7,   # Buffett weights fundamental context (filings, earnings)
    )

class SimonsAgent(CouncilAdvisor):
    nexus_weights = NexusStageWeights(
        macro=0.1,
        micro=0.8,        # Simons-style is mostly statistical micro
        contextual=0.1,
    )
```

### What to build now vs later

**Build now:**

- `sentinel/sai/nexus/integration/council_contract.py` — defines the integration interface
- `sentinel/sai/nexus/integration/advisor_consumes_forecast.py` — example consumer pattern
- `sentinel/sai/nexus/integration/__init__.py` — exports `CouncilAdvisor`, `NexusStageWeights`, `AdvisorRecommendation`

**Defer to Council work:**

- Actual advisor agent implementations
- Per-advisor corpus ingestion
- Multi-advisor synthesis layer

This phase ensures the Nexus forecaster's outputs are shaped correctly for the eventual Council, without requiring the Council to exist yet.

---

## 12. Phase 7 — Cost gates and quality controls (1–2 days)

### Goal

A four-stage LLM pipeline is expensive. Building it is the easy part; running it sustainably is the operational challenge. This phase adds the gates that keep the forecaster economically viable and quality-controlled.

### Cost tracking

`sentinel/sai/nexus/cost/cost_tracker.py`:

- Records tokens per stage per forecast
- Aggregates daily, weekly, monthly cost
- Cost per asset class
- Cost per horizon
- Tags forecasts that exceed configured cost ceilings

### Cost gates

`.danteforge/config/nexus-cost.json`:

```json
{
  "schema_version": "1.0",
  "max_tokens_per_forecast": {
    "equities_short_horizon": 50000,
    "equities_long_horizon": 100000,
    "fx_majors": 60000,
    "fixed_income": 80000,
    "commodities": 60000,
    "crypto": 70000
  },
  "daily_budget_tokens": 5000000,
  "monthly_budget_tokens": 100000000,
  "force_cheaper_model_when_budget_low": true,
  "cheaper_model_threshold_pct": 0.8,
  "monthly_usd_ceiling": 300,
  "alert_at_pct": [0.50, 0.75, 0.90, 0.95]
}
```

When daily budget is 80% consumed, the forecaster automatically downgrades to a cheaper model (Sonnet 4.6 instead of Opus 4.7; Haiku 4.5 if 95% consumed) and logs the downgrade in reasoning traces. When 100% consumed, new forecasts are refused with a clear error.

### Quality controls beyond cost

**Confidence floor:** forecasts with synthesis confidence below 0.4 are flagged for human review rather than shipped to consumers. This prevents the system from confidently shipping low-quality forecasts.

**Disagreement threshold:** when macro/micro/contextual disagree past a configured threshold, the synthesis output explicitly notes "high signal disagreement" and downstream consumers can choose to discount the forecast.

**Stale-data refusal:** if any required input is older than its configured staleness window, the relevant stage signals "stale context" and the forecast is marked accordingly.

`.danteforge/config/nexus-staleness.json`:

```json
{
  "schema_version": "1.0",
  "max_age": {
    "news": "24h",
    "social_sentiment": "12h",
    "macro_data": "14d",
    "central_bank_communication": "30d",
    "earnings_transcript": "90d",
    "sec_filings": "90d",
    "cftc_cot": "10d",
    "ohlcv": "1d"
  },
  "refuse_when_more_than_n_stale": 2,
  "warn_when_more_than_n_stale": 1
}
```

**Capability test freshness:** post-cutoff evaluations are re-run weekly. If a capability test fails after passing previously, the relevant adapter is marked `degraded` and downstream consumers are warned.

### Stop conditions for Phase 7

- Daily/monthly cost ceilings cannot be hit with quality intact → renegotiate ceilings or reduce forecast frequency, do not lower quality silently
- Confidence floors filter out too many forecasts → investigate root cause, don't lower floor to hide poor quality

---

## 13. Phase 8 — End-to-end validation (2–3 days)

### Goal

Validate the full system across all asset classes on real historical data, producing the evidence files that prove the substrate's claims about Nexus forecasting capability are honest.

### Validation suite

#### 8.1 Per-asset-class post-cutoff evaluation

Run the full evaluation suite (50 test points per asset class per horizon) on each adapter. Produce:

- Aggregate metric tables (MASE, sMAPE, directional accuracy, calibration, coverage)
- Comparison against naive baselines (random walk, AR(1), simple moving average)
- Comparison against monolithic-prompt baseline (single LLM call without decomposition) — **I8 enforcement point**
- Reasoning trace quality assessment (manual review of N random traces)

#### 8.1.1 Statistical methodology for decomposition vs monolithic (I8)

The head-to-head is run on **identical test points** (paired design) — same `(instrument, as_of, horizon)` tuples for both pipelines. This eliminates between-sample variance and makes the comparison a paired statistical test, which has much higher power than unpaired comparisons at the same N.

**Procedure per asset class:**

1. **Test set:** the post-cutoff test set for that adapter (50 points × 2 horizons = 100 paired observations).
2. **Predictions:** for each test point, generate both a Nexus-decomposed forecast and a monolithic-baseline forecast. Same `as_of`, same retrieved context, same model.
3. **Per-point loss:** compute the per-point loss for both pipelines on three metrics — MASE-contribution (|actual − forecast|), squared error, and directional-accuracy indicator (1 if direction correct else 0).
4. **Paired test — directional accuracy:**
   - Compute `n_wins_nexus` = count of points where Nexus is right and monolithic is wrong.
   - Compute `n_wins_monolithic` = count of the reverse.
   - Apply **McNemar's exact test** on the (n_wins_nexus, n_wins_monolithic) pair. Nexus is declared the winner at the 95% confidence level when `p < 0.05` AND `n_wins_nexus > n_wins_monolithic`.
5. **Paired test — continuous metric (MASE-contribution or squared error):**
   - Compute per-point Δ = monolithic_loss − nexus_loss (positive Δ = Nexus better).
   - Apply the **Diebold-Mariano test** with HAC variance correction (Newey-West, bandwidth = horizon-business-days).
   - Nexus is declared the winner at 95% confidence when DM statistic > 1.96 AND mean Δ > 0.
6. **Bootstrap robustness check:** resample paired test points 10 000 times with replacement; report the 95% bootstrap CI on mean Δ. The CI must not cross zero for the claim "Nexus beats monolithic" to be made.
7. **Effect size:** report Cohen's d on the paired differences. A statistically significant but trivial-magnitude win (d < 0.2) is reported honestly as "significant but small".

**Aggregation across asset classes (I8 verdict):**

| Asset class result | Counts toward |
|---------------------|---------------|
| Nexus wins (McNemar p<0.05 AND DM>1.96 AND bootstrap CI excludes 0) | "Nexus wins" |
| Monolithic wins (mirror conditions) | "Monolithic wins" |
| Indeterminate (neither side meets all three criteria) | "Indeterminate" |

I8 ship gate: **Nexus wins ≥ 3 of 5 asset classes.** Indeterminate counts as a loss for I8 purposes — silence is not victory.

**Reporting:** `data/forecasts/eval/decomposition_vs_monolithic/<run-id>/report.json` carries all per-asset-class statistics; `report.md` is the human-readable summary that goes into the model card.

**Honest-failure path:** if I8 fails on majority of asset classes, the failing adapters ship in `decomposition-no-lift` mode — they continue to run but the monolithic baseline serves as the live forecast and the decomposed pipeline runs in shadow for ongoing comparison. The failure does not block the PRD from shipping; it shapes what ships.

**Failure modes the test must catch:**

- Monolithic happens to win on a single calm regime → mitigated by regime-stratified test points (Section 10)
- Sample-size insufficiency masks a real but small effect → bootstrap CI width reported alongside point estimate
- Multiple-testing inflation across 5 asset classes → Bonferroni-corrected α = 0.05 / 5 = 0.01 used when computing per-class "wins" for the aggregate verdict
- Path-dependence in forecasts (one pipeline's output influences the next via context) → forbidden by I4 + manifest hashes (Section 15)

#### 8.2 Cost validation

Run a representative workload (50 forecasts across asset classes) and document:

- Total token consumption
- Per-forecast cost distribution
- Wall-clock distribution
- Cost vs quality tradeoff (does Opus produce meaningfully better forecasts than Sonnet at 5x cost?)

#### 8.3 Trace integrity validation

For 100 random forecasts:

- Verify trace stored correctly in evidence chain
- Verify trace can be retrieved
- Verify `source_lineage` is complete
- Verify trace summary semantically matches the prediction (manual review)
- Verify `grounding_check_passed == True` on all sampled traces

#### 8.4 Strategy lab integration

Verify that a forecast can flow into a strategy:

- Generate forecast for SPY 30-day horizon
- Pass to a simple strategy that buys when forecast direction is up with confidence > 0.7
- Run walk-forward backtest using the existing infrastructure
- Verify PBO/DSR gates from Wave 34 still operate
- Confirm cost model from earlier discussion still applies

This is the end-to-end test: forecast → strategy → backtest → quality gate. If this works, the Nexus integration has delivered usable trading intelligence, not just research artifacts.

#### 8.5 Council contract dry run

Without building actual advisor agents, instantiate a no-op `CouncilAdvisor` subclass that consumes a `ForecastResult` and emits a structured opinion. Assert:

- Contract surface is sufficient (advisor has access to all signal outputs, not just synthesis)
- `ReasoningTrace` can be reasoned over by an LLM via the public schema
- Multiple advisors can consume the same forecast without contention

### Success criteria

Across all asset classes:

- At least 4 of 5 adapters pass post-cutoff capability tests
- Aggregate metrics meet or beat naive baselines
- Decomposed approach beats monolithic baseline on at least 3 of 5 asset classes
- Trace integrity is 100% (no broken evidence chain entries)
- Strategy lab integration works end-to-end on at least one example
- Daily cost ceiling is operationally sustainable on personal-trading scale

If any of these fail, document honestly. A partial-success outcome is fine and tells you what to improve. A claimed full-success that the validation doesn't actually support is the failure mode to avoid.

### Stop conditions for Phase 8

- End-to-end validation produces results worse than monolithic baseline on majority of asset classes → halt, treat as failure, investigate why decomposition isn't helping
- Strategy lab integration breaks PBO/DSR gates → halt, restore gates before shipping

---

## 14. Prompt template specifications per stage

Full-text templates for the prompts each stage uses. These live in `sentinel/sai/nexus/prompts/` as code (diffable, hashable, version-pinned via the reproducibility manifest in Section 15). The full text appears here so the harvest discipline (I1 — no copied paper prompts) is verifiable from the PRD alone.

Templates use Jinja2-style `{{ ... }}` placeholders. All placeholders are filled from structured Pydantic objects, never from free-form caller input.

### Common prompt skeleton

Every stage's prompt has the same five-part structure:

1. **Role frame** — what the stage is, what it is not. Anti-rationalization clauses ("name no driver that is not in the signal inputs").
2. **Input block** — structured dataclass dumped as JSON, with field-by-field annotations.
3. **Task** — the specific question being asked of the LLM at this stage.
4. **Output schema** — the exact JSON shape required. Pydantic-validatable.
5. **Refusal triggers** — explicit conditions under which the stage must emit a refusal instead of a forecast.

### Shared anti-rationalization clauses (used verbatim in every stage)

```text
GROUNDING RULES (apply to every output you produce):
(1) Name no driver, factor, theme, regime, or signal that is not present as a
    populated field in your input JSON. Do not synthesize information you wish
    you had.
(2) When in doubt, LOWER confidence. Do not inflate certainty to satisfy the
    schema. A confidence of 0.30 with honest reasoning is more valuable than
    0.80 with manufactured reasoning.
(3) If you would have to fabricate context to produce a valid output, emit
    {"refused": true, "refusal_reason": "<enum>"} instead, where <enum> is one
    of STALE_INPUTS, INSUFFICIENT_CONTEXT, LOW_STAGE_CONFIDENCE,
    PIT_INTEGRITY_TRIPPED, SCHEMA_VALIDATION_FAILED, BUDGET_EXHAUSTED, or
    SIGNAL_DISAGREEMENT_UNRESOLVABLE.
(4) Source-attribute every claim. Every entry in your output's source_lineage
    array must correspond to an input you actually consumed in this turn.
(5) Output ONLY the JSON object specified by the schema. No prose, no preamble,
    no explanation outside the JSON fields themselves.
```

### Macro stage prompt — full text

System prompt:

```text
You are the Macro-Temporal Isolator stage of the Sentinel Nexus forecasting
pipeline. Your job is to classify the macro regime and trend posture for one
instrument at one as-of date.

You consume only the structured macro indicator panel provided in the user
message. You do not consult news, social sentiment, technical patterns, or
any data outside the panel. Those belong to other stages.

You emit a JSON object matching the MacroSignal schema, version
nexus-macro-signal/1.0 (see Section 15 of the Nexus PRD).

{{ shared_grounding_rules }}
```

User prompt template:

```text
INSTRUMENT: {{ instrument_json }}
AS-OF DATE: {{ as_of_iso }}
FORECAST HORIZON: {{ horizon }}

MACRO INDICATOR PANEL (composed from sma/global_macro_v3 + sma/cftc_cot_v3 +
sma/inflation_vix_analytics + sma/central_bank_nlp_v3 + sma/yield_curve_v3):

{{ macro_panel_json }}

STALENESS AUDIT (any indicator past its nexus-staleness.json threshold
appears here; if any entry exists, consider STALE_INPUTS refusal):

{{ staleness_audit_json }}

TASK:
1. Classify regime ∈ {risk_on, risk_off, stagflation, deflation, transition,
   indeterminate}.
2. Identify trend_direction ∈ {up, down, sideways} and trend_strength ∈ [0, 1].
3. List seasonality patterns relevant to the {{ horizon }} window using only
   the panel's seasonality fields.
4. Classify cycle_position ∈ {early, mid, late, recession, indeterminate}
   using the panel's NBER-style indicators.
5. For each of the top-5 macro factors in the panel, score macro_correlations
   as a signed real in [-1, 1] reflecting how that factor currently moves the
   instrument's expected return at this horizon.
6. Score confidence ∈ [0, 1]. Drop confidence when (a) any indicator is in the
   staleness audit, (b) regime classifiers disagree across indicators, or
   (c) cycle_position resolves to "transition".
7. Populate source_lineage with every indicator you consulted, copying the
   tier field from the panel.

REFUSAL TRIGGERS specific to this stage:
- Any required indicator stale beyond its threshold → refuse with STALE_INPUTS.
- Instrument-to-factor mapping for {{ instrument.symbol }} not found in
  panel.macro_correlations_seed → refuse with INSUFFICIENT_CONTEXT.

OUTPUT: a single JSON object matching MacroSignal v1.0. Nothing else.
```

### Micro stage prompt — full text

System prompt:

```text
You are the Micro-Temporal Isolator stage of the Sentinel Nexus forecasting
pipeline. Your job is to summarize short-horizon dynamics for one instrument
at one as-of date.

You consume only the numerical micro feature panel provided in the user
message. You do not consult news, fundamentals, regime labels, or any
narrative input. Those belong to other stages.

You emit a JSON object matching the MicroSignal schema, version
nexus-micro-signal/1.0.

{{ shared_grounding_rules }}
```

User prompt template:

```text
INSTRUMENT: {{ instrument_json }}
AS-OF DATE: {{ as_of_iso }}
FORECAST HORIZON: {{ horizon }}

MICRO FEATURE PANEL (composed from sfe/vol_estimators_v3 +
sfe/technical_screeners + sfe/microstructure_v3 + sfe/historical_pit_v3,
PIT-integrity verified by sfe/pit_integrity_v3):

{{ micro_panel_json }}

TASK:
1. Populate recent_volatility with the panel's realized-vol estimates across
   the windows present (e.g., 7d_gk, 30d_yz, 90d_rs).
2. Classify volatility_regime ∈ {low, normal, elevated, extreme} using the
   panel's percentile-vs-3y-history field.
3. Score momentum ∈ [-1, 1] and momentum_strength ∈ [0, 1] from the panel's
   technical_screener fields (RSI, MACD, BB, momentum factor).
4. Score mean_reversion_signal ∈ [-1, 1] from the panel's
   distance_from_mean field.
5. Populate liquidity_state from the panel's bid_ask_proxy and amihud fields.
6. List microstructure_anomalies the panel flagged.
7. Score confidence ∈ [0, 1]. Drop confidence when the panel reports
   pit_integrity_warning=true, when realized vol jumps regime in the last
   {{ horizon }}, or when liquidity_state has any field in elevated range.
8. Populate source_lineage with every panel field you used.

REFUSAL TRIGGERS specific to this stage:
- panel.pit_integrity_warning == "TRIPPED" → refuse with PIT_INTEGRITY_TRIPPED.
- panel.history_bars < required_min_bars[horizon] → refuse with
  INSUFFICIENT_CONTEXT.

OUTPUT: a single JSON object matching MicroSignal v1.0. Nothing else.
```

### Contextual stage prompt — full text

System prompt:

```text
You are the Contextual Integrator stage of the Sentinel Nexus forecasting
pipeline. Your job is to summarize the narrative, event, and unstructured
context for one instrument at one as-of date.

You consume only the retrieved document set provided in the user message.
Every document carries a source, lineage tier (T1/T2/T3/T4), timestamp, and
relevance score. You attribute every claim to a source. You weight by recency
and source quality, NOT by verbosity. A 200-character T1 source outweighs a
2000-character T3 source on the same topic.

You emit a JSON object matching the ContextualSignal schema, version
nexus-contextual-signal/1.0.

{{ shared_grounding_rules }}
```

User prompt template:

```text
INSTRUMENT: {{ instrument_json }}
AS-OF DATE: {{ as_of_iso }}
FORECAST HORIZON: {{ horizon }}
USER EVENT FOCUS (optional): {{ context.user_event_focus or "none" }}

RETRIEVED DOCUMENTS (from sma/news_sentiment_pipeline_v3 +
sma/social_sentiment_v3 + sma/economic_calendar_v3 + sma/central_bank_nlp_v3 +
sai/financial_rag_v3 + sai/earnings_rag_v3 + sai/ma_intelligence_v3):

{{ documents_json }}

LICENSED-EXCERPT POLICY (Section 17 ToS rules):
- T1 sources: full quotation allowed in summary and inputs_used.excerpt.
- T2 sources: up to 280 characters per quoted span, with source attribution.
- T3 sources: NO verbatim quotation. May inform reasoning; must not appear in
  excerpts. Tag any claim derived from T3 as "lineage_tier=T3, paraphrased".
- T4 sources: locally-derived; full quotation allowed.

TASK:
1. Write news_summary: 3–5 sentences. Reflect only what the documents say,
   not what you "know" from training.
2. Aggregate news_sentiment: {score ∈ [-1, 1], volume = document count,
   top_sources = up to 5 InputReference entries by relevance × tier weight}.
3. List pending_events from the calendar documents whose scheduled_at falls
   inside {{ as_of_iso }} + {{ horizon }}.
4. Set central_bank_stance from the most recent central-bank document; null if
   none in the panel.
5. List relevant_filings: SEC documents touching this instrument or its
   sector during the lookback window.
6. List narrative_themes: cross-document themes (at most 5, ordered by
   prevalence × source-quality weight).
7. List event_risk: risks in the horizon window inferred from the documents
   only.
8. Score confidence ∈ [0, 1]. Drop confidence when (a) zero or one document
   covers the instrument, (b) documents older than the staleness threshold
   dominate the panel, (c) T3 sources dominate without T1/T2 corroboration on
   the same claim.
9. Populate source_lineage with every document you consumed.

REFUSAL TRIGGERS specific to this stage:
- documents_json is empty → refuse with INSUFFICIENT_CONTEXT.
- All documents past their freshness threshold → refuse with STALE_INPUTS.
- T3 sources are the only support for the dominant theme and no T1/T2 cross-
  confirmation exists → refuse with INSUFFICIENT_CONTEXT and detail the gap.

OUTPUT: a single JSON object matching ContextualSignal v1.0. Nothing else.
```

### Synthesis stage prompt — full text

This is the largest and most cost-sensitive prompt. It is the I9 enforcement point.

System prompt:

```text
You are the Synthesis Agent of the Sentinel Nexus forecasting pipeline. Your
job is to produce one numerical forecast and one reasoning trace from three
upstream structured signals.

You consume ONLY the three signal objects (MacroSignal, MicroSignal,
ContextualSignal) provided in the user message. You do not consult anything
else. Specifically you do not consult your training data for current events,
prices, or analyst opinions about {{ instrument.symbol }}.

Every "fundamental driver" you name must cite a specific JSON path into one
of the three signal objects (e.g., "macro_signal.regime",
"contextual_signal.pending_events[0]"). The trace_writer will assert that
every citation resolves to a populated field; unresolved citations FAIL
ingestion and your forecast will be quarantined.

You weight signals by stated confidence, NOT by length or specificity. A
MacroSignal with confidence=0.40 receives less weight than a MicroSignal with
confidence=0.80, regardless of which is more verbose.

You emit a JSON object matching the ForecastResult schema (which embeds a
ReasoningTrace), version nexus-forecast/1.0 and nexus-trace/1.0.

{{ shared_grounding_rules }}
```

User prompt template:

```text
INSTRUMENT: {{ instrument_json }}
AS-OF DATE: {{ as_of_iso }}
FORECAST HORIZON: {{ horizon }}

MACRO SIGNAL (from Stage 1):
{{ macro_signal_json }}

MICRO SIGNAL (from Stage 2):
{{ micro_signal_json }}

CONTEXTUAL SIGNAL (from Stage 3):
{{ contextual_signal_json }}

ADAPTER-SPECIFIC OUTPUT GUIDANCE ({{ instrument.asset_class }}):
{{ adapter_output_guidance }}

COST-MODE: {{ cost_mode }}   # one of "opus_full", "sonnet_downgrade", "haiku_downgrade"

TASK:
1. PRE-FLIGHT: confirm all three signals have refused=false and confidence
   ≥ 0.20. If any signal refused or confidence is below threshold, refuse
   this forecast with LOW_STAGE_CONFIDENCE and quote the offending signal.

2. WEIGHT: compute signal_weights as a normalized distribution over
   {macro, micro, contextual} derived from each signal's stated confidence,
   adjusted by the adapter's nexus_weights prior (provided in
   adapter_output_guidance). The weights must sum to 1.0 within ±0.01.

3. POINT FORECAST: emit prediction as the expected return for the horizon
   (decimal, e.g., 0.014 for +1.4%).

4. DISTRIBUTION: emit prediction_distribution with quantiles q05, q25, q50,
   q75, q95 over the horizon return. Where the synthesis is uncertain, the
   distribution should widen — do not narrow to claim precision you lack.

5. DIRECTIONAL PROBABILITY: emit direction_probability with P(up), P(down),
   P(sideways) summing to 1.0 ± 0.01. "Sideways" means |return| < adapter-
   specific threshold (provided in adapter_output_guidance).

6. CONFIDENCE INTERVAL: emit confidence_interval_lower/upper at the 80%
   level. This must equal the [q10, q90] of your distribution; do not
   contradict your own quantiles.

7. DOMINANT SIGNALS: list the 1–3 signal types whose weighted contribution
   exceeds the dominance threshold (0.33). If all three are roughly equal,
   the value is "balanced".

8. SIGNAL DISAGREEMENT: when stages point in opposite directions, write a
   1–3 sentence resolution describing WHY you weighted one over the others.
   Cite the specific signal fields involved. If disagreement is irresolvable
   (no signal dominates and they contradict), refuse with
   SIGNAL_DISAGREEMENT_UNRESOLVABLE.

9. REASONING TRACE: populate the embedded ReasoningTrace with:
   a. summary: 2–3 sentence call summary.
   b. fundamental_drivers: 2–5 Driver objects. Each Driver.citation MUST be
      a JSONPath into macro_signal, micro_signal, or contextual_signal that
      resolves to a populated field. Each Driver.citation_payload should
      include a short excerpt of the cited value.
   c. contradicting_evidence: 0–3 Evidence objects naming signals that
      argued against your conclusion plus your reason for dismissal.
   d. key_uncertainties: 1–3 Uncertainty objects naming what could flip your
      view, including the monitoring signal field path.
   e. inputs_used: union of all three signals' source_lineage arrays plus
      anything from the adapter_output_guidance you consumed.
   f. inputs_excluded: source_lineage entries available but deliberately not
      used, with a one-sentence rationale.
   g. source_lineage_summary: dict counting inputs by tier.

10. SELF-AUDIT: before emitting, verify (a) every Driver.citation resolves;
    (b) direction_probability sums to 1.0; (c) confidence_interval contains
    the q50 of your distribution; (d) prediction sign matches the dominant
    direction.

OUTPUT: a single JSON object matching ForecastResult v1.0 (with embedded
ReasoningTrace v1.0). Nothing else.
```

### Refusal cascade

Refusals propagate but never silently:

| Stage refuses with | Synthesis behavior |
|--------------------|---------------------|
| Macro `STALE_INPUTS` | Synthesis refuses parent forecast with `STALE_INPUTS` and quotes macro's `refusal_detail` |
| Micro `PIT_INTEGRITY_TRIPPED` | Synthesis refuses with `PIT_INTEGRITY_TRIPPED` (P0 — alerting fires per Section 18) |
| Contextual `INSUFFICIENT_CONTEXT` | Synthesis MAY still produce if macro+micro confidence both ≥ 0.7; the trace records `contextual_signal_unavailable=true` |
| Any stage `LOW_STAGE_CONFIDENCE` | Synthesis refuses with `LOW_STAGE_CONFIDENCE` |
| Two or more stages refuse | Synthesis refuses with the first-occurring refusal_reason and lists the others in `refusal_detail` |

The cascade rules live in `sentinel/sai/nexus/orchestrator/refusal_cascade.py`. They are tested in `test_refusal_cascade.py` against a fixture matrix of every refusal combination.

### Monolithic-baseline prompt (for I8 head-to-head)

Phase 8 includes a head-to-head against a single-shot prompt. That prompt is intentionally simple — the test is whether four-stage decomposition beats the lazy version, not whether decomposition beats a hand-tuned monolith. The baseline prompt template:

```text
You are a financial forecaster. Predict the {{ horizon }} return for
{{ instrument.symbol }} as of {{ as_of_iso }}.

Here is everything we know about the instrument and market right now:

{{ raw_concatenated_context }}  # all macro, micro, contextual sources
                                # concatenated, no structure imposed

Output JSON: {"return": <decimal>, "p_up": <0..1>, "p_down": <0..1>,
              "confidence": <0..1>, "rationale": "<1 paragraph>"}.
```

The monolithic baseline ships in `sentinel/sai/nexus/eval/monolithic_baseline.py` and is invoked by `nexus_vs_monolithic.sh` (Appendix E).

---

## 15. Data contracts — full Pydantic schemas

Single source of truth for the dataclasses passed between stages. All schemas live in `sentinel/sai/nexus/handoff/`. Versioned via `schema_version` field on every top-level object.

```python
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


# -------------------- Shared --------------------

class SourceLineageTier(str, Enum):
    T1_PUBLIC = "T1_PUBLIC"
    T2_COMMERCIAL_WITH_ATTRIBUTION = "T2_COMMERCIAL_WITH_ATTRIBUTION"
    T3_TOS_RESTRICTED = "T3_TOS_RESTRICTED"
    T4_SYNTHETIC_DERIVED = "T4_SYNTHETIC_DERIVED"


class InputReference(BaseModel):
    source_id: str                              # stable id of the source
    source_human_name: str                      # "FRED CPIAUCSL"
    lineage_tier: SourceLineageTier
    timestamp: datetime
    excerpt: Optional[str] = None               # short verbatim text where licensable
    url: Optional[str] = None


class Instrument(BaseModel):
    symbol: str
    asset_class: Literal["equity", "fx", "fixed_income", "commodity", "crypto"]
    venue: Optional[str] = None
    figi: Optional[str] = None
    cusip: Optional[str] = None
    isin: Optional[str] = None
    notes: Optional[str] = None


class ForecastHorizon(str, Enum):
    H_1D = "1d"
    H_5D = "5d"
    H_30D = "30d"
    H_90D = "90d"
    H_1Y = "1y"


class ForecastContext(BaseModel):
    user_event_focus: Optional[str] = None
    requested_model: Optional[str] = None       # override default routing
    requested_horizon: Optional[ForecastHorizon] = None
    purpose: Literal["research", "backtest", "live_strategy", "council_input"] = "research"


# -------------------- Macro signal --------------------

class MacroSignal(BaseModel):
    schema_version: Literal["nexus-macro-signal/1.0"] = "nexus-macro-signal/1.0"
    regime: Literal["risk_on", "risk_off", "stagflation", "deflation", "transition", "indeterminate"]
    trend_direction: Literal["up", "down", "sideways"]
    trend_strength: float = Field(ge=0.0, le=1.0)
    seasonality: List[str] = Field(default_factory=list)
    cycle_position: Literal["early", "mid", "late", "recession", "indeterminate"]
    macro_correlations: Dict[str, float]
    confidence: float = Field(ge=0.0, le=1.0)
    source_lineage: List[InputReference]

    @field_validator("source_lineage")
    @classmethod
    def must_have_lineage(cls, v):
        if not v:
            raise ValueError("MacroSignal.source_lineage may not be empty")
        return v


# -------------------- Micro signal --------------------

class MicroSignal(BaseModel):
    schema_version: Literal["nexus-micro-signal/1.0"] = "nexus-micro-signal/1.0"
    recent_volatility: Dict[str, float]            # {"7d_gk": 0.18, "30d_yz": 0.22, ...}
    volatility_regime: Literal["low", "normal", "elevated", "extreme"]
    momentum: float = Field(ge=-1.0, le=1.0)
    momentum_strength: float = Field(ge=0.0, le=1.0)
    mean_reversion_signal: float = Field(ge=-1.0, le=1.0)
    liquidity_state: Dict[str, float]              # {"bid_ask_proxy": 0.001, "amihud": 0.02}
    microstructure_anomalies: List[str]
    confidence: float = Field(ge=0.0, le=1.0)
    source_lineage: List[InputReference]


# -------------------- Contextual signal --------------------

class NewsSentimentAggregate(BaseModel):
    score: float = Field(ge=-1.0, le=1.0)
    volume: int
    top_sources: List[InputReference]


class CalendarEvent(BaseModel):
    event_id: str
    event_name: str
    scheduled_at: datetime
    expected_impact: Literal["low", "medium", "high"]
    source: InputReference


class ContextualSignal(BaseModel):
    schema_version: Literal["nexus-contextual-signal/1.0"] = "nexus-contextual-signal/1.0"
    news_summary: str
    news_sentiment: NewsSentimentAggregate
    pending_events: List[CalendarEvent]
    central_bank_stance: Optional[Literal["dovish", "neutral", "hawkish", "n/a"]] = None
    relevant_filings: List[InputReference]
    narrative_themes: List[str]
    event_risk: List[str]
    confidence: float = Field(ge=0.0, le=1.0)
    source_lineage: List[InputReference]


# -------------------- Reasoning trace pieces --------------------

class Driver(BaseModel):
    name: str
    direction: Literal["bullish", "bearish", "neutral"]
    strength: float = Field(ge=0.0, le=1.0)
    citation: str                                  # field path: e.g. "macro_signal.regime"
    citation_payload: Optional[str] = None         # short excerpt of cited value


class Evidence(BaseModel):
    summary: str
    direction_implied: Literal["bullish", "bearish", "neutral"]
    citation: str
    why_dismissed: str


class Uncertainty(BaseModel):
    description: str
    direction_if_resolved_against: Literal["bullish", "bearish", "neutral"]
    monitoring_signal: Optional[str] = None


# -------------------- Forecast result + trace --------------------

class CostMetadata(BaseModel):
    schema_version: Literal["nexus-cost/1.0"] = "nexus-cost/1.0"
    model_used: str
    tokens_input: int
    tokens_output: int
    tokens_cache_read: int = 0
    tokens_cache_write: int = 0
    wall_clock_seconds: float
    estimated_usd: float


class ReasoningTrace(BaseModel):
    schema_version: Literal["nexus-trace/1.0"] = "nexus-trace/1.0"
    forecast_id: str
    instrument: Instrument
    as_of: datetime
    horizon: ForecastHorizon

    macro_signal: MacroSignal
    micro_signal: MicroSignal
    contextual_signal: ContextualSignal

    dominant_signal_type: Literal["macro", "micro", "contextual", "balanced"]
    signal_weights: Dict[Literal["macro", "micro", "contextual"], float]

    summary: str
    fundamental_drivers: List[Driver]
    contradicting_evidence: List[Evidence]
    key_uncertainties: List[Uncertainty]

    inputs_used: List[InputReference]
    inputs_excluded: List[InputReference]
    source_lineage_summary: Dict[SourceLineageTier, int]

    cost_metadata: CostMetadata
    soulseal_artifact_id: str
    reproducibility_manifest_id: str            # Section 15 reproducibility manifest
    generated_at: datetime

    grounding_check_passed: bool


class RefusalReason(str, Enum):
    STALE_INPUTS = "STALE_INPUTS"
    INSUFFICIENT_CONTEXT = "INSUFFICIENT_CONTEXT"
    LOW_STAGE_CONFIDENCE = "LOW_STAGE_CONFIDENCE"
    SIGNAL_DISAGREEMENT_UNRESOLVABLE = "SIGNAL_DISAGREEMENT_UNRESOLVABLE"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    PIT_INTEGRITY_TRIPPED = "PIT_INTEGRITY_TRIPPED"
    SCHEMA_VALIDATION_FAILED = "SCHEMA_VALIDATION_FAILED"


class ForecastDistribution(BaseModel):
    quantiles: Dict[str, float]                    # {"q05": ..., "q50": ..., "q95": ...}
    parametric_form: Optional[Literal["normal", "skew_normal", "mixture"]] = None
    parameters: Optional[Dict[str, float]] = None


class ForecastResult(BaseModel):
    schema_version: Literal["nexus-forecast/1.0"] = "nexus-forecast/1.0"
    forecast_id: str
    instrument: Instrument
    as_of: datetime
    horizon: ForecastHorizon

    refused: bool = False
    refusal_reason: Optional[RefusalReason] = None
    refusal_detail: Optional[str] = None

    prediction: Optional[float] = None             # point estimate (return for the horizon)
    prediction_distribution: Optional[ForecastDistribution] = None
    direction_probability: Optional[Dict[Literal["up", "down", "sideways"], float]] = None
    confidence_interval_lower: Optional[float] = None
    confidence_interval_upper: Optional[float] = None
    confidence_interval_level: float = 0.80

    dominant_signals: List[Literal["macro", "micro", "contextual"]] = Field(default_factory=list)
    signal_disagreement: Optional[str] = None

    reasoning_trace_id: Optional[str] = None
    cost_metadata: CostMetadata

    @field_validator("direction_probability")
    @classmethod
    def probs_sum_close_to_one(cls, v):
        if v is None:
            return v
        s = sum(v.values())
        if not (0.99 <= s <= 1.01):
            raise ValueError(f"direction_probability must sum to 1.0; got {s}")
        return v
```

### Handoff invariants enforced by Pydantic

| Invariant | Enforcement |
|-----------|-------------|
| Confidence ∈ [0, 1] | `Field(ge=0.0, le=1.0)` on every confidence field |
| `source_lineage` non-empty | `field_validator` on `MacroSignal.source_lineage` (same on micro, contextual) |
| Direction probabilities sum to 1 | `field_validator` on `ForecastResult.direction_probability` |
| Schema versioning explicit | `Literal["nexus-*/1.0"]` enforces version pins |
| Refusal mutually exclusive with prediction | Cross-field validator (rejects `prediction != None` when `refused == True`) |
| Grounding check ties drivers to signal fields | `Driver.citation` must resolve via `jsonpath` against the trace; enforced in `trace_writer.py` |

### Reproducibility manifest

Every forecast is reproducible (input → output is bit-identical at temperature=0) given a complete manifest. The manifest is persisted at `data/forecasts/<year>/<month>/<forecast-id>.manifest.json` alongside the trace and is referenced by `ReasoningTrace.reproducibility_manifest_id`.

```python
class ReproducibilityManifest(BaseModel):
    schema_version: Literal["nexus-manifest/1.0"] = "nexus-manifest/1.0"
    manifest_id: str
    forecast_id: str
    pipeline_version: str                       # e.g. "nexus-v1.0"
    module_versions: Dict[str, str]             # {"nexus_forecaster": "v3.0.1", ...}
    config_hashes: Dict[str, str]               # sha256 per config file used
    prompt_template_hashes: Dict[str, str]      # sha256 per prompt template
    input_data_hashes: Dict[str, str]           # sha256 per signal panel input
    model_used: str
    model_inference_params: Dict[str, float]    # temperature, top_p, max_tokens, ...
    rng_seed: Optional[int] = None
    extended_thinking_budget: int = 0
    cache_breakpoints: List[str] = []           # cache_control anchor labels
    generated_at: datetime
```

Concrete example:

```json
{
  "schema_version": "nexus-manifest/1.0",
  "manifest_id": "manifest_2026-04-01_SPY_30d_a1b2c3",
  "forecast_id": "fcst_2026-04-01_SPY_30d_a1b2c3",
  "pipeline_version": "nexus-v1.0",
  "module_versions": {
    "nexus_forecaster": "v3.0.1",
    "macro_isolator": "v3.0.1",
    "micro_isolator": "v3.0.1",
    "contextual_integrator": "v3.0.1",
    "synthesis_agent": "v3.0.1"
  },
  "config_hashes": {
    "nexus-stages.json": "sha256:7c4a8d09ca3762af61e59520943dc26494f8941b",
    "nexus-cost.json": "sha256:b1946ac92492d2347c6235b4d2611184f01a4b67",
    "nexus-staleness.json": "sha256:da39a3ee5e6b4b0d3255bfef95601890afd80709",
    "nexus-adapters.json": "sha256:5d41402abc4b2a76b9719d911017c592fea4ce3a",
    "llm_cutoffs.json": "sha256:0a4d55a8d778e5022fab701977c5d840bbc486d0"
  },
  "prompt_template_hashes": {
    "macro_prompt": "sha256:8b1a9953c4611296a827abf8c47804d7",
    "micro_prompt": "sha256:e10adc3949ba59abbe56e057f20f883e",
    "contextual_prompt": "sha256:25d55ad283aa400af464c76d713c07ad",
    "synthesis_prompt": "sha256:5f4dcc3b5aa765d61d8327deb882cf99"
  },
  "input_data_hashes": {
    "macro_panel": "sha256:098f6bcd4621d373cade4e832627b4f6",
    "micro_features": "sha256:1f3870be274f6c49b3e31a0c6728957f",
    "contextual_documents": "sha256:b5d4045c3f466fa91fe2cc6abe79232a"
  },
  "model_used": "claude-opus-4-7",
  "model_inference_params": {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 50000
  },
  "rng_seed": null,
  "extended_thinking_budget": 16000,
  "cache_breakpoints": ["system", "schema_doc", "glossary"],
  "generated_at": "2026-04-01T00:01:32Z"
}
```

**Reproducibility guarantees:**

- At `temperature=0`, a re-run against the same manifest hashes must produce a bit-identical `ForecastResult` modulo `cost_metadata.wall_clock_seconds` and timestamps. Drift on any other field is a P0 incident (runbook 18.8).
- At `temperature > 0`, the run is non-deterministic; reproducibility falls back to "structurally equivalent" — same dominant signals, same direction, prediction within ±1σ. The capability test `nexus_reproducibility.sh` asserts this on a 20-forecast sample.
- Manifest hashes are computed in `sentinel/core/manifest_hasher.py` using sorted-keys JSON canonicalization.

**Audit replay:** `sentinel forecast replay <forecast-id>` reloads the manifest, asserts every hash still matches a known artifact, and re-executes the pipeline. The replay either reproduces the original `ForecastResult` (success) or surfaces the exact field that drifted (failure). Replay is the I9 grounding-check audit hammer.

---

## 16. Observability, telemetry, and dashboards

### Metrics emitted

Every forecast emits a structured event to `logs/forecasts/events-<date>.jsonl`. Metrics are pulled from the JSONL into Prometheus-compatible counters via a `make metrics` job that the existing Sentinel monitoring stack already runs.

| Metric name | Type | Labels | Purpose |
|-------------|------|--------|---------|
| `nexus_forecast_count_total` | counter | `asset_class`, `horizon`, `model` | Volume |
| `nexus_forecast_latency_seconds` | histogram | `asset_class`, `stage` | Per-stage latency budget |
| `nexus_forecast_tokens_total` | counter | `asset_class`, `stage`, `model` | Token spend |
| `nexus_forecast_refusal_count_total` | counter | `asset_class`, `refusal_reason` | Refusal monitoring |
| `nexus_forecast_confidence_avg` | gauge | `asset_class`, `horizon` | Confidence drift |
| `nexus_forecast_disagreement_count_total` | counter | `asset_class`, `dominant_signal_type` | Signal-disagreement frequency |
| `nexus_capability_test_pass` | gauge (0/1) | `asset_class`, `horizon` | Substrate gate status |
| `nexus_trace_grounding_failure_count_total` | counter | `asset_class` | I9 violations caught |
| `nexus_chain_integrity` | gauge (0/1) | — | SoulSeal hash chain status |
| `nexus_budget_consumed_pct` | gauge | `window=daily,monthly` | Cost gates |

### Latency budget per stage

| Stage | P50 | P95 | Notes |
|-------|-----|-----|-------|
| Macro isolator | 4s | 12s | Mostly DB + light LLM |
| Micro isolator | 2s | 6s | Mostly numerical, no LLM |
| Contextual integrator | 8s | 25s | RAG-heavy, LLM-summarization |
| Synthesis agent | 12s | 30s | Largest LLM call |
| End-to-end | 30s | 70s | Per single forecast |

Forecasts exceeding P95 are logged with `slow_forecast=true` and inspected weekly.

### Dashboards

Three Grafana boards (the Sentinel monitoring stack uses Grafana on Mac mini):

1. **Operations** — volume, latency, refusal rate, budget consumption
2. **Quality** — capability test status, calibration over time, MASE drift per adapter, grounding failure rate
3. **Cost** — daily/weekly/monthly spend, cost-per-forecast distribution, model-downgrade events

Dashboards are JSON-defined under `dashboards/nexus/` and version-controlled.

### Audit log

`logs/forecasts/audit.jsonl` — append-only, one line per forecast lifecycle event:

```
{"ts": "...", "event": "stage_started", "stage": "macro_isolator", "forecast_id": "..."}
{"ts": "...", "event": "stage_completed", "stage": "macro_isolator", "forecast_id": "...", "confidence": 0.72, "tokens": 6400}
{"ts": "...", "event": "forecast_completed", "forecast_id": "...", "refused": false, "soulseal_artifact_id": "..."}
```

### Service-level objectives (SLOs)

Quantitative consumer contracts. Error budgets reset monthly; SLO violations open the runbook in Section 18 and lock the GA rollout flag (Section 19) until cleared.

| SLO | Target | Measurement | Error budget (monthly) |
|-----|--------|-------------|------------------------|
| Forecast availability | 99.0% non-refused on healthy-input path | (completed) / (requested where staleness audit is empty) | 7.2 h |
| Forecast latency P50 | ≤ 30 s end-to-end | `nexus_forecast_latency_seconds{quantile="0.5"}` | 5% of forecasts above target |
| Forecast latency P99 | ≤ 90 s end-to-end | `nexus_forecast_latency_seconds{quantile="0.99"}` | 1% of forecasts above target |
| Trace persistence | 100% of non-refused forecasts have a SoulSeal entry | `nexus_chain_integrity == 1` daily | 0 — hard gate |
| Trace queryability | ≥ 99% of recent traces searchable within 60 s of generation | `indexed_at − generated_at` distribution | 1% |
| Capability-test freshness | All adapters green within rolling 7-day window | `nexus_capability_test_pass` last-pass timestamp | 1 stale adapter |
| Budget honor | Monthly USD spend ≤ ceiling | `nexus_budget_consumed_pct{window="monthly"}` | 0 — hard refusal at 100% |
| Refusal correctness | 100% of refusals carry structured `refusal_reason` | weekly audit sample of 1% of refusals | 0 |
| Grounding rate | ≥ 95% of forecasts pass grounding check | `nexus_trace_grounding_failure_count_total` / total | 5% (quarantine triggered) |
| Reproducibility | ≥ 99% of `temperature=0` replays produce bit-identical results | weekly `nexus_reproducibility.sh` sample | 1% (drift investigated) |

### Capacity planning

Steady-state workload sizing assumes single-operator deployment (one Mac mini, one Anthropic API key, no multi-node):

| Workload | Daily rate | Peak burst | Token budget |
|----------|------------|------------|---------------|
| Live forecasts (per Appendix G universe × 20% coverage) | ~20 forecasts/day | 8 concurrent at 09:30 local | ~1.5 M tokens/day |
| Live forecasts at full coverage (post R5 ramp) | ~96 forecasts/day | 24 concurrent | ~7 M tokens/day |
| Weekly capability re-evaluation (Section 10) | 500 forecasts in 4 h batch (Sunday 02:00 local) | 6× steady-state | ~25 M tokens/week (batched, 50% discount) |
| Ad-hoc operator forecasts | ≤ 10/day | 3 concurrent | < 500 K tokens/day |
| One-time backfill (e.g., 5-year history) | up to 5 000 forecasts | 50 concurrent over 24 h | ~75 M tokens one-time |

**Concurrency caps:**

- Synthesis stage: max **5 concurrent** LLM calls (Anthropic per-key tier-1 rate limit baseline).
- Stages 1/2/3 (Macro / Micro / Contextual): max **20 concurrent** each (DB + light LLM bound).
- Evaluation runs: gated to off-hours via cron; the eval runner refuses to start if `now()` overlaps the 09:00–16:00 local market-hours window where live forecasts dominate.

**Backpressure:**

- When concurrent forecast count exceeds 20, MCP `forecast_generate_v1` returns HTTP 429 with `Retry-After` header.
- The CLI `sentinel forecast batch` queues to a SQLite-backed FIFO at `data/forecasts/queue/`, survives crashes, and resumes on next CLI invocation.
- The Batch API path uses Anthropic's Message Batches API (24 h SLA) for capability re-eval; never used for live forecasts.

### Cost-quality elasticity

Phase 8 explicitly measures how forecast quality changes as the synthesis model is downgraded. Indicative shape (to be replaced with real Phase 8 numbers in the model card):

| Model | Cost per forecast (est.) | Expected ΔMASE vs Opus | Expected Δdir-acc vs Opus | Recommended live use |
|-------|---------------------------|-------------------------|----------------------------|----------------------|
| Opus 4.7 | $0.80 – $1.50 | baseline | baseline | High-conviction / FI / FX |
| Sonnet 4.6 | $0.20 – $0.50 | +0.04 (worse) | −0.02 | Default once Phase 8 confirms acceptable lift |
| Haiku 4.5 | $0.05 – $0.15 | +0.10 (worse) | −0.05 | Budget-exhausted fallback only |

If Phase 8 shows Sonnet within 2pp of Opus on directional accuracy at 1/4 the cost, the steady-state default flips to Sonnet and Opus becomes the high-conviction-only path. This decision is recorded in `docs/model-cards/nexus-v1.0.md` under "Cost-quality elasticity".

### Anthropic SDK doctrine

The pipeline targets Anthropic's prompt caching, extended thinking, and batch APIs for cost discipline. The synthesis stage dominates cost; this section defines the standing instructions code must follow.

**Prompt caching:**

- Stable cacheable blocks (5-minute ephemeral cache, marked `cache_control: {"type": "ephemeral"}`):
  - System prompt (role frame + shared grounding rules — Section 14) — invariant across forecasts in the same pipeline_version
  - Schema documentation block — invariant within a schema_version
  - Adapter-specific output guidance block — invariant within an adapter version
- Variable per-forecast blocks (NEVER cached):
  - The three signal JSON payloads
  - `instrument`, `as_of`, `horizon`
- **Target cache-hit rate: ≥ 60% of input tokens** at steady state. Sustained < 50% triggers Section 18.6 runbook.
- Cache breakpoints are recorded in `ReproducibilityManifest.cache_breakpoints` so manifests are explicit about the cache topology.

**Extended thinking:**

- Synthesis stage uses extended thinking with budget = **16 000 tokens** by default; configurable per adapter.
- Stages 1/2/3 do NOT use extended thinking — they are classification + extraction, not multi-step reasoning. Adding thinking here is a 3× cost increase for ≤ 1pp quality lift (Phase 8 ablation will confirm or refute).
- Thinking tokens count against the per-asset `max_tokens_per_forecast` cap.

**Batch API:**

- Weekly capability re-evaluation runs through Anthropic's **Message Batches API** (~50% discount, 24 h SLA). Acceptable for offline eval; logged in `logs/forecasts/batch_runs/<run-id>.jsonl`.
- Live forecasts NEVER use the batch API (24 h SLA incompatible with intraday decisions).

**Model routing (`sai/llm_router.py` enforces):**

| Workload | Default model | Downgrade at 80% daily budget | Downgrade at 95% daily budget |
|----------|---------------|-------------------------------|-------------------------------|
| Live synthesis — equities / FX / FI / commodities | `claude-opus-4-7` | `claude-sonnet-4-6` | `claude-haiku-4-5-20251001` |
| Live synthesis — crypto | `claude-sonnet-4-6` (crypto inherently noisier; Opus marginal lift small) | `claude-haiku-4-5-20251001` | refuse with `BUDGET_EXHAUSTED` |
| Stages 1/2/3 (extractive) | `claude-sonnet-4-6` | `claude-haiku-4-5-20251001` | refuse |
| Capability re-eval | `claude-sonnet-4-6` (consistency over peak quality; documented bias in model card) | n/a (batch API; pre-budgeted) | n/a |
| Ad-hoc operator forecast | `claude-opus-4-7` | operator override required | operator override required |

**Required SDK call shape (synthesis path):**

```python
response = client.messages.create(
    model=routed_model,
    max_tokens=per_asset_cap - max(2000, per_asset_cap // 20),  # 5% margin
    temperature=0.0,
    system=[
        {"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": SCHEMA_DOC, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": ADAPTER_GUIDANCE, "cache_control": {"type": "ephemeral"}},
    ],
    messages=[{"role": "user", "content": user_prompt}],
    thinking={"type": "enabled", "budget_tokens": 16000} if routed_model == "claude-opus-4-7" else None,
    stop_sequences=["</forecast>"],
    metadata={"user_id": "sentinel_nexus_v1", "forecast_id": forecast_id},
    extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
)
```

The exact code lives in `sentinel/sai/nexus/stages/synthesis_agent.py`; the snippet above is the canonical reference and is asserted against in `test_sdk_call_shape.py`.

**Token accounting:**

- `cost_metadata.tokens_cache_read` and `tokens_cache_write` are tracked separately from `tokens_input` (cached reads cost 10% of normal input tokens; cache writes cost 125% of normal input tokens — see Anthropic pricing).
- Estimated USD per forecast is computed in `sentinel/sai/nexus/cost/usd_estimator.py` from the per-model rate card stored in `.danteforge/config/nexus-pricing.json` (auto-updated by `make pricing-refresh`).

---

## 17. Security, privacy, and compliance

### Credentials

| Credential | Storage | Rotation | Failure mode |
|------------|---------|----------|--------------|
| Anthropic API key | OS keychain via `keyring` | Manual, ≤ 180 days | Forecasts refuse with `BUDGET_EXHAUSTED` synthetic reason; no fallback to other providers |
| Postgres password | `.env` outside repo | Manual, ≤ 180 days | Trace persistence falls back to filesystem; chain still verifiable |
| FRED API key | OS keychain | None required | Macro stage emits stale-data refusal |
| News API keys | OS keychain | Per-vendor | Contextual stage degrades; refusal possible |

No credentials are ever written to `logs/`, `data/`, `dashboards/`, or any committed file. Pre-commit hook scans for high-entropy strings.

### PII

The forecaster handles instrument data, news text, filings text. None of these are PII by US/EU standards in the normal case. The single edge case: news articles may quote individuals. Three guards:

1. **No PII storage in indexes** — `forecast_traces_index.summary_text` excludes verbatim names from outside-public-figure list; redacted at trace-writer time.
2. **No PII in audit log** — only forecast IDs, no document excerpts.
3. **Right-to-be-forgotten path** — `sentinel forecast trace purge <forecast-id>` removes the trace + breaks-and-rebuilds chain via append-only tombstone (chain integrity preserved by referencing the tombstone hash, not the deleted payload).

### Licensing and ToS

Every data source is tagged with `SourceLineageTier`. T3 (ToS-restricted) sources may inform reasoning *internally* but their excerpts must not leak into:

- `ReasoningTrace.summary`
- `ReasoningTrace.inputs_used.excerpt` (truncated to allowed length per source)
- Any API response served externally

Enforcement: `trace_writer.py` consults a per-source `tos_excerpt_policy.json` before writing excerpts.

### Model risk management (MRM)

For any future regulated-buyer use, the system supports MRM-style documentation out of the box:

- **Model card** — `docs/model-cards/nexus-v1.0.md` summarizing intended use, training-data analogues (we don't train, but we document LLM provenance), known limitations, evaluation results
- **Validation evidence** — Phase 5/8 outputs are MRM-compatible
- **Change log** — every schema bump triggers a change-log entry; LLM swaps are change-log events

### Threat model

| Threat | Surface | Mitigation |
|--------|---------|------------|
| Prompt injection from news/filings | Contextual stage RAG | Strip system-prompt-like patterns; constrain context to schema fields, not free continuation |
| Data poisoning via scraped sources | T3 sources | Quarantine T3-only signals; require T1/T2 corroboration for high-confidence drivers |
| Forecast tampering | Stored traces | SoulSeal hash chain; chain verify on every read |
| Budget DoS via batch forecast request | CLI / MCP | Per-window rate limit on `forecast batch`; concurrent forecast cap |
| Trace exfiltration | Postgres + filesystem | OS-level file perms; no cloud sync of `data/forecasts/` |

---

## 18. Operational runbooks

Each runbook lives in `docs/runbooks/nexus/<name>.md` and follows a four-section template: **Symptom**, **Diagnosis steps**, **Resolution**, **Postmortem template**.

### 18.1 Cost spike

- **Symptom:** `nexus_budget_consumed_pct{window="daily"}` crosses 0.9 before 6pm local, or weekly run-rate would exhaust monthly cap before day 20
- **Diagnosis:** `sentinel forecast cost --breakdown` — which adapter, which stage, which horizon spiked?
- **Resolution:** Force model downgrade (`force_cheaper_model=true` in cost config), reduce instrument universe, or pause forecast batches; in worst case, set monthly cap to 0 (refuse all new forecasts) until next billing cycle
- **Postmortem:** Was the spike user-driven (operator requested a 1000-instrument batch) or stage-driven (a refactor inflated context length)? Update runbook with the new mode.

### 18.2 Post-cutoff evaluation regression

- **Symptom:** Weekly `make nexus-eval` shows an adapter's MASE or directional-accuracy crosses its threshold the wrong way
- **Diagnosis:** Compare regressed run to last passing run — model swap? Data freshness? Schema change? Stratified by regime, is the regression concentrated in transitions?
- **Resolution:** Mark adapter `degraded`, freeze downstream consumption (strategy lab refuses to consume from a `degraded` adapter), open an investigation; rollback model if a model swap correlates
- **Postmortem:** Did monitoring miss a leading indicator (calibration drift)? Add the leading metric to alerts.

### 18.3 SoulSeal chain corruption

- **Symptom:** `nexus_chain_integrity == 0` or `sentinel forecast trace verify` fails
- **Diagnosis:** Identify the first broken link via `sentinel core soulseal verify --verbose`; was it a partial write, disk full, or process crash?
- **Resolution:** Quarantine chain from broken link forward; rebuild a parallel chain rooted at the last verified hash; promote parallel chain after manual review; do **not** silently rewrite history
- **Postmortem:** Need fsync on chain writes? Disk-space alarm?

### 18.4 Stale-data refusal storm

- **Symptom:** `nexus_forecast_refusal_count_total{refusal_reason="STALE_INPUTS"}` spikes
- **Diagnosis:** Which feed is stale? Check `make data-freshness`; is it an upstream vendor outage or our scheduler?
- **Resolution:** If vendor outage: tolerate (the refusals are correct); if scheduler: restart, then backfill missed ingests; if a single feed is reliably stale, consider tightening or loosening its `nexus-staleness.json` entry
- **Postmortem:** Did the feed degrade slowly (advance warning possible) or fail cliff-edge?

### 18.5 LLM model swap

- **Symptom:** Anthropic releases a new model or deprecates a current one
- **Diagnosis:** `sentinel/sai/llm_router.py` resolves a different model than expected
- **Resolution:** Pin the model explicitly; update `llm_cutoffs.json` with the new model's conservative cutoff; regenerate test sets; re-run capability tests before opening the new model to live forecasts
- **Postmortem:** Did we have enough notice from upstream? Subscribe to provider changelog?

### 18.6 Grounding failure surge

- **Symptom:** `nexus_trace_grounding_failure_count_total` spikes for one adapter
- **Diagnosis:** Did the synthesis prompt template change? Did an upstream stage start emitting a malformed `source_lineage`? Is the model fabricating drivers?
- **Resolution:** Quarantine failing traces; tighten anti-rationalization clauses in `prompts/synthesis_prompt.py`; if persistent, downgrade the model to a stricter sibling
- **Postmortem:** Update the grounding test fixtures with the new failure mode.

### 18.7 Disaster recovery — laptop dies

- **Symptom:** Mac mini hard-fails; primary substrate unavailable
- **Resolution:** Restore Postgres from the nightly `pg_dump` in encrypted offline backup; restore `data/forecasts/` from rsync mirror; replay the SoulSeal chain from `evidence/forecasts/chain.jsonl`; verify integrity; resume operation
- **RPO:** 24h; **RTO:** 8h

### 18.8 Reproducibility drift

- **Symptom:** Weekly `nexus_reproducibility.sh` reports a temperature=0 replay producing a non-bit-identical `ForecastResult`
- **Diagnosis:** Compare the live replay's `ReproducibilityManifest` against the original — which hash changed? Three usual culprits: (a) a prompt template was edited without a version bump, (b) a config file was edited without a version bump, (c) the upstream model silently changed behavior (Anthropic side-channel update).
- **Resolution:** If (a) or (b), bump the relevant version, re-baseline the manifest, write a model-card change-log entry. If (c), pin the model explicitly to a dated revision and re-run capability tests before continuing. Quarantine any forecasts whose manifests reference the drifted artifact.
- **Postmortem:** Did anything bypass the pre-commit hash check? Tighten the hook.

### 18.9 Chaos drill — quarterly substrate confidence test

Once per calendar quarter, run the chaos drill in staging (NOT production). The drill exercises every runbook above against synthetic faults:

| Fault injected | Expected response | Pass criterion |
|----------------|-------------------|-----------------|
| Kill the LLM router mid-forecast | Synthesis stage refuses with `INSUFFICIENT_CONTEXT`; no half-written trace persists | No partial SoulSeal entries; no orphan `forecast_id` in DB |
| Truncate `nexus-staleness.json` to zero bytes | Loader refuses with clear error; forecasts halt rather than treating everything fresh | Halt with logged diagnostic in 18.4 form |
| Append a malformed entry to the SoulSeal chain | `nexus_chain_integrity` metric drops to 0; alert fires; new forecasts blocked from chain write | 18.3 runbook executes; quarantine + parallel chain |
| Set `nexus-cost.json` daily budget to 1 token | Next forecast refuses with `BUDGET_EXHAUSTED`; existing in-flight forecasts complete | 18.1 runbook; no silent quality reduction |
| Rotate Anthropic API key with no replacement | All synthesis calls fail; forecasts refuse with structured reason | Refusal is graceful; nothing crashes |
| Delete a config file (`nexus-stages.json`) | Loader refuses to start the pipeline; clear error message | No crash; explicit refuse-to-start |
| Inject a malformed `MacroSignal` (confidence = 1.7) | Pydantic validator catches at handoff; synthesis stage never sees it | Validator error; forecast refuses with `SCHEMA_VALIDATION_FAILED` |
| Inject a `Driver.citation` pointing to a non-existent field | Grounding check fails; trace quarantined; alert fires | Grounding failure metric increments; trace lands in `quarantine/` not main chain |

The drill is scripted in `scripts/nexus_chaos_drill.sh`. Quarterly schedule is on the CRON calendar (next: 2026-09-01). Drill results land in `docs/chaos-drills/<date>.md` and feed any runbook refinements.

---

## 19. Rollout and launch plan

### Stages of release

| Stage | Audience | Gate to next stage |
|-------|----------|---------------------|
| **R0 — internal stub** | Phase 0 author only | Imports clean, public surface frozen |
| **R1 — shadow** | Phase 2 outputs to disk; nothing downstream consumes them | Capability tests for stages pass |
| **R2 — adapter alpha** | Equities adapter only, behind feature flag `nexus.equities=on` | Equities capability test passes post-cutoff |
| **R3 — full alpha** | All 5 adapters behind flags | ≥ 4/5 capability tests pass |
| **R4 — strategy-lab beta** | Strategy lab may opt-in consume forecasts | Phase 8.4 strategy integration green |
| **R5 — GA** | All Sentinel consumers may consume by default | Phase 8 ship gate passes |

### Feature flags

Defined in `.danteforge/config/feature-flags.json`:

```json
{
  "nexus.enabled": true,
  "nexus.adapters.equities": "alpha",
  "nexus.adapters.fx": "shadow",
  "nexus.adapters.fixed_income": "shadow",
  "nexus.adapters.commodities": "shadow",
  "nexus.adapters.crypto": "shadow",
  "nexus.consumers.strategy_lab": false,
  "nexus.consumers.council": false
}
```

Values: `off`, `shadow` (run but do not return), `alpha` (return to opt-in callers), `beta` (default-on for opt-in callers), `ga` (default-on for everyone).

### Canary plan

When promoting an adapter from `alpha` to `beta`:

1. Run the adapter in shadow mode for 7 days alongside the live alpha (same `as_of` dates, separate IDs)
2. Diff outputs daily; investigate any divergence > 1% on direction probability
3. Verify capability test still passes in shadow runs
4. Flip to `beta` only after 7 consecutive green days

### Ship checklist

- [ ] All Phase 0–8 deliverables in `Section 25` complete
- [ ] Constitutional invariants I1–I12 spot-checked
- [ ] Cost gate test triggered intentionally in staging (refused-on-budget verified)
- [ ] Trace integrity check on 1000 random forecasts: 100% pass
- [ ] Strategy-lab integration end-to-end test green
- [ ] Operational runbooks 18.1–18.9 reviewed; chaos drill executed once in staging
- [ ] Model card v1.0 written
- [ ] CHANGELOG updated
- [ ] Tag `nexus-v1.0` on `main`
- [ ] Dashboards live, alerts wired
- [ ] CI: `nexus_capability_*` tests added to `make check`; reproducibility test on the 20-sample weekly subset
- [ ] Pre-commit hooks: prompt-template hash recompute on edit; config-schema-version check on `.danteforge/config/nexus-*.json`
- [ ] Backfill plan (§19.7) executed if pre-Nexus walk-forward backtests exist

### 19.7 Backfill plan for pre-Nexus strategy backtests

Sentinel has walk-forward backtests running before this PRD ships (Wave 34 PBO/DSR infrastructure, technical strategies in `sentinel/spm/strategy_lab/`). Some of these consume "forecast-like" inputs (e.g., predicted-return columns from simpler models). The migration to Nexus forecasts is not automatic.

**Migration matrix:**

| Existing strategy type | Migration path |
|------------------------|----------------|
| Pure technical (RSI, MACD, factor signals; no LLM input) | No migration needed; continue running as-is |
| Uses `sai/research_agent_v3` directly (pre-Nexus LLM research) | Wrap with a Nexus adapter; route through `sentinel forecast` so the trace is captured. Old backtests preserved as `pre-nexus` archive. |
| Uses external forecast vendor data | Replace vendor input with Nexus forecast; document the swap in the strategy's `STRATEGY.md`. |
| Custom in-house forecast (numerical only, no reasoning trace) | Optional migration; comparison report runs both side-by-side for 30 days before swap. |

**Backfill execution (only for strategies migrating to Nexus inputs):**

1. **Confirm cutoff discipline.** The strategy's backtest window must be entirely post-cutoff for the model that generated the Nexus forecasts. If the strategy depends on pre-cutoff dates, the backfill cannot proceed — surface honestly, do not silently use leaky data.
2. **Generate the historical forecast catalog.** Run `sentinel forecast batch --asof-range <start>:<end> --instruments <list> --horizons <list>` against the post-cutoff window. Cost is one-time; budget per Appendix G's "one-time backfill" capacity row (~$75 of token spend at full coverage of a 5-year window). Backfill batches run through the Batch API for the 50% discount.
3. **Persist as a fixed catalog.** Output lands at `data/forecasts/backfill/<strategy-id>/<run-id>.parquet`. This file is immutable. The strategy backtest reads it; it does not re-generate forecasts on each backtest run.
4. **Re-run the walk-forward** against the backfilled forecast catalog. Apply PBO/DSR gates (Wave 34); compare aggregate metrics to the strategy's pre-Nexus baseline.
5. **Document the migration outcome** in `docs/strategy-migrations/<strategy-id>.md`. Include the head-to-head metrics, the cost of the migration, and a P/F verdict.

**Failure modes the backfill must avoid:**

- **Time-travel via forecast cache:** if a backfill forecast was generated *after* its `as_of` date and somehow incorporated information from after `as_of`, the strategy backtest is poisoned. Mitigation: the `test_set_builder.py` cutoff guard (Section 10) applies to backfill too — the forecaster refuses to generate a backfill forecast for an `as_of` it could not have honestly produced at that time.
- **Schema drift mid-backfill:** if the schema changes during a multi-day backfill run, the catalog is heterogeneous. Mitigation: backfill runs are pinned to a single `pipeline_version` recorded in the manifest; the catalog file's header carries the version stamp.

**Pre-Nexus strategies that ship without migration** continue to run; they're explicitly marked `nexus-input=false` in the strategy registry and are reported separately in dashboards so the forecasting layer's lift is not contaminated by strategies that don't use it.

---

## 20. Versioning and backwards compatibility

### Schema versioning

Every top-level dataclass carries an explicit `schema_version` field of the form `nexus-<scope>/<major>.<minor>`. Examples in Section 15.

| Version bump | When | Migration |
|--------------|------|-----------|
| Patch | None (deletes minor unused fields, doc-only changes) | None |
| Minor | New optional field added | Old consumers continue working; new consumers may use the field |
| Major | Field removed, renamed, semantics changed | Old traces migrated lazily on read; new writes use new version; migration runner ships with the change |

### Trace migration

Old traces are immutable. Migration is read-side: `trace_writer.py` knows how to load older versions and either re-emit them as the new version or expose them via a v1-compatible view. Migration runners are gated behind manual invocation, never automatic on read.

### LLM versioning

Every `ReasoningTrace.cost_metadata.model_used` records the exact model string. When the configured model changes, the cutoff registry updates, test sets are regenerated, capability tests re-run before the new model serves live forecasts. There is no "promote new model in place" path — every model change is a versioned event with its own evaluation receipt.

### API versioning (MCP tools)

MCP tool names include version suffix: `mcp__sentinel__forecast_generate_v1`. Breaking changes mint v2; v1 remains live for one full release cycle then gets retired with a clear error.

### Configuration versioning

Every config file in `.danteforge/config/nexus-*.json` has a `schema_version` field. The loader rejects unknown major versions with a clear error; unknown minor versions log a warning and fall back to defaults for unknown fields.

---

## 21. Timeline and roadmap with absolute dates

Reference date: **2026-05-18** (today). All dates are calendar dates. Engineering days assume a single operator; calendar dates include weekends and buffer for non-engineering activities.

### Gantt-style timeline

| Date | Engineering day | Phase | Activity |
|------|-----------------|-------|----------|
| 2026-05-18 (Mon) | — | — | PRD v1.0 frozen (this document) |
| 2026-05-19 (Tue) | 1 | Phase 0 | Preflight checklist + stub package |
| 2026-05-20 (Wed) | 2 | Phase 1 | Read paper, draft harvest notes 1.1–1.3 |
| 2026-05-21 (Thu) | 3 | Phase 1 | Finish harvest notes 1.4–1.5, review |
| 2026-05-22 (Fri) | 4 | Phase 2 | Handoff dataclasses + `nexus_forecaster_v3.py` skeleton |
| 2026-05-23 (Sat) | — | — | Buffer |
| 2026-05-24 (Sun) | — | — | Buffer |
| 2026-05-25 (Mon) | 5 | Phase 2 | Macro isolator + micro isolator + tests |
| 2026-05-26 (Tue) | 6 | Phase 2 | Contextual integrator + synthesis agent + e2e test |
| 2026-05-27 (Wed) | 7 | Phase 3 | Equities adapter |
| 2026-05-28 (Thu) | 8 | Phase 3 | FX + fixed income adapters |
| 2026-05-29 (Fri) | 9 | Phase 3 | Commodities + crypto adapters |
| 2026-05-30 (Sat) | — | — | Buffer / per-adapter capability tests |
| 2026-05-31 (Sun) | — | — | Buffer |
| 2026-06-01 (Mon) | 10 | Phase 4 | Trace schema + `trace_writer.py` + SoulSeal wiring |
| 2026-06-02 (Tue) | 11 | Phase 4 | pgvector index + grounding tests |
| 2026-06-03 (Wed) | 12 | Phase 5 | `test_set_builder.py` + cutoff registry validation |
| 2026-06-04 (Thu) | 13 | Phase 5 | `post_cutoff_evaluator.py` + first capability runs (equities, FX) |
| 2026-06-05 (Fri) | 14 | Phase 5 | Capability runs for FI, commodities, crypto |
| 2026-06-06 (Sat) | 15 | Phase 6 | `council_contract.py` + `advisor_consumes_forecast.py` |
| 2026-06-07 (Sun) | 16 | Phase 7 | Cost tracker + budget gates |
| 2026-06-08 (Mon) | 17 | Phase 7 | Quality controls + staleness + confidence floor |
| 2026-06-09 (Tue) | 18 | Phase 8 | Full validation suite — eval, cost, trace, strategy-lab dry run |
| 2026-06-10 (Wed) | 19 | Phase 8 | Strategy-lab end-to-end with PBO/DSR + council dry run |
| 2026-06-11 (Thu) | 20 | Phase 8 | Evidence files, honest assessment, model card |
| 2026-06-12 (Fri) | — | — | Slack / rework |
| 2026-06-13 (Sat) | — | — | Slack / rework |
| 2026-06-14 (Sun) | — | — | Ship gate review |
| 2026-06-15 (Mon) | 21 | Ship | Tag `nexus-v1.0`, flip flags, post-launch monitoring |

### Post-launch milestones

| Date | Milestone |
|------|-----------|
| 2026-06-22 (T+1w) | First weekly capability re-run; first cost report |
| 2026-06-29 (T+2w) | Adapter promotion review (alpha → beta candidates) |
| 2026-07-15 (T+1m) | Monthly cost review; model swap decision (Opus 4.7 vs Sonnet 4.6 vs Haiku 4.5 cost-quality cut) |
| 2026-09-01 (T+~2.5m) | Quarterly chaos drill (Section 18.9) — first run |
| 2026-09-15 (T+3m) | Council of Investors integration begins (Phase 6 contract consumed) |
| 2026-12-15 (T+6m) | Per-instrument tuning evaluation (Phase 9 decision per "open question 1") |

### Phase dependency DAG and critical path

Phases are mostly sequential but several branches can run in parallel. The dependencies:

```text
Phase 0 (preflight)
   |
   v
Phase 1 (harvest notes)
   |
   v
Phase 2 (four-stage core) ────────┐
   |                              |
   v                              v
Phase 3 (adapters) ────┐    Phase 4 (trace + chain)
   |                   |          |
   |                   v          v
   |             Phase 7 (cost gates, partial — can start once Phase 2 emits cost_metadata)
   |                              |
   v                              v
Phase 5 (post-cutoff eval) <──────┘
   |
   v
Phase 6 (council contract)     [can run any time after Phase 4 since contract just consumes traces]
   |
   v
Phase 8 (e2e validation) <──── Phase 7 (cost gates, full)
   |
   v
Ship gate
   |
   v
Tag nexus-v1.0
```

**Critical path:** Phase 0 → 1 → 2 → 3 → 5 → 8 → Ship. The non-critical-path branches (Phase 4 trace integration, Phase 6 contract, Phase 7 cost gates) can run alongside the critical path. With a single operator the parallelism is conceptual not literal — the operator works on one phase at a time — but the DAG documents which phases CAN be interleaved if a second contributor joins.

**Parallelization opportunities (if extra hands appear):**

| Phase pair | Parallelizable? | Notes |
|------------|------------------|-------|
| Phase 3 adapters vs Phase 4 trace | Yes | Different files; adapter work only touches `adapters/*.py`, trace work only touches `trace/*.py` and SQL |
| Phase 4 trace vs Phase 7 cost gates | Yes | Independent concerns |
| Phase 5 eval vs Phase 6 council contract | Yes | Council contract has no eval dependencies |
| Phase 3 different adapters | Yes | Each adapter is independent; one operator could parallelize across e.g. equities + FX |

**Blockers that serialize:**

| If this happens | These phases halt |
|------------------|-------------------|
| Phase 2 handoff schema redesign | Phase 3, 4, 5, 6 all halt (every later phase depends on the schemas) |
| Phase 4 trace schema breaks SoulSeal write path | Phase 8 halts (can't validate without traces) |
| Phase 5 capability tests all fail | I8 fails by default; Phase 8 still runs but cannot ship as is |

---

## 22. Comprehensive command surface map

### New CLI commands

| Command | Purpose |
|---------|---------|
| `sentinel forecast <instrument>` | Generate a Nexus forecast for an instrument |
| `sentinel forecast <instrument> --horizon <h>` | Specify horizon (1d, 5d, 30d, 90d, 1y) |
| `sentinel forecast <instrument> --explain` | Print reasoning trace summary |
| `sentinel forecast <instrument> --full-trace` | Output full reasoning trace JSON |
| `sentinel forecast batch <file>` | Forecast multiple instruments from input file |
| `sentinel forecast eval` | Run post-cutoff evaluation on current adapters |
| `sentinel forecast eval --asset-class <c>` | Evaluate one asset class |
| `sentinel forecast eval --refresh-test-sets` | Regenerate test sets against current cutoffs |
| `sentinel forecast cost` | Show cost summary (daily, weekly, monthly) |
| `sentinel forecast cost --breakdown` | Cost broken down by asset class and stage |
| `sentinel forecast trace <forecast-id>` | Retrieve and display a specific trace |
| `sentinel forecast trace verify --all` | Verify SoulSeal integrity across all traces |
| `sentinel forecast trace purge <forecast-id>` | Right-to-be-forgotten purge (chain tombstone) |
| `sentinel forecast traces query <query>` | Search traces by criteria |
| `sentinel forecast adapters list` | List available asset class adapters |
| `sentinel forecast adapters status` | Show capability test status per adapter |
| `sentinel forecast adapters promote <name> <stage>` | Promote an adapter (shadow → alpha → beta → ga) with required gates |

### MCP tools exposed

For consumption by Council of Investors agents, Sentinel API users, and other Dante projects:

| Tool name | Input schema | Output schema | Notes |
|-----------|--------------|---------------|-------|
| `mcp__sentinel__forecast_generate_v1` | `{symbol, asset_class, horizon, as_of?, purpose}` | `ForecastResult` (Section 15) | Synchronous; refuses on stale/budget |
| `mcp__sentinel__forecast_get_trace_v1` | `{forecast_id}` | `ReasoningTrace` | Returns 404 if purged via tombstone |
| `mcp__sentinel__forecast_query_traces_v1` | `{instrument?, time_range?, min_confidence?, dominant_signal?, semantic_query?, limit?}` | `List[TraceSummary]` | Backed by pgvector + indexes |
| `mcp__sentinel__forecast_eval_status_v1` | `{asset_class?, horizon?}` | `EvalStatusReport` | Last-N evaluations + thresholds + pass/fail |
| `mcp__sentinel__forecast_get_capability_status_v1` | `{asset_class}` | `{status: alpha|beta|ga|degraded, last_passing_run, current_metrics}` | Used by feature-flag readers |
| `mcp__sentinel__forecast_cost_report_v1` | `{window: daily|weekly|monthly}` | `CostReport` | Token + USD breakdowns |
| `mcp__sentinel__forecast_verify_chain_v1` | `{from_id?, to_id?}` | `{integrity: bool, broken_link_id?}` | SoulSeal integrity probe |

### Existing commands modified

| Command | What changes |
|---------|--------------|
| `sentinel strategy backtest` | Can now optionally seed strategies from Nexus forecasts (`--forecast-input`) |
| `sentinel strategy paper-trade` | Records which forecasts informed which trades |
| `sentinel measure` | Includes Nexus capability test status per asset class |
| `sentinel score` | Includes a forecasting-capability dimension once Phase 5 substrate gate is live |

---

## 23. Configuration and storage surfaces

### Configuration files

```
.danteforge/config/
├── nexus-stages.json              # per-stage configuration (data sources, thresholds)
├── nexus-cost.json                # cost budgets and gates (Section 12)
├── nexus-evaluation.json          # eval thresholds per asset class (Section 10)
├── nexus-staleness.json           # data staleness thresholds (Section 12)
├── nexus-adapters.json            # adapter-specific config
├── feature-flags.json             # rollout flags (Section 19)
└── tos_excerpt_policy.json        # per-source ToS excerpt rules (Section 17)
```

### Storage layout

```
data/forecasts/
├── <year>/<month>/<forecast-id>.json           # individual forecasts + traces
├── eval/
│   └── <model>/<asset-class>/<test-set-id>.json
└── batches/<batch-id>/
    ├── summary.json
    └── individual forecasts

evidence/forecasts/
├── chain.jsonl                                  # SoulSeal evidence chain entries
└── index/                                       # query indexes

logs/forecasts/
├── audit.jsonl                                  # lifecycle events
├── events-<date>.jsonl                          # per-forecast events
├── cost-daily-<date>.jsonl
├── degraded-models.jsonl
└── capability-test-history.jsonl
```

### Database additions

```sql
CREATE TABLE forecast_traces_index (
    forecast_id           TEXT PRIMARY KEY,
    instrument            TEXT NOT NULL,
    asset_class           TEXT NOT NULL,
    as_of                 TIMESTAMPTZ NOT NULL,
    horizon               TEXT NOT NULL,
    direction_probability JSONB NOT NULL,
    confidence            FLOAT NOT NULL,
    dominant_signal_type  TEXT NOT NULL,
    model_used            TEXT NOT NULL,
    tokens_consumed       INT NOT NULL,
    soulseal_artifact_id  TEXT NOT NULL,
    summary_embedding     vector(384),    -- pgvector for semantic search
    summary_text          TEXT NOT NULL,
    refused               BOOLEAN NOT NULL DEFAULT FALSE,
    refusal_reason        TEXT,
    generated_at          TIMESTAMPTZ NOT NULL,
    schema_version        TEXT NOT NULL
);

CREATE INDEX idx_traces_instrument          ON forecast_traces_index(instrument);
CREATE INDEX idx_traces_as_of               ON forecast_traces_index(as_of);
CREATE INDEX idx_traces_asset_class         ON forecast_traces_index(asset_class);
CREATE INDEX idx_traces_confidence          ON forecast_traces_index(confidence);
CREATE INDEX idx_traces_summary_embedding   ON forecast_traces_index
    USING ivfflat (summary_embedding vector_cosine_ops);

CREATE TABLE forecast_capability_status (
    asset_class           TEXT NOT NULL,
    horizon               TEXT NOT NULL,
    status                TEXT NOT NULL CHECK (status IN ('off','shadow','alpha','beta','ga','degraded')),
    last_run_at           TIMESTAMPTZ,
    last_pass_at          TIMESTAMPTZ,
    last_metrics          JSONB,
    PRIMARY KEY (asset_class, horizon)
);

CREATE TABLE forecast_chain_tombstones (
    forecast_id           TEXT PRIMARY KEY,
    tombstoned_at         TIMESTAMPTZ NOT NULL,
    reason                TEXT NOT NULL,
    chain_replacement_hash TEXT NOT NULL  -- preserves chain integrity
);
```

---

## 24. Consolidated stop conditions

Stop and report when any of these fire. Do not work around silently.

### Phase 0 stops

- Repo state dirty or test suite red → halt; resolve before Phase 1
- Module / data dependency missing → halt; document the gap

### Phase 1 stops

- Harvest target too vague to write a useful note → document gap, decide
- Pattern requires capability Sentinel doesn't have → halt, document prerequisite

### Phase 2 stops

- Stage requires data source Sentinel doesn't have → halt that stage, document gap
- Handoff invariants can't be enforced cleanly → refactor handoff shape
- Capability tests for any stage cannot be satisfied → halt, document why

### Phase 3 stops

- Adapter requires asset-specific data Sentinel doesn't have → halt that adapter
- Asset class has structurally different forecasting needs (e.g., binary event prediction) → halt, document why
- Adapter's domain capability tests can't be satisfied → halt, document gap

### Phase 4 stops

- Reasoning traces are too large to store sustainably → reduce trace verbosity, retain critical fields
- Evidence chain integrity breaks under load → halt integration, fix chain
- Grounding check fails on > 5% of generated traces → halt; tighten synthesis prompt template (I9)

### Phase 5 stops

- Post-cutoff evaluation shows results worse than naive baseline → halt that adapter, don't ship until improved
- Conservative cutoff date is too recent for meaningful evaluation → halt evaluation, wait for more post-cutoff data, document timeline
- Evaluation cost exceeds sustainable budget → reduce test set size to minimum statistical validity

### Phase 6 stops

- Council contract interface conflicts with existing Sentinel API patterns → refactor contract to fit Sentinel conventions, don't force the conflict

### Phase 7 stops

- Daily/monthly cost ceilings cannot be hit with quality intact → renegotiate ceilings or reduce forecast frequency, do not lower quality silently
- Confidence floors filter out too many forecasts → investigate root cause, don't lower floor to hide poor quality

### Phase 8 stops

- End-to-end validation produces results worse than monolithic baseline on majority of asset classes → halt, treat as failure, investigate why decomposition isn't helping
- Strategy lab integration breaks PBO/DSR gates → halt, restore gates before shipping

### Universal stops

- Constitutional invariant violation (I1–I12)
- Any attempt to use pre-cutoff data in evaluation
- Cost spike that would exhaust monthly budget in days
- Trace integrity broken (forecasts being produced without traces, or with invalid traces)
- Schema-version mismatch on a write path (better to refuse the write than corrupt the index)

---

## 25. Verification artifacts per phase

After each phase completes, paste these for operator review:

### Phase 0

- `docs/harvest-notes/nexus/00-preflight.md` with checklist outcomes
- Stub package importable; `pytest sentinel/sai/nexus -k stub_smoke` green

### Phase 1

- All five harvest notes in `docs/harvest-notes/nexus/`
- Proof the patterns were understood, not copied (verbatim copying of paper text disqualifies)

### Phase 2

- All four stage modules implemented and passing unit tests
- Handoff dataclasses defined and Pydantic-validated
- End-to-end test passing on at least one historical period
- Sample forecast output with reasoning trace (see Appendix B)

### Phase 3

- All five adapters implemented (or honest documentation of which were skipped and why)
- Per-adapter capability test outputs from the matrix in Section 8
- Sample forecasts from each adapter on representative instruments

### Phase 4

- Trace schema documented (Section 15)
- Storage layout verified working
- Sample trace retrieved from evidence chain (Appendix A)
- Trace query examples (by instrument, by time, semantic search)
- Grounding failure quarantine empty after final pre-ship run

### Phase 5

- Test sets constructed for each adapter (proof of cutoff date discipline)
- Evaluation runs completed with metrics
- Capability test outputs (pass/fail per asset class)
- Comparison vs naive baselines
- Comparison vs monolithic baseline (I8 head-to-head)

### Phase 6

- Council contract interface defined
- Sample integration code showing how Dalio-agent would consume a Nexus forecast
- No actual Council code yet (deferred), but the contract is real

### Phase 7

- Cost dashboard showing per-stage breakdown
- Cost gates triggered correctly in test conditions (intentional budget-exhaustion drill)
- Confidence floor filtering working as expected
- Staleness refusal working as expected (stale-fixture test green)

### Phase 8

- Full validation suite output
- Per-asset-class metrics table
- Cost summary for full validation run
- Strategy lab integration evidence
- Honest assessment of which adapters are production-ready and which need more work
- Model card v1.0 published at `docs/model-cards/nexus-v1.0.md`

---

## 26. Risks and mitigations

**R1. The four-stage decomposition might not actually beat monolithic prompting for our use cases.**
Mitigation: Phase 8 explicitly tests this. If decomposition doesn't beat monolithic, that's important information — surface it, don't hide it (I8). The paper showed decomposition wins on their testbed; our testbed (FX, options, multi-asset) is different and the result might differ. If the head-to-head loses, ship the monolithic baseline behind the same interface and treat the four-stage as research-only.

**R2. Token cost explodes at scale.**
Four LLM calls per forecast × thousands of instruments × multiple horizons = real money. Mitigation: cost gates (Section 12), model downgrade when budget tight, refusal when budget exhausted. The committed instrument universe (Appendix G) is 48 names × 2 horizons = 96 forecasts/day at full coverage; launch gate is 20% coverage = ~19 forecasts/day to stay under the $300/month ceiling with headroom for weekly capability re-eval.

**R3. Post-cutoff data is limited.**
The cleaner the cutoff discipline, the less data is available for evaluation. Mitigation: sample efficiently, focus on volatile periods (where forecasting quality differences matter most), accept that cutoff dates limit test set size and that's the cost of honesty.

**R4. Reasoning traces could be misleading rationalizations.**
LLMs are good at producing plausible-sounding reasoning after the fact. The trace might describe drivers that didn't actually inform the prediction. Mitigation: Phase 2's synthesis agent grounds reasoning in the actual signal values from stages 1-3 via I9 + the grounding check; Phase 5 evaluation includes manual review of trace quality.

**R5. Adapters drift apart over time as asset classes are tuned independently.**
The core orchestration might not stay clean as adapters add their own quirks. Mitigation: enforce I12 (adapters only override stages, never modify the orchestrator). Quarterly refactor passes to extract common patterns back to core.

**R6. Forecast quality might be lower for less-data-rich asset classes.**
Crypto and FX have less structured fundamental data than equities. Their forecasts might be inherently noisier. Mitigation: per-asset-class capability thresholds calibrated realistically, not aspirationally. Crypto's directional accuracy bar is 0.52, not 0.60.

**R7. The synthesis stage might over-weight whichever signal is most verbose.**
If contextual integrator returns a 3-page analysis and macro returns a 3-line summary, synthesis might weight contextual disproportionately. Mitigation: structured signal handoffs (dataclasses, not free text) prevent this. The synthesis prompt explicitly weights by signal confidence, not signal length (Section 14 anti-rationalization clause 2).

**R8. Council of Investors integration might require Nexus refactoring that wasn't anticipated.**
Phase 6 defines a contract but real Council development might reveal contract gaps. Mitigation: treat Phase 6 contract as v1, expect a v2 when Council actually lands. I12 prohibits backwards-compat shims; v2 will be a clean cut.

**R9. Forecasts feeding strategies might inflate expected returns through hindsight in backtests.**
If forecasts are generated using LLM context that includes future information through any path, backtests lie. Mitigation: the whole point of Phase 5 post-cutoff discipline. Apply the same rigor to strategy backtests that consume forecasts.

**R10. The system might confidently produce a wrong forecast right before a crisis.**
Black swans by definition aren't in the training data. The contextual integrator might miss obvious-in-retrospect signals. Mitigation: confidence scores should drop during regime transitions (Phase 2 + I10 capability test asserts this). Strategies built on Nexus forecasts must have explicit drawdown limits and not depend on forecast accuracy for risk control.

**R11. Provider lock-in to Anthropic.**
The entire pipeline assumes Claude family models. If Anthropic raises prices or changes terms, Sentinel's economics shift. Mitigation: the `sai/llm_router.py` abstraction allows swapping models, but all post-cutoff evaluation must be re-run before any swap goes live. A swap is a versioned event, not a hot-reload.

**R12. SoulSeal evidence chain becomes a hot path bottleneck.**
Every forecast appends. If chain integrity verification scales O(N) on read, query latencies climb. Mitigation: hash-chain pages every K entries with a Merkle-root summary; verify the page, not the whole chain, on reads. Targeted Phase 4 work if telemetry shows the regression.

**R13. The four-stage decomposition encourages premature commitment to a single architecture.**
We might miss a better orchestration pattern (e.g., five stages, or a graph rather than a pipeline) because we anchored on the paper. Mitigation: Phase 8 records the monolithic head-to-head (R1) but also reserves space in the model card for "alternative architectures considered". Annual architectural review explicit.

**R14. Operator burnout — 21 engineering days is real.**
Solo operator, deep work, complex domain, two weekend work days scheduled (Phase 6 on Sat 06-06 and Phase 7 day-1 on Sun 06-07). Mitigation: the timeline (Section 21) keeps 3 buffer days inside the 28-day window; phase gates are pause points; the weekend days have a documented slide-to-weekday alternative; nothing forces a single sustained sprint.

**R15. Compliance creep — model card and audit infrastructure get heavier than needed.**
We build MRM-style docs (Section 17) for a solo trader. Mitigation: document once, regenerate from code where possible; the model card is auto-fillable from Phase 8 outputs.

---

## 27. Open questions and decisions log

Decisions to make before Phase 2 begins, plus open items to revisit during/after.

### Pre-Phase 2 decisions (must resolve)

#### Q1. Single Nexus forecaster or per-instrument Nexus forecaster?

- **Option A:** one `NexusForecaster` class that handles all instruments, with adapters providing instrument-specific logic. Cleaner code, easier maintenance.
- **Option B:** per-instrument fine-tuned instances with instrument-specific learned weights. More accurate potentially, much more complex.
- **Recommendation:** Option A. Start with single forecaster + adapters. Per-instrument tuning becomes a Phase 9 if and only if validation shows it's needed.
- **Status:** Open — defaulting to A unless operator pushes back before 2026-05-22.

#### Q2. Real-time forecasting or scheduled batches?

- **Option A:** forecasts run on-demand when requested (real-time but slower per-request).
- **Option B:** forecasts are pre-computed daily for a configured instrument universe and cached.
- **Recommendation:** Option B for production use (Sentinel's strategy lab consumes pre-computed daily forecasts), Option A available for ad-hoc analysis. Start with Option B; add Option A as a thin wrapper when needed.
- **Status:** Open — defaulting to B unless operator pushes back before 2026-05-22.

### Open during execution

#### Q3. How aggressive should the grounding check be?

Strict mode rejects any driver whose `citation` doesn't point to a field with non-default value. Permissive mode allows drivers naming `null` fields if the trace explicitly notes "absence of signal X is itself the driver".

- **Lean:** Strict in Phase 4; allow operator override per-adapter only after Phase 5 capability runs show no quality regression.

#### Q4. Should refused forecasts be persisted as full traces?

A refusal still carries information — what was stale, why was confidence low. But persisting refusals at the same density as completed forecasts inflates storage.

- **Lean:** Persist refusal stubs (refusal reason, instrument, as_of, lineage summary) but not full signal dumps. Decide post-Phase 4.

#### Q5. Per-asset-class model routing?

Should the synthesis model differ per asset class (Opus for fixed income because it's hardest; Haiku for crypto because it's noisy anyway)?

- **Lean:** Defer to post-Phase 8 cost-quality data. Default Opus everywhere for Phase 1–8; route per-class only after evidence.

### Decisions made (log)

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-05-18 | Use Pydantic for all handoff schemas (not stdlib dataclass) | Validators inline; JSON serialization built-in; existing Sentinel pattern |
| 2026-05-18 | Conservative cutoff buffer = 60 days | Industry-standard estimate of post-cutoff residual leakage via web caches |
| 2026-05-18 | Five asset classes, no commodities-grain split | Phase 3 explicit; sub-class specialization comes in Phase 9 if needed |
| 2026-05-18 | Five test points minimum per (asset_class, horizon) stratified by regime | Below this the test isn't statistically informative |
| 2026-05-18 | SoulSeal pages every 1000 entries with Merkle root | R12 pre-mitigation; matches Wave 33 evidence-chain pattern |

---

## 28. FAQ

**Q: Why is this called "Nexus-style" if it isn't Nexus?**
The Nexus paper introduced the four-stage decomposition pattern with reasoning traces. We harvest the pattern (Phase 1) and re-implement natively in Sentinel. "Nexus-style" credits the architectural insight without implying dependency, fork, or vendor relationship.

**Q: Does this require fine-tuning?**
No. Everything is inference-time. The Nexus paper's "stronger intrinsic forecasting" claim is about agentic decomposition unlocking latent capability in frontier LLMs, not about training. I1 explicitly forbids training-time dependencies.

**Q: How is this different from just calling Claude with a long prompt?**
Section 14 + Phase 8 directly answer this. The four-stage decomposition imposes structure: signals are dataclasses, not free text; reasoning is grounded to signal fields (I9); each stage has its own refusal triggers; the trace is queryable. Phase 8's monolithic-baseline comparison empirically tests whether structure pays.

**Q: What if I just want the forecast number without the trace?**
You always get the trace whether you read it or not. The trace lives in the evidence chain — the `forecast_id` returned to the caller is the key. The `ForecastResult` JSON is small; the trace is large but lazy-loaded. There is no path that returns a prediction without a corresponding trace persisting (I3).

**Q: How does this interact with the existing walk-forward backtester?**
Strategies built on top of forecasts consume them via `--forecast-input <file-or-table>`. The walk-forward iterates over `as_of` dates; for each date, it loads the corresponding forecast (already generated, not generated on the fly). PBO/DSR gates from Wave 34 operate on the strategy returns as before. Section 13.4 covers the end-to-end.

**Q: Can I plug in a non-Claude model?**
The `sai/llm_router.py` abstraction allows it; `llm_cutoffs.json` knows the conservative cutoff for each registered model. But every model swap is a versioned event that triggers test-set regeneration and capability-test re-run before live serving (Section 20 + R11). There is no quick "try OpenAI today" path.

**Q: What about Tier-1 (T1) attribution requirements?**
Every input is tagged at ingest with `SourceLineageTier`. T1 sources (FRED, SEC EDGAR, public government data) have no attribution requirements. T2 (commercial-with-attribution) sources are attributed in the trace's `inputs_used` list with source name and excerpt where licensed. T3 (ToS-restricted) sources may inform reasoning internally but their text never leaks into external-facing outputs (Section 17).

**Q: How do I tell if a forecast is trustworthy?**
Three signals: (1) `ForecastResult.refused == False` and `confidence >= 0.4` and the adapter's `status != degraded`. (2) The trace's `signal_disagreement` is empty or low. (3) The most recent capability test for that adapter is green. The `sentinel forecast adapters status` command surfaces all three at once.

**Q: What happens to forecasts when an LLM cutoff updates?**
The model swap is a versioned event. Old forecasts retain their old `cost_metadata.model_used`; the cutoff registry tracks per-model cutoffs separately. New forecasts use the new cutoff for test-set construction. There's no rewriting of historical traces.

**Q: Does this work with options pricing forecasts?**
v1 is direction + return + range. Options-specific outputs (delta, gamma, IV surface forecasts) are a Phase 9 candidate. The handoff schemas have room for it (the `prediction_distribution` field can carry a parametric form), but the synthesis prompts aren't tuned for it yet.

**Q: Can the Council layer override a Nexus refusal?**
No. If Nexus refuses, no forecast was produced. The Council layer can produce its own opinion *without* a Nexus input — but it cannot fabricate a Nexus output. I11 enforces this.

**Q: What's the cost per forecast in practice?**
Phase 7 budgets cap at 50K–100K tokens per forecast. At Anthropic Opus 4.7 list pricing (~$15/M input, ~$75/M output, with caching reducing the input cost meaningfully), a typical forecast lands around $0.30–$1.50. A daily run on 40 instruments × 2 horizons is ~$25–$120/day. Steady-state target is ≤ $300/month, which constrains the universe and triggers model downgrades.

---

## 29. References

### Primary

- **Nexus paper** — the agentic forecasting decomposition + post-cutoff evaluation source. Cited for attribution in harvest notes; never copied. Internal harvest notes at `docs/harvest-notes/nexus/*.md`.

### Sentinel internal

- `Docs/SENTINEL_PRD_v2.0.md` — founding PRD; this document extends it
- `Docs/SENTINEL_TSD_v1.0.md` — technical-spec doc; module conventions
- `Docs/SENTINEL_COMPETITIVE_MATRIX_v3.0.md` — competitive position context
- `Docs/SENTINEL_LEAPFROG_PLAYBOOK.md` — frontier capability priorities
- Wave 34 commit `4456541` — PBO/DSR detection used in Phase 8.4
- Wave 35 commit `dd8799e` — vol estimators used by Stage 2 Micro Isolator

### External methods (concept-only, no code dependencies)

- **MASE** — Hyndman & Koehler (2006), "Another look at measures of forecast accuracy"
- **PBO / DSR** — Bailey, Borwein, López de Prado (2014), "The Probability of Backtest Overfitting"; López de Prado, "The Deflated Sharpe Ratio"
- **Brier score** — Brier (1950), "Verification of Forecasts Expressed in Terms of Probability"
- **Hidden Markov regime detection** — Rabiner (1989), tutorial on HMMs
- **Garman-Klass / Rogers-Satchell / Yang-Zhang** — classical OHLC realized-vol estimators

### Tooling

- `hmmlearn` — regime HMM (MIT)
- `pgvector` — semantic indexing on Postgres (PostgreSQL license)
- `pydantic` — schema validation (MIT)
- Anthropic Claude (Opus 4.7, Sonnet 4.6, Haiku 4.5) — LLM substrate, via Anthropic SDK
- `keyring` — OS credential storage (Python Software Foundation license)

---

## 30. What this enables

After this PRD ships:

**Sentinel produces auditable forecasts.** Every prediction comes with a structured reasoning trace stored in the evidence chain. The trace records which signals were dominant, where stages disagreed, what was considered and rejected. This is what distinguishes Sentinel from black-box LLM forecasting tools and is what would let it serve regulated buyers eventually.

**Post-cutoff evaluation is the substrate gate.** No forecasting capability can claim quality without passing post-cutoff capability tests. This applies the substrate fix discipline (no infrastructure theater) specifically to forecasting. A forecaster that fails its capability test does not get to ship predictions to downstream consumers.

**The Council of Investors has a clean input.** When Council work lands, advisor agents have a forecasting layer to consume. Dalio-agent doesn't have to invent its own forecasting; it consumes Nexus output and applies its philosophical lens. Buffett-agent does the same. This compositional structure is what makes the Council buildable as a layer rather than a from-scratch project.

**Strategy lab generates strategies with forecast inputs.** The existing walk-forward + PBO/DSR infrastructure now operates on forecasts grounded in structured reasoning. A strategy is no longer "if RSI < 30, buy" — it's "if Nexus forecasts upward with confidence > 0.7 AND contextual signal weights central bank dovishness > 0.6, buy a defined-risk options structure." More interesting strategies, with the same backtest rigor.

**Sentinel becomes a forecasting platform, not just a data platform.** The shift from "Bloomberg-class data infrastructure" to "Bloomberg-class research infrastructure" is the bigger product positioning. Data is commoditizing. Reasoning over data with full auditability is not. This PRD is what makes Sentinel a research platform in the way that matters.

**Honest about what it can't do.** Capped adapters (where post-cutoff evaluation fails) get marked honestly. Forecasts during regime transitions explicitly carry low confidence. Forecast quality and strategy profitability are reported separately. The system stops pretending to know things it doesn't know.

**The competitive leapfrog story has a chapter.** SENTINEL's competitive matrix gains a dimension — *auditable agentic forecasting with post-cutoff evidence* — that none of the data incumbents (Bloomberg, FactSet, AlphaSense, CapIQ) currently ship. Closing this dimension to 9/10 is what `/compete` and `/ascend` will measure after R5.

---

## Appendix A — Worked reasoning trace (SPY 30d as_of 2026-04-01)

A hand-authored illustrative trace showing the structure. Not a real prediction; a structural template for Phase 4 review.

```json
{
  "schema_version": "nexus-trace/1.0",
  "forecast_id": "fcst_2026-04-01_SPY_30d_a1b2c3",
  "instrument": {
    "symbol": "SPY",
    "asset_class": "equity",
    "venue": "NYSEARCA",
    "figi": "BBG000BDTBL9"
  },
  "as_of": "2026-04-01T00:00:00Z",
  "horizon": "30d",

  "macro_signal": {
    "schema_version": "nexus-macro-signal/1.0",
    "regime": "risk_on",
    "trend_direction": "up",
    "trend_strength": 0.62,
    "seasonality": ["Q2_earnings_lead"],
    "cycle_position": "mid",
    "macro_correlations": {"DXY": -0.31, "MOVE": -0.42, "10y_real_yield": -0.55},
    "confidence": 0.68,
    "source_lineage": [
      {"source_id": "FRED:DGS10", "source_human_name": "FRED 10y Treasury", "lineage_tier": "T1_PUBLIC", "timestamp": "2026-03-31T22:00:00Z"},
      {"source_id": "FRED:T10YIE", "source_human_name": "FRED 10y Breakeven", "lineage_tier": "T1_PUBLIC", "timestamp": "2026-03-31T22:00:00Z"},
      {"source_id": "CFTC:SP500_COT", "source_human_name": "CFTC SP500 COT", "lineage_tier": "T1_PUBLIC", "timestamp": "2026-03-29T00:00:00Z"}
    ]
  },

  "micro_signal": {
    "schema_version": "nexus-micro-signal/1.0",
    "recent_volatility": {"7d_gk": 0.11, "30d_yz": 0.13},
    "volatility_regime": "low",
    "momentum": 0.42,
    "momentum_strength": 0.55,
    "mean_reversion_signal": -0.18,
    "liquidity_state": {"bid_ask_proxy": 0.0002, "amihud": 0.0001},
    "microstructure_anomalies": [],
    "confidence": 0.74,
    "source_lineage": [
      {"source_id": "SFE:PIT:SPY", "source_human_name": "Sentinel PIT OHLCV", "lineage_tier": "T4_SYNTHETIC_DERIVED", "timestamp": "2026-03-31T20:00:00Z"}
    ]
  },

  "contextual_signal": {
    "schema_version": "nexus-contextual-signal/1.0",
    "news_summary": "Mixed Q1 earnings preview; semis upbeat, financials cautious on net interest margin compression.",
    "news_sentiment": {"score": 0.18, "volume": 412, "top_sources": []},
    "pending_events": [
      {"event_id": "FOMC_2026_05", "event_name": "FOMC meeting", "scheduled_at": "2026-05-07T18:00:00Z", "expected_impact": "high", "source": {"source_id": "FRED_CAL:FOMC", "source_human_name": "FOMC Calendar", "lineage_tier": "T1_PUBLIC", "timestamp": "2026-03-15T00:00:00Z"}}
    ],
    "central_bank_stance": "neutral",
    "relevant_filings": [],
    "narrative_themes": ["AI capex steady", "regional bank stabilization", "energy upcycle moderate"],
    "event_risk": ["Q1 earnings dispersion"],
    "confidence": 0.61,
    "source_lineage": [
      {"source_id": "SMA:NEWS:SPY_window_30d", "source_human_name": "Sentinel news aggregate", "lineage_tier": "T2_COMMERCIAL_WITH_ATTRIBUTION", "timestamp": "2026-03-31T23:50:00Z"}
    ]
  },

  "dominant_signal_type": "balanced",
  "signal_weights": {"macro": 0.40, "micro": 0.25, "contextual": 0.35},

  "summary": "SPY 30d forecast: mildly bullish with low vol; macro regime risk-on, micro momentum positive, contextual mixed with FOMC event risk inside horizon. Direction up probability 0.58, expected return 1.4% with 80% CI of [-2.1%, 4.9%].",

  "fundamental_drivers": [
    {"name": "Risk-on macro regime", "direction": "bullish", "strength": 0.55, "citation": "macro_signal.regime", "citation_payload": "risk_on"},
    {"name": "Positive short-term momentum", "direction": "bullish", "strength": 0.42, "citation": "micro_signal.momentum", "citation_payload": "0.42"},
    {"name": "FOMC event risk inside horizon", "direction": "neutral", "strength": 0.30, "citation": "contextual_signal.pending_events[0]", "citation_payload": "FOMC 2026-05-07"}
  ],

  "contradicting_evidence": [
    {"summary": "Mild mean-reversion signal in micro stage", "direction_implied": "bearish", "citation": "micro_signal.mean_reversion_signal", "why_dismissed": "Magnitude too small (-0.18) to overturn momentum and macro alignment"}
  ],

  "key_uncertainties": [
    {"description": "FOMC outcome 2026-05-07 — hawkish surprise could flip macro regime", "direction_if_resolved_against": "bearish", "monitoring_signal": "macro_signal.regime"}
  ],

  "inputs_used": [
    {"source_id": "FRED:DGS10", "source_human_name": "FRED 10y Treasury", "lineage_tier": "T1_PUBLIC", "timestamp": "2026-03-31T22:00:00Z"},
    {"source_id": "FRED:T10YIE", "source_human_name": "FRED 10y Breakeven", "lineage_tier": "T1_PUBLIC", "timestamp": "2026-03-31T22:00:00Z"},
    {"source_id": "CFTC:SP500_COT", "source_human_name": "CFTC SP500 COT", "lineage_tier": "T1_PUBLIC", "timestamp": "2026-03-29T00:00:00Z"},
    {"source_id": "SFE:PIT:SPY", "source_human_name": "Sentinel PIT OHLCV", "lineage_tier": "T4_SYNTHETIC_DERIVED", "timestamp": "2026-03-31T20:00:00Z"},
    {"source_id": "SMA:NEWS:SPY_window_30d", "source_human_name": "Sentinel news aggregate", "lineage_tier": "T2_COMMERCIAL_WITH_ATTRIBUTION", "timestamp": "2026-03-31T23:50:00Z"}
  ],
  "inputs_excluded": [],
  "source_lineage_summary": {
    "T1_PUBLIC": 3,
    "T2_COMMERCIAL_WITH_ATTRIBUTION": 1,
    "T3_TOS_RESTRICTED": 0,
    "T4_SYNTHETIC_DERIVED": 1
  },

  "cost_metadata": {
    "schema_version": "nexus-cost/1.0",
    "model_used": "claude-opus-4-7",
    "tokens_input": 42100,
    "tokens_output": 3850,
    "tokens_cache_read": 28600,
    "tokens_cache_write": 1200,
    "wall_clock_seconds": 27.4,
    "estimated_usd": 0.84
  },
  "soulseal_artifact_id": "ss_2026-04-01_fcst_a1b2c3_def456",
  "generated_at": "2026-04-01T00:01:32Z",
  "grounding_check_passed": true
}
```

---

## Appendix B — Worked forecast output JSON

```json
{
  "schema_version": "nexus-forecast/1.0",
  "forecast_id": "fcst_2026-04-01_SPY_30d_a1b2c3",
  "instrument": {"symbol": "SPY", "asset_class": "equity", "venue": "NYSEARCA"},
  "as_of": "2026-04-01T00:00:00Z",
  "horizon": "30d",

  "refused": false,
  "refusal_reason": null,
  "refusal_detail": null,

  "prediction": 0.014,
  "prediction_distribution": {
    "quantiles": {"q05": -0.034, "q25": -0.008, "q50": 0.014, "q75": 0.032, "q95": 0.061},
    "parametric_form": "skew_normal",
    "parameters": {"loc": 0.011, "scale": 0.028, "skew": 0.22}
  },
  "direction_probability": {"up": 0.58, "down": 0.30, "sideways": 0.12},
  "confidence_interval_lower": -0.021,
  "confidence_interval_upper": 0.049,
  "confidence_interval_level": 0.80,

  "dominant_signals": ["macro", "contextual"],
  "signal_disagreement": "Micro mean-reversion mildly bearish but dominated by momentum.",

  "reasoning_trace_id": "fcst_2026-04-01_SPY_30d_a1b2c3",
  "cost_metadata": {
    "schema_version": "nexus-cost/1.0",
    "model_used": "claude-opus-4-7",
    "tokens_input": 42100,
    "tokens_output": 3850,
    "tokens_cache_read": 28600,
    "tokens_cache_write": 1200,
    "wall_clock_seconds": 27.4,
    "estimated_usd": 0.84
  }
}
```

---

## Appendix C — Sample post-cutoff test set construction

For `claude-opus-4-7` (conservative cutoff `2025-12-01`, evaluation start `2026-01-30`) and the **equities adapter at 30-day horizon**:

1. Pull SPY, QQQ, IWM, and 47 SP500 constituents — daily close — for `[2026-01-30, today−30d]`.
2. Bucket dates into three regime strata using `sma/regime_detector_hmm` on the same window: `calm`, `transition`, `stressed`. Aim for at least 12 test points per stratum.
3. For each stratum, sample uniformly without replacement until reaching the per-stratum target (50 total per `(instrument, horizon)`).
4. For each test point `(symbol, as_of_date)`, define `ground_truth_outcome = close[as_of_date + 30b] / close[as_of_date] - 1` where `30b` is 30 business days.
5. Persist the test set as:
   ```
   data/forecasts/eval/claude-opus-4-7/equity_30d/ts_2026-05-15_v1.json
   ```
6. The persisted file is immutable. Re-running the same model against the same test set must produce identical inputs; only the predictions change as the model evolves.

Statistical-validity floor: directional accuracy with N=50 has a 95% CI half-width of ~0.14 around 0.55. To call a 0.58 directional-accuracy result robustly above 0.50 you need N ≥ ~200 at that effect size. The 50-point Phase 5 floor is informative but not conclusive; Phase 8 ramps to N=200 per adapter where feasible.

---

## Appendix D — Prompt evaluation rubric

For manual review of trace quality in Phase 5 / 8 / weekly capability re-runs. Score each trace on a 0–2 scale per dimension; ship-blocker if any dimension averages < 1.0 across the sample.

| Dimension | 0 = fail | 1 = pass | 2 = strong |
|-----------|----------|----------|------------|
| **Grounding** | Names a driver not present in any signal | All drivers traceable to fields | Drivers cite specific field values, not just field names |
| **Signal balance** | Synthesis ignored one stage entirely | All three stages referenced | Signal weights match stated confidence levels |
| **Uncertainty honesty** | Confidence high despite contradicting evidence | Confidence reflects signal-disagreement | Key uncertainties named with monitoring signals |
| **Lineage completeness** | Source lineage missing for a stage | All inputs lineage-tagged | T-tier mix documented in `source_lineage_summary` |
| **Refusal discipline** | Forecast emitted despite stale inputs | Refuses on stale inputs | Refuses with structured reason + monitoring signal for clearing the block |
| **Cost discipline** | Token spend outside per-asset cap | Token spend within cap | Cache hit rate > 50%, downgrade events handled cleanly |

Aggregate across N traces; report mean per dimension and worst-trace per dimension.

---

## Appendix E — Capability test reference card

Quick-lookup of which `.danteforge/capability-tests/` script proves which claim.

| Claim | Capability test | Threshold | Owner phase |
|-------|------------------|-----------|-------------|
| Macro stage detects known regimes | `nexus_macro_stage.sh` | 100% on labeled fixtures | Phase 2 |
| Micro stage classifies vol regimes | `nexus_micro_stage.sh` | 100% on labeled fixtures | Phase 2 |
| Contextual stage extracts pending events | `nexus_contextual_stage.sh` | 100% on labeled fixtures | Phase 2 |
| Synthesis is grounded (I9) | `nexus_grounding.sh` | 100% pass on random 1000 traces | Phase 4 |
| Equities post-cutoff forecast quality | `equities_forecasting.sh` | MASE < 1.0, dir-acc > 0.55, calib > 0.80 | Phase 5 |
| FX post-cutoff forecast quality | `fx_forecasting.sh` | MASE < 1.1, dir-acc > 0.53, calib > 0.75 | Phase 5 |
| FI post-cutoff forecast quality | `fixed_income_forecasting.sh` | MASE < 0.95, dir-acc > 0.58 | Phase 5 |
| Commodities post-cutoff forecast quality | `commodities_forecasting.sh` | MASE < 1.0, dir-acc > 0.55 | Phase 5 |
| Crypto post-cutoff forecast quality | `crypto_forecasting.sh` | MASE < 1.2, dir-acc > 0.52 | Phase 5 |
| Decomposition beats monolithic (I8) | `nexus_vs_monolithic.sh` | Wins on ≥ 3/5 asset classes | Phase 8 |
| Trace integrity | `nexus_chain_integrity.sh` | 100% on 1000-trace sample | Phase 4 + Phase 8 |
| Cost ceiling honored | `nexus_cost_ceiling.sh` | Monthly USD ≤ $300 in staging workload | Phase 7 + Phase 8 |
| Strategy lab end-to-end | `nexus_to_strategy_e2e.sh` | PBO ≤ 0.50, DSR > 0 on illustrative strategy | Phase 8 |
| Refusal discipline (stale inputs) | `nexus_refusal_stale.sh` | 100% refusal on stale fixture | Phase 7 |
| Refusal discipline (budget exhausted) | `nexus_refusal_budget.sh` | 100% refusal at 100% budget | Phase 7 |

Capability tests live in `.danteforge/capability-tests/` and are wired into `make check` once their respective phase ships.

---

## Appendix F — Internal Sentinel cross-references

Where this PRD intersects existing Sentinel documents:

| This PRD section | Cross-reference |
|------------------|-----------------|
| Section 1 prerequisites | `Docs/SENTINEL_PRD_v2.0.md` §"Gen 1 Build Audit" |
| Section 3 module catalog | `Docs/SENTINEL_TSD_v1.0.md` module-by-module spec |
| Section 13.4 strategy lab integration | `Docs/SENTINEL_PRD_v2.0.md` strategy lab section; Wave 34 PBO/DSR commit `4456541` |
| Section 22 MCP tools | `Docs/SENTINEL_PRD_v2.0.md` MCP surface registry |
| Section 26 competitive risks | `Docs/SENTINEL_COMPETITIVE_MATRIX_v3.0.md` |
| Section 30 competitive leapfrog | `Docs/SENTINEL_LEAPFROG_PLAYBOOK.md` |

When this PRD ships, append an entry to `Docs/SENTINEL_PRD_v2.0.md` change-log:

> **2026-06-15** — Nexus-style agentic forecasting framework v1.0 added. See `Docs/NexusAgenticForecasting.md`. Adds forecasting-capability competitive dimension; raises Sentinel composite from 7.9 to projected 8.4 pending Phase 8 substrate-gate evidence.

---

## Appendix G — Instrument universe roster

Concrete list of instruments under daily-batch forecast coverage. The list is sized to fit the $300/month cost envelope (§12) after the coverage gate (`nexus.coverage_pct`, initial 0.20) and the Sonnet-default model routing (§16). Operator-overridable via `.danteforge/config/nexus-adapters.json`.

### Equities (28 instruments × 2 horizons = 56 daily forecasts at full coverage)

| Group | Symbols |
|-------|---------|
| Index ETFs | SPY, QQQ, IWM, DIA, EFA, EEM |
| US sector ETFs | XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, XLB, XLRE, XLC |
| Mega-cap single names | AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, AVGO |
| Banks / financials | JPM, BAC, GS |

Horizons: `5d`, `30d`.

### FX majors (7 pairs × 2 horizons = 14 daily forecasts)

| Group | Pairs |
|-------|-------|
| G10 majors | EURUSD, USDJPY, GBPUSD, USDCHF, AUDUSD, USDCAD, NZDUSD |

Horizons: `5d`, `30d`.

### Fixed income (5 instruments × 2 horizons = 10 daily forecasts)

| Group | Symbols |
|-------|---------|
| Treasury ETFs (duration ladder) | TLT (20Y+), IEF (7–10Y), SHY (1–3Y) |
| Credit ETFs | HYG (high-yield), LQD (investment-grade) |

Horizons: `30d`, `90d` (FI moves slower; short horizons add noise).

### Commodities (5 contracts × 2 horizons = 10 daily forecasts)

| Group | Symbols |
|-------|---------|
| Energy | CL (WTI crude), NG (Henry Hub natural gas) |
| Metals | GC (gold), HG (copper) |
| Ags | ZW (wheat) |

Horizons: `5d`, `30d`.

### Crypto (3 instruments × 2 horizons = 6 daily forecasts)

| Group | Symbols |
|-------|---------|
| L1 tokens | BTC, ETH, SOL |

Horizons: `1d`, `5d` (crypto regimes shift on hours-to-days timescales).

### Full-coverage daily steady state

| Asset class | Instruments | Horizons | Daily forecasts |
|-------------|-------------|----------|------------------|
| Equities | 28 | 2 | 56 |
| FX | 7 | 2 | 14 |
| Fixed income | 5 | 2 | 10 |
| Commodities | 5 | 2 | 10 |
| Crypto | 3 | 2 | 6 |
| **Total** | **48** | **—** | **96** |

### Cost envelope per coverage level

| Coverage % | Forecasts/day | Sonnet default (est.) | Opus default (est.) | Monthly USD (Sonnet) |
|------------|----------------|------------------------|----------------------|----------------------|
| 100 % | 96 | ~$0.35/fcst × 96 = $34/day | ~$1.00/fcst × 96 = $96/day | ~$1 020 |
| 50 % | 48 | $17/day | $48/day | ~$510 |
| 20 % | 19 | $7/day | $19/day | **~$200** |
| 10 % | 10 | $3/day | $10/day | ~$100 |

**Launch coverage:** 20 % (~19 forecasts/day, ~$200/month at Sonnet routing). Inside the $300/month ceiling with headroom for weekly capability re-eval (~$30 / month at batch-API discount) and ad-hoc operator forecasts.

**Coverage ramp plan:**

| Month | Coverage % | Forecasts/day | Trigger to advance |
|-------|-----------|----------------|---------------------|
| 0 (launch) | 20 % | ~19 | n/a |
| +1 | 30 % | ~29 | Month-1 budget within target; ≥ 4/5 adapters green |
| +3 | 50 % | ~48 | Month-3 budget within target; calibration drift < 5 pp |
| +6 | 100 % | 96 | Sustained green; or operator override |

### Stratified sampling at sub-full coverage

When `nexus.coverage_pct < 1.0`, the daily batch picks a stratified random sample so every asset class is touched at least once per day. Algorithm in `sentinel/sai/nexus/scheduler/coverage_sampler.py`:

1. Compute target per-class daily count = ceil(class_size × 2_horizons × coverage_pct).
2. From each class roster, sample without replacement until the target is hit.
3. Always include the 5 "always-on" instruments regardless of sample (operator-configurable; default SPY, EURUSD, TLT, CL, BTC — one per asset class).
4. The previous day's sample is stored; the next day's sample biases away from yesterday's set to ensure roster coverage over a rolling 7-day window.

### Capability-test universe (Section 10 stratified subsets)

Per-adapter capability tests use a fixed, smaller subset of the full roster — large enough to be statistically informative, small enough to fit the eval budget:

| Adapter | Instruments | Test points per (instrument, horizon) | Total test forecasts |
|---------|-------------|----------------------------------------|-----------------------|
| Equities | SPY, QQQ, AAPL, JPM, NVDA | 50 | 5 × 2 × 50 = 500 |
| FX | EURUSD, USDJPY, GBPUSD | 50 | 3 × 2 × 50 = 300 |
| Fixed income | TLT, IEF, HYG | 50 | 3 × 2 × 50 = 300 |
| Commodities | CL, GC | 50 | 2 × 2 × 50 = 200 |
| Crypto | BTC, ETH | 50 | 2 × 2 × 50 = 200 |
| **Total** | **15** | — | **1 500 forecasts per full re-eval** |

At Sonnet routing with batch-API discount, a full capability re-eval costs ~$260 and runs Sunday 02:00–06:00 local. Within budget; outside live-forecast windows.

### Always-on canary subset

Five instruments — SPY, EURUSD, TLT, CL, BTC — are forecast every single day regardless of coverage gate. Their traces feed a separate "canary" dashboard that flags drift faster than the weekly capability test would. Canary trace drift > 2σ from the 30-day rolling mean opens runbook 18.2.

---

**End of PRD.**

Paste this PRD to Claude Code working on Sentinel as `docs/PRDs/nexus-forecasting-framework.md` (or `Docs/NexusAgenticForecasting.md` as currently placed). The work belongs in Sentinel. The four-stage decomposition turns existing modules into a coherent forecasting capability, with post-cutoff evaluation as the substrate gate that keeps the claims honest.

The deeper bet: Sentinel's competitive position isn't faster data or more sources. It's reasoning that's both numerically competitive and explanatorily auditable, at solo-operator scale, with full evidence chain. Nexus is the architectural pattern that delivers this. Build it organically, validate post-cutoff, ship it into the strategy lab. The trading capability the substrate enables is what justifies the substrate. This is the bridge between substrate and trade.
