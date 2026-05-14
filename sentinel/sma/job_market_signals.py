from __future__ import annotations
import asyncio
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta
from typing import Optional
import pandas as pd
import httpx
from pydantic import BaseModel, ConfigDict, Field
from sentinel.core.logging import get_logger

logger = get_logger(__name__)

BLS_API = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
BLS_JOLTS_SECTORS: dict[str, str] = {
    "all": "JTS000000000000000JOR",
    "manufacturing": "JTS3100JOR",
    "retail": "JTS4400JOR",
    "information": "JTS5100JOR",
    "finance": "JTS5200JOR",
    "professional": "JTS5400JOR",
    "health_education": "JTS6100JOR",
    "leisure": "JTS7000JOR",
}
BLS_QUITS_SERIES: dict[str, str] = {
    "all": "JTS000000000000000QUR",
    "manufacturing": "JTS3100QUR",
    "retail": "JTS4400QUR",
}
CENSUS_BFS_URL = "https://www.census.gov/econ/bfs/csv/bfs_us_apps_monthly_nsa.csv"
INDEED_RSS = "https://www.indeed.com/rss?q={query}&l=&sort=date&limit=25"

SECTOR_QUERIES: dict[str, str] = {
    "information_technology": "software engineer developer",
    "consumer_staples": "store manager retail associate",
    "consumer_discretionary": "warehouse fulfillment associate",
    "financials": "financial analyst investment banking",
    "health_care": "registered nurse clinical",
    "industrials": "manufacturing engineer operations",
    "energy": "petroleum engineer field technician",
    "materials": "chemical engineer process operator",
    "real_estate": "property manager leasing agent",
    "utilities": "electrical engineer grid operator",
    "communication_services": "content creator media analyst",
}

_PERIOD_TO_MONTH: dict[str, int] = {
    f"M{i:02d}": i for i in range(1, 13)
}


class JOLTSDataPoint(BaseModel):
    model_config = ConfigDict(frozen=True)
    series_id: str
    sector: str
    period: str
    value: float


class JOLTSSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)
    as_of: date
    job_openings_rate: dict[str, float]
    quits_rate: dict[str, float]
    openings_yoy_change: dict[str, float]
    tight_labor_sectors: list[str]


class IndeedJobPosting(BaseModel):
    model_config = ConfigDict(frozen=True)
    title: str
    company: str
    location: str
    posted_date: Optional[date] = None
    description: str
    url: str
    query: str


class IndeedTrend(BaseModel):
    model_config = ConfigDict(frozen=True)
    query: str
    ticker: Optional[str] = None
    total_postings: int
    recent_30d: int
    yoy_change_pct: Optional[float] = None
    top_locations: list[str]
    top_companies: list[str]


class GoogleTrendsSignal(BaseModel):
    model_config = ConfigDict(frozen=True)
    keyword: str
    ticker: Optional[str] = None
    current_interest: float
    avg_90d: float
    trend_direction: str
    spike_today: bool
    relative_change_pct: float


class BusinessFormationSignal(BaseModel):
    model_config = ConfigDict(frozen=True)
    period: str
    total_applications: int
    high_propensity_apps: int
    yoy_change_pct: float
    sector_signal: str


class AlternativeEmploymentSignal(BaseModel):
    model_config = ConfigDict(frozen=True)
    as_of: date
    jolts: JOLTSSnapshot
    business_formation: Optional[BusinessFormationSignal] = None
    google_trends: list[GoogleTrendsSignal]
    interpretation: str


def _period_to_date(year: str, period: str) -> Optional[date]:
    month = _PERIOD_TO_MONTH.get(period)
    if month is None:
        return None
    return date(int(year), month, 1)


def _parse_pubdate(raw: str) -> Optional[date]:
    for fmt in (
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S GMT",
        "%Y-%m-%dT%H:%M:%S%z",
    ):
        try:
            return datetime.strptime(raw.strip(), fmt).date()
        except ValueError:
            continue
    return None


class JobMarketSignals:
    def __init__(self, timeout: float = 20.0, bls_api_key: str = ""):
        self._timeout = timeout
        self._bls_key = bls_api_key

    async def get_jolts_snapshot(self, n_months: int = 24) -> JOLTSSnapshot:
        n_years = max(2, (n_months // 12) + 1)
        openings_ids = list(BLS_JOLTS_SECTORS.values())
        quits_ids = list(BLS_QUITS_SERIES.values())
        all_ids = openings_ids + quits_ids

        raw = await self._bls_fetch(all_ids, n_years=n_years)
        parsed = self._parse_jolts_response(raw)

        sector_by_id = {v: k for k, v in BLS_JOLTS_SECTORS.items()}
        quits_by_id = {v: k for k, v in BLS_QUITS_SERIES.items()}

        openings_latest: dict[str, float] = {}
        openings_12m_ago: dict[str, float] = {}
        quits_latest: dict[str, float] = {}

        for sid, points in parsed.items():
            if not points:
                continue
            sorted_pts = sorted(points, key=lambda p: p.period, reverse=True)
            latest = sorted_pts[0]

            if sid in sector_by_id:
                sector = sector_by_id[sid]
                openings_latest[sector] = latest.value
                if len(sorted_pts) >= 13:
                    openings_12m_ago[sector] = sorted_pts[12].value
            elif sid in quits_by_id:
                sector = quits_by_id[sid]
                quits_latest[sector] = latest.value

        yoy: dict[str, float] = {}
        for sector, curr in openings_latest.items():
            if sector in openings_12m_ago and openings_12m_ago[sector] != 0:
                yoy[sector] = round(
                    (curr - openings_12m_ago[sector]) / openings_12m_ago[sector] * 100, 2
                )

        tight = [s for s, v in openings_latest.items() if v > 5.0]

        as_of_date = date.today()
        all_points_flat = [p for pts in parsed.values() for p in pts]
        if all_points_flat:
            try:
                most_recent = max(all_points_flat, key=lambda p: p.period)
                parsed_date = _period_to_date(
                    most_recent.period[:4], most_recent.period[5:]
                )
                if parsed_date:
                    as_of_date = parsed_date
            except Exception:
                pass

        return JOLTSSnapshot(
            as_of=as_of_date,
            job_openings_rate=openings_latest,
            quits_rate=quits_latest,
            openings_yoy_change=yoy,
            tight_labor_sectors=tight,
        )

    async def get_jolts_history(
        self, sector: str = "all", n_years: int = 5
    ) -> pd.DataFrame:
        openings_sid = BLS_JOLTS_SECTORS.get(sector, BLS_JOLTS_SECTORS["all"])
        quits_sid = BLS_QUITS_SERIES.get(sector)
        ids = [openings_sid]
        if quits_sid:
            ids.append(quits_sid)

        raw = await self._bls_fetch(ids, n_years=n_years)
        parsed = self._parse_jolts_response(raw)

        openings_pts = parsed.get(openings_sid, [])
        quits_pts = parsed.get(quits_sid or "", [])

        def pts_to_series(pts: list[JOLTSDataPoint]) -> pd.Series:
            return pd.Series(
                {p.period: p.value for p in pts}, dtype=float
            ).sort_index()

        openings_s = pts_to_series(openings_pts)
        quits_s = pts_to_series(quits_pts)

        df = pd.DataFrame({"openings_rate": openings_s, "quits_rate": quits_s})
        df.index.name = "period"
        df = df.reset_index()
        return df.dropna(subset=["openings_rate"]).reset_index(drop=True)

    async def scrape_indeed_postings(
        self, query: str, n_pages: int = 3
    ) -> list[IndeedJobPosting]:
        postings: list[IndeedJobPosting] = []
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "application/rss+xml, application/xml, text/xml, */*",
        }
        ns = {"dc": "http://purl.org/dc/elements/1.1/"}
        async with httpx.AsyncClient(
            timeout=self._timeout,
            follow_redirects=True,
            headers=headers,
        ) as client:
            for page in range(n_pages):
                start = page * 25
                url = INDEED_RSS.format(query=query.replace(" ", "+"))
                if start > 0:
                    url += f"&start={start}"
                try:
                    resp = await client.get(url)
                    resp.raise_for_status()
                    root = ET.fromstring(resp.text)
                    items = root.findall(".//item")
                    if not items:
                        break
                    for item in items:
                        title = (item.findtext("title") or "").strip()
                        link = (item.findtext("link") or "").strip()
                        pub_date_raw = (item.findtext("pubDate") or "").strip()
                        desc = (item.findtext("description") or "").strip()
                        source = item.find("source")
                        company = ""
                        if source is not None:
                            company = (source.text or "").strip()
                        location = ""
                        for tag in ("location", "georss:point", "city"):
                            val = item.findtext(tag)
                            if val:
                                location = val.strip()
                                break
                        if not location:
                            if " - " in title:
                                parts = title.rsplit(" - ", 1)
                                if len(parts) == 2:
                                    location = parts[1].strip()
                        parsed_date = _parse_pubdate(pub_date_raw) if pub_date_raw else None
                        postings.append(
                            IndeedJobPosting(
                                title=title,
                                company=company,
                                location=location,
                                posted_date=parsed_date,
                                description=desc[:500],
                                url=link,
                                query=query,
                            )
                        )
                    if len(items) < 20:
                        break
                    await asyncio.sleep(1.2)
                except Exception as exc:
                    logger.warning("indeed_rss_error", query=query, page=page, error=str(exc))
                    break
        return postings

    async def analyze_company_hiring(
        self, company: str, ticker: Optional[str] = None
    ) -> IndeedTrend:
        postings = await self.scrape_indeed_postings(query=f'"{company}"', n_pages=2)
        now = date.today()
        cutoff_30d = now - timedelta(days=30)
        recent = [p for p in postings if p.posted_date and p.posted_date >= cutoff_30d]

        location_counts: dict[str, int] = {}
        company_counts: dict[str, int] = {}
        for p in postings:
            if p.location:
                location_counts[p.location] = location_counts.get(p.location, 0) + 1
            if p.company:
                company_counts[p.company] = company_counts.get(p.company, 0) + 1

        top_locs = sorted(location_counts, key=lambda k: -location_counts[k])[:5]
        top_cos = sorted(company_counts, key=lambda k: -company_counts[k])[:5]

        return IndeedTrend(
            query=company,
            ticker=ticker,
            total_postings=len(postings),
            recent_30d=len(recent),
            yoy_change_pct=None,
            top_locations=top_locs,
            top_companies=top_cos,
        )

    async def get_sector_hiring_trends(
        self, sectors: Optional[list[str]] = None
    ) -> dict[str, IndeedTrend]:
        target_sectors = sectors if sectors else list(SECTOR_QUERIES.keys())
        tasks = {
            sector: self.scrape_indeed_postings(query=SECTOR_QUERIES.get(sector, sector), n_pages=1)
            for sector in target_sectors
            if sector in SECTOR_QUERIES
        }
        results: dict[str, IndeedTrend] = {}
        gathered = await asyncio.gather(*tasks.values(), return_exceptions=True)
        for sector, result in zip(tasks.keys(), gathered):
            if isinstance(result, Exception):
                logger.warning("sector_hiring_error", sector=sector, error=str(result))
                results[sector] = IndeedTrend(
                    query=SECTOR_QUERIES.get(sector, sector),
                    total_postings=0,
                    recent_30d=0,
                    top_locations=[],
                    top_companies=[],
                )
                continue
            postings: list[IndeedJobPosting] = result  # type: ignore[assignment]
            now = date.today()
            cutoff_30d = now - timedelta(days=30)
            recent = [p for p in postings if p.posted_date and p.posted_date >= cutoff_30d]
            loc_counts: dict[str, int] = {}
            co_counts: dict[str, int] = {}
            for p in postings:
                if p.location:
                    loc_counts[p.location] = loc_counts.get(p.location, 0) + 1
                if p.company:
                    co_counts[p.company] = co_counts.get(p.company, 0) + 1
            results[sector] = IndeedTrend(
                query=SECTOR_QUERIES.get(sector, sector),
                total_postings=len(postings),
                recent_30d=len(recent),
                top_locations=sorted(loc_counts, key=lambda k: -loc_counts[k])[:5],
                top_companies=sorted(co_counts, key=lambda k: -co_counts[k])[:5],
            )
        return results

    def get_google_trends(
        self, keywords: list[str], timeframe: str = "today 3-m"
    ) -> list[GoogleTrendsSignal]:
        try:
            from pytrends.request import TrendReq  # type: ignore[import]
        except ImportError:
            logger.warning("pytrends_not_installed")
            return []

        signals: list[GoogleTrendsSignal] = []
        pytrends = TrendReq(hl="en-US", tz=360, timeout=(10, 25))
        for kw in keywords:
            try:
                pytrends.build_payload([kw], cat=0, timeframe=timeframe, geo="US")
                df = pytrends.interest_over_time()
                if df is None or df.empty or kw not in df.columns:
                    signals.append(
                        GoogleTrendsSignal(
                            keyword=kw,
                            current_interest=0.0,
                            avg_90d=0.0,
                            trend_direction="stable",
                            spike_today=False,
                            relative_change_pct=0.0,
                        )
                    )
                    continue
                series = df[kw].dropna().astype(float)
                if series.empty:
                    signals.append(
                        GoogleTrendsSignal(
                            keyword=kw,
                            current_interest=0.0,
                            avg_90d=0.0,
                            trend_direction="stable",
                            spike_today=False,
                            relative_change_pct=0.0,
                        )
                    )
                    continue
                current = float(series.iloc[-1])
                avg = float(series.mean())
                spike = avg > 0 and current > (2.0 * avg)
                if avg > 0:
                    rel_change = (current - avg) / avg * 100.0
                else:
                    rel_change = 0.0
                if len(series) >= 4:
                    recent_avg = float(series.iloc[-4:].mean())
                    earlier_avg = float(series.iloc[:-4].mean()) if len(series) > 4 else avg
                    if earlier_avg > 0:
                        momentum = (recent_avg - earlier_avg) / earlier_avg
                        if momentum > 0.05:
                            direction = "rising"
                        elif momentum < -0.05:
                            direction = "falling"
                        else:
                            direction = "stable"
                    else:
                        direction = "stable"
                else:
                    direction = "stable"
                signals.append(
                    GoogleTrendsSignal(
                        keyword=kw,
                        current_interest=round(current, 2),
                        avg_90d=round(avg, 2),
                        trend_direction=direction,
                        spike_today=spike,
                        relative_change_pct=round(rel_change, 2),
                    )
                )
                import time as _time
                _time.sleep(0.8)
            except Exception as exc:
                logger.warning("google_trends_error", keyword=kw, error=str(exc))
        return signals

    async def get_ticker_search_trends(
        self, tickers: list[str]
    ) -> list[GoogleTrendsSignal]:
        loop = asyncio.get_event_loop()
        signals: list[GoogleTrendsSignal] = await loop.run_in_executor(
            None, lambda: self.get_google_trends(tickers, timeframe="today 3-m")
        )
        for i, sig in enumerate(signals):
            if i < len(tickers):
                signals[i] = GoogleTrendsSignal(
                    keyword=sig.keyword,
                    ticker=tickers[i],
                    current_interest=sig.current_interest,
                    avg_90d=sig.avg_90d,
                    trend_direction=sig.trend_direction,
                    spike_today=sig.spike_today,
                    relative_change_pct=sig.relative_change_pct,
                )
        return signals

    async def get_business_formation(self) -> BusinessFormationSignal:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(CENSUS_BFS_URL)
            resp.raise_for_status()

        from io import StringIO
        df = pd.read_csv(StringIO(resp.text))
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        year_col = next((c for c in df.columns if "year" in c), None)
        month_col = next((c for c in df.columns if "month" in c), None)
        total_col = next(
            (c for c in df.columns if "ba_ba" in c or "total" in c or "applications" in c),
            None,
        )
        hba_col = next(
            (c for c in df.columns if "hba" in c or "high_propensity" in c or "hpba" in c),
            None,
        )

        if year_col is None or month_col is None or total_col is None:
            col_names = list(df.columns)
            year_col = col_names[0] if len(col_names) > 0 else "year"
            month_col = col_names[1] if len(col_names) > 1 else "month"
            total_col = col_names[2] if len(col_names) > 2 else "total"
            hba_col = col_names[3] if len(col_names) > 3 else None

        df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
        df[month_col] = pd.to_numeric(df[month_col], errors="coerce")
        df = df.dropna(subset=[year_col, month_col, total_col])
        df = df.sort_values([year_col, month_col], ascending=True)

        latest = df.iloc[-1]
        year = int(latest[year_col])
        month = int(latest[month_col])
        period = f"{year}-{month:02d}"
        total = int(pd.to_numeric(latest[total_col], errors="coerce") or 0)
        hba = int(pd.to_numeric(latest[hba_col], errors="coerce") or 0) if hba_col else 0

        yoy_change = 0.0
        if len(df) >= 13:
            prior_year_row = df.iloc[-13]
            prior_total = pd.to_numeric(prior_year_row[total_col], errors="coerce")
            if prior_total and prior_total != 0:
                yoy_change = round((total - float(prior_total)) / float(prior_total) * 100, 2)

        if yoy_change > 3.0:
            signal = "expanding"
        elif yoy_change < -3.0:
            signal = "contracting"
        else:
            signal = "stable"

        return BusinessFormationSignal(
            period=period,
            total_applications=total,
            high_propensity_apps=hba,
            yoy_change_pct=yoy_change,
            sector_signal=signal,
        )

    async def composite_employment_signal(
        self, ticker: str, company_name: str
    ) -> AlternativeEmploymentSignal:
        jolts_task = self.get_jolts_snapshot(n_months=14)
        hiring_task = self.analyze_company_hiring(company=company_name, ticker=ticker)
        bfs_task = self.get_business_formation()

        jolts, hiring, bfs = await asyncio.gather(
            jolts_task, hiring_task, bfs_task, return_exceptions=True
        )

        if isinstance(jolts, Exception):
            logger.warning("jolts_failed", error=str(jolts))
            jolts = JOLTSSnapshot(
                as_of=date.today(),
                job_openings_rate={},
                quits_rate={},
                openings_yoy_change={},
                tight_labor_sectors=[],
            )
        if isinstance(bfs, Exception):
            logger.warning("bfs_failed", error=str(bfs))
            bfs = None

        loop = asyncio.get_event_loop()
        trends_signals: list[GoogleTrendsSignal] = []
        try:
            trends_signals = await loop.run_in_executor(
                None, lambda: self.get_google_trends([ticker, company_name])
            )
        except Exception as exc:
            logger.warning("trends_failed", error=str(exc))

        parts: list[str] = []
        all_rate = jolts.job_openings_rate.get("all")  # type: ignore[union-attr]
        if all_rate is not None:
            parts.append(f"Overall job openings rate is {all_rate:.1f}%.")
        if jolts.tight_labor_sectors:  # type: ignore[union-attr]
            parts.append(f"Tight labor sectors (>5% openings rate): {', '.join(jolts.tight_labor_sectors)}.")
        if not isinstance(hiring, Exception):
            parts.append(
                f"{company_name} shows {hiring.total_postings} total postings, "
                f"{hiring.recent_30d} in the last 30 days."
            )
        if bfs and not isinstance(bfs, Exception):
            parts.append(
                f"Business formation is {bfs.sector_signal} ({bfs.yoy_change_pct:+.1f}% YoY, period {bfs.period})."
            )
        for sig in trends_signals:
            if sig.keyword == ticker:
                parts.append(
                    f"Google Trends for '{ticker}' is {sig.trend_direction} "
                    f"(current interest {sig.current_interest:.0f}, avg {sig.avg_90d:.0f})."
                )

        interpretation = " ".join(parts) if parts else "Insufficient data for interpretation."

        return AlternativeEmploymentSignal(
            as_of=date.today(),
            jolts=jolts,  # type: ignore[arg-type]
            business_formation=bfs if not isinstance(bfs, Exception) else None,  # type: ignore[arg-type]
            google_trends=trends_signals,
            interpretation=interpretation,
        )

    async def hiring_velocity_screener(
        self,
        tickers_companies: list[tuple[str, str]],
        min_postings: int = 10,
    ) -> pd.DataFrame:
        tasks = [
            self.analyze_company_hiring(company=company, ticker=ticker)
            for ticker, company in tickers_companies
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        rows: list[dict] = []
        for (ticker, company), result in zip(tickers_companies, results):
            if isinstance(result, Exception):
                logger.warning("screener_error", ticker=ticker, error=str(result))
                continue
            trend: IndeedTrend = result  # type: ignore[assignment]
            if trend.total_postings < min_postings:
                continue
            rows.append(
                {
                    "ticker": ticker,
                    "company": company,
                    "recent_postings": trend.recent_30d,
                    "total_postings": trend.total_postings,
                    "yoy_change_pct": trend.yoy_change_pct,
                    "trend_direction": (
                        "rising"
                        if trend.recent_30d > (trend.total_postings / 3)
                        else "stable"
                    ),
                    "top_location": trend.top_locations[0] if trend.top_locations else "",
                }
            )
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return df.sort_values("recent_postings", ascending=False).reset_index(drop=True)

    async def _bls_fetch(self, series_ids: list[str], n_years: int = 3) -> dict:
        end_year = date.today().year
        start_year = end_year - n_years
        payload: dict = {
            "seriesid": series_ids,
            "startyear": str(start_year),
            "endyear": str(end_year),
        }
        if self._bls_key:
            payload["registrationkey"] = self._bls_key
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(BLS_API, json=payload)
            resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "REQUEST_SUCCEEDED":
            msg = data.get("message", ["BLS API error"])
            logger.warning("bls_api_error", messages=msg)
        return data

    def _parse_jolts_response(
        self, response: dict
    ) -> dict[str, list[JOLTSDataPoint]]:
        results: dict[str, list[JOLTSDataPoint]] = {}
        sid_to_sector: dict[str, str] = {v: k for k, v in BLS_JOLTS_SECTORS.items()}
        sid_to_sector.update({v: k for k, v in BLS_QUITS_SERIES.items()})

        try:
            series_list = response["Results"]["series"]
        except (KeyError, TypeError):
            logger.warning("bls_parse_error", response_keys=list(response.keys()))
            return results

        for series_obj in series_list:
            sid = series_obj.get("seriesID", "")
            sector = sid_to_sector.get(sid, sid)
            points: list[JOLTSDataPoint] = []
            for dp in series_obj.get("data", []):
                try:
                    year = dp["year"]
                    period_code = dp["period"]
                    if not period_code.startswith("M"):
                        continue
                    value = float(dp["value"])
                    period_str = f"{year}-{period_code[1:]}"
                    points.append(
                        JOLTSDataPoint(
                            series_id=sid,
                            sector=sector,
                            period=period_str,
                            value=value,
                        )
                    )
                except (KeyError, ValueError, TypeError) as exc:
                    logger.debug("bls_dp_skip", sid=sid, error=str(exc))
            results[sid] = points
        return results


async def jolts_snapshot() -> JOLTSSnapshot:
    return await JobMarketSignals().get_jolts_snapshot()


async def company_hiring(company: str, ticker: str = "") -> IndeedTrend:
    return await JobMarketSignals().analyze_company_hiring(
        company=company, ticker=ticker or None
    )
