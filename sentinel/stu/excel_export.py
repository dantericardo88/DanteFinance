"""
Bloomberg-style Excel export — Dimension 93 (Excel plugin, 3/10 export tier).

Produces xlsxwriter workbooks with Bloomberg dark-terminal aesthetics:
dark background, green/red price colouring, monospace font, Bloomberg-named
sheets (HP, FA, SRCH, PORT, ECOS). Real data portability without live plugin.

Usage:
    buffer = export_to_excel(ticker="AAPL", ohlcv_df=df, fundamentals=facts)
    # Buffer is a seeked BytesIO ready for st.download_button or file write.
"""
from __future__ import annotations
from datetime import date
from io import BytesIO
from typing import TYPE_CHECKING, Optional
import pandas as pd
from sentinel.core.logging import get_logger

try:
    import xlsxwriter
    from xlsxwriter.workbook import Workbook
    from xlsxwriter.worksheet import Worksheet
    _XLSXWRITER_AVAILABLE = True
except ImportError:
    _XLSXWRITER_AVAILABLE = False
    # Stubs so type annotations resolve at import time even without the package
    if TYPE_CHECKING:
        from xlsxwriter.workbook import Workbook
        from xlsxwriter.worksheet import Worksheet

logger = get_logger(__name__)

THEME = {
    "bg_dark": "#0a0a0a", "bg_panel": "#1a1a1a", "bg_header": "#002b36",
    "text_primary": "#e0e0e0", "text_positive": "#00ff88", "text_negative": "#ff4444",
    "text_header": "#00ff88", "border": "#333333",
    "accent_yellow": "#ffcc00", "accent_blue": "#4db8ff",
    "font": "Courier New", "font_size": 10,
}
_W_DATE, _W_NUM, _W_TICK, _W_NAME, _W_LBL = 13, 14, 10, 28, 32
_IS = {"revenue", "sales", "grossprofit", "operatingincome", "netincome", "eps", "ebit", "ebitda"}
_BS = {"assets", "liabilities", "equity", "cash", "debt", "receivables", "goodwill", "payables"}
_CF = {"operatingactivities", "investingactivities", "financingactivities", "capitalexpend", "freecashflow"}


def _section(concept: str) -> str:
    c = concept.lower().replace("_", "").replace("-", "")
    if any(k in c for k in _IS): return "Income Statement"
    if any(k in c for k in _BS): return "Balance Sheet"
    if any(k in c for k in _CF): return "Cash Flow"
    return "Other"


class BloombergExporter:
    """Bloomberg dark-theme xlsxwriter exporter. Add sheets, then call close()."""

    def __init__(self, output: str | BytesIO | None = None) -> None:
        self.output: str | BytesIO = output if output is not None else BytesIO()
        self.workbook: Workbook = xlsxwriter.Workbook(self.output, {"in_memory": True, "strings_to_numbers": True})
        self._f: dict = {}
        self._setup_formats()

    def _setup_formats(self) -> None:
        wb, T = self.workbook, THEME
        base = {"font_name": T["font"], "font_size": T["font_size"], "font_color": T["text_primary"],
                "bg_color": T["bg_dark"], "border": 1, "border_color": T["border"]}

        def add(**kw): return wb.add_format({**base, **kw})

        self._f["hdr"] = add(bg_color=T["bg_header"], font_color=T["text_header"], bold=True, align="center", valign="vcenter")
        self._f["subhdr"] = add(bg_color=T["bg_panel"], font_color=T["accent_yellow"], bold=True)
        self._f["label"] = add(bg_color=T["bg_panel"], font_color=T["accent_blue"])
        self._f["text"] = add()
        self._f["date"] = add(num_format="yyyy-mm-dd")
        self._f["num"] = add(num_format="#,##0.00")
        self._f["int"] = add(num_format="#,##0")
        self._f["cur"] = add(num_format='$#,##0.00')
        self._f["pct"] = add(num_format="0.00%")
        self._f["pct+"] = add(num_format="0.00%", font_color=T["text_positive"])
        self._f["pct-"] = add(num_format="0.00%", font_color=T["text_negative"])
        self._f["num+"] = add(num_format="#,##0.00", font_color=T["text_positive"])
        self._f["num-"] = add(num_format="#,##0.00", font_color=T["text_negative"])
        self._f["pos"] = add(font_color=T["text_positive"])
        self._f["neg"] = add(font_color=T["text_negative"])
        self._f["gbg"] = add(bg_color="#003300", font_color=T["text_positive"])
        self._f["rbg"] = add(bg_color="#330000", font_color=T["text_negative"])
        self._f["title"] = wb.add_format({"font_name": T["font"], "font_size": 14,
                                           "font_color": T["text_header"], "bg_color": T["bg_dark"], "bold": True})

    def _hdr_row(self, ws: Worksheet, row: int, cols: list[str]) -> None:
        for c, lbl in enumerate(cols): ws.write(row, c, lbl, self._f["hdr"])

    # ── HP — Historical Prices ────────────────────────────────────────────────

    def add_ohlcv_sheet(self, ticker: str, df: pd.DataFrame) -> None:
        """HP - {ticker}: OHLCV table with green/red close colouring + stats block."""
        ws: Worksheet = self.workbook.add_worksheet(f"HP - {ticker}"[:31])
        ws.set_tab_color(THEME["bg_header"]); ws.hide_gridlines(2); ws.freeze_panes(2, 0); ws.set_zoom(90)
        ws.merge_range(0, 0, 0, 6, f"SENTINEL  HP  {ticker}  Historical Prices", self._f["title"])
        self._hdr_row(ws, 1, ["Date", "Open", "High", "Low", "Close", "Volume", "Adj Close"])
        ws.set_column(0, 0, _W_DATE); ws.set_column(1, 6, _W_NUM)

        df = df.copy(); cm = {c.lower().strip(): c for c in df.columns}
        def col(*names): return next((cm[n] for n in names if n in cm), None)

        dc, oc, hc, lc, cc, vc, ac = (col("date", "time", "datetime"), col("open"), col("high"),
                                        col("low"), col("close"), col("volume", "vol"),
                                        col("adj close", "adj_close", "adjclose"))
        closes: list[float] = []; vols: list[float] = []; highs: list[float] = []; lows: list[float] = []

        for i, (_, r) in enumerate(df.iterrows()):
            row = i + 2
            d = r[dc] if dc else ""
            if hasattr(d, "strftime"):
                ws.write_datetime(row, 0, pd.Timestamp(d).to_pydatetime(), self._f["date"])
            else:
                ws.write(row, 0, str(d), self._f["text"])
            o = float(r[oc]) if oc and pd.notna(r[oc]) else None
            h = float(r[hc]) if hc and pd.notna(r[hc]) else None
            lo = float(r[lc]) if lc and pd.notna(r[lc]) else None
            c = float(r[cc]) if cc and pd.notna(r[cc]) else None
            v = float(r[vc]) if vc and pd.notna(r[vc]) else None
            adj = float(r[ac]) if ac and pd.notna(r[ac]) else c
            ws.write(row, 1, o, self._f["num"]); ws.write(row, 2, h, self._f["num"])
            ws.write(row, 3, lo, self._f["num"])
            ws.write(row, 4, c, self._f["num+"] if (c and o and c >= o) else self._f["num-"])
            ws.write(row, 5, v, self._f["int"]); ws.write(row, 6, adj, self._f["num"])
            if c: closes.append(c)
            if h: highs.append(h)
            if lo: lows.append(lo)
            if v: vols.append(v)

        sr = len(df) + 4
        ws.write(sr, 0, "52-Wk High", self._f["subhdr"]); ws.write(sr, 1, max(highs) if highs else 0, self._f["num+"])
        ws.write(sr+1, 0, "52-Wk Low", self._f["subhdr"]); ws.write(sr+1, 1, min(lows) if lows else 0, self._f["num-"])
        ws.write(sr+2, 0, "Avg Volume", self._f["subhdr"]); ws.write(sr+2, 1, sum(vols)/len(vols) if vols else 0, self._f["int"])
        tr = (closes[-1]/closes[0] - 1) if len(closes) >= 2 else 0.0
        ws.write(sr+3, 0, "Period Return", self._f["subhdr"]); ws.write(sr+3, 1, tr, self._f["pct+"] if tr >= 0 else self._f["pct-"])

    # ── FA — Financial Analysis ───────────────────────────────────────────────

    def add_fundamentals_sheet(self, ticker: str, facts: list[dict]) -> None:
        """FA - {ticker}: financial facts pivoted by period, grouped by section (millions)."""
        import collections
        ws: Worksheet = self.workbook.add_worksheet(f"FA - {ticker}"[:31])
        ws.set_tab_color(THEME["bg_header"]); ws.hide_gridlines(2); ws.freeze_panes(3, 2)
        ws.merge_range(0, 0, 0, 6, f"SENTINEL  FA  {ticker}  Financial Analysis", self._f["title"])
        if not facts:
            ws.write(2, 0, "No financial data available.", self._f["text"]); return

        periods = sorted({str(f.get("period_end","")) for f in facts if f.get("period_end")}, reverse=True)[:8]
        pivot: dict[tuple, float] = {}; labels: dict[str, str] = {}; secs: dict[str, str] = {}
        for f in facts:
            con, lbl, per, val = str(f.get("concept","")), str(f.get("label") or f.get("concept","")), str(f.get("period_end","")), f.get("value")
            if con and per and val is not None:
                try: pivot[(con, per)] = float(val) / 1e6; labels[con] = lbl; secs[con] = _section(con)
                except (TypeError, ValueError): pass

        by_sec: dict[str, list] = collections.defaultdict(list)
        seen: set[str] = set()
        for con in labels:
            if con not in seen: by_sec[secs[con]].append(con); seen.add(con)

        ws.set_column(0, 0, _W_LBL); ws.set_column(1, 1, 8)
        for ci in range(len(periods)): ws.set_column(ci+2, ci+2, _W_NUM)
        self._hdr_row(ws, 2, ["Concept", "Sec"] + periods)

        row = 3
        for sec in ["Income Statement", "Balance Sheet", "Cash Flow", "Other"]:
            if not by_sec.get(sec): continue
            ws.merge_range(row, 0, row, len(periods)+1, f"  {sec.upper()}", self._f["subhdr"]); row += 1
            for con in by_sec[sec]:
                ws.write(row, 0, labels[con], self._f["label"]); ws.write(row, 1, sec[:2], self._f["text"])
                for ci, per in enumerate(periods):
                    v = pivot.get((con, per))
                    ws.write(row, ci+2, v, self._f["num+"] if v and v >= 0 else self._f["num-"]) if v is not None else ws.write(row, ci+2, "—", self._f["text"])
                row += 1
        ws.write(row+1, 0, "Values in USD millions. Source: SEC EDGAR.", self._f["label"])

    # ── SRCH — Screener ───────────────────────────────────────────────────────

    def add_screener_sheet(self, results: list[dict]) -> None:
        """SRCH - Screener: ranked table, top-10 green, bottom-10 red."""
        ws: Worksheet = self.workbook.add_worksheet("SRCH - Screener")
        ws.set_tab_color(THEME["bg_header"]); ws.hide_gridlines(2); ws.freeze_panes(2, 0)
        ws.merge_range(0, 0, 0, 7, "SENTINEL  SRCH  Equity Screener Results", self._f["title"])
        if not results:
            ws.write(2, 0, "No screener results.", self._f["text"]); return

        ranked = sorted(results, key=lambda x: float(x.get("score") or 0), reverse=True)
        n = len(ranked); bot_start = max(0, n - 10)
        all_fk: list[str] = []
        seen: set[str] = set()
        for r in ranked:
            for k in (r.get("fields") or {}):
                if k not in seen: all_fk.append(k); seen.add(k)
        all_fk = all_fk[:6]
        hdrs = ["Rank", "Ticker", "Name", "Score"] + [k.replace("_"," ").title() for k in all_fk]
        self._hdr_row(ws, 1, hdrs)
        ws.set_column(0,0,6); ws.set_column(1,1,_W_TICK); ws.set_column(2,2,_W_NAME); ws.set_column(3,3,8)
        for ci in range(len(all_fk)): ws.set_column(4+ci, 4+ci, _W_NUM)

        for i, r in enumerate(ranked):
            row = i + 2
            bf = self._f["gbg"] if i < 10 else (self._f["rbg"] if i >= bot_start else self._f["text"])
            nf = self._f["num+"] if i < 10 else (self._f["num-"] if i >= bot_start else self._f["num"])
            ws.write(row,0,i+1,bf); ws.write(row,1,str(r.get("ticker") or ""),bf); ws.write(row,2,str(r.get("name") or ""),bf)
            try: ws.write(row,3,float(r.get("score") or 0),nf)
            except (TypeError,ValueError): ws.write(row,3,str(r.get("score","")),bf)
            for ci, fk in enumerate(all_fk):
                v = (r.get("fields") or {}).get(fk)
                try: ws.write(row,4+ci,float(v),nf) if v is not None else ws.write(row,4+ci,"—",bf)
                except (TypeError,ValueError): ws.write(row,4+ci,str(v or ""),bf)

    # ── PORT — Portfolio ──────────────────────────────────────────────────────

    def add_portfolio_sheet(self, holdings: dict[str, float], metrics: dict) -> None:
        """PORT - Portfolio: holdings table + performance metrics block."""
        ws: Worksheet = self.workbook.add_worksheet("PORT - Portfolio")
        ws.set_tab_color(THEME["bg_header"]); ws.hide_gridlines(2); ws.freeze_panes(2, 0)
        ws.merge_range(0, 0, 0, 3, "SENTINEL  PORT  Portfolio Analysis", self._f["title"])
        total = sum(holdings.values()) or 1.0
        ws.set_column(0,0,_W_TICK); ws.set_column(1,1,_W_NUM); ws.set_column(2,2,_W_NUM); ws.set_column(3,3,_W_NUM)
        self._hdr_row(ws, 1, ["Ticker", "Weight %", "Value ($)", "Bar"])
        for i, (tk, amt) in enumerate(sorted(holdings.items(), key=lambda x: x[1], reverse=True)):
            row = i + 2; w = amt / total
            ws.write(row,0,tk,self._f["text"]); ws.write(row,1,w,self._f["pct"])
            ws.write(row,2,amt,self._f["cur"]); ws.write(row,3,"█"*max(1,int(w*40)),self._f["pos"])
        tr = len(holdings) + 2
        ws.write(tr,0,"TOTAL",self._f["subhdr"]); ws.write(tr,1,1.0,self._f["pct"]); ws.write(tr,2,total,self._f["cur"])

        mr = tr + 3; ws.write(mr, 0, "─── PERFORMANCE METRICS ───", self._f["subhdr"]); mr += 1
        rows_m = [("CAGR","cagr",True,False),("Sharpe","sharpe",False,False),
                  ("Max Drawdown","max_drawdown",True,True),("Volatility","volatility",True,True),
                  ("VaR 95%","var_95",True,True),("Sortino","sortino",False,False),("Beta","beta",False,False)]
        for i, (lbl, key, is_pct, inv) in enumerate(rows_m):
            val = metrics.get(key) or metrics.get(lbl.lower().replace(" ","_"))
            ws.write(mr+i, 0, lbl, self._f["label"])
            if val is not None:
                try:
                    fv = float(val); ok = (fv >= 0) != inv
                    ws.write(mr+i, 1, fv, self._f["pct+" if ok else "pct-"] if is_pct else self._f["pos" if ok else "neg"])
                except (TypeError, ValueError): ws.write(mr+i,1,str(val),self._f["text"])
            else: ws.write(mr+i,1,"N/A",self._f["text"])

    # ── ECOS — Macro ──────────────────────────────────────────────────────────

    def add_macro_sheet(self, series_dict: dict[str, list[dict]]) -> None:
        """ECOS - Macro: date rows × series columns, YoY change colouring."""
        import datetime as _dt
        ws: Worksheet = self.workbook.add_worksheet("ECOS - Macro")
        ws.set_tab_color(THEME["bg_header"]); ws.hide_gridlines(2); ws.freeze_panes(2, 1)
        ws.merge_range(0, 0, 0, len(series_dict), "SENTINEL  ECOS  Macro Economic Series", self._f["title"])
        if not series_dict:
            ws.write(2, 0, "No macro data.", self._f["text"]); return

        all_dates = sorted({str(pt.get("date",""))[:10] for pts in series_dict.values() for pt in pts if pt.get("date")}, reverse=True)
        sids = list(series_dict.keys())
        idx: dict[str, dict[str, Optional[float]]] = {sid: {} for sid in sids}
        for sid, pts in series_dict.items():
            for pt in pts:
                d = str(pt.get("date",""))[:10]
                try: idx[sid][d] = float(pt["value"]) if pt.get("value") is not None else None
                except (TypeError, ValueError): idx[sid][d] = None

        ws.set_column(0,0,_W_DATE)
        for ci in range(len(sids)): ws.set_column(ci+1,ci+1,_W_NUM)
        self._hdr_row(ws, 1, ["Date"] + sids)

        for ri, ds in enumerate(all_dates):
            row = ri + 2
            try:
                dt = _dt.datetime.combine(_dt.date.fromisoformat(ds), _dt.time())
                ws.write_datetime(row, 0, dt, self._f["date"])
            except Exception: ws.write(row, 0, ds, self._f["text"])
            for ci, sid in enumerate(sids):
                cur = idx[sid].get(ds)
                if cur is None: ws.write(row, ci+1, "—", self._f["text"]); continue
                try: prior_d = (_dt.date.fromisoformat(ds) - _dt.timedelta(364)).isoformat()
                except Exception: prior_d = None
                prior = idx[sid].get(prior_d) if prior_d else None
                fmt = (self._f["num+"] if (prior and cur >= prior) else self._f["num-"]) if prior else self._f["num"]
                ws.write(row, ci+1, cur, fmt)

        ws.write(len(all_dates)+3, 0, "Green = YoY improvement  |  Red = YoY deterioration", self._f["label"])

    # ── Close ─────────────────────────────────────────────────────────────────

    def close(self) -> Optional[BytesIO]:
        """Close workbook. Returns seeked BytesIO (or None for file-path output)."""
        self.workbook.close()
        if isinstance(self.output, BytesIO):
            self.output.seek(0); return self.output
        return None


# ── Convenience ───────────────────────────────────────────────────────────────

def export_to_excel(
    ticker: Optional[str] = None,
    ohlcv_df: Optional[pd.DataFrame] = None,
    fundamentals: Optional[list[dict]] = None,
    screener_results: Optional[list[dict]] = None,
    portfolio: Optional[dict] = None,
    macro_series: Optional[dict] = None,
    output_path: Optional[str] = None,
) -> BytesIO:
    """
    Convenience function: create exporter, add sheets for provided data, return buffer.
    Sheets are only added when the corresponding data argument is non-empty.
    """
    exp = BloombergExporter(output=output_path)
    if ticker and ohlcv_df is not None and not ohlcv_df.empty:
        try: exp.add_ohlcv_sheet(ticker=ticker, df=ohlcv_df)
        except Exception as exc: logger.error("add_ohlcv_sheet failed", error=str(exc))
    if ticker and fundamentals:
        try: exp.add_fundamentals_sheet(ticker=ticker, facts=fundamentals)
        except Exception as exc: logger.error("add_fundamentals_sheet failed", error=str(exc))
    if screener_results:
        try: exp.add_screener_sheet(results=screener_results)
        except Exception as exc: logger.error("add_screener_sheet failed", error=str(exc))
    if portfolio:
        hld = portfolio.get("holdings") or {}
        met = {k: v for k, v in portfolio.items() if k != "holdings"}
        try: exp.add_portfolio_sheet(holdings=hld, metrics=met)
        except Exception as exc: logger.error("add_portfolio_sheet failed", error=str(exc))
    if macro_series:
        try: exp.add_macro_sheet(series_dict=macro_series)
        except Exception as exc: logger.error("add_macro_sheet failed", error=str(exc))
    return exp.close() or BytesIO()


# ── Streamlit integration ─────────────────────────────────────────────────────

def streamlit_download_button(
    st,
    ticker: Optional[str] = None,
    ohlcv_df: Optional[pd.DataFrame] = None,
    fundamentals: Optional[list[dict]] = None,
    label: str = "Export to Excel",
) -> None:
    """
    Render a Streamlit download button triggering Bloomberg-dark Excel export.

    Usage in terminal.py:
        from sentinel.stu.excel_export import streamlit_download_button
        streamlit_download_button(st, ticker="AAPL", ohlcv_df=df, fundamentals=facts)
    """
    try:
        buf = export_to_excel(ticker=ticker, ohlcv_df=ohlcv_df, fundamentals=fundamentals)
        fname = f"SENTINEL_{ticker or 'export'}_{date.today().isoformat()}.xlsx"
        st.download_button(label=label, data=buf, file_name=fname,
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception as exc:
        logger.error("streamlit_download_button error", error=str(exc))
        st.error(f"Export failed: {exc}")
