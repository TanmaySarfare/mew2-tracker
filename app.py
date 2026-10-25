"""
Mew2 Tracker
============
Look up any stock, ETF, mutual fund, index, currency or crypto listed on Yahoo Finance,
see its data on the page, and download every table as CSV.

Built for DCF study and single-stock analysis with Streamlit + yfinance.

Run it:
    pip install -r requirements.txt
    streamlit run app.py
"""
from __future__ import annotations

import base64
import html
import io
import os
import math
import re
import urllib.request
import zipfile
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

APP_NAME = "Mew2 Tracker"
APP_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".gif": "image/gif"}


def logo_path() -> str | None:
    """Your logo: any image in the app's folder whose name starts with "logo", such as logo.png."""
    try:
        names = sorted(os.listdir(APP_DIR))
    except OSError:
        return None
    for name in names:
        path = os.path.join(APP_DIR, name)
        if name.lower().startswith("logo") and os.path.splitext(name.lower())[1] in LOGO_TYPES and os.path.isfile(path):
            try:
                from PIL import Image  # installed with Streamlit
                with Image.open(path) as image:
                    image.verify()  # skip files that aren't really images
            except Exception:
                continue
            return path
    return None


st.set_page_config(page_title=APP_NAME, page_icon=logo_path() or "📊", layout="wide",
                   initial_sidebar_state="collapsed")

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
UP, DOWN = "#1E8E5A", "#C0392B"          # ledger green / accounting red
LINE, BENCH = "#2F5D8A", "#8A8F98"       # ink blue / graphite
SMA_FAST, SMA_SLOW = "#C98A1B", "#6B4FA0"

UNITS = {"Raw": 1.0, "Thousands": 1e3, "Millions": 1e6, "Billions": 1e9}
INTERVALS = {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo", "Quarterly": "3mo"}
BAR_WORD = {"Daily": "day", "Weekly": "week", "Monthly": "month", "Quarterly": "quarter"}
TYPE_LABELS = {
    "EQUITY": "Stock", "ETF": "ETF", "MUTUALFUND": "Mutual fund", "INDEX": "Index",
    "CRYPTOCURRENCY": "Crypto", "CURRENCY": "Currency", "FUTURE": "Future",
}
FUND_TYPES = {"ETF", "MUTUALFUND"}
MARKET_ONLY_TYPES = {"INDEX", "CURRENCY", "CRYPTOCURRENCY", "FUTURE"}

DEFAULT_RF = 0.0425      # used only if the live 10-year Treasury yield can't be loaded
DEFAULT_ERP = 0.045      # equity risk premium: an assumption you should set yourself
DEFAULT_TAX = 0.21
DEFAULT_TERMINAL_GROWTH = 0.025

# Prices quoted in a currency's minor unit (pence, cents, agorot).
SUBUNITS = {"GBp": ("GBP", 100.0), "GBX": ("GBP", 100.0), "ZAc": ("ZAR", 100.0), "ILA": ("ILS", 100.0)}

TICKER_PATTERN = re.compile(r"^[\^A-Za-z0-9.\-=]{1,20}$")
UNSCALED_ROWS = re.compile(r"\bEPS\b|Per Share|Tax Rate")  # per-share and ratio rows keep their units

STATEMENTS = [("income", "Income statement"), ("balance", "Balance sheet"), ("cashflow", "Cash flow")]
STATEMENT_METHODS = {"income": "get_income_stmt", "balance": "get_balance_sheet", "cashflow": "get_cashflow"}

# Yahoo line-item names, in order of preference.
IS_ROWS = {
    "revenue": ["Total Revenue", "Operating Revenue"],
    "ebit": ["Operating Income", "EBIT"],
    "pretax": ["Pretax Income"],
    "tax": ["Tax Provision"],
    "tax_rate": ["Tax Rate For Calcs"],
    "interest": ["Interest Expense", "Interest Expense Non Operating"],
    "da": ["Reconciled Depreciation"],
}
CF_ROWS = {
    "ocf": ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"],
    "capex": ["Capital Expenditure"],
    "fcf": ["Free Cash Flow"],
    "da": ["Depreciation And Amortization", "Depreciation Amortization Depletion"],
    "wc": ["Change In Working Capital"],
}
BS_ROWS = {
    "debt": ["Total Debt"],
    "cash": ["Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents"],
    "shares": ["Ordinary Shares Number", "Share Issued"],
    "minority": ["Minority Interest"],
    "preferred": ["Preferred Stock"],
}
FCFF = "Unlevered free cash flow (FCFF)"
FCF = "Free cash flow (OCF - CapEx)"
PCT_ROWS = {"Revenue growth (%)", "EBIT margin (%)", "Effective tax rate (%)"}

FUND_FIELDS = ("description", "fund_overview", "fund_operations", "asset_classes", "top_holdings",
               "equity_holdings", "bond_holdings", "bond_ratings", "sector_weightings")
ESTIMATE_FIELDS = ("analyst_price_targets", "recommendations", "earnings_estimate", "revenue_estimate",
                   "eps_trend", "growth_estimates", "upgrades_downgrades")
HOLDER_FIELDS = ("major_holders", "institutional_holders", "mutualfund_holders")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def num(x) -> float | None:
    """Return x as a finite float, or None."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def safe_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_") or "ticker"


def md(text) -> str:
    """Stop dollar signs in Yahoo text from being read as LaTeX by Streamlit's markdown."""
    return str(text).replace("$", "\\$")


def fmt_money(x, decimals: int = 2) -> str:
    v = num(x)
    if v is None:
        return "n/a"
    for suffix, div in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(v) >= div:
            return f"{v / div:,.{decimals}f}{suffix}"
    return f"{v:,.{decimals}f}"


def fmt_pct(x, decimals: int = 2) -> str:
    v = num(x)
    return "n/a" if v is None else f"{v * 100:.{decimals}f}%"


def price_decimals(values) -> int:
    v = num(pd.Series(values).abs().median())
    if v is None or v >= 10:
        return 2
    return 4 if v >= 0.1 else 6


def fmt_price(x) -> str:
    v = num(x)
    return "n/a" if v is None else f"{v:,.{price_decimals([v])}f}"


def to_csv_bytes(df: pd.DataFrame, index: bool = True) -> bytes:
    return df.to_csv(index=index, date_format="%Y-%m-%d").encode("utf-8")


def dict_table(d, key_name: str = "Field", value_name: str = "Value") -> pd.DataFrame:
    if not isinstance(d, dict) or not d:
        return pd.DataFrame()
    return pd.DataFrame(list(d.items()), columns=[key_name, value_name])


def as_text_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Copy of df with one column turned into display strings (mixed types break Arrow)."""
    out = df.copy()
    out[column] = out[column].map(lambda v: "" if v is None else (f"{v:,}" if isinstance(v, int) and not isinstance(v, bool) else str(v)))
    return out


def pretty_key(key: str) -> str:
    special = {"realestate": "Real estate", "us_government": "U.S. government", "below_b": "Below B"}
    if key in special:
        return special[key]
    if key.lower() in {"aaa", "aa", "a", "bbb", "bb", "b"}:
        return key.upper()
    words = re.sub(r"(?<!^)(?=[A-Z])", " ", key).replace("_", " ")
    words = re.sub(r"\bposition\b", "", words, flags=re.IGNORECASE).strip()
    return (words[:1].upper() + words[1:].lower()) if words else key


def describe_error(err: Exception, symbol: str = "") -> str:
    text = f"{type(err).__name__}: {err}"
    low = text.lower()
    if "ratelimit" in low or "too many requests" in low or "429" in low:
        return ("Yahoo Finance is limiting requests right now. Wait a minute, then refresh the page. "
                "Anything already loaded stays cached, so it won't be requested again.")
    if any(s in low for s in ("delisted", "no data found", "not found", "pricesmissing", "tzmissing",
                              "no timezone", "404", "no price data")):
        return (f"Yahoo Finance has no price data for \"{symbol}\". Check the symbol. Listings outside the U.S. "
                "need an exchange suffix, for example RELIANCE.NS (India), VOD.L (London) or SHOP.TO (Toronto).")
    return f"Couldn't load data for \"{symbol}\". Yahoo Finance returned: {err}"


def attempt(fn, *args, **kwargs):
    """Call a loader and return (value, error_message) instead of raising."""
    try:
        return fn(*args, **kwargs), None
    except Exception as exc:  # yfinance raises many exception types
        return None, describe_error(exc, str(args[0]) if args else "")


# ---------------------------------------------------------------------------
# Data loading (cached so Yahoo isn't asked for the same thing twice)
# Failures raise instead of returning empty data, so they are never cached.
# ---------------------------------------------------------------------------
@st.cache_data(ttl=24 * 3600, show_spinner=False)
def search_symbols(query: str) -> list[dict]:
    results = []
    for q in yf.Search(query, max_results=10, news_count=0).quotes or []:
        symbol = q.get("symbol")
        if symbol:
            results.append({
                "symbol": symbol,
                "name": q.get("longname") or q.get("shortname") or "",
                "type": str(q.get("quoteType") or q.get("typeDisp") or ""),
                "exchange": q.get("exchDisp") or q.get("exchange") or "",
            })
    return results


def _naive_index(index) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index)
    return idx.tz_localize(None) if idx.tz is not None else idx


@st.cache_data(ttl=15 * 60, show_spinner=False)
def load_history(symbol: str, period: str | None = "max", start: str | None = None,
                 end: str | None = None, interval: str = "1d") -> tuple[pd.DataFrame, dict]:
    ticker = yf.Ticker(symbol)
    kwargs = {"interval": interval, "auto_adjust": False, "actions": True, "raise_errors": True}
    if period:
        kwargs["period"] = period
    else:
        kwargs["start"], kwargs["end"] = start, end
    df = ticker.history(**kwargs)
    if df is None or df.empty:
        raise LookupError(f"No price data found for {symbol}")
    df = df.copy()
    df.index = _naive_index(df.index)
    df.index.name = "Date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    keep = [c for c in ("Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends",
                        "Stock Splits", "Capital Gains") if c in df.columns]
    prices = [c for c in ("Open", "High", "Low", "Close") if c in keep]
    df = df[keep].dropna(how="all", subset=prices or None)
    if df.empty:
        raise LookupError(f"No price data found for {symbol}")
    try:
        meta = ticker.history_metadata or {}
    except Exception:
        meta = {}
    meta = {k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool))}
    return df, meta


@st.cache_data(ttl=3600, show_spinner=False)
def load_info(symbol: str) -> dict:
    return dict(yf.Ticker(symbol).info or {})


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def load_statement(symbol: str, kind: str, freq: str) -> pd.DataFrame:
    method = getattr(yf.Ticker(symbol), STATEMENT_METHODS[kind])
    df = method(pretty=True, freq=freq)
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()


def _collect(obj, names) -> dict:
    out, errors = {}, []
    for name in names:
        try:
            out[name] = getattr(obj, name)
        except Exception as exc:
            out[name] = None
            errors.append(exc)
    if errors and len(errors) == len(names):
        raise errors[-1]  # nothing came back, so don't cache the failure
    return out


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def load_fund_data(symbol: str) -> dict:
    return _collect(yf.Ticker(symbol).funds_data, FUND_FIELDS)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_estimates(symbol: str) -> dict:
    return _collect(yf.Ticker(symbol), ESTIMATE_FIELDS)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_holders(symbol: str) -> dict:
    return _collect(yf.Ticker(symbol), HOLDER_FIELDS)


def risk_free_rate() -> tuple[float, str]:
    try:
        tnx, _ = load_history("^TNX", "1mo")
        last = num(tnx["Close"].dropna().iloc[-1])
        if last and last > 20:  # some feeds quote the yield x10
            last /= 10
        if last and 0 < last < 20:
            return last / 100, "Latest 10-year U.S. Treasury yield (^TNX) from Yahoo Finance."
    except Exception:
        pass
    return DEFAULT_RF, "Default assumption, because the live 10-year Treasury yield couldn't be loaded."


def fx_factor(fin_ccy: str, price_ccy: str) -> float | None:
    """Multiply an amount in the reporting currency by this to get the trading currency."""
    if not fin_ccy or not price_ccy:
        return 1.0
    major, mult = SUBUNITS.get(price_ccy, (price_ccy, 1.0))
    if fin_ccy == major:
        return mult
    try:
        fx, _ = load_history(f"{fin_ccy}{major}=X", "5d")
        rate = num(fx["Close"].dropna().iloc[-1])
        return rate * mult if rate else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Calculations (plain pandas, no Streamlit)
# ---------------------------------------------------------------------------
def adjusted_close(df: pd.DataFrame) -> pd.Series:
    col = "Adj Close" if "Adj Close" in df and df["Adj Close"].notna().any() else "Close"
    s = df[col].dropna()
    return s[s > 0]


def nonzero(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df:
        return pd.Series(dtype=float)
    s = df[column].fillna(0)
    return s[s != 0]


def ttm_dividends(daily: pd.DataFrame) -> float:
    divs = nonzero(daily, "Dividends")
    if divs.empty:
        return 0.0
    return float(divs[divs.index > daily.index.max() - pd.Timedelta(days=365)].sum())


def window_start(index: pd.DatetimeIndex, label: str):
    end = index.max()
    if label == "All":
        return None
    if label == "YTD":
        return pd.Timestamp(year=end.year, month=1, day=1)
    n, unit = int(label[:-1]), label[-1]
    return end - (pd.DateOffset(months=n) if unit == "M" else pd.DateOffset(years=n))


def slice_window(obj, label: str):
    start = window_start(obj.index, label)
    if start is None:
        return obj
    sub = obj[obj.index >= start]
    return sub if len(sub) >= 2 else obj


RETURN_PERIODS = [("1M", pd.DateOffset(months=1)), ("3M", pd.DateOffset(months=3)),
                  ("6M", pd.DateOffset(months=6)), ("YTD", "ytd"), ("1Y", pd.DateOffset(years=1)),
                  ("3Y", pd.DateOffset(years=3)), ("5Y", pd.DateOffset(years=5)),
                  ("10Y", pd.DateOffset(years=10)), ("Since start", "start")]


def period_return(s: pd.Series, spec) -> tuple[float, float]:
    """Total and annualized return over a trailing window ending at the last observation."""
    s = s.dropna()
    nan = float("nan")
    if len(s) < 2:
        return nan, nan
    end_date, end_val = s.index[-1], s.iloc[-1]
    if spec == "start":
        base_date, base_val = s.index[0], s.iloc[0]
    elif spec == "ytd":
        prior = s[s.index < pd.Timestamp(year=end_date.year, month=1, day=1)]
        if prior.empty:
            return nan, nan
        base_date, base_val = prior.index[-1], prior.iloc[-1]
    else:
        target = end_date - spec
        prior = s[s.index <= target]
        if prior.empty:
            if s.index[0] - target > pd.Timedelta(days=7):
                return nan, nan
            base_date, base_val = s.index[0], s.iloc[0]
        else:
            base_date, base_val = prior.index[-1], prior.iloc[-1]
    total = end_val / base_val - 1
    years = (end_date - base_date).days / 365.25
    annualized = (1 + total) ** (1 / years) - 1 if years >= 1 else nan
    return total, annualized


def returns_table(adj: pd.Series, bench: pd.Series | None, symbol: str, benchmark: str) -> pd.DataFrame:
    rows = []
    b_all = bench[bench.index <= adj.index[-1]] if bench is not None else None
    for label, spec in RETURN_PERIODS:
        total, ann = period_return(adj, spec)
        row = {"Period": label, f"{symbol} total return (%)": total * 100, f"{symbol} annualized (%)": ann * 100}
        if b_all is not None:
            b = b_all[b_all.index >= adj.index[0]] if spec == "start" else b_all
            b_total, b_ann = period_return(b, spec)
            row[f"{benchmark} total return (%)"] = b_total * 100
            row[f"{benchmark} annualized (%)"] = b_ann * 100
        rows.append(row)
    return pd.DataFrame(rows)


RISK_WINDOWS = [("1Y", 1), ("3Y", 3), ("5Y", 5), ("10Y", 10), ("Full history", None)]
RISK_COLUMNS = ["Annualized return (%)", "Annualized volatility (%)", "Sharpe ratio", "Max drawdown (%)"]


def risk_stats(s: pd.Series, rf: float) -> dict:
    s = s.dropna()
    stats = {c: float("nan") for c in RISK_COLUMNS}
    stats["Worst drawdown date"] = None
    years = (s.index[-1] - s.index[0]).days / 365.25 if len(s) > 1 else 0
    if len(s) < 20 or years <= 0:
        return stats
    rets = (s / s.shift(1) - 1).dropna()
    ann_ret = (s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1
    ann_vol = rets.std() * math.sqrt(len(rets) / years)  # uses the real number of trading days per year
    dd = s / s.cummax() - 1
    stats.update({
        "Annualized return (%)": ann_ret * 100,
        "Annualized volatility (%)": ann_vol * 100,
        "Sharpe ratio": (ann_ret - rf) / ann_vol if ann_vol > 0 else float("nan"),
        "Max drawdown (%)": dd.min() * 100,
        "Worst drawdown date": dd.idxmin().strftime("%Y-%m-%d"),
    })
    return stats


def risk_table(adj: pd.Series, rf: float) -> pd.DataFrame:
    end = adj.index[-1]
    rows = []
    for label, years in RISK_WINDOWS:
        if years:
            start = end - pd.DateOffset(years=years)
            if adj.index[0] - start > pd.Timedelta(days=7):
                rows.append({"Window": label, **risk_stats(adj.iloc[:0], rf)})
                continue
            sub = adj[adj.index >= start]
        else:
            sub = adj
        rows.append({"Window": label, **risk_stats(sub, rf)})
    return pd.DataFrame(rows)


def monthly_beta(s: pd.Series, b: pd.Series, months: int = 60) -> tuple[float, float, int]:
    ms = s.groupby(s.index.to_period("M")).last()
    mb = b.groupby(b.index.to_period("M")).last()
    df = pd.concat([ms / ms.shift(1) - 1, mb / mb.shift(1) - 1], axis=1, keys=["s", "b"]).dropna().iloc[-months:]
    if len(df) < 12 or df["b"].var() == 0:
        return float("nan"), float("nan"), len(df)
    return df["s"].cov(df["b"]) / df["b"].var(), df["s"].corr(df["b"]), len(df)


def choose_beta(computed: float | None, months: int, benchmark: str, yahoo_beta: float | None) -> tuple[float, str]:
    """Pick the beta to pre-fill: computed if plausible, else Yahoo's, else 1.0."""
    def plausible(b):
        return b is not None and np.isfinite(b) and 0.3 <= b <= 3.0

    if plausible(computed):
        return float(computed), f"Computed from {months} months of returns against {benchmark}."
    reason = ""
    if computed is not None and np.isfinite(computed):
        reason = (f"The beta computed against {benchmark} ({computed:.2f}) is outside 0.3 to 3, which usually "
                  "means the price history is thin or doesn't move with that benchmark. ")
    if plausible(yahoo_beta):
        return float(yahoo_beta), reason + "Using the beta Yahoo Finance reports instead."
    return 1.0, reason + "Using a market-average beta of 1.0."


def tidy_statement(df: pd.DataFrame | None, scale: float) -> pd.DataFrame:
    """Oldest period on the left, numbers scaled to the chosen unit (per-share rows untouched)."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out.columns = pd.to_datetime(out.columns, errors="coerce")
    out = out.loc[:, out.columns.notna()]
    out = out.apply(pd.to_numeric, errors="coerce").dropna(how="all").dropna(axis=1, how="all")
    if out.empty:
        return pd.DataFrame()
    out = out.reindex(sorted(out.columns), axis=1)
    scaled = ~out.index.to_series().astype(str).str.contains(UNSCALED_ROWS).to_numpy()
    out.loc[scaled] = out.loc[scaled] / scale
    out.columns = [c.strftime("%Y-%m-%d") for c in out.columns]
    out.index.name = "Line item"
    return out


def get_row(df: pd.DataFrame | None, names) -> pd.Series:
    """First matching line item as a float series indexed by period end, oldest first."""
    if df is None or df.empty:
        return pd.Series(dtype=float)
    for name in names:
        if name in df.index:
            values = df.loc[name]
            if isinstance(values, pd.DataFrame):
                values = values.iloc[0]
            s = pd.to_numeric(values, errors="coerce")
            s.index = pd.to_datetime(s.index, errors="coerce")
            s = s[s.index.notna()].sort_index().dropna()
            if not s.empty:
                return s.astype(float)
    return pd.Series(dtype=float)


def latest(s: pd.Series) -> float | None:
    s = s.dropna()
    return float(s.iloc[-1]) if len(s) else None


def cagr(s: pd.Series) -> float | None:
    s = s.dropna()
    if len(s) < 2 or s.iloc[0] <= 0 or s.iloc[-1] <= 0:
        return None
    years = (s.index[-1] - s.index[0]).days / 365.25
    return (s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1 if years >= 0.9 else None


def dcf_history(inc, cf, bs) -> pd.DataFrame:
    """Annual building blocks for a DCF, one row per fiscal year (oldest first)."""
    revenue, ebit = get_row(inc, IS_ROWS["revenue"]), get_row(inc, IS_ROWS["ebit"])
    pretax, tax = get_row(inc, IS_ROWS["pretax"]), get_row(inc, IS_ROWS["tax"])
    interest = get_row(inc, IS_ROWS["interest"])
    da = get_row(cf, CF_ROWS["da"])
    if da.empty:
        da = get_row(inc, IS_ROWS["da"])
    capex, wc = get_row(cf, CF_ROWS["capex"]), get_row(cf, CF_ROWS["wc"])
    ocf, fcf_reported = get_row(cf, CF_ROWS["ocf"]), get_row(cf, CF_ROWS["fcf"])

    dates = sorted(set().union(*(s.index for s in (revenue, ebit, ocf, capex))))
    if not dates:
        return pd.DataFrame()
    idx = pd.DatetimeIndex(dates)

    def a(s):
        return s.reindex(idx)

    tax_rate = (a(tax) / a(pretax)).where(a(pretax) > 0).clip(lower=0, upper=0.5)
    tax_rate = tax_rate.fillna(a(get_row(inc, IS_ROWS["tax_rate"])))
    capex_out = -a(capex).abs()
    nopat = a(ebit) * (1 - tax_rate.fillna(DEFAULT_TAX))
    table = pd.DataFrame({
        "Revenue": a(revenue),
        "Revenue growth (%)": (a(revenue) / a(revenue).shift(1) - 1) * 100,
        "Operating income (EBIT)": a(ebit),
        "EBIT margin (%)": a(ebit) / a(revenue) * 100,
        "Effective tax rate (%)": tax_rate * 100,
        "NOPAT = EBIT x (1 - tax rate)": nopat,
        "Depreciation & amortization": a(da),
        "Capital expenditure": capex_out,
        "Change in working capital": a(wc),
        FCFF: nopat + a(da) + capex_out + a(wc).fillna(0),
        "Operating cash flow": a(ocf),
        FCF: a(fcf_reported).fillna(a(ocf) + capex_out),
        "Interest expense": a(interest).abs(),
    }, index=idx)
    table.index.name = "Fiscal year end"
    return table.replace([np.inf, -np.inf], np.nan)


def blocks_for_display(table: pd.DataFrame, scale: float) -> pd.DataFrame:
    out = table.T.copy()
    money = [r for r in out.index if r not in PCT_ROWS]
    out.loc[money] = out.loc[money] / scale
    out.columns = [c.strftime("%Y-%m-%d") for c in out.columns]
    out.index.name = "Item"
    return out


def capital_structure(info: dict, bs: pd.DataFrame | None) -> dict:
    debt = latest(get_row(bs, BS_ROWS["debt"]))
    cash = latest(get_row(bs, BS_ROWS["cash"]))
    return {
        "debt": debt if debt is not None else (num(info.get("totalDebt")) or 0.0),
        "cash": cash if cash is not None else (num(info.get("totalCash")) or 0.0),
        "minority": latest(get_row(bs, BS_ROWS["minority"])) or 0.0,
        "preferred": latest(get_row(bs, BS_ROWS["preferred"])) or 0.0,
        "shares": num(info.get("sharesOutstanding")) or num(info.get("impliedSharesOutstanding"))
        or latest(get_row(bs, BS_ROWS["shares"])),
    }


def compute_wacc(rf, beta, erp, kd, tax, equity_value, debt_value) -> dict:
    ke = rf + beta * erp
    kd_after = kd * (1 - tax)
    equity_value, debt_value = max(equity_value or 0.0, 0.0), max(debt_value or 0.0, 0.0)
    total = equity_value + debt_value
    we = equity_value / total if total > 0 and equity_value > 0 else 1.0
    return {"ke": ke, "kd_after": kd_after, "we": we, "wd": 1 - we, "wacc": we * ke + (1 - we) * kd_after}


def run_dcf(base_cf, growth, years, terminal_growth, wacc, fade, net_claims, shares, first_year) -> dict:
    if wacc <= terminal_growth:
        raise ValueError("WACC must be higher than the terminal growth rate.")
    rows, cf = [], base_cf
    for i in range(1, years + 1):
        g = growth + (terminal_growth - growth) * (i - 1) / (years - 1) if (fade and years > 1) else growth
        cf *= 1 + g
        factor = 1 / (1 + wacc) ** i
        rows.append({"Fiscal year": (first_year + i) if first_year else i, "Growth (%)": g * 100,
                     "Cash flow": cf, "Discount factor": factor, "Present value": cf * factor})
    projection = pd.DataFrame(rows)
    terminal = cf * (1 + terminal_growth) / (wacc - terminal_growth)
    pv_terminal = terminal / (1 + wacc) ** years
    pv_explicit = float(projection["Present value"].sum())
    ev = pv_explicit + pv_terminal
    equity = ev - net_claims
    return {"projection": projection, "terminal_value": terminal, "pv_terminal": pv_terminal,
            "pv_explicit": pv_explicit, "enterprise_value": ev, "equity_value": equity,
            "per_share": equity / shares if shares else float("nan")}


def sensitivity_grid(params: dict, wacc: float, terminal_growth: float, step: float = 0.005, n: int = 2) -> pd.DataFrame:
    waccs = [wacc + step * k for k in range(-n, n + 1)]
    growths = [terminal_growth + step * k for k in range(-n, n + 1)]
    data = []
    for w in waccs:
        row = []
        for g in growths:
            if w <= g or w <= 0:
                row.append(float("nan"))
            else:
                row.append(run_dcf(**{**params, "wacc": w, "terminal_growth": g})["per_share"])
        data.append(row)
    grid = pd.DataFrame(data, index=[f"{w * 100:.2f}%" for w in waccs], columns=[f"{g * 100:.2f}%" for g in growths])
    grid.index.name, grid.columns.name = "WACC", "Terminal growth"
    return grid


def analyst_growth(est: dict | None) -> dict:
    out = {}
    if not est:
        return out
    ge = est.get("growth_estimates")
    if isinstance(ge, pd.DataFrame) and not ge.empty:
        col = ge.columns[0]
        for key in ("LTG", "+5y"):
            if key in ge.index and num(ge.loc[key, col]) is not None:
                out["analysts' long-term EPS growth"] = num(ge.loc[key, col])
                break
    rev = est.get("revenue_estimate")
    if isinstance(rev, pd.DataFrame) and "growth" in rev.columns and "+1y" in rev.index:
        v = num(rev.loc["+1y", "growth"])
        if v is not None:
            out["analysts' revenue growth for next fiscal year"] = v
    return out


def weights_table(d, label: str) -> pd.DataFrame:
    if not isinstance(d, dict) or not d:
        return pd.DataFrame()
    rows = [(pretty_key(k), num(v) * 100) for k, v in d.items() if num(v) is not None]
    return pd.DataFrame(rows, columns=[label, "Weight (%)"]).sort_values("Weight (%)", ascending=False)


# ---------------------------------------------------------------------------
# More data: earnings and industry peers
# ---------------------------------------------------------------------------
EARNINGS_FIELDS = ("earnings_dates", "earnings_history", "calendar")


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_earnings(symbol: str) -> dict:
    return _collect(yf.Ticker(symbol), EARNINGS_FIELDS)


@st.cache_data(ttl=7 * 24 * 3600, show_spinner=False)
def load_industry_peers(industry_key: str) -> list:
    top = yf.Industry(industry_key).top_companies
    if not isinstance(top, pd.DataFrame) or top.empty:
        return []
    symbols = top["symbol"] if "symbol" in top.columns else top.index
    return [str(s).strip() for s in symbols if str(s).strip()]


# ---------------------------------------------------------------------------
# Technical indicators
# ---------------------------------------------------------------------------
INDICATOR_OPTIONS = ["Moving averages", "Bollinger bands", "RSI", "MACD"]


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative strength index with Wilder's smoothing."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return 100 - 100 / (1 + gain / loss)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    sig = line.ewm(span=signal, adjust=False).mean()
    out = pd.DataFrame({"MACD": line, "MACD signal": sig, "MACD histogram": line - sig})
    out.iloc[: slow - 1] = np.nan  # not enough history yet for a meaningful value
    return out


def bollinger(close: pd.Series, window: int = 20, width: float = 2.0) -> pd.DataFrame:
    mid = close.rolling(window).mean()
    sd = close.rolling(window).std(ddof=0)
    return pd.DataFrame({"Bollinger upper": mid + width * sd, "Bollinger middle": mid, "Bollinger lower": mid - width * sd})


def indicator_frame(hist: pd.DataFrame, chosen) -> pd.DataFrame:
    """Indicator columns for the whole history (so values at the start of a view are already warmed up)."""
    close = hist["Close"]
    parts = []
    if "Moving averages" in chosen:
        parts.append(pd.DataFrame({"SMA 50": close.rolling(50).mean(), "SMA 200": close.rolling(200).mean()}))
    if "Bollinger bands" in chosen:
        parts.append(bollinger(close))
    if "RSI" in chosen:
        parts.append(rsi(close).rename("RSI 14").to_frame())
    if "MACD" in chosen:
        parts.append(macd(close))
    return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=hist.index)


# ---------------------------------------------------------------------------
# Financial ratios over time
# ---------------------------------------------------------------------------
RATIO_GROUPS = [
    ("Profitability", ["Gross margin (%)", "Operating margin (%)", "EBITDA margin (%)", "Net margin (%)",
                       "Free cash flow margin (%)"]),
    ("Returns", ["Return on equity, ROE (%)", "Return on assets, ROA (%)", "Return on invested capital, ROIC (%)"]),
    ("Growth", ["Revenue growth (%)", "Earnings per share growth (%)", "Free cash flow growth (%)"]),
    ("Financial health", ["Debt to equity (x)", "Net debt to EBITDA (x)", "Interest coverage (x)", "Current ratio (x)",
                          "Quick ratio (x)"]),
    ("Efficiency and cash", ["Asset turnover (x)", "Operating cash flow to net income (x)",
                             "Capital expenditure to revenue (%)", "Dividend payout ratio (%)"]),
    ("Valuation at fiscal year end", ["Price to earnings, P/E (x)", "Price to book, P/B (x)", "Price to sales, P/S (x)",
                                      "EV to EBITDA (x)", "EV to revenue (x)", "Free cash flow yield (%)"]),
]


def split_factor_after(daily: pd.DataFrame, when) -> float:
    splits = nonzero(daily, "Stock Splits")
    later = splits[splits.index > when]
    return float(np.prod(later.to_numpy())) if len(later) else 1.0


def to_current_share_basis(shares: pd.Series, daily: pd.DataFrame) -> pd.Series:
    """Restate share counts on today's basis when a later stock split isn't reflected in them.

    Yahoo's prices are split-adjusted, so market values need share counts on the same basis.
    Share counts drift slowly, so a count that only lines up with the latest one after applying
    the later splits was reported before those splits.
    """
    s = shares.dropna()
    s = s[s > 0]
    if len(s) < 1:
        return shares
    latest = s.iloc[-1]
    out = shares.astype(float).copy()
    for when, value in s.items():
        factor = split_factor_after(daily, when)
        if factor != 1.0 and abs(math.log(value * factor / latest)) < abs(math.log(value / latest)):
            out[when] = value * factor
    return out


def price_on(daily: pd.DataFrame, when) -> float | None:
    closes = daily["Close"].dropna()
    closes = closes[closes.index <= when]
    return float(closes.iloc[-1]) if len(closes) else None


def financial_ratios(inc, bs, cf, daily: pd.DataFrame) -> pd.DataFrame:
    """Ratios by fiscal year: index is the ratio name, columns are fiscal year ends (oldest first)."""
    revenue = get_row(inc, IS_ROWS["revenue"])
    if revenue.empty:
        return pd.DataFrame()
    idx = revenue.index

    def a(df, names):
        return get_row(df, names).reindex(idx)

    def ratio(n, d):
        return n / d.where(d > 0)

    def avg(s):
        return ((s + s.shift(1)) / 2).fillna(s)

    def growth(s):
        prev = s.shift(1)
        return (s / prev - 1).where(prev > 0)

    gross = a(inc, ["Gross Profit"])
    if gross.isna().all():
        gross = revenue - a(inc, ["Cost Of Revenue"])
    ebit = a(inc, IS_ROWS["ebit"])
    ebitda = a(inc, ["EBITDA", "Normalized EBITDA"])
    net = a(inc, ["Net Income Common Stockholders", "Net Income"])
    pretax, tax = a(inc, IS_ROWS["pretax"]), a(inc, IS_ROWS["tax"])
    interest = a(inc, IS_ROWS["interest"]).abs()
    diluted = a(inc, ["Diluted Average Shares", "Basic Average Shares"])
    equity = a(bs, ["Stockholders Equity", "Common Stock Equity"])
    assets = a(bs, ["Total Assets"])
    debt = a(bs, BS_ROWS["debt"])
    cash = a(bs, BS_ROWS["cash"])
    current_assets, current_liabilities = a(bs, ["Current Assets"]), a(bs, ["Current Liabilities"])
    inventory = a(bs, ["Inventory"]).fillna(0)
    minority = a(bs, BS_ROWS["minority"]).fillna(0)
    shares = a(bs, BS_ROWS["shares"])
    ocf = a(cf, CF_ROWS["ocf"])
    capex = a(cf, CF_ROWS["capex"]).abs()
    fcf = a(cf, CF_ROWS["fcf"]).fillna(ocf - capex)
    dividends = a(cf, ["Cash Dividends Paid", "Common Stock Dividend Paid"]).abs()

    tax_rate = (tax / pretax).where(pretax > 0).clip(lower=0, upper=0.5).fillna(DEFAULT_TAX)
    invested = debt.fillna(0) + equity - cash.fillna(0)
    shares_now = to_current_share_basis(shares if shares.notna().any() else diluted, daily)
    diluted_now = to_current_share_basis(diluted if diluted.notna().any() else shares, daily)
    prices = pd.Series([price_on(daily, d) for d in idx], index=idx, dtype=float)
    market_cap = prices * shares_now
    ev = market_cap + debt.fillna(0) - cash.fillna(0) + minority

    rows = {
        "Gross margin (%)": ratio(gross, revenue) * 100,
        "Operating margin (%)": ratio(ebit, revenue) * 100,
        "EBITDA margin (%)": ratio(ebitda, revenue) * 100,
        "Net margin (%)": ratio(net, revenue) * 100,
        "Free cash flow margin (%)": ratio(fcf, revenue) * 100,
        "Return on equity, ROE (%)": ratio(net, avg(equity)) * 100,
        "Return on assets, ROA (%)": ratio(net, avg(assets)) * 100,
        "Return on invested capital, ROIC (%)": ratio(ebit * (1 - tax_rate), avg(invested)) * 100,
        "Revenue growth (%)": growth(revenue) * 100,
        "Earnings per share growth (%)": growth(net / diluted_now) * 100,
        "Free cash flow growth (%)": growth(fcf) * 100,
        "Debt to equity (x)": ratio(debt, equity),
        "Net debt to EBITDA (x)": ratio(debt - cash.fillna(0), ebitda),
        "Interest coverage (x)": ratio(ebit, interest),
        "Current ratio (x)": ratio(current_assets, current_liabilities),
        "Quick ratio (x)": ratio(current_assets - inventory, current_liabilities),
        "Asset turnover (x)": ratio(revenue, avg(assets)),
        "Operating cash flow to net income (x)": ratio(ocf, net),
        "Capital expenditure to revenue (%)": ratio(capex, revenue) * 100,
        "Dividend payout ratio (%)": ratio(dividends, net) * 100,
        "Price to earnings, P/E (x)": ratio(market_cap, net),
        "Price to book, P/B (x)": ratio(market_cap, equity),
        "Price to sales, P/S (x)": ratio(market_cap, revenue),
        "EV to EBITDA (x)": ratio(ev, ebitda),
        "EV to revenue (x)": ratio(ev, revenue),
        "Free cash flow yield (%)": ratio(fcf, market_cap) * 100,
    }
    table = pd.DataFrame(rows, index=idx).T.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    table = table.dropna(axis=1, how="all")
    table.columns = [c.strftime("%Y-%m-%d") for c in table.columns]
    table.index.name = "Ratio"
    return table


# ---------------------------------------------------------------------------
# More valuation models
# ---------------------------------------------------------------------------
def solve_increasing(fn, target: float, lo: float, hi: float) -> float | None:
    """Bisection for fn(x) = target where fn rises with x. None if the target is out of range."""
    try:
        f_lo, f_hi = fn(lo) - target, fn(hi) - target
    except (ValueError, ZeroDivisionError):
        return None
    if not (np.isfinite(f_lo) and np.isfinite(f_hi)) or f_lo > 0 or f_hi < 0:
        return None
    for _ in range(200):
        mid = (lo + hi) / 2
        if fn(mid) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-9:
            break
    return (lo + hi) / 2


IMPLIED_GROWTH_RANGE = (-0.9, 5.0)


def implied_growth(params: dict, target_per_share: float) -> float | None:
    """Year-1 growth that makes the DCF value per share equal the target (other inputs unchanged)."""
    if params["base_cf"] <= 0 or not params.get("shares"):
        return None
    return solve_increasing(lambda g: run_dcf(**{**params, "growth": g})["per_share"], target_per_share,
                            IMPLIED_GROWTH_RANGE[0], IMPLIED_GROWTH_RANGE[1])


def implied_wacc(params: dict, target_per_share: float) -> float | None:
    """Discount rate that makes the DCF value per share equal the target (the return the price implies)."""
    if params["base_cf"] <= 0 or not params.get("shares"):
        return None
    lo = params["terminal_growth"] + 0.0005
    return solve_increasing(lambda w: -run_dcf(**{**params, "wacc": w})["per_share"], -target_per_share, lo, 0.6)


SCENARIO_DEFAULTS = pd.DataFrame({
    "Scenario": ["Bear", "Base", "Bull"],
    "Probability (%)": [25.0, 50.0, 25.0],
    "Growth change (pts)": [-5.0, 0.0, 5.0],
    "WACC change (pts)": [1.0, 0.0, -1.0],
    "Terminal growth change (pts)": [-0.5, 0.0, 0.5],
})


def scenario_results(params: dict, edits: pd.DataFrame, factor: float, price: float | None) -> tuple[pd.DataFrame, float | None]:
    rows = []
    for _, r in edits.iterrows():
        growth = params["growth"] + (num(r.get("Growth change (pts)")) or 0) / 100
        wacc = params["wacc"] + (num(r.get("WACC change (pts)")) or 0) / 100
        terminal = params["terminal_growth"] + (num(r.get("Terminal growth change (pts)")) or 0) / 100
        value = float("nan")
        if wacc > terminal and params.get("shares"):
            value = run_dcf(**{**params, "growth": growth, "wacc": wacc, "terminal_growth": terminal})["per_share"] * factor
        rows.append({"Scenario": str(r.get("Scenario") or ""), "Probability (%)": num(r.get("Probability (%)")) or 0.0,
                     "Year-1 growth (%)": growth * 100, "WACC (%)": wacc * 100, "Terminal growth (%)": terminal * 100,
                     "Value per share": value,
                     "Vs last price (%)": (value / price - 1) * 100 if (price and np.isfinite(value)) else float("nan")})
    table = pd.DataFrame(rows)
    usable = table[np.isfinite(table["Value per share"]) & (table["Probability (%)"] > 0)]
    weighted = None
    if len(usable) and usable["Probability (%)"].sum() > 0:
        weighted = float((usable["Value per share"] * usable["Probability (%)"]).sum() / usable["Probability (%)"].sum())
    return table, weighted


def run_ddm(d0: float, growth: float, years: int, terminal_growth: float, ke: float, fade: bool) -> dict:
    """Two-stage dividend discount model: value per share from the dividends it's expected to pay."""
    if ke <= terminal_growth:
        raise ValueError("The cost of equity must be higher than the terminal growth rate.")
    rows, dividend = [], d0
    for i in range(1, years + 1):
        g = growth + (terminal_growth - growth) * (i - 1) / (years - 1) if (fade and years > 1) else growth
        dividend *= 1 + g
        factor = 1 / (1 + ke) ** i
        rows.append({"Year": i, "Growth (%)": g * 100, "Dividend per share": dividend, "Discount factor": factor,
                     "Present value": dividend * factor})
    projection = pd.DataFrame(rows)
    terminal = dividend * (1 + terminal_growth) / (ke - terminal_growth)
    pv_terminal = terminal / (1 + ke) ** years
    return {"projection": projection, "terminal_value": terminal, "pv_terminal": pv_terminal,
            "value": float(projection["Present value"].sum()) + pv_terminal}


def dividend_growth_5y(daily: pd.DataFrame) -> float | None:
    divs = nonzero(daily, "Dividends")
    if divs.empty:
        return None
    annual = divs.groupby(divs.index.year).sum()
    complete = annual[annual.index < daily.index.max().year]
    if not len(complete):
        return None
    last = complete.index[-1]
    if (last - 5) in complete.index and complete.loc[last - 5] > 0 and complete.loc[last] > 0:
        return (complete.loc[last] / complete.loc[last - 5]) ** 0.2 - 1
    return None


# ---------------------------------------------------------------------------
# Comparing tickers
# ---------------------------------------------------------------------------
def compare_snapshot(symbols: list, infos: dict, histories: dict, rf: float) -> pd.DataFrame:
    rows = []
    for s in symbols:
        info, daily = infos.get(s) or {}, histories.get(s)
        adj = adjusted_close(daily) if daily is not None else pd.Series(dtype=float)
        last = num(daily["Close"].dropna().iloc[-1]) if daily is not None and len(daily["Close"].dropna()) else None
        qt = str(info.get("quoteType") or "").upper()

        def pct(key):
            v = num(info.get(key))
            return v * 100 if v is not None else float("nan")

        def ret(years):
            if len(adj) < 2:
                return float("nan")
            total, ann = period_return(adj, pd.DateOffset(years=years))
            return (total if years == 1 else ann) * 100

        def risk(years, column):
            if len(adj) < 2:
                return float("nan")
            sub = adj[adj.index >= adj.index[-1] - pd.DateOffset(years=years)]
            return risk_stats(sub, rf).get(column, float("nan"))

        paid = ttm_dividends(daily) if daily is not None else 0.0
        rows.append({
            "Ticker": display_symbol(s), "Name": str(info.get("shortName") or info.get("longName") or display_symbol(s)),
            "Type": TYPE_LABELS.get(qt, qt.title()), "Currency": str(info.get("currency") or ""),
            "Price": last, "Market cap or fund assets": num(info.get("marketCap")) or num(info.get("totalAssets")),
            "P/E, trailing": num(info.get("trailingPE")), "P/E, forward": num(info.get("forwardPE")),
            "P/B": num(info.get("priceToBook")), "EV/EBITDA": num(info.get("enterpriseToEbitda")),
            "EV/revenue": num(info.get("enterpriseToRevenue")), "Operating margin (%)": pct("operatingMargins"),
            "Profit margin (%)": pct("profitMargins"), "ROE (%)": pct("returnOnEquity"),
            "Revenue growth (%)": pct("revenueGrowth"),
            "Dividend yield, 12 months (%)": (paid / last * 100) if last else float("nan"),
            "Beta, per Yahoo": num(info.get("beta")), "Return, 1 year (%)": ret(1),
            "Return, 3 years a year (%)": ret(3), "Return, 5 years a year (%)": ret(5),
            "Volatility, 1 year (%)": risk(1, "Annualized volatility (%)"),
            "Max drawdown, 5 years (%)": risk(5, "Max drawdown (%)"),
        })
    table = pd.DataFrame(rows).set_index("Ticker")
    numeric = table.select_dtypes("number").columns
    table[numeric] = table[numeric].astype(float)
    return table


def with_peer_median(table: pd.DataFrame, main: str) -> pd.DataFrame:
    peers = table.drop(index=main, errors="ignore")
    if len(peers) < 2:
        return table
    median = peers.select_dtypes("number").median()
    row = pd.DataFrame([{**{c: "" for c in table.columns}, **median.to_dict(), "Name": "Median of the others"}],
                       index=["Peer median"])
    return pd.concat([table, row[table.columns]])


def monthly_correlations(series: dict, months: int = 60) -> pd.DataFrame:
    monthly = {}
    for sym, s in series.items():
        m = s.groupby(s.index.to_period("M")).last()
        monthly[sym] = (m / m.shift(1) - 1)
    frame = pd.DataFrame(monthly).dropna(how="all").iloc[-months:]
    return frame.corr(min_periods=12)


# ---------------------------------------------------------------------------
# Earnings
# ---------------------------------------------------------------------------
def tidy_earnings(dates_df, history_df) -> pd.DataFrame:
    """Past and upcoming earnings: EPS estimate, reported EPS and surprise, newest first."""
    def pick(df, *keys):
        return next((c for c in df.columns if any(k in str(c).lower() for k in keys)), None)

    for df in (dates_df, history_df):
        if isinstance(df, pd.DataFrame) and not df.empty:
            est, rep = pick(df, "estimate"), pick(df, "reported", "actual")
            if est is None and rep is None:
                continue
            out = pd.DataFrame(index=_naive_index(pd.to_datetime(df.index, errors="coerce")))
            out["EPS estimate"] = pd.to_numeric(df[est], errors="coerce").to_numpy() if est else np.nan
            out["Reported EPS"] = pd.to_numeric(df[rep], errors="coerce").to_numpy() if rep else np.nan
            surprise = (out["Reported EPS"] - out["EPS estimate"]) / out["EPS estimate"].abs() * 100
            out["Surprise (%)"] = surprise.where(out["EPS estimate"].abs() > 0)
            out = out[out.index.notna()]
            out.index.name = "Earnings date"
            return out[~out.index.duplicated()].sort_index(ascending=False)
    return pd.DataFrame()


def calendar_facts(calendar) -> dict:
    if not isinstance(calendar, dict):
        return {}
    dates = calendar.get("Earnings Date") or []
    if not isinstance(dates, (list, tuple)):
        dates = [dates]
    dates = [pd.Timestamp(d) for d in dates if d is not None]
    return {
        "next_date": min(dates) if dates else None,
        "last_possible": max(dates) if len(dates) > 1 else None,
        "eps_avg": num(calendar.get("Earnings Average")), "eps_low": num(calendar.get("Earnings Low")),
        "eps_high": num(calendar.get("Earnings High")), "revenue_avg": num(calendar.get("Revenue Average")),
        "ex_dividend": calendar.get("Ex-Dividend Date"), "dividend_date": calendar.get("Dividend Date"),
    }


# ---------------------------------------------------------------------------
# Charts: one style everywhere, with colors for light and dark mode
# ---------------------------------------------------------------------------
PALETTES = {
    "light": {"line": "#5B45B0", "up": "#1E8E5A", "down": "#D6452F", "muted": "#8A8F98", "fast": "#C98A1B",
              "slow": "#2A8C9A", "band": "#7C6BC4", "grid": "rgba(110, 104, 140, 0.16)", "accent": "#6B4FBB",
              "accent_soft": "rgba(107, 79, 187, 0.28)", "ring": "#FFFFFF"},
    "dark": {"line": "#B79CF0", "up": "#3CC48A", "down": "#F0705A", "muted": "#9AA0AB", "fast": "#E2B04A",
             "slow": "#5CC6D3", "band": "#A99AE6", "grid": "rgba(200, 196, 224, 0.13)", "accent": "#B79CF0",
             "accent_soft": "rgba(183, 156, 240, 0.32)", "ring": "#14121C"},
}
CHART_FONT = '"IBM Plex Sans", "Source Sans Pro", sans-serif'


def theme_mode() -> str:
    """'light' or 'dark', following the theme the viewer picked (light when Streamlit can't tell)."""
    try:
        theme = st.context.theme
    except Exception:
        return "light"
    for attr in ("type", "base"):
        value = getattr(theme, attr, None)
        if value is None and hasattr(theme, "get"):
            value = theme.get(attr)
        if value in ("light", "dark"):
            return value
    return "light"


def palette() -> dict:
    return PALETTES[theme_mode()]


def series_colors() -> list:
    p = palette()
    return [p["line"], p["fast"], p["up"], p["down"], p["slow"], p["band"], p["muted"], "#B5651D"]


def style_figure(fig: go.Figure, height: int, *, hover: str = "x unified", dates: bool = True, legend: bool = True,
                 **extra) -> go.Figure:
    p = palette()
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=36 if legend else 12, b=8), hovermode=hover,
                      font=dict(family=CHART_FONT, size=13),
                      hoverlabel=dict(font=dict(family=CHART_FONT, size=13), namelength=-1),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title_text=""),
                      showlegend=legend, **extra)
    spikes = {"hoverformat": "%b %d, %Y", "showspikes": True, "spikemode": "across", "spikesnap": "cursor",
              "spikethickness": 1, "spikedash": "dot", "spikecolor": p["muted"]} if dates else {}
    fig.update_xaxes(showgrid=False, ticks="outside", ticklen=4, **spikes)
    fig.update_yaxes(gridcolor=p["grid"], zeroline=False)
    return fig


def chart_config(name: str) -> dict:
    """Plotly toolbar: no clutter, and the camera button saves a sharp PNG named after the chart."""
    return {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d", "toggleSpikelines"],
            "toImageButtonOptions": {"format": "png", "filename": safe_filename(name), "scale": 2}}


def show_chart(fig: go.Figure, name: str, symbol: str = ""):
    st.plotly_chart(fig, config=chart_config(f"{display_symbol(symbol)}_{name}" if symbol else name), key=f"chart_{name}")


def growth_figure(s: pd.Series, bench: pd.Series | None, symbol: str, benchmark: str) -> go.Figure:
    p = palette()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=10000 * s / s.iloc[0], name=symbol, mode="lines",
                             line=dict(color=p["line"], width=2), hovertemplate="%{y:,.0f}"))
    if bench is not None:
        b = bench[(bench.index >= s.index[0]) & (bench.index <= s.index[-1])]
        if len(b) > 1:
            fig.add_trace(go.Scatter(x=b.index, y=10000 * b / b.iloc[0], name=benchmark, mode="lines",
                                     line=dict(color=p["muted"], width=1.5), hovertemplate="%{y:,.0f}"))
    fig.update_yaxes(title_text="Value of 10,000 invested")
    return style_figure(fig, 380)


def drawdown_figure(s: pd.Series) -> go.Figure:
    p = palette()
    dd = (s / s.cummax() - 1) * 100
    fig = go.Figure(go.Scatter(x=dd.index, y=dd, name="Below previous peak", mode="lines", fill="tozeroy",
                               line=dict(color=p["down"], width=1), hovertemplate="%{y:.1f}%"))
    fig.update_yaxes(title_text="Below previous peak (%)")
    return style_figure(fig, 260, legend=False)


def bar_figure(labels, values, title: str, color=None, horizontal: bool = True, fmt: str = ",.2f") -> go.Figure:
    color = color or palette()["line"]
    if horizontal:
        fig = go.Figure(go.Bar(x=list(values)[::-1], y=list(labels)[::-1], orientation="h", marker_color=color,
                               hovertemplate=f"%{{x:{fmt}}}<extra></extra>"))
        fig.update_xaxes(title_text=title)
        height = max(220, 28 * len(labels) + 60)
    else:
        fig = go.Figure(go.Bar(x=list(labels), y=list(values), marker_color=color,
                               hovertemplate=f"%{{y:{fmt}}}<extra></extra>"))
        fig.update_yaxes(title_text=title)
        height = 320
    return style_figure(fig, height, hover="closest", dates=False, legend=False)


def sensitivity_figure(grid: pd.DataFrame, price: float | None) -> go.Figure:
    z = grid.to_numpy(dtype=float)
    colour = (z / price - 1) * 100 if price else np.zeros_like(z)
    text_values = [["n/a" if not np.isfinite(v) else f"{v:,.2f}" for v in row] for row in z]
    fig = go.Figure(go.Heatmap(
        z=colour, x=list(grid.columns), y=list(grid.index), text=text_values, texttemplate="%{text}",
        colorscale=[[0, "#E8A39C"], [0.5, "#F4F4F2"], [1, "#9FD3B5"]], zmid=0, showscale=False,
        textfont=dict(color="#1A1A1A", family=CHART_FONT),
        hovertemplate="WACC %{y}<br>Terminal growth %{x}<br>Value per share %{text}<extra></extra>"))
    fig.update_xaxes(title_text="Terminal growth", side="top")
    fig.update_yaxes(title_text="WACC", autorange="reversed")
    return style_figure(fig, 300, hover="closest", dates=False, legend=False)


def price_figure(view: pd.DataFrame, indicators: pd.DataFrame, style: str, log_scale: bool, currency: str,
                 bar_word: str) -> go.Figure:
    p = palette()
    ind = indicators.reindex(view.index) if indicators is not None else pd.DataFrame(index=view.index)
    has_volume = "Volume" in view and view["Volume"].fillna(0).sum() > 0
    panels = ["price"] + (["volume"] if has_volume else []) + (["rsi"] if "RSI 14" in ind else []) \
        + (["macd"] if "MACD" in ind else [])
    weights = {"price": 3.4, "volume": 0.9, "rsi": 1.15, "macd": 1.25}
    total = sum(weights[x] for x in panels)
    fig = make_subplots(rows=len(panels), cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[weights[x] / total for x in panels])
    row = {x: i + 1 for i, x in enumerate(panels)}

    if style == "Candles":
        fig.add_trace(go.Candlestick(
            x=view.index, open=view["Open"], high=view["High"], low=view["Low"], close=view["Close"],
            name="Price", increasing_line_color=p["up"], increasing_fillcolor=p["up"],
            decreasing_line_color=p["down"], decreasing_fillcolor=p["down"], showlegend=False), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=view.index, y=view["Close"], name="Close", mode="lines",
                                 line=dict(color=p["line"], width=1.8), hovertemplate="%{y:,.2f}"), row=1, col=1)
        if "Adj Close" in view:
            fig.add_trace(go.Scatter(x=view.index, y=view["Adj Close"], name="Adj close", mode="lines",
                                     line=dict(color=p["muted"], width=1.2, dash="dot"), hovertemplate="%{y:,.2f}"),
                          row=1, col=1)
    for col, color, name in (("SMA 50", p["fast"], f"50-{bar_word} average"), ("SMA 200", p["slow"], f"200-{bar_word} average")):
        if col in ind and ind[col].notna().any():
            fig.add_trace(go.Scatter(x=view.index, y=ind[col], name=name, mode="lines",
                                     line=dict(color=color, width=1.4), hovertemplate="%{y:,.2f}"), row=1, col=1)
    if "Bollinger upper" in ind:
        for col, dash in (("Bollinger upper", "dot"), ("Bollinger middle", "dash"), ("Bollinger lower", "dot")):
            fig.add_trace(go.Scatter(x=view.index, y=ind[col], name=col, mode="lines",
                                     line=dict(color=p["band"], width=1, dash=dash), hovertemplate="%{y:,.2f}"),
                          row=1, col=1)
    if has_volume:
        colors = np.where(view["Close"] >= view["Open"], p["up"], p["down"]).tolist()
        fig.add_trace(go.Bar(x=view.index, y=view["Volume"], name="Volume", marker_color=colors, opacity=0.55,
                             showlegend=False, hovertemplate="%{y:,.0f}"), row=row["volume"], col=1)
        fig.update_yaxes(title_text="Volume", row=row["volume"], col=1)
    if "rsi" in row:
        r = row["rsi"]
        fig.add_trace(go.Scatter(x=view.index, y=ind["RSI 14"], name="RSI (14)", mode="lines",
                                 line=dict(color=p["slow"], width=1.4), hovertemplate="%{y:.1f}"), row=r, col=1)
        for level in (70, 30):
            fig.add_hline(y=level, line_dash="dot", line_color=p["muted"], line_width=1, row=r, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=r, col=1)
    if "macd" in row:
        r = row["macd"]
        hist_colors = np.where(ind["MACD histogram"].fillna(0) >= 0, p["up"], p["down"]).tolist()
        fig.add_trace(go.Bar(x=view.index, y=ind["MACD histogram"], name="MACD histogram", marker_color=hist_colors,
                             opacity=0.5, showlegend=False, hovertemplate="%{y:,.3f}"), row=r, col=1)
        fig.add_trace(go.Scatter(x=view.index, y=ind["MACD"], name="MACD", mode="lines",
                                 line=dict(color=p["line"], width=1.4), hovertemplate="%{y:,.3f}"), row=r, col=1)
        fig.add_trace(go.Scatter(x=view.index, y=ind["MACD signal"], name="MACD signal", mode="lines",
                                 line=dict(color=p["fast"], width=1.4), hovertemplate="%{y:,.3f}"), row=r, col=1)
        fig.update_yaxes(title_text="MACD", row=r, col=1)
    fig.update_yaxes(title_text=f"Price ({currency})" if currency else "Price",
                     type="log" if log_scale else "linear", row=1, col=1)
    return style_figure(fig, 560 + 150 * (len(panels) - 1), xaxis_rangeslider_visible=False, bargap=0)


def compare_figure(series: dict) -> go.Figure:
    colors = series_colors()
    fig = go.Figure()
    for i, (sym, s) in enumerate(series.items()):
        fig.add_trace(go.Scatter(x=s.index, y=10000 * s / s.iloc[0], name=sym, mode="lines",
                                 line=dict(color=colors[i % len(colors)], width=2.4 if i == 0 else 1.4),
                                 hovertemplate="%{y:,.0f}"))
    fig.update_yaxes(title_text="Value of 10,000 invested")
    return style_figure(fig, 420)


def earnings_figure(table: pd.DataFrame) -> go.Figure:
    p = palette()
    past = table.dropna(subset=["Reported EPS"]).sort_index().tail(12)
    labels = [f"{d:%b %Y}" for d in past.index]
    beat = [p["up"] if (np.isfinite(e) and r >= e) else (p["down"] if np.isfinite(e) else p["line"])
            for r, e in zip(past["Reported EPS"], past["EPS estimate"])]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=past["EPS estimate"], name="Analysts expected", marker_color=p["muted"],
                         opacity=0.55, hovertemplate="%{y:.2f}"))
    fig.add_trace(go.Bar(x=labels, y=past["Reported EPS"], name="Reported", marker_color=beat, hovertemplate="%{y:.2f}"))
    fig.update_yaxes(title_text="Earnings per share")
    return style_figure(fig, 340, dates=False, barmode="group")


def quarterly_figure(quarters: pd.DataFrame, unit_label: str) -> go.Figure:
    p = palette()
    labels = [f"{d:%b %Y}" for d in quarters.index]
    fig = go.Figure()
    for col, color in (("Revenue", p["line"]), ("Net income", p["up"])):
        if col in quarters and quarters[col].notna().any():
            fig.add_trace(go.Bar(x=labels, y=quarters[col], name=col, marker_color=color, hovertemplate="%{y:,.1f}"))
    fig.update_yaxes(title_text=unit_label)
    return style_figure(fig, 340, dates=False, barmode="group")


# ---------------------------------------------------------------------------
# Excel model with working formulas
# ---------------------------------------------------------------------------
XL_YEARS = 15  # forecast columns in the workbook; the "Forecast years" input switches them on


def build_excel_model(m: dict) -> bytes:
    """The DCF as an .xlsx workbook. Every result is a formula fed by the Inputs sheet."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.datavalidation import DataValidation

    scale = m["scale"]
    money = '#,##0;(#,##0);"-"' if scale == 1 else '#,##0.0;(#,##0.0);"-"'
    pct = '0.0%;(0.0%);"-"'
    rate = '0.00%;(0.00%);"-"'
    per_share = '#,##0.00;(#,##0.00);"-"'
    factor_fmt = '0.0000;(0.0000);"-"'
    whole = "0"
    unit_word = {"Raw": "", "Thousands": "thousands", "Millions": "millions", "Billions": "billions"}[m["units_label"]]
    unit = f"{unit_word} of {m['fin_ccy']}".strip() if unit_word else m["fin_ccy"]
    share_unit = unit_word or "number"
    colors = {"input": "0000FF", "link": "008000", "calc": "000000", "label": "000000"}
    key_fill = PatternFill("solid", fgColor="FFFF00")
    band = PatternFill("solid", fgColor="D9E1F2")
    last_col = get_column_letter(2 + XL_YEARS)

    def put(ws, ref, value, kind="calc", fmt=None, bold=False, fill=None, italic=False, right=False):
        cell = ws[ref]
        cell.value = value
        cell.font = Font(name="Arial", size=10, bold=bold, italic=italic, color=colors[kind])
        if right:
            cell.alignment = Alignment(horizontal="right")
        if fmt:
            cell.number_format = fmt
        if fill:
            cell.fill = fill
        return cell

    def heading(ws, row, title, width=3):
        put(ws, f"A{row}", title, "label", bold=True, fill=band)
        for col in range(2, width + 1):
            ws.cell(row=row, column=col).fill = band

    def title(ws, text_line, note):
        put(ws, "A1", text_line, "label", bold=True)
        ws["A1"].font = Font(name="Arial", size=12, bold=True)
        put(ws, "A2", note, "label", italic=True)

    wb = Workbook()

    # Inputs ------------------------------------------------------------------------------
    inp = wb.active
    inp.title = "Inputs"
    title(inp, f"{m['name']} ({m['symbol']}): DCF model inputs",
          f"Blue cells are inputs you can change; black cells are formulas; green cells link to another sheet. "
          f"Money is in {unit}.")
    ref, local = {}, {}
    state = {"row": 4}

    def section(name):
        heading(inp, state["row"], name)
        state["row"] += 1

    def field(key, label, value, kind="input", fmt=None, note=None, key_assumption=False):
        r = state["row"]
        put(inp, f"A{r}", label, "label")
        put(inp, f"B{r}", value, kind, fmt, fill=key_fill if key_assumption else None)
        if note:
            put(inp, f"C{r}", note, "label", italic=True)
        ref[key], local[key] = f"Inputs!$B${r}", f"B{r}"
        state["row"] += 1

    section("Market data")
    field("price", f"Last price ({m['price_ccy']})", m["price"], fmt=per_share,
          note=f"Yahoo Finance close on {m['as_of']}")
    field("fx", f"{m['price_ccy']} per 1 {m['fin_ccy']}", m["fx"], fmt=factor_fmt,
          note="Converts the reporting currency into the trading currency (1 when they're the same)")
    field("shares", f"Shares outstanding ({share_unit})", m["shares"] / scale, fmt=money, note="Yahoo Finance")
    field("debt", f"Total debt ({unit})", m["debt"] / scale, fmt=money, note="Latest balance sheet")
    field("cash", f"Cash and short-term investments ({unit})", m["cash"] / scale, fmt=money, note="Latest balance sheet")
    field("minority", f"Minority interest ({unit})", m["minority"] / scale, fmt=money, note="Latest balance sheet")
    field("preferred", f"Preferred stock ({unit})", m["preferred"] / scale, fmt=money, note="Latest balance sheet")
    field("equity_mv", f"Market value of equity ({unit})", None, kind="calc", fmt=money,
          note="Price divided by the exchange rate, times shares")
    inp[local["equity_mv"]].value = f"={ref['price']}/{ref['fx']}*{ref['shares']}"
    state["row"] += 1
    section("Discount rate")
    field("rf", "Risk-free rate", m["rf"], fmt=rate, note=m["rf_note"])
    field("beta", "Beta", m["beta"], fmt="0.00", note=m["beta_note"])
    field("erp", "Equity risk premium", m["erp"], fmt=rate, key_assumption=True,
          note="An assumption. Damodaran publishes a monthly implied premium at pages.stern.nyu.edu/~adamodar")
    field("kd", "Pre-tax cost of debt", m["kd"], fmt=rate, note=m["kd_note"])
    field("tax", "Tax rate", m["tax"], fmt=rate, note="Latest effective tax rate from the income statement")
    field("manual", "Use the WACC typed below? (1 = yes, 0 = no)", 1 if m["manual_wacc"] else 0, fmt=whole)
    field("wacc_manual", "WACC typed by hand", m["wacc_manual"], fmt=rate, note="Used only when the cell above is 1")
    state["row"] += 1
    section("Cash flow forecast")
    field("base", f"Starting cash flow ({unit})", m["base_cf"] / scale, fmt=money, note=f"{m['kind']}, {m['base_note']}")
    field("growth", "Growth in year 1", m["growth"], fmt=pct, key_assumption=True)
    field("tg", "Terminal growth", m["terminal_growth"], fmt=pct, key_assumption=True,
          note="Growth forever after the forecast. Must be lower than WACC")
    field("fade", "Fade growth to the terminal rate? (1 = yes, 0 = no)", 1 if m["fade"] else 0, fmt=whole)
    field("years", f"Forecast years (3 to {XL_YEARS})", m["years"], fmt=whole)
    field("first_year", "Last reported fiscal year", m["first_year"], fmt=whole)
    ddm = m.get("ddm")
    if ddm:
        state["row"] += 1
        section("Dividend discount model")
        field("d0", f"Dividends per share, last 12 months ({m['price_ccy']})", ddm["d0"], fmt=per_share,
              note="From Yahoo Finance dividend history")
        field("ddm_growth", "Dividend growth in year 1", ddm["growth"], fmt=pct, key_assumption=True)
        field("ddm_tg", "Terminal dividend growth", ddm["terminal_growth"], fmt=pct, key_assumption=True)
        field("ddm_fade", "Fade dividend growth to the terminal rate? (1 = yes, 0 = no)", 1 if ddm["fade"] else 0, fmt=whole)
        field("ddm_years", f"Dividend forecast years (1 to {XL_YEARS})", ddm["years"], fmt=whole)

    def validation(lo, hi, keys, message):
        dv = DataValidation(type="whole", operator="between", formula1=str(lo), formula2=str(hi),
                            showErrorMessage=True, errorTitle="Check this input", error=message)
        inp.add_data_validation(dv)
        for k in keys:
            dv.add(local[k])

    validation(3, XL_YEARS, ["years"], f"Enter a whole number from 3 to {XL_YEARS}.")
    validation(0, 1, ["manual", "fade"] + (["ddm_fade"] if ddm else []), "Enter 1 for yes or 0 for no.")
    if ddm:
        validation(1, XL_YEARS, ["ddm_years"], f"Enter a whole number from 1 to {XL_YEARS}.")
    inp.column_dimensions["A"].width = 52
    inp.column_dimensions["B"].width = 16
    inp.column_dimensions["C"].width = 80

    # One projection block, reused for the DCF, each scenario and the dividend model ----------------------
    def projection(ws, top, *, title_text, base_ref, g_ref, tg_ref, w_ref, years_ref, fade_ref, bridge,
                   flow_label, fiscal_ref=None):
        heading(ws, top, title_text, width=2 + XL_YEARS)
        r = top + 1
        rows = {}
        for key, label in (("year", "Year"), ("fiscal", "Fiscal year"), ("flag", "In the forecast (1 = yes)"),
                           ("growth", "Growth"), ("flow", flow_label), ("df", "Discount factor"),
                           ("pv", "Present value")):
            if key == "fiscal" and not fiscal_ref:
                continue
            put(ws, f"A{r}", label, "label")
            rows[key] = r
            r += 1
        put(ws, f"B{rows['year']}", "Base", "label", right=True)
        put(ws, f"B{rows['flow']}", f"={base_ref}", "link", per_share if not bridge else money)
        if fiscal_ref:
            put(ws, f"B{rows['fiscal']}", f"={fiscal_ref}", "link", whole)
        for j in range(XL_YEARS):
            col, prev = get_column_letter(3 + j), get_column_letter(2 + j)
            put(ws, f"{col}{rows['year']}", j + 1, "label", whole)
            if fiscal_ref:
                put(ws, f"{col}{rows['fiscal']}", f"=$B${rows['fiscal']}+{col}${rows['year']}", "calc", whole)
            put(ws, f"{col}{rows['flag']}", f"=IF({col}${rows['year']}<={years_ref},1,0)", "calc", whole)
            put(ws, f"{col}{rows['growth']}",
                f"=IF({col}${rows['flag']}=1,IF(AND({fade_ref}=1,{years_ref}>1),"
                f"{g_ref}+({tg_ref}-{g_ref})*({col}${rows['year']}-1)/({years_ref}-1),{g_ref}),0)", "calc", pct)
            put(ws, f"{col}{rows['flow']}", f"=IF({col}${rows['flag']}=1,{prev}{rows['flow']}*(1+{col}{rows['growth']}),0)",
                "calc", per_share if not bridge else money)
            put(ws, f"{col}{rows['df']}", f"=IF({col}${rows['flag']}=1,1/(1+{w_ref})^{col}${rows['year']},0)",
                "calc", factor_fmt)
            put(ws, f"{col}{rows['pv']}", f"={col}{rows['flow']}*{col}{rows['df']}", "calc",
                per_share if not bridge else money)
        r += 1
        flow_fmt = per_share if not bridge else money
        out = {"rows": rows}
        lines = [("pv_sum", "Present value of the forecast", f"=SUM(C{rows['pv']}:{last_col}{rows['pv']})", flow_fmt),
                 ("final", "Cash flow in the last forecast year" if bridge else "Dividend in the last forecast year",
                  f"=INDEX(C{rows['flow']}:{last_col}{rows['flow']},1,{years_ref})", flow_fmt),
                 ("tv", "Terminal value", None, flow_fmt), ("pv_tv", "Present value of the terminal value", None, flow_fmt)]
        for key, label, formula, fmt in lines:
            put(ws, f"A{r}", label, "label")
            out[key] = f"$B${r}"
            if formula:
                put(ws, f"B{r}", formula, "calc", fmt)
            r += 1
        put(ws, out["tv"].replace("$", ""), f"=IF({w_ref}>{tg_ref},{out['final']}*(1+{tg_ref})/({w_ref}-{tg_ref}),0)",
            "calc", flow_fmt)
        put(ws, out["pv_tv"].replace("$", ""), f"=IF({w_ref}>{tg_ref},{out['tv']}/(1+{w_ref})^{years_ref},0)",
            "calc", flow_fmt)
        if bridge:
            for key, label, formula, fmt, kind in (
                    ("ev", "Enterprise value", f"={out['pv_sum']}+{out['pv_tv']}", money, "calc"),
                    ("less_debt", "Less total debt", f"={ref['debt']}", money, "link"),
                    ("plus_cash", "Plus cash and short-term investments", f"={ref['cash']}", money, "link"),
                    ("less_minority", "Less minority interest", f"={ref['minority']}", money, "link"),
                    ("less_preferred", "Less preferred stock", f"={ref['preferred']}", money, "link")):
                put(ws, f"A{r}", label, "label")
                put(ws, f"B{r}", formula, kind, fmt)
                out[key] = f"$B${r}"
                r += 1
            put(ws, f"A{r}", "Equity value", "label")
            put(ws, f"B{r}", f"={out['ev']}-{out['less_debt']}+{out['plus_cash']}-{out['less_minority']}-{out['less_preferred']}",
                "calc", money)
            out["equity"] = f"$B${r}"
            r += 1
            put(ws, f"A{r}", f"Value per share ({m['fin_ccy']})", "label")
            put(ws, f"B{r}", f"=IF({ref['shares']}>0,{out['equity']}/{ref['shares']},0)", "calc", per_share)
            out["per_share_fin"] = f"$B${r}"
            r += 1
            put(ws, f"A{r}", f"Value per share ({m['price_ccy']})", "label", bold=True)
            put(ws, f"B{r}", f"={out['per_share_fin']}*{ref['fx']}", "calc", per_share, bold=True)
        else:
            put(ws, f"A{r}", f"Value per share ({m['price_ccy']})", "label", bold=True)
            put(ws, f"B{r}", f"={out['pv_sum']}+{out['pv_tv']}", "calc", per_share, bold=True)
        out["value"] = f"$B${r}"
        r += 1
        put(ws, f"A{r}", f"Last price ({m['price_ccy']})", "label")
        put(ws, f"B{r}", f"={ref['price']}", "link", per_share)
        out["price"] = f"$B${r}"
        r += 1
        put(ws, f"A{r}", "Upside or downside versus the last price", "label")
        put(ws, f"B{r}", f"=IF({out['price']}>0,{out['value']}/{out['price']}-1,0)", "calc", pct)
        out["upside"] = f"$B${r}"
        r += 1
        put(ws, f"A{r}", "Check", "label")
        put(ws, f"B{r}", f'=IF({w_ref}>{tg_ref},"OK","The discount rate must be higher than terminal growth")', "calc",
            right=True)
        out["check"] = f"$B${r}"
        out["end"] = r
        return out

    def widths(ws, first=50):
        ws.column_dimensions["A"].width = first
        for j in range(2, 3 + XL_YEARS):
            ws.column_dimensions[get_column_letter(j)].width = 13

    # DCF -----------------------------------------------------------------------------------
    dcf = wb.create_sheet("DCF")
    title(dcf, f"{m['name']} ({m['symbol']}): discounted cash flow",
          f"Money in {unit}. Everything here is a formula: change the blue cells on the Inputs sheet.")
    heading(dcf, 4, "Discount rate (WACC)")
    wacc_rows = [
        ("ke", "Cost of equity (CAPM: risk-free rate + beta x premium)", f"={ref['rf']}+{ref['beta']}*{ref['erp']}", rate, "calc"),
        ("kd_after", "After-tax cost of debt", f"={ref['kd']}*(1-{ref['tax']})", rate, "calc"),
        ("equity_mv", f"Market value of equity ({unit})", f"={ref['equity_mv']}", money, "link"),
        ("debt", f"Total debt ({unit})", f"={ref['debt']}", money, "link"),
        ("we", "Equity share of capital", None, pct, "calc"),
        ("wd", "Debt share of capital", None, pct, "calc"),
        ("wacc_calc", "WACC, calculated", None, rate, "calc"),
        ("wacc", "WACC used", None, rate, "calc"),
    ]
    cells = {}
    for i, (key, label, formula, fmt, kind) in enumerate(wacc_rows):
        r = 5 + i
        cells[key] = f"$B${r}"
        put(dcf, f"A{r}", label, "label", bold=key == "wacc")
        if formula:
            put(dcf, f"B{r}", formula, kind, fmt)
    put(dcf, cells["we"].replace("$", ""), f"=IF({cells['equity_mv']}+{cells['debt']}>0,{cells['equity_mv']}/({cells['equity_mv']}+{cells['debt']}),1)", "calc", pct)
    put(dcf, cells["wd"].replace("$", ""), f"=1-{cells['we']}", "calc", pct)
    put(dcf, cells["wacc_calc"].replace("$", ""), f"={cells['we']}*{cells['ke']}+{cells['wd']}*{cells['kd_after']}", "calc", rate)
    put(dcf, cells["wacc"].replace("$", ""), f"=IF({ref['manual']}=1,{ref['wacc_manual']},{cells['wacc_calc']})", "calc", rate, bold=True)
    dcf_out = projection(dcf, 14, title_text="Cash flow forecast and valuation", base_ref=ref["base"], g_ref=ref["growth"],
                         tg_ref=ref["tg"], w_ref=f"DCF!{cells['wacc']}", years_ref=ref["years"], fade_ref=ref["fade"],
                         bridge=True, flow_label=f"{m['kind']} ({unit})", fiscal_ref=ref["first_year"])
    r = dcf_out["end"] + 1
    put(dcf, f"A{r}", "Terminal value as a share of enterprise value", "label")
    put(dcf, f"B{r}", f"=IF({dcf_out['ev']}<>0,{dcf_out['pv_tv']}/{dcf_out['ev']},0)", "calc", pct)
    widths(dcf)
    dcf.freeze_panes = "C4"

    # Sensitivity -------------------------------------------------------------------------------
    sen = wb.create_sheet("Sensitivity")
    title(sen, "Sensitivity of value per share",
          f"Value per share in {m['price_ccy']}. Rows move WACC and columns move terminal growth, half a point at a time.")
    steps = [-0.01, -0.005, 0.0, 0.005, 0.01]
    put(sen, "A4", "WACC (down) and terminal growth (across)", "label", bold=True)
    year_row, flag_row = dcf_out["rows"]["year"], dcf_out["rows"]["flag"]
    for j, step in enumerate(steps):
        put(sen, f"{get_column_letter(2 + j)}4", f"={ref['tg']}{step:+.3f}", "calc", rate, bold=True)
    for i, step in enumerate(steps):
        put(sen, f"A{5 + i}", f"=DCF!{cells['wacc']}{step:+.3f}", "calc", rate, bold=True)
    # Working rows. With fading switched on, terminal growth also shapes the forecast path,
    # so each terminal-growth column gets its own growth and cash flow rows.
    work = 13
    put(sen, f"A{work}", "Working: forecast for each terminal growth rate, and discount factors for each WACC",
        "label", italic=True)
    growth_rows, flow_rows, df_rows = [], [], []
    r = work + 1
    for j in range(len(steps)):
        tg_cell = f"${get_column_letter(2 + j)}$4"
        put(sen, f"A{r}", f"Growth, terminal growth column {j + 1}", "label")
        put(sen, f"A{r + 1}", f"Cash flow, terminal growth column {j + 1}", "label")
        put(sen, f"B{r + 1}", f"={ref['base']}", "link", money)
        for k in range(XL_YEARS):
            col, prev = get_column_letter(3 + k), get_column_letter(2 + k)
            put(sen, f"{col}{r}",
                f"=IF(DCF!{col}${flag_row}=1,IF(AND({ref['fade']}=1,{ref['years']}>1),{ref['growth']}+({tg_cell}-"
                f"{ref['growth']})*(DCF!{col}${year_row}-1)/({ref['years']}-1),{ref['growth']}),0)", "calc", pct)
            put(sen, f"{col}{r + 1}", f"=IF(DCF!{col}${flag_row}=1,{prev}{r + 1}*(1+{col}{r}),0)", "calc", money)
        growth_rows.append(r)
        flow_rows.append(r + 1)
        r += 2
    for i in range(len(steps)):
        put(sen, f"A{r}", f"Discount factor, WACC row {i + 1}", "label")
        put(sen, f"B{r}", f"=$A${5 + i}", "calc", rate)
        for k in range(XL_YEARS):
            col = get_column_letter(3 + k)
            put(sen, f"{col}{r}", f"=IF(DCF!{col}${flag_row}=1,1/(1+$B{r})^DCF!{col}${year_row},0)", "calc", factor_fmt)
        df_rows.append(r)
        r += 1
    for i in range(len(steps)):
        for j in range(len(steps)):
            cell = f"{get_column_letter(2 + j)}{5 + i}"
            w_cell, tg_cell = f"$A${5 + i}", f"{get_column_letter(2 + j)}$4"
            flows = f"$C${flow_rows[j]}:${last_col}${flow_rows[j]}"
            put(sen, cell,
                f"=IF({w_cell}>{tg_cell},(SUMPRODUCT({flows},$C${df_rows[i]}:${last_col}${df_rows[i]})"
                f"+INDEX({flows},1,{ref['years']})*(1+{tg_cell})/({w_cell}-{tg_cell})/(1+{w_cell})^{ref['years']}"
                f"-{ref['debt']}+{ref['cash']}-{ref['minority']}-{ref['preferred']})/{ref['shares']}*{ref['fx']},0)",
                "calc", per_share)
    widths(sen, 44)

    # Scenarios ------------------------------------------------------------------------------------
    sc = wb.create_sheet("Scenarios")
    title(sc, "Bear, base and bull scenarios",
          "Change the blue cells: each scenario moves year-1 growth, WACC and terminal growth away from the Inputs sheet.")
    names = list(m["scenarios"]["Scenario"])
    for j, name in enumerate(names):
        put(sc, f"{get_column_letter(2 + j)}4", name, "label", bold=True, right=True)
    edit_rows = [("Probability", "Probability (%)"), ("Change to year-1 growth", "Growth change (pts)"),
                 ("Change to WACC", "WACC change (pts)"), ("Change to terminal growth", "Terminal growth change (pts)")]
    for i, (label, column) in enumerate(edit_rows):
        r = 5 + i
        put(sc, f"A{r}", label, "label")
        for j, value in enumerate(m["scenarios"][column]):
            put(sc, f"{get_column_letter(2 + j)}{r}", (num(value) or 0.0) / 100, "input", pct)
    derived = [("Year-1 growth", lambda c: f"={ref['growth']}+{c}6"), ("WACC", lambda c: f"=DCF!{cells['wacc']}+{c}7"),
               ("Terminal growth", lambda c: f"={ref['tg']}+{c}8")]
    for i, (label, formula) in enumerate(derived):
        r = 10 + i
        put(sc, f"A{r}", label, "label")
        for j in range(len(names)):
            col = get_column_letter(2 + j)
            put(sc, f"{col}{r}", formula(col), "calc", rate if label == "WACC" else pct)
    put(sc, "A13", f"Value per share ({m['price_ccy']})", "label", bold=True)
    put(sc, "A14", "Upside or downside versus the last price", "label")
    put(sc, "A15", "Check", "label")
    top = 20
    for j, name in enumerate(names):
        col = get_column_letter(2 + j)
        block = projection(sc, top, title_text=f"Working: {name} scenario", base_ref=ref["base"], g_ref=f"${col}$10",
                           tg_ref=f"${col}$12", w_ref=f"${col}$11", years_ref=ref["years"], fade_ref=ref["fade"],
                           bridge=True, flow_label=f"Cash flow ({unit})")
        put(sc, f"{col}13", f"={block['value']}", "calc", per_share, bold=True)
        put(sc, f"{col}14", f"={block['upside']}", "calc", pct)
        put(sc, f"{col}15", f"={block['check']}", "calc", right=True)
        top = block["end"] + 3
    last_sc = get_column_letter(1 + len(names))
    put(sc, "A17", f"Probability-weighted value per share ({m['price_ccy']})", "label", bold=True)
    put(sc, "B17", f"=IF(SUM(B5:{last_sc}5)>0,SUMPRODUCT(B5:{last_sc}5,B13:{last_sc}13)/SUM(B5:{last_sc}5),0)",
        "calc", per_share, bold=True)
    put(sc, "A18", "Upside or downside versus the last price", "label")
    put(sc, "B18", f"=IF({ref['price']}>0,B17/{ref['price']}-1,0)", "calc", pct)
    widths(sc, 44)

    # Dividend model -------------------------------------------------------------------------------
    if ddm:
        dd = wb.create_sheet("DDM")
        title(dd, "Dividend discount model",
              f"Per share, in {m['price_ccy']}. Discounted at the cost of equity from the DCF sheet. Companies that "
              "also buy back shares return more cash than dividends show, so this tends to undervalue them.")
        put(dd, "A4", "Cost of equity", "label", bold=True)
        put(dd, "B4", f"=DCF!{cells['ke']}", "link", rate, bold=True)
        projection(dd, 6, title_text="Dividend forecast and value", base_ref=ref["d0"], g_ref=ref["ddm_growth"],
                   tg_ref=ref["ddm_tg"], w_ref="$B$4", years_ref=ref["ddm_years"], fade_ref=ref["ddm_fade"],
                   bridge=False, flow_label=f"Dividend per share ({m['price_ccy']})")
        widths(dd, 44)

    # Historical ---------------------------------------------------------------------------------------
    hist_ws = wb.create_sheet("Historical")
    history = m["history"]
    title(hist_ws, "Historical figures used for the forecast",
          f"Source: Yahoo Finance annual statements through yfinance, downloaded {m['downloaded']}. Money in {unit}.")
    put(hist_ws, "A4", "Item", "label", bold=True)
    for j, when in enumerate(history.index):
        put(hist_ws, f"{get_column_letter(2 + j)}4", f"{when:%Y-%m-%d}", "label", bold=True)
    for i, column in enumerate(history.columns):
        r = 5 + i
        is_pct = column in PCT_ROWS
        put(hist_ws, f"A{r}", column.replace(" (%)", ""), "label")
        for j, value in enumerate(history[column]):
            v = num(value)
            if v is not None:
                put(hist_ws, f"{get_column_letter(2 + j)}{r}", v / 100 if is_pct else v / scale, "input",
                    pct if is_pct else money)
    hist_ws.column_dimensions["A"].width = 44
    for j in range(len(history.index)):
        hist_ws.column_dimensions[get_column_letter(2 + j)].width = 14

    # Notes ---------------------------------------------------------------------------------------------
    notes = wb.create_sheet("Notes")
    lines = [
        ("How to use this workbook", True),
        ("Change the blue cells on the Inputs sheet (and the scenario cells on Scenarios). Every other number recalculates.", False),
        ("Yellow cells are the assumptions that move the value most.", False),
        (f"Forecast years can be anything from 3 to {XL_YEARS}: years past your choice show zeros and drop out.", False),
        ("", False),
        ("Reverse DCF in Excel", True),
        ("Data > What-If Analysis > Goal Seek. Set cell DCF value per share (the bold cell on the DCF sheet) to the last "
         "price, by changing Inputs 'Growth in year 1'. The answer is the growth the current price implies.", False),
        ("", False),
        ("Colors", True),
        ("Blue text: inputs. Black text: formulas. Green text: links to another sheet.", False),
        ("", False),
        ("Source", True),
        (f"Market and financial statement data from Yahoo Finance via the open-source yfinance library, downloaded "
         f"{m['downloaded']}. Meant for personal, research and educational use.", False),
    ]
    for i, (line, bold) in enumerate(lines):
        put(notes, f"A{1 + i}", line, "label", bold=bold)
    notes.column_dimensions["A"].width = 120

    for ws in wb.worksheets:
        ws.page_setup.orientation = "landscape"
        ws.page_setup.fitToWidth = 1
        ws.page_setup.fitToHeight = 0
        ws.sheet_properties.pageSetUpPr.fitToPage = True
    wb.calculation.fullCalcOnLoad = True  # Excel and Numbers compute every formula when the file opens
    buffer = io.BytesIO()
    wb.save(buffer)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Search list: every U.S.-listed stock and ETF from Nasdaq's public symbol
# directory, plus indices, crypto, currencies, futures and popular funds
# ---------------------------------------------------------------------------
NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
LISTED_SYMBOL = re.compile(r"^[A-Z]{1,5}(\.[A-Z])?$")
SKIP_SECURITY = re.compile(r"\b(warrants?|rights?|units?|notes? due|debentures?|preferred (stock|shares?))\b",
                           re.IGNORECASE)
NAME_SUFFIX = re.compile(r"\s+(common stock|ordinary shares?|common shares?|american depositary (shares?|receipts?))\b.*$",
                         re.IGNORECASE)

POPULAR = {
    "AAPL": "Apple Inc.", "MSFT": "Microsoft Corporation", "NVDA": "NVIDIA Corporation", "AMZN": "Amazon.com, Inc.",
    "GOOGL": "Alphabet Inc. Class A", "GOOG": "Alphabet Inc. Class C", "META": "Meta Platforms, Inc.",
    "TSLA": "Tesla, Inc.", "BRK-B": "Berkshire Hathaway Inc. Class B", "JPM": "JPMorgan Chase & Co.",
    "V": "Visa Inc.", "MA": "Mastercard Incorporated", "UNH": "UnitedHealth Group Incorporated",
    "JNJ": "Johnson & Johnson", "XOM": "Exxon Mobil Corporation", "WMT": "Walmart Inc.",
    "PG": "The Procter & Gamble Company", "KO": "The Coca-Cola Company", "PEP": "PepsiCo, Inc.",
    "COST": "Costco Wholesale Corporation", "HD": "The Home Depot, Inc.", "NFLX": "Netflix, Inc.",
    "AMD": "Advanced Micro Devices, Inc.", "INTC": "Intel Corporation", "DIS": "The Walt Disney Company",
    "BAC": "Bank of America Corporation", "SPY": "SPDR S&P 500 ETF Trust", "VOO": "Vanguard S&P 500 ETF",
    "VTI": "Vanguard Total Stock Market ETF", "QQQ": "Invesco QQQ Trust", "IVV": "iShares Core S&P 500 ETF",
    "VXUS": "Vanguard Total International Stock ETF", "BND": "Vanguard Total Bond Market ETF",
    "AGG": "iShares Core U.S. Aggregate Bond ETF", "GLD": "SPDR Gold Shares", "VT": "Vanguard Total World Stock ETF",
    "SCHD": "Schwab U.S. Dividend Equity ETF",
}
EXTRA_SYMBOLS = {
    "^GSPC": "S&P 500 index", "^DJI": "Dow Jones Industrial Average", "^IXIC": "Nasdaq Composite index",
    "^RUT": "Russell 2000 index", "^VIX": "CBOE Volatility Index", "^TNX": "10-year U.S. Treasury yield",
    "^IRX": "13-week U.S. Treasury bill yield", "^TYX": "30-year U.S. Treasury yield",
    "^FTSE": "FTSE 100 index (UK)", "^GDAXI": "DAX index (Germany)", "^FCHI": "CAC 40 index (France)",
    "^STOXX50E": "Euro Stoxx 50 index", "^N225": "Nikkei 225 index (Japan)", "^HSI": "Hang Seng index (Hong Kong)",
    "000001.SS": "Shanghai Composite index", "^NSEI": "Nifty 50 index (India)", "^BSESN": "BSE Sensex (India)",
    "^GSPTSE": "S&P/TSX Composite index (Canada)", "^AXJO": "S&P/ASX 200 index (Australia)",
    "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "SOL-USD": "Solana",
    "EURUSD=X": "Euro to U.S. dollar", "GBPUSD=X": "British pound to U.S. dollar",
    "USDJPY=X": "U.S. dollar to Japanese yen", "USDINR=X": "U.S. dollar to Indian rupee",
    "USDCAD=X": "U.S. dollar to Canadian dollar", "GC=F": "Gold futures", "SI=F": "Silver futures",
    "CL=F": "Crude oil (WTI) futures", "NG=F": "Natural gas futures",
    "VTSAX": "Vanguard Total Stock Market Index Fund Admiral Shares",
    "VFIAX": "Vanguard 500 Index Fund Admiral Shares", "FXAIX": "Fidelity 500 Index Fund",
    "VTIAX": "Vanguard Total International Stock Index Fund Admiral Shares",
    "VBTLX": "Vanguard Total Bond Market Index Fund Admiral Shares", "SWPPX": "Schwab S&P 500 Index Fund",
}
QUICK_PICKS = [("AAPL", "Apple"), ("VTI", "Vanguard Total Stock Market ETF"), ("^GSPC", "S&P 500"),
               ("BTC-USD", "Bitcoin")]
BENCHMARKS = {
    "^GSPC": "S&P 500", "^IXIC": "Nasdaq Composite", "^DJI": "Dow Jones Industrial Average", "^RUT": "Russell 2000",
    "VT": "Vanguard Total World Stock ETF", "AGG": "U.S. investment-grade bonds", "^FTSE": "FTSE 100 (UK)",
    "^GDAXI": "DAX (Germany)", "^N225": "Nikkei 225 (Japan)", "^HSI": "Hang Seng (Hong Kong)",
    "^NSEI": "Nifty 50 (India)", "^BSESN": "BSE Sensex (India)", "^GSPTSE": "S&P/TSX Composite (Canada)",
    "^AXJO": "S&P/ASX 200 (Australia)",
}
INDEX_NAMES = {
    "^GSPC": "S&P 500", "^IXIC": "Nasdaq Composite", "^DJI": "Dow Jones Industrial Average", "^RUT": "Russell 2000",
    "^VIX": "Cboe Volatility Index", "^TNX": "10-year Treasury yield", "^IRX": "13-week Treasury bill yield",
    "^TYX": "30-year Treasury yield", "^FTSE": "FTSE 100", "^GDAXI": "DAX", "^FCHI": "CAC 40",
    "^STOXX50E": "Euro Stoxx 50", "^N225": "Nikkei 225", "^HSI": "Hang Seng", "^NSEI": "Nifty 50",
    "^BSESN": "BSE Sensex", "^GSPTSE": "S&P/TSX Composite", "^AXJO": "S&P/ASX 200",
}


def display_symbol(symbol) -> str:
    """A symbol the way people read it, without Yahoo's codes such as ^ for indices or =X for currencies."""
    s = str(symbol)
    if s in INDEX_NAMES:
        return INDEX_NAMES[s]
    if s.startswith("^"):
        return s[1:]
    if s.endswith("=X"):
        core = s[:-2]
        return f"{core[:3]}/{core[3:]}" if len(core) == 6 else core
    if s.endswith("=F"):
        return s[:-2]
    return s


def directory_label(symbol: str, name: str) -> str:
    """How a search suggestion reads: indices and futures by name; currencies, crypto and stocks with a short code."""
    if symbol.startswith("^") or symbol.endswith("=F"):
        return name
    if symbol.endswith("=X"):
        return f"{name} ({display_symbol(symbol)})"
    if symbol.endswith("-USD"):
        return f"{name} ({symbol[:-4]})"
    return f"{name} ({symbol})"


SECTION_SETS = {
    "EQUITY": ["Price", "Performance", "Statements", "Ratios", "Valuation", "Earnings", "Dividends", "Compare", "Profile"],
    "FUND": ["Price", "Performance", "Holdings", "Valuation", "Dividends", "Compare", "Profile"],
    "MARKET": ["Price", "Performance", "Compare", "Profile"],
    "ANY": ["Price", "Performance", "Statements", "Ratios", "Valuation", "Earnings", "Holdings", "Dividends", "Compare",
            "Profile"],
}
PRICE_PERIODS = ["1M", "3M", "6M", "YTD", "1Y", "5Y", "10Y", "Max", "Custom"]
DATA_NOTE = ("Data comes from Yahoo Finance through the open-source yfinance library and is meant for personal, "
             "research and educational use. Prices can be delayed.")


def clean_security_name(name: str) -> str:
    original = str(name).strip()
    short = re.split(r"\s+-\s+", original, maxsplit=1)[0]
    short = NAME_SUFFIX.sub("", short).strip(" ,")
    return short or original


def _fetch_text(url: str) -> str:
    headers = {"User-Agent": f"Mozilla/5.0 ({APP_NAME})"}
    try:
        import requests  # installed with Streamlit; brings its own certificates, which matters on macOS
    except ImportError:
        requests = None
    if requests is not None:
        response = requests.get(url, headers=headers, timeout=8)
        response.raise_for_status()
        return response.text
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=8) as response:
        return response.read().decode("utf-8", "replace")


@st.cache_data(ttl=24 * 3600, show_spinner="Loading the list of tickers (first time only)...")
def load_us_listings() -> dict:
    listings = {}
    for url, column in ((NASDAQ_LISTED_URL, "Symbol"), (OTHER_LISTED_URL, "ACT Symbol")):
        table = pd.read_csv(io.StringIO(_fetch_text(url)), sep="|", dtype=str, keep_default_na=False)
        if "Test Issue" in table:
            table = table[table["Test Issue"] != "Y"]
        for symbol, name in zip(table[column], table["Security Name"]):
            symbol = str(symbol).strip()
            if LISTED_SYMBOL.match(symbol) and not SKIP_SECURITY.search(str(name)):
                listings.setdefault(symbol.replace(".", "-"), clean_security_name(name))
    if len(listings) < 1000:
        raise ValueError("The list of U.S. tickers came back incomplete.")
    return listings


def search_directory() -> tuple[dict, str | None]:
    """Options for the search box as {symbol: label}, and a note if the full list didn't load."""
    listings = None
    if not st.session_state.get("listings_failed"):
        listings, err = attempt(load_us_listings)
        if err:
            st.session_state["listings_failed"] = True  # don't wait on an unreachable list at every click
    note = None
    if not listings:
        listings = {}
        note = ("The full list of U.S. tickers didn't load, so suggestions cover popular names only. "
                "Any ticker still works: type it and press Enter.")
    names = {**POPULAR, **EXTRA_SYMBOLS}
    ordered = list(POPULAR) + list(EXTRA_SYMBOLS) + sorted(s for s in listings if s not in names)
    return {s: directory_label(s, listings.get(s) or names.get(s, s)) for s in ordered}, note


# ---------------------------------------------------------------------------
# Look and feel
# ---------------------------------------------------------------------------
CHART_CONFIG = {"displaylogo": False}
CSS = """<style>
[data-testid="stMainBlockContainer"], .block-container {max-width: 1240px; margin-left: auto; margin-right: auto; padding-top: 4.5rem;}
[data-testid="stAppDeployButton"], .stAppDeployButton {display: none !important;}
.td-brand {display: flex; align-items: center; gap: 0.75rem; font-size: 1.6rem; font-weight: 700; line-height: 1.2; letter-spacing: 0.01em; margin-bottom: 0.75rem;}
.td-logo {height: 3.5rem; width: auto; flex: none;}
.td-name {font-size: 2.15rem; font-weight: 700; line-height: 1.15; margin: 0.4rem 0 0.25rem 0;}
.td-ticker {font-size: 1.15rem; font-weight: 500; opacity: 0.55; margin-left: 0.35rem; letter-spacing: 0.02em;}
.td-sub {font-size: 0.95rem; opacity: 0.7;}
.td-quote {display: flex; flex-wrap: wrap; align-items: baseline; column-gap: 0.75rem; row-gap: 0.25rem; margin: 0.9rem 0 0.2rem 0;}
.td-price {font-size: 2.6rem; font-weight: 600; line-height: 1.05; font-variant-numeric: tabular-nums;}
.td-ccy {font-size: 1rem; opacity: 0.6;}
.td-change {font-size: 1.15rem; font-weight: 600; font-variant-numeric: tabular-nums;}
.td-up {color: var(--td-up, #1E8E5A);}
.td-down {color: var(--td-down, #D6452F);}
.td-flat {opacity: 0.7;}
.td-asof {font-size: 0.85rem; opacity: 0.6;}
.td-stats {display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 0.9rem 1.5rem; margin: 1.1rem 0 1.2rem 0; padding: 0.95rem 0; border-top: 1px solid rgba(128, 128, 128, 0.25); border-bottom: 1px solid rgba(128, 128, 128, 0.25);}
.td-label {font-size: 0.8rem; opacity: 0.65;}
.td-stats-plain {border: none; padding: 0; margin: 0.4rem 0 1rem 0;}
.td-about {font-size: 1rem; line-height: 1.7; text-align: justify; hyphens: auto; -webkit-hyphens: auto; margin: 0 0 0.5rem 0;}
.td-facts {display: grid; gap: 0.9rem; padding: 1rem 1.25rem; border: 1px solid rgba(128, 128, 128, 0.28); border-radius: 0.5rem;}
.td-fact {font-size: 1rem; font-weight: 600; line-height: 1.35; margin-top: 0.1rem; overflow-wrap: anywhere;}
.td-fact a {color: inherit; text-decoration: underline; text-underline-offset: 2px;}
[data-testid="stMetricValue"] {font-size: 1.75rem;}
[data-testid="stMetricValue"] > div {white-space: normal; overflow: visible; text-overflow: clip;}
.td-name, .td-brand {font-family: "Sora", "IBM Plex Sans", "Source Sans Pro", sans-serif;}
.td-name {letter-spacing: -0.01em;}
.td-range {grid-column: span 2; min-width: 0;}
.td-range-track {position: relative; height: 6px; margin: 0.6rem 0 0.4rem 0; border-radius: 3px; background: rgba(128, 128, 128, 0.22);}
.td-range-fill {position: absolute; left: 0; top: 0; bottom: 0; border-radius: 3px; background: var(--td-accent-soft, rgba(107, 79, 187, 0.28));}
.td-range-dot {position: absolute; top: 50%; width: 12px; height: 12px; margin: -6px 0 0 -6px; border-radius: 50%; background: var(--td-accent, #6B4FBB); box-shadow: 0 0 0 2px var(--td-ring, #FFFFFF);}
.td-range-ends {display: flex; justify-content: space-between; font-size: 0.85rem; opacity: 0.75; font-variant-numeric: tabular-nums;}
.td-spark {display: inline-flex; align-items: center; gap: 0.45rem;}
.td-spark img {display: block;}
.td-spark-label {font-size: 0.85rem; opacity: 0.7; font-variant-numeric: tabular-nums;}
.td-skeleton {display: grid; gap: 0.8rem; margin: 0.8rem 0 1.4rem 0;}
.td-skeleton span {display: block; border-radius: 0.4rem; background: linear-gradient(90deg, rgba(128, 128, 128, 0.10) 0%, rgba(128, 128, 128, 0.22) 50%, rgba(128, 128, 128, 0.10) 100%); background-size: 200% 100%; animation: td-shimmer 1.3s ease-in-out infinite;}
@keyframes td-shimmer {from {background-position: 200% 0;} to {background-position: -200% 0;}}
[data-stale="true"] {opacity: 0.6 !important;}
@media (max-width: 640px) {
  [data-testid="stMainBlockContainer"], .block-container {padding-top: 4rem;}
  .td-brand {font-size: 1.3rem; gap: 0.55rem;}
  .td-logo {height: 2.6rem;}
  .td-name {font-size: 1.6rem;}
  .td-ticker {font-size: 0.95rem;}
  .td-price {font-size: 2.1rem;}
  .td-change {font-size: 1rem;}
  .td-about {text-align: left; hyphens: manual; -webkit-hyphens: manual;}
  .td-stats {gap: 0.75rem 1rem;}
}
@media (prefers-reduced-motion: reduce) {
  .td-skeleton span {animation: none;}
}
.td-value {font-size: 1.12rem; font-weight: 600; margin-top: 0.15rem; font-variant-numeric: tabular-nums; overflow-wrap: anywhere;}
</style>"""


@st.cache_data(show_spinner=False)
def logo_data_uri(path: str, modified: float) -> str:
    """The logo as a data URI; the file's modified time is part of the cache key, so a new logo shows at once."""
    with open(path, "rb") as fh:
        encoded = base64.b64encode(fh.read()).decode("ascii")
    return f"data:{LOGO_TYPES[os.path.splitext(path.lower())[1]]};base64,{encoded}"


def text(value) -> str:
    """Escape text for the HTML header (and keep $ from being read as math)."""
    return html.escape(str(value)).replace("$", "&#36;")


def fmt_day(ts) -> str:
    return f"{ts:%b} {ts.day}, {ts.year}"


def fmt_amount(x) -> str:
    v = num(x)
    if v is None:
        return "n/a"
    return f"{v:,.4f}" if 0 < abs(v) < 0.1 else f"{v:,.2f}"


def theme_css() -> str:
    """CSS variables for the custom HTML, so it matches light or dark mode."""
    p = palette()
    return (f"<style>:root {{--td-up: {p['up']}; --td-down: {p['down']}; --td-accent: {p['accent']}; "
            f"--td-accent-soft: {p['accent_soft']}; --td-ring: {p['ring']};}}</style>")


SKELETON = ('<div class="td-skeleton" aria-busy="true" aria-label="Loading">'
            '<span style="width: 42%; height: 2.3rem"></span><span style="width: 26%; height: 1rem"></span>'
            '<span style="width: 34%; height: 2.7rem"></span><span style="width: 100%; height: 3.6rem"></span></div>')


def flash(message: str, icon: str = ":material/check_circle:"):
    """Queue a confirmation to show as a toast on the next run (toasts called right before a rerun get lost)."""
    st.session_state.setdefault("flash", []).append((message, icon))


def show_flash():
    for message, icon in st.session_state.pop("flash", []):
        st.toast(message, icon=icon)


def range_bar_html(low, high, last) -> str:
    low, high, last = num(low), num(high), num(last)
    if low is None or high is None or last is None or high <= low:
        return ""
    pos = min(max((last - low) / (high - low), 0.0), 1.0) * 100
    return (f'<div class="td-range" title="Today’s price is {pos:.0f}% of the way from the 52-week low to the high">'
            f'<div class="td-label">52-week range</div>'
            f'<div class="td-range-track"><span class="td-range-fill" style="width: {pos:.1f}%"></span>'
            f'<span class="td-range-dot" style="left: {pos:.1f}%"></span></div>'
            f'<div class="td-range-ends"><span>{fmt_price(low)}</span><span>{fmt_price(high)}</span></div></div>')


def sparkline_html(daily: pd.DataFrame, days: int = 31) -> str:
    """A small line of the last month's closes, green if it rose and red if it fell."""
    closes = daily["Close"].dropna()
    recent = closes[closes.index >= closes.index.max() - pd.Timedelta(days=days)]
    if len(recent) < 5:
        return ""
    p = palette()
    change = recent.iloc[-1] / recent.iloc[0] - 1
    color = p["up"] if change >= 0 else p["down"]
    w, h, pad = 120, 34, 3
    lo, hi = float(recent.min()), float(recent.max())
    span = (hi - lo) or 1.0
    xs = [pad + i * (w - 2 * pad) / (len(recent) - 1) for i in range(len(recent))]
    ys = [h - pad - (float(v) - lo) / span * (h - 2 * pad) for v in recent]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    area = f"{xs[0]:.1f},{h - pad} {line} {xs[-1]:.1f},{h - pad}"
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
           f'<polygon points="{area}" fill="{color}" fill-opacity="0.13"/>'
           f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="1.8" stroke-linejoin="round" '
           f'stroke-linecap="round"/></svg>')
    uri = "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")
    label = f"{change:+.1%}".replace("-", "−")
    return (f'<span class="td-spark"><img src="{uri}" width="{w}" height="{h}" alt="Last month, {text(label)}">'
            f'<span class="td-spark-label">1 month {text(label)}</span></span>')


def fmt_range(first, last=None) -> str:
    if last is None or last == first:
        return fmt_day(first)
    if first.year == last.year:
        return f"{first:%b} {first.day} – {last:%b} {last.day}, {last.year}"
    return f"{fmt_day(first)} – {fmt_day(last)}"


def stat_grid(items) -> str:
    cells = "".join(f'<div><div class="td-label">{text(k)}</div><div class="td-value">{text(v)}</div></div>'
                    for k, v in items)
    return f'<div class="td-stats td-stats-plain">{cells}</div>'


def epoch_day(value):
    v = num(value)
    return fmt_day(pd.Timestamp(int(v), unit="s")) if v and 1e8 < v < 4.2e9 else None


def company_facts(info: dict, qtype: str) -> list:
    facts = []

    def add(label, value):
        if value not in (None, ""):
            facts.append((label, str(value)))

    if qtype in FUND_TYPES:
        add("Category", info.get("category"))
        add("Fund family", info.get("fundFamily"))
        add("Legal type", info.get("legalType"))
        add("Started", epoch_day(info.get("fundInceptionDate")))
    else:
        add("Sector", info.get("sector"))
        add("Industry", info.get("industry"))
        add("Headquarters", ", ".join(str(x) for x in (info.get("city"), info.get("state"), info.get("country")) if x))
        employees = num(info.get("fullTimeEmployees"))
        add("Employees", f"{int(employees):,}" if employees else None)
        add("Latest fiscal year ended", epoch_day(info.get("lastFiscalYearEnd")))
    return facts


def facts_html(facts, website=None) -> str:
    rows = [f'<div><div class="td-label">{text(k)}</div><div class="td-fact">{text(v)}</div></div>' for k, v in facts]
    url = str(website or "")
    if url.lower().startswith(("http://", "https://")):
        domain = re.sub(r"^https?://(www\.)?", "", url, flags=re.IGNORECASE).rstrip("/")
        rows.append(f'<div><div class="td-label">Website</div><div class="td-fact"><a href="{html.escape(url, quote=True)}" '
                    f'target="_blank" rel="noopener">{text(domain)}</a></div></div>')
    return f'<div class="td-facts">{"".join(rows)}</div>' if rows else ""


FIELD_LABELS = {
    "52WeekChange": "52-week change", "SandP52WeekChange": "S&P 500 52-week change", "trailingPE": "P/E, trailing",
    "forwardPE": "P/E, forward", "trailingEps": "EPS, trailing", "forwardEps": "EPS, forward", "pegRatio": "PEG ratio",
    "trailingPegRatio": "PEG ratio, trailing", "enterpriseToEbitda": "EV/EBITDA", "enterpriseToRevenue": "EV/revenue",
    "address1": "Address", "address2": "Address, line 2", "zip": "Postal code", "fullTimeEmployees": "Employees",
    "irWebsite": "Investor relations website", "heldPercentInsiders": "Held by insiders",
    "heldPercentInstitutions": "Held by institutions", "quoteType": "Security type", "exchange": "Exchange code",
    "fullExchangeName": "Exchange", "longName": "Name", "shortName": "Short name", "symbol": "Symbol",
    "recommendationKey": "Analyst recommendation", "recommendationMean": "Analyst recommendation score",
    "numberOfAnalystOpinions": "Number of analysts", "priceToBook": "Price to book", "bookValue": "Book value per share",
    "YF repair?": "Prices repaired by yfinance",
}
PERCENT_FIELDS = {"52WeekChange", "SandP52WeekChange", "heldPercentInsiders", "heldPercentInstitutions", "profitMargins",
                  "grossMargins", "operatingMargins", "ebitdaMargins", "revenueGrowth", "earningsGrowth",
                  "earningsQuarterlyGrowth", "returnOnAssets", "returnOnEquity", "payoutRatio",
                  "trailingAnnualDividendYield", "shortPercentOfFloat", "sharesPercentSharesOut"}
ACRONYMS = {"eps": "EPS", "ebitda": "EBITDA", "pe": "P/E", "peg": "PEG", "esg": "ESG", "ir": "IR", "ceo": "CEO",
            "nav": "NAV", "ytd": "YTD", "gmt": "GMT", "utc": "UTC", "id": "ID", "url": "URL", "etf": "ETF", "ttm": "TTM",
            "us": "U.S.", "yf": "YF", "fcf": "FCF", "roe": "ROE", "roa": "ROA", "ebit": "EBIT"}
DATE_FIELD = re.compile(r"date|epoch|time|fiscalyearend|mostrecentquarter", re.IGNORECASE)


def humanize_field(key) -> str:
    """Yahoo's camelCase field names as plain words, e.g. heldPercentInsiders -> Held percent insiders."""
    key = str(key)
    if key in FIELD_LABELS:
        return FIELD_LABELS[key]
    k = (key.replace("fiftyTwoWeek", "52Week").replace("twoHundredDay", "200Day").replace("fiftyDay", "50Day")
         .replace("SandP", "S&P ").replace("_", " "))
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Za-z])(?=[0-9])|(?<=[0-9])(?=[A-Za-z])|(?<=[A-Z])(?=[A-Z][a-z])",
                    " ", k)
    words = []
    for w in spaced.split():
        low = w.lower()
        words.append(ACRONYMS.get(low, w if (w.isupper() and len(w) > 1) or "&" in w else low))
    label = " ".join(words)
    for old, new in ((" epoch", ""), ("52 week", "52-week"), ("200 day", "200-day"), ("50 day", "50-day"),
                     ("ex dividend", "ex-dividend")):
        label = label.replace(old, new)
    return label[:1].upper() + label[1:]


def format_field(key, value) -> str:
    key = str(key)
    if value is None:
        return ""
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        v = num(value)
        if v is None:
            return ""
        if key in PERCENT_FIELDS:
            return f"{v * 100:.2f}%"
        if DATE_FIELD.search(key) and "employees" not in key.lower() and epoch_day(v):
            return epoch_day(v)
        if abs(v) >= 1e6:
            return fmt_money(v)
        if float(v).is_integer():
            return f"{int(v):,}"
        return f"{v:,.4f}".rstrip("0").rstrip(".") if abs(v) < 1000 else f"{v:,.2f}"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(x) for x in value)
    return str(value)


def fields_tables(fields: dict):
    """(display table, CSV table) for a dict of Yahoo fields."""
    rows = sorted(fields.items(), key=lambda kv: humanize_field(kv[0]).lower())
    display = pd.DataFrame([(humanize_field(k), format_field(k, v)) for k, v in rows], columns=["Field", "Value"])
    raw = pd.DataFrame([(humanize_field(k), k, v if isinstance(v, (str, int, float, bool)) or v is None else str(v))
                        for k, v in rows], columns=["Field", "Yahoo name", "Value"])
    return display, raw


HOLDER_SUMMARY = {"insidersPercentHeld": ("Held by insiders", "pct"),
                  "institutionsPercentHeld": ("Held by institutions", "pct"),
                  "institutionsFloatPercentHeld": ("Institutions' share of the shares that trade", "pct"),
                  "institutionsCount": ("Number of institutions", "count")}
HOLDER_COLUMNS = {"Date Reported": "Reported", "Holder": "Holder", "pctHeld": "Share of company (%)", "Shares": "Shares",
                  "Value": "Value", "pctChange": "Change in position (%)"}


def holder_summary(df) -> list:
    if not isinstance(df, pd.DataFrame) or df.empty or "Value" not in df.columns:
        return []
    items = []
    for key, (label, kind) in HOLDER_SUMMARY.items():
        if key in df.index and num(df.loc[key, "Value"]) is not None:
            v = num(df.loc[key, "Value"])
            items.append((label, f"{v:.2%}" if kind == "pct" else f"{int(v):,}"))
    return items


def tidy_holders(df: pd.DataFrame):
    out = df.copy()
    for col in ("pctHeld", "pctChange"):
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce") * 100
    if "Date Reported" in out:
        out["Date Reported"] = pd.to_datetime(out["Date Reported"], errors="coerce").dt.strftime("%Y-%m-%d")
    out = out.rename(columns=HOLDER_COLUMNS)
    config = {"Share of company (%)": st.column_config.NumberColumn(format="%.2f%%"),
              "Change in position (%)": st.column_config.NumberColumn(format="%+.2f%%"),
              "Shares": st.column_config.NumberColumn(format="localized"),
              "Value": st.column_config.NumberColumn(format="compact")}
    return out, {k: v for k, v in config.items() if k in out.columns}


ESTIMATE_PERIODS = {"0q": "This quarter", "+1q": "Next quarter", "0y": "This fiscal year", "+1y": "Next fiscal year",
                    "+5y": "Next 5 years, per year", "LTG": "Long term, per year", "-5y": "Past 5 years, per year"}
RATING_PERIODS = {"0m": "This month", "-1m": "Last month", "-2m": "2 months ago", "-3m": "3 months ago"}
ANALYST_COLUMNS = {"avg": "Average", "low": "Low", "high": "High", "yearAgoEps": "Year-ago EPS",
                   "yearAgoRevenue": "Year-ago revenue", "numberOfAnalysts": "Analysts", "growth": "Growth (%)",
                   "current": "Now", "7daysAgo": "7 days ago", "30daysAgo": "30 days ago", "60daysAgo": "60 days ago",
                   "90daysAgo": "90 days ago", "stockTrend": "This stock (%)", "indexTrend": "S&P 500 (%)",
                   "stock": "This stock (%)", "industry": "Industry (%)", "sector": "Sector (%)", "index": "S&P 500 (%)",
                   "strongBuy": "Strong buy", "buy": "Buy", "hold": "Hold", "sell": "Sell", "strongSell": "Strong sell",
                   "ToGrade": "New rating", "FromGrade": "Previous rating", "priceTargetAction": "Price target change",
                   "currentPriceTarget": "New price target", "priorPriceTarget": "Previous price target"}
RATING_ACTIONS = {"up": "Upgrade", "down": "Downgrade", "main": "Maintained", "init": "Initiated", "reit": "Reiterated"}


def tidy_analyst_table(kind: str, df: pd.DataFrame):
    out = df.copy()
    if kind == "growth_estimates":
        out = out.apply(pd.to_numeric, errors="coerce") * 100
    elif "growth" in out.columns:
        out["growth"] = pd.to_numeric(out["growth"], errors="coerce") * 100
    if kind == "recommendations" and "period" in out.columns:
        out = out.set_index("period")
        out.index = [RATING_PERIODS.get(str(p), str(p)) for p in out.index]
    elif kind == "upgrades_downgrades":
        out.index = pd.DatetimeIndex(pd.to_datetime(out.index, errors="coerce")).strftime("%Y-%m-%d")
        if "Action" in out:
            out["Action"] = out["Action"].map(lambda a: RATING_ACTIONS.get(str(a), str(a)))
    else:
        out.index = [ESTIMATE_PERIODS.get(str(p), str(p)) for p in out.index]
    out.index.name = "Date" if kind == "upgrades_downgrades" else "Period"
    out = out.rename(columns=ANALYST_COLUMNS)
    config = {}
    for col in out.columns:
        if not pd.api.types.is_numeric_dtype(out[col]):
            continue
        if str(col).endswith("(%)"):
            config[col] = st.column_config.NumberColumn(format="%+.1f%%")
        elif col == "Analysts" or kind == "recommendations":
            config[col] = st.column_config.NumberColumn(format="%d")
        else:
            config[col] = st.column_config.NumberColumn(format="compact" if kind == "revenue_estimate" else "%.2f")
    return out, config


def tidy_fund_table(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Fund statistics as readable text: ratios as percentages and big amounts shortened."""
    out = pd.DataFrame(index=[str(i) for i in df.index])
    for col in df.columns:
        values = []
        for row in df.index:
            v, name = num(df.loc[row, col]), str(row).lower()
            if v is None:
                values.append("")
            elif any(w in name for w in ("ratio", "turnover", "growth", "yield")):
                values.append(f"{v:.2%}")
            elif any(w in name for w in ("assets", "market cap")):
                values.append(fmt_money(v))
            else:
                values.append(f"{v:,.2f}")
        out[display_symbol(symbol) if str(col) == symbol else str(col).capitalize()] = values
    out.index.name = "Measure"
    return out


# ---------------------------------------------------------------------------
# UI building blocks
# ---------------------------------------------------------------------------
def show_table(df, *, key: str, filename: str, column_config=None, height=None, hide_index: bool = False,
               csv_df=None, download_label: str = "Download CSV", empty_text: str = "Not available for this ticker."):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        st.caption(empty_text)
        return
    kwargs = {"hide_index": hide_index, "key": f"table_{key}"}
    if column_config:
        kwargs["column_config"] = column_config
    if height:
        kwargs["height"] = int(height)
    st.dataframe(df, **kwargs)
    out = df if csv_df is None else csv_df
    st.download_button(download_label, data=to_csv_bytes(out, index=not hide_index), file_name=filename,
                       mime="text/csv", key=f"download_{key}")


def table_height(rows: int, cap: int = 720) -> int:
    return min(36 * (rows + 1) + 4, cap)


def accounting_config(columns, scale: float) -> dict:
    step = 1.0 if scale == 1 else 0.01
    return {c: st.column_config.NumberColumn(format="accounting", step=step) for c in columns}


def percent_config(columns) -> dict:
    return {c: st.column_config.NumberColumn(format="%.2f%%") for c in columns}


def price_column_config(table: pd.DataFrame, reference) -> dict:
    fmt = f"%.{price_decimals(reference)}f"
    config = {c: st.column_config.NumberColumn(format=fmt) for c in ("Open", "High", "Low", "Close", "Adj Close") if c in table}
    if "Volume" in table:
        config["Volume"] = st.column_config.NumberColumn(format="localized")
    for c in ("Dividends", "Capital Gains"):
        if c in table:
            config[c] = st.column_config.NumberColumn(format="%.4f")
    return config


def choice_bar(label: str, options: list, default, key: str, *, label_visibility: str = "visible", help=None):
    """A segmented control that always has one option selected."""
    last_key = f"{key}__last"
    fallback = st.session_state.get(last_key, default)
    if fallback not in options:
        fallback = default

    def keep_selection():
        if st.session_state.get(key) is None:  # clicking the active option would otherwise clear it
            st.session_state[key] = st.session_state.get(last_key, default)

    if st.session_state.get(key) not in (None, *options):
        st.session_state.pop(key, None)
    value = st.segmented_control(label, options, default=None if key in st.session_state else fallback, key=key,
                                 on_change=keep_selection, label_visibility=label_visibility, help=help)
    if value is None:
        value = fallback
    st.session_state[last_key] = value
    return value


def units_picker() -> str:
    options = list(UNITS)
    previous = st.session_state.get("units__saved")
    value = st.selectbox("Figures in", options, index=options.index(current_units()), key="units",
                         help="Scales money amounts in tables and downloads. Per-share values and tax rates stay as they are.")
    st.session_state["units__saved"] = value
    if previous is not None and value != previous:
        st.rerun()  # tabs refresh on their own, but the ZIP button above them needs the new units too
    return value


def current_units() -> str:
    """The units choice, read at the start of a run so everything drawn above the picker already uses it."""
    value = st.session_state.get("units") or st.session_state.get("units__saved", "Millions")
    return value if value in UNITS else "Millions"


def current_benchmark() -> str:
    value = st.session_state.get("benchmark") or st.session_state.get("benchmark__saved", "^GSPC")
    return str(value).strip().upper()


def benchmark_picker() -> str:
    previous = st.session_state.get("benchmark__saved")
    saved = current_benchmark()
    options = list(BENCHMARKS) + ([saved] if saved not in BENCHMARKS else [])
    value = st.selectbox("Compare with", options, index=options.index(saved), key="benchmark", accept_new_options=True,
                         format_func=lambda s: BENCHMARKS[s] if s in BENCHMARKS else display_symbol(s),
                         help="Used for relative returns and for beta, including the beta in the DCF. "
                              "Type any Yahoo symbol to compare with something else.")
    value = str(value or "^GSPC").strip().upper()
    st.session_state["benchmark__saved"] = value
    if previous is not None and value != previous:
        st.rerun()  # the ZIP button outside this tab uses the benchmark too
    return value


def _choose(symbol: str):
    st.session_state["ticker"] = symbol


def pick_symbol(query: str) -> str:
    """Turn text that isn't in the suggestion list into a Yahoo symbol, asking when it's ambiguous."""
    q = query.strip()
    typed = q.upper()
    matches, err = attempt(search_symbols, q)
    matches = matches or []
    symbols = list(dict.fromkeys(m["symbol"] for m in matches))
    if typed in symbols:
        return typed
    if not symbols:
        if TICKER_PATTERN.match(q):
            if err:
                st.caption("Yahoo's search isn't responding, so what you typed is used as the ticker.")
            return typed
        st.warning(f'Nothing on Yahoo Finance matches "{q}". Check the spelling, or type the exact ticker.')
        return ""
    labels = {}
    for m in matches:
        if m["symbol"] not in labels:
            kind = TYPE_LABELS.get(m["type"].upper(), m["type"].title())
            where = ", ".join(x for x in (kind, m["exchange"]) if x)
            labels[m["symbol"]] = f"{m['name'] or m['symbol']} ({m['symbol']})" + (f", {where}" if where else "")
    options = symbols[:]
    if TICKER_PATTERN.match(q):
        labels[typed] = f"{typed}, exactly as typed"
        options.append(typed)
    return st.selectbox(f'Matches on Yahoo Finance for "{q}"', options, format_func=lambda s: labels.get(s, s),
                        key=f"match_{typed}")


def price_snapshot(info: dict, meta: dict, daily: pd.DataFrame) -> tuple[float | None, float | None]:
    """Last price and the close before it, taken from the same price history so they always match."""
    closes = daily["Close"].dropna()
    if len(closes) >= 2:
        return float(closes.iloc[-1]), float(closes.iloc[-2])
    last = (num(meta.get("regularMarketPrice")) or num(info.get("regularMarketPrice"))
            or (num(closes.iloc[-1]) if len(closes) else None))
    return last, num(info.get("regularMarketPreviousClose")) or num(info.get("previousClose"))


def quote_stats(qtype: str, info: dict, daily: pd.DataFrame, last: float | None) -> list[tuple[str, str]]:
    adj = adjusted_close(daily)
    paid = ttm_dividends(daily)
    dividend_yield = fmt_pct(paid / last) if (paid and last) else "No dividend"
    if qtype in FUND_TYPES:
        return [("Total assets", fmt_money(info.get("totalAssets"))), ("Dividend yield, 12 months", dividend_yield),
                ("1-year total return", fmt_pct(period_return(adj, pd.DateOffset(years=1))[0])),
                ("Category", info.get("category") or "n/a"), ("Fund family", info.get("fundFamily") or "n/a")]
    if qtype in MARKET_ONLY_TYPES:
        stats = [("1-year change", fmt_pct(period_return(adj, pd.DateOffset(years=1))[0])),
                 ("Change this year", fmt_pct(period_return(adj, "ytd")[0]))]
        volume = num(daily["Volume"].iloc[-1]) if "Volume" in daily else None
        if volume:
            stats.append(("Volume, last day", fmt_money(volume, 1)))
        return stats
    pe, beta = num(info.get("trailingPE")), num(info.get("beta"))
    return [("Market cap", fmt_money(info.get("marketCap"))), ("P/E, trailing", f"{pe:.1f}" if pe else "n/a"),
            ("Beta, per Yahoo", f"{beta:.2f}" if beta is not None else "n/a"),
            ("Dividend yield, 12 months", dividend_yield), ("Sector", info.get("sector") or "n/a")]


def render_quote(symbol: str, info: dict, meta: dict, daily: pd.DataFrame, qtype: str, price_ccy: str):
    name = info.get("longName") or info.get("shortName") or meta.get("longName") or meta.get("shortName") or symbol
    exchange = info.get("fullExchangeName") or meta.get("fullExchangeName") or info.get("exchange") or meta.get("exchangeName")
    sub = TYPE_LABELS.get(qtype, qtype.title() if qtype else "Security")
    if exchange and qtype in ("EQUITY", "ETF"):
        sub += f" listed on {exchange}"
    if price_ccy:
        sub += f", {'values' if qtype == 'INDEX' else 'prices'} in {price_ccy}"
    chip = "" if str(symbol).startswith("^") else f'<span class="td-ticker">{text(display_symbol(symbol))}</span>'
    last, prev = price_snapshot(info, meta, daily)
    change = ""
    if last is not None and prev:
        diff = last - prev
        css = "td-up" if diff > 0 else "td-down" if diff < 0 else "td-flat"
        sign = "+" if diff > 0 else "−" if diff < 0 else ""
        change = (f'<span class="td-change {css}">{sign}{abs(diff):,.{price_decimals([last])}f} '
                  f'({sign}{abs(diff / prev):.2%})</span>')
    year = daily[daily.index >= daily.index.max() - pd.Timedelta(days=365)]
    stats = range_bar_html(year["Low"].min(), year["High"].max(), last) + "".join(
                    f'<div><div class="td-label">{text(label)}</div>'
                    f'<div class="td-value">{text(re.sub(r"^-(?=[0-9])", "−", value))}</div></div>'
                    for label, value in quote_stats(qtype, info, daily, last))
    st.markdown(
        f'<div class="td-name">{text(name)}{chip}</div>'
        f'<div class="td-sub">{text(sub)}</div>'
        f'<div class="td-quote"><span class="td-price">{fmt_price(last)}</span>'
        f'<span class="td-ccy">{text(price_ccy)}</span>{change}{sparkline_html(daily)}'
        f'<span class="td-asof">as of {fmt_day(daily.index.max())}</span></div>'
        f'<div class="td-stats">{stats}</div>',
        unsafe_allow_html=True)


def sections_for(qtype: str) -> list[str]:
    if qtype == "EQUITY":
        return SECTION_SETS["EQUITY"]
    if qtype in FUND_TYPES:
        return SECTION_SETS["FUND"]
    if qtype in MARKET_ONLY_TYPES:
        return SECTION_SETS["MARKET"]
    return SECTION_SETS["ANY"]


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

@st.fragment
def section_performance(symbol: str, daily: pd.DataFrame, rf: float):
    adj = adjusted_close(daily)
    if len(adj) < 2:
        st.info("There isn't enough price history to measure performance.")
        return
    c1, _ = st.columns([2, 3])
    with c1:
        benchmark = benchmark_picker()
    bench = None
    if benchmark and benchmark != symbol.upper():
        loaded, err = attempt(load_history, benchmark, "max")
        if loaded is not None:
            bench = adjusted_close(loaded[0])
        else:
            st.warning(f"{display_symbol(benchmark)} didn't load, so comparisons are left out. {err}")
    fname = safe_filename(symbol)

    st.subheader("Trailing returns")
    st.caption("Based on adjusted closes, so dividends count as reinvested. Periods over a year are also annualized.")
    returns = returns_table(adj, bench, display_symbol(symbol), display_symbol(benchmark))
    show_table(returns, key="returns", hide_index=True, filename=f"{fname}_trailing_returns.csv",
               column_config=percent_config([c for c in returns.columns if c != "Period"]))

    st.subheader("Risk and return")
    st.caption(f"The Sharpe ratio uses a risk-free rate of {rf:.2%}. Volatility is annualized from daily returns.")
    risk = risk_table(adj, rf)
    config = percent_config([c for c in RISK_COLUMNS if c != "Sharpe ratio"])
    config["Sharpe ratio"] = st.column_config.NumberColumn(format="%.2f")
    show_table(risk, key="risk", hide_index=True, filename=f"{fname}_risk_return.csv", column_config=config)
    if bench is not None:
        beta, corr, months = monthly_beta(adj, bench)
        m1, m2, _ = st.columns([1, 1, 2])
        m1.metric(f"Beta vs {display_symbol(benchmark)}", f"{beta:.2f}" if np.isfinite(beta) else "n/a",
                  help=f"Slope of monthly returns against the {display_symbol(benchmark)} over the last {months} months "
                       "(Yahoo's method).")
        m2.metric("Correlation", f"{corr:.2f}" if np.isfinite(corr) else "n/a", help="Over the same months.")

    st.subheader("Growth of 10,000 and drawdowns")
    window = choice_bar("Chart period", ["1Y", "3Y", "5Y", "10Y", "Max"], "5Y", "perf_window")
    view = slice_window(adj, "All" if window == "Max" else window)
    show_chart(growth_figure(view, bench, display_symbol(symbol), display_symbol(benchmark)), "growth", symbol)
    show_chart(drawdown_figure(view), "drawdown", symbol)

    me, them = display_symbol(symbol), display_symbol(benchmark)
    data = pd.DataFrame({f"{me} adj close": adj, f"{me} daily return (%)": (adj / adj.shift(1) - 1) * 100})
    if bench is not None:
        b = bench.reindex(adj.index).ffill()
        data[f"{them} close"] = b
        data[f"{them} daily return (%)"] = (b / b.shift(1) - 1) * 100
    st.download_button("Download daily returns for regressions (CSV)", data=to_csv_bytes(data),
                       file_name=f"{fname}_daily_returns.csv", mime="text/csv", key="download_daily_returns")


@st.fragment
def section_statements(symbol: str, info: dict):
    c1, c2, _ = st.columns([1.6, 1.3, 3], vertical_alignment="bottom")
    with c1:
        freq_label = choice_bar("Period", ["Annual", "Quarterly"], "Annual", "statement_period")
    with c2:
        units_label = units_picker()
    scale = UNITS[units_label]
    freq = "yearly" if freq_label == "Annual" else "quarterly"
    fin_ccy = info.get("financialCurrency") or info.get("currency") or "the reporting currency"
    unit_text = "" if units_label == "Raw" else f" in {units_label.lower()}"
    st.caption(f"Figures{unit_text} of {fin_ccy}. Negative numbers are in parentheses. Per-share values and tax rates "
               "aren't scaled. Yahoo Finance usually covers the last 4 to 5 fiscal years and 5 to 6 quarters.")
    tabs = st.tabs([title for _, title in STATEMENTS])
    for tab, (kind, title) in zip(tabs, STATEMENTS):
        with tab:
            raw, err = attempt(load_statement, symbol, kind, freq)
            if err:
                st.warning(err)
                continue
            table = tidy_statement(raw, scale)
            show_table(table, key=f"statement_{kind}_{freq}", column_config=accounting_config(table.columns, scale),
                       height=table_height(len(table)), download_label=f"Download {title.lower()} (CSV)",
                       filename=f"{safe_filename(symbol)}_{kind}_{freq_label.lower()}_{units_label.lower()}.csv",
                       empty_text=f"Yahoo Finance has no {title.lower()} for {symbol}.")



@st.fragment
def section_dividends(symbol: str, daily: pd.DataFrame, price: float | None):
    divs, splits, gains = nonzero(daily, "Dividends"), nonzero(daily, "Stock Splits"), nonzero(daily, "Capital Gains")
    fname = safe_filename(symbol)
    if divs.empty and splits.empty and gains.empty:
        st.info(f"Yahoo Finance shows no dividends, capital gains distributions or splits for {symbol}.")
        return
    if not divs.empty:
        st.subheader("Dividends")
        paid = ttm_dividends(daily)
        annual = divs.groupby(divs.index.year).agg(["sum", "count"])
        annual.columns = ["Dividends per share", "Payments"]
        annual.index.name = "Year"
        this_year = daily.index.max().year
        complete = annual[annual.index < this_year]
        growth5 = None
        if len(complete):
            y1 = complete.index[-1]
            if (y1 - 5) in complete.index and complete.loc[y1 - 5, "Dividends per share"] > 0:
                growth5 = (complete.loc[y1, "Dividends per share"] / complete.loc[y1 - 5, "Dividends per share"]) ** 0.2 - 1
        m = st.columns(4)
        m[0].metric("Paid in the last 12 months", fmt_amount(paid))
        m[1].metric("Yield on last price", fmt_pct(paid / price) if price else "n/a")
        m[2].metric("Growth per year, last 5 years", fmt_pct(growth5) if growth5 is not None else "n/a",
                    help="Compound growth in total dividends between the last full year and five years before it.")
        m[3].metric("Payments in the last full year", f"{int(complete['Payments'].iloc[-1])}" if len(complete) else "n/a")
        p = palette()
        colors = [p["up"] if y < this_year else p["muted"] for y in annual.index]
        show_chart(bar_figure(annual.index.astype(str), annual["Dividends per share"], "Dividends per share",
                                   color=colors, horizontal=False), "dividends", symbol)
        st.caption(f"Grey marks {this_year}, which isn't over yet. Amounts are per share, adjusted for later stock splits.")
        c1, c2 = st.columns(2)
        with c1:
            history = divs.sort_index(ascending=False).to_frame("Dividend per share")
            history.index = history.index.strftime("%Y-%m-%d")
            show_table(history, key="dividend_history", filename=f"{fname}_dividends.csv",
                       column_config={"Dividend per share": st.column_config.NumberColumn(format="%.4f")},
                       download_label="Download dividend history (CSV)")
        with c2:
            show_table(annual.sort_index(ascending=False), key="dividend_annual", filename=f"{fname}_dividends_by_year.csv",
                       column_config={"Dividends per share": st.column_config.NumberColumn(format="%.4f")},
                       download_label="Download yearly totals (CSV)")
    if not gains.empty:
        st.subheader("Capital gains distributions")
        table = gains.sort_index(ascending=False).to_frame("Capital gain per share")
        table.index = table.index.strftime("%Y-%m-%d")
        show_table(table, key="capital_gains", filename=f"{fname}_capital_gains.csv")
    if not splits.empty:
        st.subheader("Stock splits")
        table = splits.sort_index(ascending=False).to_frame("Ratio")
        table["Split"] = [f"{v:g}-for-1" if v >= 1 else f"1-for-{1 / v:g}" for v in table["Ratio"]]
        table.index = table.index.strftime("%Y-%m-%d")
        show_table(table, key="splits", filename=f"{fname}_splits.csv")


@st.fragment
def section_fund(symbol: str):
    fd, err = attempt(load_fund_data, symbol)
    if err:
        st.warning(err)
        return
    fname = safe_filename(symbol)
    if fd.get("description"):
        st.markdown(md(fd["description"]))
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Fund facts")
        overview = dict_table(fd.get("fund_overview"))
        if not overview.empty:
            overview["Field"] = overview["Field"].map(pretty_key)
        show_table(as_text_column(overview, "Value") if not overview.empty else overview, key="fund_overview",
                   hide_index=True, csv_df=overview, filename=f"{fname}_fund_overview.csv")
        ops = fd.get("fund_operations")
        if isinstance(ops, pd.DataFrame) and not ops.empty:
            st.caption("Fund compared with its category average")
            show_table(tidy_fund_table(ops, symbol), key="fund_operations", csv_df=ops,
                       filename=f"{fname}_fund_operations.csv")
    with c2:
        st.subheader("Asset mix")
        assets = weights_table(fd.get("asset_classes"), "Asset class")
        assets = assets[assets["Weight (%)"].abs() > 0] if not assets.empty else assets
        show_table(assets, key="fund_assets", hide_index=True, filename=f"{fname}_asset_classes.csv",
                   column_config=percent_config(["Weight (%)"]))

    st.subheader("Top holdings")
    top = fd.get("top_holdings")
    if isinstance(top, pd.DataFrame) and not top.empty:
        top = top.copy()
        weight_col = next((c for c in top.columns if "percent" in str(c).lower()), None)
        config = None
        if weight_col:
            top["Weight (%)"] = pd.to_numeric(top.pop(weight_col), errors="coerce") * 100
            labels = top["Name"] if "Name" in top else top.index.astype(str)
            show_chart(bar_figure(labels, top["Weight (%)"], "Weight (%)"), "holdings", symbol)
            config = percent_config(["Weight (%)"])
        show_table(top, key="fund_top_holdings", filename=f"{fname}_top_holdings.csv", column_config=config)
        st.caption("Yahoo Finance lists only the largest holdings. The fund provider's website has the full list.")
    else:
        st.caption("Not available for this fund.")

    sectors = weights_table(fd.get("sector_weightings"), "Sector")
    if not sectors.empty:
        st.subheader("Sector weights")
        show_chart(bar_figure(sectors["Sector"], sectors["Weight (%)"], "Weight (%)"), "sectors", symbol)
        show_table(sectors, key="fund_sectors", hide_index=True, filename=f"{fname}_sector_weights.csv",
                   column_config=percent_config(["Weight (%)"]))
    for field, title in (("equity_holdings", "Valuation of the stocks held"), ("bond_holdings", "Bond holdings")):
        table = fd.get(field)
        if isinstance(table, pd.DataFrame) and not table.empty:
            st.subheader(title)
            show_table(tidy_fund_table(table, symbol), key=f"fund_{field}", csv_df=table,
                       filename=f"{fname}_{field}.csv")
    ratings = weights_table(fd.get("bond_ratings"), "Rating")
    if not ratings.empty and ratings["Weight (%)"].abs().sum() > 0:
        st.subheader("Bond credit ratings")
        show_table(ratings, key="fund_ratings", hide_index=True, filename=f"{fname}_bond_ratings.csv",
                   column_config=percent_config(["Weight (%)"]))




def multi_pills(label: str, options: list, default: list, key: str, *, help=None) -> list:
    """Multi-select chips that remember their choice when you switch tabs."""
    saved_key = f"{key}__saved"
    initial = [o for o in st.session_state.get(saved_key, default) if o in options]
    value = st.pills(label, options, selection_mode="multi", default=None if key in st.session_state else initial,
                     key=key, help=help)
    value = [o for o in (value or []) if o in options]
    st.session_state[saved_key] = value
    return value


INDICATOR_HELP = ("Moving averages: 50 and 200 bars. Bollinger bands: 20 bars, 2 standard deviations. "
                  "RSI (14 bars): above 70 is often read as overbought and below 30 as oversold. "
                  "MACD (12, 26, 9): the MACD line crossing above its signal line is often read as a bullish turn.")


@st.fragment
def section_price(symbol: str, daily: pd.DataFrame, currency: str):
    period = choice_bar("Period", PRICE_PERIODS, "1Y", "price_period")
    first_day, last_day = daily.index.min().date(), daily.index.max().date()
    start = end = None
    if period == "Custom":
        c1, c2, _ = st.columns([1, 1, 2])
        start = c1.date_input("From", value=max(first_day, last_day - timedelta(days=5 * 365)), min_value=first_day,
                              max_value=last_day, key="price_from")
        end = c2.date_input("To", value=last_day, min_value=first_day, max_value=last_day, key="price_to")
        if start > end:
            st.error("The start date has to be on or before the end date.")
            return
    c1, c2, c3 = st.columns([2.4, 1.5, 1.2], vertical_alignment="bottom")
    with c1:
        bar = choice_bar("Bar size", list(INTERVALS), "Daily", "price_bar")
    with c2:
        style = choice_bar("Chart", ["Candles", "Line"], "Candles", "price_style")
    log_scale = c3.toggle("Log scale", value=False, key="price_log")
    chosen = multi_pills("Indicators", INDICATOR_OPTIONS, ["Moving averages"], "price_indicators", help=INDICATOR_HELP)
    bar_word = BAR_WORD[bar]

    if bar == "Daily":
        hist = daily
    else:
        loaded, err = attempt(load_history, symbol, "max", None, None, INTERVALS[bar])
        if err:
            st.warning(err)
            return
        hist = loaded[0]
    if period == "Custom":
        view = hist[(hist.index >= pd.Timestamp(start)) & (hist.index < pd.Timestamp(end) + pd.Timedelta(days=1))]
    else:
        begin = window_start(hist.index, "All" if period == "Max" else period)
        view = hist if begin is None else hist[hist.index >= begin]
    if view.empty:
        st.info("There's no price data in that period. Choose a longer period or a bigger bar size.")
        return

    indicators = indicator_frame(hist, chosen)
    show_chart(price_figure(view, indicators, style, log_scale, currency, bar_word), "price", symbol)

    st.subheader(f"{bar} prices, {fmt_day(view.index.min())} to {fmt_day(view.index.max())}")
    note = " Indicator columns you switched on are included." if chosen else ""
    st.caption(f"{len(view):,} rows, newest first. Close is the traded price, adjusted only for stock splits. "
               f"Adj Close also folds in dividends, so use it to measure returns. Downloads run oldest to newest.{note}")
    full = hist.join(indicators) if len(indicators.columns) else hist
    shown = full.loc[view.index]
    table = shown.sort_index(ascending=False)
    table.index = table.index.strftime("%Y-%m-%d")
    config = price_column_config(table, hist["Close"])
    decimals = price_decimals(hist["Close"])
    for col in indicators.columns:
        fmt = "%.1f" if col == "RSI 14" else (f"%.{decimals + 2}f" if col.startswith("MACD") else f"%.{decimals}f")
        config[col] = st.column_config.NumberColumn(format=fmt)
    st.dataframe(table, column_config=config, height=420, key="table_prices")
    name = safe_filename(symbol)
    d1, d2, _ = st.columns([1.3, 1.3, 2])
    d1.download_button(f"Download these {len(view):,} rows (CSV)", data=to_csv_bytes(shown), mime="text/csv",
                       file_name=f"{name}_{INTERVALS[bar]}_{view.index.min():%Y%m%d}_{view.index.max():%Y%m%d}.csv",
                       key="download_prices_view")
    d2.download_button("Download full history (CSV)", data=to_csv_bytes(full), mime="text/csv",
                       file_name=f"{name}_{INTERVALS[bar]}_full_history.csv", key="download_prices_all")


@st.fragment
def section_ratios(symbol: str, daily: pd.DataFrame):
    statements, errors = {}, set()
    for kind in ("income", "balance", "cashflow"):
        statements[kind], err = attempt(load_statement, symbol, kind, "yearly")
        if err:
            errors.add(err)
    for err in errors:
        st.warning(err)
    table = financial_ratios(statements["income"], statements["balance"], statements["cashflow"], daily)
    if table.empty:
        st.info(f"Yahoo Finance doesn't have enough financial statement data to calculate ratios for {symbol}.")
        return
    st.caption("Calculated from Yahoo Finance's annual statements, which usually cover the last 4 to 5 fiscal years. "
               "Returns use the average of opening and closing balances. Valuation ratios use the share price on "
               "each fiscal year-end date. Blank cells mean an input wasn't reported or the ratio isn't meaningful "
               "(for example a P/E when earnings are negative).")
    options = list(table.index)
    default = options.index("Operating margin (%)") if "Operating margin (%)" in options else 0
    c1, _ = st.columns([2, 3])
    pick = c1.selectbox("Chart a ratio", options, index=default, key="ratio_pick")
    show_chart(bar_figure(list(table.columns), table.loc[pick], pick, horizontal=False), "ratio", symbol)
    for group, names in RATIO_GROUPS:
        rows = [n for n in names if n in table.index]
        if not rows:
            continue
        st.subheader(group)
        sub = table.loc[rows]
        st.dataframe(sub, column_config={c: st.column_config.NumberColumn(format="%.2f") for c in sub.columns},
                     key=f"table_ratios_{safe_filename(group)}")
    export = table.copy()
    export.insert(0, "Group", [next((g for g, names in RATIO_GROUPS if r in names), "") for r in export.index])
    st.download_button("Download all ratios (CSV)", data=to_csv_bytes(export), mime="text/csv",
                       file_name=f"{safe_filename(symbol)}_financial_ratios.csv", key="download_ratios")


def cost_of_equity_inputs(symbol: str, info: dict, daily: pd.DataFrame, benchmark: str) -> float:
    """Inputs for the CAPM cost of equity, used where there's no DCF above to take it from."""
    rf, rf_note = risk_free_rate()
    computed, months = None, 0
    if benchmark and benchmark != symbol.upper():
        loaded, _ = attempt(load_history, benchmark, "max")
        if loaded is not None:
            computed, _, months = monthly_beta(adjusted_close(daily), adjusted_close(loaded[0]))
    beta_default, beta_note = choose_beta(computed, months, display_symbol(benchmark), num(info.get("beta")))
    c = st.columns(4)
    rf_in = c[0].number_input("Risk-free rate (%)", value=float(round(rf * 100, 2)), step=0.05, format="%.2f",
                              key=f"coe_rf_{symbol}", help=rf_note)
    beta_in = c[1].number_input("Beta", value=float(round(beta_default, 2)), step=0.05, format="%.2f",
                                key=f"coe_beta_{symbol}", help=beta_note)
    erp_in = c[2].number_input("Equity risk premium (%)", value=float(DEFAULT_ERP * 100), step=0.25, format="%.2f",
                               key=f"coe_erp_{symbol}", help="An assumption. Damodaran publishes a monthly implied "
                                                              "premium for the S&P 500 at pages.stern.nyu.edu/~adamodar.")
    ke = (rf_in + beta_in * erp_in) / 100
    c[3].metric("Cost of equity", fmt_pct(ke), help="Risk-free rate + beta x equity risk premium (CAPM).")
    return ke


def ddm_block(symbol: str, daily: pd.DataFrame, price: float | None, price_ccy: str, ke: float, heading: str,
              ke_note: str) -> dict | None:
    st.subheader(heading)
    paid = ttm_dividends(daily)
    if paid <= 0:
        st.info(f"{symbol} hasn't paid dividends in the last 12 months, so a dividend discount model doesn't apply.")
        return None
    st.caption("Values a share as today's value of the dividends it's expected to pay, discounted at the cost of "
               "equity. Companies that also buy back shares return more cash than their dividends show, so this "
               "model tends to undervalue them. It suits banks, insurers, utilities and funds best.")
    g5 = dividend_growth_5y(daily)
    g_default = min(max(g5, 0.0), 0.10) if g5 is not None else 0.03
    tg_default = min(DEFAULT_TERMINAL_GROWTH, ke - 0.01)
    c = st.columns(4)
    d0 = c[0].number_input(f"Dividends per share, last 12 months ({price_ccy})", value=float(round(paid, 4)),
                           format="%.4f", key=f"ddm_d0_{symbol}", help="From Yahoo Finance dividend history.")
    g_in = c[1].number_input("Dividend growth in year 1 (%)", value=float(round(g_default * 100, 2)), step=0.5,
                             format="%.2f", key=f"ddm_g_{symbol}",
                             help="Pre-filled with the last 5 years' dividend growth, kept between 0% and 10%.")
    years = c[2].number_input("Forecast years", min_value=1, max_value=15, value=5, step=1, key=f"ddm_years_{symbol}")
    tg_in = c[3].number_input("Terminal growth (%)", value=float(round(tg_default * 100, 2)), step=0.25, format="%.2f",
                              key=f"ddm_tg_{symbol}")
    fade = st.toggle("Fade dividend growth to the terminal rate", value=True, key=f"ddm_fade_{symbol}")
    if ke <= tg_in / 100:
        st.error(f"The cost of equity ({ke:.2%}) has to be higher than terminal growth ({tg_in / 100:.2%}).")
        return None
    result = run_ddm(d0, g_in / 100, int(years), tg_in / 100, ke, fade)
    value = result["value"]
    upside = value / price - 1 if price else None
    m = st.columns(4)
    m[0].metric(f"Dividend model value per share ({price_ccy})", fmt_price(value),
                delta=f"{upside:+.1%} vs last price" if upside is not None else None)
    m[1].metric(f"Last price ({price_ccy})", fmt_price(price))
    m[2].metric("Cost of equity used", fmt_pct(ke), help=ke_note)
    m[3].metric("Share from the terminal value", fmt_pct(result["pv_terminal"] / value) if value else "n/a")
    projection = result["projection"].rename(columns={"Dividend per share": f"Dividend per share ({price_ccy})",
                                                      "Present value": f"Present value ({price_ccy})"})
    config = {"Year": st.column_config.NumberColumn(format="%d"),
              "Growth (%)": st.column_config.NumberColumn(format="%.2f%%"),
              "Discount factor": st.column_config.NumberColumn(format="%.4f"),
              f"Dividend per share ({price_ccy})": st.column_config.NumberColumn(format="%.4f"),
              f"Present value ({price_ccy})": st.column_config.NumberColumn(format="%.4f")}
    show_table(projection, key="ddm_projection", hide_index=True, column_config=config,
               filename=f"{safe_filename(symbol)}_dividend_model.csv", download_label="Download dividend model (CSV)")
    return {"d0": d0, "growth": g_in / 100, "terminal_growth": tg_in / 100, "years": int(years), "fade": fade,
            "value": value}


@st.fragment
def section_valuation(symbol: str, info: dict, daily: pd.DataFrame, price: float | None, price_ccy: str):
    c1, _ = st.columns([1.3, 4])
    with c1:
        units_label = units_picker()
    benchmark = current_benchmark()
    scale = UNITS[units_label]
    unit_word = "" if units_label == "Raw" else units_label.lower()
    fin_ccy = info.get("financialCurrency") or price_ccy or ""
    fname = safe_filename(symbol)
    step = iter(range(1, 20))

    statements, errors = {}, set()
    for kind in ("income", "cashflow", "balance"):
        statements[kind], err = attempt(load_statement, symbol, kind, "yearly")
        if err:
            errors.add(err)
    for err in errors:
        st.warning(err)
    table = dcf_history(statements["income"], statements["cashflow"], statements["balance"])
    if table.empty:
        st.info(f"Yahoo Finance doesn't have enough financial statement data to build a DCF for {symbol}.")
        if ttm_dividends(daily) > 0:
            st.subheader(f"{next(step)}. Cost of equity")
            ke = cost_of_equity_inputs(symbol, info, daily, benchmark)
            ddm_block(symbol, daily, price, price_ccy, ke, f"{next(step)}. Dividend discount model",
                      "From the inputs above.")
        return

    sector, industry = str(info.get("sector") or "").lower(), str(info.get("industry") or "").lower()
    if sector == "financial services" and any(w in industry for w in ("bank", "insurance", "credit", "mortgage", "capital markets")):
        st.info("Banks, insurers and other lenders don't suit a free-cash-flow DCF, because debt is part of how they "
                "operate. The dividend discount model further down fits them better.")

    major = SUBUNITS.get(price_ccy, (price_ccy, 1.0))[0]
    factor = fx_factor(fin_ccy, price_ccy)
    if fin_ccy and price_ccy and fin_ccy != major:
        if factor:
            st.info(f"{symbol} reports in {fin_ccy} but trades in {price_ccy}. Values per share are converted at the "
                    f"latest exchange rate. If this is an ADR, one ADR can stand for several ordinary shares, so check "
                    f"the ratio before comparing with the price.")
        else:
            st.warning(f"{symbol} reports in {fin_ccy} but trades in {price_ccy}, and the exchange rate didn't load. "
                       f"Values per share stay in {fin_ccy} and aren't compared with the price.")

    st.subheader(f"{next(step)}. Historical cash flows")
    st.caption(f"Annual figures{' in ' + unit_word if unit_word else ''} of {fin_ccy}. FCFF = EBIT x (1 - tax rate) "
               "+ depreciation & amortization - capital expenditure + change in working capital (as reported in "
               "the cash flow statement, where a negative number means cash was tied up).")
    blocks = blocks_for_display(table, scale)
    show_table(blocks, key="dcf_blocks", column_config=accounting_config(blocks.columns, scale),
               height=table_height(len(blocks)), download_label="Download historical cash flows (CSV)",
               filename=f"{fname}_dcf_building_blocks_{units_label.lower()}.csv")

    rf, rf_note = risk_free_rate()
    computed_beta, months = None, 0
    bench_loaded, _ = attempt(load_history, benchmark, "max") if benchmark else (None, None)
    if bench_loaded is not None:
        computed_beta, _, months = monthly_beta(adjusted_close(daily), adjusted_close(bench_loaded[0]))
    beta_default, beta_source = choose_beta(computed_beta, months, display_symbol(benchmark), num(info.get("beta")))
    beta_note = beta_source + " To measure beta against another index, change \"Compare with\" in Performance."

    cap = capital_structure(info, statements["balance"])
    shares = cap["shares"]
    equity_value = None
    if price and shares and factor:
        equity_value = price * shares / factor  # market value of equity in the reporting currency
    elif factor == 1.0:
        equity_value = num(info.get("marketCap"))
    tax_hist = latest(table["Effective tax rate (%)"])
    tax_default = min(max(tax_hist / 100, 0.0), 0.4) if tax_hist is not None else DEFAULT_TAX
    interest = latest(table["Interest expense"])
    if interest and cap["debt"] > 0:
        kd_default, kd_note = min(max(interest / cap["debt"], 0.01), 0.15), "Latest interest expense divided by total debt."
    else:
        kd_default, kd_note = rf + 0.015, "Risk-free rate plus 1.5%, because interest expense or debt wasn't reported."

    st.subheader(f"{next(step)}. Discount rate (WACC)")
    st.caption("Pre-filled from the data. Change any number and everything below updates.")
    c = st.columns(4)
    rf_in = c[0].number_input("Risk-free rate (%)", value=float(round(rf * 100, 2)), step=0.05, format="%.2f",
                              key=f"dcf_rf_{symbol}", help=rf_note)
    beta_in = c[1].number_input("Beta", value=float(round(beta_default, 2)), step=0.05, format="%.2f",
                                key=f"dcf_beta_{symbol}", help=beta_note)
    erp_in = c[2].number_input("Equity risk premium (%)", value=float(DEFAULT_ERP * 100), step=0.25, format="%.2f",
                               key=f"dcf_erp_{symbol}",
                               help="An assumption. Aswath Damodaran publishes a monthly implied premium for the "
                                    "S&P 500 at pages.stern.nyu.edu/~adamodar.")
    kd_in = c[3].number_input("Pre-tax cost of debt (%)", value=float(round(kd_default * 100, 2)), step=0.25,
                              format="%.2f", key=f"dcf_kd_{symbol}", help=kd_note)
    c = st.columns(4)
    tax_in = c[0].number_input("Tax rate (%)", min_value=0.0, max_value=60.0, value=float(round(tax_default * 100, 2)),
                               step=0.5, format="%.2f", key=f"dcf_tax_{symbol}",
                               help="Latest effective tax rate from the income statement.")
    w = compute_wacc(rf_in / 100, beta_in, erp_in / 100, kd_in / 100, tax_in / 100, equity_value, cap["debt"])
    c[1].metric("Cost of equity", fmt_pct(w["ke"]), help="Risk-free rate + beta x equity risk premium (CAPM).")
    c[2].metric("Debt share of capital", fmt_pct(w["wd"]), help="Total debt / (total debt + market value of equity).")
    c[3].metric("WACC", fmt_pct(w["wacc"]))
    wacc, manual = w["wacc"], st.toggle("Enter WACC by hand", key=f"dcf_wacc_manual_{symbol}")
    if manual:
        wacc = st.number_input("WACC (%)", value=float(round(w["wacc"] * 100, 2)), step=0.25, format="%.2f",
                               key=f"dcf_wacc_{symbol}") / 100
    ke_note = "Risk-free rate + beta x equity risk premium, from step 2."
    ddm_heading = lambda: f"{next(step)}. Dividend discount model"  # noqa: E731

    st.subheader(f"{next(step)}. Cash flow forecast")
    kinds = [k for k in (FCFF, FCF) if table[k].notna().any()]
    if not kinds:
        st.info("Yahoo Finance doesn't report enough cash flow data to project free cash flow for this company.")
        ddm_block(symbol, daily, price, price_ccy, w["ke"], ddm_heading(), ke_note)
        return
    c = st.columns([3, 2], vertical_alignment="bottom")
    kind = c[0].selectbox("Cash flow to project", kinds, key=f"dcf_kind_{symbol}",
                          help="FCFF is the textbook pairing with WACC. Reported free cash flow is simpler but has "
                               "already paid interest, so it slightly understates cash available to all investors.")
    use_avg = c[1].toggle("Start from the 3-year average", key=f"dcf_avg_{symbol}", help="Smooths out a single unusual year.")
    series = table[kind].dropna()
    if use_avg and len(series) >= 3:
        base_default, base_note = float(series.iloc[-3:].mean()), f"average of fiscal {series.index[-3].year} to {series.index[-1].year}"
    else:
        base_default, base_note = float(series.iloc[-1]), f"fiscal year ending {fmt_day(series.index[-1])}"

    rev_cagr, cf_cagr = cagr(table["Revenue"]), cagr(series)
    est, _ = attempt(load_estimates, symbol)
    references = analyst_growth(est)
    candidate = rev_cagr if rev_cagr is not None else references.get("analysts' long-term EPS growth", 0.05)
    growth_default = min(max(candidate, 0.0), 0.15)

    c = st.columns(4)
    unit_label = f"{unit_word} {fin_ccy}".strip()
    base_in = c[0].number_input(f"Starting cash flow ({unit_label})", value=float(round(base_default / scale, 2)),
                                format="%.2f", key=f"dcf_base_{symbol}_{kind}_{use_avg}_{units_label}",
                                help=f"Pre-filled with {kind} for the {base_note}.")
    growth_in = c[1].number_input("Growth in year 1 (%)", value=float(round(growth_default * 100, 2)), step=0.5,
                                  format="%.2f", key=f"dcf_growth_{symbol}",
                                  help="Pre-filled with historical revenue growth, kept between 0% and 15%.")
    years_in = c[2].number_input("Forecast years", min_value=3, max_value=15, value=5, step=1, key=f"dcf_years_{symbol}")
    tg_in = c[3].number_input("Terminal growth (%)", value=float(DEFAULT_TERMINAL_GROWTH * 100), step=0.25, format="%.2f",
                              key=f"dcf_tg_{symbol}",
                              help="Growth forever after the forecast. Usually no higher than long-run GDP growth.")
    fade = st.toggle("Fade growth in a straight line to the terminal rate by the last forecast year", value=True,
                     key=f"dcf_fade_{symbol}")
    refs = []
    if rev_cagr is not None:
        refs.append(f"revenue grew {rev_cagr:.1%} a year over the statements shown")
    if cf_cagr is not None:
        refs.append(f"{'FCFF' if kind == FCFF else 'free cash flow'} grew {cf_cagr:.1%} a year")
    refs += [f"{label} is {value:.1%}" for label, value in references.items()]
    if refs:
        st.caption("For reference: " + "; ".join(refs) + ".")

    terminal = tg_in / 100
    if wacc <= terminal:
        st.error(f"WACC has to be higher than terminal growth, otherwise the terminal value is infinite. "
                 f"WACC is {wacc:.2%} and terminal growth is {terminal:.2%}: raise WACC (for example the beta or "
                 f"equity risk premium) or lower terminal growth.")
        ddm_block(symbol, daily, price, price_ccy, w["ke"], ddm_heading(), ke_note)
        return
    if base_in <= 0:
        st.warning("The starting cash flow is zero or negative, so the model values the business below zero. "
                   "Try the 3-year average, a normalized figure you choose, or a different valuation method.")
    net_claims = cap["debt"] - cap["cash"] + cap["minority"] + cap["preferred"]
    params = {"base_cf": base_in * scale, "growth": growth_in / 100, "years": int(years_in),
              "terminal_growth": terminal, "wacc": wacc, "fade": fade, "net_claims": net_claims,
              "shares": shares, "first_year": int(table.index[-1].year)}
    result = run_dcf(**params)
    per_share = result["per_share"] if shares else None
    value_ccy = price_ccy if factor else fin_ccy
    value_px = per_share * factor if (per_share is not None and factor) else per_share
    upside = value_px / price - 1 if (value_px is not None and price and factor) else None

    st.subheader(f"{next(step)}. Valuation")
    m = st.columns(5)
    m[0].metric(f"Enterprise value ({fin_ccy})", fmt_money(result["enterprise_value"]))
    m[1].metric("Less net debt and other claims", fmt_money(net_claims),
                help="Total debt - cash and short-term investments + minority interest + preferred stock.")
    m[2].metric("Equity value", fmt_money(result["equity_value"]))
    m[3].metric(f"DCF value per share ({value_ccy})", fmt_price(value_px) if value_px is not None else "n/a",
                delta=f"{upside:+.1%} vs last price" if upside is not None else None)
    m[4].metric(f"Last price ({price_ccy})", fmt_price(price))
    if not shares:
        st.caption("Shares outstanding weren't available, so there's no value per share.")
    if result["enterprise_value"] > 0:
        st.caption(f"The terminal value makes up {result['pv_terminal'] / result['enterprise_value']:.0%} of "
                   f"enterprise value. Shares outstanding used: {fmt_money(shares)}.")
    projection = result["projection"].copy()
    cf_col, pv_col = f"Cash flow ({unit_label})", f"Present value ({unit_label})"
    projection[cf_col] = projection.pop("Cash flow") / scale
    projection[pv_col] = projection.pop("Present value") / scale
    projection = projection[["Fiscal year", "Growth (%)", cf_col, "Discount factor", pv_col]]
    config = {"Fiscal year": st.column_config.NumberColumn(format="%d"),
              "Growth (%)": st.column_config.NumberColumn(format="%.2f%%"),
              "Discount factor": st.column_config.NumberColumn(format="%.4f"),
              **accounting_config([cf_col, pv_col], scale)}
    show_table(projection, key="dcf_projection", hide_index=True, column_config=config,
               filename=f"{fname}_dcf_projection.csv", download_label="Download projection (CSV)")

    st.subheader(f"{next(step)}. Sensitivity of value per share")
    grid = sensitivity_grid(params, wacc, terminal)
    if per_share is not None and factor:
        grid = grid * factor
    show_chart(sensitivity_figure(grid, price if upside is not None else None), "sensitivity", symbol)
    st.caption(f"Value per share in {value_ccy}. Rows move WACC and columns move terminal growth in half-point "
               "steps. Green cells sit above the last price, red cells below.")
    st.download_button("Download sensitivity table (CSV)", data=to_csv_bytes(grid), mime="text/csv",
                       file_name=f"{fname}_dcf_sensitivity.csv", key="download_dcf_sensitivity")

    # Reverse DCF ---------------------------------------------------------------------------------------
    st.subheader(f"{next(step)}. Reverse DCF: what today's price implies")
    if price and factor and shares and base_in > 0:
        target = price / factor
        g_implied, w_implied = implied_growth(params, target), implied_wacc(params, target)
        c = st.columns(3)
        if g_implied is None:
            top = run_dcf(**{**params, "growth": IMPLIED_GROWTH_RANGE[1]})["per_share"]
            g_text = (f"Above {IMPLIED_GROWTH_RANGE[1]:.0%}" if top < target
                      else f"Below {IMPLIED_GROWTH_RANGE[0]:.0%}")
        else:
            g_text = fmt_pct(g_implied)
        c[0].metric("Year-1 growth the price implies", g_text,
                    delta=f"{(g_implied - params['growth']) * 100:+.1f} pts vs your {params['growth']:.1%}"
                    if g_implied is not None else None, delta_color="off",
                    help="Holding your WACC, terminal growth, forecast years and fade fixed.")
        if w_implied is None:
            floor = params["terminal_growth"] + 0.0005
            w_text = (f"Below {floor:.1%}" if run_dcf(**{**params, "wacc": floor})["per_share"] < target
                      else "Above 60%")
        else:
            w_text = fmt_pct(w_implied)
        c[1].metric("Return the price implies", w_text,
                    delta=f"{(w_implied - wacc) * 100:+.1f} pts vs your WACC" if w_implied is not None else None,
                    delta_color="off",
                    help="The discount rate at which your forecast is worth exactly today's price: roughly the yearly "
                         "return you'd earn buying now if your forecast comes true.")
        c[2].metric("Your WACC", fmt_pct(wacc))
        if g_implied is not None:
            st.caption(f"To justify {fmt_price(price)} {price_ccy}, cash flow would need to grow about "
                       f"{g_implied:.1%} in year 1"
                       + (f", fading to {terminal:.1%}" if fade else "") + f", discounted at your WACC of {wacc:.2%}. "
                       "If that looks too high, the price is pricing in more than your forecast; if it looks low, "
                       "the market is more cautious than you.")
        else:
            st.caption(f"No year-1 growth between {IMPLIED_GROWTH_RANGE[0]:.0%} and {IMPLIED_GROWTH_RANGE[1]:.0%} "
                       f"matches the price at these inputs, so the price and your forecast are far apart. Check the "
                       f"starting cash flow and the discount rate.")
    else:
        st.caption("A reverse DCF needs a positive starting cash flow, shares outstanding and a price in a currency the "
                   "model can convert.")

    # Scenarios -------------------------------------------------------------------------------------------
    st.subheader(f"{next(step)}. Bear, base and bull scenarios")
    st.caption("Each scenario shifts the inputs above by the amounts in the table, in percentage points. Edit any "
               "white cell, including the probabilities.")
    edits = st.data_editor(
        SCENARIO_DEFAULTS, key=f"scenarios_{symbol}", hide_index=True, disabled=["Scenario"], num_rows="fixed",
        column_config={"Probability (%)": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=5.0, format="%.0f"),
                       "Growth change (pts)": st.column_config.NumberColumn(step=0.5, format="%+.1f"),
                       "WACC change (pts)": st.column_config.NumberColumn(step=0.25, format="%+.2f"),
                       "Terminal growth change (pts)": st.column_config.NumberColumn(step=0.25, format="%+.2f")})
    scenarios, weighted = scenario_results(params, edits, factor or 1.0, price if factor else None)
    show_table(scenarios, key="scenario_results", hide_index=True, filename=f"{fname}_scenarios.csv",
               download_label="Download scenarios (CSV)",
               column_config={c_: st.column_config.NumberColumn(format="%.2f") for c_ in scenarios.columns if c_ != "Scenario"})
    if weighted is not None:
        weighted_upside = weighted / price - 1 if (price and factor) else None
        st.metric(f"Probability-weighted value per share ({value_ccy})", fmt_price(weighted),
                  delta=f"{weighted_upside:+.1%} vs last price" if weighted_upside is not None else None)
        total_prob = float(edits["Probability (%)"].fillna(0).sum())
        if abs(total_prob - 100) > 0.01:
            st.caption(f"The probabilities add up to {total_prob:g}%, so they're rescaled to 100% for the weighted value.")

    ddm = ddm_block(symbol, daily, price, price_ccy, w["ke"], ddm_heading(), ke_note)

    # Excel -------------------------------------------------------------------------------------------------
    st.subheader(f"{next(step)}. Excel model")
    st.caption("Downloads this valuation as a workbook with working formulas: change any blue input cell and the DCF, "
               "sensitivity table, scenarios and dividend model all recalculate. It opens in Excel, Numbers, "
               "Google Sheets and LibreOffice.")
    model = {
        "name": info.get("longName") or info.get("shortName") or symbol, "symbol": symbol, "fin_ccy": fin_ccy or price_ccy,
        "price_ccy": price_ccy, "units_label": units_label, "scale": scale, "price": price or 0.0,
        "fx": factor or 1.0, "shares": shares or 0.0, "debt": cap["debt"], "cash": cap["cash"],
        "minority": cap["minority"], "preferred": cap["preferred"], "rf": rf_in / 100, "beta": beta_in,
        "erp": erp_in / 100, "kd": kd_in / 100, "tax": tax_in / 100, "manual_wacc": manual, "wacc_manual": wacc,
        "kind": kind, "base_cf": base_in * scale, "base_note": base_note, "growth": growth_in / 100,
        "terminal_growth": terminal, "years": int(years_in), "fade": fade, "first_year": int(table.index[-1].year),
        "as_of": fmt_day(daily.index.max()), "rf_note": rf_note, "beta_note": beta_source, "kd_note": kd_note,
        "history": table, "scenarios": edits, "ddm": ddm, "downloaded": f"{date.today():%Y-%m-%d}",
    }
    try:
        workbook = build_excel_model(model)
    except ImportError:
        st.warning("The Excel download needs the openpyxl package. In Terminal, run: python3 -m pip install openpyxl")
    else:
        st.download_button("Download Excel model (.xlsx)", data=workbook, file_name=f"{fname}_valuation_model.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           key="download_excel_model", type="primary")


@st.fragment
def section_fund_valuation(symbol: str, info: dict, daily: pd.DataFrame, price: float | None, price_ccy: str):
    st.caption("A fund is worth what its holdings are worth, so there are no company cash flows to discount. "
               "A dividend discount model on its distributions shows how much of the price the payout alone supports.")
    st.subheader("1. Cost of equity")
    ke = cost_of_equity_inputs(symbol, info, daily, current_benchmark())
    ddm_block(symbol, daily, price, price_ccy, ke, "2. Dividend discount model", "From the inputs above.")


@st.fragment
def section_earnings(symbol: str, info: dict, price: float | None):
    fname = safe_filename(symbol)
    data, err = attempt(load_earnings, symbol)
    if err:
        st.warning(err)
    data = data or {}
    facts = calendar_facts(data.get("calendar"))
    earnings = tidy_earnings(data.get("earnings_dates"), data.get("earnings_history"))
    today = pd.Timestamp.today().normalize()
    upcoming = earnings[(earnings.index >= today) & earnings["Reported EPS"].isna()] if not earnings.empty else earnings
    past = earnings.dropna(subset=["Reported EPS"]) if not earnings.empty else earnings
    next_date = facts.get("next_date") or (upcoming.index.min() if len(upcoming) else None)

    st.subheader("Next earnings")
    m = st.columns(4)
    when = fmt_range(next_date, facts.get("last_possible")) if next_date is not None else "Not announced"
    m[0].metric("Next earnings date", when, help="Companies often confirm the exact day a few weeks ahead.")
    eps_expected = facts.get("eps_avg")
    if eps_expected is None and len(upcoming):
        eps_expected = num(upcoming["EPS estimate"].iloc[-1])
    eps_range = (f"Analysts' range: {fmt_amount(facts['eps_low'])} to {fmt_amount(facts['eps_high'])}."
                 if facts.get("eps_low") is not None and facts.get("eps_high") is not None else None)
    m[1].metric("Earnings per share expected", fmt_amount(eps_expected) if eps_expected is not None else "n/a", help=eps_range)
    m[2].metric("Revenue expected", fmt_money(facts.get("revenue_avg")) if facts.get("revenue_avg") else "n/a")
    recent = past.head(8)
    if len(recent) and recent["Surprise (%)"].notna().any():
        beats = int((recent["Surprise (%)"] > 0).sum())
        m[3].metric("Beat analysts' estimate", f"{beats} of the last {len(recent)}")

    st.subheader("Earnings versus estimates")
    if past.empty:
        st.caption("Yahoo Finance has no earnings history for this ticker.")
    else:
        show_chart(earnings_figure(earnings), "earnings", symbol)
        st.caption("Green bars beat the average analyst estimate, red bars missed it. Surprise is reported EPS minus "
                   "the estimate, as a percentage of the estimate.")
        shown = earnings.copy()
        shown.index = shown.index.strftime("%Y-%m-%d")
        show_table(shown, key="earnings_history", filename=f"{fname}_earnings.csv",
                   download_label="Download earnings history (CSV)",
                   column_config={"EPS estimate": st.column_config.NumberColumn(format="%.2f"),
                                  "Reported EPS": st.column_config.NumberColumn(format="%.2f"),
                                  "Surprise (%)": st.column_config.NumberColumn(format="%+.1f%%")})

    st.subheader("Quarterly results")
    raw, qerr = attempt(load_statement, symbol, "income", "quarterly")
    if qerr:
        st.warning(qerr)
    else:
        units_label = current_units()
        scale = UNITS[units_label]
        rev, net = get_row(raw, IS_ROWS["revenue"]), get_row(raw, ["Net Income Common Stockholders", "Net Income"])
        eps = get_row(raw, ["Diluted EPS", "Basic EPS"])
        quarters = pd.DataFrame({"Revenue": rev / scale, "Net income": net / scale, "Diluted EPS": eps}).sort_index().tail(8)
        if quarters.dropna(how="all").empty:
            st.caption("Yahoo Finance has no quarterly income statement for this ticker.")
        else:
            fin_ccy = info.get("financialCurrency") or info.get("currency") or ""
            unit_text = f"{'' if units_label == 'Raw' else units_label.lower() + ' of '}{fin_ccy}".strip()
            show_chart(quarterly_figure(quarters, unit_text), "quarters", symbol)
            shown = quarters.copy()
            shown.index = shown.index.strftime("%Y-%m-%d")
            shown.index.name = "Quarter end"
            st.caption(f"Revenue and net income in {unit_text}; earnings per share as reported.")
            show_table(shown, key="quarterly_results", filename=f"{fname}_quarterly_results.csv",
                       download_label="Download quarterly results (CSV)",
                       column_config={**accounting_config(["Revenue", "Net income"], scale),
                                      "Diluted EPS": st.column_config.NumberColumn(format="%.2f")})

    est, err = attempt(load_estimates, symbol)
    if err:
        st.warning(err)
        return
    if not est:
        return
    targets = est.get("analyst_price_targets")
    if isinstance(targets, dict) and targets:
        st.subheader("Analyst price targets")
        m = st.columns(4)
        for col, key in zip(m, ("low", "mean", "median", "high")):
            value = num(targets.get(key))
            delta = f"{value / price - 1:+.1%} vs last price" if (value and price) else None
            col.metric(key.title(), fmt_price(value), delta)
        st.download_button("Download price targets (CSV)", data=to_csv_bytes(dict_table(targets, "Measure", "Price"), index=False),
                           file_name=f"{fname}_price_targets.csv", mime="text/csv", key="download_targets")
    titles = {"earnings_estimate": "EPS estimates", "revenue_estimate": "Revenue estimates",
              "eps_trend": "How EPS estimates moved", "growth_estimates": "Growth estimates",
              "recommendations": "Ratings by month", "upgrades_downgrades": "Upgrades and downgrades"}
    available = [k for k in titles if isinstance(est.get(k), pd.DataFrame) and not est[k].empty]
    if available:
        st.subheader("Analyst estimates and ratings")
        for tab, k in zip(st.tabs([titles[k] for k in available]), available):
            with tab:
                tidy, config = tidy_analyst_table(k, est[k])
                show_table(tidy, key=f"analyst_{k}", filename=f"{fname}_{k}.csv", column_config=config)


@st.fragment
def section_compare(symbol: str, qtype: str, info: dict, directory: dict, rf: float):
    peers = []
    if qtype == "EQUITY" and info.get("industryKey"):
        found, _ = attempt(load_industry_peers, str(info["industryKey"]))
        peers = [p for p in (found or []) if p.upper() != symbol.upper()][:4]
    key = f"compare_{safe_filename(symbol)}"
    options = list(directory) + [p for p in peers if p not in directory]
    picks = st.multiselect(
        "Compare with", options, default=None if key in st.session_state else peers, key=key, max_selections=7,
        format_func=lambda s: directory.get(s, str(s)), accept_new_options=True,
        placeholder="Type a company name or ticker",
        help="Starts with the largest companies in the same industry. Add or remove up to 7 tickers.")
    others = [p if p in directory else str(p).strip().upper() for p in picks]
    symbols = [symbol] + [p for p in dict.fromkeys(others) if p and p.upper() != symbol.upper()]
    if len(symbols) == 1:
        st.info("Add one or more tickers above to compare them with " + display_symbol(symbol) + ".")
        return

    histories, infos, failed = {}, {}, []
    with st.spinner("Loading the comparison..."):
        for s in symbols:
            loaded, err = attempt(load_history, s, "max")
            if loaded is None:
                failed.append(s)
                continue
            histories[s] = loaded[0]
            infos[s] = attempt(load_info, s)[0] or {}
    if failed:
        st.warning(f"Couldn't load {', '.join(failed)}. Check the symbol, or wait a minute if Yahoo is limiting requests.")
    symbols = [s for s in symbols if s in histories]
    if len(symbols) < 2:
        return

    st.subheader("Growth of 10,000")
    window = choice_bar("Chart period", ["1Y", "3Y", "5Y", "10Y", "Max"], "5Y", "compare_window")
    series = {s: adjusted_close(histories[s]) for s in symbols}
    series = {s: v for s, v in series.items() if len(v) > 1}
    latest_day = max(v.index[-1] for v in series.values())
    begin = window_start(pd.DatetimeIndex([latest_day]), "All" if window == "Max" else window)
    start = max([v.index[0] for v in series.values()] + ([begin] if begin is not None else []))
    trimmed = {s: v[v.index >= start] for s, v in series.items() if (v.index >= start).sum() > 1}
    show_chart(compare_figure({display_symbol(k): v for k, v in trimmed.items()}), "compare", symbol)
    currencies = sorted({str(infos[s].get("currency") or "") for s in symbols} - {""})
    st.caption(f"Every line starts at 10,000 on {fmt_day(start)}, the first date all of them have prices in this "
               "period. Adjusted closes, so dividends count as reinvested."
               + (f" Prices are in different currencies ({', '.join(currencies)}), so returns are in each one's own "
                  "currency." if len(currencies) > 1 else ""))

    st.subheader("Side by side")
    snapshot = with_peer_median(compare_snapshot(symbols, infos, histories, rf), display_symbol(symbol))
    config = {c: st.column_config.NumberColumn(format="%.2f") for c in snapshot.select_dtypes("number").columns}
    config["Market cap or fund assets"] = st.column_config.NumberColumn(format="compact")
    st.dataframe(snapshot, column_config=config, key="table_compare")
    st.caption("Valuation and profitability figures are Yahoo Finance's latest; returns, volatility and drawdowns are "
               "calculated from prices. Returns over 3 and 5 years are per year."
               + (f" The peer median leaves out {symbol}." if "Peer median" in snapshot.index else ""))

    st.subheader("How closely they move together")
    corr = monthly_correlations({display_symbol(k): v for k, v in series.items()})
    st.dataframe(corr, column_config={c: st.column_config.NumberColumn(format="%.2f") for c in corr.columns},
                 key="table_correlation")
    st.caption("Correlation of monthly returns over up to the last 5 years: 1 means they move together, 0 unrelated, "
               "-1 opposite.")
    prices = pd.DataFrame({display_symbol(k): v for k, v in series.items()})
    prices.index.name = "Date"
    d1, d2, d3, _ = st.columns([1.2, 1.2, 1.2, 1])
    d1.download_button("Download comparison table (CSV)", data=to_csv_bytes(snapshot), mime="text/csv",
                       file_name=f"{safe_filename(symbol)}_comparison.csv", key="download_compare")
    d2.download_button("Download adjusted prices (CSV)", data=to_csv_bytes(prices), mime="text/csv",
                       file_name=f"{safe_filename(symbol)}_comparison_prices.csv", key="download_compare_prices")
    d3.download_button("Download correlations (CSV)", data=to_csv_bytes(corr), mime="text/csv",
                       file_name=f"{safe_filename(symbol)}_correlations.csv", key="download_correlations")


@st.fragment
def section_profile(symbol: str, info: dict, meta: dict, qtype: str):
    fname = safe_filename(symbol)
    name = info.get("longName") or info.get("shortName") or meta.get("longName") or display_symbol(symbol)
    summary = info.get("longBusinessSummary")
    panel = facts_html(company_facts(info, qtype), info.get("website"))
    if summary or panel:
        st.subheader(f"About {name}")
        if summary and panel:
            left, right = st.columns([2, 1], gap="large")
            left.markdown(f'<div class="td-about">{text(summary)}</div>', unsafe_allow_html=True)
            right.markdown(panel, unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="td-about">{text(summary)}</div>' if summary else panel, unsafe_allow_html=True)

    officers = info.get("companyOfficers")
    if isinstance(officers, list) and officers:
        st.subheader("Officers")
        table = pd.DataFrame(officers)
        keep = [c for c in ("name", "title", "age", "yearBorn", "totalPay") if c in table]
        table = (table[keep] if keep else table).rename(
            columns={"name": "Name", "title": "Title", "age": "Age", "yearBorn": "Born", "totalPay": "Total pay"})
        show_table(table, key="officers", hide_index=True, filename=f"{fname}_officers.csv",
                   column_config={"Age": st.column_config.NumberColumn(format="%d"),
                                  "Born": st.column_config.NumberColumn(format="%d"),
                                  "Total pay": st.column_config.NumberColumn(format="compact")})

    if qtype in ("EQUITY", ""):
        holders, _ = attempt(load_holders, symbol)
        holders = holders or {}
        summary_items = holder_summary(holders.get("major_holders"))
        lists = {k: t for k, t in (("institutional_holders", "Largest institutions"), ("mutualfund_holders", "Largest funds"))
                 if isinstance(holders.get(k), pd.DataFrame) and not holders[k].empty}
        if summary_items or lists:
            st.subheader("Ownership")
            if summary_items:
                st.markdown(stat_grid(summary_items), unsafe_allow_html=True)
            if lists:
                for tab, k in zip(st.tabs(list(lists.values())), lists):
                    with tab:
                        tidy, config = tidy_holders(holders[k])
                        show_table(tidy, key=f"holders_{k}", hide_index=True, filename=f"{fname}_{k}.csv",
                                   column_config=config)

    fields = {k: v for k, v in info.items() if k not in ("companyOfficers", "longBusinessSummary")}
    if fields or meta:
        with st.expander(f"All the data fields Yahoo Finance returns ({len(fields) + len(meta)})"):
            column, _ = st.columns([3, 1])
            with column:
                if fields:
                    display, raw = fields_tables(fields)
                    st.caption("Company and market fields")
                    show_table(display, key="all_fields", hide_index=True, csv_df=raw, height=420,
                               filename=f"{fname}_all_fields.csv")
                if meta:
                    display, raw = fields_tables(meta)
                    st.caption("Price data fields")
                    show_table(display, key="price_meta", hide_index=True, csv_df=raw, height=320,
                               filename=f"{fname}_price_metadata.csv")


# ---------------------------------------------------------------------------
# Download everything as one ZIP
# ---------------------------------------------------------------------------
def build_zip(symbol: str, qtype: str, info: dict, meta: dict, daily: pd.DataFrame, units_label: str,
              benchmark: str, rf: float) -> bytes:
    scale = UNITS[units_label]
    folder = safe_filename(symbol)
    notes, written = [], []
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        def add(name, df, index=True):
            if isinstance(df, pd.DataFrame) and not df.empty:
                zf.writestr(f"{folder}/{name}", to_csv_bytes(df, index=index))
                written.append(name)

        add("prices_daily_full_history.csv", daily)
        add("dividends.csv", nonzero(daily, "Dividends").to_frame("Dividend per share"))
        add("splits.csv", nonzero(daily, "Stock Splits").to_frame("Ratio"))
        add("capital_gains.csv", nonzero(daily, "Capital Gains").to_frame("Capital gain per share"))
        add("profile_all_fields.csv", dict_table({k: v for k, v in sorted(info.items()) if k != "companyOfficers"}), index=False)
        add("price_metadata.csv", dict_table(meta), index=False)

        adj = adjusted_close(daily)
        if len(adj) > 1:
            bench = None
            if benchmark and benchmark != symbol.upper():
                loaded, err = attempt(load_history, benchmark, "max")
                bench = adjusted_close(loaded[0]) if loaded is not None else None
                if err:
                    notes.append(f"Benchmark {display_symbol(benchmark)}: {err}")
            add("trailing_returns.csv", returns_table(adj, bench, display_symbol(symbol), display_symbol(benchmark)),
                index=False)
            add("risk_return.csv", risk_table(adj, rf), index=False)

        if qtype in ("EQUITY", ""):
            for kind, title in STATEMENTS:
                for freq in ("yearly", "quarterly"):
                    raw, err = attempt(load_statement, symbol, kind, freq)
                    if err:
                        notes.append(f"{title} ({freq}): {err}")
                    add(f"{kind}_{freq}_{units_label.lower()}.csv", tidy_statement(raw, scale))
            annual = {k: attempt(load_statement, symbol, k, "yearly")[0] for k in ("income", "cashflow", "balance")}
            blocks = dcf_history(annual["income"], annual["cashflow"], annual["balance"])
            if not blocks.empty:
                add(f"dcf_building_blocks_{units_label.lower()}.csv", blocks_for_display(blocks, scale))
            add("financial_ratios.csv", financial_ratios(annual["income"], annual["balance"], annual["cashflow"], daily))
            earnings, err = attempt(load_earnings, symbol)
            if err:
                notes.append(f"Earnings: {err}")
            elif earnings:
                add("earnings.csv", tidy_earnings(earnings.get("earnings_dates"), earnings.get("earnings_history")))
            for loader, label in ((load_estimates, "analyst data"), (load_holders, "ownership data")):
                data, err = attempt(loader, symbol)
                if err:
                    notes.append(f"{label.capitalize()}: {err}")
                for k, v in (data or {}).items():
                    if isinstance(v, pd.DataFrame):
                        add(f"{k}.csv", v)
                    elif isinstance(v, dict):
                        add(f"{k}.csv", dict_table(v), index=False)
        if qtype in FUND_TYPES or qtype == "":
            data, err = attempt(load_fund_data, symbol)
            if err and qtype in FUND_TYPES:
                notes.append(f"Fund data: {err}")
            for k, v in (data or {}).items():
                if isinstance(v, pd.DataFrame):
                    add(f"fund_{k}.csv", v)
                elif isinstance(v, dict):
                    add(f"fund_{k}.csv", dict_table(v), index=False)

        readme = [f"Data for {symbol} from Yahoo Finance via yfinance, downloaded {date.today():%Y-%m-%d}.",
                  f"Financial statement figures are in {units_label.lower()} of the reporting currency "
                  "(per-share values and tax rates unscaled).",
                  "Prices: Close is adjusted for splits only; Adj Close also reflects dividends.", "",
                  "Files:"] + [f"  {name}" for name in written]
        if notes:
            readme += ["", "Not included:"] + [f"  {n}" for n in notes]
        zf.writestr(f"{folder}/README.txt", "\n".join(readme))
    return buffer.getvalue()


def render_export(symbol: str, qtype: str, info: dict, meta: dict, daily: pd.DataFrame):
    units_label = current_units()
    benchmark = current_benchmark()
    export_id = f"{symbol}|{units_label}|{benchmark}"
    saved = st.session_state.get("zip_payload")
    if saved and saved[0] == export_id:
        st.download_button("Download all data (ZIP)", data=saved[1], file_name=f"{safe_filename(symbol)}_data.zip",
                           mime="application/zip", key="zip_download", type="primary",
                           help="Every table for this ticker as CSV files.")
    elif st.button("Get all data as a ZIP", key="zip_prepare",
                   help="Collects every table for this ticker into one ZIP of CSV files. Takes a few seconds."):
        with st.spinner("Collecting every table...", show_time=True):
            payload = build_zip(symbol, qtype, info, meta, daily, units_label, benchmark, risk_free_rate()[0])
        st.session_state["zip_payload"] = (export_id, payload)
        files = len(zipfile.ZipFile(io.BytesIO(payload)).namelist())
        flash(f"Your ZIP is ready with {files} files. Click Download all data.", ":material/folder_zip:")
        st.rerun()


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def render_welcome():
    st.subheader("Look up any stock, ETF, fund, index or crypto")
    st.markdown("Click the search box above and type a few letters of a company name or ticker. Pick a suggestion "
                "to see its prices, financial statements, DCF inputs and more. Every table can be downloaded as CSV.")
    st.caption("Or start with one of these:")
    for col, (symbol, label) in zip(st.columns(len(QUICK_PICKS)), QUICK_PICKS):
        col.button(label, key=f"quick_{safe_filename(symbol)}", on_click=_choose, args=(symbol,), width="stretch")


def main():
    st.markdown(CSS + theme_css(), unsafe_allow_html=True)
    show_flash()
    path = logo_path()
    logo = f'<img class="td-logo" src="{logo_data_uri(path, os.path.getmtime(path))}" alt="">' if path else ""
    st.markdown(f'<div class="td-brand">{logo}<span>{text(APP_NAME)}</span></div>', unsafe_allow_html=True)
    directory, note = search_directory()
    search_col, refresh_col = st.columns([6, 1], vertical_alignment="bottom")
    with search_col:
        choice = st.selectbox("Find a stock, ETF, fund, index or crypto", list(directory), index=None,
                              format_func=lambda s: directory.get(s, str(s)), key="ticker", accept_new_options=True,
                              placeholder="Start typing a company name or ticker, like Apple or VTI")
    with refresh_col:
        if st.button("Refresh prices", key="refresh", width="stretch",
                     help="Prices are kept for 15 minutes. This fetches them again now."):
            load_history.clear()
            load_info.clear()
            st.toast("Prices refreshed", icon=":material/refresh:")
    st.caption(note or "Not in the suggestions? Type the Yahoo Finance symbol, for example RELIANCE.NS or VOD.L, "
                       "and press Enter.")

    symbol = ""
    if choice:
        symbol = choice if choice in directory else pick_symbol(str(choice))
    if not symbol:
        render_welcome()
        return

    placeholder = st.empty()
    if st.session_state.get("shown_symbol") != symbol:  # only for a new ticker, so tab switches don't flicker
        placeholder.markdown(SKELETON, unsafe_allow_html=True)
    loaded, err = attempt(load_history, symbol, "max")
    if err:
        placeholder.empty()
        st.error(err)
        return
    daily, meta = loaded
    info, info_err = attempt(load_info, symbol)
    info = info or {}
    placeholder.empty()
    st.session_state["shown_symbol"] = symbol
    qtype = str(info.get("quoteType") or meta.get("instrumentType") or "").upper()
    price_ccy = info.get("currency") or meta.get("currency") or ""

    render_quote(symbol, info, meta, daily, qtype, price_ccy)
    if info_err:
        st.caption(f"Company details are incomplete because the profile didn't load. {info_err}")
    price, _ = price_snapshot(info, meta, daily)

    sections = sections_for(qtype)
    nav_col, export_col = st.columns([5, 1.6], vertical_alignment="center")
    with nav_col:
        section = choice_bar("Section", sections, sections[0], f"section_{qtype or 'ANY'}", label_visibility="collapsed")
    with export_col:
        render_export(symbol, qtype, info, meta, daily)

    body = st.empty()  # one slot, so the old tab is swapped out at once instead of fading piece by piece
    with body.container():
        if section == "Price":
            section_price(symbol, daily, price_ccy)
        elif section == "Performance":
            section_performance(symbol, daily, risk_free_rate()[0])
        elif section == "Statements":
            section_statements(symbol, info)
        elif section == "Ratios":
            section_ratios(symbol, daily)
        elif section == "Valuation":
            if qtype in FUND_TYPES:
                section_fund_valuation(symbol, info, daily, price, price_ccy)
            else:
                section_valuation(symbol, info, daily, price, price_ccy)
        elif section == "Earnings":
            section_earnings(symbol, info, price)
        elif section == "Holdings":
            section_fund(symbol)
        elif section == "Dividends":
            section_dividends(symbol, daily, price)
        elif section == "Compare":
            section_compare(symbol, qtype, info, directory, risk_free_rate()[0])
        else:
            section_profile(symbol, info, meta, qtype)
    st.divider()
    st.caption(DATA_NOTE)


main()
