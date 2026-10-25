"""Synthetic stand-in for yfinance. Shapes, names and dtypes follow what yfinance returns."""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

__version__ = "1.3.0-stub"
CALLS: list = []          # (symbol, what) for request counting
TODAY = pd.Timestamp.today().normalize()


class YFRateLimitError(Exception):
    def __init__(self):
        super().__init__("Too Many Requests. Rate limited. Try after a while.")


class YFPricesMissingError(Exception):
    pass


EQ = "EQUITY"
PROFILES = {
    "AAPL": dict(kind=EQ, start="1980-12-12", p0=0.1, mu=0.2, sigma=0.33, div_start="2012-08-09", div=0.095,
                 split=("2020-08-31", 4.0), ccy="USD", fin="USD", exch="NasdaqGS", name="Apple Inc.",
                 sector="Technology", industry="Consumer Electronics", shares=14.8e9, rev=391e9),
    "MSFT": dict(kind=EQ, start="1986-03-13", p0=0.1, mu=0.2, sigma=0.3, div_start="2003-02-19", div=0.08,
                 ccy="USD", fin="USD", exch="NasdaqGS", name="Microsoft Corporation", sector="Technology",
                 industry="Software - Infrastructure", shares=7.4e9, rev=280e9),
    "VTI": dict(kind="ETF", start="2001-06-15", p0=55, mu=0.08, sigma=0.18, div_start="2001-12-26", div=0.3,
                ccy="USD", exch="NYSEArca", name="Vanguard Total Stock Market Index Fund ETF Shares"),
    "VTSAX": dict(kind="MUTUALFUND", start="2000-11-13", p0=28, mu=0.08, sigma=0.18, div_start="2001-03-26",
                  div=0.12, gains=True, ccy="USD", exch="Nasdaq", name="Vanguard Total Stock Market Index Fund Admiral Shares"),
    "^GSPC": dict(kind="INDEX", start="1985-01-02", p0=165, mu=0.09, sigma=0.17, ccy="USD", exch="SNP", name="S&P 500"),
    "^NSEI": dict(kind="INDEX", start="2007-09-17", p0=4500, mu=0.11, sigma=0.2, ccy="INR", exch="NSE", name="NIFTY 50"),
    "^TNX": dict(kind="RATE", start="1990-01-02", level=4.31, ccy="USD", exch="CBOE", name="CBOE Interest Rate 10 Year T No"),
    "TWDUSD=X": dict(kind="FX", start="2003-12-01", level=0.0312, ccy="USD", exch="CCY", name="TWD/USD"),
    "BTC-USD": dict(kind="CRYPTOCURRENCY", start="2014-09-17", p0=450, mu=0.6, sigma=0.75, ccy="USD", exch="CCC",
                    name="Bitcoin USD", weekends=True),
    "TSM": dict(kind=EQ, start="1997-10-09", p0=8, mu=0.15, sigma=0.35, div_start="2004-07-14", div=0.3, ccy="USD",
                fin="TWD", exch="NYSE", name="Taiwan Semiconductor Manufacturing Company Limited",
                sector="Technology", industry="Semiconductors", shares=5.19e9, rev=2.89e12),
    "VOD.L": dict(kind=EQ, start="1990-01-02", p0=60, mu=0.01, sigma=0.3, div_start="1995-06-01", div=4.0,
                  ccy="GBp", fin="GBP", exch="LSE", name="Vodafone Group Public Limited Company",
                  sector="Communication Services", industry="Telecom Services", shares=25.0e9, rev=37e9, tz="Europe/London"),
    "BRLX": dict(kind=EQ, start="2010-01-04", p0=10, mu=0.05, sigma=0.4, ccy="USD", fin="BRL", exch="NYSE",
                 name="Brazil Example S.A.", sector="Utilities", industry="Utilities - Regulated Electric",
                 shares=1.0e9, rev=40e9),
    "JPM": dict(kind=EQ, start="1980-03-17", p0=3, mu=0.1, sigma=0.3, div_start="1990-01-04", div=0.4, ccy="USD",
                fin="USD", exch="NYSE", name="JPMorgan Chase & Co.", sector="Financial Services",
                industry="Banks - Diversified", shares=2.8e9, rev=160e9, bank=True),
    "NEGF": dict(kind=EQ, start="2019-06-03", p0=20, mu=-0.05, sigma=0.6, ccy="USD", fin="USD", exch="NasdaqGS",
                 name="Negative Cash Corp", sector="Healthcare", industry="Biotechnology", shares=0.3e9, rev=0.2e9,
                 negative=True),
    "INFOFAIL": dict(kind=EQ, start="2005-01-03", p0=30, mu=0.07, sigma=0.25, ccy="USD", fin="USD", exch="NYSE",
                     name="Info Fails Inc.", shares=1e9, rev=10e9, info_fails=True),
    "EMPTYFIN": dict(kind=EQ, start="2021-01-04", p0=10, mu=0.0, sigma=0.4, ccy="USD", fin="USD", exch="NYSE",
                     name="No Statements Corp", shares=1e8, rev=0, no_statements=True),
    "RATELIM": dict(kind=EQ, rate_limited=True),
    "ODDB": dict(kind=EQ, start="2008-01-02", p0=15, mu=0.06, sigma=0.3, beta=0.0, ccy="USD", fin="USD", exch="NYSE",
                 name="Odd Beta Holdings", sector="Industrials", industry="Conglomerates", shares=0.5e9, rev=8e9),
}


def _seed(symbol: str) -> int:
    return zlib.crc32(symbol.encode()) & 0xFFFFFFFF


def _base_history(symbol: str) -> pd.DataFrame:
    p = PROFILES[symbol]
    rng = np.random.default_rng(_seed(symbol))
    dates = pd.date_range(p["start"], TODAY, freq="D" if p.get("weekends") else "B")
    n = len(dates)
    if p["kind"] in ("RATE", "FX"):
        close = p["level"] * np.exp(np.cumsum(rng.normal(0, 0.004, n)) - np.cumsum(rng.normal(0, 0.004, n)).mean() * 0)
        close = close / close[-1] * p["level"]
    else:
        days = 365 if p.get("weekends") else 252
        if p["kind"] in (EQ, "ETF", "MUTUALFUND"):
            mkt = _history("^GSPC")["Close"]
            mkt.index = mkt.index.tz_localize(None)
            mret = np.log(mkt).diff().reindex(dates).fillna(0).to_numpy()
            beta = p.get("beta", 1.0)
            idio = np.sqrt(max(p["sigma"] ** 2 - (beta * 0.17) ** 2, 0.0025))
            steps = (p["mu"] - beta * 0.09) / days + beta * mret + rng.normal(0, idio / np.sqrt(days), n)
        else:
            steps = rng.normal(p["mu"] / days, p["sigma"] / np.sqrt(days), n)
        close = p["p0"] * np.exp(np.cumsum(steps))
    open_ = close * (1 + rng.normal(0, 0.004, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
    volume = rng.integers(1_000_000, 90_000_000, n).astype("int64") if p["kind"] not in ("RATE", "FX") else np.zeros(n, "int64")
    df = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=dates)
    df["Dividends"] = 0.0
    df["Stock Splits"] = 0.0
    if p.get("div_start"):
        pay_dates = pd.date_range(p["div_start"], TODAY, freq="QS-FEB") + pd.Timedelta(days=8)
        for i, d in enumerate(pay_dates):
            pos = df.index.searchsorted(d)
            if pos < n:
                df.iloc[pos, df.columns.get_loc("Dividends")] = round(p["div"] * (1.06 ** (i / 4)), 4)
    if p.get("split"):
        d, ratio = p["split"]
        pos = df.index.searchsorted(pd.Timestamp(d))
        if pos < n:
            df.iloc[pos, df.columns.get_loc("Stock Splits")] = ratio
    if p["kind"] in ("ETF", "MUTUALFUND"):
        df["Capital Gains"] = 0.0
        if p.get("gains"):
            for d in pd.date_range(p["start"], TODAY, freq="YS-DEC"):
                pos = df.index.searchsorted(d + pd.Timedelta(days=14))
                if pos < n:
                    df.iloc[pos, df.columns.get_loc("Capital Gains")] = 0.35
    # Adj Close: back-adjust for dividends paid after each date
    factor = np.ones(n)
    divs = df["Dividends"].to_numpy()
    running = 1.0
    for i in range(n - 1, -1, -1):
        factor[i] = running
        if divs[i] > 0 and i > 0:
            running *= 1 - divs[i] / close[i - 1]
    df["Adj Close"] = close * factor
    df = df[["Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits"]
            + (["Capital Gains"] if "Capital Gains" in df else [])]
    return df.tz_localize(p.get("tz", "UTC" if p.get("weekends") else "America/New_York"))


_CACHE: dict = {}


def _history(symbol):
    if symbol not in _CACHE:
        _CACHE[symbol] = _base_history(symbol)
    return _CACHE[symbol]


def _statement(p, kind, freq):
    if p.get("no_statements"):
        return pd.DataFrame()
    if freq == "yearly":
        ends = [pd.Timestamp(TODAY.year - k - 1, 9, 30) for k in range(5)]  # newest first, like Yahoo
        span = 1.0
    elif freq == "quarterly":
        ends = [(pd.Timestamp(TODAY.year, 1, 1) - pd.offsets.QuarterEnd(k + 1)).normalize() for k in range(6)]
        span = 0.25
    else:
        raise ValueError(f"Illegal argument: freq={freq!r}")
    neg = p.get("negative")
    rows = {}
    for i, end in enumerate(ends):
        g = (1.06) ** (-i)
        rev = p["rev"] * g * span
        ebit = rev * (-1.5 if neg else 0.3)
        pretax = ebit * 1.02
        tax = max(pretax * 0.16, 0)
        col = {}
        if kind == "income":
            col = {"Total Revenue": rev, "Operating Revenue": rev, "Cost Of Revenue": rev * 0.55,
                   "Gross Profit": rev * 0.45, "Pretax Income": pretax, "Tax Provision": tax,
                   "Net Income": pretax - tax, "Interest Expense": rev * 0.009, "Reconciled Depreciation": rev * 0.03,
                   "EBITDA": ebit + rev * 0.03, "Basic EPS": (pretax - tax) / p["shares"], "Diluted EPS": (pretax - tax) / p["shares"] * 0.99,
                   "Diluted Average Shares": p["shares"] * 1.01, "Tax Rate For Calcs": 0.16}
            if not p.get("bank"):
                col.update({"Operating Income": ebit, "EBIT": pretax + rev * 0.009})
        elif kind == "cashflow":
            ocf = rev * (-1.2 if neg else 0.3)
            col = {"Operating Cash Flow": ocf, "Free Cash Flow": ocf - rev * 0.03, "Repurchase Of Capital Stock": -rev * 0.2,
                   "Cash Dividends Paid": -rev * 0.04}
            if not p.get("bank"):
                col.update({"Capital Expenditure": -rev * 0.03, "Depreciation And Amortization": rev * 0.03,
                            "Change In Working Capital": rev * 0.004 * (-1) ** i})
        elif kind == "balance":
            col = {"Total Assets": rev * 0.9, "Total Debt": rev * 0.27, "Stockholders Equity": rev * 0.17 * (1 + 0.05 * i),
                   "Current Assets": rev * 0.37, "Current Liabilities": rev * 0.4, "Inventory": rev * 0.02,
                   "Cash Cash Equivalents And Short Term Investments": rev * 0.17, "Ordinary Shares Number": p["shares"],
                   "Share Issued": p["shares"], "Minority Interest": rev * 0.002 if p.get("bank") else np.nan,
                   "Working Capital": rev * 0.05}
        for k, v in col.items():
            rows.setdefault(k, {})[end] = v
    df = pd.DataFrame(rows).T[ends].astype(object)
    if freq == "yearly":
        df[ends[-1]] = np.nan  # Yahoo often returns an empty oldest year
    df.loc["Tax Provision", ends[0]] = None
    return df


class _FundsData:
    def __init__(self, symbol):
        self._symbol, self._p = symbol, PROFILES.get(symbol, {})

    def __getattr__(self, name):
        if self._p.get("kind") not in ("ETF", "MUTUALFUND"):
            raise Exception(f"No Fund data found for {self._symbol}")
        s = self._symbol
        data = {
            "description": "The fund seeks to track the CRSP US Total Market Index, about $1.9 trillion in assets.",
            "fund_overview": {"categoryName": "Large Blend", "family": "Vanguard", "legalType": "Exchange Traded Fund"},
            "fund_operations": pd.DataFrame({s: [0.0003, 0.02, 1.9e12], "Category Average": [0.0085, 0.4, 5.4e9]},
                                            index=pd.Index(["Annual Report Expense Ratio", "Annual Holdings Turnover", "Total Net Assets"], name="Attributes")),
            "asset_classes": {"cashPosition": 0.0021, "stockPosition": 0.9968, "bondPosition": 0.0, "preferredPosition": 0.0,
                              "convertiblePosition": 0.0, "otherPosition": 0.0011},
            "top_holdings": pd.DataFrame({"Name": ["NVIDIA Corp", "Microsoft Corp", "Apple Inc", "Amazon.com Inc", "Meta Platforms Inc"],
                                          "Holding Percent": [0.068, 0.061, 0.057, 0.036, 0.025]},
                                         index=pd.Index(["NVDA", "MSFT", "AAPL", "AMZN", "META"], name="Symbol")),
            "equity_holdings": pd.DataFrame({s: [26.1, 4.5, 3.1, 17.0, 175e9, 0.12], "Category Average": [25.0, 4.2, 3.0, 16.0, 150e9, 0.11]},
                                            index=pd.Index(["Price/Earnings", "Price/Book", "Price/Sales", "Price/Cashflow",
                                                            "Median Market Cap", "3 Year Earnings Growth"], name="Average")),
            "bond_holdings": pd.DataFrame(),
            "bond_ratings": {"us_government": 0.0, "aaa": 0.0, "bbb": 0.0, "below_b": 0.0},
            "sector_weightings": {"realestate": 0.025, "consumer_cyclical": 0.105, "basic_materials": 0.02,
                                  "technology": 0.335, "financial_services": 0.13, "healthcare": 0.095},
        }
        if name not in data:
            raise AttributeError(name)
        return data[name]


class Ticker:
    def __init__(self, ticker, session=None):
        self.ticker = ticker.upper() if not ticker.endswith("=X") else ticker
        self._p = PROFILES.get(self.ticker)
        self._meta = None

    # --- prices -----------------------------------------------------------
    def history(self, period=None, interval="1d", start=None, end=None, prepost=False, actions=True,
                auto_adjust=True, back_adjust=False, repair=False, keepna=False, rounding=False, timeout=10,
                raise_errors=False):
        CALLS.append((self.ticker, f"history {period} {interval}"))
        if self._p is None:
            if raise_errors:
                raise YFPricesMissingError(f"${self.ticker}: possibly delisted; no price data found  (period={period})")
            return pd.DataFrame()
        if self._p.get("rate_limited"):
            raise YFRateLimitError()
        df = _history(self.ticker).copy()
        tz = df.index.tz
        if period and start is None:
            if period == "max":
                pass
            else:
                days = {"5d": 7, "1mo": 31, "1y": 366}[period]
                df = df[df.index >= df.index[-1] - pd.Timedelta(days=days)]
        else:
            s = pd.Timestamp(start).tz_localize(tz)
            e = pd.Timestamp(end).tz_localize(tz) if end else df.index[-1] + pd.Timedelta(days=1)
            df = df[(df.index >= s) & (df.index < e)]
        if interval != "1d":
            rule = {"1wk": "W-MON", "1mo": "MS", "3mo": "QS"}[interval]
            agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Adj Close": "last",
                   "Volume": "sum", "Dividends": "sum", "Stock Splits": "max"}
            if "Capital Gains" in df:
                agg["Capital Gains"] = "sum"
            df = df.resample(rule, label="left", closed="left").agg(agg).dropna(subset=["Close"])
        if auto_adjust:
            ratio = df["Adj Close"] / df["Close"]
            for c in ("Open", "High", "Low", "Close"):
                df[c] = df[c] * ratio
            df = df.drop(columns="Adj Close")
        if not actions:
            df = df.drop(columns=[c for c in ("Dividends", "Stock Splits", "Capital Gains") if c in df])
        if df.empty and raise_errors:
            raise YFPricesMissingError(f"${self.ticker}: possibly delisted; no price data found")
        self._meta = {
            "currency": self._p["ccy"], "symbol": self.ticker, "exchangeName": self._p["exch"],
            "fullExchangeName": self._p["exch"], "instrumentType": "EQUITY" if self._p["kind"] == "RATE" else self._p["kind"],
            "firstTradeDate": 345479400, "regularMarketPrice": float(_history(self.ticker)["Close"].iloc[-1]),
            "longName": self._p["name"], "shortName": self._p["name"][:20], "timezone": "EST",
            "tradingPeriods": pd.DataFrame({"start": [1], "end": [2]}), "validRanges": ["1d", "5d", "max"],
            "hasPrePostMarketData": True,
        }
        return df

    @property
    def history_metadata(self):
        return self._meta or {}

    # --- profile --------------------------------------------------------------
    @property
    def info(self):
        CALLS.append((self.ticker, "info"))
        p = self._p
        if p is None:
            return {"trailingPegRatio": None}
        if p.get("info_fails"):
            raise Exception("401 Client Error: Unauthorized for url")
        last = float(_history(self.ticker)["Close"].iloc[-1])
        base = {"symbol": self.ticker, "quoteType": "INDEX" if p["kind"] == "RATE" else ("CURRENCY" if p["kind"] == "FX" else p["kind"]),
                "longName": p["name"], "shortName": p["name"][:24], "currency": p["ccy"], "exchange": p["exch"],
                "fullExchangeName": p["exch"], "regularMarketPrice": last, "regularMarketPreviousClose": last * 0.99,
                "fiftyTwoWeekHigh": last * 1.2, "maxAge": 86400, "tradeable": False, "validRanges": None}
        if p["kind"] == EQ:
            base.update({"financialCurrency": p["fin"], "sector": p["sector"], "industry": p["industry"],
                         "country": "United States", "website": "https://example.com", "fullTimeEmployees": 164000,
                         "longBusinessSummary": "Designs products and returned $110 billion to shareholders in buybacks.",
                         "marketCap": int(last * p["shares"]), "beta": 1.21, "trailingPE": 33.4,
                         "sharesOutstanding": int(p["shares"]), "totalDebt": p["rev"] * 0.27, "totalCash": p["rev"] * 0.17,
                         "dividendYield": 0.44, "companyOfficers": [
                             {"maxAge": 1, "name": "Ms. Jane Doe", "age": 61, "title": "CEO & Director", "yearBorn": 1964, "totalPay": 16_500_000},
                             {"maxAge": 1, "name": "Mr. John Roe", "title": "CFO", "fiscalYear": 2025}],
                         "governanceEpochDate": 1717200000, "numberOfAnalystOpinions": 40, "targetMeanPrice": last * 1.1,
                         "underlyingSymbol": None, "isEsgPopulated": False, "sectorDisp": p["sector"],
                         "industryKey": p["industry"].lower().replace(" - ", "-").replace(" ", "-"),
                         "forwardPE": 28.9, "priceToBook": 45.1, "enterpriseToEbitda": 24.3, "enterpriseToRevenue": 8.2,
                         "operatingMargins": 0.31, "profitMargins": 0.24, "returnOnEquity": 1.5, "revenueGrowth": 0.06})
        elif p["kind"] in ("ETF", "MUTUALFUND"):
            base.update({"totalAssets": 1.9e12, "category": "Large Blend", "fundFamily": "Vanguard",
                         "longBusinessSummary": "Tracks the CRSP US Total Market Index ($ amounts in USD).",
                         "yield": 0.0121, "navPrice": last})
        return base

    # --- statements -------------------------------------------------------------
    def _stmt(self, kind, pretty, freq):
        CALLS.append((self.ticker, f"{kind} {freq}"))
        if self._p and self._p.get("rate_limited"):
            raise YFRateLimitError()
        if not pretty:
            raise AssertionError("app should request pretty row names")
        if self._p is None or self._p["kind"] != EQ:
            return pd.DataFrame()
        return _statement(self._p, kind, freq)

    def get_income_stmt(self, as_dict=False, pretty=False, freq="yearly"):
        return self._stmt("income", pretty, freq)

    def get_balance_sheet(self, as_dict=False, pretty=False, freq="yearly"):
        return self._stmt("balance", pretty, freq)

    def get_cashflow(self, as_dict=False, pretty=False, freq="yearly"):
        return self._stmt("cashflow", pretty, freq)

    @property
    def funds_data(self):
        return _FundsData(self.ticker)

    # --- analysts & holders -------------------------------------------------------
    def _equity_only(self, name):
        CALLS.append((self.ticker, name))
        if self._p is None or self._p["kind"] != EQ:
            raise Exception(f"No {name} data for {self.ticker}")

    @property
    def analyst_price_targets(self):
        self._equity_only("targets")
        last = float(_history(self.ticker)["Close"].iloc[-1])
        return {"current": last, "high": last * 1.4, "low": last * 0.7, "mean": last * 1.1, "median": last * 1.08}

    @property
    def recommendations(self):
        self._equity_only("recommendations")
        return pd.DataFrame({"period": ["0m", "-1m", "-2m", "-3m"], "strongBuy": [5, 6, 6, 7], "buy": [24, 23, 22, 22],
                             "hold": [15, 15, 16, 16], "sell": [1, 1, 2, 2], "strongSell": [2, 2, 1, 1]})

    def _estimate_frame(self, scale):
        idx = pd.Index(["0q", "+1q", "0y", "+1y"], name="period")
        return pd.DataFrame({"avg": [1.6 * scale, 1.7 * scale, 7.4 * scale, 8.1 * scale],
                             "low": [1.5 * scale] * 4, "high": [1.8 * scale] * 4,
                             "numberOfAnalysts": [29, 27, 40, 40], "growth": [0.08, 0.07, 0.09, 0.095]}, index=idx)

    @property
    def earnings_estimate(self):
        self._equity_only("earnings_estimate")
        return self._estimate_frame(1)

    @property
    def revenue_estimate(self):
        self._equity_only("revenue_estimate")
        return self._estimate_frame(1e9)

    @property
    def eps_trend(self):
        self._equity_only("eps_trend")
        return pd.DataFrame({"current": [1.6, 1.7], "7daysAgo": [1.6, 1.69], "30daysAgo": [1.58, 1.68]},
                            index=pd.Index(["0q", "+1q"], name="period"))

    @property
    def growth_estimates(self):
        self._equity_only("growth_estimates")
        return pd.DataFrame({"stockTrend": [0.08, 0.07, 0.09, 0.095, 0.104], "indexTrend": [0.05, 0.06, 0.07, 0.11, np.nan]},
                            index=pd.Index(["0q", "+1q", "0y", "+1y", "LTG"], name="period"))

    @property
    def upgrades_downgrades(self):
        self._equity_only("upgrades_downgrades")
        idx = pd.DatetimeIndex(["2026-10-01 10:05:00", "2026-09-12 08:00:00"], name="GradeDate")
        return pd.DataFrame({"Firm": ["Morgan Stanley", "Barclays"], "ToGrade": ["Overweight", "Underweight"],
                             "FromGrade": ["Overweight", "Equal-Weight"], "Action": ["main", "down"]}, index=idx)

    @property
    def major_holders(self):
        self._equity_only("major_holders")
        return pd.DataFrame({"Value": [0.0207, 0.623, 0.636, 6214.0]},
                            index=pd.Index(["insidersPercentHeld", "institutionsPercentHeld",
                                            "institutionsFloatPercentHeld", "institutionsCount"], name="Breakdown"))

    @property
    def institutional_holders(self):
        self._equity_only("institutional_holders")
        return pd.DataFrame({"Date Reported": pd.to_datetime(["2026-06-30", "2026-06-30"]),
                             "Holder": ["Vanguard Group Inc", "Blackrock Inc."], "pctHeld": [0.093, 0.075],
                             "Shares": [1_390_000_000, 1_120_000_000], "Value": [3.1e11, 2.5e11], "pctChange": [0.01, -0.002]})

    @property
    def mutualfund_holders(self):
        self._equity_only("mutualfund_holders")
        return pd.DataFrame()


class Search:
    UNIVERSE = [
        {"symbol": "AAPL", "shortname": "Apple Inc.", "longname": "Apple Inc.", "quoteType": "EQUITY", "exchDisp": "NASDAQ", "exchange": "NMS"},
        {"symbol": "APLE", "shortname": "Apple Hospitality REIT, Inc.", "longname": "Apple Hospitality REIT, Inc.", "quoteType": "EQUITY", "exchDisp": "NYSE"},
        {"symbol": "VTI", "shortname": "Vanguard Total Stock Market ETF", "quoteType": "ETF", "exchDisp": "NYSEArca"},
        {"symbol": "VTSAX", "shortname": "Vanguard Total Stock Mkt Idx Adm", "quoteType": "MUTUALFUND", "typeDisp": "Fund"},
        {"symbol": "^GSPC", "shortname": "S&P 500", "quoteType": "INDEX", "exchDisp": "SNP"},
        {"symbol": "VOD.L", "shortname": "VODAFONE GROUP PLC", "quoteType": "EQUITY", "exchDisp": "London"},
        {"symbol": "TSM", "longname": "Taiwan Semiconductor Manufacturing Company Limited", "quoteType": "EQUITY", "exchDisp": "NYSE"},
        {"symbol": "BTC-USD", "shortname": "Bitcoin USD", "quoteType": "CRYPTOCURRENCY", "exchDisp": "CCC"},
    ]

    def __init__(self, query, max_results=8, news_count=8, lists_count=8, include_cb=True, include_nav_links=False,
                 include_research=False, include_cultural_assets=False, enable_fuzzy_query=False, recommended=8,
                 session=None, timeout=30, raise_errors=True):
        CALLS.append((query, "search"))
        if query.upper() == "SEARCHFAIL":
            raise Exception("Failed to perform search for query 'SEARCHFAIL'")
        q = query.lower()
        hits = [d for d in self.UNIVERSE if q in d["symbol"].lower() or q in (d.get("longname") or d.get("shortname") or "").lower()]
        if q == "vanguard":
            hits = [d for d in self.UNIVERSE if d["symbol"] in ("VTI", "VTSAX")]
        self.quotes = hits[:max_results] + ([{"shortname": "no symbol entry"}] if q == "apple" else [])
        self.news = []


def _quarter_ends(n_past=8, n_future=2):
    ends = [(TODAY - pd.offsets.QuarterEnd(k)).normalize() for k in range(n_past, 0, -1)]
    future = [(TODAY + pd.offsets.QuarterEnd(k)).normalize() for k in range(1, n_future + 1)]
    return ends, future


def _earnings_dates(self):
    self._equity_only("earnings_dates")
    past, future = _quarter_ends()
    when = [d + pd.Timedelta(days=30) for d in past + future]
    rng = np.random.default_rng(_seed(self.ticker))
    est = np.round(np.linspace(1.2, 1.9, len(when)), 2)
    rep = np.round(est * (1 + rng.normal(0.03, 0.04, len(when))), 2)
    rep[len(past):] = np.nan
    idx = pd.DatetimeIndex(when).tz_localize("America/New_York")
    idx.name = "Earnings Date"
    df = pd.DataFrame({"EPS Estimate": est, "Reported EPS": rep, "Surprise(%)": np.round((rep / est - 1) * 100, 2)}, index=idx)
    return df.sort_index(ascending=False)


def _earnings_history(self):
    self._equity_only("earnings_history")
    past, _ = _quarter_ends(4, 0)
    return pd.DataFrame({"epsEstimate": [1.5, 1.6, 1.7, 1.8], "epsActual": [1.55, 1.58, 1.8, 1.9],
                         "epsDifference": [0.05, -0.02, 0.1, 0.1], "surprisePercent": [0.033, -0.012, 0.059, 0.055]},
                        index=pd.DatetimeIndex(past, name="quarter"))


def _calendar(self):
    self._equity_only("calendar")
    _, future = _quarter_ends()
    first = (future[0] + pd.Timedelta(days=30)).date()
    return {"Dividend Date": first, "Ex-Dividend Date": first, "Earnings Date": [first, first + pd.Timedelta(days=4)],
            "Earnings High": 2.1, "Earnings Low": 1.7, "Earnings Average": 1.92,
            "Revenue High": 1.1e11, "Revenue Low": 0.98e11, "Revenue Average": 1.04e11}


Ticker.earnings_dates = property(_earnings_dates)
Ticker.earnings_history = property(_earnings_history)
Ticker.calendar = property(_calendar)


class Industry:
    TOP = {"consumer-electronics": ["AAPL", "SONY", "MSFT", "TSM", "VTI"], "semiconductors": ["NVDA", "TSM", "AVGO"]}

    def __init__(self, key, session=None):
        CALLS.append((key, "industry"))
        self.key = key

    @property
    def top_companies(self):
        if self.key == "broken-industry":
            raise Exception("industry lookup failed")
        symbols = self.TOP.get(self.key, [])
        return pd.DataFrame({"name": symbols, "rating": ["Buy"] * len(symbols), "market weight": [0.1] * len(symbols)},
                            index=pd.Index(symbols, name="symbol"))
