# Mew2 Tracker

A website for project work. Search any stock, ETF, mutual fund, index, currency or crypto on Yahoo Finance, analyze it, value it, and download every table as CSV, plus the valuation as an Excel model with working formulas. Built with Streamlit and yfinance.

## Run it

You need Python 3.10 or newer.

```bash
cd mew2-tracker-v2
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

On Windows, use `py` instead of `python3`. Your browser opens at http://localhost:8501. Leave the terminal open while you use the site, and press Ctrl+C there to stop it.

## Your logo

Save an image in this folder with a name starting with `logo`, such as `logo.png` (PNG, JPG, WEBP or GIF). It appears to the left of the name and as the browser-tab icon; refresh the page after adding it. Without one, the site shows just the name. When the site is online, the repository's `logo.png` is used, so use an image you have the rights to publish.

## Look and feel

- **Light and dark mode.** Open the ⋮ menu at the top right, choose Settings, and pick Light, Dark or your system setting. Charts and the header switch colors with it.
- **Fonts.** Sora for headings and IBM Plex Sans for everything else. Both load from Google Fonts, so they need an internet connection; otherwise Streamlit's own font is used.
- **Smoother updates.** Each tab refreshes on its own: changing a chart period or a DCF input redraws only that tab, not the whole page.
- **Chart images.** Hover over a chart and click the camera icon to save it as a sharp PNG for reports.

## What's on the site

**Search.** Click the box at the top and type a few letters of a company name or ticker. Suggestions appear as you type. They cover every U.S.-listed stock and ETF (from Nasdaq's public symbol directory), plus major indices, crypto, currencies, futures and popular mutual funds. For other markets, type the Yahoo symbol, such as RELIANCE.NS or VOD.L, and press Enter.

**Tabs.** Each tab has its settings at the top, next to what they change.

| Tab | What you get |
|---|---|
| Price | Candlestick or line chart. Indicators you can switch on: 50/200 moving averages, Bollinger bands, RSI and MACD. Price table for any period and bar size. Downloads include the indicators you switched on. |
| Performance | Trailing returns vs a benchmark, volatility, Sharpe ratio, max drawdown, beta, and growth and drawdown charts. |
| Statements | Income statement, balance sheet and cash flow, annual or quarterly, in your choice of units. |
| Ratios | Margins, ROE, ROA, ROIC, growth, debt and liquidity ratios, and valuation multiples at each fiscal year end, plus a chart of any ratio. |
| Valuation | Nine numbered steps. Historical cash flows, WACC, forecast, valuation and sensitivity, then: a reverse DCF (the growth and return today's price implies); bear, base and bull scenarios you can edit, with a probability-weighted value; a dividend discount model; and an Excel download of the whole model with working formulas. |
| Earnings | Next earnings date and estimates, a beat/miss history against analysts' estimates, quarterly revenue and profit, analyst price targets and estimates. |
| Dividends | Dividend history, yearly totals, trailing yield, 5-year growth, splits. |
| Compare | Starts with the biggest companies in the same industry; add up to 7 tickers of any kind. Shows a growth-of-10,000 chart, a side-by-side table with a peer median, and how closely returns move together. |
| Holdings (ETFs and funds) | Fund facts, asset mix, top holdings, sector weights. |
| Profile | Business summary, officers, ownership, and every field Yahoo returns. |

ETFs and funds get a Valuation tab with a dividend discount model, since a fund is worth what its holdings are worth.

**Get all data as a ZIP**, next to the tabs, bundles every table for the current ticker. **Refresh prices**, next to the search box, fetches fresh prices; otherwise they're kept for 15 minutes.

## The valuation, step by step

1. **Historical cash flows.** Revenue, EBIT, tax rate, NOPAT, D&A, capital expenditure, working capital, FCFF and reported free cash flow.
2. **WACC.** The cost of equity comes from CAPM. The risk-free rate is the live 10-year Treasury yield; beta is 5-year monthly against your benchmark. The cost of debt is interest divided by debt. The weights use market value of equity and total debt. Every number can be edited.
3. **Forecast.** Starting cash flow, year-1 growth (pre-filled with revenue growth), forecast years, and terminal growth. Growth can fade in a straight line to the terminal rate.
4. **Valuation.** Enterprise value, then minus debt, plus cash, minus minority interest and preferred stock, gives equity value and value per share.
5. **Sensitivity.** Value per share across WACC and terminal growth.
6. **Reverse DCF.** The year-1 growth, and separately the discount rate, at which your forecast equals today's price.
7. **Scenarios.** Bear, base and bull shifts to growth, WACC and terminal growth, each with a probability.
8. **Dividend discount model.** A two-stage model on the last 12 months' dividends, discounted at the cost of equity. Best for banks, insurers, utilities and funds; it undervalues companies that also buy back shares.
9. **Excel model.** The same valuation as a workbook:
   - Inputs, DCF, Sensitivity, Scenarios, DDM, Historical and Notes sheets.
   - Blue cells are inputs, black cells are formulas, green cells link between sheets.
   - Change any blue cell and everything recalculates.
   - Forecast years can be changed from 3 to 15 inside the workbook.
   - For a reverse DCF in Excel, use Goal Seek (instructions on the Notes sheet).

Set the equity risk premium (default 4.5%) and terminal growth (default 2.5%) yourself; they're assumptions. Aswath Damodaran publishes a monthly implied equity risk premium at pages.stern.nyu.edu/~adamodar.

Things to keep in mind:

- **Short history.** Yahoo usually provides only 4–5 fiscal years of statements and 5–6 quarters.
- **Banks and insurers.** Free-cash-flow DCFs don't suit lenders; the app flags them and points to the dividend model.
- **Foreign listings and ADRs.** Values are converted at the latest exchange rate. Check the ADR ratio before comparing with the price.
- **Ratios and stock splits.** Valuation multiples use the price at each fiscal year end. Share counts reported before a stock split are restated so they match split-adjusted prices.

## Put it online (optional)

1. Push this folder to a GitHub repository, including `app.py`, `requirements.txt` and `.streamlit/config.toml`.
2. Sign in at share.streamlit.io with GitHub.
3. Click Create app and choose the repository with `app.py` as the main file.

Updating later is easy: upload a new `app.py` to the repository and the site redeploys within a minute or so. Yahoo limits requests from shared cloud servers more than from home connections, so the local version is the more dependable one.

## If something goes wrong

| Message or symptom | Fix |
|---|---|
| "Yahoo Finance is limiting requests" | Wait a minute and refresh. Loaded data is cached, so moving between tabs doesn't re-download it. |
| "No price data" | Check the symbol and add the exchange suffix for non-U.S. listings. |
| Everything stops loading at once | Yahoo has probably changed something. Run `python3 -m pip install -U yfinance`. |
| The Excel download is missing | Run `python3 -m pip install openpyxl`. |
| Prices look stale | Click "Refresh prices". |

## Tests

`tests/run_scenarios.py` runs the whole app with fake Streamlit, Plotly and yfinance modules and synthetic data, so it needs no internet:

```bash
python3 tests/run_scenarios.py
```

It covers 120 scenarios: every tab, stocks, ETFs, funds, indices and crypto, foreign currencies, missing data, failed requests and the ZIP export. It also checks the math:

- the DCF against a hand calculation
- that the reverse DCF recovers known inputs
- RSI and MACD on known series
- share counts restated across stock splits

If LibreOffice is installed, it also recalculates the downloaded Excel model and confirms it matches the app.

## Data terms

yfinance is an open-source project that isn't affiliated with Yahoo. Yahoo's data is for personal, research and educational use. Prices may be delayed.
