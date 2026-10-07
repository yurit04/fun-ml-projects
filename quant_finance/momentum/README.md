# Momentum notebooks

Work from the lecture *Everything You Always Wanted to Know About Momentum* (September 2026), implemented on the data already in this repo. Strategy Sharpes are gross of a risk-free rate, and gross of costs unless a cost is stated. Daily series are annualized with \(\sqrt{252}\); monthly series with \(\sqrt{12}\). Significance uses Newey–West HAC \(t\)-statistics, Mertens Sharpe standard errors, a stationary bootstrap, and Benjamini–Hochberg where several series are screened together.

## Files

| Notebook | What it is |
|---|---|
| `momentum_strategies.ipynb` | TSMOM, XSMOM, vol targeting, CUSUM, TS+XS blend, turnover and costs, risk stats, significance |
| `kalman_bocpd.ipynb` | SMA / EWMA / Donchian equivalence, Kalman local linear trend, BOCPD vs CUSUM |
| `multi_style_premia.ipynb` | Baz et al. (2015): TS momentum, XS momentum (12–1), 5-year value. Carry omitted |
| `dmn_sharpe_loss.ipynb` | Lim et al. (2019) Sharpe-loss LSTM, plus a one-day horizon-attention stand-in for Wood et al. (2021). Methods only |
| `../factor_models/03_factor_momentum.ipynb` | Ehsani & Linnainmaa (2022) on Ken French factors |

Data: `real_market_data.csv` (10 ETFs, 2007-01-03 to 2024-05-31: SPY, QQQ, IWM, EEM, EFA, TLT, LQD, GLD, DBC, VNQ). Factor momentum uses the Ken French cache in `../factor_models/data_cache/`.

All five notebooks were re-executed top to bottom on 2026-10-07 with the repo `.venv`; every number below is from that run.

## What the October 2026 review changed

1. **Start-up leverage (all four ETF notebooks).** Volatility estimates were used from their second or third observation. TSMOM opened with about 10× leverage in one ETF and lost 2.3% on its first trading day; vol-scaled XSMOM opened at 7.5× because its volatility had been estimated on a run of zero-filled returns. That one artifact moved the TSMOM Sharpe from 0.55 to 0.49. Every volatility estimate now needs 60 observations, no book trades in the first 126 days, and warmup days are NaN rather than zero.
2. **One evaluation window.** `momentum_strategies.ipynb` and the regime filters in `kalman_bocpd.ipynb` are scored on 2008-01-02 to 2024-05-31 for every series, SPY included. Previously each series was scored from its own first non-missing day.
3. **Costs.** `momentum_strategies.ipynb` and `dmn_sharpe_loss.ipynb` now report turnover, net Sharpe and break-even cost.
4. **Factor momentum is benchmarked against holding the factors.** The headline Sharpe did not distinguish timing from static premia.
5. **Deep momentum uses the paper's vol-scaled returns** alongside the raw-return version, adds seed ensembles, and tests them against a long-only book.
6. **BOCPD** no longer sees a zero-padded warmup, and its first 21 days are not counted as detections.

Earlier fixes still in place: the markdown repair, and the causal (lagged, expanding) CUSUM standardization that replaced a full-sample \(z\)-score.

## `momentum_strategies.ipynb`

Window 2008-01-02 to 2024-05-31 (16.4 years). An annualized Sharpe of about **0.49** is required for \(t = 2\).

| | Sharpe | Max DD | Calmar | HAC \(t\) | BH \(p\) | Turnover (× NAV / yr) | Sharpe at 5 bp | Break-even (bp) |
|---|---|---|---|---|---|---|---|---|
| TSMOM (vol-scaled trend) | 0.56 | −13% | 0.33 | 2.52 | 0.035 | 9.8 | 0.50 | 46.9 |
| Vol-scaled XSMOM | 0.32 | −33% | 0.11 | 1.35 | 0.27 | 81.2 | 0.06 | 6.0 |
| CUSUM-filtered XSMOM | 0.10 | −25% | 0.02 | 0.46 | 0.77 | 40.5 | −0.08 | 2.7 |
| Unscaled XSMOM | 0.05 | −25% | 0.00 | 0.28 | 0.78 | 36.1 | −0.10 | 1.8 |
| Combined TS+XS | 0.54 | −28% | 0.26 | 2.36 | 0.037 | 71.8 | 0.31 | 11.6 |
| SPY | 0.58 | −52% | 0.20 | 2.74 | 0.035 | — | — | — |

- TSMOM and the combined book reject zero after Benjamini–Hochberg. No momentum book is distinguishable from SPY (Jobson–Korkie \(p \ge 0.21\)).
- **The combination does not help here.** The two sleeves correlate at 0.45 and the cross-sectional leg is the weaker one. The blend's Sharpe (0.54) is below TSMOM alone (0.56 as built, 0.61 when vol-targeted to 15%), with a deeper drawdown and seven times the turnover.
- **Costs decide the ranking.** XSMOM is re-ranked daily and trades 36–81× NAV a year. The unscaled book breaks even at 1.8 bp. TSMOM is the only series that survives both the significance tests and a plausible cost.
- **CUSUM.** Causal standardization, 126-day warmup, live from the first week of 2008. It fires on 70 days (31 in 2008, 13 in 2011, 15 in 2020) and lifts unscaled XSMOM from 0.05 to 0.10. Across thresholds 2.0 to 4.0 the filtered Sharpe runs 0.09 to 0.16; a 2-year rolling window gives 0.04. None of these is distinguishable from the unfiltered book.

Drawdown length counts only episodes that trough below −5%.

## Factor momentum

`../factor_models/03_factor_momentum.ipynb`. Each Ken French factor except UMD is timed on the sign of its trailing 12-month return (`rolling(12).sum().shift(1)`). FactorMom is the equal-weight of those sleeves.

UMD on FactorMom has an insignificant intercept (long sample \(t = 1.49\), short sample \(t = 1.02\)). The reverse regression keeps a large intercept (\(t = 4.52\) and \(4.14\)). That is the paper's spanning asymmetry, on 5–7 published factors rather than a stock-level \(\beta_{ik}\) replication. In March–May 2009, UMD was −49% and FactorMom was −4.8%.

**The Sharpe is not the evidence.** FactorMom's Sharpe is 0.69 (1932–) and 0.74 (1964–) with HAC \(t\) above 6, but it is long about two-thirds of the time in factors with positive premia, and simply holding those factors gives 0.68 and 0.81. What does isolate timing:

| | Long sample (1932–) | Short sample (1964–) |
|---|---|---|
| Avg factor return after an up year / a down year | 7.9% / 1.1% | 5.7% / 0.1% |
| Up-minus-down spread, HAC \(t\) | 5.40 | 5.13 |
| FactorMom alpha over the passive factor book | 3.2% a year, \(t = 4.67\) | 3.3% a year, \(t = 5.66\) |
| UMD alpha over the passive factor book | 11.2% a year, \(t = 8.03\) | 10.7% a year, \(t = 6.02\) |

Static factor exposure leaves UMD's alpha intact. Timed factor exposure removes it.

## Kalman and BOCPD

`kalman_bocpd.ipynb`.

SMA vs EWMA strategy-return correlation is **0.91** (the linear-filter claim). Donchian vs SMA is **0.27**; Donchian is a breakout rule, so the theorem does not apply. Kalman vs EWMA return correlation is **0.75** and position correlation is **0.50**: the time-varying gain changes the trades. Kalman \(Q, R\) are fit on the first two years of each asset and then frozen; the signal is the filtered velocity, lagged one day. Post-warmup (2009–2024) Sharpe is 0.62 vs 0.55 for EWMA, Jobson–Korkie \(p = 0.73\).

BOCPD uses \(\Pr(r_t < 21)\), not \(\Pr(r_t = 0)\). Under a constant hazard the latter equals the hazard on every day, so it cannot be a detector. On 2008–2024, CUSUM (70 hits) moves unscaled XSMOM from 0.05 to 0.10 (\(p = 0.50\) vs raw). BOCPD flags 207 young-regime days, including 2014–2016, 2018, 2019 and 2022, and leaves the Sharpe at 0.04 (\(p = 0.86\) vs raw).

## Multi-style premia

`multi_style_premia.ipynb`. Carry is omitted: this ETF universe has no futures curve, rate differential, or clean dividend-yield carry. Value is the Asness–Moskowitz–Pedersen proxy, the negative return from \(t-1260\) to \(t-21\). The aligned window is 2012-03-30 to 2024-05-31 (12.2 years). Sharpe needed for \(t = 2\) is **0.58**.

Value correlates −0.20 with TS momentum and −0.47 with 12–1 XS momentum. Pre-scale vol of the three-style blend is 7.9% against a 15.4% sleeve average, so the diversification identity holds. The 5-year reversal Sharpe on this window is **−0.49**. Adding it to TS+XS cuts the blend Sharpe from 0.60 to 0.44 (Jobson–Korkie \(p = 0.39\)). Ken French HML is 0.35 since 1926 and −0.06 over the same window, so the ETF reversal is tracking a bad decade for value, not a broken formula.

On this 12-year window nothing survives Benjamini–Hochberg: TS momentum is the closest (Sharpe 0.62, \(t = 2.32\), BH \(p = 0.061\)). The same engine passes on the 16-year window above.

## Deep momentum

`dmn_sharpe_loss.ipynb`. Train through 2015, test from 2016. Features are lagged 1/5/21/63/126-day returns plus lagged volatility, standardized on the training window only. The loss is the negative Sharpe of the equal-weight portfolio of \(\tanh\) positions. Twelve seeds, 40 epochs, under two return definitions: raw ETF returns, and returns scaled to 15% volatility as in Lim et al.

| Returns | Model | Train Sharpe | Test Sharpe (seed range) | Seed std | Net of 2 bp (mean) | Days long |
|---|---|---|---|---|---|---|
| raw | LSTM (5,153 parameters) | 2.40 to 4.22 | 0.42 to 1.26 | 0.22 | 0.77 | 66% |
| raw | Horizon attention (241 parameters) | −0.40 to 1.04 | −0.72 to 1.01 | 0.55 | 0.50 | 74% |
| vol-scaled | LSTM | 2.49 to 4.59 | 0.60 to 1.49 | 0.28 | 0.92 | 63% |
| vol-scaled | Horizon attention | 0.60 to 1.01 | 0.69 to 0.92 | 0.06 | 0.65 | 89% |

Same test window: equal-weight long 0.73, inverse-vol long 0.84, EWMA TSMOM 0.63, SPY 0.81.

- The attention model's −0.72 to 1.01 range was the previous headline for seed variance. Under the paper's vol-scaled returns it shrinks to 0.69 to 0.92, and the model becomes a long book: long 89% of days, ensemble correlation 0.89 with inverse-vol long, the same Sharpe (0.84).
- The LSTM stays seed-sensitive under both definitions.
- The vol-scaled LSTM seed ensemble has a test Sharpe of 1.14 and a correlation of 0.32 with the long book. That is not distinguishable from the long book's 0.84 (Jobson–Korkie \(p = 0.46\)); net of 2 bp it is 0.94 (\(p = 0.80\)). Break-even cost is about 12 bp.
- The LSTM's in-sample Sharpe of 2.4 to 4.6 is the size of the overfit, not a strategy. There is no validation set, early stopping or hyperparameter search. A full temporal fusion transformer is not in this repo.

## Limits that apply to everything above

- Parameters (32/96 spans, 126-day lookback, top/bottom 3, CUSUM threshold, the 2016 split) were chosen with this sample in view. None of the \(p\)-values corrects for that.
- No risk-free rate is deducted. That flatters SPY and any net-long book relative to the dollar-neutral ones.
- Turnover ignores weight drift between rebalances, and costs are a flat rate per unit traded with no impact or borrow fee.

## Left out on purpose

- **Carry.** Needs FX or a futures curve, not another sleeve on these 10 ETFs.
- **Stock-level Ehsani–Linnainmaa.** Would need name-level betas. The factor-level spanning test is what the Ken French files support.
- **200-year trend-following sample, Martin–Zou skewness proof, Fung–Hsieh lookback straddle.** Cited in the lecture; not replications.
