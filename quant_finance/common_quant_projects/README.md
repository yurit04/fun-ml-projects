# Common Quant Projects

Nine end-to-end research notebooks built from the "don't build the same quant projects everyone has on their résumé" list.
Each notebook has the theory with equations, a from-scratch implementation on **real data** wherever it exists locally, honest out-of-sample
evaluation, and a closing section on **strengths, weaknesses and extensions**. The narrative in each notebook is written from its *actual* results,
including the negative ones.

| # | Notebook | Data | Headline finding |
|---|---|---|---|
| 1 | [Volatility surface & arbitrage detection](01_vol_surface_arbitrage.ipynb) | SPX & SPY option chains (Yahoo snapshot, 2026-09-28) | Parity-implied forwards/rates; SVI, SSVI and eSSVI fits; Durrleman & calendar checks. Almost all "executable arbitrage" is stale quotes, and a freshness filter cuts ~500 flags to 2 one-tick cases. SSVI can't fit short-dated SPX skew even without constraints. |
| 2 | [Yield-curve modelling with ML](02_yield_curve_ml.ipynb) | FRED CMT curve 1994–2026 | Nelson–Siegel vs PCA vs autoencoder (the AE extrapolates *worst* out of sample); GP interpolation with uncertainty bands; the random walk is never significantly beaten by DNS, PCA-VAR, AE, GP or LSTM at 1–3 months. |
| 3 | [News-sentiment factor (FinBERT)](03_sentiment_factor.ipynb) | Massive news (750 k articles), liquid-1500 universe | Point-in-time FinBERT pipeline. In liquid stocks sentiment has no reliable predictive power: news *follows* prices, and the signal is a disguised momentum/reversal bet. Post-news drift survives only in smaller, less liquid stocks. |
| 4 | [HMM regime-switching portfolio](04_hmm_regime_portfolio.ipynb) | Databento CME futures (ES, NQ, ZN, ZB, GC, 6J) 2010–2025 | Filtered vs smoothed look-ahead trap; walk-forward HMM with regime-dependent optimisers. It helped in 2020, hurt in the 2021–22 bond sell-off, and ties static risk parity overall. |
| 5 | [Liquidity risk in ETF–bond arbitrage (TLT)](05_etf_bond_liquidity_arb.ipynb) | Massive TLT (daily, 1-min, spreads), Databento UB/ZB 1-min, FRED | TLT as a 20–30y ladder (duration, low-coupon effect); futures-implied fair value at synchronised timestamps; dislocations vs liquidity. The intraday stat-arb edge (~1–2 bp) is smaller than costs (~3.5 bp). |
| 6 | [CDS bootstrapping & synthetic CDOs](06_cds_bootstrap_cdo.ipynb) | FRED discount curve + *illustrative* CDS quotes | Hazard-rate bootstrap, upfront/coupon, CS01/bucketed/recovery/JTD, arbitrage-inconsistent curves; Gaussian-copula tranches (LHP, exact recursion, MC). A systemic-state correlation model generates the observed base-correlation skew; a t-copula doesn't. |
| 7 | [Intraday volume profiles & VWAP](07_intraday_volume_profile.ipynb) | Massive 1-min bars, 32 liquid names, 2022–2026 | GMM description of the intraday density; clustering finds FOMC, expiry/rebalance and data-artefact day types; profile schedules beat TWAP by ~⅓ in mean slippage. The dynamic update lowers typical error but fattens tails. |
| 8 | [Vol-of-vol: SABR, Bergomi, rough vol](08_vol_of_vol.ipynb) | SPX/VIX option chains, VIX/VVIX history, Databento ES 1-min | H ≈ 0.16 from both realised vol and the SPX skew power law; SABR ν term structure; exact VIX-option pricing under 1F and rough Bergomi, whose near-flat VIX smiles miss the steep market skew (the SPX/VIX joint-calibration puzzle) and understate VIX-spike probabilities. |
| 9 | [Limit order book simulation](09_lob_simulation.ipynb) | Databento MBO for ES & MES, 2025-05-14 | Numba book reconstruction from 20 M messages; CST queueing model calibrated and simulated (right shape, wrong dynamics); OFI explains 64–75 % of price changes; metaorder impact in the ZI model is linear and permanent; ES→MES latency-arbitrage windows last a median ≈ 0.3 ms. |

## Data sources

| Source | Location | Used in |
|---|---|---|
| Massive (US equities: daily panel, 1-min SIP bars, intraday features, news, universes) | `~/Documents/market_data/massive/curated` | 3, 5, 7 |
| Databento CME Globex (OHLCV-1d/1m, MBO) | `~/Documents/market_data/Databento/CME-Futures` | 4, 5, 8, 9 |
| FRED Treasury CMT yields, SOFR | cached `data_cache/fred_rates.parquet` | 2, 5, 6, 8 |
| Yahoo option chains (^SPX, SPY, ^VIX) and index levels (VIX, VVIX, …) | cached `data_cache/option_chains_20260928.parquet`, `yahoo_index_levels.parquet` | 1, 8 |

Neither local data set contains options, Treasury yields, CDS or tranche quotes, so those come from the public snapshots above (options, rates). For CDS/CDO,
the quotes are clearly labelled illustrative. The option snapshot is a single point in time and can't be re-downloaded, which is why it's cached.

## Layout

```
common_quant_projects/
├── 0X_*.ipynb                  executed notebooks (outputs included)
├── src/0X_*.py                 the same notebooks in percent-format (diff-friendly source)
├── qdata.py                    shared data access: Massive loaders, continuous futures, Databento extracts (cached), public snapshots
├── lob_tools.py                numba limit-order-book reconstruction from MBO messages
├── scripts/score_news_finbert.py   one-off FinBERT scoring of all news (≈1–2 h on Apple-silicon GPU)
└── data_cache/                 cached extracts and snapshots (large Databento/Massive extracts are git-ignored)
```

## Running

Use the repo's virtual environment (`uv pip install -e ".[notebook]"` at the repo root installs everything, including `pyarrow`, `hmmlearn`,
`databento`, `transformers` and `numba`). Then, from this folder:

```bash
python scripts/score_news_finbert.py            # only needed once, for notebook 3
jupyter nbconvert --to notebook --execute --inplace 0X_name.ipynb
```

The first run of notebooks 4, 5, 7, 8 and 9 builds their caches from the raw Databento/Massive files (seconds to ~2 minutes each). After that every
notebook runs in 1–5 minutes, except notebook 2 (~15 minutes for the annual LSTM/GP re-fits).
