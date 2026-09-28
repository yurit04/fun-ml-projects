"""Shared data access for the common-quant-projects notebooks.

Sources
-------
* Massive (ex-Polygon) US equities, curated parquet  -> ~/Documents/market_data/massive/curated
    panel/           daily OHLCV + split/dividend-adjusted returns, keyed by security_id (FIGI)
    minute_aggs/     1-minute SIP bars, one file per day
    intraday/        per-day intraday summary features (spreads, volume shares, RV ...)
    news/            news headlines + descriptions with ticker tags
    universes/       point-in-time liquid / cap-ranked universes
* Databento CME Globex (GLBX.MDP3) futures        -> ~/Documents/market_data/Databento/CME-Futures
    OHLCV-1d/1h/1m/1s, MBO (full order-by-order), MBP-10, TBBO
* Public snapshots cached in ./data_cache (FRED Treasury curve, Yahoo option chains / index levels,
  FinBERT scores) so that every notebook re-runs offline and reproducibly.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

MARKET_DATA = Path.home() / "Documents" / "market_data"
MASSIVE = MARKET_DATA / "massive"
CURATED = MASSIVE / "curated"
DATABENTO = MARKET_DATA / "Databento" / "CME-Futures"
HERE = Path(__file__).resolve().parent
CACHE = HERE / "data_cache"
CACHE.mkdir(exist_ok=True)

MONTH_CODES = "FGHJKMNQUVXZ"


# ----------------------------------------------------------------------------- equities
@lru_cache(maxsize=1)
def security_master() -> pd.DataFrame:
    f = next((CURATED / "security_master").glob("asof=*/*.parquet"))
    return pd.read_parquet(f)


def _panel_dataset(name: str = "panel") -> ds.Dataset:
    return ds.dataset(CURATED / name, format="parquet", partitioning="hive")


def load_panel(tickers=None, start=None, end=None, columns=("ticker", "adj_close", "ret", "volume", "close"),
               security_ids=None, dataset="panel") -> pd.DataFrame:
    """Long daily panel (date, security_id, ...). Filter by ticker and/or security_id.

    `ret` is the split- and dividend-adjusted total return; `adj_close` the adjusted price.
    """
    cols = list(dict.fromkeys(["date", "security_id", *columns]))
    flt = None
    if tickers is not None:
        flt = ds.field("ticker").isin(list(tickers))
    if security_ids is not None:
        f2 = ds.field("security_id").isin(list(security_ids))
        flt = f2 if flt is None else (flt & f2)
    if start is not None:
        f3 = ds.field("date") >= pd.Timestamp(start).date()
        flt = f3 if flt is None else (flt & f3)
    if end is not None:
        f4 = ds.field("date") <= pd.Timestamp(end).date()
        flt = f4 if flt is None else (flt & f4)
    t = _panel_dataset(dataset).to_table(columns=cols, filter=flt).to_pandas()
    t["date"] = pd.to_datetime(t["date"])
    return t.sort_values(["date", "security_id"]).reset_index(drop=True)


def wide(panel: pd.DataFrame, value: str, key: str = "ticker") -> pd.DataFrame:
    return panel.pivot_table(index="date", columns=key, values=value, aggfunc="last").sort_index()


def load_minute_bars(tickers, start, end, regular_hours=True) -> pd.DataFrame:
    """1-minute SIP bars from Massive flat files (one parquet per day)."""
    days = pd.bdate_range(start, end)
    out = []
    for d in days:
        f = CURATED / "minute_aggs" / f"year={d.year}" / f"month={d.month:02d}" / f"{d.date()}.parquet"
        if not f.exists():
            continue
        t = ds.dataset(f).to_table(filter=ds.field("ticker").isin(list(tickers)),
                                   columns=["ticker", "ts_ny", "open", "high", "low", "close",
                                            "volume", "transactions"]).to_pandas()
        out.append(t)
    if not out:
        return pd.DataFrame()
    m = pd.concat(out, ignore_index=True)
    m["date"] = m["ts_ny"].dt.normalize()
    m["minute"] = m["ts_ny"].dt.hour * 60 + m["ts_ny"].dt.minute
    if regular_hours:
        m = m[(m["minute"] >= 570) & (m["minute"] < 960)]
    return m.sort_values(["ticker", "ts_ny"]).reset_index(drop=True)


def load_intraday_features(tickers, start, end, columns=None) -> pd.DataFrame:
    flt = ds.field("ticker").isin(list(tickers))
    t = ds.dataset(CURATED / "intraday", format="parquet", partitioning="hive")
    cols = None if columns is None else list(dict.fromkeys(["date", "ticker", *columns]))
    df = t.to_table(filter=flt, columns=cols).to_pandas()
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def load_universe(name="liquid_1500") -> pd.DataFrame:
    df = ds.dataset(CURATED / "universes" / f"name={name}", format="parquet",
                    partitioning="hive").to_table().to_pandas()
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_news(columns=("id", "published_utc", "tickers", "title", "publisher.name")) -> pd.DataFrame:
    fs = sorted((CURATED / "news").glob("year=*/month=*/data.parquet"))
    df = pd.concat([pd.read_parquet(f, columns=list(columns)) for f in fs], ignore_index=True)
    return df.drop_duplicates("id")


# ----------------------------------------------------------------------------- futures
def parse_contract(sym: str):
    """'ESZ4' -> ('ES', 'Z', 4). Returns None for spreads/options."""
    m = re.fullmatch(r"([A-Z0-9]+?)([FGHJKMNQUVXZ])(\d{1,2})", sym)
    return None if m is None else (m.group(1), m.group(2), int(m.group(3)))


@lru_cache(maxsize=1)
def futures_daily() -> pd.DataFrame:
    """All outright GLBX daily bars 2010-06 .. 2025-06 (cached from the DBN file)."""
    f = CACHE / "glbx_ohlcv1d_outrights.parquet"
    if not f.exists():
        import databento as db
        src = next((DATABENTO / "OHLCV-1d").glob("GLBX*/*.ohlcv-1d.dbn"))
        df = db.DBNStore.from_file(src).to_df().reset_index()
        df = df[~df.symbol.str.contains("[- ]", regex=True)]
        df["date"] = df.ts_event.dt.tz_convert(None).dt.normalize()
        df[["symbol", "instrument_id", "open", "high", "low", "close", "volume", "date"]].to_parquet(f)
    df = pd.read_parquet(f)
    p = df["symbol"].map(parse_contract)
    df = df[p.notna()].copy()
    df["root"] = p[p.notna()].str[0]
    return df


def continuous_future(root: str, min_volume=0, fold_sunday=True) -> pd.DataFrame:
    """Back-adjusted (ratio) continuous front contract.

    Each day we hold the most-traded contract, but only ever roll *forward* (never back
    to an earlier expiry). Returns are computed within a contract, so the series is
    free of roll gaps: `ret` is the daily excess return of a fully-collateralised
    futures position, `px` the ratio back-adjusted price.
    """
    df = futures_daily()
    df = df[(df["root"] == root) & (df["volume"] > min_volume)].copy()
    # decode expiry year: single-digit year codes are resolved relative to the bar date
    code = df["symbol"].str[len(root):]
    mon = code.str[0].map(lambda c: MONTH_CODES.index(c) + 1)
    yy = code.str[1:].astype(int)
    base = df["date"].dt.year
    if (code.str.len() == 2).all():
        yr = base - base % 10 + yy
        yr = np.where(yr < base - 1, yr + 10, yr)
    else:
        yr = 2000 + yy
    df["expiry_key"] = np.asarray(yr) * 100 + mon.values
    df = df.sort_values(["date", "expiry_key"])
    wide_close = df.pivot_table(index="date", columns="expiry_key", values="close", aggfunc="last")
    wide_vol = df.pivot_table(index="date", columns="expiry_key", values="volume", aggfunc="sum").fillna(0)
    held, cur = [], None
    for d, row in wide_vol.iterrows():
        best = row.idxmax()
        if cur is None or (best > cur and row[best] > row.get(cur, 0)):
            cur = best
        # the held contract expired / disappeared -> move to the most active later one
        if pd.isna(wide_close.at[d, cur]):
            later = row[row.index > cur]
            cur = later.idxmax() if len(later) else best
        held.append(cur)
    held = pd.Series(held, index=wide_vol.index)
    prev_held = held.shift(1)
    # return of the contract held *yesterday*, measured from yesterday's close to today's
    ret = pd.Series(np.nan, index=held.index)
    for i in range(1, len(held)):
        k = prev_held.iloc[i]
        c0, c1 = wide_close.iloc[i - 1].get(k, np.nan), wide_close.iloc[i].get(k, np.nan)
        ret.iloc[i] = c1 / c0 - 1 if (c0 and not np.isnan(c0) and not np.isnan(c1)) else np.nan
    out = pd.DataFrame({"contract": held, "close": [wide_close.at[d, k] for d, k in held.items()],
                        "volume": [wide_vol.at[d, k] for d, k in held.items()], "ret": ret})
    if fold_sunday:
        # Globex daily bars are UTC days, so the Sunday-evening open gets its own small bar:
        # compound it into Monday so the series is on business days.
        d = out.index + pd.to_timedelta((out.index.dayofweek == 6).astype(int), unit="D")
        out = out.groupby(d).agg(contract=("contract", "last"), close=("close", "last"), volume=("volume", "sum"),
                                 ret=("ret", lambda r: np.prod(1 + r.fillna(0)) - 1 if r.notna().any() else np.nan))
        out.index.name = "date"
    out["px"] = (1 + out["ret"].fillna(0)).cumprod() * out["close"].iloc[0]
    return out


def databento_files(schema: str) -> list[Path]:
    """List raw DBN files of a given schema folder (MBO, MBO-10, TTBO, OHLCV-1m ...)."""
    return sorted((DATABENTO / schema).glob("GLBX*/*.dbn"))


# ----------------------------------------------------------------------------- public snapshots
def fred_rates() -> pd.DataFrame:
    """FRED constant-maturity Treasury yields (percent), DFF and SOFR. Cached snapshot."""
    f = CACHE / "fred_rates.parquet"
    if not f.exists():
        import io
        import requests
        ids = ["DGS1MO", "DGS3MO", "DGS6MO", "DGS1", "DGS2", "DGS3", "DGS5", "DGS7",
               "DGS10", "DGS20", "DGS30", "DFF", "SOFR"]
        parts = []
        for i in ids:
            txt = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={i}", timeout=60).text
            parts.append(pd.read_csv(io.StringIO(txt), index_col=0, parse_dates=True, na_values="."))
        fr = pd.concat(parts, axis=1, sort=True)
        fr.index.name = "date"
        fr[fr.index >= "1990-01-01"].to_parquet(f)
    return pd.read_parquet(f)


def option_chains() -> pd.DataFrame:
    """Yahoo Finance option-chain snapshot (^SPX, SPY, ^VIX), intraday 2026-09-28 (~15:20 ET)."""
    return pd.read_parquet(CACHE / "option_chains_20260928.parquet")


def index_levels() -> pd.DataFrame:
    """Daily closes of ^SPX, SPY, ^VIX, ^VVIX, ^VIX9D, ^VIX3M, ^VIX6M, ^IRX (Yahoo)."""
    return pd.read_parquet(CACHE / "yahoo_index_levels.parquet")


# ----------------------------------------------------------------------------- Databento extracts (cached)
def _symbol_map(job_dir: Path, pattern: str) -> pd.DataFrame:
    """instrument_id -> (symbol, valid date range) for outright contracts matching `pattern` (regex)."""
    import json
    sym = json.load(open(job_dir / "symbology.json"))["result"]
    rows = [(int(iv["s"]), s, pd.Timestamp(iv["d0"]), pd.Timestamp(iv["d1"]))
            for s, v in sym.items() if re.fullmatch(pattern, s) for iv in v]
    return pd.DataFrame(rows, columns=["iid", "symbol", "d0", "d1"])


def _scan_ohlcv1m(pattern: str, start: str | None = None) -> pd.DataFrame:
    """Stream the 1-minute OHLCV DBN file (~28 GB) in chunks, keeping only instruments matching `pattern`."""
    import databento as db
    job = next((DATABENTO / "OHLCV-1m").glob("GLBX*"))
    mp = _symbol_map(job, pattern)
    ids, t0 = set(mp.iid), (pd.Timestamp(start, tz="UTC") if start else None)
    out = []
    for ch in db.DBNStore.from_file(next(job.glob("*.dbn"))).to_df(count=20_000_000, map_symbols=False):
        if t0 is not None and ch.index[-1] < t0:
            continue
        c = ch[ch.instrument_id.isin(ids)]
        if t0 is not None:
            c = c[c.index >= t0]
        out.append(c[["instrument_id", "open", "high", "low", "close", "volume"]])
    df = pd.concat(out).reset_index()
    df = df.merge(mp, left_on="instrument_id", right_on="iid")
    day = df.ts_event.dt.tz_convert(None).dt.normalize()
    return df[(day >= df.d0) & (day < df.d1)].drop(columns=["iid", "d0", "d1"])


def treasury_futures_1m() -> pd.DataFrame:
    """UB / ZB / ZN 1-minute bars from Sep-2021 (Databento OHLCV-1m), cached."""
    f = CACHE / "glbx_ohlcv1m_ub_zb_zn_2021on.parquet"
    if not f.exists():
        _scan_ohlcv1m(r"(UB|ZB|ZN)[FGHJKMNQUVXZ]\d{1,2}", "2021-09-01").to_parquet(f)
    return pd.read_parquet(f)


def es_daily_realized_variance() -> pd.DataFrame:
    """Daily realised variance of 5-minute ES log returns during US hours (09:30–16:00 ET), front contract, cached."""
    f = CACHE / "es_daily_rv5m.parquet"
    if not f.exists():
        df = _scan_ohlcv1m(r"ES[HMUZ]\d{1,2}")
        df["ts_ny"] = df.ts_event.dt.tz_convert("America/New_York").dt.tz_localize(None)
        df["date"] = df.ts_ny.dt.normalize()
        mins = df.ts_ny.dt.hour * 60 + df.ts_ny.dt.minute
        df = df[(mins >= 570) & (mins < 960)]
        front = df.groupby(["date", "symbol"]).volume.sum().reset_index().sort_values("volume").drop_duplicates("date", keep="last")
        df = df.merge(front[["date", "symbol"]], on=["date", "symbol"])
        b = df.set_index("ts_ny").groupby([pd.Grouper(freq="5min"), "date"]).close.last().reset_index().dropna()
        b["r"] = np.log(b.close).groupby(b.date).diff()
        rv = b.groupby("date").agg(rv=("r", lambda x: (x ** 2).sum()), n=("r", "count"), close=("close", "last"))
        rv[rv.n >= 60].to_parquet(f)
    return pd.read_parquet(f)


def mbo_events(date: str = "2025-05-14", symbols=("ESM5", "MESM5")) -> pd.DataFrame:
    """All MBO messages for the given contracts on one Globex session (from the daily DBN file), cached.
    Prices stay in Databento fixed-point (1e-9) units."""
    f = CACHE / f"mbo_{'_'.join(s.lower() for s in symbols)}_{date.replace('-', '')}.parquet"
    if not f.exists():
        import databento as db
        src = next(p for p in (DATABENTO / "MBO").glob(f"GLBX*/glbx-mdp3-{date.replace('-', '')}.mbo.dbn"))
        first = next(db.DBNStore.from_file(src).to_df(count=3_000_000, map_symbols=True))
        ids = first[first.symbol.isin(symbols)].groupby("symbol").instrument_id.first()
        inv = {v: k for k, v in ids.items()}
        out = []
        for ch in db.DBNStore.from_file(src).to_df(count=20_000_000, map_symbols=False, price_type="fixed"):
            c = ch[ch.instrument_id.isin(list(inv))].reset_index()
            out.append(c[["ts_recv", "ts_event", "instrument_id", "action", "side", "price", "size", "order_id", "flags", "sequence"]])
        df = pd.concat(out, ignore_index=True)
        df["symbol"] = df.instrument_id.map(inv)
        df.to_parquet(f)
    return pd.read_parquet(f)
