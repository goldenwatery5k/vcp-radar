"""Monthly momentum leaderboard: risk-adjusted 12-1 momentum among the 1,000 most-traded US stocks.

Rule (same as backtest research_c.py "risk-adj mom (12-1/vol)", N=20):
  score = (close[t-21] / close[t-252] - 1) / (60d daily-return stdev * sqrt(252))
  universe = top 1000 by 50d $ volume, price >= $5, close > 200d average, 60d vol >= 12%
  signal date = last trading day before the month's first trading day; model portfolio = top 20, equal weight.
Run:  python momentum.py            (downloads ~5,000 tickers, a few minutes)
      python momentum.py --cache    (reuse data/bt_prices_all.pkl, for local tests)
Out:  data/momentum.json
"""
import io
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import requests
import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "momentum.json")
H = {"User-Agent": "Mozilla/5.0"}
TOP_N, SHOW = 20, 60


def listed_us():
    bad = re.compile(r"warrant|right|unit|preferred|note|debenture|trust|fund|etf|acquisition|depositary receipt|%", re.I)
    out = {}
    for url, sym in (("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt", "Symbol"),
                     ("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt", "ACT Symbol")):
        df = pd.read_csv(io.StringIO(requests.get(url, headers=H, timeout=30).text), sep="|", dtype=str)
        df = df[df[sym].notna() & ~df[sym].str.contains("File Creation", na=False)]
        df = df[(df["ETF"] != "Y") & (df["Test Issue"] != "Y")]
        for t, nm in zip(df[sym], df["Security Name"]):
            if re.fullmatch(r"[A-Z]{1,5}", t) and not bad.search(nm or ""):
                nm = re.split(r" - | Common Stock| Ordinary Shares| Class [A-C]| American Depositary", nm or "")[0].strip()
                out[t] = nm
    return out


def download(tickers):
    out = {}
    n = 100
    for i in range(0, len(tickers), n):
        ch = tickers[i:i + n]
        try:
            df = yf.download(ch, period="2y", interval="1d", auto_adjust=True, progress=False,
                             group_by="ticker", threads=True)
        except Exception as e:
            print("chunk failed", e)
            continue
        for t in ch:
            try:
                d = df[t].dropna(subset=["Close"])
                if len(d) > 260:
                    out[t] = d[["Close", "Volume"]].astype("float32")
            except Exception:
                pass
        if (i // n) % 10 == 0:
            print(f"  {min(i + n, len(tickers))}/{len(tickers)} ({len(out)} ok)", flush=True)
    return out


def score_at(C, V, i):
    """series of risk-adjusted momentum for all tickers using data through row i (inclusive)."""
    c = C.iloc[:i + 1]
    mom = c.iloc[-22] / c.iloc[-253] - 1
    vol = c.pct_change().iloc[-60:].std() * np.sqrt(252)
    dv = (c * V.iloc[:i + 1]).iloc[-50:].mean()
    s200 = c.iloc[-200:].mean()
    ok = (dv.rank(ascending=False) <= 1000) & (c.iloc[-1] >= 5) & (c.iloc[-1] > s200) & mom.notna() & (vol >= 0.12)   # <12% vol = price pinned (e.g. cash takeover), not momentum
    sc = (mom / vol).where(ok)
    return sc, mom, vol, dv, s200


def main():
    t0 = time.time()
    names = listed_us()
    if "--cache" in sys.argv:
        raw = pd.read_pickle(os.path.join(HERE, "data", "bt_prices_all.pkl"))
        px = {t: d[["Close", "Volume"]] for t, d in raw.items() if t in names}
    else:
        px = download(sorted(names))
    print(f"{len(px)} series ({time.time()-t0:.0f}s)", flush=True)
    C = pd.DataFrame({t: d["Close"] for t, d in px.items()}).sort_index().ffill(limit=5)
    V = pd.DataFrame({t: d["Volume"] for t, d in px.items()}).reindex(C.index).fillna(0)
    last = len(C) - 1
    months = C.index.to_period("M")
    cur_m = months[-1]
    first_i = int(np.argmax(months == cur_m))          # first trading day of current month
    if first_i == 0:
        raise SystemExit("not enough history")
    i_sig = first_i - 1
    # previous month's signal day
    prev_m = months[first_i - 1]
    prev_first = int(np.argmax(months == prev_m))
    i_prev = prev_first - 1 if prev_first > 0 else None
    sc, mom, vol, dv, s200 = score_at(C, V, i_sig)
    ranked = sc.dropna().sort_values(ascending=False)
    prev_hold = []
    if i_prev is not None and i_prev >= 252:
        psc = score_at(C, V, i_prev)[0].dropna().sort_values(ascending=False)
        prev_hold = list(psc.index[:TOP_N])
    hi52 = C.iloc[max(0, i_sig - 251):i_sig + 1].max()
    c_sig = C.iloc[i_sig]; c_now = C.iloc[last]; c_reb = C.iloc[first_i]
    c3 = C.iloc[i_sig] / C.iloc[i_sig - 63] - 1
    rows = []
    for rank, (t, v) in enumerate(ranked.head(SHOW).items(), 1):
        rows.append(dict(rank=rank, ticker=t, name=names.get(t, ""), score=round(float(v), 2),
                         mom=round(float(mom[t]) * 100, 1), vol=round(float(vol[t]) * 100, 1),
                         r3m=round(float(c3[t]) * 100, 1), from_hi=round(float(c_sig[t] / hi52[t] - 1) * 100, 1),
                         price=round(float(c_sig[t]), 2), now=round(float(c_now[t]), 2),
                         since=round(float(c_now[t] / c_reb[t] - 1) * 100, 1),
                         dollar_vol=round(float(dv[t]) / 1e6, 1), new=bool(prev_hold and t not in prev_hold)))
    dropped = [t for t in prev_hold if t not in list(ranked.index[:TOP_N])]
    out = dict(
        signal_date=str(C.index[i_sig].date()), rebalance_date=str(C.index[first_i].date()),
        price_date=str(C.index[last].date()), universe=int(sc.notna().sum()), top_n=TOP_N,
        dropped=[dict(ticker=t, name=names.get(t, ""), rank=int(list(ranked.index).index(t) + 1) if t in ranked.index else None) for t in dropped],
        rows=rows,
        generated=pd.Timestamp.now(tz="Asia/Hong_Kong").strftime("%Y-%m-%d %H:%M") + " HKT")
    json.dump(out, open(OUT, "w"), ensure_ascii=False)
    print(f"signal {out['signal_date']}  rebalance {out['rebalance_date']}  universe {out['universe']}  "
          f"top: {[r['ticker'] for r in rows[:10]]}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
