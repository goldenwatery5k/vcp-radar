"""Backtest of the Stage 2 + VCP breakout system (same rules as screener.py).

Run:  .venv/bin/python backtest.py [start_year]      (default 2018)
Out:  out/backtest.json  out/backtest.html

Design (daily bars only, so every fill assumption is explicit):
- Universe  = today's constituents (survivorship bias! results are optimistic).
- Liquidity = point-in-time: close >= $20 and 50d avg $ volume >= $15M.
- Signal    = day i closes above the pivot of a base that ended on day i-1,
              all 8 trend-template criteria true, not Stage 3, volume >= 1.5x 50d avg,
              close <= pivot +5% (no chasing).
- Entry     = that day's close (+0.1% slippage)  [= "method 2": confirm into the close].
- Stop      = handle low -1% (min 5% away; if > 10% try 10/20DMA -1%; else skip).
- Exit      = live rules: early stage 2 (<=1 base) -> sell into weakness (close < 20DMA once
              20DMA is above the initial stop), 2R -> stop to +2%;
              late stage 2 (>=2 bases) -> sell half at 2R, rest at 3R, stop to +2% after 2R.
              Variant "strength": everyone uses 2R half / 3R rest.
- Worst-case ordering: if a bar touches both stop and target, the stop fills first.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import yfinance as yf

import screener as S
import universe

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
CACHE = os.path.join(HERE, "data", "bt_prices_all.pkl")
EARN_CACHE = os.path.join(HERE, "data", "earnings_cache.json")
START_YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else 2018
SLIP = 0.001
ENTRY = os.environ.get("ENTRY", "close")   # close = method 2 (confirm into close); pivot = method 1 (buy-stop at pivot)
E, B, T = S.E, S.B, S.T


# ----------------------------------------------------------------- data
def listed_us():
    import requests, io, re
    H = {"User-Agent": "Mozilla/5.0"}
    out = set()
    bad = re.compile(r"warrant|right|unit|preferred|note|debenture|trust|fund|etf|acquisition|depositary receipt|%", re.I)
    for url, sym, etf, test, name in (
            ("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt", "Symbol", "ETF", "Test Issue", "Security Name"),
            ("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt", "ACT Symbol", "ETF", "Test Issue", "Security Name")):
        df = pd.read_csv(io.StringIO(requests.get(url, headers=H, timeout=30).text), sep="|", dtype=str)
        df = df[df[sym].notna() & ~df[sym].str.contains("File Creation", na=False)]
        df = df[(df[etf] != "Y") & (df[test] != "Y")]
        for t, nm in zip(df[sym], df[name]):
            if not re.fullmatch(r"[A-Z]{1,5}", t) or bad.search(nm or ""):
                continue
            out.add(t)
    return sorted(out)


def load_prices():
    if os.path.exists(CACHE) and time.time() - os.path.getmtime(CACHE) < 3 * 86400:
        return pd.read_pickle(CACHE)
    base = [u["ticker"] for u in universe.load() if u["market"] == "US"]
    tick = sorted(set(listed_us()) | set(base) | {"SPY", "QQQ", "IWM"})
    print(f"downloading {len(tick)} tickers")
    out = {}
    n = 100
    for i in range(0, len(tick), n):
        ch = tick[i:i + n]
        try:
            df = yf.download(ch, start=f"{START_YEAR - 2}-01-01", interval="1d", auto_adjust=True,
                             progress=False, group_by="ticker", threads=True)
        except Exception as e:
            print("chunk failed", e)
            continue
        for t in ch:
            try:
                d = df[t].dropna(subset=["Close"])
                if len(d) > 260 and float(d["Close"].iloc[-1]) >= 5 and float((d["Close"] * d["Volume"]).tail(50).mean()) >= 2e6:
                    out[t] = d[["Open", "High", "Low", "Close", "Volume"]].astype("float32")
            except Exception:
                pass
        print(f"  downloaded {min(i + n, len(tick))}/{len(tick)} ({len(out)} kept)", flush=True)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    pd.to_pickle(out, CACHE)
    return out


def rolling(df, n):
    return df.rolling(n, min_periods=n).mean()


# ----------------------------------------------------------------- signals
def build_signals(px, tick, min_price, min_dv):
    idx = px["SPY"].index
    def mat(col):
        return pd.DataFrame({t: px[t][col] for t in tick}).reindex(idx)
    C, H, L, O, V = (mat(c) for c in ("Close", "High", "Low", "Open", "Volume"))
    C = C.ffill(limit=5); H = H.where(H.notna(), C); L = L.where(L.notna(), C); O = O.where(O.notna(), C)
    V = V.fillna(0)
    s10, s20, s50, s150, s200 = (rolling(C, n) for n in (10, 20, 50, 150, 200))
    hi52 = H.rolling(252, min_periods=200).max(); lo52 = L.rolling(252, min_periods=200).min()
    dv50 = rolling(C * V, 50)
    liquid = (C >= min_price) & (dv50 >= min_dv)
    r = lambda nb: C / C.shift(nb) - 1
    raw = 0.4 * r(63) + 0.2 * r(126) + 0.2 * r(189) + 0.2 * r(251)
    raw = raw.where(liquid)
    rs = raw.rank(axis=1, pct=True) * 98 + 1
    tt = ((C > s150) & (C > s200) & (s150 > s200) & (s200 > s200.shift(T["ma200_up_days"]))
          & (s50 > s150) & (s50 > s200) & (C >= hi52 * (1 - T["within_52w_high"]))
          & (C >= lo52 * (1 + T["above_52w_low"])) & (rs >= T["min_rs"]) & (C > s50))
    stage3 = (s150 / s150.shift(21) - 1 < T.get("stage3_flat_150_pct", 0.01)) & (hi52 / lo52 - 1 >= T.get("stage3_min_run", 0.30))
    volr = V / V.shift(1).rolling(50, min_periods=50).mean()
    ev = liquid & tt & ~stage3 & (volr >= E["breakout_vol_ratio"]) & (C > C.shift(1))
    # regime points (same scoring as the live page)
    nliq = liquid.sum(axis=1).clip(lower=1)
    b50 = ((C > s50) & liquid).sum(axis=1) / nliq
    s2p = ((tt | ((C > s50) & False)) & liquid).sum(axis=1) / nliq
    nh = ((C >= hi52 * 0.98) & liquid).sum(axis=1); nl = ((C <= lo52 * 1.02) & liquid).sum(axis=1)
    pts = pd.Series(0, index=idx, dtype=float)
    for k in ("SPY", "QQQ", "IWM"):
        c = px[k]["Close"].reindex(idx).ffill()
        pts += (c > c.rolling(50).mean()).astype(float)
        if k != "IWM":
            pts += (c > c.rolling(200).mean()).astype(float)
    pts += np.where(b50 >= 0.5, 2, np.where(b50 >= 0.35, 1, 0))
    pts += (s2p >= 0.15).astype(float) + (nh > nl).astype(float)
    return dict(idx=idx, tick=tick, C=C, H=H, L=L, O=O, V=V, s10=s10, s20=s20, ev=ev, rs=rs, regime=pts,
                spy=px["SPY"]["Close"].reindex(idx).ffill())


def count_bases(h, l):
    zz = S.zigzag(h[-252:], l[-252:], 0.10)
    mx = 0; nb = 0
    for p in zz:
        if p[2] != "P":
            continue
        if p[1] >= mx * 0.995:
            nb += 1
        mx = max(mx, p[1])
    return max(0, nb - 1)


def find_trades(sig, first_i):
    C, H, L, V = (sig[k].to_numpy(float) for k in ("C", "H", "L", "V"))
    s10, s20 = sig["s10"].to_numpy(float), sig["s20"].to_numpy(float)
    ev = sig["ev"].to_numpy(bool); rs = sig["rs"].to_numpy(float)
    trades = []
    ii, jj = np.nonzero(ev)
    for i, j in zip(ii, jj):
        if i < first_i or i < 260:
            continue
        c_, h_, l_, v_ = C[:i + 1, j], H[:i + 1, j], L[:i + 1, j], V[:i + 1, j]
        f = int(np.argmax(~np.isnan(c_)))
        c_, h_, l_, v_ = c_[f:], h_[f:], l_[f:], v_[f:]
        e = len(c_) - 2
        base = S.base_at(h_, l_, c_, v_, e)
        if not base:
            continue
        piv = base["pivot"]
        if not (c_[-1] > piv and c_[-2] <= piv * 1.0005):
            continue
        if c_[-1] > piv * (1 + E["chase_limit_pct"]):
            continue
        if ENTRY == "pivot":
            o_ = float(sig["O"].to_numpy(float)[i, j])
            entry = max(o_, piv * 1.0015) * (1 + SLIP)
        else:
            entry = float(c_[-1]) * (1 + SLIP)
        stop = base["handle_low"] * (1 - E["stop_below_handle_low"])
        risk = (entry - stop) / entry
        if risk < E["min_stop_pct"]:
            stop = entry * (1 - E["min_stop_pct"]); risk = E["min_stop_pct"]
        elif risk > E["max_stop_pct"]:
            for arr in (s10, s20):
                st = arr[i, j] * (1 - E["stop_below_handle_low"])
                rp = (entry - st) / entry
                if st < entry and E["min_stop_pct"] <= rp <= E["max_stop_pct"]:
                    stop, risk = st, rp
                    break
            else:
                continue
        trades.append(dict(j=int(j), i=int(i), ticker=sig["tick"][j], entry=entry, stop=float(stop), risk=float(risk),
                           rs=float(rs[i, j]), bases=count_bases(h_, l_), regime=float(sig["regime"].iloc[i]),
                           dec=bool(base["decreasing"]), vol_dry=base["vol_dry"], hd=base["handle_depth"], pa=base["prior_adv"],
                           ext=float(c_[-1] / piv - 1), vr=float(v_[-1] / max(1.0, np.mean(v_[-51:-1]))),
                           date=sig["idx"][i]))
    return trades


# ----------------------------------------------------------------- exits
def simulate(tr, sig, policy):
    """returns dict(exit_i, R, path{day_i: pnl_R}) ; daily path of cumulative R"""
    O, H, L, C = (sig[k].to_numpy(float) for k in ("O", "H", "L", "C"))
    s20 = sig["s20"].to_numpy(float)
    j, i0 = tr["j"], tr["i"]
    entry, stop0 = tr["entry"], tr["stop"]
    rk = entry - stop0
    t2, t3 = entry + 2 * rk, entry + 3 * rk
    weak = (policy == "mixed") and tr["bases"] <= 1
    stop = stop0
    rem, realized = 1.0, 0.0
    half = False
    if L[i0, j] <= stop0 and ENTRY == "pivot":
        r_ = (stop0 * (1 - SLIP) - entry) / rk
        return dict(exit_i=i0, R=r_, path={i0: r_})
    path = {i0: (C[i0, j] - entry) / rk}
    n = C.shape[0]
    for d in range(i0 + 1, n):
        o, h, l, c = O[d, j], H[d, j], L[d, j], C[d, j]
        if np.isnan(c):
            continue
        if o <= stop:
            px_ = o * (1 - SLIP); realized += rem * (px_ - entry) / rk; rem = 0
        elif l <= stop:
            px_ = stop * (1 - SLIP); realized += rem * (px_ - entry) / rk; rem = 0
        if rem == 0:
            path[d] = realized
            return dict(exit_i=d, R=realized, path=path)
        if not half and h >= t2:
            half = True
            stop = max(stop, entry * 1.02)
            if not weak:
                px_ = max(o, t2) * (1 - SLIP); realized += 0.5 * (px_ - entry) / rk; rem = 0.5
        if (not weak) and half and h >= t3 and rem > 0:
            px_ = max(o, t3) * (1 - SLIP); realized += rem * (px_ - entry) / rk; rem = 0
            path[d] = realized
            return dict(exit_i=d, R=realized, path=path)
        if weak and s20[d, j] > stop0 and c < s20[d, j]:
            px_ = c * (1 - SLIP); realized += rem * (px_ - entry) / rk; rem = 0
            path[d] = realized
            return dict(exit_i=d, R=realized, path=path)
        path[d] = realized + rem * (c - entry) / rk
    return dict(exit_i=None, R=path[n - 1], path=path)


# ----------------------------------------------------------------- portfolio
def portfolio(trades, sims, sig, mode, first_i, equity0=100000.0, skip=frozenset()):
    n = len(sig["idx"])
    by_day = {}
    for k, t in enumerate(trades):
        by_day.setdefault(t["i"], []).append(k)
    open_pos = {}
    cash_realized = equity0
    eq_series = np.full(n, np.nan)
    closed = []
    taken = []
    pnl_open = {}

    def mtm(d):
        return cash_realized + sum(p["rd"] * sims[k]["path"].get(d, sims[k]["path"][max(sims[k]["path"])]) for k, p in open_pos.items())

    for d in range(first_i, n):
        # close positions that exited today
        for k in list(open_pos):
            s = sims[k]
            if s["exit_i"] is not None and s["exit_i"] <= d:
                cash_realized += open_pos[k]["rd"] * s["R"]
                closed.append(k)
                del open_pos[k]
        equity = mtm(d)
        eq_series[d] = equity
        reg = trades[0]["regime"] if False else float(sig["regime"].iloc[d])
        if mode == "all":
            scale, max_r = 1.0, 10
        elif mode == "green":
            if reg < 7: continue
            scale, max_r = 1.0, 10
        elif mode == "notred":
            if reg < 4: continue
            scale, max_r = 1.0, 10
        else:  # advice
            scale, max_r = (1.0, 10) if reg >= 7 else ((1.0, 6) if reg >= 4 else (0.5, 3))
        cands = sorted((k for k in by_day.get(d, []) if k not in skip), key=lambda k: -trades[k]["rs"])
        for k in cands:
            if len(open_pos) * 1.0 >= max_r or sims[k] is None:
                break
            rd = equity * 0.01 * scale
            expo = rd / trades[k]["risk"]
            expo = min(expo, equity * 0.25)
            rd = expo * trades[k]["risk"]
            used = sum(p["expo"] for p in open_pos.values())
            if used + expo > equity:
                expo = max(0.0, equity - used)
                rd = expo * trades[k]["risk"]
                if expo < 0.4 * min(equity * 0.01 * scale / trades[k]["risk"], equity * 0.25):
                    continue
            open_pos[k] = dict(rd=rd, expo=expo)
            taken.append(k)
    for k in list(open_pos):
        cash_realized += open_pos[k]["rd"] * sims[k]["R"]
    eq = pd.Series(eq_series, index=sig["idx"]).ffill()
    return eq, taken


def stats(eq, trades, sims, taken, spy):
    e = eq.dropna()
    e = e[e.index >= e.index[0]]
    yrs = (e.index[-1] - e.index[0]).days / 365.25
    cagr = (e.iloc[-1] / e.iloc[0]) ** (1 / yrs) - 1
    dd = (e / e.cummax() - 1)
    ret = e.pct_change().dropna()
    sharpe = float(ret.mean() / ret.std() * np.sqrt(252)) if ret.std() > 0 else 0
    R = np.array([sims[k]["R"] for k in taken]) if taken else np.array([0.0])
    wins, loss = R[R > 0], R[R <= 0]
    year = e.groupby(e.index.year).apply(lambda s: s.iloc[-1] / s.iloc[0] - 1)
    # yearly using prior year-end as base
    ye = e.groupby(e.index.year).last()
    yrets = {}
    prev = e.iloc[0]
    for y, v in ye.items():
        yrets[int(y)] = round(float(v / prev - 1) * 100, 1); prev = v
    return dict(trades=int(len(R)), win_rate=round(float((R > 0).mean()) * 100, 1),
                avg_win_R=round(float(wins.mean()) if len(wins) else 0, 2),
                avg_loss_R=round(float(loss.mean()) if len(loss) else 0, 2),
                expectancy_R=round(float(R.mean()), 2),
                profit_factor=round(float(wins.sum() / -loss.sum()) if loss.sum() < 0 else 99, 2),
                cagr=round(float(cagr) * 100, 1), maxdd=round(float(dd.min()) * 100, 1),
                sharpe=round(sharpe, 2), final=round(float(e.iloc[-1]), 0), years=yrets)


def fetch_earnings(tickers):
    from concurrent.futures import ThreadPoolExecutor
    cache = json.load(open(EARN_CACHE)) if os.path.exists(EARN_CACHE) else {}
    todo = [t for t in tickers if t not in cache]
    print(f"earnings dates: {len(todo)} to fetch ({len(cache)} cached)", flush=True)
    def one(t):
        for _ in range(3):
            try:
                time.sleep(0.4)
                d = yf.Ticker(t).get_earnings_dates(limit=100)
                if d is None or len(d) == 0:
                    time.sleep(3)
                    continue
                return t, sorted({str(x.date()) for x in d.index})
            except Exception:
                time.sleep(3)
        return t, None
    with ThreadPoolExecutor(2) as ex:
        for k, (t, r) in enumerate(ex.map(one, todo), 1):
            if r is not None:
                cache[t] = r
            if k % 100 == 0:
                print(f"  {k}/{len(todo)}", flush=True)
                json.dump(cache, open(EARN_CACHE, "w"))
    json.dump(cache, open(EARN_CACHE, "w"))
    return cache


def main():
    t0 = time.time()
    px = load_prices()
    print(f"prices: {len(px)} series ({time.time()-t0:.0f}s)", flush=True)
    idx_tick = {u["ticker"] for u in universe.load() if u["market"] == "US"}
    UNIS = {"index": ([t for t in px if t in idx_tick], 20, 15e6),
            "expanded": ([t for t in px if t not in ("SPY", "QQQ", "IWM")], 10, 5e6)}
    res, curves, regs, tr_out = {}, {}, {}, {}
    spy_stat = None
    for uname, (tick, mp, mdv) in UNIS.items():
        sig = build_signals(px, tick, mp, mdv)
        first_i = int(np.argmax(sig["idx"] >= pd.Timestamp(f"{START_YEAR}-01-01")))
        trades = find_trades(sig, first_i)
        print(f"[{uname}] events {int(sig['ev'].to_numpy().sum())} trades {len(trades)} ({time.time()-t0:.0f}s)", flush=True)
        earn = fetch_earnings(sorted({t["ticker"] for t in trades}))
        skip = set()
        for k, t in enumerate(trades):
            ds = earn.get(t["ticker"]) or []
            d0 = t["date"].date()
            for x in ds:
                dd = (pd.Timestamp(x).date() - d0).days
                if 0 <= dd <= 7:
                    skip.add(k); break
        print(f"[{uname}] earnings within 7d of entry: {len(skip)} of {len(trades)}", flush=True)
        spyc = sig["spy"][sig["spy"].index >= sig["idx"][first_i]]
        for policy in ("mixed", "strength"):
            sims = [simulate(t, sig, policy) for t in trades]
            for en, sk in (("noearn", frozenset(skip)), ("earn", frozenset())):
                for mode in ("all", "notred", "green", "advice"):
                    eq, taken = portfolio(trades, sims, sig, mode, first_i, skip=sk)
                    key = f"{uname}/{en}/{policy}/{mode}"
                    res[key] = stats(eq, trades, sims, taken, spyc)
                    if policy == "mixed":
                        curves[key] = [round(float(v), 0) for v in eq.iloc[first_i::5].fillna(100000).tolist()]
                    print(key, {k: v for k, v in res[key].items() if k != "years"}, flush=True)
        # per-trade expectancy (mixed), by regime and earnings flag
        sims = [simulate(t, sig, "mixed") for t in trades]
        reg = {}
        for lbl, lo, hi in (("green", 7, 99), ("yellow", 4, 6.99), ("red", -1, 3.99)):
            R = np.array([sims[k]["R"] for k, t in enumerate(trades) if lo <= t["regime"] <= hi])
            if len(R):
                reg[lbl] = dict(n=len(R), win=round(float((R > 0).mean()) * 100, 1), exp=round(float(R.mean()), 2))
        Re = np.array([sims[k]["R"] for k in range(len(trades)) if k in skip]); Rn = np.array([sims[k]["R"] for k in range(len(trades)) if k not in skip])
        reg["earnings_near"] = dict(n=int(len(Re)), win=round(float((Re > 0).mean()) * 100, 1) if len(Re) else 0, exp=round(float(Re.mean()), 2) if len(Re) else 0)
        reg["earnings_far"] = dict(n=int(len(Rn)), win=round(float((Rn > 0).mean()) * 100, 1), exp=round(float(Rn.mean()), 2))
        regs[uname] = reg
        tr_out[uname] = dict(n=len(trades), n_universe=len(tick))
        if spy_stat is None:
            ye = spyc.groupby(spyc.index.year).last(); prev = spyc.iloc[0]; yy = {}
            for y, v in ye.items():
                yy[int(y)] = round(float(v / prev - 1) * 100, 1); prev = v
            spy_stat = dict(cagr=round(float((spyc.iloc[-1] / spyc.iloc[0]) ** (365.25 / ((spyc.index[-1] - spyc.index[0]).days)) - 1) * 100, 1),
                            maxdd=round(float((spyc / spyc.cummax() - 1).min()) * 100, 1), years=yy)
            dates = [str(d.date()) for d in sig["idx"][first_i::5]]
            spy_curve = [round(float(v / spyc.iloc[0] * 100000), 0) for v in spyc.iloc[::5].tolist()]
        del sig
    out = dict(start=dates[0], end=dates[-1], entry=ENTRY, results=res, curves=curves, regime=regs, unis=tr_out,
               spy=spy_stat, dates=dates, spy_curve=spy_curve)
    json.dump(out, open(os.path.join(OUT, f"backtest2_{ENTRY}.json"), "w"), ensure_ascii=False)
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
