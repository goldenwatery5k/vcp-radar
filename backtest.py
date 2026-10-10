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
CACHE = os.path.join(HERE, "data", "bt_prices.pkl")
START_YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else 2018
SLIP = 0.001
ENTRY = os.environ.get("ENTRY", "close")   # close = method 2 (confirm into close); pivot = method 1 (buy-stop at pivot)
E, B, T = S.E, S.B, S.T


# ----------------------------------------------------------------- data
def load_prices():
    if os.path.exists(CACHE) and time.time() - os.path.getmtime(CACHE) < 3 * 86400:
        return pd.read_pickle(CACHE)
    uni = [u["ticker"] for u in universe.load() if u["market"] == "US"]
    tick = sorted(set(uni) | {"SPY", "QQQ", "IWM"})
    out = {}
    n = 80
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
                if len(d) > 260:
                    out[t] = d[["Open", "High", "Low", "Close", "Volume"]]
            except Exception:
                pass
        print(f"  downloaded {min(i + n, len(tick))}/{len(tick)} ({len(out)} ok)", flush=True)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    pd.to_pickle(out, CACHE)
    return out


def rolling(df, n):
    return df.rolling(n, min_periods=n).mean()


# ----------------------------------------------------------------- signals
def build_signals(px):
    idx = px["SPY"].index
    tick = [t for t in px if t not in ("SPY", "QQQ", "IWM")]
    def mat(col):
        return pd.DataFrame({t: px[t][col] for t in tick}).reindex(idx)
    C, H, L, O, V = (mat(c) for c in ("Close", "High", "Low", "Open", "Volume"))
    C = C.ffill(limit=5); H = H.where(H.notna(), C); L = L.where(L.notna(), C); O = O.where(O.notna(), C)
    V = V.fillna(0)
    s10, s20, s50, s150, s200 = (rolling(C, n) for n in (10, 20, 50, 150, 200))
    hi52 = H.rolling(252, min_periods=200).max(); lo52 = L.rolling(252, min_periods=200).min()
    dv50 = rolling(C * V, 50)
    liquid = (C >= 20) & (dv50 >= 15e6)
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
                           ext=float(c_[-1] / piv - 1), vr=float(v_[-1] / max(1.0, np.mean(v_[-51:-1])))))
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
def portfolio(trades, sims, sig, mode, first_i, equity0=100000.0):
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
        cands = sorted(by_day.get(d, []), key=lambda k: -trades[k]["rs"])
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


def main():
    t0 = time.time()
    px = load_prices()
    print(f"prices: {len(px)} series ({time.time()-t0:.0f}s)")
    sig = build_signals(px)
    first_i = int(np.argmax(sig["idx"] >= pd.Timestamp(f"{START_YEAR}-01-01")))
    print(f"signals built ({time.time()-t0:.0f}s); events: {int(sig['ev'].to_numpy().sum())}")
    trades = find_trades(sig, first_i)
    print(f"trades: {len(trades)} ({time.time()-t0:.0f}s)")
    res = {}
    curves = {}
    spy = sig["spy"]
    spyc = spy[spy.index >= sig["idx"][first_i]]
    for policy in ("mixed", "strength"):
        sims = [simulate(t, sig, policy) for t in trades]
        for mode in ("all", "notred", "green", "advice"):
            eq, taken = portfolio(trades, sims, sig, mode, first_i)
            key = f"{policy}/{mode}"
            res[key] = stats(eq, trades, sims, taken, spyc)
            curves[key] = [round(float(v), 0) for v in eq.iloc[first_i::5].fillna(100000).tolist()]
            print(key, {k: v for k, v in res[key].items() if k != "years"})
    # by-regime expectancy on unfiltered trades (mixed policy), independent of sizing
    sims = [simulate(t, sig, "mixed") for t in trades]
    reg = {}
    for lbl, lo, hi in (("green (≥7)", 7, 99), ("yellow (4–6)", 4, 6.99), ("red (≤3)", -1, 3.99)):
        R = [sims[k]["R"] for k, t in enumerate(trades) if lo <= t["regime"] <= hi]
        if R:
            R = np.array(R)
            reg[lbl] = dict(n=len(R), win=round(float((R > 0).mean()) * 100, 1), exp=round(float(R.mean()), 2),
                            avgwin=round(float(R[R > 0].mean()) if (R > 0).any() else 0, 2),
                            avgloss=round(float(R[R <= 0].mean()) if (R <= 0).any() else 0, 2))
    ty = {}
    for k, t in enumerate(trades):
        y = str(sig["idx"][t["i"]].year)
        ty.setdefault(y, []).append(sims[k]["R"])
    byyear = {y: dict(n=len(v), exp=round(float(np.mean(v)), 2), win=round(float((np.array(v) > 0).mean()) * 100, 1))
              for y, v in sorted(ty.items())}
    spy_c = round(float((spyc.iloc[-1] / spyc.iloc[0]) ** (365.25 / ((spyc.index[-1] - spyc.index[0]).days)) - 1) * 100, 1)
    spy_dd = round(float((spyc / spyc.cummax() - 1).min()) * 100, 1)
    spy_y = {}
    ye = spyc.groupby(spyc.index.year).last(); prev = spyc.iloc[0]
    for y, v in ye.items():
        spy_y[int(y)] = round(float(v / prev - 1) * 100, 1); prev = v
    dates = [str(d.date()) for d in sig["idx"][first_i::5]]
    spy_curve = [round(float(v / spyc.iloc[0] * 100000), 0) for v in spyc.iloc[::5].tolist()]
    out = dict(start=str(sig["idx"][first_i].date()), end=str(sig["idx"][-1].date()), n_universe=len(sig["tick"]),
               results=res, regime_expectancy=reg, by_year=byyear, spy=dict(cagr=spy_c, maxdd=spy_dd, years=spy_y),
               dates=dates, curves=curves, spy_curve=spy_curve,
               trades_sample=[dict(ticker=t["ticker"], date=str(sig["idx"][t["i"]].date()), R=round(sims[k]["R"], 2),
                                   regime=t["regime"], bases=t["bases"], risk=round(t["risk"] * 100, 1), rs=round(t["rs"]),
                                   dec=t["dec"], vol_dry=t["vol_dry"], hd=t["hd"], pa=t["pa"], ext=round(t["ext"] * 100, 1),
                                   vr=round(t["vr"], 2))
                              for k, t in enumerate(trades)])
    json.dump(out, open(os.path.join(OUT, "backtest.json" if ENTRY == "close" else f"backtest_{ENTRY}.json"), "w"), ensure_ascii=False)
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
