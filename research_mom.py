"""Cross-sectional momentum rotation (Stage-2 style leaders), long-only, equal-weight, regime-gated. IS 2018-22 / OOS 2023-now."""
import sys, itertools, numpy as np, pandas as pd
import backtest as BT, universe
px = BT.load_prices()
idx_t = {u["ticker"] for u in universe.load() if u["market"] == "US"}
UNIS = {"index": ([t for t in px if t in idx_t], 20, 15e6), "expanded": ([t for t in px if t not in ("SPY","QQQ","IWM")], 10, 5e6)}
COST = 0.001
def met(r):
    r = r.dropna(); eq = (1 + r).cumprod(); yrs = len(r) / 252
    return dict(cagr=round((eq.iloc[-1] ** (1 / yrs) - 1) * 100, 1), dd=round((eq / eq.cummax() - 1).min() * 100, 1),
                sh=round(r.mean() / r.std() * np.sqrt(252), 2) if r.std() > 0 else 0)
for uname, (tick, mp, mdv) in UNIS.items():
    sig = BT.build_signals(px, tick, mp, mdv)
    idx = sig["idx"]; C = sig["C"]; ret = C.pct_change().fillna(0).clip(-0.5, 1.0).to_numpy()
    liquid = sig["liquid"].to_numpy(bool); raw = sig["raw"].to_numpy(float)
    s200 = C.rolling(200).mean().to_numpy(); Cn = C.to_numpy(float)
    spy = sig["spy"]; spy_r = spy.pct_change().fillna(0).to_numpy(); spy200 = spy.rolling(200).mean().to_numpy(); spyv = spy.to_numpy()
    reg = sig["regime"].to_numpy()
    first = int(np.argmax(idx >= pd.Timestamp("2018-01-01"))); split = int(np.argmax(idx >= pd.Timestamp("2023-01-01"))); n = len(idx)
    base = pd.Series(spy_r, index=idx)
    print(f"\n#### universe {uname} ({len(tick)} tickers)   SPY full {met(base.iloc[first:])}  IS {met(base.iloc[first:split])}  OOS {met(base.iloc[split:])}")
    REGS = {"none": lambda d: True, "SPY>200d": lambda d: spyv[d] > spy200[d], "green>=7": lambda d: reg[d] >= 7, "green>=4": lambda d: reg[d] >= 4}
    for N, reb, rg in itertools.product((10, 20, 40), (21, 63), REGS):
        w = np.zeros(Cn.shape[1]); port = np.zeros(n); prev = w.copy()
        for d in range(first, n):
            if (d - first) % reb == 0:
                w_new = np.zeros_like(w)
                if REGS[rg](d - 1):
                    ok = liquid[d - 1] & ~np.isnan(raw[d - 1]) & (Cn[d - 1] > s200[d - 1])
                    cand = np.where(ok)[0]
                    if len(cand):
                        top = cand[np.argsort(-raw[d - 1, cand])[:N]]
                        w_new[top] = 1.0 / N
                turn = np.abs(w_new - w).sum()
                port[d] -= turn * COST
                w = w_new
            elif not REGS[rg](d - 1) and w.sum() > 0:   # regime flips off mid-period -> go to cash
                port[d] -= w.sum() * COST; w = np.zeros_like(w)
            port[d] += float((w * ret[d]).sum())
            # drift weights
            w = w * (1 + ret[d]); s = w.sum()
            if s > 0:
                w = w / s * min(1.0, s)
        r = pd.Series(port, index=idx)
        f, i, o = met(r.iloc[first:]), met(r.iloc[first:split]), met(r.iloc[split:])
        flag = "  <== beats SPY full+IS+OOS" if (f["cagr"] > 14.6 and i["cagr"] > 9.2 and o["cagr"] > 22.4) else ("  * beats full" if f["cagr"] > 14.6 else "")
        print(f"N={N:2d} reb={reb:2d} {rg:9s} FULL {f['cagr']:+5.1f}%/{f['dd']:6.1f}%/sh{f['sh']:.2f} | IS {i['cagr']:+5.1f}% | OOS {o['cagr']:+5.1f}%/{o['dd']:6.1f}%{flag}", flush=True)
