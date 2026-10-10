"""Momentum rotation on POINT-IN-TIME S&P 500 membership (no index-inclusion look-ahead). Survivorship: members with no Yahoo data are missing."""
import itertools, numpy as np, pandas as pd
import backtest as BT
px = BT.load_prices()
d = pd.read_csv("data/sp500_hist_upd.csv"); d["date"] = pd.to_datetime(d["date"])
def base(t): return t.split("-")[0] if "-" in t and t.split("-")[-1].isdigit() and len(t.split("-")[-1]) == 6 else t
snap = [(r.date, {base(x) for x in r.tickers.split(",")}) for r in d.itertuples()]
allm = set().union(*[s for dt, s in snap if dt >= pd.Timestamp("2016-06-01")])
tick = [t for t in px if t in allm and t not in ("SPY", "QQQ", "IWM")]
sig = BT.build_signals(px, tick, 1, 1e6)
idx = sig["idx"]; n = len(idx); col = {t: j for j, t in enumerate(sig["C"].columns)}
mem = np.zeros((n, len(col)), bool); k = 0
for di, dt in enumerate(idx):
    while k + 1 < len(snap) and snap[k + 1][0] <= dt:
        k += 1
    for t in snap[k][1]:
        j = col.get(t)
        if j is not None:
            mem[di, j] = True
print("tickers", len(tick), " avg members/day with data", mem.sum(1)[-1500:].mean().round(0))
C = sig["C"]; Cn = C.to_numpy(float); ret = C.pct_change().fillna(0).clip(-0.5, 1.0).to_numpy()
raw = sig["raw"].to_numpy(float); s200 = C.rolling(200).mean().to_numpy()
liq = sig["C"].notna().to_numpy()   # membership defines universe
spy = sig["spy"]; spyv = spy.to_numpy(); spy200 = spy.rolling(200).mean().to_numpy(); spyr = spy.pct_change().fillna(0)
rg = sig["regime"].to_numpy()
first = int(np.argmax(idx >= pd.Timestamp("2018-01-01"))); split = int(np.argmax(idx >= pd.Timestamp("2023-01-01")))
# raw uses `liquid` (price/vol); recompute raw w/o liquidity gate
r = lambda nb: C / C.shift(nb) - 1
rawfull = (0.4 * r(63) + 0.2 * r(126) + 0.2 * r(189) + 0.2 * r(251)).to_numpy()
COST = 0.001
def met(x):
    x = pd.Series(x).dropna(); eq = (1 + x).cumprod(); y = len(x) / 252
    return dict(cagr=round((eq.iloc[-1] ** (1 / y) - 1) * 100, 1), dd=round((eq / eq.cummax() - 1).min() * 100, 1),
                sh=round(x.mean() / x.std() * np.sqrt(252), 2))
print("SPY  FULL", met(spyr.iloc[first:]), "IS", met(spyr.iloc[first:split])["cagr"], "OOS", met(spyr.iloc[split:]))
REGS = {"none": lambda dd: True, "SPY>200d": lambda dd: spyv[dd] > spy200[dd], "green>=4": lambda dd: rg[dd] >= 4}
def run(N, reb, regf, score, trend=True):
    w = np.zeros(Cn.shape[1]); port = np.zeros(n)
    for dd in range(first, n):
        if (dd - first) % reb == 0:
            wn = np.zeros_like(w)
            if regf(dd - 1):
                ok = mem[dd - 1] & ~np.isnan(score[dd - 1])
                if trend:
                    ok &= Cn[dd - 1] > s200[dd - 1]
                cand = np.where(ok)[0]
                if len(cand):
                    top = cand[np.argsort(-score[dd - 1, cand])[:N]] if N else cand
                    wn[top] = 1.0 / len(top)
            port[dd] -= np.abs(wn - w).sum() * COST; w = wn
        port[dd] += float((w * ret[dd]).sum())
        w = w * (1 + ret[dd]); s = w.sum()
        if s > 0: w = w / s * min(1.0, s)
    return port
# benchmark: equal-weight all members, monthly
ew = run(0, 21, REGS["none"], rawfull, trend=False)
print("EW S&P500 members FULL", met(ew[first:]), "IS", met(ew[first:split])["cagr"], "OOS", met(ew[split:])["cagr"])
print()
for N, reb, rn in itertools.product((10, 20, 30), (21, 63), REGS):
    p = run(N, reb, REGS[rn], rawfull)
    f, i, o = met(p[first:]), met(p[first:split]), met(p[split:])
    tag = "  <== beats SPY full+IS+OOS" if (f["cagr"] > 14.7 and i["cagr"] > 9.3 and o["cagr"] > 22.4) else ("  * beats full" if f["cagr"] > 14.7 else "")
    print(f"N={N:2d} reb={reb:2d} {rn:9s} FULL {f['cagr']:+5.1f}%/{f['dd']:6.1f}%/sh{f['sh']:.2f} | IS {i['cagr']:+5.1f}% | OOS {o['cagr']:+5.1f}%/{o['dd']:6.1f}%{tag}", flush=True)
