"""Exit / sizing research: can the Stage2+VCP breakout entries beat buy-and-hold SPY?
Split: in-sample 2018-2022, out-of-sample 2023-now. Everything equal-weight, long-only, no leverage."""
import json, sys, numpy as np, pandas as pd
import backtest as BT
import universe

UNI = sys.argv[1] if len(sys.argv) > 1 else "index"
px = BT.load_prices()
idx_t = {u["ticker"] for u in universe.load() if u["market"] == "US"}
tick = [t for t in px if t in idx_t] if UNI == "index" else [t for t in px if t not in ("SPY", "QQQ", "IWM")]
mp, mdv = (20, 15e6) if UNI == "index" else (10, 5e6)
sig = BT.build_signals(px, tick, mp, mdv)
first_i = int(np.argmax(sig["idx"] >= pd.Timestamp("2018-01-01")))
trades = BT.find_trades(sig, first_i)
O, H, L, C = (sig[k].to_numpy(float) for k in ("O", "H", "L", "C"))
S20, S50 = sig["s20"].to_numpy(float), sig["s50"].to_numpy(float)
spy = sig["spy"].to_numpy(float); n = len(spy); idx = sig["idx"]
SLIP = BT.SLIP
spy200 = pd.Series(spy).rolling(200).mean().to_numpy()
print(UNI, "trades", len(trades), flush=True)

def run_exit(tr, rule):
    j, i0 = tr["j"], tr["i"]; e = tr["entry"]; stop0 = tr["stop"]
    hi_c = C[i0, j]; path = {}
    for d in range(i0 + 1, n):
        c = C[d, j]
        if np.isnan(c):
            continue
        o, l = O[d, j], L[d, j]
        stop = stop0
        if rule.get("be") and hi_c >= e * (1 + rule["be"]):
            stop = max(stop, e * 1.01)
        if rule.get("trail_pct") and hi_c >= e * (1 + rule.get("trail_after", 0)):
            stop = max(stop, hi_c * (1 - rule["trail_pct"]))
        if o <= stop:
            return d, o * (1 - SLIP) / e - 1, path
        if l <= stop:
            return d, stop * (1 - SLIP) / e - 1, path
        hi_c = max(hi_c, c)
        ma = rule.get("ma")
        if ma:
            m = (S50 if ma == 50 else S20)[d, j]
            if not np.isnan(m) and c < m and (not rule.get("ma_after") or hi_c >= e * (1 + rule["ma_after"])):
                return d, c * (1 - SLIP) / e - 1, path
        if rule.get("max_hold") and d - i0 >= rule["max_hold"]:
            return d, c * (1 - SLIP) / e - 1, path
        path[d] = c / e - 1
    return None, C[n - 1, j] / e - 1, path

def portfolio(trades, exits, slots, regime, rank="rs", start=first_i, end=n):
    eq = 100000.0; cash = eq; opens = {}; curve = np.full(n, np.nan)
    by_day = {}
    for k, t in enumerate(trades):
        by_day.setdefault(t["i"], []).append(k)
    for d in range(start, end):
        # exits today
        for k in list(opens):
            xd, r, _ = exits[k]
            if xd is not None and xd <= d:
                cash += opens[k] * (1 + r); del opens[k]
        val = cash
        for k, amt in opens.items():
            xd, r, path = exits[k]
            val += amt * (1 + path.get(d, path[max(path)] if path else 0.0))
        curve[d] = val
        if regime(d) and len(opens) < slots:
            for k in sorted(by_day.get(d, []), key=lambda q: -trades[q][rank]):
                if len(opens) >= slots:
                    break
                alloc = min(val / slots, cash)
                if alloc < 0.5 * val / slots:
                    continue
                opens[k] = alloc; cash -= alloc
    for k in list(opens):
        xd, r, _ = exits[k]; cash += opens[k] * (1 + r)
    s = pd.Series(curve, index=idx).ffill().iloc[start:end]
    return s

def metrics(s):
    s = s.dropna(); yrs = (s.index[-1] - s.index[0]).days / 365.25
    ret = s.pct_change().dropna()
    return dict(cagr=round(((s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1) * 100, 1), dd=round((s / s.cummax() - 1).min() * 100, 1),
                sh=round(ret.mean() / ret.std() * np.sqrt(252), 2) if ret.std() > 0 else 0)

def spy_m(a, b):
    s = pd.Series(spy[a:b], index=idx[a:b]); return metrics(s)

split = int(np.argmax(idx >= pd.Timestamp("2023-01-01")))
RULES = {
 "stop only (hold)":            dict(),
 "ma50 exit":                   dict(ma=50),
 "ma20 exit":                   dict(ma=20),
 "ma50 + BE@+15%":              dict(ma=50, be=0.15),
 "trail 20% after +20%":        dict(trail_pct=0.20, trail_after=0.20),
 "trail 15%":                   dict(trail_pct=0.15),
 "ma50 + trail 25%":            dict(ma=50, trail_pct=0.25),
}
REG = {
 "no filter": lambda d: True,
 "SPY>200d": lambda d: spy[d] > spy200[d],
 "green(>=7)": lambda d: sig["regime"].iloc[d] >= 7,
}
res = []
print("SPY full:", spy_m(first_i, n), " IS:", spy_m(first_i, split), " OOS:", spy_m(split, n))
for rn, rule in RULES.items():
    ex = [run_exit(t, rule) for t in trades]
    R = np.array([e[1] for e in ex]); hold = None
    print(f"\n== {rn}: per-trade mean {R.mean()*100:+.1f}%  median {np.median(R)*100:+.1f}%  win {(R>0).mean()*100:.0f}%  avg win {R[R>0].mean()*100:+.0f}%  avg loss {R[R<=0].mean()*100:.0f}%", flush=True)
    for gn, g in REG.items():
        for slots in (10, 20):
            s = portfolio(trades, ex, slots, g)
            full = metrics(s); isr = metrics(s.iloc[:split - first_i]); oos = metrics(s.iloc[split - first_i:])
            res.append(dict(rule=rn, reg=gn, slots=slots, full=full, IS=isr, OOS=oos))
            print(f"  {gn:11s} slots={slots:2d} FULL {full['cagr']:+5.1f}%/{full['dd']:6.1f}%/sh{full['sh']:.2f} | IS {isr['cagr']:+5.1f}% | OOS {oos['cagr']:+5.1f}%/{oos['dd']:6.1f}%", flush=True)
json.dump(res, open(f"out/research_{UNI}.json", "w"))
