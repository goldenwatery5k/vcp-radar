"""Pre-registered stock-selection test. Success = beats SPY CAGR in IS(2018-22), OOS(2023-now) AND full period, maxDD within 5pts of SPY.
Universes: PIT S&P500 members; top-1000 by 50d $ volume (point-in-time, but survivors-only prices). Long-only, equal weight, monthly rebalance, 0.1%/side cost.
All 20 configs declared up front; nothing tuned after seeing results."""
import itertools, numpy as np, pandas as pd, json
px = pd.read_pickle("data/bt_prices_all.pkl")
idx = px["SPY"].index
tick = [t for t in px if t not in ("SPY", "QQQ", "IWM")]
def mat(col):
    return pd.DataFrame({t: px[t][col] for t in tick}).reindex(idx).astype("float32")
C = mat("Close").ffill(limit=5); V = mat("Volume").fillna(0); Hh = mat("High").where(lambda x: x.notna(), C)
n = len(idx); cols = {t: j for j, t in enumerate(C.columns)}
dv50 = (C * V).rolling(50, min_periods=50).mean()
hi252 = Hh.rolling(252, min_periods=200).max()
ret = C.pct_change().fillna(0).clip(-0.5, 1.0).to_numpy(np.float32)
mom12_1 = (C.shift(21) / C.shift(252) - 1)
mom6 = C / C.shift(126) - 1
rs_raw = 0.4 * (C / C.shift(63) - 1) + 0.2 * (C / C.shift(126) - 1) + 0.2 * (C / C.shift(189) - 1) + 0.2 * (C / C.shift(251) - 1)
vol60 = C.pct_change().rolling(60, min_periods=60).std() * np.sqrt(252)
prox = C / hi252
s200 = C.rolling(200).mean()
rk = lambda df: df.rank(axis=1, pct=True)
SCORES = {
    "RS composite": rs_raw,
    "12-1 momentum": mom12_1,
    "52w-high proximity": prox,
    "risk-adj mom (12-1/vol)": mom12_1 / vol60,
    "rank(prox)+rank(12-1)": rk(prox) + rk(mom12_1),
}
SC = {k: v.to_numpy(np.float32) for k, v in SCORES.items()}
Cn = C.to_numpy(np.float32); S200 = s200.to_numpy(np.float32)
# universes
d = pd.read_csv("data/sp500_hist_upd.csv"); d["date"] = pd.to_datetime(d["date"])
def base(t): return t.split("-")[0] if "-" in t and t.split("-")[-1].isdigit() and len(t.split("-")[-1]) == 6 else t
snap = [(r.date, {base(x) for x in r.tickers.split(",")}) for r in d.itertuples()]
mem500 = np.zeros((n, len(cols)), bool); k = 0
for di, dt in enumerate(idx):
    while k + 1 < len(snap) and snap[k + 1][0] <= dt:
        k += 1
    for t in snap[k][1]:
        j = cols.get(t)
        if j is not None:
            mem500[di, j] = True
rank_dv = dv50.rank(axis=1, ascending=False).to_numpy()
mem1000 = (rank_dv <= 1000) & (Cn >= 5)
UNI = {"S&P500 PIT": mem500, "top1000 $vol": mem1000}
first = int(np.argmax(idx >= pd.Timestamp("2018-01-01"))); split = int(np.argmax(idx >= pd.Timestamp("2023-01-01")))
spy = px["SPY"]["Close"].astype(float).reindex(idx).ffill(); spyr = spy.pct_change().fillna(0)
def met(x):
    x = pd.Series(x).dropna(); eq = (1 + x).cumprod(); y = len(x) / 252
    return dict(cagr=round(float((eq.iloc[-1] ** (1 / y) - 1) * 100), 1), dd=round(float((eq / eq.cummax() - 1).min() * 100), 1),
                sh=round(float(x.mean() / x.std() * np.sqrt(252)), 2))
bf, bi, bo = met(spyr.iloc[first:]), met(spyr.iloc[first:split]), met(spyr.iloc[split:])
def run(mem, score, N, reb=21, trend=True, cost=0.001):
    w = np.zeros(Cn.shape[1], np.float32); port = np.zeros(n)
    for dd in range(first, n):
        if (dd - first) % reb == 0:
            ok = mem[dd - 1] & ~np.isnan(score[dd - 1])
            if trend:
                ok &= Cn[dd - 1] > S200[dd - 1]
            cand = np.where(ok)[0]; wn = np.zeros_like(w)
            if len(cand):
                top = cand[np.argsort(-score[dd - 1, cand])[:N]]
                wn[top] = 1.0 / len(top)
            port[dd] -= float(np.abs(wn - w).sum()) * cost; w = wn
        port[dd] += float((w * ret[dd]).sum())
        w = w * (1 + ret[dd]); s = w.sum()
        if s > 0: w = w / s
    return port
def main():
    rows = []
    for (un, mem), (sn, sc), N in itertools.product(UNI.items(), SC.items(), (20, 40)):
        p = run(mem, sc, N)
        f, i, o = met(p[first:]), met(p[first:split]), met(p[split:])
        ok = f["cagr"] > bf["cagr"] and i["cagr"] > bi["cagr"] and o["cagr"] > bo["cagr"] and f["dd"] >= bf["dd"] - 5
        rows.append(dict(uni=un, score=sn, N=N, full=f, IS=i, OOS=o, success=bool(ok)))
        print(f"{un:13s} {sn:24s} N={N:2d} FULL {f['cagr']:+5.1f}%/{f['dd']:6.1f}%/sh{f['sh']:.2f} | IS {i['cagr']:+5.1f}% | OOS {o['cagr']:+5.1f}%{'  <== PASS' if ok else ''}", flush=True)
    json.dump(rows, open("out/research_c.json", "w"))
    print("passes:", sum(r["success"] for r in rows), "of", len(rows))

if __name__ == '__main__':
    main()
