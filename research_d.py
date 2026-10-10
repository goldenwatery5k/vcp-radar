"""Risk-managed momentum: position stops x weighting x portfolio overlay (24 configs declared up front).
Base signal fixed = risk-adjusted 12-1 momentum, top 20, universe top-1000 by $ volume, monthly rebalance, cost 0.1%/side (stop exits also pay it).
Judged on risk-adjusted terms: Sharpe, maxDD, Calmar, IS/OOS."""
import itertools, json, numpy as np, pandas as pd
import research_c as R

n = R.n; idx = R.idx; Cn = R.Cn
sc = R.SC["risk-adj mom (12-1/vol)"].copy()
vol60 = R.vol60.to_numpy(np.float32)
sc = np.where(vol60 >= 0.12, sc, np.nan)
mem = R.UNI["top1000 $vol"]; S200 = R.S200; ret = R.ret
spy = R.spy.to_numpy(); spy200 = R.spy.rolling(200).mean().to_numpy()
first, split = R.first, R.split
COST = 0.001; N = 20; REB = 21

def pick(d):
    ok = mem[d] & ~np.isnan(sc[d]) & (Cn[d] > S200[d])
    cand = np.where(ok)[0]
    return cand[np.argsort(-sc[d, cand])[:N]] if len(cand) else cand

def run(stop, weighting, overlay):
    ncol = Cn.shape[1]
    w = np.zeros(ncol, np.float32); hi = np.zeros(ncol, np.float32)
    out = np.zeros(n); unscaled = np.zeros(n)
    prev_on = True
    since = 10**9
    for d in range(first, n):
        on = True
        if overlay == "regime200":
            on = bool(spy[d - 1] > spy200[d - 1])
        sched = (d - first) % REB == 0
        flip_on = overlay == "regime200" and on and not prev_on
        prev_on = on
        scale = 1.0
        if overlay == "voltarget":
            h = unscaled[max(first, d - 60):d]
            if len(h) >= 20 and h.std() > 0:
                scale = float(min(1.0, 0.20 / (h.std() * np.sqrt(252))))
        if not on and w.sum() > 0:
            out[d] -= float(w.sum()) * COST * scale; w[:] = 0
        elif on and (sched or flip_on):
            top = pick(d - 1)
            wn = np.zeros(ncol, np.float32)
            if len(top):
                if weighting == "invvol":
                    iv = 1.0 / np.maximum(vol60[d - 1, top], 0.10)
                    ww = iv / iv.sum()
                    ww = np.minimum(ww, 0.10)             # cap 10% per name; remainder stays in cash
                    wn[top] = ww
                else:
                    wn[top] = 1.0 / N
            out[d] -= float(np.abs(wn - w).sum()) * COST * scale
            w = wn; hi[:] = Cn[d - 1]
        r_u = float((w * ret[d]).sum())
        unscaled[d] = r_u
        out[d] += r_u * scale
        w = w * (1 + ret[d])                               # drift
        tot = w.sum()
        # hold weights as fraction of ORIGINAL equity; renormalise only if >1
        if tot > 1: w = w / tot
        held = w > 0
        hi[held] = np.maximum(hi[held], Cn[d, held])
        if stop != "none" and held.any():
            if stop.startswith("trail"):
                lvl = float(stop[5:]) / 100
                hit = held & (Cn[d] < hi * (1 - lvl))
            else:   # ma50 position-level trend exit
                m50 = R.C.rolling(50).mean().to_numpy()[d] if False else MA50[d]
                hit = held & (Cn[d] < m50)
            if hit.any():
                out[d] -= float(w[hit].sum()) * COST * scale
                w[hit] = 0
    return out

MA50 = R.C.rolling(50).mean().to_numpy(np.float32)

def met(x):
    x = pd.Series(x).dropna(); eq = (1 + x).cumprod(); y = len(x) / 252
    c = float(eq.iloc[-1] ** (1 / y) - 1) * 100; dd = float((eq / eq.cummax() - 1).min() * 100)
    return dict(cagr=round(c, 1), dd=round(dd, 1), sh=round(float(x.mean() / x.std() * np.sqrt(252)), 2), calmar=round(c / abs(dd), 2))

bf = R.met(R.spyr.iloc[first:]); bi = R.met(R.spyr.iloc[first:split]); bo = R.met(R.spyr.iloc[split:])
print("SPY FULL", bf, "calmar", round(bf["cagr"] / abs(bf["dd"]), 2), "IS", bi["cagr"], "OOS", bo["cagr"], flush=True)
def main():
    rows = []
    for stop, wt, ov in itertools.product(("none", "trail20", "trail30", "ma50"), ("equal", "invvol"), ("none", "regime200", "voltarget")):
        p = run(stop, wt, ov)
        f, i, o = met(p[first:]), met(p[first:split]), met(p[split:])
        yrs = pd.Series(p[first:], index=idx[first:]).groupby(idx[first:].year).apply(lambda r: round(((1 + r).prod() - 1) * 100, 1)).to_dict()
        ok = f["sh"] > 0.82 and f["dd"] > -35 and f["cagr"] >= 14.7
        rows.append(dict(stop=stop, weighting=wt, overlay=ov, full=f, IS=i, OOS=o, years={int(k): v for k, v in yrs.items()}, ok=bool(ok)))
        print(f"{stop:8s} {wt:7s} {ov:10s} FULL {f['cagr']:+5.1f}%/{f['dd']:6.1f}%/sh{f['sh']:.2f}/cal{f['calmar']:.2f} | IS {i['cagr']:+5.1f}% | OOS {o['cagr']:+5.1f}% | 2018 {yrs.get(2018):+6.1f} 2022 {yrs.get(2022):+6.1f}{'  <== PASS' if ok else ''}", flush=True)
    json.dump(rows, open("out/research_d.json", "w"))
    print("pass:", sum(r["ok"] for r in rows), "of", len(rows))

if __name__ == '__main__':
    main()
