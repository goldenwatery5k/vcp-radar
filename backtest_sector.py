"""Does sector / industry-group strength improve the breakout system? (index universe, GICS from data/universe.json)
Run: ENTRY=close .venv/bin/python backtest_sector.py 2018"""
import json, sys, numpy as np, pandas as pd
import backtest as BT
import universe

px = BT.load_prices()
meta = {u["ticker"]: u for u in universe.load() if u["market"] == "US" and u.get("group") and u.get("sector")}
tick = [t for t in px if t in meta]
sig = BT.build_signals(px, tick, 20, 15e6)
first_i = int(np.argmax(sig["idx"] >= pd.Timestamp(f"{BT.START_YEAR}-01-01")))
trades = BT.find_trades(sig, first_i)
raw = sig["raw"]                       # days x tickers, NaN when illiquid
def strength(key, min_n):
    g = pd.Series({t: meta[t][key] for t in tick})
    med = raw.T.groupby(g).median().T
    cnt = raw.notna().T.groupby(g).sum().T
    med = med.where(cnt >= min_n)
    return med.rank(axis=1, pct=True), g
gp, gmap = strength("group", 4)
sp, smap = strength("sector", 4)
for t in trades:
    d = sig["idx"][t["i"]]
    t["gp"] = float(gp.at[d, gmap[t["ticker"]]]) if gmap[t["ticker"]] in gp.columns else np.nan
    t["sp"] = float(sp.at[d, smap[t["ticker"]]]) if smap[t["ticker"]] in sp.columns else np.nan
print("trades", len(trades), "with group strength", sum(1 for t in trades if not np.isnan(t["gp"])))
spyc = sig["spy"][sig["spy"].index >= sig["idx"][first_i]]
out = {}
for policy in ("mixed", "strength"):
    sims = [BT.simulate(t, sig, policy) for t in trades]
    R = np.array([s["R"] for s in sims])
    if policy == "mixed":
        print("\nper-trade expectancy by industry-group strength (percentile among ~150 groups):")
        gpa = np.array([t["gp"] for t in trades]); spa = np.array([t["sp"] for t in trades])
        for lo, hi in ((0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)):
            m = (gpa >= lo) & (gpa < hi)
            print(f"  group pct {lo:.1f}-{min(hi,1):.1f}: n={m.sum():4d} win={(R[m]>0).mean()*100:4.1f}% exp={R[m].mean():+.3f}R")
        print("by sector strength (11 sectors):")
        for lo, hi in ((0, .34), (.34, .67), (.67, 1.01)):
            m = (spa >= lo) & (spa < hi)
            print(f"  sector pct {lo:.2f}-{min(hi,1):.2f}: n={m.sum():4d} win={(R[m]>0).mean()*100:4.1f}% exp={R[m].mean():+.3f}R")
    variants = {
        "none": lambda t: False,
        "group top30%": lambda t: not (t["gp"] >= 0.70),
        "group top20%": lambda t: not (t["gp"] >= 0.80),
        "sector top3": lambda t: not (t["sp"] >= 0.70),
        "group top30% & sector top3": lambda t: not (t["gp"] >= 0.70 and t["sp"] >= 0.70),
    }
    print(f"\n[{policy}] portfolio results")
    for name, f in variants.items():
        skip = frozenset(k for k, t in enumerate(trades) if f(t))
        for mode in ("all", "green"):
            eq, taken = BT.portfolio(trades, sims, sig, mode, first_i, skip=skip)
            st = BT.stats(eq, trades, sims, taken, spyc)
            out[f"{policy}/{name}/{mode}"] = {k: v for k, v in st.items() if k != "years"}
            print(f"  {name:28s} {mode:5s} n={st['trades']:4d} win={st['win_rate']:4.1f}% exp={st['expectancy_R']:+.2f}R PF={st['profit_factor']:.2f} CAGR={st['cagr']:+.1f}% maxDD={st['maxdd']:.1f}% sharpe={st['sharpe']:.2f}")
json.dump(out, open("out/backtest_sector.json", "w"), ensure_ascii=False)
