import numpy as np, pandas as pd
import research_c as R
cfgs = [("top1000 $vol","risk-adj mom (12-1/vol)",20),("top1000 $vol","risk-adj mom (12-1/vol)",40),("top1000 $vol","12-1 momentum",40),("top1000 $vol","RS composite",40)]
spyy = R.spyr.iloc[R.first:].groupby(R.idx[R.first:].year).apply(lambda r: ((1 + r).prod() - 1) * 100)
print("year   SPY   " + "   ".join(f"{c[1][:10]}/{c[2]}" for c in cfgs))
res = {}
for un, sn, N in cfgs:
    for cost in (0.001, 0.003):
        p = R.run(R.UNI[un], R.SC[sn], N, cost=cost)
        res[(sn, N, cost)] = pd.Series(p[R.first:], index=R.idx[R.first:])
for y in spyy.index:
    row = f"{y}  {spyy[y]:+6.1f}  " + "  ".join(f"{((1+res[(c[1],c[2],0.001)][R.idx[R.first:].year==y]).prod()-1)*100:+9.1f}" for c in cfgs)
    print(row)
print()
for c in cfgs:
    for cost in (0.001, 0.003):
        s = res[(c[1], c[2], cost)]
        print(f"{c[1]:24s} N={c[2]:2d} cost {cost*100:.1f}%/side: {R.met(s.values)}  IS {R.met(s.iloc[:R.split-R.first].values)['cagr']} OOS {R.met(s.iloc[R.split-R.first:].values)['cagr']}")
