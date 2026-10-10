import json, os
H = os.path.dirname(os.path.abspath(__file__))
A = json.load(open(f"{H}/out/backtest.json")); P = json.load(open(f"{H}/out/backtest_pivot.json"))
def slim(d):
    return dict(results={k: v for k, v in d["results"].items()}, curves={k: d["curves"][k] for k in d["curves"] if k.startswith("mixed")},
                regime=d["regime_expectancy"], byyear=d["by_year"], ntr=len(d["trades_sample"]))
data = dict(close=slim(A), pivot=slim(P), dates=A["dates"], spy_curve=A["spy_curve"], spy=A["spy"], start=A["start"], end=A["end"], nuni=A["n_universe"])
html = open(f"{H}/bt_template.html").read().replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False))
open(f"{H}/out/backtest.html", "w").write(html)
print(len(html))
