"""VCP / Stage-2 screener + top-down mind map -> out/data.json + out/index.html

Stage 2 trend template + VCP breakout rules, 1R risk framework. Run:  ./run.sh
"""
import datetime as dt
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import yfinance as yf

import universe

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = json.load(open(os.path.join(HERE, "config.json")))
MM = json.load(open(os.path.join(HERE, "mindmap.json")))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

B, E, T, R = CFG["base"], CFG["entry"], CFG["trend_template"], CFG["risk"]


# ----------------------------------------------------------------- data
def fetch(tickers, period):
    out = {}
    chunks = [tickers[i:i + CFG["chunk"]] for i in range(0, len(tickers), CFG["chunk"])]
    for n, ch in enumerate(chunks, 1):
        try:
            df = yf.download(ch, period=period, interval="1d", auto_adjust=True, progress=False,
                             group_by="ticker", threads=True)
        except Exception as e:  # network hiccup: skip chunk
            print(f"  chunk {n}/{len(chunks)} failed: {e}")
            continue
        for t in ch:
            try:
                d = df[t] if len(ch) > 1 else df
                d = d.dropna(subset=["Close"])
                if len(d) >= 60:
                    out[t] = d[["Open", "High", "Low", "Close", "Volume"]]
            except Exception:
                pass
        print(f"  downloaded {n}/{len(chunks)}  ({len(out)} ok)", flush=True)
    return out


def sma(a, n):
    return pd.Series(a).rolling(n).mean().to_numpy()


def zigzag(h, l, thr):
    """alternating swing points [(idx, price, 'P'|'T')] using % reversal threshold"""
    n = len(h)
    if n < 3:
        return []
    piv = []
    direction = 0
    hi_i, hi = 0, h[0]
    lo_i, lo = 0, l[0]
    for i in range(1, n):
        if direction == 0:
            if h[i] > hi:
                hi_i, hi = i, h[i]
            if l[i] < lo:
                lo_i, lo = i, l[i]
            if hi_i > lo_i and hi / lo - 1 >= thr:
                piv.append((lo_i, lo, "T")); direction = 1
            elif lo_i > hi_i and 1 - lo / hi >= thr:
                piv.append((hi_i, hi, "P")); direction = -1
        elif direction == 1:
            if h[i] > hi:
                hi_i, hi = i, h[i]
            elif l[i] <= hi * (1 - thr):
                piv.append((hi_i, hi, "P")); direction = -1; lo_i, lo = i, l[i]
        else:
            if l[i] < lo:
                lo_i, lo = i, l[i]
            elif h[i] >= lo * (1 + thr):
                piv.append((lo_i, lo, "T")); direction = 1; hi_i, hi = i, h[i]
    piv.append((hi_i, hi, "P") if direction >= 0 else (lo_i, lo, "T"))
    return piv


# ----------------------------------------------------------------- base detection
def base_at(h, l, c, v, e):
    """Evaluate a VCP-type base using bars [..e]. Returns dict or None."""
    if e < 60:
        return None
    s = max(0, e - B["lookback"] + 1)
    ip = s + int(np.argmax(h[s:e + 1]))
    if e - ip < B["min_len"] or e - ip > B["max_len"]:
        return None
    peak = h[ip]
    it = ip + int(np.argmin(l[ip:e + 1]))
    low = l[it]
    depth = (peak - low) / peak
    if depth < B["depth_min"] or depth > B["depth_max"]:
        return None
    irh = it + int(np.argmax(h[it:e + 1]))
    rh = h[irh]
    rim_gap = (peak - rh) / peak
    hl = float(l[irh:e + 1].min())
    handle_len = e - irh
    handle_depth = (rh - hl) / rh
    pos = (hl - low) / (peak - low) if peak > low else 1.0
    # 杯柄(handle 喺杯上三分一)要求反彈到近左頂;cheat 容許較闊
    gap_max = B.get("rim_gap_cup_max", B["rim_gap_max"]) if pos >= 0.667 else B["rim_gap_max"]
    if rim_gap > gap_max:
        return None
    if pos < B["handle_pos_min"] or handle_depth > B["handle_depth_max"]:
        return None
    # contractions (down-legs after the left peak)
    zz = zigzag(h[ip:e + 1], l[ip:e + 1], B["zigzag_pct"])
    legs = []
    for a, b in zip(zz[:-1], zz[1:]):
        if a[2] == "P" and b[2] == "T":
            legs.append(round((a[1] - b[1]) / a[1] * 100, 1))
    if handle_depth * 100 < B["zigzag_pct"] * 100 and handle_len >= 3:
        legs.append(round(handle_depth * 100, 1))
    tail = 1
    for i in range(len(legs) - 1, 0, -1):
        if legs[i] <= legs[i - 1] * 1.15:
            tail += 1
        else:
            break
    tail = tail if legs else 0
    dec = len(legs) >= 2 and legs[-1] <= max(legs) * 0.6 and legs[-1] <= 9.0 and legs[-1] <= float(np.median(legs)) * 1.05
    vol_dry = float(np.mean(v[e - 9:e + 1]) / max(1.0, np.mean(v[max(0, e - 49):e + 1])))
    lo_adv = float(np.min(l[max(0, ip - 120):ip + 1]))
    prior_adv = peak / lo_adv - 1 if lo_adv > 0 else 0
    atr10 = float(np.mean((h[e - 9:e + 1] - l[e - 9:e + 1]) / c[e - 9:e + 1]))
    if pos >= 0.667:
        kind = "杯柄/高位整固 (Cup&Handle)"
    elif pos >= 0.333:
        kind = "中位 cheat"
    else:
        kind = "低位 cheat (成功率較低)"
    if depth < 0.15 and e - ip >= 20:
        kind = "平底整固 (Flat base)" if pos >= 0.667 else kind
    ihl = irh + int(np.argmin(l[irh:e + 1]))
    zzp = [(ip + a, float(b_), t_) for a, b_, t_ in zz]
    return dict(geo=dict(peak_i=int(ip), low_i=int(it), rim_i=int(irh), hl_i=int(ihl), zz=zzp),
                pivot=float(rh), peak=float(peak), low=float(low), depth=round(depth * 100, 1),
                length=int(e - ip), handle_len=int(handle_len), handle_depth=round(handle_depth * 100, 1),
                handle_low=hl, pos=round(pos, 2), legs=legs[-6:], n_legs=len(legs), tail=int(tail), decreasing=bool(dec),
                vol_dry=round(vol_dry, 2), prior_adv=round(prior_adv * 100, 0), atr10=round(atr10 * 100, 1),
                kind=kind, peak_i=int(ip), rim_gap=round(rim_gap * 100, 1))


def analyze(t, d, rs_raw_all, meta, mkt):
    o, h, l, c, v = (d[k].to_numpy(float) for k in ("Open", "High", "Low", "Close", "Volume"))
    n = len(c)
    if n < 210:
        return None
    t_i = n - 1
    s10, s20, s50, s150, s200 = sma(c, 10), sma(c, 20), sma(c, 50), sma(c, 150), sma(c, 200)
    hi52 = float(h[-252:].max()); lo52 = float(l[-252:].min())
    last = float(c[-1])
    avg_dv = float(np.mean(c[-50:] * v[-50:]))
    # trend template (course: 8 criteria)
    tt = {
        "c1": bool(last > s150[-1] and last > s200[-1]),
        "c2": bool(s150[-1] > s200[-1]),
        "c3": bool(s200[-1] > s200[-1 - T["ma200_up_days"]]),
        "c4": bool(s50[-1] > s150[-1] and s50[-1] > s200[-1]),
        "c5": bool(last >= hi52 * (1 - T["within_52w_high"])),
        "c6": bool(last >= lo52 * (1 + T["above_52w_low"])),
        "c7": None,  # RS filled later
        "c8": bool(last > s50[-1]),
    }
    # Stage 3 (見頂):已經大升過 + 150 日線走平(升幅 < flat_150_pct / 21 日)
    flat150 = bool(s150[-1] / s150[-22] - 1 < T.get("stage3_flat_150_pct", 0.01))
    ran = bool(hi52 / lo52 - 1 >= T.get("stage3_min_run", 0.30))
    stage3 = bool(T.get("exclude_stage3", True) and flat150 and ran)
    return dict(ticker=t, o=o, h=h, l=l, c=c, v=v, s10=s10, s20=s20, s50=s50, s150=s150, s200=s200,
               hi52=hi52, lo52=lo52, last=last, avg_dv=avg_dv, tt=tt, n=n, meta=meta, mkt=mkt, stage3=stage3)


def event_scan(x):
    """Find latest base-breakout (<= scan_days_back) or evaluate the pending base."""
    h, l, c, v = x["h"], x["l"], x["c"], x["v"]
    n = x["n"]
    best = None
    for k in range(0, E["scan_days_back"] + 1):
        e = n - 2 - k  # base ends the day before the breakout day
        base = base_at(h, l, c, v, e)
        if not base:
            continue
        bi = n - 1 - k
        if c[bi] > base["pivot"] and c[bi - 1] <= base["pivot"] * 1.0005:
            vr = float(v[bi] / max(1.0, np.mean(v[max(0, bi - 50):bi])))
            best = dict(k=k, base=base, vol_ratio=round(vr, 2), bi=bi)
            break
    if best:
        return best
    base = base_at(h, l, c, v, n - 1)
    if base:
        return dict(k=None, base=base, vol_ratio=round(float(v[-1] / max(1.0, np.mean(v[-51:-1]))), 2), bi=None)
    return None


def build_stock(x, rs, ev):
    base = ev["base"]
    c, h, l, v = x["c"], x["h"], x["l"], x["v"]
    last = x["last"]
    n = x["n"]
    pivot = base["pivot"]
    stop = base["handle_low"] * (1 - E["stop_below_handle_low"])
    entry = pivot * 1.0015
    k = ev["k"]
    if k is None:
        dist = (pivot / last - 1)  # +ve = below pivot
        if dist <= E["near_pivot_pct"]:
            status, code = "近買點 (≤3% 到突破位)", "NEAR"
        elif dist <= E["forming_pct"]:
            status, code = "成形中 (等收縮完成)", "FORM"
        else:
            return None
        ref = entry
    else:
        ext = last / pivot - 1
        if last < pivot * 0.97:
            return None
        if k == 0:
            if ev["vol_ratio"] >= E["breakout_vol_ratio"]:
                status, code = "今日突破 ✔量足", "BO0"
            else:
                status, code = "今日突破但量未夠 (觀察/等跟進)", "BOW"
        else:
            status, code = f"突破後第{k}日", "BOK"
        if ext > E["chase_limit_pct"]:
            status = f"已離買點 +{ext*100:.1f}% (勿追)"
            code = "EXT"
        dist = -ext
        ref = last
    stop_adj = False
    if (ref - stop) / ref < E["min_stop_pct"]:
        stop = ref * (1 - E["min_stop_pct"]); stop_adj = True
    risk_pct = (ref - stop) / ref
    stop_src = "handle 低位"
    no_entry = False
    if risk_pct > E["max_stop_pct"]:
        # 教學:止蝕過闊時,喺 10% 內搵其他技術支持(10/20 日線);搵唔到就唔入
        for nm, key in (("10 日線", "s10"), ("20 日線", "s20")):
            ma_v = float(x[key][-1])
            st = ma_v * (1 - E["stop_below_handle_low"])
            rp = (ref - st) / ref
            if st < ref and E["min_stop_pct"] <= rp <= E["max_stop_pct"]:
                stop, risk_pct, stop_src = st, rp, nm
                break
        else:
            no_entry = True
    r1 = ref * (1 + risk_pct * 1.5); r2 = ref * (1 + risk_pct * 2); r3 = ref * (1 + risk_pct * 3)
    # 左邊阻力:近一年、現價之上最近一個局部高位(教學:有明顯阻力就用嚟決定止賺位)
    resist = None
    hh = h[-252:-3]
    for i in range(5, len(hh) - 5):
        if hh[i] >= max(hh[i - 5:i + 6]) and hh[i] > ref * (1 + risk_pct * 1.0):
            if resist is None or hh[i] < resist:
                resist = float(hh[i])
    resist = round(resist, 2) if resist else None
    # late stage: count breakouts to new 52w highs over past year
    zz = zigzag(h[-252:], l[-252:], 0.10)
    peaks = [p for p in zz if p[2] == "P"]
    mx = 0; nbases = 0
    for p in peaks:
        if p[1] >= mx * 0.995:
            nbases += 1
        mx = max(mx, p[1])
    nbases = max(0, nbases - 1)
    r10 = last / float(np.min(c[-11:-1])) - 1
    climax = bool(r10 >= 0.25 and last / x["s50"][-1] - 1 >= 0.30)
    # quality score 0..100
    tt_n = sum(1 for kk, vv in x["tt"].items() if vv)
    score = tt_n / 8 * 30
    score += 18 if (base["decreasing"] and base["tail"] >= 3) else (14 if base["decreasing"] else (8 if base["tail"] >= 2 else 0))
    score += max(0, 14 - base["handle_depth"]) / 14 * 10
    score += max(0, min(1.0, (1.0 - base["vol_dry"]) / 0.5)) * 10
    score += min(10, max(0, (rs - 60) / 4))
    score += 8 if base["prior_adv"] >= CFG["base"]["prior_adv_min"] * 100 else 0
    score += 6 if (base["pos"] >= 0.667) else 3
    score += 8 if (code in ("NEAR", "BO0")) else 2
    score = round(min(100, score), 0)
    # chart payload
    N = 220
    idx = list(range(max(0, n - N), n))
    dates = x["dates"][idx[0]:]
    bars = [[str(dates[i - idx[0]].date()), round(float(x["o"][i]), 2), round(float(x["h"][i]), 2),
             round(float(x["l"][i]), 2), round(float(c[i]), 2), int(v[i])] for i in idx]
    gd = lambda i: str(x["dates"][i].date())
    g = base["geo"]
    geo = dict(peak=[gd(g["peak_i"]), round(float(h[g["peak_i"]]), 2)],
               low=[gd(g["low_i"]), round(float(l[g["low_i"]]), 2)],
               rim=[gd(g["rim_i"]), round(float(h[g["rim_i"]]), 2)],
               hl=[gd(g["hl_i"]), round(float(l[g["hl_i"]]), 2)],
               zz=[[gd(i), round(p_, 2), t_] for i, p_, t_ in g["zz"]],
               breakout=gd(ev["bi"]) if ev.get("bi") is not None else None)
    ma = {nm: [None if np.isnan(x[nm][i]) else round(float(x[nm][i]), 2) for i in idx]
          for nm in ("s20", "s50", "s150", "s200")}
    return dict(
        ticker=x["ticker"], market=x["mkt"], name=x["meta"].get("name", ""), sector=x["meta"].get("sector", ""),
        group=x["meta"].get("group", ""), tags=x["meta"].get("tags", []),
        close=round(last, 2), pivot=round(pivot, 2), entry=round(entry, 2), stop=round(stop, 2),
        risk_pct=round(risk_pct * 100, 1), t15=round(r1, 2), t2=round(r2, 2), t3=round(r3, 2), resist=resist,
        dist_pct=round(dist * 100, 1), status=status, code=code, k=k, vol_ratio=ev["vol_ratio"],
        rs=int(rs), tt=x["tt"], tt_n=tt_n, base={kk: vv for kk, vv in base.items() if kk not in ("peak_i", "geo")}, geo=geo,
        score=int(score), bases=int(nbases), climax=climax, stop_wide=bool(risk_pct > E["max_stop_pct"]), stop_adj=stop_adj, stop_src=stop_src, no_entry=no_entry,
        hi52=round(x["hi52"], 2), pct_from_hi=round((last / x["hi52"] - 1) * 100, 1),
        avg_dollar_vol=int(x["avg_dv"]), bars=bars, ma=ma)


def build_s2_only(x, rs):
    tt_n = sum(1 for vv in x["tt"].values() if vv)
    last = x["last"]
    n = x["n"]
    N = 220
    idx = list(range(max(0, n - N), n))
    dates = x["dates"][idx[0]:]
    bars = [[str(dates[i - idx[0]].date()), round(float(x["o"][i]), 2), round(float(x["h"][i]), 2),
             round(float(x["l"][i]), 2), round(float(x["c"][i]), 2), int(x["v"][i])] for i in idx]
    ma = {nm: [None if np.isnan(x[nm][i]) else round(float(x[nm][i]), 2) for i in idx]
          for nm in ("s20", "s50", "s150", "s200")}
    return dict(
        ticker=x["ticker"], market=x["mkt"], name=x["meta"].get("name", ""), sector=x["meta"].get("sector", ""),
        group=x["meta"].get("group", ""), tags=x["meta"].get("tags", []), close=round(last, 2), pivot=None,
        entry=None, stop=None, risk_pct=None, t15=None, t2=None, t3=None, resist=None,
        dist_pct=round((x["hi52"] / last - 1) * 100, 1), status="Stage 2 領導股 (未成形,放入 watchlist)",
        code="S2", k=None, vol_ratio=round(float(x["v"][-1] / max(1.0, np.mean(x["v"][-51:-1]))), 2),
        rs=int(rs), tt=x["tt"], tt_n=tt_n, base=None, score=int(tt_n / 8 * 40 + max(0, rs - 60) * 1.2),
        bases=0, climax=False, stop_wide=False, hi52=round(x["hi52"], 2),
        pct_from_hi=round((last / x["hi52"] - 1) * 100, 1), avg_dollar_vol=int(x["avg_dv"]), bars=bars, ma=ma)


# ----------------------------------------------------------------- fundamentals (finalists only)
def fundamentals(tickers):
    def one(t):
        r = {}
        try:
            tk = yf.Ticker(t)
            info = tk.info or {}
            r["mcap"] = info.get("marketCap")
            r["sales_growth"] = info.get("revenueGrowth")
            r["earn_growth"] = info.get("earningsGrowth")
            r["long_name"] = info.get("longName") or info.get("shortName")
            r["yf_sector"] = info.get("sector")
            r["yf_industry"] = info.get("industry")
            try:
                cal = tk.calendar
                ed = None
                if isinstance(cal, dict):
                    ed = cal.get("Earnings Date")
                    if isinstance(ed, (list, tuple)) and ed:
                        ed = ed[0]
                if ed is not None:
                    r["earnings"] = str(pd.to_datetime(ed).date())
            except Exception:
                pass
        except Exception:
            pass
        return t, r
    out = {}
    with ThreadPoolExecutor(8) as ex:
        for t, r in ex.map(one, tickers):
            out[t] = r
    return out


# ----------------------------------------------------------------- macro / regime
def trend_state(d):
    c = d["Close"].dropna().to_numpy(float)
    if len(c) < 60:
        return None
    s50 = pd.Series(c).rolling(50).mean().iloc[-1]
    ret1 = c[-1] / c[-22] - 1 if len(c) > 22 else None
    ret3 = c[-1] / c[-64] - 1 if len(c) > 64 else None
    return dict(last=round(float(c[-1]), 2), ret1m=round(float(ret1) * 100, 1) if ret1 is not None else None,
                ret3m=round(float(ret3) * 100, 1) if ret3 is not None else None,
                above50=bool(c[-1] > s50), chg1m_abs=round(float(c[-1] - c[-22]), 2) if len(c) > 22 else None)


def main():
    t0 = time.time()
    markets = CFG.get("markets", ["US"])
    uni = [u for u in universe.load() if u["market"] in markets]
    meta = {u["ticker"]: u for u in uni}
    print(f"universe: {len(uni)} tickers  markets={markets}")

    bench = ["SPY", "QQQ", "IWM", "^GSPC", "^IXIC", "^VIX"] + [d["id"] for d in MM["drivers"] if d["id"] not in ("^VIX",)]
    etfs = list(MM["sector_etfs"].keys())
    if "HK" in markets:
        bench += ["^HSI", "000300.SS"]
    print("downloading prices ...")
    allp = fetch(sorted(set(meta) | set(bench) | set(etfs)), CFG["history_period"])
    print(f"  got {len(allp)} series in {time.time()-t0:.0f}s")

    # ---- per ticker base metrics + RS raw
    xs = {}
    rs_raw = {}
    for t, d in allp.items():
        if t not in meta:
            continue
        mkt = meta[t]["market"]
        lq = CFG["liquidity"][mkt]
        c = d["Close"].to_numpy(float)
        if len(c) < 210:
            continue
        x = analyze(t, d, None, meta[t], mkt)
        if not x:
            continue
        x["dates"] = d.index
        if x["last"] < lq["min_price"] or x["avg_dv"] < lq["min_avg_dollar_vol"]:
            continue
        def r(nb):
            return c[-1] / c[-1 - nb] - 1 if len(c) > nb else np.nan
        raw = 0.4 * r(63) + 0.2 * r(126) + 0.2 * r(189) + 0.2 * r(251)
        if np.isnan(raw):
            continue
        rs_raw[t] = raw
        xs[t] = x
    ser = pd.Series(rs_raw)
    rs_pct = (ser.rank(pct=True) * 98 + 1).round().astype(int)
    print(f"  liquid universe: {len(xs)}")

    for t, x in xs.items():
        x["tt"]["c7"] = bool(rs_pct[t] >= T["min_rs"])

    # ---- market health (breadth)
    stage2 = [t for t, x in xs.items() if all(x["tt"].values())]
    above50 = sum(1 for x in xs.values() if x["tt"]["c8"])
    new_hi = sum(1 for x in xs.values() if x["last"] >= x["hi52"] * 0.98)
    new_lo = sum(1 for x in xs.values() if x["last"] <= x["lo52"] * 1.02)
    idx_state = {}
    for k in ("SPY", "QQQ", "IWM"):
        if k in allp:
            c = allp[k]["Close"].to_numpy(float)
            s50 = pd.Series(c).rolling(50).mean().iloc[-1]; s200 = pd.Series(c).rolling(200).mean().iloc[-1]
            idx_state[k] = dict(last=round(float(c[-1]), 2), above50=bool(c[-1] > s50), above200=bool(c[-1] > s200),
                                golden=bool(s50 > s200), ret1m=round(float(c[-1] / c[-22] - 1) * 100, 1))
    pts = 0; reasons = []
    for k, st in idx_state.items():
        if st["above50"]:
            pts += 1; reasons.append(f"{k} 在 50 日線上")
        else:
            reasons.append(f"{k} 跌穿 50 日線")
        if k != "IWM":
            pts += 1 if st["above200"] else 0
    b50 = above50 / max(1, len(xs)); s2p = len(stage2) / max(1, len(xs))
    if b50 >= 0.5: pts += 2
    elif b50 >= 0.35: pts += 1
    if s2p >= 0.15: pts += 1
    if new_hi > new_lo: pts += 1
    reasons += [f"{b50*100:.0f}% 股票在 50 日線上", f"{s2p*100:.0f}% 股票符合 Stage 2",
                f"近 52 週新高 {new_hi} / 新低 {new_lo}"]
    if pts >= 7:
        light, advice = "green", "進取:可用方法①(一穿阻力即買),正常注碼;總風險可到 10R。"
    elif pts >= 4:
        light, advice = "yellow", "中性:優先用方法②(收市前確認突破先買)或方法③(回踩再升);總風險建議 ≤ 6R,揀最靚嘅。"
    else:
        light, advice = "red", "防守:少入場,只做最靚嘅;用方法③(回踩確認)、注碼減半,總風險 ≤ 3R。"
    regime = dict(light=light, points=pts, max_points=9, advice=advice, reasons=reasons, index=idx_state,
                  breadth=dict(universe=len(xs), above50_pct=round(b50 * 100, 1), stage2=len(stage2),
                               stage2_pct=round(s2p * 100, 1), new_highs=new_hi, new_lows=new_lo))

    # ---- groups
    gr = {}
    for t, x in xs.items():
        g = x["meta"].get("group") or x["meta"].get("sector") or "其他"
        gr.setdefault(g, []).append(t)
    groups = []
    for g, ts in gr.items():
        if len(ts) < 3:
            continue
        rs_list = [int(rs_pct[t]) for t in ts]
        s2 = [t for t in ts if all(xs[t]["tt"].values())]
        groups.append(dict(group=g, sector=xs[ts[0]]["meta"].get("sector", ""), n=len(ts), med_rs=int(np.median(rs_list)),
                           stage2_pct=round(len(s2) / len(ts) * 100), leaders=sorted(
                               ts, key=lambda q: -int(rs_pct[q]))[:5]))
    sec = {}
    for t, x in xs.items():
        sec.setdefault(x["meta"].get("sector") or "其他", []).append(t)
    sectors = []
    for sname, ts in sec.items():
        rs_list = [int(rs_pct[t]) for t in ts]
        s2 = [t for t in ts if all(xs[t]["tt"].values())]
        sectors.append(dict(group=sname, sector=sname, n=len(ts), med_rs=int(np.median(rs_list)),
                            stage2_pct=round(len(s2) / len(ts) * 100)))
    sectors.sort(key=lambda g: (-g["med_rs"], -g["stage2_pct"]))
    for i, g in enumerate(sectors):
        g["rank"] = i + 1; g["pctile"] = round((1 - i / max(1, len(sectors))) * 100)
    smap = {g["group"]: g for g in sectors}
    groups.sort(key=lambda g: (-g["med_rs"], -g["stage2_pct"]))
    for i, g in enumerate(groups):
        g["rank"] = i + 1
        g["pctile"] = round((1 - i / max(1, len(groups))) * 100)
    gmap = {g["group"]: g for g in groups}

    # ---- candidates
    cands = []
    s2_only = []
    n_stage3 = 0
    for t, x in xs.items():
        tt_n = sum(1 for vv in x["tt"].values() if vv)
        if tt_n < CFG["output"]["min_tt_n"]:
            continue
        if x["stage3"]:
            n_stage3 += 1
            continue
        rs = int(rs_pct[t])
        ev = event_scan(x)
        s = build_stock(x, rs, ev) if ev else None
        if s:
            cands.append(s)
        elif all(x["tt"].values()) and x["last"] >= x["hi52"] * 0.85:
            s2_only.append((rs, x))
    s2_only.sort(key=lambda a: -a[0])
    for rs, x in s2_only[:CFG["output"]["max_s2_only"]]:
        cands.append(build_s2_only(x, rs))
    print(f"  candidates: {len(cands)}  (breakout/near/forming: {sum(1 for c in cands if c['code']!='S2')})  · Stage 3 已剔走: {n_stage3}")

    # ---- fundamentals for finalists
    fin = [c["ticker"] for c in cands]
    print(f"fundamentals for {len(fin)} ...")
    fd = fundamentals(fin)
    today = dt.date.today()
    for c in cands:
        f = fd.get(c["ticker"], {})
        c["mcap"] = f.get("mcap"); c["sales_growth"] = f.get("sales_growth")
        if f.get("long_name"): c["name"] = f["long_name"]
        if not c["sector"] and f.get("yf_sector"): c["sector"] = f["yf_sector"]
        if not c["group"] and f.get("yf_industry"): c["group"] = f["yf_industry"]
        c["earnings"] = f.get("earnings")
        c["earn_soon"] = False
        if c["earnings"]:
            try:
                dd = (pd.to_datetime(c["earnings"]).date() - today).days
                if dd < 0:
                    c["earnings"] = None
                else:
                    c["earn_days"] = dd
                    c["earn_soon"] = bool(dd <= 7)
            except Exception:
                pass
        mcap_min = CFG["liquidity"][c["market"]]["min_mcap"]
        c["mcap_ok"] = (c["mcap"] is None) or (c["mcap"] >= mcap_min)
        sg = c["sales_growth"]
        c["sales_ok"] = None if sg is None else bool(sg >= 0.15)
        g = gmap.get(c["group"])
        c["group_level"] = "子行業"
        if not g:
            g = smap.get(c["sector"]); c["group_level"] = "板塊"
        c["group_rank"] = g["rank"] if g else None
        c["group_pctile"] = g["pctile"] if g else None
        c["group_n"] = len(groups) if c["group_level"] == "子行業" else len(sectors)
        c["leader"] = c["group"] and any(c["ticker"] in gg["leaders"][:2] for gg in groups if gg["group"] == c["group"])

    # ---- macro mind map
    macro = []
    state = {}
    for d in MM["drivers"]:
        if d["id"] not in allp:
            continue
        st = trend_state(allp[d["id"]])
        if not st:
            continue
        active = None
        if d["kind"] == "pct":
            if st["ret1m"] is not None and st["ret1m"] / 100 >= d["threshold"] and st["above50"]:
                active = "up"
            elif st["ret1m"] is not None and st["ret1m"] / 100 <= -d["threshold"] and not st["above50"]:
                active = "down"
        else:
            if st["chg1m_abs"] is not None and st["chg1m_abs"] >= d["threshold"] and st["above50"]:
                active = "up"
            elif st["chg1m_abs"] is not None and st["chg1m_abs"] <= -d["threshold"] and not st["above50"]:
                active = "down"
        macro.append(dict(id=d["id"], name=d["name"], **st, active=active,
                          thesis=d[active]["thesis"] if active else None,
                          up=d["up"], down=d["down"]))
        state[d["id"]] = active
    # tag candidates
    def hit(c, spec):
        if c["ticker"] in spec.get("benefit", []):
            return "benefit"
        if c["ticker"] in spec.get("hurt", []):
            return "hurt"
        txt = f"{c['group']} {c['sector']}"
        if any(k.lower() in txt.lower() for k in spec.get("benefit_groups", [])):
            return "benefit"
        if any(k.lower() in txt.lower() for k in spec.get("hurt_groups", [])):
            return "hurt"
        return None
    for c in cands:
        c["macro"] = []
        for m in macro:
            if not m["active"]:
                continue
            h = hit(c, m[m["active"]])
            if h:
                c["macro"].append(dict(driver=m["name"], dir=m["active"], effect=h))

    sector_perf = []
    for tk, nm in MM["sector_etfs"].items():
        if tk in allp:
            st = trend_state(allp[tk])
            if st:
                sector_perf.append(dict(ticker=tk, name=nm, **st))
    sector_perf.sort(key=lambda s: -(s["ret3m"] or -999))

    order = {"BO0": 0, "NEAR": 1, "BOK": 2, "BOW": 3, "FORM": 4, "EXT": 5, "S2": 6}
    cands.sort(key=lambda c: (order.get(c["code"], 9), -c["score"]))
    asof = max(x["dates"][-1] for x in xs.values())
    from zoneinfo import ZoneInfo
    _now = dt.datetime.now(dt.timezone.utc)
    _et = _now.astimezone(ZoneInfo("America/New_York"))
    _intraday = _et.weekday() < 5 and (9, 30) <= (_et.hour, _et.minute) < (16, 0)
    data = dict(generated=_now.astimezone(ZoneInfo("Asia/Hong_Kong")).strftime("%Y-%m-%d %H:%M") + " HKT",
                intraday=_intraday, asof=str(asof.date()),
                config=dict(markets=markets, risk=R, entry=E, liquidity=CFG["liquidity"], trend=T),
                regime=regime, macro=macro, sector_perf=sector_perf, groups=groups[:40], sectors=sectors,
                note_pairs=MM.get("note_pairs", []), stocks=cands)
    mp = os.path.join(HERE, "data", "momentum.json")
    data["momentum"] = json.load(open(mp)) if os.path.exists(mp) else None
    if data["momentum"]:
        try:
            import momentum as _mom
            data["momentum"] = _mom.refresh_prices(data["momentum"])
        except Exception as e:
            print("momentum refresh skipped:", e)
    json.dump(data, open(os.path.join(OUT, "data.json"), "w"), ensure_ascii=False, separators=(",", ":"))
    build_html(data)
    print(f"done in {time.time()-t0:.0f}s  -> out/index.html")


def build_html(data):
    tpl = open(os.path.join(HERE, "template.html"), encoding="utf-8").read()
    lib = open(os.path.join(HERE, "vendor", "lightweight-charts.js"), encoding="utf-8").read()
    html = tpl.replace("/*__LWC__*/", lib).replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    open(os.path.join(OUT, "index.html"), "w", encoding="utf-8").write(html)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "html":
        build_html(json.load(open(os.path.join(OUT, "data.json"))))
        print("html rebuilt")
    else:
        main()
