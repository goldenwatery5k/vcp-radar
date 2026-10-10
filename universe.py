"""Build / cache the screening universe (US S&P500 + S&P400 + Nasdaq-100, HK HSI + HSCEI + HS-Tech-ish list)."""
import io, json, os, re, time
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data", "universe.json")
H = {"User-Agent": "Mozilla/5.0 (Macintosh) AppleWebKit/537.36 Chrome/120 Safari/537.36"}

# HK growth / tech names that sit outside HSI/HSCEI but matter for this style (incl. course examples)
HK_EXTRA = {
    "1810": "Xiaomi", "9618": "JD.com", "9888": "Baidu", "9999": "NetEase", "1024": "Kuaishou",
    "981": "SMIC", "992": "Lenovo", "285": "BYD Electronic", "1211": "BYD", "2015": "Li Auto",
    "9868": "XPeng", "9866": "NIO", "6618": "JD Health", "241": "Alibaba Health", "2013": "Weimob",
    "268": "Kingdee", "522": "ASMPT", "3888": "Kingsoft", "1347": "Hua Hong Semi", "9961": "Trip.com",
    "9626": "Bilibili", "2382": "Sunny Optical", "1833": "Ping An Healthcare", "6690": "Haier Smart Home",
    "780": "Tongcheng", "909": "Ming Yuan Cloud", "1357": "Meitu", "9992": "Pop Mart", "9633": "Nongfu Spring",
    "2020": "Anta", "2331": "Li Ning", "1186": "China Railway Construction", "9923": "Yeahka",
    "1137": "HKTVmall", "2137": "Dunhuang? (course ex.)", "6969": "Smoore", "6060": "ZhongAn",
    "1801": "Innovent Bio", "2269": "WuXi Bio", "1177": "Sino Biopharm", "6160": "BeiGene", "9926": "Akeso",
    "1530": "3SBio", "2359": "WuXi AppTec", "3692": "Hansoh", "1093": "CSPC", "6862": "Haidilao",
    "9987": "Yum China", "2899": "Zijin Mining", "1772": "Ganfeng", "9696": "Tianqi Lithium",
    "3750": "CATL", "1919": "COSCO Shipping", "883": "CNOOC", "857": "PetroChina", "386": "Sinopec",
    "1398": "ICBC", "3968": "CM Bank", "2318": "Ping An", "1299": "AIA", "388": "HKEX", "5": "HSBC",
}


# US momentum / growth / ADR names that are not in the S&P indices (course examples included)
US_EXTRA = {t: t for t in ("ARM ASML PDD MELI TEAM ZS DDOG SNOW NET MDB S PATH HIMS RKLB IONQ OKLO RDDT CELH DUOL TOST "
    "IOT SHOP SE NU SPOT COIN MSTR MARA RIOT CLSK HUT HOOD SOFI AFRM UPST LMND DKNG RBLX U RIVN LCID NIO XPEV LI BABA JD "
    "BIDU TSM NVO ONON CAVA BROS WING ELF CART KVYO DOCS ALAB CRDO VRT SMCI APP AXON CRWD PANW FTNT TTD ENPH FSLR NXT "
    "ASTS LUNR JOBY ACHR SOUN QBTS RGTI TEM CRCL HOOD GLXY BTDR CIFR WULF IREN CORZ").split()}


def _tables(url):
    r = requests.get(url, headers=H, timeout=40)
    r.raise_for_status()
    return pd.read_html(io.StringIO(r.text))


def _us_symbol(s):
    return str(s).strip().replace(".", "-")


def _hk_symbol(s):
    m = re.search(r"(\d+)", str(s))
    return f"{int(m.group(1)):04d}.HK" if m else None


def build():
    rows = {}

    def add(t, name, market, sector, group, tag):
        if not t or t == "nan":
            return
        r = rows.setdefault(t, dict(ticker=t, name=name, market=market, sector=sector, group=group, tags=[]))
        if tag not in r["tags"]:
            r["tags"].append(tag)
        if sector and not r.get("sector"):
            r["sector"] = sector
        if group and not r.get("group"):
            r["group"] = group

    for tag, url in [("S&P500", "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"),
                     ("S&P400", "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"),
                     ("S&P600", "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies")]:
        try:
            df = _tables(url)[0]
            for _, x in df.iterrows():
                add(_us_symbol(x["Symbol"]), x["Security"], "US", x["GICS Sector"], x["GICS Sub-Industry"], tag)
        except Exception as e:
            print("WARN", tag, e)

    try:
        for t in _tables("https://en.wikipedia.org/wiki/List_of_Nasdaq-100_companies"):
            cols = [str(c) for c in t.columns]
            if any(c.lower().startswith("ticker") for c in cols) and len(t) > 80:
                tc = [c for c in cols if c.lower().startswith("ticker")][0]
                nc = [c for c in cols if c.lower().startswith("company")][0]
                sc = [c for c in cols if "sector" in c.lower()]
                gc = [c for c in cols if "industry" in c.lower()]
                t.columns = cols
                for _, x in t.iterrows():
                    add(_us_symbol(x[tc]), x[nc], "US", x[sc[0]] if sc else "", x[gc[0]] if gc else "", "NDX100")
                break
    except Exception as e:
        print("WARN NDX", e)

    try:
        t = _tables("https://en.wikipedia.org/wiki/Hang_Seng_Index")
        t = [x for x in t if "Sub-index" in [str(c) for c in x.columns]][0]
        for _, x in t.iterrows():
            add(_hk_symbol(x["Ticker"]), x["Name"], "HK", x["Sub-index"], x["Sub-index"], "HSI")
    except Exception as e:
        print("WARN HSI", e)
    try:
        t = _tables("https://en.wikipedia.org/wiki/Hang_Seng_China_Enterprises_Index")
        t = [x for x in t if "Industry" in [str(c) for c in x.columns]][0]
        for _, x in t.iterrows():
            add(_hk_symbol(x["Ticker"]), x["Name"], "HK", x["Industry"], x["Industry"], "HSCEI")
    except Exception as e:
        print("WARN HSCEI", e)
    for t, nm in US_EXTRA.items():
        add(t, nm, "US", "", "", "US-extra")
    for code, nm in HK_EXTRA.items():
        add(f"{int(code):04d}.HK", nm, "HK", "", "", "HK-extra")

    return list(rows.values())


def load(refresh_days=7):
    if os.path.exists(CACHE) and (time.time() - os.path.getmtime(CACHE)) < refresh_days * 86400:
        return json.load(open(CACHE))
    u = build()
    if len(u) > 300:
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        json.dump(u, open(CACHE, "w"), ensure_ascii=False)
    elif os.path.exists(CACHE):
        return json.load(open(CACHE))
    return u


if __name__ == "__main__":
    u = load(0)
    us = [x for x in u if x["market"] == "US"]
    hk = [x for x in u if x["market"] == "HK"]
    print(len(u), "total;", len(us), "US;", len(hk), "HK")
