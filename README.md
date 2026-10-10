# VCP 入場雷達 (Stage 2 + VCP 趨勢突破篩選)

- `./run.sh`            每日收市後行一次(約 1 分鐘),完成自動開 `out/index.html`
- `config.json`         所有門檻(Stage 2、base、入場、止蝕、1R 比例、流動性、市場 US/HK)
- `mindmap.json`        宏觀因果鏈(driver 升/跌 → 受惠/受壓股票或板塊),可自行增刪
- `universe.py`         股票池(S&P 500/400/600 + 動能股名單;config 加 "HK" 就包埋港股),每 7 日自動更新
- `screener.py`         偵測引擎;`template.html` 網頁模板;`out/data.json` 原始結果
