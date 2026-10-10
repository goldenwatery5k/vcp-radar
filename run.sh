#!/bin/zsh
# 每日收市後行一次:下載價量 → 篩選 → 生成 out/index.html 並用瀏覽器打開
cd "$(dirname "$0")"
.venv/bin/python screener.py && open out/index.html
