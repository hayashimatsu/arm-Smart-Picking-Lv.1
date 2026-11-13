#!/usr/bin/env bash
# 建立「工作副本」：GUI 要開、要存檔的是副本，正本 scenes/lab_cell_g1.usda 永遠不被改。
# 副本放在同一個資料夾（正本用相對路徑引用 ./nhc-d455-ik.usd 和 layers/）。
# 加 --lock 才會把正本設成唯讀（等沒有 agent 正在使用 GUI 時再做；可用 chmod u+w 還原）。
set -euo pipefail
cd "$(dirname "$0")/.."
SRC=scenes/lab_cell_g1.usda
DST=scenes/lab_cell_g1_work.usda
[ -f "$SRC" ] || { echo "找不到 $SRC" >&2; exit 1; }
if [ -e "$DST" ]; then
  echo "$DST 已存在，不覆寫；要重建請先自己移走它（覆寫場景是只有 Lin 能做的動作）" >&2
else
  cp "$SRC" "$DST" && echo "已建立 $DST"
fi
if [ "${1:-}" = "--lock" ]; then chmod a-w "$SRC" && echo "已把 $SRC 設為唯讀"; fi
python3 tools/protect.py verify
