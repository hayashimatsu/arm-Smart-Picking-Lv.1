#!/usr/bin/env bash
# 回報前跑這支：先證明檢查器自己會失敗（selftest），再檢查真實狀態。
# 用法：tools/run_checks.sh [result.json]   有給 result.json 才檢查驗收門檻
set -euo pipefail
cd "$(dirname "$0")/.."
python3 tools/protect.py --selftest
python3 tools/check_acceptance.py --selftest
python3 tools/lint_project.py --selftest
echo "---- 真實狀態 ----"
python3 tools/protect.py verify
python3 tools/lint_project.py
python3 tools/lint_project.py --orphans   # 放在驗收檢查之前：驗收有缺口時 set -e 會中止腳本，孤兒檢查才不會被擋掉
if [ -n "${1:-}" ]; then python3 tools/check_acceptance.py "$1"; fi
