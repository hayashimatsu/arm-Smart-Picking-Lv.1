#!/usr/bin/env python3
"""active.py [分鐘=30] -- 現在有誰在動這個專案？（只讀；從檔案活動推斷，看不到別台機器的程序）
  1) 最近 N 分鐘被改的檔案，依目錄分組   2) 最近 N 分鐘動過、但沒有 TASK.md 的任務資料夾
  3) 最近 15 分鐘被改的『權威檔』（別碰）   4) Isaac GUI log 與 flock 鎖的狀態
"""
import os, pathlib, sys, time

ROOT = pathlib.Path(__file__).resolve().parent.parent
N = float(sys.argv[1]) if len(sys.argv) > 1 else 30
SKIP = (".venv", "__pycache__", "/.git/", ".pyc", "/frames/")
AUTH = ["CLAUDE.md", "AGENTS.md", "STATUS.md", "README.md", "docs/principles.md", "docs/operations.md", "history/failures.md", "history/improvements.md", "docs/acceptance.json", "docs/protected.globs"]
now = time.time()
age = lambda p: (now - p.stat().st_mtime) / 60

recent = {}
for p in ROOT.rglob("*"):
    s = "/" + p.relative_to(ROOT).as_posix()
    if p.is_file() and not any(k in s for k in SKIP) and age(p) <= N:
        recent.setdefault("/".join(p.relative_to(ROOT).parts[:3]), []).append(p)

print(f"== 最近 {N:g} 分鐘有活動的位置")
for d, ps in sorted(recent.items(), key=lambda kv: -max(x.stat().st_mtime for x in kv[1])):
    last = min(age(x) for x in ps)
    print(f"  {d:55s} {len(ps):3d} 檔，最新 {last:4.1f} 分鐘前")
if not recent:
    print("  （沒有）→ 目前看起來沒有 agent 在寫檔")

print("== 有活動但沒有 TASK.md 的任務資料夾（沒有 spec）")
miss = [d for d in recent if d.startswith("outputs/tests/") and not (ROOT / d / "TASK.md").exists() and len(d.split("/")) == 3]
print("\n".join(f"  {d}" for d in miss) or "  （無）")

print("== 最近 15 分鐘被改過的權威檔（別人可能還在改，先不要編輯）")
hot = [f for f in AUTH if (ROOT / f).exists() and age(ROOT / f) <= 15]
print("\n".join(f"  {f}（{age(ROOT / f):.1f} 分鐘前）" for f in hot) or "  （無）")

print("== Isaac")
logs = sorted(ROOT.glob("outputs/tests/*/gui_launch.log"), key=lambda p: p.stat().st_mtime, reverse=True)
if logs:
    print(f"  最新 GUI log：{logs[0].relative_to(ROOT)}，{age(logs[0]):.1f} 分鐘前寫入 → {'GUI 仍在跑（有人在操作或剛結束）' if age(logs[0]) < 5 else '已安靜'}")
lock = pathlib.Path("/tmp/smart_picking_isaac.lock")
print(f"  flock 鎖檔：{'存在' if lock.exists() else '不存在（在這個沙箱看不到主機的 /tmp，僅供參考）'}")
