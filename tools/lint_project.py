#!/usr/bin/env python3
"""lint_project.py -- 掃自己的程式有沒有「寫死到別的專案」或「指向已不存在的場景」。
只是警告（不擋任務，因為比 Lin 嚴的檢查只記錄）；--strict 才讓警告變成失敗。
排除：.venv、outputs、history、reference、yolo/runs、.git。n=0 視為 FAIL。
  --selftest  證明它抓得到壞路徑、乾淨的樹不誤報、空樹會 FAIL；孤兒檢查同樣各證一次。
  --orphans   孤兒檔：lab／grasp／perception／scripts 裡 GUI 入口 demo_start.py 載不到的 .py → FAIL；
              列在 docs/orphan_allow.txt（Lin 維護，移植期「待接上」的檔）的只 WARN。n=0 或入口不存在 → FAIL。
              「載得到」＝從入口靜態追 exec/open 的檔名字串與 import（入口用 exec(open(...)) 載入，只看 import 會漏）。
"""
import os, pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SKIP = (".venv", "site-packages", "__pycache__", "/.git/", "/outputs/", "/history/", "/reference/", "/yolo/runs/", "/tools/")
ABS = re.compile(r"""/home/[A-Za-z0-9_.-]+/[^\s"'`)\]]*""")
STALE = "arm310d_d455_ik_demo"


def inside(path, root):
    """屬於本專案？沙箱掛載路徑和主機路徑不同，所以用『資料夾名稱（完整一層）』判斷，而不是整串前綴。"""
    return path.startswith(str(root)) or re.search(r"/" + re.escape(root.name) + r"(/|$)", path) is not None


def scan(root):
    root = pathlib.Path(root).resolve()
    files, warns = 0, []
    for p in root.rglob("*"):
        s = "/" + p.relative_to(root).as_posix()
        if not p.is_file() or any(k in s + "/" for k in SKIP) or p.suffix not in (".py", ".sh", ".json"):
            continue
        files += 1
        text = p.read_text(encoding="utf-8", errors="ignore")
        for m in ABS.finditer(text):
            if not inside(m.group(0), root):
                warns.append(f"{p.relative_to(root)}: 指向專案外的絕對路徑 {m.group(0)[:90]}")
                break
        if p.suffix == ".py" and STALE in text:
            warns.append(f"{p.relative_to(root)}: 還寫著舊場景名 {STALE}")
    return files, warns


ENTRY = "demo_start.py"
PRODUCT = ("lab", "grasp", "perception", "scripts")
ALLOW = "docs/orphan_allow.txt"
REACH_SKIP = (".venv", "site-packages", "__pycache__", ".git", "outputs", "history", "reference", "yolo", "tools")


def reachable(root, entry=ENTRY):
    """Project .py files the entry can load (ported from outputs/tests/20261108_reachability/reach.py)."""
    root = pathlib.Path(root).resolve()
    files = set()
    for d, dirs, fs in os.walk(root):
        dirs[:] = [x for x in dirs if x not in REACH_SKIP]
        files |= {(pathlib.Path(d) / f).relative_to(root).as_posix() for f in fs if f.endswith(".py")}
    byname = {}
    for f in files:
        byname.setdefault(f.rsplit("/", 1)[-1], []).append(f)

    def refs(f):
        src = (root / f).read_text(encoding="utf-8", errors="ignore")
        out = set()
        for m in re.finditer(r'["\']([\w./-]*?)([\w-]+\.py)["\']', src):
            cand = byname.get(m.group(2), [])
            dirs = [x for x in m.group(1).split("/") if x and x not in (".", "..")]
            if dirs:
                cand = [p for p in cand if p.split("/")[-2:-1] == dirs[-1:]] or cand
            if len(cand) > 1:   # same name in several folders: prefer the folder of f
                cand = [p for p in cand if os.path.dirname(p) == os.path.dirname(f)] or cand
            out |= set(cand)
        for m in re.finditer(r'join\([^)]*?"([\w-]+)"\s*,\s*"([\w-]+\.py)"', src):
            p = f"{m.group(1)}/{m.group(2)}"
            if p in files:
                out.add(p)
        for m in re.finditer(r'^\s*(?:from|import)\s+([\w.]+)', src, re.M):
            p = m.group(1).replace(".", "/") + ".py"
            if p in files:
                out.add(p)
        return out - {f}

    seen, todo = set(), [entry]
    while todo:
        f = todo.pop()
        if f in seen or f not in files:
            continue
        seen.add(f); todo += list(refs(f))
    return files, seen


def orphans(root):
    """-> (status, n product .py, fail list, warn list)."""
    root = pathlib.Path(root).resolve()
    if not (root / ENTRY).is_file():
        return "FAIL", 0, [f"入口 {ENTRY} 不存在"], []
    files, seen = reachable(root)
    prod = sorted(f for f in files if f.split("/")[0] in PRODUCT)
    allow = set()
    if (root / ALLOW).is_file():
        for line in (root / ALLOW).read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                allow.add(line.split()[0])
    lost = [f for f in prod if f not in seen]
    fail = [f"{f}: 入口載不到（封存到 history/archive/，或請 Lin 列入 {ALLOW}）" for f in lost if f not in allow]
    warn = [f"{f}: 入口載不到，在容許清單內（待接上）" for f in lost if f in allow]
    status = "FAIL" if not prod or fail else ("WARN" if warn else "PASS")
    return status, len(prod), fail, warn


def orphan_selftest():
    import tempfile
    bad = 0
    with tempfile.TemporaryDirectory() as td:
        r = pathlib.Path(td); (r / "lab").mkdir(); (r / "docs").mkdir()
        (r / ENTRY).write_text('exec(open(os.path.join(_root, "lab", "api2.py")).read())\n')
        (r / "lab" / "api2.py").write_text('exec(open("live_pick.py").read())\n')
        (r / "lab" / "live_pick.py").write_text("print('hi')\n")
        s, n, f, w = orphans(r)
        good = s == "PASS" and n == 2
        bad += not good; print(f"  selftest {'ok ' if good else 'BAD'}: 全部載得到 → PASS（n={n}）")
        (r / "lab" / "stray.py").write_text("print('orphan')\n")
        s, n, f, w = orphans(r)
        good = s == "FAIL" and len(f) == 1 and "stray.py" in f[0]
        bad += not good; print(f"  selftest {'ok ' if good else 'BAD'}: 故意放的孤兒檔 → FAIL")
        (r / ALLOW).write_text("# 說明\nlab/stray.py   # 待接上：TASK.md 某任務\n")
        s, n, f, w = orphans(r)
        good = s == "WARN" and not f and len(w) == 1
        bad += not good; print(f"  selftest {'ok ' if good else 'BAD'}: 容許清單內的孤兒檔 → 只 WARN")
    with tempfile.TemporaryDirectory() as td:
        s, n, f, w = orphans(td)
        good = s == "FAIL" and n == 0
        bad += not good; print(f"  selftest {'ok ' if good else 'BAD'}: 空樹 → FAIL")
    return bad


def selftest():
    import tempfile
    bad = orphan_selftest()
    with tempfile.TemporaryDirectory() as td:
        r = pathlib.Path(td)
        (r / "ok.py").write_text("print('hi')\n")
        n, w = scan(r)
        good = n == 1 and not w
        bad += not good; print(f"  selftest {'ok ' if good else 'BAD'}: 乾淨的樹不誤報（n={n}）")
        (r / "bad.py").write_text('ROOT = "/home/someone/other-project/x"\n')
        n, w = scan(r)
        good = n == 2 and len(w) == 1
        bad += not good; print(f"  selftest {'ok ' if good else 'BAD'}: 抓得到專案外的絕對路徑")
    with tempfile.TemporaryDirectory() as td:
        n, w = scan(td)
        good = n == 0
        bad += not good; print(f"  selftest {'ok ' if good else 'BAD'}: 空樹 n=0 會被視為 FAIL")
    print(f"CHECK lint_selftest n=7 {'PASS' if not bad else 'FAIL'}")
    return not bad


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    if "--orphans" in sys.argv:
        status, n, fail, warn = orphans(ROOT)
        print(f"CHECK orphans n={n} fail={len(fail)} allowed={len(warn)} {status}")
        for w in fail + warn:
            print("   -", w)
        sys.exit(1 if status == "FAIL" else 0)
    n, warns = scan(ROOT)
    status = "FAIL" if n == 0 or (warns and "--strict" in sys.argv) else ("WARN" if warns else "PASS")
    print(f"CHECK lint n={n} warnings={len(warns)} {status}")
    for w in warns:
        print("   -", w)
    sys.exit(1 if status == "FAIL" else 0)
