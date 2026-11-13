#!/usr/bin/env python3
"""protect.py -- 不可動清單的雜湊檢查（docs/protected.globs -> docs/protected.sha256）。

  build [--force]  產生清單。這是「只有 Lin 能做」的動作：清單已存在時必須 --force。
  verify           逐檔比對。印出 n；n==0、檔案被改、檔案不見 = FAIL。新增的檔案只警告。
  --selftest       用暫存專案證明 verify 在 竄改／缺檔／空清單 時真的會 FAIL。
"""
import fnmatch, glob, hashlib, pathlib, sys, tempfile

GLOBS, MANIFEST = "docs/protected.globs", "docs/protected.sha256"


def read_globs(root):
    inc, exc = [], []
    for line in (root / GLOBS).read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            (exc if s.startswith("!") else inc).append(s.lstrip("!"))
    return inc, exc


def expand(root, inc, exc):
    files = set()
    for pat in inc:
        for p in glob.glob(str(root / pat), recursive=True):
            pp = pathlib.Path(p)
            cands = [q for q in pp.rglob("*") if q.is_file()] if pp.is_dir() else [pp]
            for q in cands:
                rel = q.relative_to(root).as_posix()
                if "__pycache__" in rel or rel.endswith(".pyc"):
                    continue
                files.add(rel)
    return sorted(f for f in files if not any(fnmatch.fnmatch(f, e) for e in exc))


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(root, force=False):
    mf = root / MANIFEST
    if mf.exists() and not force:
        return False, f"{MANIFEST} 已存在；重建是只有 Lin 能做的動作，需要 --force"
    inc, exc = read_globs(root)
    files = expand(root, inc, exc)
    if not files:
        return False, "清單展開後 0 個檔案，拒絕產生空清單"
    mf.write_text("".join(f"{sha(root / f)}  {f}\n" for f in files), encoding="utf-8")
    return True, f"已寫入 {MANIFEST}：{len(files)} 個檔案"


def verify(root):
    mf = root / MANIFEST
    if not mf.exists():
        return False, "CHECK protected_files n=0 清單不存在 FAIL"
    entries = {}
    for line in mf.read_text(encoding="utf-8").splitlines():
        if line.strip():
            h, p = line.split("  ", 1)
            entries[p] = h
    changed, missing = [], []
    for p, h in entries.items():
        fp = root / p
        if not fp.is_file():
            missing.append(p)
        elif sha(fp) != h:
            changed.append(p)
    inc, exc = read_globs(root)
    new = [f for f in expand(root, inc, exc) if f not in entries]
    n = len(entries)
    ok = n > 0 and not changed and not missing
    msg = f"CHECK protected_files n={n} changed={len(changed)} missing={len(missing)} new={len(new)} {'PASS' if ok else 'FAIL'}"
    detail = [f"  changed: {p}" for p in changed[:10]] + [f"  missing: {p}" for p in missing[:10]] + [f"  new(警告): {p}" for p in new[:10]]
    return ok, "\n".join([msg] + detail)


def selftest():
    results = []
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        (root / "docs").mkdir()
        (root / "frozen").mkdir()
        (root / GLOBS).write_text("frozen/**\n", encoding="utf-8")
        (root / "frozen/a.bin").write_bytes(b"AAAA")
        (root / "frozen/b.bin").write_bytes(b"BBBB")
        ok, _ = build(root)
        results.append(("建立清單", ok, True))
        results.append(("原樣 verify 應 PASS", verify(root)[0], True))
        (root / "frozen/a.bin").write_bytes(b"AAAX")
        results.append(("竄改一檔應 FAIL", verify(root)[0], False))
        (root / "frozen/a.bin").write_bytes(b"AAAA")
        (root / "frozen/b.bin").unlink()
        results.append(("刪除一檔應 FAIL", verify(root)[0], False))
        (root / MANIFEST).write_text("", encoding="utf-8")
        results.append(("空清單 n=0 應 FAIL", verify(root)[0], False))
        ok2, _ = build(root)
        results.append(("清單已存在、不加 --force 應拒絕", ok2, False))
    bad = [name for name, got, want in results if got != want]
    for name, got, want in results:
        print(f"  selftest {'ok ' if got == want else 'BAD'}: {name}")
    print(f"CHECK protect_selftest n={len(results)} {'PASS' if not bad else 'FAIL'}")
    return not bad


if __name__ == "__main__":
    root = pathlib.Path(__file__).resolve().parent.parent
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verify"
    if cmd == "--selftest":
        sys.exit(0 if selftest() else 1)
    if cmd == "build":
        ok, msg = build(root, force="--force" in sys.argv)
        print(msg)
        sys.exit(0 if ok else 1)
    ok, msg = verify(root)
    print(msg)
    sys.exit(0 if ok else 1)
