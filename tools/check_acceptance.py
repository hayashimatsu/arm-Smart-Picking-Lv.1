#!/usr/bin/env python3
"""check_acceptance.py <result.json> -- 用 Lin 的失敗定義（STATUS 第 1 行）檢查一份結果。

門檻只讀 docs/acceptance.json，且必須和 STATUS.md 第 1 行一致，否則 FAIL（防止門檻被默默改）。
每項檢查都印出 n（看了幾件）。n==0、結構不認得 = FAIL，不會因為「沒東西可看」而過關。
目前認得的結構：單件夾取的 result.json（頂層有 picks）。sort() 移植後要加 sort 結構與它的 selftest。
  --selftest  用造出來的壞資料證明每一項都會 FAIL。
"""
import json, math, pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def load_cfg(root=ROOT, status_line=None):
    cfg = json.loads((root / "docs/acceptance.json").read_text(encoding="utf-8"))
    if status_line is None:
        status_line = (root / "STATUS.md").read_text(encoding="utf-8").splitlines()[0]
    wall = re.search(r"牆上時鐘\s*≤\s*([0-9.]+)\s*s", status_line)
    peak = re.search(r"關節峰值\s*≤\s*([0-9.]+)", status_line)
    problems = []
    if not wall or float(wall.group(1)) != cfg["per_item_wall_s_max"]:
        problems.append(f"每件牆上時鐘門檻：STATUS 第 1 行={wall.group(1) if wall else '讀不到'}，acceptance.json={cfg['per_item_wall_s_max']}")
    if not peak or float(peak.group(1)) != cfg["joint_peak_max"]:
        problems.append(f"關節峰值門檻：STATUS 第 1 行={peak.group(1) if peak else '讀不到'}，acceptance.json={cfg['joint_peak_max']}")
    return cfg, problems


def evaluate(result, cfg):
    """回傳 [(名稱, n, 失敗清單)]。失敗清單空且 n>0 才算 PASS。"""
    picks = result.get("picks") if isinstance(result, dict) else None
    if not isinstance(picks, list):
        return [("結構", 0, ["認不得這份結果的結構（沒有 picks）；不當作通過"])]
    placed_fail, wall_fail, peak_fail = [], [], []
    for p in picks:
        cls = p.get("cls", "?")
        names = p.get("names") or []
        finished = any(str(n).endswith(":released") for n in names)            # 11/11 corner 1a: one run_live places several parts; its error is copied to
        if p.get("error") is not None and not finished:                         # every piece -> only the piece that did not finish (no release) fails on it
            placed_fail.append(f"{cls}: error={p['error']}")
            continue
        sc = p.get("after_error_scoring") or {}                                 # 11/11 corner 1a: the error path's truth (report only)
        own = p.get("in_own_bin_after") or sc.get("in_own_bin_after")
        dest = {k: bool(v.get("inside")) for k, v in own.items()} if isinstance(own, dict) and own else (p.get("in_dest_after") or {})   # each part vs ITS bin
        before, after = p.get("truth_before") or sc.get("truth_before") or {}, p.get("truth_after") or sc.get("truth_after") or {}
        tn = p.get("target_name")
        if tn and tn in dest:                                                   # 11/11 corner 1a: this pick's own part (stricter than "any part of the class")
            if not (dest[tn] and tn in before and tn in after and math.dist(before[tn][:3], after[tn][:3]) > 0.05):
                placed_fail.append(f"{cls}: {tn} 沒有『進自己類別那一箱且真的被搬動過』")
        else:
            moved_in = [k for k, v in dest.items() if k.startswith(cls + "_") and v and k in before and k in after and math.dist(before[k][:3], after[k][:3]) > 0.05]
            if not moved_in:
                placed_fail.append(f"{cls}: 沒有任何同類零件『進箱且真的被搬動過』")
        w = p.get("turn_wall_s")
        if w is None or w > cfg["per_item_wall_s_max"]:
            wall_fail.append(f"{cls}: 每件一輪牆上 {w} s > {cfg['per_item_wall_s_max']} s")
        j = p.get("joint_peak_ratio")
        if j is None or j > cfg["joint_peak_max"]:
            peak_fail.append(f"{cls}: 關節峰值 {j} > {cfg['joint_peak_max']}")
    n = len(picks)
    return [("放對箱（物理真值）", n, placed_fail), ("每件一輪牆上時鐘", n, wall_fail), ("關節峰值", n, peak_fail)]


def report(rows):
    ok = True
    for name, n, fails in rows:
        good = n > 0 and not fails
        ok &= good
        print(f"CHECK {name} n={n} fail={len(fails)} {'PASS' if good else 'FAIL'}")
        for f in fails:
            print("   -", f)
    return ok


def _pick(cls="flange_round", wall=4.0, peak=0.9, moved=True, error=None):
    key = f"{cls}_galvanized"
    return {"cls": cls, "error": error, "turn_wall_s": wall, "joint_peak_ratio": peak,
            "in_dest_after": {key: True}, "truth_before": {key: [0.4, 0.0, 0.6]},
            "truth_after": {key: [0.0, -0.1, 0.6] if moved else [0.4, 0.0, 0.6]}}


def selftest():
    cfg = {"per_item_wall_s_max": 5.0, "joint_peak_max": 1.0}
    verdict = lambda r: all(n > 0 and not f for _, n, f in evaluate(r, cfg))
    cases = [
        ("全部達標應 PASS", {"picks": [_pick(), _pick("bracket_mid")]}, True),
        ("一件牆上 6 s 應 FAIL", {"picks": [_pick(), _pick(wall=6.0)]}, False),
        ("關節峰值 1.2 應 FAIL", {"picks": [_pick(peak=1.2)]}, False),
        ("進箱但沒被搬動（空殼）應 FAIL", {"picks": [_pick(moved=False)]}, False),
        ("有 error 應 FAIL", {"picks": [_pick(error="timeout")]}, False),
        ("picks 為空 n=0 應 FAIL", {"picks": []}, False),
        ("不認得的結構應 FAIL", {"something": 1}, False),
    ]
    bad = 0
    for name, data, want in cases:
        got = verdict(data)
        bad += got != want
        print(f"  selftest {'ok ' if got == want else 'BAD'}: {name}")
    line_ok = "每件一輪牆上時鐘 ≤5 s、關節峰值 ≤1.00"
    _, p_ok = load_cfg(ROOT, line_ok)
    _, p_bad = load_cfg(ROOT, "每件一輪牆上時鐘 ≤9 s、關節峰值 ≤1.00")
    for name, got, want in [("STATUS 與 acceptance.json 一致應無問題", not p_ok, True), ("STATUS 被改成 9 s 應被抓到", not p_bad, False)]:
        bad += got != want
        print(f"  selftest {'ok ' if got == want else 'BAD'}: {name}")
    print(f"CHECK acceptance_selftest n={len(cases) + 2} {'PASS' if not bad else 'FAIL'}")
    return not bad


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        sys.exit(0 if selftest() else 1)
    if len(sys.argv) != 2:
        sys.exit("用法：check_acceptance.py <result.json> | --selftest")
    cfg, problems = load_cfg()
    for pr in problems:
        print("CHECK 門檻一致性 n=1 FAIL\n   -", pr)
    ok = not problems
    ok &= report(evaluate(json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")), cfg))
    print("總結：", "PASS" if ok else "FAIL（未達標的項目就是缺口，不要改門檻）")
    sys.exit(0 if ok else 1)
