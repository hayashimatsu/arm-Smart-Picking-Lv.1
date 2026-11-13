"""lab/batch2.py -- run the USER entry points (demo_start2 layout()/detect_pick()) over several cases and write a summary.
    exec(open(".../lab/batch2.py").read()); start_batch2([("sphere", 31), ("cube", 32)], "t1")      (needs builtins._lab_ds2 = demo_start2 globals)"""
import asyncio, builtins, json, os
import omni.kit.app, omni.timeline




async def _wait_idle(reg):
    for _ in range(60 * 60 * 10):
        await omni.kit.app.get_app().next_update_async()
        if reg["task"] is None or reg["task"].done():
            return
    raise RuntimeError("timeout")


async def _b2(cases, tag):
    g = builtins._lab_ds2; reg = builtins._nhc_d455_yolo2_demo; rows = []
    root = g["ROOT"]
    for target, seed in cases:
        row = dict(target=target, seed=seed)
        try:
            omni.timeline.get_timeline_interface().set_auto_update(True)
            g["layout"](target, seed=seed); await _wait_idle(reg)
            for _ in range(30): await omni.kit.app.get_app().next_update_async()
            g["detect_pick"](target); await _wait_idle(reg)
            row.update(state=reg.get("state"), error=reg.get("error"), result=reg.get("result"))
            run = (reg.get("result") or {}).get("report", "")
            if run:
                d = json.load(open(os.path.join(os.path.dirname(run), "result.json")))
                tb, inb = d.get("truth_before", {}), d.get("in_basket_after", {})
                onT = [n for n, p in tb.items() if n.split("_")[0] == target and p[0] < -0.2]
                row.update(n_target=len(onT), n_in_basket=sum(1 for n in onT if inb.get(n)), moved_other=[n for n in d["truth_after"] if d["truth_after"][n] != tb.get(n) and n not in onT and max(abs(a - b) for a, b in zip(d["truth_after"][n], tb[n])) > 0.01])
        except Exception:
            import traceback; row["error"] = traceback.format_exc()
        rows.append(row)
        json.dump(dict(done=False, rows=rows), open(os.path.join(root, "outputs", "lab", "live", f"batch2_{tag}.json"), "w"), indent=1, default=str)
    json.dump(dict(done=True, rows=rows), open(os.path.join(root, "outputs", "lab", "live", f"batch2_{tag}.json"), "w"), indent=1, default=str)


def start_batch2(cases, tag="t"):
    builtins._lab_b2_task = asyncio.ensure_future(_b2(cases, tag))
