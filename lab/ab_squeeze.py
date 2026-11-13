"""lab/ab_squeeze.py -- feedback-2 #2: same seed, squeeze 3 deg vs 5 deg, N runs each; counts blocks taken back out of the basket,
tug tests, aborts, in-basket result.   exec(open(...).read()); start_ab("cube", 33, [3.0, 5.0], 3, "a1")   (needs builtins._lab_ds2)"""
import asyncio, builtins, json, os
import omni.kit.app, omni.timeline


async def _idle(reg):
    for _ in range(60 * 60 * 5):
        await omni.kit.app.get_app().next_update_async()
        if reg["task"] is None or reg["task"].done():
            return


def _metrics(res):
    rd = res["run_dir"]; w = json.load(open(os.path.join(rd, "truth_watch.json")))
    bmin, bmax = None, None
    tb, ta = res["truth_before"], res["truth_after"]
    ins = lambda p: 0.044 < p[0] < 0.656 and -0.206 < p[1] < 0.206 and p[2] < 0.70           # basket box (scene constants)
    target = [n for n, p in tb.items() if p[0] < -0.2 and n.split("_")[0] == res["target"]]
    taken_back = 0; ever_in = 0
    for n in target:
        seen = [ins(tr[n]) for t, tr in w if n in tr]
        if any(seen):
            ever_in += 1
            last_in = max(i for i, s in enumerate(seen) if s)
            if not seen[-1] or any((not s) for s in seen[last_in:]):
                taken_back += 1
    cyc = res.get("cycles", [])
    return dict(n_target=len(target), ever_in_basket=ever_in, final_in_basket=sum(1 for n in target if ins(ta[n])), taken_back_out=taken_back,
                attempts=len(cyc), ok=sum(1 for c in cyc if c.get("ok")), aborts=sum(1 for c in cyc if c.get("width_mismatch") or not c.get("ok")),
                tug_tests=[(c.get("tug") or {}).get("tests") for c in cyc], total_sim_s=res.get("total_sim_s"))


async def _ab(target, seed, squeezes, n, tag):
    g = builtins._lab_ds2; reg = builtins._nhc_d455_yolo2_demo; root = g["ROOT"]; rows = []
    for sq in squeezes:
        for i in range(n):
            row = dict(squeeze=sq, trial=i)
            try:
                omni.timeline.get_timeline_interface().set_auto_update(True)
                g["layout"](target, seed=seed); await _idle(reg)
                for _ in range(30): await omni.kit.app.get_app().next_update_async()
                g["_remove_original_object"]()
                ns = g["_load_lab"](); ns["P"]["squeeze_deg"] = sq
                res = await ns["LivePicker"]().run_live(target, 12, True, False)
                row.update(error=res.get("error"), run_dir=res.get("run_dir"))
                if not res.get("error"):
                    row.update(_metrics(res))
            except Exception:
                import traceback; row["error"] = traceback.format_exc()
            rows.append(row)
            json.dump(dict(done=False, rows=rows), open(os.path.join(root, "outputs", "lab", "live", f"ab_{tag}.json"), "w"), indent=1, default=str)
    json.dump(dict(done=True, rows=rows), open(os.path.join(root, "outputs", "lab", "live", f"ab_{tag}.json"), "w"), indent=1, default=str)


def start_ab(target, seed, squeezes, n, tag="a1"):
    builtins._lab_ab_task = asyncio.ensure_future(_ab(target, seed, squeezes, n, tag))
