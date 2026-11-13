"""lab/seq_eval.py -- the user's scenario: ONE layout, then detect_pick for several classes in a row (the basket fills up; later runs find earlier blocks in it).
Per layout: every block of every class asked for must end in the basket, none on top of another, none left on the table.
   exec(open(...).read()); start_sq([(seed, ["cube","cuboid","cylinder","sphere"]), ...], "s1")     needs builtins._lab_ds2
Writes outputs/lab/live/seq_<tag>.json"""
import asyncio, builtins, json, math, os
import omni.kit.app, omni.timeline

HALF_H = {"cube": lambda n: float(n.split("_")[1]), "sphere": lambda n: float(n.split("_")[1]),
          "cylinder": lambda n: 40.0 if "lying" in n else float(n.split("_")[2][1:]),
          "cuboid": lambda n: {"2_1_1": 30.0, "2_2_1": 30.0, "3_2_1": 20.0}[n.split("_", 1)[1]]}


async def _idle(reg):
    for _ in range(60 * 60 * 8):
        await omni.kit.app.get_app().next_update_async()
        if reg["task"] is None or reg["task"].done():
            return


async def _sq(cases, tag):
    g = builtins._lab_ds2; reg = builtins._nhc_d455_yolo2_demo; root = g["ROOT"]; rows = []
    path = os.path.join(root, "outputs", "lab", "live", f"seq_{tag}.json")
    for seed, targets in cases:
        row = dict(seed=seed, runs=[])
        try:
            omni.timeline.get_timeline_interface().set_auto_update(True)
            g["layout"](None, seed=seed); await _idle(reg)
            for _ in range(30): await omni.kit.app.get_app().next_update_async()
            for tgt in targets:
                g["detect_pick"](tgt); await _idle(reg)
                rep = (reg.get("result") or {}).get("report", ""); run = dict(target=tgt, state=reg.get("state"), error=reg.get("error"))
                if rep:
                    rd = os.path.dirname(rep); res = json.load(open(os.path.join(rd, "result.json"))); ta = res.get("truth_after", {}); tb = res.get("truth_before", {}); inb = res.get("in_basket_after", {})
                    names = [n for n in tb if n.split("_")[0] == tgt]
                    stacked = []
                    for n, p in ta.items():
                        if inb.get(n) and n != "object":
                            try:
                                if p[2] - 0.6035 > HALF_H[n.split("_")[0]](n) / 2000.0 + 0.010 and any(m != n and inb.get(m) and math.dist(ta[m][:2], p[:2]) < 0.05 for m in ta): stacked.append(n)      # higher than it should be AND another block within 5 cm (a tilted block alone is not a stack)
                            except Exception: pass
                    cyc = res.get("cycles", [])
                    run.update(run_dir=rd, n_target=len(names), in_basket=[n for n in names if inb.get(n)], on_table=[n for n in names if not inb.get(n)], stacked_now=stacked,
                               picks=len(cyc), place_changes=sum(len(c.get("place_changes", [])) for c in cyc), spots=[c.get("cell") for c in cyc], total_sim_s=res.get("total_sim_s"),
                               skipped=[e.get("skipped") for e in res.get("events", []) if e["name"] == "nothing_graspable"], arrive_to_release_s=[c.get("from_arrive_to_release_s") for c in cyc])
                row["runs"].append(run)
        except Exception:
            import traceback; row["error"] = traceback.format_exc()
        rows.append(row)
        json.dump(dict(done=False, rows=rows), open(path, "w"), indent=1, default=str)
    json.dump(dict(done=True, rows=rows), open(path, "w"), indent=1, default=str)


def start_sq(cases, tag="s1"):
    builtins._lab_sq_task = asyncio.ensure_future(_sq(cases, tag))
