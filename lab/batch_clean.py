"""lab/batch_clean.py -- feedback-2 #4: the USER entry points (layout(), detect_pick() of demo_start2.py) over several cases, with the
problems counted separately: release (a block taken back out of the basket), grasp (aborted grasp / more than one tug test),
not attempted (skipped as blocked by neighbours or not seen).  Writes outputs/lab/live/batch2_clean1.json"""
import asyncio, builtins, json, os
import omni.kit.app, omni.timeline
exec(open(os.path.join(builtins._lab_ds2["LAB"], "ab_squeeze.py")).read().split("async def _ab")[0])      # reuse _idle and _metrics


async def _bc(cases, tag):
    g = builtins._lab_ds2; reg = builtins._nhc_d455_yolo2_demo; root = g["ROOT"]; rows = []
    for target, seed in cases:
        row = dict(target=target, seed=seed)
        try:
            omni.timeline.get_timeline_interface().set_auto_update(True)
            g["layout"](target, seed=seed); await _idle(reg)
            for _ in range(30): await omni.kit.app.get_app().next_update_async()
            g["detect_pick"](target); await _idle(reg)
            rep = (reg.get("result") or {}).get("report", "")
            row.update(state=reg.get("state"), error=reg.get("error"))
            if rep:
                rd = os.path.dirname(rep); res = json.load(open(os.path.join(rd, "result.json"))); m = _metrics(res)
                cyc = res.get("cycles", []); skipped = [e.get("skipped") for e in res.get("events", []) if e["name"] == "nothing_graspable"]
                row.update(run_dir=rd, n_target=m["n_target"], final_in_basket=m["final_in_basket"],
                           release_problem=m["taken_back_out"],
                           grasp_aborted=sum(1 for c in cyc if not c.get("ok") and c.get("tug") is None),
                           grasp_needed_retug=sum(1 for c in cyc if (c.get("tug") or {}).get("tests", 1) > 1),
                           first_try_ok=sum(1 for c in cyc if c.get("ok") and (c.get("tug") or {}).get("tests") == 1),
                           attempts=len(cyc), not_attempted=max(0, m["n_target"] - len({c["tid"] for c in cyc})), skipped_reasons=skipped,
                           arrive_to_release_s=[c["from_arrive_to_release_s"] for c in cyc],
                           total_sim_s=m["total_sim_s"])
        except Exception:
            import traceback; row["error"] = traceback.format_exc()
        rows.append(row)
        json.dump(dict(done=False, rows=rows), open(os.path.join(root, "outputs", "lab", "live", f"batch2_{tag}.json"), "w"), indent=1, default=str)
    json.dump(dict(done=True, rows=rows), open(os.path.join(root, "outputs", "lab", "live", f"batch2_{tag}.json"), "w"), indent=1, default=str)


def start_bc(cases, tag="clean1"):
    builtins._lab_bc_task = asyncio.ensure_future(_bc(cases, tag))
