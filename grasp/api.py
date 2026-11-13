"""grasp/api.py -- the functions behind demo_start.py (detect, pick, detect_pick, layout, refresh, status).

Loaded by demo_start.py from disk on every Ctrl+Enter, so the newest version always runs even if the Script
Editor still shows an old copy of demo_start.py (the Script Editor does not reload files changed on disk).
See demo_start.py / README for usage.
"""
API_VERSION = "2026-10-01b"
import asyncio, builtins, datetime, math, os, traceback

def _stage_root():
    """Project root = parent of the folder holding the open scene (scenes/*.usd). No path is hard-coded."""
    import omni.usd
    layer = omni.usd.get_context().get_stage().GetRootLayer()
    return os.path.dirname(os.path.dirname(os.path.abspath(layer.realPath)))


ROOT = _stage_root()      # the scene of this folder must be open (see README)
_REG = "_nhc_d455_yolo2_demo"


def _reg():
    if not hasattr(builtins, _REG):
        setattr(builtins, _REG, dict(start_task=None, task=None, action=None, state="idle", result=None, error=None,
                                     detection=None, started=None))
    return getattr(builtins, _REG)


def _load(rel):
    p = os.path.join(ROOT, rel); g = dict(NO_AUTORUN=True, __file__=p)
    exec(compile(open(p, encoding="utf-8").read(), p, "exec"), g)       # always the latest code on disk
    return g


# ------------------------------------------------------------------ start (runs once when this file is executed)
async def _start():
    oc = _load("perception/one_click.py")
    pk = _load("grasp/pick.py")
    await pk["ensure_sim"]()
    ns = oc["_load_arm_stack"]()                 # checks the open stage is this folder's scene; loads IK + D455 capture
    ns["ik_follow_start"]()
    import omni.kit.app
    for _ in range(600):
        await omni.kit.app.get_app().next_update_async()
        if ns["ik_follow_status"]().get("running"):
            break
    else:
        raise RuntimeError(f"IK did not start: {ns['ik_follow_status']()}")
    await oc["_ensure_server"]()                 # YOLO server (own Python env), so the first detect() is fast
    return "ready"


def _ik_target_visible(on):
    """Show / hide the /World/IKTarget marker while an action runs (session layer only, never saved).
    Hiding also deselects it: a selected prim keeps its move gizmo / outline in the viewport even when invisible,
    so the marker would still seem to follow the arm. The previous selection comes back with the marker."""
    try:
        import omni.usd
        from pxr import Usd, UsdGeom
        ctx = omni.usd.get_context(); st = ctx.get_stage(); prim = st.GetPrimAtPath("/World/IKTarget")
        if not prim.IsValid():
            return
        with Usd.EditContext(st, st.GetSessionLayer()):
            UsdGeom.Imageable(prim).GetVisibilityAttr().Set(UsdGeom.Tokens.inherited if on else UsdGeom.Tokens.invisible)
        sel = ctx.get_selection(); r = _reg()
        if not on:
            paths = list(sel.get_selected_prim_paths())
            if any(p == "/World/IKTarget" or p.startswith("/World/IKTarget/") for p in paths):
                r["ik_selected"] = paths
                sel.set_selected_prim_paths([p for p in paths if not p.startswith("/World/IKTarget")], True)
        elif r.get("ik_selected"):
            sel.set_selected_prim_paths(r.pop("ik_selected"), True)
    except Exception:
        pass


def _schedule(action, coro_fn):
    r = _reg()
    if r["task"] is not None and not r["task"].done():
        return dict(state="busy", running=r["action"], hint="wait, or check status()")

    async def body():
        _ik_target_visible(False)
        try:
            if r["start_task"] is not None and not r["start_task"].done():
                await r["start_task"]
            if r["start_task"] is not None and r["start_task"].exception():
                raise RuntimeError(f"demo_start failed: {r['start_task'].exception()}")
            res = await coro_fn()
            r.update(state="done", result=res, error=None)
            print(f"[{action}] done:", res)
        except Exception as exc:
            r.update(state="error", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
            print(f"[{action}] ERROR:", r["error"])
        except asyncio.CancelledError:
            r.update(state="cancelled", error="cancelled"); raise
        finally:
            _ik_target_visible(True)
    r.update(action=action, state="running", result=None, error=None, traceback=None,
             started=datetime.datetime.now().isoformat(timespec="seconds"))
    r["task"] = asyncio.ensure_future(body())
    return dict(state="scheduled", action=action, hint="poll status()")


def _say(ret):
    """The Script Editor does not show return values: print every answer (scheduled / busy / refused) as well."""
    if isinstance(ret, dict) and ret.get("state") == "refused":
        print(f"[refused] {ret['why']}")
    elif isinstance(ret, dict) and ret.get("state") == "busy":
        print(f"[busy] {ret['running']} is still running -> wait, or check status()")
    elif isinstance(ret, dict) and ret.get("state") == "scheduled":
        print(f"[{ret['action']}] started -> progress: status()")
    return ret


def _stale(reason):
    """The table changed: the last detect() result may no longer be used by pick(). Keeps the newest reason."""
    d = _reg()["detection"]
    if d is not None:
        d.update(stale=True, stale_reason=reason)


def _check(target, order_by):
    """None if the arguments are fine, else a refusal (printed by _say)."""
    if target not in ("cube", "cuboid", "cylinder", "sphere"):
        return dict(state="refused", why=f'target must be "cube", "cuboid", "cylinder" or "sphere", got {target!r}')
    if order_by not in ("camera", "arm_base"):
        return dict(state="refused", why=f'order_by must be "camera" or "arm_base", got {order_by!r}')
    return None


# ------------------------------------------------------------------ public API
def detect():
    """Pictures -> objects with class, confidence, world center; arm back home. Result kept for pick()."""
    async def run():
        pk = _load("grasp/pick.py")
        await pk["ensure_sim"]()
        doc = await pk["Picker"]().perceive()
        cam = doc["views"][0]["camera"]["position_m"]
        objs = [dict(id=o["id"], class_name=o["class_name"], conf=round(o["conf"], 3),
                     center_mm=[round(v * 1000, 1) for v in o["center_xyz_m"]] if o.get("center_xyz_m") else None,
                     dist_from_camera_m=round(math.dist(o["center_xyz_m"], cam), 3) if o.get("center_xyz_m") else None)
                for o in doc["objects"]]
        _reg()["detection"] = dict(doc=doc, run_id=doc["run_id"], stale=False, objects=objs)
        return dict(run_id=doc["run_id"], n_objects=len(objs), objects=sorted(objs, key=lambda o: o["dist_from_camera_m"] or 9),
                    report=os.path.join(ROOT, "outputs", "oneclick", doc["run_id"], "index.html"))
    return _say(_schedule("detect", run))


def pick(target="cube", id=None, order_by="camera"):
    """ONE object of `target` into the basket, from the last detect() (nearest first, or the given id)."""
    bad = _check(target, order_by)
    if bad:
        return _say(bad)
    d = _reg()["detection"]
    if d is None:
        return _say(dict(state="refused", why="no detect() result yet -> run detect() first, then pick()"))
    if d.get("stale"):
        return _say(dict(state="refused", why=f"the last detect() is out of date ({d['stale_reason']}) -> run detect() again, then pick()"))
    cands = [o for o in d["objects"] if o["class_name"] == target and (id is None or o["id"] == id)]
    if not cands:
        seen = ", ".join(f"#{o['id']} {o['class_name']}" for o in d["objects"]) or "nothing"
        return _say(dict(state="refused", why=f"the last detect() ({d['run_id']}) found no {target}" + (f" with id {id}" if id is not None else "")
                         + f" (it found: {seen})"))

    async def run():
        pk = _load("grasp/pick.py")
        _stale("a pick() moved objects")
        doc = await pk["Picker"]().run(target, 1, order_by, open_report=False, doc=d["doc"], only_id=id)
        return _summary(doc)
    return _say(_schedule(f"pick({target!r})", run))


def detect_pick(target="cube", max_picks=12, order_by="camera", open_report=True):
    """detect() -> pick() repeated until no `target` is left (or max_picks attempts). Opens the report at the end."""
    bad = _check(target, order_by)
    if bad:
        return _say(bad)

    async def run():
        pk = _load("grasp/pick.py")
        _stale("detect_pick() moved objects")
        doc = await pk["Picker"]().run(target, max_picks, order_by, open_report=open_report)
        return _summary(doc)
    return _say(_schedule(f"detect_pick({target!r})", run))


def layout(target=None, n=10, seed=None):
    """(testing) Stop -> n new random test blocks (<=60 mm, with physics) -> play. Basket emptied.
    target (optional): make sure EVERY size of that type is on the table (e.g. "sphere" -> spheres 40/50/60 mm, the
    rest random), so a following detect_pick(target) has something to pick. layout() alone: any n random blocks,
    which may contain few or none of the type you want to pick."""
    async def run():
        placed = await _load("grasp/series_test.py")["relayout"](target, seed, n)
        _stale("layout() replaced the blocks")
        from collections import Counter
        count = Counter(p["cls"] for p in placed)
        return dict(placed=len(placed), per_type=dict(count), blocks=[p["name"] for p in placed])
    return _say(_schedule(f"layout({target!r})", run))


def refresh():
    """Reopen this folder's scene (never saved) and start again like demo_start.py.
    Clears what builds up over a long session and slows the simulation down: camera render products left by
    earlier code, Replicator state, physics/IK callbacks. Test blocks are removed -> layout() again if needed."""
    async def run():
        import omni.usd, omni.timeline, omni.kit.app
        ctx = omni.usd.get_context(); path = ctx.get_stage().GetRootLayer().realPath
        ik = {"__file__": os.path.join(ROOT, "scripts", "ik_controller.py")}
        exec(compile(open(ik["__file__"], encoding="utf-8").read(), ik["__file__"], "exec"), ik)
        ik["ik_follow_stop"]()
        omni.timeline.get_timeline_interface().stop()
        for name in ("_nhc_d455_capture_left", "_nhc12_d455_capture_registry"):   # cached render products
            if hasattr(builtins, name):
                delattr(builtins, name)
        for _ in range(10):
            await omni.kit.app.get_app().next_update_async()
        ok, err = await ctx.open_stage_async(path)
        if not ok:
            raise RuntimeError(f"could not reopen {path}: {err}")
        for _ in range(60):
            await omni.kit.app.get_app().next_update_async()
        _stale("refresh() reopened the scene")
        await _start()
        return dict(reopened=path, note="test blocks removed; call layout() for a test scene")
    return _say(_schedule("refresh()", run))


def _summary(doc):
    d = doc["diagnosis"]
    picks = [c for c in doc["cycles"] if not c.get("done")]
    end = next((c for c in doc["cycles"] if c.get("done")), None)
    return dict(in_basket=d["in_basket"], expected=d["expected_truth"],
                picks=[dict(id=c["target_id"], cls=c["target_class"], outcome=c["outcome"], dist_from_camera_m=c.get("dist_from_camera_m"))
                       for c in picks],
                stopped_because=end["reason"] if end else ("one pick() done" if len(picks) == 1 else "max_picks reached"),
                issues=[i["what"] for i in d["issues"]], report=os.path.join(doc["run_dir"], "index.html"), total_s=doc["total_s"])


def status():
    """What is running now (with the current step) and the last result."""
    import json
    r = _reg()
    out = {k: r.get(k) for k in ("action", "state", "started", "error", "result")}
    if r["start_task"] is not None:
        out["start"] = "running" if not r["start_task"].done() else (f"ERROR {r['start_task'].exception()}" if r["start_task"].exception() else "ready")
    if r["state"] == "running":
        for f in ("pick", "oneclick"):
            try:
                out[f"{f}_step"] = json.load(open(os.path.join(ROOT, "outputs", f, "latest_status.json"))).get("step")
            except Exception:
                pass
    d = r["detection"]
    out["last_detection"] = None if d is None else dict(run_id=d["run_id"], stale=d["stale"], objects=[(o["id"], o["class_name"]) for o in d["objects"]])
    import pprint; pprint.pprint(out, width=120, sort_dicts=False)      # the Script Editor shows prints only
    return out


# ------------------------------------------------------------------ executed on Ctrl+Enter
def _boot():
    r = _reg()
    if r["task"] is not None and not r["task"].done():
        print("demo_start: an action is still running:", r["action"], "-> not restarting"); return
    if r["start_task"] is None or r["start_task"].done():
        r["start_task"] = asyncio.ensure_future(_start())
        r["start_task"].add_done_callback(lambda t: print(
            "demo_start: READY. Nothing runs by itself -- call detect(), pick(\"cube\") or detect_pick(\"cube\")."
            if not t.cancelled() and t.exception() is None else f"demo_start: FAILED: {t.exception()}"))
    print(f"demo_start (api {API_VERSION}): starting arm IK, camera and YOLO server (about 10 s). "
          "It does not run detect() / pick() / detect_pick().")


if not globals().get("NO_AUTORUN"):
    _boot()
