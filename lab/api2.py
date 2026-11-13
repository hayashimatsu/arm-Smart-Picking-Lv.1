"""lab/api2.py -- the functions behind demo_start.py: layout(seed, 8) and sort() (lab cell); the blocks-case layout_blocks() / detect_pick() stay.

detect_pick() here is the video-mode version (lab/live_pick.py):
  * no two-photo detect(): the wrist D455 streams, YOLO + depth run on the video, object class / confidence / world position
    are tracked continuously (lab/live_perception.py);
  * the arm moves along one continuous trajectory (no stop except at grasp and place);
  * grasp = one continuous gripper close (no 1-degree steps); place = lowered onto the basket floor, never dropped;
  * the original red /World/object cube is removed for the session (it is not one of the random blocks).
Everything is read from disk on every call, so the newest code runs. Nothing here is saved into scenes/*.usd.
"""
API2_VERSION = "2026-10-21"
import asyncio, builtins, os, traceback

_G = globals()
_base_ns = {"NO_AUTORUN": True, "__file__": os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "grasp", "api.py")}
exec(compile(open(_base_ns["__file__"], encoding="utf-8").read(), _base_ns["__file__"], "exec"), _base_ns)   # _schedule, _say, _reg, _load ...
ROOT = _base_ns["ROOT"]
LAB = os.path.join(ROOT, "lab")


def _load_lab():
    """lab modules executed in one private namespace (FastPicker, Live, LivePicker, P ...), always the latest from disk."""
    ns = {"NO_AUTORUN": True}
    for name in ("fast_pick.py", "live_perception.py", "live_pick.py"):
        p = os.path.join(LAB, name); ns["__file__"] = p
        exec(compile(open(p, encoding="utf-8").read(), p, "exec"), ns)
    return ns


def _remove_original_object():
    """Deactivate /World/object (session layer only, never saved): layout() brings back the physics state of everything else."""
    try:
        import omni.usd
        from pxr import Usd
        st = omni.usd.get_context().get_stage(); prim = st.GetPrimAtPath("/World/object")
        if prim.IsValid() and prim.IsActive():
            with Usd.EditContext(st, st.GetSessionLayer()):
                prim.SetActive(False)
    except Exception:
        traceback.print_exc()


# Physics of the test scene (feedback-4 #1), authored in the SESSION layer after layout(), never saved into scenes/*.usd.
#  - table and basket had no physics material (PhysX default); give them one close to the blocks (friction 1.0 = the blocks', restitution 0)
#  - balls and cylinders: angular + linear damping, because friction cannot stop a rolling ball (measured: friction 1.0 and 5.0 change nothing,
#    outputs/lab/live/phys_p1.json / phys_p2.json); with angular damping 1.5 and linear damping 0.3 a ball launched at 0.3 m/s rolls about 18 cm (phys_p4.json); lying cylinders 0.5 / 0.3
PHYS = dict(table_mu=1.0, sphere=(1.5, 0.3), cylinder=(0.5, 0.3))      # (angular, linear) damping; a lying cylinder needs less than a ball for the same distance (phys_pc2.json)


def _apply_physics():
    """Call with the timeline stopped (changing physics settings while it runs invalidates the physics views)."""
    from pxr import Usd, UsdGeom, UsdShade, UsdPhysics, PhysxSchema
    import omni.usd
    st = omni.usd.get_context().get_stage()
    with Usd.EditContext(st, st.GetSessionLayer()):
        path = "/World/LabPhysicsMaterial"
        m = UsdShade.Material.Define(st, path); a = UsdPhysics.MaterialAPI.Apply(m.GetPrim())
        a.CreateStaticFrictionAttr(PHYS["table_mu"]); a.CreateDynamicFrictionAttr(PHYS["table_mu"]); a.CreateRestitutionAttr(0.0)
        mat = UsdShade.Material(st.GetPrimAtPath(path))
        for p in st.Traverse():
            if "SM_HeavyDutyPackingTable_C02_01" in str(p.GetPath()) and p.HasAPI(UsdPhysics.CollisionAPI):
                UsdShade.MaterialBindingAPI.Apply(p).Bind(mat, UsdShade.Tokens.strongerThanDescendants, "physics")
        cont = st.GetPrimAtPath("/World/container_h20")
        if cont.IsValid():
            UsdShade.MaterialBindingAPI.Apply(cont).Bind(mat, UsdShade.Tokens.strongerThanDescendants, "physics")
        tb = st.GetPrimAtPath("/World/TestBlocks")
        for c in (tb.GetChildren() if tb.IsValid() else []):
            kind = c.GetName().split("_")[0]
            if c.IsA(UsdGeom.Gprim) and kind in ("sphere", "cylinder"):
                r = PhysxSchema.PhysxRigidBodyAPI.Apply(c); ang, lin = PHYS[kind]
                r.CreateAngularDampingAttr(ang); r.CreateLinearDampingAttr(lin)


async def _start2():
    """Arm IK + camera stack (no YOLO photo server needed: the video YOLO server starts with the first detect_pick())."""
    import omni.kit.app
    _remove_original_object()
    oc = _base_ns["_load"]("perception/one_click.py")
    pk = _base_ns["_load"]("grasp/pick.py")
    await pk["ensure_sim"]()
    ns = oc["_load_arm_stack"]()
    ns["ik_follow_start"]()
    for _ in range(600):
        await omni.kit.app.get_app().next_update_async()
        if ns["ik_follow_status"]().get("running"):
            break
    else:
        raise RuntimeError(f"IK did not start: {ns['ik_follow_status']()}")
    return "ready"


def _load_sort():
    """lab namespace + lab/sort_parts.py (layout / sort of the three-part cell), always the latest from disk."""
    ns = _load_lab(); p = os.path.join(LAB, "sort_parts.py"); ns["__file__"] = p
    exec(compile(open(p, encoding="utf-8").read(), p, "exec"), ns)
    return ns


def layout(seed=None, count=8):
    """Lab cell (scenes/lab_cell_g1.usda): `count` parts (8 = 3/3/2) dropped into the source basket, all upright, random turn;
    bins emptied. Session layer only, never saved. Plan + settled positions: outputs/sort/<time>_s<seed>_n<count>/layout.json."""
    async def run():
        return await _load_sort()["layout_run"](seed, count)
    return _base_ns["_say"](_base_ns["_schedule"](f"layout(seed={seed}, count={count})", run))


def sort():
    """Sort the source basket: at view A decide -> grasp -> the bin of its class (red / blue / yellow) -> back to view A, until
    nothing is left. Prints PASSED / NOT PASSED; result: outputs/sort/<layout run>/result.json (tools/check_acceptance.py)."""
    async def run():
        ns = _load_sort()
        return await ns["sort_run"](ns, _base_ns)
    return _base_ns["_say"](_base_ns["_schedule"]("sort()", run))


def layout_blocks(target=None, n=10, seed=None):
    """(blocks scene, the old layout()) n new random blocks (<=60 mm, with physics); basket emptied; the original red cube is removed.
    target (optional): every size of that type is on the table (e.g. "sphere" -> 40/50/60 mm), so detect_pick(target) has something to pick."""
    async def run():
        _remove_original_object()
        placed = await _base_ns["_load"]("grasp/series_test.py")["relayout"](target, seed, n)
        import builtins
        builtins._lab_basket = None; builtins._lab_basket_empty = True          # the basket was emptied: the next detect_pick() knows it is empty without looking
        _remove_original_object()
        import omni.timeline, omni.kit.app
        tl = omni.timeline.get_timeline_interface(); tl.stop()               # physics settings only with the timeline stopped, then play and let the blocks settle
        for _ in range(10):
            await omni.kit.app.get_app().next_update_async()
        _apply_physics(); tl.play()
        for _ in range(120):
            await omni.kit.app.get_app().next_update_async()
        from collections import Counter
        return dict(placed=len(placed), per_type=dict(Counter(p["cls"] for p in placed)), blocks=[p["name"] for p in placed])
    return _base_ns["_say"](_base_ns["_schedule"](f"layout({target!r})", run))


def detect_pick(target="cube", max_picks=12):
    """Pick every `target` block ("cube" / "cuboid" / "cylinder" / "sphere") into the basket in video mode.
    Report: outputs/lab/live/run_<time>/index.html (video, times, what moved)."""
    if target not in ("cube", "cuboid", "cylinder", "sphere"):
        return _base_ns["_say"](dict(state="refused", why=f'target must be "cube", "cuboid", "cylinder" or "sphere", got {target!r}'))

    async def run():
        _remove_original_object()
        ns = _load_lab()
        res = await ns["LivePicker"]().run_live(target, max_picks)
        if res.get("error"):
            raise RuntimeError(res["error"])
        cyc = res.get("cycles", [])
        rel = [e["sim_s"] for e in res.get("events", []) if e["name"].endswith(":released")]
        return dict(picked_ok=sum(1 for c in cyc if c.get("ok") and c.get("kind") != "relocate"), moved_aside=sum(1 for c in cyc if c.get("ok") and c.get("kind") == "relocate"), attempts=len(cyc),
                    arrive_to_release_s=[c["from_arrive_to_release_s"] for c in cyc],
                    release_to_release_s=[round(b - a, 2) for a, b in zip(rel, rel[1:])],
                    report=os.path.join(res["run_dir"], "index.html"))
    return _base_ns["_say"](_base_ns["_schedule"](f"detect_pick({target!r})", run))


def _boot2():
    r = _base_ns["_reg"]()
    if r["task"] is not None and not r["task"].done():
        print("demo_start: an action is still running:", r["action"], "-> not restarting"); return
    if r["start_task"] is None or r["start_task"].done():
        r["start_task"] = asyncio.ensure_future(_start2())
        r["start_task"].add_done_callback(lambda t: print(
            'demo_start: READY. layout(seed, 8) then sort().'
            if not t.cancelled() and t.exception() is None else f"demo_start: FAILED: {t.exception()}"))
    print(f"demo_start (api2 {API2_VERSION}): starting arm IK, wrist D455 and YOLO (about 10 s). Nothing moves until you call layout(seed, 8) / sort().")


if not globals().get("NO_AUTORUN"):
    _boot2()
