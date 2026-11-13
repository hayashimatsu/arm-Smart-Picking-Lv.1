"""lab/sort_parts.py -- layout(seed, count) and sort() for the three-part lab cell (S0, 2026-11-09; simulation only).

Loaded by lab/api2.py into the lab namespace (FastPicker, LivePicker, Live, P ... already defined there), always from disk.

layout(seed, count)  (test preparation, session layer only, never saved)
  * class counts as reference/smart_picking_v1 layout_gen.class_counts: 8 = 3/3/2, the class with 2 = CLASSES[seed % 3];
    finishes count//2 black, the rest galvanized, shuffled
  * every part UPRIGHT (stable pose 0, the pose the single-pick tests used); turn about the vertical uniform 0-360 deg
  * centre = a reachable cell of scenes/area.json +- 10 mm; top-view outlines (parts/stable_pose_geometry.json, pose 0 convex hull)
    >= 30 mm apart and inside area.json basket_inner (already 50 mm inside the basket walls); rng = default_rng([seed, count, attempt])
  * the part outlines are used ONLY to place the test parts, never by the arm's decisions (D5)
  * dropped 5 mm above the basket floor with the timeline playing (edit target = session layer first, so the picture follows PhysX)
sort()
  * one cycle = baseline LivePicker.run_live(cls, max_picks=1) with home = view A: at view A scan + decide -> grasp -> the bin of
    the class -> back to view A (D7). Classes in a fixed order; a class ends after 2 cycles in a row without a part in the bin
    (or at once when nothing of it is tracked). Ends when every class has ended.
  * grasp settings of the 11/09 single picks (outputs/tests/20261108_single_pick/run_pick.py, ported here because outputs/ is not in git)
  * result.json: picks[] in the structure tools/check_acceptance.py reads; physical (PhysX truth, report only) and system columns apart
"""
import asyncio, builtins, datetime, json, math, os, time, traceback
import numpy as np

SORT_VERSION = "2026-11-09 S0"
SROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(globals().get("__file__") or __file__)), ".."))
CLASSES = ("flange_round", "bracket_mid", "bracket_end")
BINS = {"flange_round": "/World/Bins/red", "bracket_mid": "/World/Bins/blue", "bracket_end": "/World/Bins/yellow"}
PARTS = "/World/Parts"
ORIGINAL_PARTS = [f"{c}_{f}" for c in CLASSES for f in ("galvanized", "black")]      # the 6 prims of scenes/layers/parts_demo.usda
FLOOR_Z = 0.6047                     # source basket floor (perception/config.json table_z_m, 11/08)
DROP_DZ = 0.005                      # lowest point this far above the floor before the drop
GAP, CELL_JITTER, TRIES, PLACE_ATTEMPTS = 0.03, 0.01, 1500, 200
# grasp settings of the 11/09 single picks (outputs/tests/20261109_post_archive_pick/run_110413/result.json)
P_SET = dict(transit_z=0.80, floor_z=0.6052, max_relocations=0)
P_EXTRA = dict(v_carry=1.0, a_carry=1.2,              # 11/11 Lin "Yes, use 1.0"（轉自 PM-Reviewer），驗證：Q2–Q5、T1: v_carry 0.6 -> 1.0 (a_carry kept). 11/10 D+E run 2 (supervisor, Lin "lift slowly too"): carry back to the baseline defaults (was 1.0 / 2.0)
               place_outline=True, place_rank="rows", place_yaw_offsets=(0.0, 15.0, -15.0, 30.0, -30.0, 45.0, -45.0, 60.0, -60.0, 75.0, -75.0, 90.0, -90.0),
               place_map_res_m=0.005,
               place_wrist_guard=True, place_gap_tiers=(0.008, 0.003), place_release_open="grasp", place_wall_m=0.02,
               place_release_margin_m=0.005, place_fallback="emptiest", width_check="neighbor",
               outline_diag=True, grasp_arrive_s=1.0, grasp_height_tol_m=0.005,
               save_tcp_log=True,
               rotate_before_descent=True, approach=dict(pre_above_m=0.04, v=0.12, a=0.6, lift_m=0.03),
               fold_yaw=True, save_map_snapshot=True,
               drop_detect_deg=2.5, drop_window_s=0.2,
               corner_limit=False, carry_lag_strict=False, j1_slow_ratio=None, ik_cap_scale=False,   # 11/10 P5: G3 four off (were True / True / 0.85 / True)
               step_follow=True,                                                                    # 11/10 P5 (Lin P4 A): F7b during fly() (fast_pick default False)
               squeeze_deg=9.5,                                                                     # 11/11 Lin 採用；T1 outputs/share/20261111_GF (fast_pick default 5.0 unchanged)
               floor_filter_m=0.003, floor_filter_classes=("bracket_mid", "bracket_end"),
               abort_up_trapezoid=True)   # 11/11 1e (supervisor: plan C defaults, GUI r5c): live_pick abort_up leg as a trapezoid; off = P_EXTRA.pop("abort_up_trapezoid")   # 11/10 G (supervisor: G4 for the brackets only; the flange as before)                                          # 11/10 F1 wrist plan A (Lin: "the mid still turns round")   # 11/10 E + D on for the sort (Lin saw the drawings 11/10)
# 11/10 (supervisor dispatch, Lin approved): wrist turns 0, +-15 ... +-90 deg (smallest first) only when the grasp yaw has no spot;
# release opening = max(planned, measured) + 5 mm; no spot -> the emptiest point (no fixed fallback cells)
# 11/09 Lin chose B (keep the grasp yaw, turn only when no spot; no alignment) and asked for denser bins:
# outputs/tests/20261109_s0_place/density.py variant B3 = 3 flange / 3 bracket_mid / 3-4 bracket_end per bin in 5/5 seeds, 0-4 of 15 parts turned
# 11/09 Lin: place by the measured outline, in rows from the corner nearest the arm, every long side along world y (offline comparison
# outputs/tests/20261109_s0_place/compare_place.py: rows-along-y fit 3 flange / 2 bracket_mid / 4 bracket_end per bin in 5/5 seeds; "any free spot" 2-3 flanges)
VARIANT_BY_CLASS = {"bracket_mid": {"dyaw_deg": 90.0, "width_mm": 60.0}}
SOUT = os.path.join(SROOT, "outputs", "sort")
_ST = "_sort_parts_state"            # builtins: last layout + one basket map per bin (bins emptied by layout)


def _state():
    if not hasattr(builtins, _ST):
        setattr(builtins, _ST, dict(layout=None, maps={}))
    return getattr(builtins, _ST)


# ------------------------------------------------------------------ layout plan (pure numpy, also runs outside Isaac)
def _geometry():
    g = json.load(open(os.path.join(SROOT, "parts", "stable_pose_geometry.json")))
    hull = {}
    for c in CLASSES:
        p0 = next(p for p in g[c]["poses"] if p["index"] == 0)
        assert p0["up_in_part_frame"][2] > 0.99, f"{c} pose 0 is not upright"
        hull[c] = np.asarray(p0["footprint"], float)
    a = json.load(open(os.path.join(SROOT, "scenes", "area.json")))
    return hull, np.asarray(a["reachable_cells"], float), a["basket_inner"]


def _dense(poly, step=0.002):
    pts = []
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        pts.append(a + (b - a) * np.linspace(0, 1, n, endpoint=False)[:, None])
    return np.vstack(pts)


def _rotz2(deg):
    t = math.radians(deg); return np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])


def class_counts(seed, n):
    c = {k: n // 3 for k in CLASSES}
    for i in range(n - 3 * (n // 3)):                     # 8 -> 3/3/2 with CLASSES[seed % 3] = 2 (as reference layout_gen for 8)
        c[CLASSES[(seed + 1 + i) % 3]] += 1
    return c


def plan_layout(seed, count=8, attempt=0):
    hull, cells, inner = _geometry()
    x0, x1 = inner["x"]; y0, y1 = inner["y"]
    rng = np.random.default_rng([int(seed), int(count), int(attempt)])
    counts = class_counts(seed, count)
    classes = [c for c in CLASSES for _ in range(counts[c])]
    finishes = ["black"] * (count // 2) + ["galvanized"] * (count - count // 2); rng.shuffle(finishes)
    spec = []
    for i, (cls, fin) in enumerate(zip(classes, finishes)):
        idx = sum(1 for c in classes[:i] if c == cls) + 1
        spec.append((f"{cls}_{fin}_{idx:02d}", cls, fin))
    spec = [spec[i] for i in rng.permutation(len(spec))]
    spec.sort(key=lambda s: -float(np.ptp(hull[s[1]], axis=0).max()))          # largest first
    for a in range(PLACE_ATTEMPTS):
        placed, out = [], []
        for name, cls, fin in spec:
            for _ in range(TRIES):
                yaw = float(rng.uniform(0, 360)); P = hull[cls] @ _rotz2(yaw).T
                c = (P.min(0) + P.max(0)) / 2
                cell = cells[rng.integers(len(cells))] + rng.uniform(-CELL_JITTER, CELL_JITTER, 2)
                xy = P - c + cell
                if xy[:, 0].min() < x0 or xy[:, 0].max() > x1 or xy[:, 1].min() < y0 or xy[:, 1].max() > y1:
                    continue
                d = _dense(xy)
                if any(np.min(np.linalg.norm(d[:, None, :] - d2[None, :, :], axis=2)) < GAP for d2 in placed):
                    continue
                placed.append(d)
                origin = cell - c
                out.append(dict(name=name, cls=cls, finish=fin, yaw_deg=round(yaw, 2), origin_xy=[round(float(v), 5) for v in origin],
                                centre_xy=[round(float(v), 4) for v in cell], outline_xy=np.round(xy, 4).tolist()))
                break
            else:
                break
        if len(out) == len(spec):
            return dict(seed=int(seed), count=int(count), attempt=int(attempt), placement_tries=a + 1, counts=counts,
                        finishes={f: finishes.count(f) for f in sorted(set(finishes))}, pose="upright (stable pose 0)",
                        rules=dict(gap_m=GAP, inside="scenes/area.json basket_inner (50 mm inside the walls)", cell_jitter_m=CELL_JITTER,
                                   cells=len(cells), rng="default_rng([seed, count, attempt])"), basket_inner=inner, parts=out)
    raise RuntimeError(f"layout seed {seed} count {count}: not placed in {PLACE_ATTEMPTS} tries")


# ------------------------------------------------------------------ Isaac side
async def _frames(n):
    import omni.kit.app
    for _ in range(n):
        await omni.kit.app.get_app().next_update_async()


def part_names():
    """Active parts under /World/Parts (the layout's, or the original 6 if no layout ran)."""
    import omni.usd
    st = omni.usd.get_context().get_stage(); root = st.GetPrimAtPath(PARTS)
    return [p.GetName() for p in root.GetChildren() if p.IsActive()] if root.IsValid() else []


def part_pose(name):
    """PhysX pose of a part: [x, y, z, qw, qx, qy, qz] (report only)."""
    from omni.physx import get_physx_interface
    t = get_physx_interface().get_rigidbody_transformation(f"{PARTS}/{name}")
    x, y, z = [float(v) for v in t["position"]]; qx, qy, qz, qw = [float(v) for v in t["rotation"]]
    return [x, y, z, qw, qx, qy, qz]


def _box(path):
    import omni.usd
    from pxr import UsdGeom, Usd
    r = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(
        omni.usd.get_context().get_stage().GetPrimAtPath(path)).ComputeAlignedRange()
    return [float(v) for v in r.GetMin()], [float(v) for v in r.GetMax()]


def where_is(xyz, boxes):
    """'red' / 'blue' / 'yellow' / 'source' / 'elsewhere' from a PhysX position and the bin / basket boxes (report only)."""
    for k, (lo, hi) in boxes.items():
        if lo[0] < xyz[0] < hi[0] and lo[1] < xyz[1] < hi[1] and xyz[2] < hi[2] + 0.02:
            return k
    return "elsewhere"


def _boxes():
    b = {BINS[c].rsplit("/", 1)[-1]: _box(BINS[c]) for c in CLASSES}
    b["source"] = _box("/World/container_h20")
    return b


async def apply_layout(lay, steps=None):
    """Session layer only. Original 6 parts off, the previous layout's parts removed, the new parts dropped and settled."""
    import omni.usd, omni.timeline
    from pxr import Usd, UsdGeom, Gf
    st = omni.usd.get_context().get_stage(); tl = omni.timeline.get_timeline_interface(); sl = st.GetSessionLayer()
    tl.stop(); await _frames(10)                                       # rigid bodies are (re)created on play
    steps_info = set_steps_while_stopped(steps) if steps else None     # 11/10 speed C: only here, while stopped (views rebuilt on play)
    st.SetEditTarget(Usd.EditTarget(sl))                               # the picture follows PhysX (11/08, history/failures.md row 72)
    removed = []; others = [l for l in st.GetLayerStack(includeSessionLayers=False)]
    with Usd.EditContext(st, sl):
        root = st.GetPrimAtPath(PARTS)
        for ch in list(root.GetAllChildren()):
            n = ch.GetName(); path = ch.GetPath()
            if n in ORIGINAL_PARTS:
                ch.SetActive(False)
            elif sl.GetPrimAtPath(path) and not any(l.GetPrimAtPath(path) for l in others):
                st.RemovePrim(path); removed.append(n)                       # only prims that exist in the session layer alone (ours)
        for p in lay["parts"]:
            path = f"{PARTS}/{p['name']}"
            prim = st.DefinePrim(path, "Xform")
            prim.GetReferences().AddReference(os.path.join(SROOT, "parts", "usd", f"{p['cls']}.usd"))
            vs = prim.GetVariantSets().GetVariantSet("finish")
            if vs:
                vs.SetVariantSelection(p["finish"])
            x = UsdGeom.Xformable(prim)
            t = prim.GetAttribute("xformOp:translate"); r = prim.GetAttribute("xformOp:rotateZ")
            if not t:
                t = x.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).GetAttr()
            if not r:
                r = x.AddRotateZOp(UsdGeom.XformOp.PrecisionFloat).GetAttr()
            t.Set(Gf.Vec3d(p["origin_xy"][0], p["origin_xy"][1], FLOOR_Z + DROP_DZ)); r.Set(float(p["yaw_deg"]))
            x.GetXformOpOrderAttr().Set(["xformOp:translate", "xformOp:rotateZ"])
    s = _state(); s["layout"] = lay; s["maps"] = {}                    # every bin is empty now (its parts are gone)
    builtins._lab_basket = None; builtins._lab_basket_empty = True
    tl.play(); await _frames(180)                                      # ~3 s sim: the parts settle
    boxes = _boxes(); settled = {}
    for p in lay["parts"]:
        try:
            pose = part_pose(p["name"]); settled[p["name"]] = dict(xyz=[round(v, 4) for v in pose[:3]], at=where_is(pose, boxes),
                                                                   below_floor_mm=round(1000 * (FLOOR_Z - pose[2]), 1))
        except Exception as e:
            settled[p["name"]] = dict(error=repr(e))
    return dict(removed=removed, settled=settled, boxes=boxes, steps=steps_info)


async def layout_run(seed=None, count=8):
    seed = int(time.time()) % 100000 if seed is None else int(seed)
    lay = plan_layout(seed, count)
    run = os.path.join(SOUT, datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + f"_s{seed}_n{count}"); os.makedirs(run, exist_ok=True)
    res = await apply_layout(lay, steps=SPEED["steps_per_frame"])     # 11/10 speed C: the sort flow's layout sets it (default None = unchanged)
    lay.update(run_dir=run, applied=res)
    json.dump(lay, open(os.path.join(run, "layout.json"), "w"), indent=1)
    _state()["layout"] = lay
    n_src = sum(1 for v in res["settled"].values() if v.get("at") == "source")
    ok = n_src == count and all(v.get("below_floor_mm", 99) <= 3.0 for v in res["settled"].values())
    print(f"layout seed {seed}: {lay['counts']} -> {n_src}/{count} in the source basket after the settle"
          f"{'' if ok else '  (CHECK: not all in the basket / through the floor)'}  {run}/layout.json")
    if ok:
        print("READY: call sort()")
    return dict(seed=seed, counts=lay["counts"], in_source=n_src, ok=ok, run_dir=run)


# ------------------------------------------------------------------ sort
def make_class(LivePicker):
    """The 11/08-09 single-pick TestPicker (outputs/tests/20261108_single_pick/run_pick.py), ported: destination bin, PhysX truth of the
    layout's parts (report only), wall clock in mark(), joint speed peak in frames(), the class variant in choose()."""
    class SortPicker(LivePicker):
        dest = None

        def basket_box(self):
            return _box(self.dest)

        def own_bin_box(self, name):
            """11/10 Q3-6 (report only): (bin label, bmin, bmax) of the bin of this part's class (name = layout name, class prefix)."""
            cls = next((c for c in CLASSES if name.startswith(c + "_")), None)
            return None if cls is None else (BINS[cls].rsplit("/", 1)[-1],) + tuple(_box(BINS[cls]))

        def use_bin(self, cls):
            """11/10 Q1 (queue): the next part goes to the bin of ITS class. The map of the bin in hand is stored back into the sort state
            first, then the remembered map of the new bin is handed to run_live (none yet = an empty bin). True = the bin changed."""
            if self.dest == BINS[cls]:
                return False
            st = _state(); prev = getattr(self, "dest_cls", None)
            if prev is not None:
                st["maps"][prev] = getattr(builtins, "_lab_basket", None)
            self.dest = BINS[cls]; self.dest_cls = cls
            builtins._lab_basket = st["maps"].get(cls); builtins._lab_basket_empty = cls not in st["maps"]
            if not hasattr(self, "bin_log"):
                self.bin_log = []
            self.bin_log.append(dict(cls=cls, stored=prev, map_known=builtins._lab_basket is not None))
            return True

        def truth(self):
            from isaacsim.core.experimental.prims import RigidPrim
            out = {}
            for n in getattr(self, "names", None) or part_names():
                pos, _ = self.oc["_cached_prim"](RigidPrim, f"{PARTS}/{n}").get_world_poses()
                out[n] = [round(float(v), 4) for v in pos.numpy()[0]]
            return out

        def truth_q(self):
            """11/11 C2 fix (record only): every part's PhysX quaternion [qw, qx, qy, qz]; truth() stays 3 values (every consumer uses xyz)."""
            from isaacsim.core.experimental.prims import RigidPrim
            out = {}
            for n in getattr(self, "names", None) or part_names():
                _, q_ = self.oc["_cached_prim"](RigidPrim, f"{PARTS}/{n}").get_world_poses()
                out[n] = [round(float(v), 5) for v in q_.numpy()[0]]
            return out

        def mark(self, name, **kw):
            self.ev.append(dict(name=name, sim_s=round(self.f / 60.0, 3), wall_s=round(time.time() - self.t0, 3), **kw))

        async def frames(self, n):
            for _ in range(n):
                await LivePicker.frames(self, 1)
                try:
                    if getattr(self, "_vlim", None) is None:
                        self._vlim = joint_vel_limits(); self.jpeak = 0.0; self.jpeak_at = None
                    ctrl = builtins._arm310d_ik_follow_controller_registry["instance"]
                    qd = np.abs(ctrl._art.get_dof_velocities().numpy()[0][:6])
                    r = float(np.max(qd / self._vlim)) if self._vlim is not None else float("nan")
                    if r > self.jpeak:
                        self.jpeak = r; self.jpeak_at = (self.ev[-1]["name"] if self.ev else None, [round(float(v), 3) for v in qd])
                except Exception as e:
                    self.jpeak_err = repr(e)

        def _variant(self, q, target, table_z):
            var = VARIANT_BY_CLASS.get(target)
            if var and q is not None and q["plan"].get("mid_mode") == "M1" and P.get("mid_tube_m1"):   # 11/11 fix item 2 (2): M1 = across the tube, no turn
                var = dict(var, dyaw_deg=0.0)
            if q is not None and var:
                pl = q["plan"]; g = list(pl["tcp_grasp"]); q["plan_before_variant"] = dict(tcp_grasp=list(g), yaw_deg=pl["yaw_deg"], open_deg=pl.get("open_deg"))
                if "z_above_floor_m" in var:
                    g[2] = table_z + var["z_above_floor_m"]
                g[2] += var.get("dz_m", 0.0); pl["tcp_grasp"] = g
                if var.get("dyaw_deg"):
                    pl["yaw_deg"] = pl["yaw_deg"] + var["dyaw_deg"]
                if var.get("width_mm"):
                    q["width"] = float(var["width_mm"])
                    pl["open_deg"] = self.planner.open_deg_for_gap(min(self.planner.FINGER_GAP_OPEN_M, q["width"] / 1000.0 + self.planner.OPEN_MARGIN_M))
                q["variant"] = dict(var)
            return q

        def _alt_plan(self, live, q, target, table_z, fails):
            """11/10: the same part again, but only with a closing-axis yaw that differs from every earlier failed one (the planner's own
            candidates and neighbour check, plan_grasp, with candidate_yaws filtered). None = no such yaw is free."""
            o = q["obj"]; others = [x for x in self._objs(live) if x["_tid"] != o["_tid"]]
            dy = (VARIANT_BY_CLASS.get(target) or {}).get("dyaw_deg", 0.0); xy = q["plan"]["tcp_grasp"][:2]
            orig = self.planner.candidate_yaws
            keep = lambda y: retry_change(fails, xy, (((y + 90.0) % 180.0) - 90.0) + dy) == "yaw"
            try:
                self.planner.candidate_yaws = lambda fp, _o=orig: [y for y in _o(fp) if keep(y)]
                pl = self.planner.plan_grasp(o, others, table_z)
            finally:
                self.planner.candidate_yaws = orig
            if not pl["ok"]:
                return None
            width = next((t["width_mm"] for t in pl["tried"] if abs(t["yaw_deg"] - round(pl["yaw_deg"], 1)) < 0.2), None)
            return self._variant(dict(q, plan=pl, width=width), target, table_z)

        def choose(self, live, target, tcp_xy, table_z, exclude_ids, failed_keys=None):
            """Baseline choose (+ the class variant). 11/10: a part that failed before is tried again only with another yaw or position
            (retry_change); if the planner has no other free yaw it is abandoned (PartFails.abandon) and the next part is chosen."""
            excl = set(exclude_ids); self.retry_info = None
            while True:
                q, n = LivePicker.choose(self, live, target, tcp_xy, table_z, excl, failed_keys)
                if q is None:
                    break
                tgt = target if target is not None else q["obj"]["class_name"]     # 11/10 Q1: queue (target None) = the part's own class
                q = self._variant(q, tgt, table_z)
                cxy = q["obj"]["center_xyz_m"][:2]
                fails = failed_keys.near(cxy) if isinstance(failed_keys, PartFails) else []
                ch = retry_change(fails, q["plan"]["tcp_grasp"][:2], q["plan"]["yaw_deg"])
                if ch is None:
                    q2 = self._alt_plan(live, q, tgt, table_z, fails)
                    if q2 is None:
                        failed_keys.abandon(cxy, "a retry would repeat the failed yaw and position; no other free yaw")
                        if target is None:
                            failed_keys.records[-1]["cls"] = tgt                 # 11/10 Q1: one PartFails for every class -> keep the class
                        excl.add(q["tid"]); continue
                    q = q2; ch = retry_change(fails, q["plan"]["tcp_grasp"][:2], q["plan"]["yaw_deg"])
                q["retry_change"] = ch; self.retry_info = dict(change=ch, earlier_failures=len(fails), yaw_deg=round(float(q["plan"]["yaw_deg"]), 2))
                break
            if q is not None:                                                  # report only (never used for a decision)
                tr = self.truth()
                self.target_name = min(tr.items(), key=lambda kv: np.hypot(kv[1][0] - q["obj"]["center_xyz_m"][0], kv[1][1] - q["obj"]["center_xyz_m"][1]))[0] if tr else None
                if not hasattr(self, "target_names"):
                    self.target_names = {}
                self.target_names[q["tid"]] = self.target_name                 # 11/10 D: per track (one run_live may pick several parts), report only
            return q, n
    return SortPicker


def joint_vel_limits():
    """rad/s of the 6 arm joints from USD physxJoint:maxJointVelocity (deg/s), in articulation dof order."""
    import omni.usd
    from pxr import Usd
    ctrl = builtins._arm310d_ik_follow_controller_registry["instance"]
    names = list(ctrl._art.dof_names)[:6]
    st = omni.usd.get_context().get_stage(); lim = {}
    for p in Usd.PrimRange(st.GetPrimAtPath("/World/ArmWithHandOnly")):
        a = p.GetAttribute("physxJoint:maxJointVelocity")
        if a and a.HasAuthoredValue() and p.GetName() in names:
            lim[p.GetName()] = float(a.Get()) * np.pi / 180.0
    return np.array([lim[n] for n in names]) if len(lim) == 6 else None


async def to_pose(oc, p1, q1, secs=3.0):
    import omni.kit.app, omni.timeline
    app = omni.kit.app.get_app(); tl = omni.timeline.get_timeline_interface()
    p0, q0 = oc["_pose"]("/World/IKTarget"); ts = tl.get_current_time()
    while True:
        await app.next_update_async()
        u = min(1.0, (tl.get_current_time() - ts) / secs); s = u * u * (3 - 2 * u)
        oc["_pose"]("/World/IKTarget", (np.asarray(p0) + s * (np.asarray(p1) - np.asarray(p0))).tolist(), oc["_slerp"](q0, q1, s))
        if u >= 1.0:
            break
    for _ in range(60):
        await app.next_update_async()


async def back_to_view_a(oc, va, pk, tol_m=0.03):
    """11/10: after an error, the sort's own to_pose leg back to view A; ok when the TCP is within tol_m of it (unverified offline)."""
    try:
        await to_pose(oc, va["position_m"], va["quaternion_wxyz"])
        d = math.dist(pk.tcp()[0], va["position_m"])
        return d <= tol_m, d
    except Exception:
        return False, None


def split_run_live(res):
    """11/10 D (pure): a run_live result with several picks -> one result-like dict per pick, so every pick is one record (system and report
    columns per part). Events of pick k are those named 'k<k>:...'; 'scan_done' of piece k = its own leg start (the first piece keeps the real
    scan_done), 'home' = the next piece's leg start (the last keeps the real home): turn_wall of a piece = its own share of the time.
    One pick (or none) -> [res] unchanged."""
    cyc = res.get("cycles") or []
    if len(cyc) <= 1:
        return [res]
    ev = res.get("events", []); out = []
    starts = {c["k"]: next((e for e in ev if e["name"] == f"k{c['k']}:leg_start"), None) for c in cyc}
    for i, c in enumerate(cyc):
        mine = [e for e in ev if e["name"].startswith(f"k{c['k']}:")]
        first = dict(starts[c["k"]] or {}, name="scan_done") if i > 0 else next((e for e in ev if e["name"] == "scan_done"), None)
        nxt = starts.get(cyc[i + 1]["k"]) if i + 1 < len(cyc) else next((e for e in ev if e["name"] == "home"), None)
        last = dict(nxt, name="home") if nxt is not None else None
        pre = [e for e in ev if not e["name"].startswith("k") and e["name"] not in ("scan_done", "home")] if i == 0 else []
        events = pre + ([first] if first else []) + mine + ([last] if last else [])
        out.append(dict(res, cycles=[c], events=events, piece=i))
    return out


def summarize(res, cls):
    """run_pick.summarize: one run_live (= one cycle) -> one picks[] row of check_acceptance.py."""
    ev = {e["name"]: e for e in res.get("events", [])}
    s = dict(cls=cls, run_dir=res.get("run_dir"), error=res.get("error"), cycles=[])
    a, b = ev.get("scan_done"), ev.get("home")
    if a and b:
        s["turn_sim_s"] = round(b["sim_s"] - a["sim_s"], 2); s["turn_wall_s"] = round(b["wall_s"] - a["wall_s"], 2)
    s["legs"] = [dict(name=e1["name"], sim_s=round(e1["sim_s"] - e0["sim_s"], 2), wall_s=round(e1["wall_s"] - e0["wall_s"], 2))
                 for e0, e1 in zip(res.get("events", []), res.get("events", [])[1:]) if "wall_s" in e0 and "wall_s" in e1]
    for c in res.get("cycles", []):
        s["cycles"].append({k: c.get(k) for k in ("k", "cls", "conf", "ok", "plan_grasp", "grasp_err_mm", "grip_width_mm", "width_mismatch", "tug", "cell",
                                                  "holding_at_place", "from_arrive_to_release_s", "retargets",
                                                  "grasp_yaw_deg", "place_yaw_deg", "spot", "place_search_ms", "outline", "place_changes", "wrist_guard", "release_gap_mm",
                                                  "place_verified", "release_width", "spot_recheck", "width_check", "retry_change", "outline_diag", "finger_at_grasp")})
        for k_ in ("round_yaw", "round_place"):                            # 11/11 Q4 records (only when the switch made them)
            if c.get(k_) is not None:
                s["cycles"][-1][k_] = c[k_]
        for k_ in ("hand_at_close", "tcp_at_close", "part_at_close"):      # 11/11 R5 A1 records for tilt_check (only when a rule switch made hand_at_close)
            if c.get("hand_at_close") is not None and c.get(k_) is not None:
                s["cycles"][-1][k_] = c[k_]
    s["in_dest_after"] = res.get("in_basket_after"); s["truth_before"] = res.get("truth_before"); s["truth_after"] = res.get("truth_after")
    ae_ = res.get("after_error") or {}
    if any(k_ in ae_ for k_ in ("truth_before", "truth_after", "in_own_bin_after")):   # 11/11 corner 1a (scoring, report only): the error path's truth
        s["after_error_scoring"] = {k_: ae_.get(k_) for k_ in ("truth_before", "truth_after", "in_own_bin_after")}   # (check_acceptance reads it)
    if res.get("in_own_bin_after") is not None:
        s["in_own_bin_after"] = res["in_own_bin_after"]                  # 11/10 Q3-6 report only: each part against its own class's bin
    s["names"] = [e["name"] for e in res.get("events", [])]
    return s


TRUTH_KEYS = ("physical", "physical_target_in_dest", "in_dest_after", "truth_before", "truth_after", "poses_after", "target_name", "grasp_err_mm")
SAME_PART_M = 0.025                  # two grasp plans of one class within 25 mm = the same part (the arm's own failed_keys use a 20 mm grid)


def _err_line(e):
    return (str(e).strip().splitlines() or [""])[-1][:200]


def pick_left(ce, cls):
    """11/11 R5 A1 (pure, report text only): (n, text) when cls ended "nothing graspable" and every recorded skip of cls was the A1 selection
    ("needs a tilted grasp" = the motion is A2 / "no grasp direction meets the rules"); else (0, None). Changes no decision."""
    if "nothing graspable" not in str((ce or {}).get("why")):
        return 0, None
    sk = [x for x in ((ce or {}).get("skipped") or []) if x.get("cls") == cls]
    keys = ("needs a tilted grasp", "no grasp direction meets the rules")
    if not sk or not all(str(x.get("reason", "")).startswith(keys) for x in sk):
        return 0, None
    n = len({x.get("track") for x in sk})
    kinds = sorted({k for x in sk for k in keys if str(x.get("reason", "")).startswith(k)})
    return n, " / ".join(("needs a tilted grasp (motion not in lab yet)" if k == keys[0] else "no direction meets L1-L3 + reach") for k in kinds)


def reach_left(doc, cls, ce):
    """11/11 R5 step 2b (pure, report text only; supervisor: "say it as it is, not 'no usable frames'"): the number of parts of cls the reach
    table kept out, when cls ended "nothing graspable" and every recorded skip of cls was "unreachable (reach table)"; else 0. Changes no decision."""
    rr = (doc.get("reach") or {}).get("unreachable") or []
    if not rr or "nothing graspable" not in str((ce or {}).get("why")):
        return 0
    sk = [x for x in ((ce or {}).get("skipped") or []) if x.get("cls") == cls]
    if not sk or not all(str(x.get("reason", "")).startswith("unreachable (reach table)") for x in sk):
        return 0
    return sum(1 for r in rr if r.get("cls") == cls)


def honest_counts(doc):
    """11/10 stage 0: what the SYSTEM did, from a sort result doc. Pure function; reads only system fields (never TRUTH_KEYS: principle 19).
    Part identity = class + grasp plan position (within SAME_PART_M), as the arm itself identifies a part. Expected part counts come from the
    layout plan names (doc['parts'], the test input), not from PhysX."""
    picks = doc.get("picks") or []; looks = doc.get("empty_looks") or []
    classes = list((doc.get("settings") or {}).get("class_order") or CLASSES)
    expected = {c: sum(1 for n in (doc.get("parts") or []) if n.startswith(c + "_")) for c in classes}
    fallback, no_spot, box_full, pv_none, placed, errors, attempts = [], [], [], [], [], [], []
    for i, p in enumerate(picks, 1):
        names = p.get("names") or []; cyc = p.get("cycles") or []
        if p.get("error"):
            errors.append(dict(cycle=i, cls=p.get("cls"), error=_err_line(p["error"])))
        for c in cyc:
            sp = c.get("spot") or {}
            went = any(n.endswith(":at_place") for n in names)
            if sp.get("fallback"):
                fallback.append(i)
            if sp.get("no_spot"):
                no_spot.append(i)
            if sp.get("box_full"):
                box_full.append(i)
            if went:
                placed.append(i)
                if c.get("place_verified") is None:
                    pv_none.append(i)
            g = c.get("plan_grasp")
            if g is not None and any(n.endswith(":leg_start") for n in names):
                attempts.append(dict(cycle=i, cls=p.get("cls"), xy=[float(g[0]), float(g[1])], ok=bool(c.get("ok")),
                                     why=None if c.get("ok") else ("height not reached" if c.get("height_miss") else ("width check" if c.get("width_mismatch") else "not holding")),
                                     grip=c.get("grip_width_mm"), retry=c.get("retry_change")))
        if not cyc and any(n.endswith(":leg_start") for n in names):   # errored before 11/09 kept the cycle rows: an attempt, position unknown
            att_ok = any(n.endswith(":released") for n in names)
            attempts.append(dict(cycle=i, cls=p.get("cls"), xy=None, ok=att_ok, why=None if att_ok else "error", grip=None))
            if any(n.endswith(":at_place") for n in names):
                placed.append(i); pv_none.append(i)
    parts = []                                                          # group attempts into parts
    for a in attempts:
        prt = None
        if a.get("retry") in ("yaw", "position"):                       # 11/10: the system matched this attempt to an earlier failure (PartFails at choose):
            prt = next((q for q in reversed(parts) if q["cls"] == a["cls"] and not q["tries"][-1]["ok"]), None)   # same part, even if it moved
        if prt is None and a["xy"] is not None:
            prt = next((q for q in parts if q["cls"] == a["cls"] and q["xy"] is not None and math.dist(q["xy"], a["xy"]) <= SAME_PART_M), None)
        if prt is None:
            prt = dict(cls=a["cls"], xy=a["xy"], tries=[]); parts.append(prt)
        prt["tries"].append(a); prt["xy"] = a["xy"]
    abandoned = []
    for q in parts:
        if not q["tries"][-1]["ok"]:
            fails = [t for t in q["tries"] if not t["ok"]]
            abandoned.append(dict(cls=q["cls"], grasp_xy=None if q["xy"] is None else [round(v, 4) for v in q["xy"]], failed_cycles=[t["cycle"] for t in fails],
                                  reason="; ".join(f"cycle {t['cycle']}: {t['why']} (closed {(t['grip'] or {}).get('closed')} mm, planned {(t['grip'] or {}).get('planned')} mm)" for t in fails)))
    by_pos = list(abandoned)
    if isinstance(doc.get("abandoned"), dict):                          # 11/10: the sort's own list (PartFails: failed twice / no other grasp) decides
        abandoned = [dict(cls=c, grasp_xy=a.get("xy"), failed_cycles=None, reason="; ".join(str(f.get("why") or f.get("abandoned")) for f in a.get("failures", [])))
                     for c, lst in doc["abandoned"].items() for a in lst]
    class_end = {}
    for c in classes:
        seq = [(i, p) for i, p in enumerate(picks, 1) if p.get("cls") == c]
        lk = next((l for l in looks if l.get("cls") == c), None); ce = (doc.get("class_ends") or {}).get(c)
        att = sum(1 for q in parts if q["cls"] == c)
        row = dict(expected_by_layout=expected.get(c), cycles=len(seq), parts_attempted=att, never_attempted=max(0, (expected.get(c) or 0) - att))
        if ce is not None:                                              # 11/10 speed A: ended by the view-A count (shared tracker), no empty look
            lks, li = doc.get("looks") or [], ce.get("look"); lk_ = lks[li] if isinstance(li, int) and 0 <= li < len(lks) else None
            ok_look = ce.get("look_valid") if "look_valid" in ce else bool(lk_ and lk_.get("frames", 0) >= lk_.get("k", 1))
            row["end"] = ce.get("why") + ("" if ok_look else " -- but that look had no usable frames (看不到畫面): not a count")
            row["view_a_clear"] = ce.get("n") == 0 and bool(ok_look)
            n_rt = reach_left(doc, c, ce)                                  # 11/11 R5 step 2b (report text only): every skip of this class was the reach table
            if n_rt:
                row["end"] = f"{n_rt} left unreachable (reach table)"; row["reach_left"] = n_rt
            n_ps, why_ps = pick_left(ce, c)                                    # 11/11 R5 A1 (report text only): every skip of this class was the rules / tilt
            if n_ps:
                row["end"] = f"{n_ps} left: {why_ps}"; row["pick_left"] = n_ps
        elif lk is not None:
            nt = lk.get("n_target")
            row["end"] = ("view A: nothing of the class seen" if nt == 0 else
                          f"view A: {nt} seen but none graspable" if nt else "view A: nothing graspable (n seen not recorded)")
            row["view_a_clear"] = nt == 0
        elif not seq:
            row["end"] = "never started"; row["view_a_clear"] = False
        else:
            last = seq[-2:]
            if len(last) == 2 and all(not p.get("system_ok") for _, p in last):
                row["end"] = "2 cycles in a row without a system ok" + (" (errors)" if all(p.get("error") for _, p in last) else "")
            else:
                row["end"] = "sort stopped (" + str((doc.get("system") or {}).get("stop") or "max_cycles / exception") + ")"
            row["view_a_clear"] = False
        class_end[c] = row
    out = dict(cycles=len(picks), placements=len(placed), fallback=len(fallback), fallback_cycles=fallback, no_spot=len(no_spot), box_full=len(box_full),
               place_verified_none=len(pv_none), place_verified_none_of=len(placed), place_verified_none_cycles=pv_none,
               error_cycles=len(errors), errors=errors, abandoned=abandoned, n_abandoned=len(abandoned), abandoned_by_position=by_pos, class_end=class_end,
               never_attempted={c: r["never_attempted"] for c, r in class_end.items() if r["never_attempted"]})
    fc = doc.get("final_checks") or []
    fin_looks = [l for l in (doc.get("looks") or []) if str(l.get("at", "")).startswith("final")]
    fc_valid = (fc[-1].get("look_valid") if fc and "look_valid" in fc[-1] else (bool(fin_looks) and fin_looks[-1].get("frames", 0) >= fin_looks[-1].get("k", 1))) if fc else True
    final_ok = not fc or (not any((fc[-1].get("counts") or {}).values()) and fc_valid)
    blind = [l for l in (doc.get("looks") or []) if "frames" in l and l.get("frames", 0) < l.get("k", 1)]
    clean = not (fallback or pv_none or abandoned or errors or blind) and all(r["view_a_clear"] for r in class_end.values()) and final_ok
    out["blind_looks"] = [l.get("at") for l in blind]
    out["final_check"] = fc[-1] if fc else None
    out["autonomous_complete"] = bool(clean)
    if clean:
        out["end_reason"] = "nothing left"
    else:
        why = []
        if fallback: why.append(f"fallback spot {len(fallback)}x (cycles {fallback})")
        if pv_none: why.append(f"place not verified {len(pv_none)}/{len(placed)} placements")
        if abandoned: why.append(f"{len(abandoned)} parts abandoned")
        if errors: why.append(f"{len(errors)} cycles with an error")
        why += [f"{c}: {r['end']}" for c, r in class_end.items() if not r["view_a_clear"]]
        if blind:
            why.insert(0, f"view A saw no usable frames (看不到畫面) at {len(blind)} look(s): {[l.get('at') for l in blind]}")
        if not final_ok:
            why.append(f"final view-A check still counts {fc[-1].get('counts')}" if fc_valid else "final view-A check had no usable frames (看不到畫面)")
        out["end_reason"] = "; ".join(why)
    return out


# ------------------------------------------------------------------ 11/10 stage 2: a failed part is tried again from view A (system data only)
class PartFails(dict):
    """Failed grasps per part, kept by sort across cycles of one class and handed to run_live(failed_keys=...).
    run_live's own skip rule reads get(key) >= 2 with key = object centre on its 20 mm grid; here a key counts every failure recorded within
    NEAR_M of that grid point (a part seen a few mm apart from one view to the next is the same part; parts are > 60 mm apart).
    Records come from run_live (record()) or from SortPicker.choose (abandon(): no other yaw or position to try)."""
    NEAR_M = 0.035

    def __init__(self):
        super().__init__(); self.records = []

    def __bool__(self):
        return bool(self.records)

    def near(self, xy):
        return [r for r in self.records if math.dist(r["xy"], xy) <= self.NEAR_M]

    def get(self, fk, default=0):
        n = sum(r.get("count", 1) for r in self.near((fk[0] * 0.02, fk[1] * 0.02)))
        return n or default

    def record(self, xy, **info):
        self.records.append(dict(xy=[round(float(xy[0]), 4), round(float(xy[1]), 4)], count=1, **info))

    def abandon(self, xy, why):
        self.records.append(dict(xy=[round(float(xy[0]), 4), round(float(xy[1]), 4)], count=2, abandoned=why))

    def abandoned(self):
        out = []
        for r in self.records:
            if any(math.dist(r["xy"], a["xy"]) <= self.NEAR_M for a in out):
                continue
            grp = self.near(r["xy"])
            if sum(g.get("count", 1) for g in grp) >= 2:
                out.append(dict(xy=r["xy"], cls=r.get("cls"), failures=[{k: v for k, v in g.items() if k != "xy"} for g in grp]))
        return out


def retry_change(fails, xy, yaw_deg, pos_m=0.005, yaw_tol_deg=5.0):
    """What a new attempt changes against the earlier failed attempts of the same part: 'yaw' / 'position' / 'first try';
    None = the same yaw (closing axis, mod 180) AND the same position as an earlier failure -> must not be retried."""
    if not fails:
        return "first try"
    same = lambda f: (abs(((yaw_deg - f["yaw"]) + 90.0) % 180.0 - 90.0) < yaw_tol_deg and math.dist(xy, f["grasp_xy"]) < pos_m)
    if any(same(f) for f in fails if f.get("yaw") is not None):
        return None
    return "yaw" if all(abs(((yaw_deg - f["yaw"]) + 90.0) % 180.0 - 90.0) >= yaw_tol_deg for f in fails if f.get("yaw") is not None) else "position"


def after_cycle(empty_look, error):
    """Sort's decision after one run_live of a class: 'class_end' only when view A found nothing graspable (empty look);
    'recover' after an error (back to view A, then go on); else 'continue'. A failed or abandoned part never ends the class."""
    if error:
        return "recover"
    return "class_end" if empty_look else "continue"


# ------------------------------------------------------------------ 11/10 speed A / C
SPEED = dict(steps_per_frame=3, final_checks_max=2,              # sort always shares one Live (11/10 speed B)
             chain_same_class=True)                              # 11/10 D: True = run_live(cls, max_picks=view-A count): same class one after another
STAGE3 = dict(free_corner_limit=True, a_free_corner=8.0,        # 11/11 Lin "Yes, use 1.0"（轉自 PM-Reviewer），驗證：Q2–Q5、T1: default on (was False / 4.0). 11/10 Q3 (Lin 23:5x "first try stage 3"); all off = as before
              short_leg_a=True, approach_a=1.2, leave_slow_a=2.0,  # short legs: approach a 0.6 -> 1.2 (to_grasp, lift_slow, carry_to_place), leave_slow 1.0 -> 2.0
              report_defer=True,                                  # 11/11 Lin "Yes, use 1.0"（轉自 PM-Reviewer），驗證：Q2–Q5、T1 (was False). timing of run_live's bookkeeping only (record), no behaviour change
              fallback_outline_check=True)                        # 11/11 Lin "Yes, use 1.0"（轉自 PM-Reviewer），驗證：Q2–Q5、T1 (was False). Q3-5 (Lin 11/11 "fix 1"): the fallback place point checked like a normal place
Q4 = dict(round_yaw_along_carry=True,                            # 11/11 Lin "Yes, use 1.0"（轉自 PM-Reviewer），驗證：Q2–Q5、T1 (was False). 11/11 Q4 A (Lin "A + B"): flange_round closing axis along grasp -> place
          round_no_wrist_turn=True)                              # 11/11 Lin "Yes, use 1.0"（轉自 PM-Reviewer），驗證：Q2–Q5、T1 (was False). 11/11 Q4 B: flange_round place yaw with the smallest wrist turn
PICK_RULES = dict(pick_wall_check=True, tilt_grasp=True)            # 11/11 1e (supervisor: plan C defaults, GUI r5c; were False). R5 A1 (Lin's L1-L3): source-wall check / tilted-grasp selection
C2_RULES = dict(tilt_keep=True, plate_raise_end=True, viewa_all_pts=True, truth_quat=True)       # 11/11 1e: on (were False). C2 (Lin "do it"): no turn after a tilted grasp,
                                                                  # the 25 deg end raised over the plate (2-6 mm), record-only points / part quaternions; off
TUBE_RULES = dict(vertical_tube_rule=True, mid_tube_m1=True)     # 11/11 1e: on (were False). fix item 2 (Lin "aim at the tube"): (1) flange + end one vertical-tube rule, (2) mid M1 first
CORNER290 = dict(keep080=True, carry_sag_m=0.0126)               # 11/12 seed 290 fix (Lin "O2 + O1 ... use this"): O2 keep080 = a lowered-carry direction
                                                                  # uses the kept tilt that reaches 0.80 (reach_table KEEP080, with tilt_keep); O1 carry_sag_m =
                                                                  # the measured carry sag added to the crate-wall clearance. off: keep080 False / carry_sag_m None
WIDTH = dict(tube_width_pct=0.5)                                  # 11/12 W1 (supervisor; engineer width report): the tube's centre / width from the
                                                                  # 0.5-99.5 percentiles of its view-A top layer (2-98 cut the ring's ends, -3 mm); None = 2-98
REACH = dict(reach_table=False)                                  # 11/11 R5 step 2 (c): view-A reach table (lab/reach_table.py); default off (Lin decides after step 4)
R13 = dict(recover_unpause=True,                                  # 11/11 R1 (P6c diag 3): after a run_live error the shared Live's paused is put back (False = as before)
           exception_logs=True)                                   # 11/11 R3: the error path also writes truth_watch / tcp_log / obs_log (record only)
SPEED["queue"] = "distance"                                      # 11/11 Lin "Yes, use 1.0"（轉自 PM-Reviewer），驗證：Q2–Q5、T1 (was None). 11/10 Q1: "distance" = one queue over every class, nearest the arm base first (queue_sort)
MFR_KEY = "/persistent/simulation/minFrameRate"


def next_class(order, counts, ended, valid=True):
    """Speed A (pure): classes view A counts 0 now end at once (no empty look); returns (the first class in the fixed order still counted, newly ended).
    11/10 fix: valid=False (the look got fewer usable frames than asked) ends NOTHING -- 'saw nothing' is not 'counted 0' (sort 145729)."""
    if not valid:
        return None, []
    newly = [c for c in order if c not in ended and not counts.get(c)]
    left = [c for c in order if c not in ended and c not in newly]
    return (left[0] if left else None), newly


def final_revisit(order, counts):
    """Speed A (pure): the one look over every class before the end; a class still counted (e.g. hidden before) is picked again."""
    return [c for c in order if counts.get(c)]


def set_steps_while_stopped(n):
    """Speed C, the 11/08 run_spf.py settings (time codes/s = 60/n, minFrameRate = 60/n; physics stays 60 Hz), applied ONLY while the timeline is
    stopped: changed during play (sort 11/10 14:57) PhysX deleted its physics data and every tensor view (the IK follower's) was invalidated.
    Called by apply_layout between its stop and play (the views are rebuilt on play); kept until the next layout. n=1 = back to 60/60.
    Returns the values before / after and the root layer dirty flag (never saved)."""
    import carb, omni.timeline, omni.usd
    tl = omni.timeline.get_timeline_interface(); cs = carb.settings.get_settings(); rl = omni.usd.get_context().get_stage().GetRootLayer()
    if not tl.is_stopped():
        raise RuntimeError("set_steps_while_stopped: the timeline is not stopped (changing it during play invalidates the physics views)")
    info = dict(n=int(n), tcps_before=tl.get_time_codes_per_seconds(), mfr_before=cs.get(MFR_KEY), root_dirty_before=rl.dirty)
    tl.set_time_codes_per_second(60.0 / n); cs.set(MFR_KEY, int(60 / n))
    info.update(tcps=tl.get_time_codes_per_seconds(), mfr=cs.get(MFR_KEY), root_dirty_after=rl.dirty)
    return info


def timeline_steps():
    """The current setting, read only (for the sort result)."""
    import carb, omni.timeline
    tl = omni.timeline.get_timeline_interface(); tc = tl.get_time_codes_per_seconds(); mfr = carb.settings.get_settings().get(MFR_KEY) or 60
    return dict(tcps=tc, mfr=mfr, steps_per_frame=max(1, int(round(60.0 * min(1.0 / tc, 1.0 / mfr)))))


def _acceptance():
    p = os.path.join(SROOT, "tools", "check_acceptance.py"); g = {"__name__": "check_acceptance", "__file__": p}
    exec(compile(open(p, encoding="utf-8").read(), p, "exec"), g); return g


def split_fails(fq, classes=CLASSES):
    """11/10 Q1 (pure): the one PartFails of the queue -> one per class (records by their 'cls'), so the sort's per-class
    part_fails / abandoned keep their form."""
    out = {}
    for c in classes:
        f = PartFails(); f.records = [r for r in fq.records if r.get("cls") == c]; out[c] = f
    return out


def view_a_reason(c, last):
    """11/10 Q1 (pure, record only): why the arm went back to view A after this part. Lin's three: 'pick failed', 'slip in the tug test',
    'next part within the finger range'; else None (flew straight on), or 'round end' for the last part of a run_live (all done / nothing
    graspable: run_live goes home = view A, then the final check)."""
    why = (c.get("chain") or {}).get("back_to_view_a")
    if why is None and not c.get("ok") and c.get("chain") is None:
        why = "pick failed"
    return why if why is not None else ("round end" if last else None)


def pick_note(doc, res, where):
    """11/11 R5 A1 (record only): run_live's grasp selections (pick_select), one row per class + rounded grasp xy (a part seen again later is
    the same row), with the chosen yaw / tilt or why none."""
    rows = (res or {}).get("pick_select")
    if not rows:
        return
    ps = doc.setdefault("pick_select", dict(rows=[], n_needs_tilt=0, n_no_direction=0))
    for r in rows:
        ch = r.get("chosen") or {}
        ps["rows"].append(dict(where=where, track=r.get("track"), cls=r.get("cls"), chosen_yaw=ch.get("yaw"), tilt=ch.get("tilt", 0) if ch else None,
                               wall_mm=ch.get("wall_mm"), floor_mm=ch.get("floor_mm"), pad_minus_centroid_mm=r.get("pad_minus_centroid_mm"),
                               z_transfer=ch.get("z_transfer"), transit_080=ch.get("transit_080"),
                               n_ok=r.get("n_ok"), why_counts=r.get("why_counts")))
    ps["n_needs_tilt"] = len({(x["cls"], x["track"]) for x in ps["rows"] if x.get("tilt") or (x.get("tilt") == 0 and x.get("transit_080") is False)})   # = the A2 motion (tilt or a lower carry)
    ps["n_no_direction"] = len({(x["cls"], x["track"]) for x in ps["rows"] if x.get("tilt") is None})


def reach_note(doc, res, where):
    """11/11 R5 step 2 (record only): the parts run_live's reach table kept out ("unreachable"), one row per class + table cell (a part seen
    again in a later round / look is the same row; its track id changes)."""
    rows = ((res or {}).get("reach") or {}).get("unreachable")
    if not rows:
        return
    rr = doc.setdefault("reach", dict(unreachable=[], n_unreachable=0))
    for r in rows:
        k = [r.get("cls"), r.get("cell") or r.get("xy")]
        old = next((x for x in rr["unreachable"] if [x.get("cls"), x.get("cell") or x.get("xy")] == k), None)
        if old is None:
            rr["unreachable"].append(dict(r, seen_at=[where]))
        elif where not in old["seen_at"]:
            old["seen_at"].append(where)
    rr["n_unreachable"] = len(rr["unreachable"])


async def queue_sort(doc, live, oc, va, Picker, names, look, max_cycles, w, fq):
    """11/10 Q1, SPEED["queue"] = "distance" (Lin 11/10): at view A every part (any class) in one queue, nearest to the arm base first; one
    part after another, each to the bin of its YOLO class, straight from the bin to the next part; back to view A in the middle only after a
    failed pick, a slip in the tug test or when the next part lies within this grasp's finger range (run_live chain_after_place). When the
    round is done (run_live home = view A) the look there is the final check; parts still counted -> a new round (new queue).
    Ends: counted 0 (valid look), a round that picked nothing, max_cycles, or a look without usable frames. Returns (end_reason, n_cyc).
    One PartFails (fq) for every class: failed twice / no other grasp -> abandoned, not counted (as the class flow)."""
    s = _state(); n_cyc = 0; rnd = 0
    excl_q = lambda tk: fq.get((round(tk["center"][0] / 0.02), round(tk["center"][1] / 0.02)), 0) >= 2
    live.count_exclude = excl_q
    doc["queue_rounds"] = []
    while True:
        if not look.get("valid"):
            doc["view_a_blind"] = doc["looks"][-1]
            return f"view A saw no usable frames (看不到畫面): {look.get('frames')} of {look.get('k')} frames at look '{doc['looks'][-1]['at']}'", n_cyc
        counts = look["counts"]; total = sum(int(v or 0) for v in counts.values())
        if total == 0:
            for c in CLASSES:
                doc["class_ends"][c] = dict(why="queue: view A count 0", n=0, look=len(doc["looks"]) - 1, look_valid=True)
            return None, n_cyc
        if n_cyc >= max_cycles:
            return f"max_cycles {max_cycles}", n_cyc
        rnd += 1
        pk = Picker(); pk.names = names; pk.dest = None; pk.dest_cls = None; pk.bin_log = []
        pk.use_bin(CLASSES[0])                                                 # a start bin for run_live; every part switches to its own
        doc["progress"] = f"queue round {rnd}"; w()
        t0 = time.time()
        try:
            res = await pk.run_live(None, max_picks=total, rec=True, failed_keys=fq, live=live)
        except Exception:
            res = dict(error=traceback.format_exc())
        reach_note(doc, res, f"queue round {rnd}")                             # 11/11 R5 step 2 (record; nothing when the switch is off)
        pick_note(doc, res, f"queue round {rnd}")                              # 11/11 R5 A1 (record; nothing when the switches are off)
        if pk.dest_cls is not None:
            s["maps"][pk.dest_cls] = getattr(builtins, "_lab_basket", None)      # the last bin's map stored back
        cyc = res.get("cycles") or []
        doc["queue_rounds"].append(dict(round=rnd, counts=dict(counts), max_picks=total, run_dir=res.get("run_dir"), arm_base_xy=res.get("arm_base_xy"),
                                        bins=pk.bin_log, n_picks=len(cyc), wall_s=round(time.time() - t0, 2)))
        if not cyc and not res.get("error"):                                   # counted at view A but nothing graspable
            ev = res.get("events", [])
            ng = next((e for e in ev if e["name"] == "nothing_graspable"), {}); sd = next((e for e in ev if e["name"] == "scan_done"), {})
            doc["empty_looks"].append(dict(cls="queue", wall_s=round(time.time() - t0, 2), events=[e["name"] for e in ev][-3:], run_dir=res.get("run_dir"),
                                           n_target=sd.get("n_target"), n_tracks=sd.get("n_tracks"), skipped=ng.get("skipped")))
            for c in CLASSES:
                doc["class_ends"][c] = dict(why="view A: counted but nothing graspable", n=counts.get(c, 0), skipped=ng.get("skipped")) if counts.get(c) else \
                    dict(why="queue: view A count 0", n=0, look=len(doc["looks"]) - 1, look_valid=True)
            return "queue: counted at view A but nothing graspable", n_cyc
        pieces = split_run_live(res)
        for ip_, sub in enumerate(pieces):
            n_cyc += 1
            c0 = (sub.get("cycles") or [{}])[0]; cls = c0.get("cls")
            row = summarize(sub, cls); row["run_live_wall_s"] = round(time.time() - t0, 2); row["total_sim_s"] = res.get("total_sim_s") if ip_ == 0 else 0.0
            row["joint_peak_ratio"] = round(float(getattr(pk, "jpeak", float("nan"))), 3); row["joint_peak_at"] = getattr(pk, "jpeak_at", None)
            row["joint_peak_err"] = getattr(pk, "jpeak_err", None); row["target_name"] = (getattr(pk, "target_names", {}) or {}).get(c0.get("tid"), getattr(pk, "target_name", None))
            tn = row["target_name"]
            row["physical_target_in_dest"] = bool(tn and (row.get("in_dest_after") or {}).get(tn))      # report only (principle 19)
            row["system_ok"] = bool(sub.get("cycles") and c0.get("ok"))
            row["retry_change"] = c0.get("retry_change")
            row["confirm_used"] = res.get("confirm") if ip_ == 0 else None; row["open_skipped_deg"] = res.get("open_skipped_deg") if ip_ == 0 else None
            row["piece"] = dict(index=ip_, of=len(pieces))
            if getattr(pk, "report_timing", None) is not None and ip_ == 0:
                row["report_timing"] = pk.report_timing                     # 11/10 Q3 report_defer: run_live bookkeeping times (s since run_live entry)
            qi = c0.get("queue") or {}
            row["queue"] = dict(round=rnd, pos=qi.get("pos"), of=qi.get("of"), dist_m=qi.get("dist_m"), cls=cls, conf=c0.get("conf"), dest=qi.get("dest"),
                                deferred=qi.get("deferred"), bin_switched=qi.get("bin_switched"), chained_in=qi.get("chained_in"), base_xy=qi.get("base_xy"), next_touch=c0.get("next_touch"),
                                view_a=view_a_reason(c0, ip_ == len(pieces) - 1), turn_wall_s=row.get("turn_wall_s"), legs=row.get("legs"))
            try:                                                                # report only: every part's PhysX pose after this run_live
                row["poses_after"] = {n: [round(v, 5) for v in part_pose(n)] for n in names}
            except Exception as e:
                row["poses_after"] = repr(e)
            doc["picks"].append(row)
        if res.get("error"):                                                    # as the class flow: back to view A, then go on
            back, d_m = await back_to_view_a(oc, va, pk)
            if doc["picks"]:
                doc["picks"][-1]["recovery"] = dict(back_at_view_a=back, tcp_dist_mm=None if d_m is None else round(1000 * d_m, 1))
            if not back:
                return f"queue: error and could not return to view A ({_err_line(res.get('error'))})", n_cyc
        t_l = time.time(); look = await pk.confirm_view(live, exclude=excl_q)       # the round is done, back at view A: the final check
        left = sum(int(v or 0) for v in look["counts"].values())
        at = f"final {rnd}" if (left == 0 or not look.get("valid")) else f"after cycle {n_cyc} (queue round {rnd})"
        doc["looks"].append(dict(at=at, **look))
        if at.startswith("final"):
            doc["final_checks"].append(dict(counts=look["counts"], revisit=[c for c in CLASSES if look["counts"].get(c)], wall_s=round(time.time() - t_l, 2),
                                            look_valid=bool(look.get("valid"))))
        if doc["picks"]:
            doc["picks"][-1]["look_after"] = dict(wall_s=look["wall_s"], frames=look["frames"], decision=look["decision"], counts=look["counts"])
        doc["part_fails"] = {c: f.records for c, f in split_fails(fq).items()}; w()


END_LABELS = {"放對箱（物理真值）": "right bin", "每件一輪牆上時鐘": "per-item time", "關節峰值": "joint peak"}   # 11/12 supervisor: the console is English


def end_lines(doc, n_parts, path):
    """11/12 (Lin: "print the right-bin count first"; console text only -- doc, passed and check_acceptance unchanged). The lines sort() prints at
    the end: (1) the physical result + the time per part (whole sort's wall clock / parts), (2) the target check (doc["passed"]) naming what did
    not pass (doc["acceptance"]["rows"] names; thresholds / the right-bin count / an acceptance error), then the old system / per-item / rules /
    reach / result lines as before. Pure."""
    ph = doc.get("physical") or {}; wc = doc.get("wall_clock") or {}
    t_ = (wc["t_end"] - wc["t_start"]) / n_parts if n_parts and wc.get("t_end") is not None and wc.get("t_start") is not None else None
    out = [f"sort: {ph.get('right_bin')}/{n_parts} parts in the right bin (left in basket {len(ph.get('left_in_source') or [])}, "
           f"wrong bin {len(ph.get('wrong_bin') or [])}, elsewhere {len(ph.get('elsewhere') or [])}); "
           + (f"{t_:.1f} s per part (whole sort / parts)" if t_ is not None else "time per part n/a")]
    acc = doc.get("acceptance") or {}; why = []
    if acc.get("error"):
        why.append("acceptance error")
    why += [END_LABELS.get(r["name"], r["name"]) for r in acc.get("rows") or [] if not r.get("ok")]   # English label; unknown names as they are
    if acc.get("threshold_problems"):
        why.append("thresholds: " + "; ".join(map(str, acc["threshold_problems"])))
    if (ph.get("right_bin") != n_parts or not n_parts) and "right bin" not in why:
        why.append("right bin")
    out.append("target check: " + ("PASSED" if doc.get("passed") else "NOT PASSED" + (f" ({', '.join(why)})" if why else "")))
    walls = [p.get("turn_wall_s") for p in doc["picks"]]
    out.append(f"  system  : {doc['system']['end_reason']}; {doc['system']['system_ok_cycles']}/{doc['system']['cycles']} cycles ok")
    out.append(f"  per item wall s (view A decision -> back at view A): {walls}; joint peak {[p.get('joint_peak_ratio') for p in doc['picks']]}")
    if doc.get("pick_select"):                                                # 11/11 R5 A1
        out.append(f"  rules   : needs a tilted grasp {doc['pick_select']['n_needs_tilt']}, no rule-compliant direction {doc['pick_select']['n_no_direction']} (track ids)")
    if doc.get("reach"):                                                      # 11/11 R5 step 2
        out.append(f"  reach   : {doc['reach']['n_unreachable']} part(s) the reach table kept out: {[(r['cls'], r.get('cell'), r['mode']) for r in doc['reach']['unreachable']]}")
    out.append(f"  result: {path}")
    return out


async def sort_run(ns, base, max_cycles=None):
    """ns = the lab namespace (lab/api2._load_lab + this file); base = grasp/api namespace (_load). Runs inside _schedule()."""
    s = _state(); lay = s.get("layout")
    names = [p["name"] for p in lay["parts"]] if lay else part_names()
    n_parts = len(names); max_cycles = max_cycles or 3 * n_parts           # guard against an endless loop only (recorded, not a failure)
    run = (lay or {}).get("run_dir") or os.path.join(SOUT, datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_nolayout")
    os.makedirs(run, exist_ok=True); path = os.path.join(run, "result.json")
    doc = dict(task="sort", version=SORT_VERSION, started=datetime.datetime.now().isoformat(timespec="seconds"), done=False,
               seed=(lay or {}).get("seed"), count=n_parts, parts=names, settings=dict(P_set=P_SET, P_extra=P_EXTRA, variant_by_class=VARIANT_BY_CLASS,
               class_order=list(CLASSES), class_ends_after="11/10 speed A: view A (shared tracker) counts 0 of the class, or counts it but nothing graspable; one final look over every class before the end", max_cycles=max_cycles),
               picks=[], empty_looks=[], system={}, physical={})

    def w():
        doc["updated"] = datetime.datetime.now().isoformat(timespec="seconds")
        json.dump(doc, open(path + ".tmp", "w"), indent=1, default=str); os.replace(path + ".tmp", path)
    w(); t_sort = time.time(); end_reason = None; live = None
    try:
        ns["P"].update(P_SET); ns["P"].update(P_EXTRA)
        ns["P"]["chain"] = bool(SPEED.get("chain_same_class"))                 # 11/10 D
        if ns["P"]["chain"]:
            ns["P"]["choose_rank"] = "clear"                                  # the view-A queue: the part with the most room first
        if SPEED.get("queue") == "distance":                                   # 11/10 Q1: one queue over every class (chain on: straight to the next part)
            ns["P"]["queue"] = "distance"; ns["P"]["chain"] = True; doc["settings"]["queue"] = "distance"
        else:
            ns["P"].pop("queue", None)                                        # off: P as before (no key)
        for k_ in ("free_corner_limit", "a_free_corner", "leave_slow_a", "report_defer", "fallback_outline_check"):
            ns["P"].pop(k_, None)                                             # 11/10 Q3: off = P as before (no keys)
        if STAGE3.get("free_corner_limit"):
            ns["P"]["free_corner_limit"] = True; ns["P"]["a_free_corner"] = float(STAGE3["a_free_corner"])
        if STAGE3.get("short_leg_a"):                                         # a new dict: P_EXTRA's approach stays as it is
            ns["P"]["approach"] = dict(ns["P"]["approach"], a=float(STAGE3["approach_a"])); ns["P"]["leave_slow_a"] = float(STAGE3["leave_slow_a"])
        if STAGE3.get("report_defer"):
            ns["P"]["report_defer"] = True
        if STAGE3.get("fallback_outline_check"):
            ns["P"]["fallback_outline_check"] = True
        for k_ in ("pick_wall_check", "tilt_grasp"):                           # 11/11 R5 A1: off = no key in P (as before)
            ns["P"].pop(k_, None)
            if PICK_RULES.get(k_):
                ns["P"][k_] = True; doc["settings"].setdefault("pick_rules", {})[k_] = True
        for k_ in ("tilt_keep", "plate_raise_end", "viewa_all_pts", "truth_quat"):     # 11/11 C2: off = no key in P (as before)
            ns["P"].pop(k_, None)
            if C2_RULES.get(k_):
                ns["P"][k_] = True; doc["settings"].setdefault("pick_rules", {})[k_] = True
        for k_ in ("vertical_tube_rule", "mid_tube_m1"):                       # 11/11 fix item 2: off = no key in P (as before)
            ns["P"].pop(k_, None)
            if TUBE_RULES.get(k_):
                ns["P"][k_] = True; doc["settings"].setdefault("pick_rules", {})[k_] = True
        for k_ in ("keep080", "carry_sag_m"):                                 # 11/12 seed 290 O2 + O1: off = no key in P (as before)
            ns["P"].pop(k_, None)
            if CORNER290.get(k_):
                ns["P"][k_] = CORNER290[k_]; doc["settings"].setdefault("corner290", {})[k_] = CORNER290[k_]
        ns["P"].pop("tube_width_pct", None)                                    # 11/12 W1: off = no key in P (as before)
        if WIDTH.get("tube_width_pct") is not None:
            ns["P"]["tube_width_pct"] = float(WIDTH["tube_width_pct"]); doc["settings"]["tube_width_pct"] = float(WIDTH["tube_width_pct"])
        ns["P"].pop("reach_table", None)                                       # 11/11 R5 step 2: off = no key in P (as before)
        if REACH.get("reach_table"):
            ns["P"]["reach_table"] = True; doc["settings"]["reach_table"] = True
        for k_ in ("recover_unpause", "exception_logs"):                       # 11/11 R1/R3: default on = no key in P (live_pick's default); off is written
            ns["P"].pop(k_, None)
            if not R13.get(k_, True):
                ns["P"][k_] = False; doc["settings"].setdefault("r13_off", []).append(k_)
        for k_ in ("round_yaw_along_carry", "round_no_wrist_turn"):          # 11/11 Q4: off = no key in P (as before)
            ns["P"].pop(k_, None)
            if Q4.get(k_):
                ns["P"][k_] = True; doc["settings"].setdefault("q4", {})[k_] = True
        if any(STAGE3.get(k_) for k_ in ("free_corner_limit", "short_leg_a", "report_defer", "fallback_outline_check")):
            doc["settings"]["stage3"] = dict(STAGE3, approach_used=ns["P"].get("approach"))
        oc = base["_load"]("perception/one_click.py"); va = oc["_view_poses"]()[0]
        await to_pose(oc, va["position_m"], va["quaternion_wxyz"])          # home = view A (D7: decide there, come back there)
        doc["speed"] = dict(timeline_steps(), set_by="layout (while stopped)", layout_steps=((lay or {}).get("applied") or {}).get("steps"))   # 11/10 speed C: read only here
        import omni.timeline
        doc["tl_start"] = omni.timeline.get_timeline_interface().get_current_time()
        Picker = make_class(ns["LivePicker"]); n_cyc = 0; fails = {c: PartFails() for c in CLASSES}; doc["part_fails"] = {}
        doc.update(looks=[], class_ends={}, final_checks=[])
        live = ns["Live"](oc); live.fp_diag = bool(ns["P"].get("outline_diag")); await live.start()     # 11/10 speed B: one Live for the whole sort
        excl = lambda tk: fails[tk["class_name"]].get((round(tk["center"][0] / 0.02), round(tk["center"][1] / 0.02)), 0) >= 2 if tk["class_name"] in fails else False
        live.count_exclude = excl                                              # abandoned parts are not counted (else a class would never end)
        pk0 = Picker(); t_l = time.time(); await pk0.scan(live, None, 0.6, 3.0); look = await pk0.confirm_view(live, exclude=excl)
        doc["looks"].append(dict(at="start", scan_wall_s=round(time.time() - t_l, 2), **look)); counts = look["counts"]; ended = set(); finals = 0
        while True:
            if SPEED.get("queue") == "distance":                               # 11/10 Q1: the queue over every class instead of the class loop below
                fq = PartFails(); end_reason, n_cyc = await queue_sort(doc, live, oc, va, Picker, names, look, max_cycles, w, fq)
                fails = split_fails(fq); doc["part_fails"] = {c: f.records for c, f in fails.items()}; w(); break
            if not look.get("valid"):                                          # 11/10 fix: a look without usable frames ends nothing; stop and say so
                end_reason = f"view A saw no usable frames (看不到畫面): {look.get('frames')} of {look.get('k')} frames at look '{doc['looks'][-1]['at']}'"
                doc["view_a_blind"] = doc["looks"][-1]; break
            cls, newly = next_class(list(CLASSES), counts, ended, valid=look.get("valid", False))
            for c in newly:                                                    # 11/10 speed A: counted 0 at view A -> ended, no empty look
                ended.add(c); doc["class_ends"][c] = dict(why="view A count 0", n=0, look=len(doc["looks"]) - 1, look_valid=True)
            if cls is None:
                if finals >= SPEED["final_checks_max"]:
                    break
                finals += 1; t_l = time.time(); pkf = Picker()
                await pkf.scan(live, None, 0.6, 3.0); look = await pkf.confirm_view(live, exclude=excl)
                doc["looks"].append(dict(at=f"final {finals}", scan_wall_s=round(time.time() - t_l, 2), **look)); counts = look["counts"]
                rv = final_revisit(list(CLASSES), counts); doc["final_checks"].append(dict(counts=counts, revisit=rv, wall_s=round(time.time() - t_l, 2), look_valid=bool(look.get("valid"))))
                w()
                if not look.get("valid"):
                    continue                                                    # the top of the loop stops with 'no usable frames'
                if not rv:
                    break
                for c in rv:
                    ended.discard(c); doc["class_ends"].pop(c, None)
                continue
            if n_cyc >= max_cycles:
                end_reason = f"max_cycles {max_cycles}"; break
            fk = fails[cls]
            pk = Picker(); pk.dest = BINS[cls]; pk.names = names
            builtins._lab_basket = s["maps"].get(cls); builtins._lab_basket_empty = cls not in s["maps"]   # one remembered map per bin
            doc["progress"] = f"{cls} cycle {n_cyc + 1}"; w()
            t0 = time.time()
            try:
                res = await pk.run_live(cls, max_picks=max(1, int(counts.get(cls) or 1)) if SPEED.get("chain_same_class") else 1, rec=True, failed_keys=fk, live=live)
            except Exception:
                res = dict(error=traceback.format_exc())
            reach_note(doc, res, cls)                                            # 11/11 R5 step 2 (record; nothing when the switch is off)
            pick_note(doc, res, cls)                                             # 11/11 R5 A1 (record; nothing when the switches are off)
            s["maps"][cls] = getattr(builtins, "_lab_basket", None)
            cyc = res.get("cycles") or []
            act = after_cycle(empty_look=not cyc, error=res.get("error"))
            if act == "class_end":                                             # counted at view A but nothing graspable
                ev = [e["name"] for e in res.get("events", [])]
                sd = next((e for e in res.get("events", []) if e["name"] == "scan_done"), {})
                ng = next((e for e in res.get("events", []) if e["name"] == "nothing_graspable"), {})
                doc["empty_looks"].append(dict(cls=cls, wall_s=round(time.time() - t0, 2), events=ev[-3:], run_dir=res.get("run_dir"),
                                               n_target=sd.get("n_target"), n_tracks=sd.get("n_tracks"), skipped=ng.get("skipped")))
                ended.add(cls); doc["class_ends"][cls] = dict(why="view A: counted but nothing graspable", n=counts.get(cls), skipped=ng.get("skipped"))
            else:
                pieces = split_run_live(res)                                    # 11/10 D: one record per pick (one piece when max_picks = 1)
                for ip_, sub in enumerate(pieces):
                    n_cyc += 1
                    row = summarize(sub, cls); row["run_live_wall_s"] = round(time.time() - t0, 2); row["total_sim_s"] = res.get("total_sim_s") if ip_ == 0 else 0.0
                    row["joint_peak_ratio"] = round(float(getattr(pk, "jpeak", float("nan"))), 3); row["joint_peak_at"] = getattr(pk, "jpeak_at", None)
                    tid_ = ((sub.get("cycles") or [{}])[0]).get("tid")
                    row["joint_peak_err"] = getattr(pk, "jpeak_err", None); row["target_name"] = (getattr(pk, "target_names", {}) or {}).get(tid_, getattr(pk, "target_name", None))
                    tn = row["target_name"]
                    row["physical_target_in_dest"] = bool(tn and (row.get("in_dest_after") or {}).get(tn))      # report only (principle 19)
                    row["system_ok"] = bool(sub.get("cycles") and sub["cycles"][0].get("ok"))
                    row["retry_change"] = getattr(pk, "retry_info", None) if len(pieces) == 1 else ((sub.get("cycles") or [{}])[0]).get("retry_change")
                    row["confirm_used"] = res.get("confirm") if ip_ == 0 else None; row["open_skipped_deg"] = res.get("open_skipped_deg") if ip_ == 0 else None
                    row["piece"] = dict(index=ip_, of=len(pieces))
                    if getattr(pk, "report_timing", None) is not None and ip_ == 0:
                        row["report_timing"] = pk.report_timing             # 11/10 Q3 report_defer: run_live bookkeeping times (s since run_live entry)
                    try:                                                        # report only: every part's PhysX pose after this run_live (top-view review)
                        row["poses_after"] = {n: [round(v, 5) for v in part_pose(n)] for n in names}
                    except Exception as e:
                        row["poses_after"] = repr(e)
                    doc["picks"].append(row)
                if act == "recover":                                            # 11/10: an error (e.g. a leg too slow): back to view A, then go on
                    back, d_m = await back_to_view_a(oc, va, pk)
                    doc["picks"][-1]["recovery"] = dict(back_at_view_a=back, tcp_dist_mm=None if d_m is None else round(1000 * d_m, 1))
                    if not back:
                        end_reason = f"{cls}: error and could not return to view A ({_err_line(res.get('error'))})"
            if end_reason:
                doc["part_fails"] = {c: f.records for c, f in fails.items()}; w(); break
            t_l = time.time(); look = await pk.confirm_view(live, exclude=excl)          # back at view A: confirm the tracks, count every class
            doc["looks"].append(dict(at=f"after cycle {n_cyc} ({cls})", **look)); counts = look["counts"]
            if doc["picks"] and act != "class_end":
                doc["picks"][-1]["look_after"] = dict(wall_s=look["wall_s"], frames=look["frames"], decision=look["decision"], counts=counts)
            doc["part_fails"] = {c: f.records for c, f in fails.items()}; w()
        doc["abandoned"] = {c: f.abandoned() for c, f in fails.items()}
        end_reason = end_reason or "loop finished: every class ended"
    except Exception:
        doc["error"] = traceback.format_exc(); end_reason = "exception"
    finally:
        try:
            if live is not None:
                await live.stop()
        except Exception:
            doc.setdefault("cleanup_errors", []).append(traceback.format_exc())
        try:                                                                    # 11/10 speed C: sim time of the sort = run_live clocks + view-A looks (+ timeline difference)
            import omni.timeline
            doc["sim_time"] = dict(run_live_sum_s=round(sum(p.get("total_sim_s") or 0 for p in doc["picks"]) + sum(l.get("sim_s") or 0 for l in doc.get("looks", [])), 2),
                                   timeline_s=None if doc.get("tl_start") is None else round(omni.timeline.get_timeline_interface().get_current_time() - doc["tl_start"], 2))
        except Exception:
            doc["sim_time"] = dict(error=traceback.format_exc())
        try:
            boxes = _boxes(); fin = {}
            doc["bin_boxes"] = {k: [[round(float(v), 4) for v in c] for c in bx] if isinstance(bx, (list, tuple)) else str(bx) for k, bx in boxes.items()}   # 11/10 P5: record only
            for n in names:
                p = part_pose(n); fin[n] = dict(xyz=[round(v, 4) for v in p[:3]], at=where_is(p, boxes))
            want = {c: BINS[c].rsplit("/", 1)[-1] for c in CLASSES}
            cls_of = lambda n: next(c for c in CLASSES if n.startswith(c + "_"))
            right = [n for n in names if fin[n]["at"] == want[cls_of(n)]]
            doc["physical"] = dict(n=n_parts, right_bin=len(right), wrong_bin=[n for n in names if fin[n]["at"] in want.values() and n not in right],
                                   left_in_source=[n for n in names if fin[n]["at"] == "source"], elsewhere=[n for n in names if fin[n]["at"] == "elsewhere"],
                                   final=fin, note="PhysX truth, report only (never used by the arm)")
        except Exception:
            doc["physical"] = dict(error=traceback.format_exc())
        try:                                                                    # 11/10 stage 0: system counts only (never PhysX truth)
            hc = honest_counts(dict(doc, physical=None))
        except Exception:
            hc = dict(error=traceback.format_exc(), autonomous_complete=False, end_reason="honest_counts failed")
        doc["system"] = dict(stop=end_reason, end_reason=hc.get("end_reason"), autonomous_complete=bool(hc.get("autonomous_complete")),
                             counts=hc, cycles=len(doc["picks"]), system_ok_cycles=sum(1 for p in doc["picks"] if p.get("system_ok")),
                             sort_wall_s=round(time.time() - t_sort, 1))
        doc["wall_clock"] = dict(t_start=t_sort, t_end=time.time(), clock="time.time()")          # 11/10 P5: start / end on the same clock (record only)
        try:
            ca = _acceptance(); cfg, problems = ca["load_cfg"]()
            rows = ca["evaluate"](doc, cfg)
            doc["acceptance"] = dict(threshold_problems=problems, rows=[dict(name=r[0], n=r[1], fails=r[2], ok=r[1] > 0 and not r[2]) for r in rows])
            acc_ok = not problems and all(r[1] > 0 and not r[2] for r in rows)
        except Exception:
            doc["acceptance"] = dict(error=traceback.format_exc()); acc_ok = False
        ph = doc.get("physical", {})
        doc["passed"] = bool(acc_ok and ph.get("right_bin") == n_parts and n_parts > 0)
        doc["done"] = True; w()
        for line_ in end_lines(doc, n_parts, path):                            # 11/12 Lin "print the right-bin count first" (console text only)
            print(line_)
    return dict(passed=doc["passed"], right_bin=doc.get("physical", {}).get("right_bin"), n=n_parts, result=path)
