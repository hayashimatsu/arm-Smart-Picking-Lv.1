"""Pick blocks of one class and drop them into the basket (Isaac Sim Script Editor).

Usage (Script Editor): open this file, edit TARGET / MAX_PICKS at the bottom if needed, Ctrl+Enter.
Or from the console after running it once: pick(target="cube", max_picks=1); pick_status().

One cycle:
  (1) perceive (one_click: view A, B if needed)      -> world-frame centers, class, footprints
  (2) choose nearest target (world-XY distance from arm base), plan yaw (natural, then +90 deg)
  (3) open -> above target -> straight down to the object center (raised if the fingertips would hit the table)
      -> close in steps (contact = finger stops following) -> lift
  (4) up to transit height -> above the basket -> depth picture of the basket -> choose a free cell
  (5) lower -> open (drop) -> up
  (6) back to the start pose
  (7) next target: perceive again (things may have moved)
Results: outputs/pick/<time>/index.html + result.json; status in outputs/pick/latest_status.json.
Stage truth is read ONLY for the report (did the object end up in the basket?), never by the method.
"""
import asyncio, builtins, datetime, json, math, os, sys, time, traceback, types
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # loaded by demo_start.py with __file__ set
OUT = os.path.join(ROOT, "outputs", "pick")
REG = "_pick_registry"
TCP = "/World/ArmWithHandOnly/Robotiq_2F_85/Robotiq_2F_85/base_link/tcp"
ARM_BASE = "/World/ArmWithHandOnly/NHC12_A00/base_link"
BASKET = "/World/container_h20"
TRANSIT_Z = 0.85                  # TCP height for moves between table and basket
DROP_ABOVE_RIM_M = 0.08           # TCP height above the basket rim when opening
GRIPPER_CLOSE_DEG = 47.0          # fully closed (nothing between the fingers)
APPROACH_MARGIN_MM = 6.0          # fast approach: one command to the angle where the pads are still 3 mm/side
                                  # from the object (width from perception), then the slow 1-degree steps
CLOSE_STEP_DEG = 1.0              # close in 1-degree steps ...
CLOSE_STEP_FRAMES = 8             # ... letting each step settle (a free finger follows within these frames)
CONTACT_LAG_DEG = 1.0             # after a settled step the finger is this far behind -> it touches the object
                                  # (detected at low force, before the pads can sink in)
SQUEEZE_DEG = 5.0                 # then the command goes only this far past contact (same 1-degree steps): a fixed,
                                  # small grip force. Full force (47 deg) pushed pads up to 10 mm into spheres and a
                                  # continuous close with a stall rule still went ~3 deg too deep on cuboids.
HOLD_MARGIN_DEG = 1.5             # holding = finger stopped at least this far before full close


def _load(path, **extra):
    g = dict(NO_AUTORUN=True, __file__=path, **extra)
    exec(compile(open(path, encoding="utf-8").read(), path, "exec"), g)
    return g


def _module(path, name):
    m = types.ModuleType(name); m.__file__ = path
    exec(compile(open(path, encoding="utf-8").read(), path, "exec"), m.__dict__)
    return m


def _reg():
    if not hasattr(builtins, REG):
        setattr(builtins, REG, {"task": None, "status": {"state": "idle"}})
    return getattr(builtins, REG)


def _status(**kw):
    st = _reg()["status"]; st.update(kw); st["updated"] = datetime.datetime.now().isoformat(timespec="seconds")
    os.makedirs(OUT, exist_ok=True)
    json.dump(st, open(os.path.join(OUT, "latest_status.json"), "w"), indent=1, default=str)


CLASSES = ("cube", "cuboid", "cylinder", "sphere")


def check_target(target):
    if target not in CLASSES:
        raise ValueError(f"target must be one of {CLASSES}, got {target!r}")


async def ensure_sim(frames=20):
    """Timeline playing with auto-update on (a stalled Replicator step leaves it off -> arm and physics freeze)."""
    import omni.timeline, omni.kit.app
    tl = omni.timeline.get_timeline_interface()
    tl.set_auto_update(True)
    if not tl.is_playing():
        tl.play()
    for _ in range(frames):
        await omni.kit.app.get_app().next_update_async()


class Picker:
    def __init__(self):
        self.oc = _load(os.path.join(ROOT, "perception", "one_click.py"))
        self.oc["CFG"]["open_browser"] = False
        self.planner = _module(os.path.join(ROOT, "grasp", "planner.py"), "grasp_planner")
        self.geo = _module(os.path.join(ROOT, "perception", "geometry.py"), "grasp_geometry")
        self.pm = _module(os.path.join(ROOT, "scripts", "pose_math.py"), "grasp_pose_math")
        self.grip = _load(os.path.join(ROOT, "scripts", "gripper_controller.py"))
        self.ns = None
        self.hold_cmd = GRIPPER_CLOSE_DEG
        self.order_by = "arm_base"
        self.log = []
        self.images = []

    # ---------------------------------------------------------------- helpers
    async def frames(self, n):
        import omni.kit.app
        app = omni.kit.app.get_app()
        for _ in range(n):
            await app.next_update_async()

    def tcp(self):
        return self.oc["_pose"](TCP)

    def step(self, name, **kw):
        rec = dict(step=name, t=round(time.time() - self.t0, 2), **kw)
        self.log.append(rec); _status(step=name)
        return rec

    async def move(self, xyz, quat, label, tol_m=0.002, max_frames=900, carrying=False, slow=None):
        """Interpolate the IK target with a smooth (ease-in/ease-out) profile, then wait for the TCP to arrive.
        carrying=True: slower (about 0.1 m/s peak) so the held object does not slip -- the 2026-09-29 first run
        lost the cube because lift/transit accelerated too hard (a slow 1 cm/20-frame lift held it)."""
        p0, q0 = self.oc["_pose"]("/World/IKTarget")
        slow = carrying if slow is None else slow   # carrying: log the finger angle (holding check);
        step = 0.003 if slow else 0.01               # slow: only the first lift off the table (user 2026-09-30:
        n = max(30 if slow else 20, int(math.dist(p0, xyz) / step))   # the carry to the basket = approach speed)
        for i in range(1, n + 1):
            u = i / n
            t = u * u * (3 - 2 * u)
            self.oc["_pose"]("/World/IKTarget", (np.asarray(p0) + t * (np.asarray(xyz) - np.asarray(p0))).tolist(),
                             self.oc["_slerp"](q0, quat, t))
            await self.frames(2)
        for f in range(max_frames):
            await self.frames(1)
            p, q = self.tcp()
            if math.dist(p, xyz) <= tol_m and self.pm.quat_angle_deg(q, quat) <= 2.0:
                break
        p, q = self.tcp()
        err = [round(1000 * (p[k] - xyz[k]), 1) for k in range(3)]
        extra = {}
        if carrying:
            extra["finger_deg"] = round(self.grip["gripper_status"]()["achieved_deg"], 2)
        return self.step(label, target=[round(v, 4) for v in xyz], tcp=[round(v, 4) for v in p], err_mm=err,
                         arrived=math.dist(p, xyz) <= tol_m, frames=f + 1, **extra)

    async def gripper(self, deg, wait=60):
        self.grip["gripper_set_angle_deg"](deg); await self.frames(wait)
        return self.grip["gripper_status"]()

    async def close(self, width_mm=None):
        """(0) Fast approach (if the object width is known): one command to the angle where each pad is still
        APPROACH_MARGIN_MM/2 from the object; stops short there = the object is wider than measured = contact.
        (1) Close in 1-degree steps; after each step wait until it settles. The first step the finger cannot follow
        (>= CONTACT_LAG_DEG behind) = contact. (2) Squeeze: SQUEEZE_DEG more, in the same steps, then settle.
        holding = contact found and the finger is still held back by the object."""
        cmd = self.grip["gripper_status"]()["achieved_deg"]; contact = None; frames = 0; fast_to = None
        if width_mm:
            fast_to = self.planner.open_deg_for_gap((width_mm + APPROACH_MARGIN_MM) / 1000.0)
            if fast_to > cmd + CLOSE_STEP_DEG:
                cmd = fast_to
                self.grip["gripper_set_angle_deg"](cmd)
                for _ in range(40):
                    await self.frames(1); frames += 1
                    if cmd - self.grip["gripper_status"]()["achieved_deg"] < 0.3:
                        break
                a = self.grip["gripper_status"]()["achieved_deg"]
                if cmd - a >= CONTACT_LAG_DEG:
                    contact = round(a, 2)
        while contact is None and cmd < GRIPPER_CLOSE_DEG:              # (1) approach in steps
            cmd = min(GRIPPER_CLOSE_DEG, cmd + CLOSE_STEP_DEG)
            self.grip["gripper_set_angle_deg"](cmd); await self.frames(CLOSE_STEP_FRAMES); frames += CLOSE_STEP_FRAMES
            a = self.grip["gripper_status"]()["achieved_deg"]
            if cmd - a >= CONTACT_LAG_DEG:
                contact = round(a, 2)
        if contact is not None:                                         # (2) squeeze
            goal = min(GRIPPER_CLOSE_DEG, contact + SQUEEZE_DEG)
            while cmd < goal:
                cmd = min(goal, cmd + CLOSE_STEP_DEG)
                self.grip["gripper_set_angle_deg"](cmd); await self.frames(CLOSE_STEP_FRAMES); frames += CLOSE_STEP_FRAMES
        await self.frames(30); frames += 30
        st = self.grip["gripper_status"]()
        self.hold_cmd = cmd
        holding = contact is not None and st["achieved_deg"] < cmd - HOLD_MARGIN_DEG
        return self.step("close", fast_to_deg=fast_to, contact_at_deg=contact, squeeze_cmd_deg=round(cmd, 2),
                         achieved_deg=round(st["achieved_deg"], 2), holding=holding, frames=frames,
                         finger_separation_mm=round(st["finger_origin_separation_mm"], 1))

    def pad_gap_mm(self, yaw_deg):
        """Distance between the inner faces of the two fingertip pads along the closing direction (yaw 0 = world +X),
        mm, from the pad mesh vertices. Robot geometry only. Compared with the object's measured width it shows how
        deep the pads went in (gap < width)."""
        import omni.usd
        from pxr import Usd, UsdGeom
        st = omni.usd.get_context().get_stage(); u = np.array([math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg)), 0.0])
        proj = []
        for side in ("left", "right"):
            root = st.GetPrimAtPath(f"/World/ArmWithHandOnly/Robotiq_2F_85/Robotiq_2F_85/{side}_inner_finger")
            m = next(q for q in Usd.PrimRange(root, Usd.TraverseInstanceProxies()) if "fingertipsstep" in q.GetName() and q.GetTypeName() == "Mesh")
            M = UsdGeom.Xformable(m).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            proj.append(np.array([np.dot(np.array(M.Transform(v)), u) for v in UsdGeom.Mesh(m).GetPointsAttr().Get()]))
        lo, hi = sorted(proj, key=lambda p: p.mean())
        return round(float(hi.min() - lo.max()) * 1000, 1)

    def truth(self):
        """Report only: world positions of the test blocks (fabric) -- never used by the method."""
        import omni.usd
        from isaacsim.core.experimental.prims import XformPrim
        import isaacsim.core.experimental.utils.backend as bu
        from pxr import UsdGeom
        st = omni.usd.get_context().get_stage(); out = {}
        tb = st.GetPrimAtPath("/World/TestBlocks")
        paths = [str(c.GetPath()) for c in tb.GetChildren() if c.IsA(UsdGeom.Gprim)] if tb.IsValid() else []
        paths.append("/World/object")
        from isaacsim.core.experimental.prims import RigidPrim
        for p in paths:      # RigidPrim = live physics state (XformPrim/fabric reads were stale for these bodies)
            pos, _ = self.oc["_cached_prim"](RigidPrim, p).get_world_poses(); out[p.split("/")[-1]] = [round(float(v), 4) for v in pos.numpy()[0]]
        return out

    async def front_snapshot(self, run, name):
        """Overview picture from the fixed /World/FrontCamera (does not move the arm) -- for review."""
        await self.ns["snapshot"]("/World/FrontCamera", os.path.join(run, name))
        self.images.append(name)

    def basket_box(self):
        import omni.usd
        from pxr import UsdGeom, Usd
        st = omni.usd.get_context().get_stage()
        r = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(st.GetPrimAtPath(BASKET)).ComputeAlignedRange()
        return list(r.GetMin()), list(r.GetMax())

    def arm_base_xy(self):
        import omni.usd
        from pxr import UsdGeom, Usd
        st = omni.usd.get_context().get_stage()
        t = UsdGeom.Xformable(st.GetPrimAtPath(ARM_BASE)).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
        return [float(t[0]), float(t[1])]

    async def basket_depth(self, run, name="basket_view"):
        """Depth picture from the current (above-basket) pose -> world points."""
        import omni.usd
        vdir = os.path.join(run, f"{name}_{len(self.log)}"); os.makedirs(vdir, exist_ok=True)
        self.images.append(os.path.relpath(os.path.join(vdir, "capture", "rgb_left.png"), run))
        await self.ns["capture"](os.path.join(vdir, "capture"))
        stage = omni.usd.get_context().get_stage(); left = self.ns["CAMERA_PATHS"]["left"]
        fr = self.ns["camera_frame"](stage, left); K = self.ns["camera_intrinsics"](stage, left)
        cam = {k: [float(x) for x in fr[k]] for k in ("position_m", "right", "up", "forward")}
        depth = np.load(os.path.join(vdir, "capture", "depth_axial_left.npy"))
        return depth, self.geo.world_points(depth, cam, K, 0.5)

    # ---------------------------------------------------------------- one pick cycle
    def truth_class(self, name):
        return "cube" if name == "object" else name.split("_")[0]

    async def perceive(self):
        """detect(): one_click perception (view A, B if needed, arm back home). Returns the one_click doc."""
        return await self.oc["_run"]()

    async def cycle(self, run, target, idx, failed, doc=None, only_id=None):
        """One (perceive) -> pick -> basket -> home cycle. `failed` = {xy key: attempts} of earlier failures.
        doc: an earlier detect() result to use instead of perceiving again; only_id: pick this detected id only."""
        cfg = self.oc["CFG"]; table_z = cfg["table_z_m"]; n = idx + 1
        if doc is None:
            self.step(f"pick{n}:perceive")
            doc = await self.perceive()
        else:
            self.step(f"pick{n}:use_detection", perception_run=doc["run_id"])
        A = {d["id"]: d for d in doc["views"][0]["detections"]}
        objs = [dict(A[o["id"]], class_name=o["class_name"], conf=o["conf"], center_xyz_m=o["center_xyz_m"]) for o in doc["objects"]]
        base_xy = self.arm_base_xy()
        cam = doc["views"][0]["camera"]["position_m"] if self.order_by == "camera" else None
        ordered = self.planner.order_targets(objs, target, base_xy, cam)
        if only_id is not None:
            ordered = [o for o in ordered if o["id"] == only_id]
        key = lambda o: (round(o["center_xyz_m"][0] / 0.02), round(o["center_xyz_m"][1] / 0.02))
        tgt = plan = None; skipped = []
        for o in ordered:
            if failed.get(key(o), 0) >= 2:
                skipped.append(dict(id=o["id"], class_name=o["class_name"], reason="failed twice before")); continue
            p = self.planner.plan_grasp(o, [x for x in objs if x is not o and x.get("center_xyz_m")], table_z)
            if p["ok"]:
                tgt, plan = o, p; break
            skipped.append(dict(id=o["id"], class_name=o["class_name"], reason=p["reason"], tried=p["tried"]))
        info = dict(pick=n, perception_run=doc["run_id"], seen_targets=len(ordered),
                    order_by=self.order_by, order=[(o["id"], o["order_key_m"]) for o in ordered], skipped=skipped,
                    perceived=[dict(id=o["id"], class_name=o["class_name"], conf=o["conf"], center=o["center_xyz_m"]) for o in objs])
        self.step(f"pick{n}:plan", chosen=tgt and tgt["id"], plan=plan, **{k: info[k] for k in ("order", "skipped")})
        if tgt is None:
            return dict(info, done=True, reason="no graspable target left" if ordered else f"no {target} seen")
        truth0 = self.truth()
        gt_name = min(truth0, key=lambda k: math.dist(truth0[k][:2], tgt["center_xyz_m"][:2]))
        home = self.oc["_pose"]("/World/IKTarget")
        quat = list(self.pm.quat_down_with_yaw(plan["yaw_deg"]))
        res = dict(info, done=False, target_id=tgt["id"], target_class=tgt["class_name"], target_conf=tgt["conf"],
                   target_center=tgt["center_xyz_m"], dist_from_arm_base_m=tgt["dist_from_arm_base_m"],
                   dist_from_camera_m=tgt.get("dist_from_camera_m"), plan=plan,
                   fail_key=key(tgt), images={})
        # grasp: open only as wide as this object needs (plan["open_deg"])
        await self.gripper(0.0, 30)
        await self.gripper(plan.get("open_deg", 0.0), 60)
        await self.move(plan["tcp_pre"], quat, f"pick{n}:above_target")
        g = list(plan["tcp_grasp"])
        r = await self.move(g, quat, f"pick{n}:down_to_center", max_frames=600)
        if not r["arrived"] and r["err_mm"][2] > 2.0:          # stopped above the target: fingertips hit the table/object
            g[2] += r["err_mm"][2] / 1000 + 0.002
            await self.move(g, quat, f"pick{n}:raised_for_table", max_frames=300)
            res["raised_for_table_mm"] = round(r["err_mm"][2] + 2, 1)
        width = next((t["width_mm"] for t in plan["tried"] if abs(t["yaw_deg"] - round(plan["yaw_deg"], 1)) < 0.2), None)
        res["grip"] = await self.close(width)
        res["grip"]["pad_gap_mm"] = self.pad_gap_mm(plan["yaw_deg"])
        res["grip"]["object_width_mm"] = width
        lost_at = None
        if not res["grip"]["holding"]:
            lost_at = "close (nothing between the fingers)"
        else:
            for label, xyz, slow in ((f"pick{n}:lift", plan["tcp_pre"], True),
                                     (f"pick{n}:up_to_transit", [plan["tcp_pre"][0], plan["tcp_pre"][1], TRANSIT_Z], False)):
                s = await self.move(xyz, quat, label, carrying=True, slow=slow)
                if s["finger_deg"] >= self.hold_cmd - HOLD_MARGIN_DEG:
                    lost_at = label.split(":")[1]; break
        if lost_at is None:
            bmin, bmax = self.basket_box(); bc = [(bmin[0] + bmax[0]) / 2, (bmin[1] + bmax[1]) / 2]
            s = await self.move([bc[0], bc[1], TRANSIT_Z], quat, f"pick{n}:above_basket", carrying=True, slow=False)
            if s["finger_deg"] >= self.hold_cmd - HOLD_MARGIN_DEG:
                lost_at = "transit to basket"
            depth, P = await self.basket_depth(run, f"pick{n}_holding_over_basket")
            res["images"]["holding_over_basket"] = self.images[-1]
            cell = self.planner.choose_drop_cell(P, depth, bmin, bmax, base_xy)
            self.step(f"pick{n}:basket_depth", chosen=cell and cell["chosen"])
            drop_xy = cell["chosen"]["center_xy"] if cell and cell["chosen"] else bc
            res["drop_xy"] = drop_xy
            await self.move([drop_xy[0], drop_xy[1], TRANSIT_Z], quat, f"pick{n}:above_cell", carrying=True, slow=False)
            await self.move([drop_xy[0], drop_xy[1], bmax[2] + DROP_ABOVE_RIM_M], quat, f"pick{n}:lower_to_drop", carrying=True, slow=False)
            await self.gripper(0.0, 90); self.step(f"pick{n}:release"); await self.frames(60)
            await self.move([drop_xy[0], drop_xy[1], TRANSIT_Z], quat, f"pick{n}:up_after_drop")
        else:
            await self.gripper(0.0, 60)
            await self.move([g[0], g[1], TRANSIT_Z], quat, f"pick{n}:retreat_after_failure")
        await self.move([home[0][0], home[0][1], TRANSIT_Z], home[1], f"pick{n}:toward_home")
        await self.move(home[0], home[1], f"pick{n}:home")
        await self.frames(60)
        # evaluation (stage truth, report only)
        truth1 = self.truth(); bmin, bmax = self.basket_box()
        p = truth1.get(gt_name)
        in_basket = bool(p and bmin[0] < p[0] < bmax[0] and bmin[1] < p[1] < bmax[1] and p[2] < bmax[2] + 0.02)
        true_cls = self.truth_class(gt_name)
        if in_basket and true_cls == target:
            outcome = "ok"
        elif in_basket:
            outcome = "wrong_class"
        elif lost_at:
            outcome = "missed_grasp" if lost_at.startswith("close") else "dropped"
        else:
            outcome = "not_in_basket"
        res.update(lost_at=lost_at, outcome=outcome,
                   truth_eval_only=dict(name=gt_name, true_class=true_cls, before=truth0.get(gt_name), after=p, in_basket=in_basket))
        return res

    def diagnose(self, target, cycles, truth_start, truth_end, bmin, bmax, only=None):
        """Self-check against the stage truth (report only): what went wrong, why, and what was decided.
        only: judge just these objects (pick() = one object), not every target on the table."""
        issues = []
        inb = lambda p: bool(p and bmin[0] < p[0] < bmax[0] and bmin[1] < p[1] < bmax[1] and p[2] < bmax[2] + 0.02)
        should = sorted(k for k in truth_start if self.truth_class(k) == target and not inb(truth_start[k]))
        if only is not None:
            should = [k for k in should if k in only]
        done = [k for k in should if inb(truth_end.get(k))]
        first = next((c for c in cycles if c.get("perceived")), None)
        for c in cycles:
            if c.get("done"):
                continue
            o, n = c["outcome"], c["pick"]
            if o == "wrong_class":
                issues.append(dict(pick=n, what=f"picked {c['truth_eval_only']['name']} (a {c['truth_eval_only']['true_class']}) although the order was '{target}'",
                                   why=f"perception labelled it '{c['target_class']}' (conf {c['target_conf']:.2f})",
                                   decision="perception class error: for cube/cuboid check the depth size check; otherwise raise the confidence needed to pick (second view) and review the YOLO error"))
            elif o == "missed_grasp":
                issues.append(dict(pick=n, what=f"gripper closed on nothing at {c['truth_eval_only']['name']}",
                                   why="finger closed fully -> object not between the pads (position/yaw off, or the object moved)",
                                   decision="retried once in a later pick after re-perceiving; skipped after 2 failures"))
            elif o == "dropped":
                issues.append(dict(pick=n, what=f"{c['truth_eval_only']['name']} slipped during '{c['lost_at']}'",
                                   why="finger opened to full close while carrying -> grip lost (too fast / too little friction / off-center grasp)",
                                   decision="object is re-perceived and retried; if it repeats: slower carry or grasp lower on the object"))
            elif o == "not_in_basket":
                issues.append(dict(pick=n, what=f"{c['truth_eval_only']['name']} was carried but is not inside the basket (now {c['truth_eval_only']['after']})",
                                   why="released but it bounced/rolled out, or the drop cell is too close to a wall",
                                   decision="use a lower drop height / cells further from the walls"))
        for k in should:
            if k in done:
                continue
            p = truth_start[k]
            seen = [o for o in (first or {}).get("perceived", []) if o["center"] and math.dist(o["center"][:2], p[:2]) < 0.03]
            if not seen:
                why = "never detected by perception"
                dec = "perception miss: check the depth-only 'unknown' objects / second view"
            elif seen[0]["class_name"] != target:
                why = f"detected but classified as '{seen[0]['class_name']}' (conf {seen[0]['conf']:.2f})"
                dec = "classification error -> not picked; review the class check for this shape"
            else:
                sk = [s for c in cycles for s in c.get("skipped", []) if s["id"] == seen[0]["id"]]
                why = f"detected as {target} but skipped: {sk[-1]['reason']}" if sk else "detected as target but not picked (attempts failed)"
                dec = "grasp planning limit (every yaw candidate blocked / failed twice) -> reported, left for a later improvement"
            issues.append(dict(pick=None, what=f"{k} ({target}) is still on the table", why=why, decision=dec))
        return dict(expected_truth=len(should), expected_perceived=first and first["seen_targets"],
                    in_basket=len(done), issues=issues, remaining=[k for k in should if k not in done])

    async def run(self, target, max_picks, order_by="arm_base", open_report=False, doc=None, only_id=None):
        """doc given -> pick() mode: one pick from that detection, no re-perception, report on that object only."""
        self.order_by = order_by
        await ensure_sim()
        import omni.kit.app, omni.usd
        from pxr import Usd
        self.t0 = time.time()
        run = os.path.join(OUT, datetime.datetime.now().strftime("%Y%m%d_%H%M%S")); os.makedirs(run, exist_ok=True)
        _status(state="running", run_dir=run, target=target, error=None)
        st = omni.usd.get_context().get_stage()
        if st.GetPrimAtPath("/World/TestBlocks").IsValid():   # see place_test_blocks._session_edit_target
            st.SetEditTarget(Usd.EditTarget(st.GetSessionLayer()))
        self.ns = self.oc["_load_arm_stack"]()
        self.ns["ik_follow_start"]()
        for _ in range(600):
            await omni.kit.app.get_app().next_update_async()
            if self.ns["ik_follow_status"]().get("running"):
                break
        start = self.oc["_pose"]("/World/IKTarget")
        truth_start = self.truth()
        await self.front_snapshot(run, "front_start.png")
        cycles, failed = [], {}
        for i in range(max_picks):
            c = await self.cycle(run, target, i, failed, doc=doc if i == 0 else None, only_id=only_id); cycles.append(c)
            if c.get("done") or doc is not None:
                break
            if c["outcome"] != "ok" and not c["truth_eval_only"]["in_basket"]:
                failed[c["fail_key"]] = failed.get(c["fail_key"], 0) + 1
            await self.front_snapshot(run, f"front_after_pick{i + 1}.png")
            c["images"]["front_after"] = self.images[-1]
            json.dump(dict(target=target, cycles=cycles, log=self.log), open(os.path.join(run, "result.json"), "w"), indent=1, default=str)
        bmin, bmax = self.basket_box()
        only = [c["truth_eval_only"]["name"] for c in cycles if not c.get("done")] if doc is not None else None
        diag = self.diagnose(target, cycles, truth_start, self.truth(), bmin, bmax, only=only)
        doc = dict(target=target, run_dir=run, start_pose=start, cycles=cycles, diagnosis=diag, log=self.log, images=self.images,
                   total_s=round(time.time() - self.t0, 1))
        json.dump(doc, open(os.path.join(run, "result.json"), "w"), indent=1, default=str)
        _html(run, doc)
        _status(state="done", html=os.path.join(run, "index.html"), in_basket=diag["in_basket"],
                expected=diag["expected_truth"], issues=len(diag["issues"]))
        if open_report:
            try:
                import webbrowser; webbrowser.open("file://" + os.path.join(run, "index.html"))
            except Exception:
                pass
        return doc


def _html(run, doc):
    d = doc["diagnosis"]; t = doc["target"]
    picks = [c for c in doc["cycles"] if not c.get("done")]
    ok = d["in_basket"] == d["expected_truth"] and not d["issues"]
    head = (f"<div class='{'good' if ok else 'bad'}'><b>{'All done' if ok else 'ERROR REPORT'}</b> — "
            f"{d['in_basket']} / {d['expected_truth']} {t}(s) in the basket "
            f"(perception saw {d['expected_perceived']} {t}(s) at the start). {len(picks)} pick attempt(s), {doc['total_s']:.0f} s.</div>")
    issues = "".join(f"<tr><td>{i['pick'] or '-'}</td><td>{i['what']}</td><td>{i['why']}</td><td>{i['decision']}</td></tr>" for i in d["issues"])
    issues = (f"<h2>Error report</h2><table><tr><th>pick</th><th>what went wrong</th><th>why (my judgement)</th><th>decision / fix</th></tr>{issues}</table>"
              if issues else "<p>No errors: every requested object ended up in the basket and nothing else was picked.</p>")
    cards = []
    for c in picks:
        tr = c["truth_eval_only"]; im = c.get("images", {})
        badge = {"ok": "OK", "wrong_class": "WRONG CLASS", "missed_grasp": "MISSED GRASP", "dropped": "DROPPED", "not_in_basket": "NOT IN BASKET"}[c["outcome"]]
        cards.append(f"""<div class=card><h3>Pick {c['pick']}: #{c['target_id']} {c['target_class']} ({c['target_conf']:.2f}) — <span class='{'ok' if c['outcome'] == 'ok' else 'ng'}'>{badge}</span></h3>
<p>distance from {'camera' if c.get('dist_from_camera_m') is not None else 'arm base'} {c.get('dist_from_camera_m') or c['dist_from_arm_base_m']:.3f} m | gripper yaw {c['plan']['yaw_deg']:.1f}° | finger stopped at {c['grip']['achieved_deg']}°
| stage object: {tr['name']} ({tr['true_class']}) {tr['before']} → {tr['after']}
| <a href='../../oneclick/{c['perception_run']}/index.html'>perception report</a></p>
<div class=g>{f"<figure><img src='{im['holding_over_basket']}'><figcaption>holding it above the basket (wrist camera)</figcaption></figure>" if im.get('holding_over_basket') else ''}
{f"<figure><img src='{im['front_after']}'><figcaption>after pick {c['pick']} (front camera)</figcaption></figure>" if im.get('front_after') else ''}</div></div>""")
    end = next((c for c in doc["cycles"] if c.get("done")), None)
    html = f"""<!doctype html><meta charset=utf-8><title>Pick report: {t}</title>
<style>body{{font:14px system-ui;max-width:1200px;margin:24px auto;padding:0 16px;color:#222}}table{{border-collapse:collapse;width:100%}}
td,th{{border-bottom:1px solid #ddd;padding:6px;text-align:left;vertical-align:top}}img{{max-width:100%;border:1px solid #ccc}}
.g{{display:grid;grid-template-columns:1fr 1fr;gap:10px}}.card{{border:1px solid #ddd;border-radius:6px;padding:10px 14px;margin:12px 0}}
.good{{background:#e6f4ea;padding:10px;border-radius:6px}}.bad{{background:#fde2e2;padding:10px;border-radius:6px}}.ok{{color:#1a7f37}}.ng{{color:#b3261e}}</style>
<h1>Pick report: all "{t}" → basket</h1>{head}{issues}
<div class=g><figure><img src='front_start.png'><figcaption>start (front camera)</figcaption></figure>
<figure><img src='{picks[-1]['images'].get('front_after', 'front_start.png') if picks else 'front_start.png'}'><figcaption>end (front camera)</figcaption></figure></div>
{''.join(cards)}
<p>Stopped because: {end['reason'] if end else 'pick limit reached'}.</p>
<p class=s>Stage truth is used only in this report to check the result; the robot decisions use perception only.</p>"""
    open(os.path.join(run, "index.html"), "w", encoding="utf-8").write(html)


async def _wrapped(target, max_picks, order_by="arm_base", open_report=False):
    try:
        check_target(target)
        return await Picker().run(target, max_picks, order_by, open_report)
    except Exception as exc:
        _status(state="error", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())


def pick(target="cube", max_picks=12, order_by="arm_base", open_report=False):
    reg = _reg()
    if reg["task"] is not None and not reg["task"].done():
        return {"state": "already_running"}
    reg["task"] = asyncio.ensure_future(_wrapped(target, max_picks, order_by, open_report))
    return {"state": "scheduled", "status_file": os.path.join(OUT, "latest_status.json")}


def pick_status():
    return dict(_reg()["status"])


TARGET = "cube"
MAX_PICKS = 12
if not globals().get("NO_AUTORUN"):
    print(pick(TARGET, MAX_PICKS))
