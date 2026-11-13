"""lab/fast_pick.py -- continuous-motion pick-and-place (phase 1: one detect(), then no stops except grasp and place).

What is different from grasp/pick.py:
  * ONE continuous IK-target trajectory per leg (rounded corners, trapezoid speed, set every frame by sim time);
    the arm only stops at the grasp point and at the place point.
  * IK slew cap raised (MAX_JOINT_DELTA_PER_STEP) so the TCP can follow ~1 m/s.
  * Gripper: ONE close command (contact angle from the measured width + squeeze), no 1-degree steps, no fixed waits.
  * No re-detect between picks, no basket depth photo, no return home between picks.
  * Place: the object is lowered onto the basket floor (in a grid cell) and released there -- no drop from height.
Reuses grasp/pick.py Picker (planner, gripper, pose helpers, truth for the report only).

Run (MCP execute_script, after grasp.api booted and the np.cross shim):
    exec(open(".../lab/fast_pick.py").read()); start("sphere")
Result: outputs/lab/fast/<time>/result.json (+ tcp.npy).
"""
import asyncio, builtins, datetime, json, math, os, time, traceback
import numpy as np

def _project_root():
    """Project root: parent of lab/ when loaded with __file__ set, else parent of the folder holding the open scene."""
    f = globals().get("__file__")
    if f:
        return os.path.dirname(os.path.dirname(os.path.abspath(f)))
    import omni.usd
    return os.path.dirname(os.path.dirname(os.path.abspath(omni.usd.get_context().get_stage().GetRootLayer().realPath)))


ROOT = _project_root()
OUT = os.path.join(ROOT, "outputs", "lab", "fast")

P = dict(
    ik_cap=0.08,            # MAX_JOINT_DELTA_PER_STEP while this runs (rad per update, whole arm)
    v_free=1.0, a_free=4.0,         # TCP speed / accel when nothing is held
    v_carry=0.6, a_carry=1.2,       # holding an object
    v_final=0.35,                   # (unused) reserved
    transit_z=0.76,                 # TCP height for travel (basket rim 0.6856 + held object + margin)
    fillet=0.07, fillet_down=0.04,  # corner rounding radius (m)
    pre_above_top=0.06,             # TCP height above the tallest object top at the start of the descent
    floor_z=0.6035,                  # basket inner floor (estimate; calibrated from the run report)
    place_clear=0.003,              # object bottom this far above the floor when released
    squeeze_deg=5.0,                # firm squeeze: close command = contact angle(measured) + this (the pads self-centre the object)
    squeeze_extra_deg=3.0,          # extra closing per re-squeeze when the tug test shows the object is slipping
    tug_lift_m=0.012, tug_slip_deg=1.5, min_lag_deg=1.0,   # tug test: lift this far, finger must stay put and keep pushing against the object
    close_pre_deg=4.0,              # fast jump to (predicted contact - this), then a continuous ramp
    close_ramp_deg_per_frame=0.35, contact_lag_deg=1.5,
    wall_margin=0.06, grid=(4, 3),
    lag_lo=0.025, lag_hi=0.07,      # m: carrot slows between these TCP-to-target distances, stops at lag_hi
    rot_lo=6.0, rot_hi=20.0,        # deg: same for orientation lag
    width_tol_mm=(0.65, 6.0),        # a grasp is plausible when 0.65 x short side <= closed width <= planned width + 6 (the firm squeeze turns the object onto its true
                                    # short side, which can be narrower than the perceived one: lying cylinders were measured 49-55 mm, are 40 mm;
                                    # a 66 mm neighbour between the pads for a 40 mm target is still rejected). Evidence: batch2_clean1.json
    preshape_clear_m=0.04,          # pre-shape the gripper for the next object only when the TCP is within this of the travel height (the hand is then clear of the block just released)
    visual_refine=False,            # option, default OFF: False / "change" / "estimate" (lab/README-lab.md); see live_pick.py retarget()
    refine_z_m=(0.03, 0.30),        # corrections only while the TCP is this far above the grasp height (low = close to the object, hand not yet in the way)
    moved_m=0.012,                  # "change" mode: only when the depth blob is this far from the scanned position (the block moved after the scan)
    tight_margin=0.002,             # m: finger clearance in the last-resort tier (normal 0.006)
    touch_margin=-0.003,            # m: finger 'clearance' of the very last tier (negative = the fingers may touch a neighbour's edge by 3 mm)
    tight_open_margin=0.010,        # m: gripper opening = block width + this in the last-resort tiers (normal 0.020)
    max_relocations=6,              # neighbours that may be moved aside in one run when every target is blocked by one
    rot_speed=3.0,                  # rad/s: leg time is at least angle / rot_speed
    place_outline=False,            # 11/09 options for the place spot (defaults = before): True = check / remember the measured footprint, not the top-face disc
    place_rank="base",              # "base" = nearest free spot to the arm base; "clearance" = the free spot farthest from occupied cells (ties: nearest the base)
    place_yaw_offsets=(0.0,),       # closing-axis turns (deg) tried in order when no spot is free at the grasp yaw
    place_map_res_m=None,           # basket map cell size (None = 10 mm as before)
    place_wrist_guard=False,        # True = choose place_yaw or place_yaw +-180 so joint_6_t stays inside its range for grasp -> place -> home (11/09)
    place_release_open="full",      # "full" = open fully at the place (before); "grasp" = only as wide as for the grasp (width + OPEN_MARGIN): narrower fingers in the bin
    place_wall_m=None,              # bin map wall margin (None = 30 mm from the bin box as before; the box includes the 10 mm wall)
    place_gap_tiers=None,           # free-space margins tried in order (None = 15 / 8 / 3 mm as before)
    place_align_deg=None,           # None = keep the grasp yaw; else turn every part so its long side points this way (world deg) -- for place_rank "rows"
    place_release_margin_m=None,    # 11/10: None = before; else release opening = max(planned, measured after close) + this (lab/live_perception.release_gap_m),
                                    #   the spot is checked again with it after the tug test
    place_fallback=None,            # 11/10: None = the fixed fallback cells (before); "emptiest" = the emptiest map point, no_spot / box_full recorded
    width_check="abort",            # 11/10: "abort" = closed width out of tolerance aborts (before); "neighbor" = abort only if a neighbour is near the fingers
    grasp_arrive_s=None,            # 11/10: sim seconds the hand may take to reach the grasp point (None = 150 app frames as before; with 3 physics steps per frame that is 7.5 s)
    grasp_height_tol_m=None,        # 11/10: None = close wherever the hand is (before); else do not close when the TCP is more than this off the planned grasp height
    rotate_before_descent=False,    # 11/10 E: True = the wrist turn is finished at a pre-grasp / pre-place height above the spot; the last part of the descent is straight down
    approach=None,                  # 11/10 E: None = one leg down (before); dict(pre_above_m, v, a, lift_m): fast to pre_above_m above the part top, then a slow
                                    #   trapezoid (v m/s, a m/s2) straight down; after the tug test a slow lift of lift_m before the carry
    save_tcp_log=False,             # 11/10 E: True = tcp_log.json (sim t, TCP, target, finger deg, target / TCP turn to go deg) in every run_live folder
    choose_rank="near",             # 11/10 D: "near" = nearest graspable target (before); "clear" = the one with the most room around it first
    chain=False,                    # 11/10 D: True = after a good place fly straight to the next part planned during the carry; back to view A only after a
                                    #   failure, a slip in the tug test, or when the next part lies within this grasp's finger range + chain_touch_m
    chain_touch_m=0.006,
    save_map_snapshot=False,        # 11/10 G0: True = k<n>_bin_map.npz at every place decision (record only)
    fold_yaw=False,
    corner_limit=False,             # 11/10 G3 U2b: True = carry legs use speed_profile (normal and tangential acceleration <= a / sqrt 2 each)
    carry_lag_strict=False,         # 11/10 G3 U2c: True = no extra lag allowance (el, er) while carrying with several physics steps per frame
    j1_slow_ratio=None,             # 11/10 G3 U2c: e.g. 0.85 = while carrying, slow the carrot once joint 1 runs above this share of its speed limit
    ik_cap_scale=False,             # 11/10 G3: True = MAX_JOINT_DELTA_PER_STEP <= 0.95 x slowest joint limit / 60 (the follower's speed then stays under it)
    drop_detect_deg=None,           # 11/10 G2 U3: e.g. 2.5 = in the carry, the finger closing this much more within drop_window_s means the part fell
    drop_window_s=0.2,
    floor_filter_m=None,            # 11/10 G4 U1b: e.g. 0.003 = drop points lower than floor - this before the view-A footprint
    floor_filter_classes=(),        # 11/10 G4: classes the floor filter applies to (empty = every class)                 # 11/10 F1 (wrist plan A): True = grasp at the 180-deg twin yaw nearest the wrist's present angle; a place turn > 90 deg flips too
    outline_diag=False,             # 11/10 stage 4a: True = record footprint variants + the chosen target's mask / points (npz) at view A; never used by a decision
    step_follow=False,              # 11/10 P5 (Lin chose P4 A): True = during fly() only, the IK follower and the carrot update every physics step (R F7b:
                                    #   outputs/tests/20261110_R/r1_fly_f7.py); TCP / hand read from the PhysX link transform; restored after every leg
)


# ------------------------------------------------------------------ path helpers
def rounded_path(pts, radii, n=14):
    """Polyline through pts; interior corners i replaced by a quadratic Bezier of radius radii[i]. Returns dense (N,3)."""
    pts = [np.asarray(p, float) for p in pts]
    out = [pts[0]]
    for i in range(1, len(pts) - 1):
        a, b, c = pts[i - 1], pts[i], pts[i + 1]
        l1, l2 = np.linalg.norm(b - a), np.linalg.norm(c - b)
        if l1 < 1e-6 or l2 < 1e-6:
            continue
        d = min(radii[i], 0.5 * l1, 0.5 * l2)
        p0 = b - (b - a) / l1 * d; p2 = b + (c - b) / l2 * d
        for k in range(n + 1):
            t = k / n
            out.append((1 - t) ** 2 * p0 + 2 * t * (1 - t) * b + t ** 2 * p2)
    out.append(pts[-1])
    return np.array(out)


def trapezoid(L, vmax, amax):
    ta = vmax / amax
    if amax * ta * ta > L:
        ta = math.sqrt(L / amax); vmax = amax * ta; tc = 0.0
    else:
        tc = (L - amax * ta * ta) / vmax
    T = 2 * ta + tc

    def s(t):
        t = min(max(t, 0.0), T)
        if t < ta: return 0.5 * amax * t * t
        if t < ta + tc: return 0.5 * amax * ta * ta + vmax * (t - ta)
        td = T - t
        return L - 0.5 * amax * td * td
    return T, s


def speed_profile(path, vmax, amax):
    """11/10 G3 U2b (pure): a speed profile along the dense path (N,3) that keeps BOTH the tangential and the normal (v^2 / R) acceleration
    <= amax / sqrt(2) (so the total <= amax): node speed <= sqrt(a_n R) at curves (R = circumradius of 3 neighbouring points), forward /
    backward passes for a_t. Returns (T, s(t)) like trapezoid()."""
    path = np.asarray(path, float); seg = np.linalg.norm(np.diff(path, axis=0), axis=1); keep = np.r_[True, seg > 1e-9]
    path = path[keep]; seg = np.linalg.norm(np.diff(path, axis=0), axis=1); n = len(path)
    if n < 2 or seg.sum() < 1e-6:
        return trapezoid(max(float(seg.sum()), 1e-4), vmax, amax)
    a_c = amax / math.sqrt(2.0); vl = np.full(n, float(vmax)); vl[0] = vl[-1] = 0.0
    for i in range(1, n - 1):
        a_, b_, c_ = path[i - 1], path[i], path[i + 1]
        ab, bc, ca = np.linalg.norm(b_ - a_), np.linalg.norm(c_ - b_), np.linalg.norm(a_ - c_)
        area2 = np.linalg.norm(np.cross(b_ - a_, c_ - a_))
        if area2 > 1e-12:
            vl[i] = min(vl[i], math.sqrt(a_c * (ab * bc * ca / (2.0 * area2))))
    v = vl.copy()
    for i in range(1, n):
        v[i] = min(v[i], math.sqrt(v[i - 1] ** 2 + 2 * a_c * seg[i - 1]))
    for i in range(n - 2, -1, -1):
        v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2 * a_c * seg[i]))
    cum = np.r_[0.0, np.cumsum(seg)]; t = np.r_[0.0, np.cumsum(2 * seg / np.maximum(v[:-1] + v[1:], 1e-6))]
    acc = (v[1:] ** 2 - v[:-1] ** 2) / (2 * np.maximum(seg, 1e-12)); T = float(t[-1])

    def s(tt, t=t, cum=cum, v=v, acc=acc):                             # constant acceleration inside each segment (no speed jumps at the nodes)
        tt = min(max(tt, 0.0), T); i = min(int(np.searchsorted(t, tt, side="right")) - 1, len(acc) - 1); tau = tt - t[i]
        return float(min(cum[i + 1], cum[i] + v[i] * tau + 0.5 * acc[i] * tau * tau))
    return T, s


def rounded_path_curvature(pts, radii, n=14):
    """11/10 Q3 (pure): the same dense path as rounded_path(pts, radii, n) and, for every node, the curvature (1/m) of the rounded corner it lies on:
    the EXACT curvature of the quadratic Bezier B(t) = (1-t)^2 p0 + 2t(1-t) b + t^2 p2 at t = k/n, |B' x B''| / |B'|^3 (0 on straight lines and
    at the path ends). The corner is symmetric (both control legs d long), so its largest curvature is at t = 0.5 (a node for even n)."""
    pts = [np.asarray(p, float) for p in pts]
    out = [pts[0]]; kap = [0.0]
    for i in range(1, len(pts) - 1):
        a, b, c = pts[i - 1], pts[i], pts[i + 1]
        l1, l2 = np.linalg.norm(b - a), np.linalg.norm(c - b)
        if l1 < 1e-6 or l2 < 1e-6:
            continue
        d = min(radii[i], 0.5 * l1, 0.5 * l2)
        p0 = b - (b - a) / l1 * d; p2 = b + (c - b) / l2 * d
        dd = 2.0 * (p2 - 2.0 * b + p0)
        for k in range(n + 1):
            t = k / n
            out.append((1 - t) ** 2 * p0 + 2 * t * (1 - t) * b + t ** 2 * p2)
            d1 = 2.0 * (1 - t) * (b - p0) + 2.0 * t * (p2 - b); nd1 = float(np.linalg.norm(d1))
            kap.append(float(np.linalg.norm(np.cross(d1, dd))) / nd1 ** 3 if nd1 > 1e-12 else 0.0)
    out.append(pts[-1]); kap.append(0.0)
    return np.array(out), np.array(kap)


def corner_profile(path, kappa, vmax, amax):
    """11/10 Q3 free_corner_limit (pure): speed along the dense path (N,3) such that on every segment the TOTAL acceleration of the target,
    sqrt(a_t^2 + (v^2 kappa)^2), stays <= amax (no a / sqrt 2 split): kappa = the per-node curvature of rounded_path_curvature, a segment uses the
    larger of its two nodes (inside a symmetric Bezier the curvature is monotone between nodes) and its larger end speed; node speed
    <= sqrt(amax / kappa); a forward pass solves the accelerating segments, a backward pass the braking ones (the larger root of
    (x k)^2 + ((x - v0^2) / 2 ds)^2 = amax^2, x = v^2). Constant acceleration inside each segment, as speed_profile. Returns (T, s(t))."""
    path = np.asarray(path, float); kappa = np.asarray(kappa, float)
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1); keep = np.r_[True, seg > 1e-9]
    path = path[keep]; kappa = kappa[keep]; seg = np.linalg.norm(np.diff(path, axis=0), axis=1); n = len(path)
    if n < 2 or seg.sum() < 1e-6:
        return trapezoid(max(float(seg.sum()), 1e-4), vmax, amax)
    ks = np.maximum(kappa[:-1], kappa[1:])                              # per segment
    kn = np.maximum(np.r_[ks, 0.0], np.r_[0.0, ks])                     # per node: the larger of its two segments
    vl = np.minimum(float(vmax), np.sqrt(amax / np.maximum(kn, 1e-12))); vl[0] = vl[-1] = 0.0

    def x_max(x0, k, ds):                                               # largest x = v_end^2 with (x k)^2 + ((x - x0) / 2ds)^2 <= amax^2
        A = k * k + 1.0 / (4 * ds * ds); B = -2.0 * x0 / (4 * ds * ds); C = x0 * x0 / (4 * ds * ds) - amax * amax
        disc = B * B - 4 * A * C
        return max(x0, (-B + math.sqrt(max(disc, 0.0))) / (2 * A))
    x = vl ** 2
    for i in range(n - 1):
        x[i + 1] = min(x[i + 1], x_max(x[i], ks[i], seg[i]))
    for i in range(n - 2, -1, -1):
        x[i] = min(x[i], x_max(x[i + 1], ks[i], seg[i]))
    v = np.sqrt(np.maximum(x, 0.0))
    cum = np.r_[0.0, np.cumsum(seg)]; t = np.r_[0.0, np.cumsum(2 * seg / np.maximum(v[:-1] + v[1:], 1e-6))]
    acc = (v[1:] ** 2 - v[:-1] ** 2) / (2 * np.maximum(seg, 1e-12)); T = float(t[-1])

    def s(tt, t=t, cum=cum, v=v, acc=acc):
        tt = min(max(tt, 0.0), T); i = min(int(np.searchsorted(t, tt, side="right")) - 1, len(acc) - 1); tau = tt - t[i]
        return float(min(cum[i + 1], cum[i] + v[i] * tau + 0.5 * acc[i] * tau * tau))
    return T, s


def scaled_ik_cap(ik_cap, vlims, margin=0.95):
    """11/10 G3 (pure): the follower moves at most MAX_JOINT_DELTA_PER_STEP * k per update and an update is k / 60 s of sim, so its joint speed is
    at most 60 * cap rad/s whatever k is. Cap it so that this is <= the slowest arm joint's limit (x margin)."""
    if vlims is None or not len(vlims):
        return ik_cap
    return min(float(ik_cap), margin * float(np.min(vlims)) / 60.0)


def _joint_vel_limits():
    """rad/s of the 6 arm joints from USD physxJoint:maxJointVelocity (deg/s), in articulation dof order (as lab/sort_parts.joint_vel_limits)."""
    import omni.usd
    from pxr import Usd
    art = builtins._arm310d_ik_follow_controller_registry["instance"]._art; names = list(art.dof_names)[:6]
    st = omni.usd.get_context().get_stage(); lim = {}
    for p_ in Usd.PrimRange(st.GetPrimAtPath("/World/ArmWithHandOnly")):
        a_ = p_.GetAttribute("physxJoint:maxJointVelocity")
        if a_ and a_.HasAuthoredValue() and p_.GetName() in names:
            lim[p_.GetName()] = float(a_.Get()) * math.pi / 180.0
    return np.array([lim[n] for n in names]) if len(lim) == 6 else None


class _StepPose:
    """11/10 P5 (ported from outputs/tests/20261110_R/r1_fly_f7.py TensorPose, F7b): the follower's TCP / hand poses from the PhysX tensor link
    transform of the hand link (fresh every physics step; fabric is synced once per app frame) x the fixed hand->tcp offset read once from fabric."""
    def __init__(self, inst):
        self.inst = inst; self.pv = inst._art._physics_articulation_view; self.hi = inst._hand_jac_row + 1
        self.fab = inst._on_update.__globals__["_get_fabric_world_poses"]
        hp, hq = self.fabric(inst._hand); tp, tq = self.fabric(inst._tcp)
        self.off_p = _qrot(_qconj(hq), tp - hp); self.off_q = _qmul(_qconj(hq), tq)

    def fabric(self, prim):
        p, q = self.fab(prim); return p.numpy()[0].astype(float), q.numpy()[0].astype(float)

    def hand(self):
        t = self.pv.get_link_transforms().numpy().reshape(-1, 7)[self.hi].astype(float)
        return t[:3], np.array([t[6], t[3], t[4], t[5]])          # xyzw -> wxyz

    def tcp(self):
        hp, hq = self.hand(); return hp + _qrot(hq, self.off_p), _qmul(hq, self.off_q)

    def reader(self):
        class _Arr:                                               # what _on_update expects: .numpy()[0]
            def __init__(s_, a): s_.a = np.asarray(a, float)[None, :]
            def numpy(s_): return s_.a
        def get(prim):
            if prim is self.inst._tcp:
                p, q = self.tcp(); return _Arr(p), _Arr(q)
            if prim is self.inst._hand:
                p, q = self.hand(); return _Arr(p), _Arr(q)
            return self.fab(prim)
        return get


def _qmul(a, b):          # wxyz
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2, w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2])


def _qconj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _qrot(q, v):
    return _qmul(_qmul(q, np.r_[0.0, v]), _qconj(q))[1:]


def load_picker_class():
    p = os.path.join(ROOT, "grasp", "pick.py"); g = dict(__file__=p, __name__="lab_pick_ns", NO_AUTORUN=True)      # pick.py would otherwise start pick("cube") on load
    exec(compile(open(p, encoding="utf-8").read(), p, "exec"), g)
    return g["Picker"], g


Picker, _PK = load_picker_class()


class ReleaseDebug:
    """Per-frame record around a release (diagnosis only, never used for control): block position in the TCP frame (mm),
    block height / vertical velocity, block-to-TCP distance, finger joint, and which gripper collider touches the block
    (PhysX contact report on the block). Stage truth is read here for the report only."""
    def __init__(self, picker, gt_name):
        self.p = picker; self.name = gt_name; self.rows = []; self.cur = set(); self.sub = None; self.err = None
        import omni.usd
        from pxr import PhysxSchema, PhysicsSchemaTools, Usd
        from omni.physx import get_physx_simulation_interface
        self.path = f"/World/TestBlocks/{gt_name}"
        st = omni.usd.get_context().get_stage(); prim = st.GetPrimAtPath(self.path)
        self.had_api = prim.HasAPI(PhysxSchema.PhysxContactReportAPI)
        try:
            with Usd.EditContext(st, st.GetSessionLayer()):
                api = PhysxSchema.PhysxContactReportAPI.Apply(prim); api.CreateThresholdAttr().Set(0.0)
            self._tools = PhysicsSchemaTools
            self.sub = get_physx_simulation_interface().subscribe_contact_report_events(self._on_contact)
        except Exception as e:
            self.err = repr(e)

    def _on_contact(self, headers, data):
        for h in headers:
            try:
                c = [str(self._tools.intToSdfPath(h.collider0)), str(self._tools.intToSdfPath(h.collider1))]
                a = [str(self._tools.intToSdfPath(h.actor0)), str(self._tools.intToSdfPath(h.actor1))]
                if self.path in a or self.path in c:
                    other = c[1] if (self.path in a[0] or self.path in c[0]) else c[0]
                    self.cur.add(other.replace("/World/ArmWithHandOnly/Robotiq_2F_85/", "").replace("/World/", "").split("/visuals")[0].split("/collisions")[0])
            except Exception:
                pass

    def sample(self, label):
        gp = self.p.truth().get(self.name); tp, tq = self.p.tcp()
        rel = None if gp is None else [round(1000 * float(v), 1) for v in self.p._rot(tq).T @ (np.asarray(gp) - np.asarray(tp))]
        from isaacsim.core.experimental.prims import RigidPrim
        vz = None
        try:
            v = self.p.oc["_cached_prim"](RigidPrim, self.path).get_linear_velocities().numpy()[0]; vz = round(float(v[2]), 3)
        except Exception:
            pass
        self.rows.append(dict(f=self.p.f, t=round(self.p.f / 60.0, 3), phase=label, block=None if gp is None else [round(float(x), 4) for x in gp],
                              tcp=[round(float(x), 4) for x in tp], rel_mm=rel, dist_mm=None if rel is None else round(float(np.linalg.norm(rel)), 1),
                              vz=vz, finger=round(self.p.finger_deg(), 2), cmd=round(float(self.p.grip["gripper_status"]()["target_deg"] or 0.0), 2), contacts=sorted(self.cur)))
        self.cur = set()

    def close(self):
        try:
            if self.sub is not None:
                self.sub.unsubscribe()
        except Exception:
            pass
        try:
            import omni.usd
            from pxr import PhysxSchema, Usd
            st = omni.usd.get_context().get_stage()
            if not self.had_api:
                with Usd.EditContext(st, st.GetSessionLayer()):
                    st.GetPrimAtPath(self.path).RemoveAPI(PhysxSchema.PhysxContactReportAPI)
        except Exception:
            pass


def physics_steps_per_frame():
    """Physics steps (1/60 s each) the timeline advances per app frame: 1 with the default timeline (60 time codes/s);
    2 when time codes/s and /persistent/simulation/minFrameRate are both 30 (11/08: physics costs ~1.2 ms per step,
    rendering ~25-35 ms per frame, so 2 steps per frame brings sim time close to wall time)."""
    import carb, omni.timeline
    tc = omni.timeline.get_timeline_interface().get_time_codes_per_seconds()
    mfr = carb.settings.get_settings().get("/persistent/simulation/minFrameRate") or 60
    return max(1, int(round(60.0 * min(1.0 / tc, 1.0 / mfr))))


class FastPicker(Picker):
    def __init__(self):
        super().__init__()
        self.f = 0                       # sim frames (60 per sim second)
        self.spf = physics_steps_per_frame()   # sim frames per app frame (1 = old behaviour)
        self.ev = []                     # (name, sim_s)
        self.tcp_log = []                # (sim_s, x, y, z, target x, y, z)
        self.cur_tgt = None
        self.retargets = 0
        self.watch = False                # report-only: stage truth every 3 frames (debug)
        self.watch_log = []

    def truth(self):
        """Report only: world positions of the test blocks (the original /World/object cube is skipped when it is removed)."""
        import omni.usd
        from pxr import UsdGeom
        from isaacsim.core.experimental.prims import RigidPrim
        st = omni.usd.get_context().get_stage(); out = {}
        tb = st.GetPrimAtPath("/World/TestBlocks")
        paths = [str(c.GetPath()) for c in tb.GetChildren() if c.IsA(UsdGeom.Gprim)] if tb.IsValid() else []
        ob = st.GetPrimAtPath("/World/object")
        if ob.IsValid() and ob.IsActive():
            paths.append("/World/object")
        for p in paths:
            pos, _ = self.oc["_cached_prim"](RigidPrim, p).get_world_poses(); out[p.split("/")[-1]] = [round(float(v), 4) for v in pos.numpy()[0]]
        return out

    def mark(self, name, **kw):
        self.ev.append(dict(name=name, sim_s=round(self.f / 60.0, 3), **kw))

    async def frames(self, n):
        import omni.kit.app
        app = omni.kit.app.get_app()
        for _ in range(-(-n // self.spf)):                    # n sim frames, spf of them per app frame
            await app.next_update_async(); self.f += self.spf
            if getattr(self, "dbg", None) is not None:
                self.dbg.sample(self.dbg_label)
            if self.watch and self.f % 3 == 0:
                self.watch_log.append((round(self.f / 60.0, 3), self.truth()))

    def _set_target(self, p, q):
        self.oc["_pose"]("/World/IKTarget", [float(v) for v in p], [float(v) for v in q])
        self.cur_tgt = np.asarray(p, float)

    async def fly(self, pts, q1, label, radii, carrying=False, retarget=None, on_frame=None, vmax=None, amax=None):
        """One continuous leg: IK target follows the rounded polyline pts (pts[0] = current target) with a trapezoid
        speed profile; orientation slerps to q1 along the path. Ends at rest at pts[-1] (carrot only)."""
        p0, q0 = self.oc["_pose"]("/World/IKTarget")
        path = rounded_path([p0] + [np.asarray(p, float) for p in pts], [0.0] + list(radii) + [0.0])
        seg = np.linalg.norm(np.diff(path, axis=0), axis=1); cum = np.concatenate([[0], np.cumsum(seg)]); L = float(cum[-1])
        v, a = (P["v_carry"], P["a_carry"]) if carrying else (P["v_free"], P["a_free"])
        v, a = vmax or v, amax or a
        if carrying and P.get("corner_limit"):                          # 11/10 G3 U2b: slow through the corners (normal acceleration <= a)
            T, s = speed_profile(path, v, a)
        else:
            T, s = trapezoid(max(L, 1e-4), v, a)
        if not carrying and vmax is None and amax is None and P.get("free_corner_limit"):     # 11/10 Q3: empty-hand fast legs (to_pre_grasp, to_home, to_view,
            _, kap_ = rounded_path_curvature([p0] + [np.asarray(p, float) for p in pts], [0.0] + list(radii) + [0.0])   # view_check, ...): total
            T, s = corner_profile(path, kap_, v, P.get("a_free_corner", 4.0))                 # acceleration <= a_free_corner, Bezier curvature
        ang = 2 * math.acos(min(1.0, abs(float(np.dot(q0, q1)))))
        if ang / P["rot_speed"] > T:                         # rotation is the slow part: stretch the leg
            T2 = ang / P["rot_speed"]; k = T / T2; s0 = s; T = T2; s = lambda t, s0=s0, k=k: s0(t * k)
        if P.get("step_follow"):                                   # 11/10 P5: F7b during this leg only (the loop below is not used then)
            return await self._fly_step(path, seg, cum, L, T, s, q0, q1, v, label, carrying, retarget, on_frame, p0)
        f0 = self.f; n_i = 0; tau = 0.0; scale = 1.0; held = 0
        while True:
            tau += scale * self.spf / 60.0
            sv = s(tau); i = min(int(np.searchsorted(cum, sv, side="right")) - 1, len(seg) - 1)
            u = (sv - cum[i]) / max(seg[i], 1e-9)
            p = path[i] + u * (path[i + 1] - path[i])
            qt = self.oc["_slerp"](q0, q1, min(1.0, sv / max(L, 1e-9)))
            self._set_target(p, qt)
            await self.frames(1)
            tp, tq = self.tcp()
            jr = self.joint_ratios()
            if on_frame is not None:
                on_frame()
            # the carrot waits for the arm: time advances only while the TCP is within lag_hi of the target
            lag = math.dist(tp, p); rlag = self.pm.quat_angle_deg(tq, qt)
            el = v * (self.spf - 1) / 60.0; er = math.degrees(P["rot_speed"]) * (self.spf - 1) / 60.0   # the carrot moves spf sim frames per update: allow that much more lag (0 when spf = 1)
            if carrying and P.get("carry_lag_strict"):                         # 11/10 G3 U2c: no extra lag allowance while something is held
                el = er = 0.0
            scale = max(0.0, min(1.0, (P["lag_hi"] + el - lag) / (P["lag_hi"] - P["lag_lo"]), (P["rot_hi"] + er - rlag) / (P["rot_hi"] - P["rot_lo"])))
            if carrying and P.get("j1_slow_ratio") and jr is not None and jr[0] > P["j1_slow_ratio"]:   # 11/10 G3 U2c: the base joint near its limit -> slow the carrot
                scale = min(scale, max(0.0, (1.0 - float(jr[0])) / (1.0 - P["j1_slow_ratio"])))
            if scale < 0.999: held += 1
            self.tcp_log.append((self.f / 60.0, *tp, *p, self.finger_deg(), round(self.pm.quat_angle_deg(qt, q1), 2), round(self.pm.quat_angle_deg(tq, q1), 2),   # 11/10: + turn to go
                                 round(scale, 3), held, *([round(float(x), 3) for x in jr] if jr is not None else [None] * 6)))   # 11/10 G1 U2a: scale, held, j1-j6 speed / limit
            if getattr(self, "abort_leg", None):                               # 11/10 G2: a drop detected in the carry -> stop this leg here
                break
            if retarget is not None:                          # live correction of the END point (mm-level), blended along the rest of the path
                new = retarget(tp)
                if new is not None:
                    dxy = np.asarray(new[:2], float) - path[-1][:2]
                    if np.linalg.norm(dxy) > 0.0015:
                        w = np.clip((cum - sv) / max(L - sv, 1e-6), 0.0, 1.0)
                        path[:, :2] += 0.5 * dxy[None, :] * w[:, None]; self.retargets += 1
            if tau >= T:
                break
            if n_i == 0 and tau > 0.5:                        # watchdog: target moved but the arm did not follow
                n_i = 1
                if math.dist(tp, self.tcp_log[-min(len(self.tcp_log), -(-30 // self.spf))][1:4]) < 0.003 and math.dist(p, p0) > 0.05:   # 30 sim frames back
                    raise RuntimeError(f"arm not following IK target in {label}: tcp={list(tp)}, target={list(p)}")
            if self.f - f0 > 900:
                raise RuntimeError(f"leg {label} too slow (>15 s)")
        self.leg_held = held
        self.mark(label, leg_s=round((self.f - f0) / 60.0, 2), len_m=round(L, 3), held_frames=held)

    async def _fly_step(self, path, seg, cum, L, T, s, q0, q1, v, label, carrying, retarget, on_frame, p0):
        """11/10 P5 step_follow (R F7b, outputs/tests/20261110_R/r1_fly_f7.py swap_step_carrot): for this leg only, the follower's app-update
        subscription is replaced by one call of its _on_update per physics step (its per-update limits at k = 1), the carrot advances 1/60 s x scale
        per physics step, and the follower reads TCP / hand from _StepPose. Per app frame as fly(): tcp_log, on_frame, abort_leg, retarget, watchdog.
        Restored in `finally` (also on exceptions): the three globals, one app-update subscription, physics-step subscription removed."""
        import omni.kit.app
        from omni.physx import get_physx_interface
        inst = builtins._arm310d_ik_follow_controller_registry["instance"]; gl = inst._on_update.__globals__
        saved = dict(spu=gl["_steps_per_update"], fab=gl["_get_fabric_world_poses"]); old_sub = None; sub = None
        tpose = _StepPose(inst)
        el = v * (self.spf - 1) / 60.0; er = math.degrees(P["rot_speed"]) * (self.spf - 1) / 60.0     # as fly(): the same lag allowance
        if carrying and P.get("carry_lag_strict"):
            el = er = 0.0
        st = dict(tau=0.0, scale=1.0, held=0, p=None, qt=None, sv=0.0, calls=0, err=None)
        fol0 = dict(running=inst._running, error_count=inst._error_count)                  # record only

        def carrot(tau):
            sv = s(tau); i = min(int(np.searchsorted(cum, sv, side="right")) - 1, len(seg) - 1)
            u = (sv - cum[i]) / max(seg[i], 1e-9)
            return path[i] + u * (path[i + 1] - path[i]), self.oc["_slerp"](q0, q1, min(1.0, sv / max(L, 1e-9))), sv

        def scale_of(tp, tq, p, qt):
            lag = math.dist(tp, p); rlag = self.pm.quat_angle_deg(tq, qt)
            sc = max(0.0, min(1.0, (P["lag_hi"] + el - lag) / (P["lag_hi"] - P["lag_lo"]), (P["rot_hi"] + er - rlag) / (P["rot_hi"] - P["rot_lo"])))
            if carrying and P.get("j1_slow_ratio"):
                jr = self.joint_ratios()
                if jr is not None and jr[0] > P["j1_slow_ratio"]:
                    sc = min(sc, max(0.0, (1.0 - float(jr[0])) / (1.0 - P["j1_slow_ratio"])))
            return sc

        def cb(dt):
            try:
                if st["tau"] < T and not getattr(self, "abort_leg", None):
                    tp, tq = tpose.tcp(); st["scale"] = scale_of(tp, tq, st["p"], st["qt"])
                    if st["scale"] < 0.999: st["held"] += 1
                    st["tau"] = min(T, st["tau"] + st["scale"] / 60.0); st["p"], st["qt"], st["sv"] = carrot(st["tau"]); self._set_target(st["p"], st["qt"])
                inst._on_update(None); st["calls"] += 1
            except Exception as exc:
                st["err"] = f"{type(exc).__name__}: {exc}"
        try:
            st["p"], st["qt"], st["sv"] = carrot(0.0); self._set_target(st["p"], st["qt"])
            gl["_get_fabric_world_poses"] = tpose.reader(); gl["_steps_per_update"] = lambda: 1
            old_sub = inst._sub; old_sub.unsubscribe(); inst._sub = None
            sub = get_physx_interface().subscribe_physics_step_events(cb)
            f0 = self.f; n_i = 0
            while True:
                await self.frames(1)
                tp, tq = self.tcp()
                jr = self.joint_ratios()
                if on_frame is not None:
                    on_frame()
                p, qt = st["p"], st["qt"]
                self.tcp_log.append((self.f / 60.0, *tp, *p, self.finger_deg(), round(self.pm.quat_angle_deg(qt, q1), 2), round(self.pm.quat_angle_deg(tq, q1), 2),
                                     round(st["scale"], 3), st["held"], *([round(float(x), 3) for x in jr] if jr is not None else [None] * 6)))
                if st["err"] is not None or not inst._running:
                    raise RuntimeError(f"step_follow in {label}: {st['err'] or 'follower stopped'}; follower last error: {str(inst._last_error)[-300:]}")
                if getattr(self, "abort_leg", None):
                    break
                if retarget is not None:
                    new = retarget(tp)
                    if new is not None:
                        dxy = np.asarray(new[:2], float) - path[-1][:2]
                        if np.linalg.norm(dxy) > 0.0015:
                            w = np.clip((cum - st["sv"]) / max(L - st["sv"], 1e-6), 0.0, 1.0)
                            path[:, :2] += 0.5 * dxy[None, :] * w[:, None]; self.retargets += 1
                if st["tau"] >= T:
                    break
                if n_i == 0 and st["tau"] > 0.5:
                    n_i = 1
                    if math.dist(tp, self.tcp_log[-min(len(self.tcp_log), -(-30 // self.spf))][1:4]) < 0.003 and math.dist(p, p0) > 0.05:
                        raise RuntimeError(f"arm not following IK target in {label}: tcp={list(tp)}, target={list(p)}")
                if self.f - f0 > 900:
                    raise RuntimeError(f"leg {label} too slow (>15 s)")
        finally:
            if sub is not None:
                sub.unsubscribe()
            gl["_steps_per_update"] = saved["spu"]; gl["_get_fabric_world_poses"] = saved["fab"]
            if old_sub is not None and inst._running:
                inst._sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(inst._on_update, name="arm310d_ik_follow_controller_update")
        self.leg_held = st["held"]
        self.mark(label, leg_s=round((self.f - f0) / 60.0, 2), len_m=round(L, 3), held_frames=st["held"], step_follow_calls=st["calls"],
                  follower_before=fol0, follower_after=dict(running=inst._running, error_count=inst._error_count))

    def joint_ratios(self):
        """11/10 G1 U2a: |joint speed| / limit of the 6 arm joints (None if unavailable)."""
        try:
            if getattr(self, "_jvlim", None) is None:
                self._jvlim = _joint_vel_limits()
            if self._jvlim is None:
                return None
            art = builtins._arm310d_ik_follow_controller_registry["instance"]._art
            return np.abs(art.get_dof_velocities().numpy()[0][:6]) / self._jvlim
        except Exception:
            return None

    async def arrive(self, xyz, tol=0.003, max_frames=150, q=None):
        n = 0
        while n < max_frames:
            p, tq = self.tcp()
            if math.dist(p, xyz) <= tol and (q is None or self.pm.quat_angle_deg(tq, q) <= 2.0):
                break
            await self.frames(1); n += 1
        return n

    # -------------------------------------------------------------- gripper
    def finger_deg(self):
        c = builtins._arm310d_ik_follow_controller_registry["instance"]
        return float(c._art.get_dof_positions().numpy()[0][6]) * 180.0 / math.pi

    async def close_once(self, width_mm):
        """Jump to (predicted contact - close_pre_deg), then ramp the command continuously (no settle waits) until the
        finger lags the command by contact_lag_deg = contact; then command contact + squeeze_deg and hold."""
        theta_c = self.planner.open_deg_for_gap(width_mm / 1000.0)
        cmd = max(self.finger_deg(), theta_c - P["close_pre_deg"])
        self.grip["gripper_set_angle_deg"](cmd)
        n = 0
        for _ in range(-(-12 // self.spf)):                   # arrive at the pre-contact angle
            await self.frames(1); n += 1
            if abs(self.finger_deg() - cmd) < 0.5: break
        contact = None
        while cmd < _PK["GRIPPER_CLOSE_DEG"] and n < 80:
            cmd = min(_PK["GRIPPER_CLOSE_DEG"], cmd + P["close_ramp_deg_per_frame"] * self.spf)
            self.grip["gripper_set_angle_deg"](cmd); await self.frames(1); n += 1
            a = self.finger_deg()
            if cmd - a >= P["contact_lag_deg"]:
                contact = a; break
        if contact is not None:
            cmd = min(_PK["GRIPPER_CLOSE_DEG"], contact + P["squeeze_deg"])
            self.grip["gripper_set_angle_deg"](cmd)
            for _ in range(-(-12 // self.spf)):                  # stay put while the pads squeeze: a slightly misaligned object turns flat between them
                await self.frames(1); n += 1
        a = self.finger_deg()
        self.hold_cmd = cmd
        lag = cmd - a
        return dict(theta_contact_pred=round(theta_c, 2), contact_deg=None if contact is None else round(contact, 2),
                    cmd_deg=round(cmd, 2), achieved_deg=round(a, 2), lag_deg=round(lag, 2), frames=n,
                    holding=bool(contact is not None and lag >= P["min_lag_deg"]))

    def gap_mm_for_deg(self, deg):
        """Pad gap (mm) at a finger angle: the planner's GAP_CURVE read the other way round."""
        pts = sorted(self.planner.GAP_CURVE)
        return float(np.interp(deg, [p[0] for p in pts], [p[1] for p in pts]))

    def short_side_mm(self, obj):
        fp = self.planner.footprint(obj)
        return 2000.0 * fp[2] if fp[0] == "circle" else 1000.0 * min(fp[2])

    def grip_firm(self):
        """(achieved deg, lag deg): the finger is held back by the object by `lag` degrees while the command pushes = a firm grip."""
        a = self.finger_deg(); return a, self.hold_cmd - a

    async def squeeze_more(self, extra):
        self.hold_cmd = min(_PK["GRIPPER_CLOSE_DEG"], self.hold_cmd + extra)
        self.grip["gripper_set_angle_deg"](self.hold_cmd)
        await self.frames(8)
        return self.grip_firm()

    def dofs_deg(self):
        c = builtins._arm310d_ik_follow_controller_registry["instance"]
        return [round(float(v) * 180.0 / math.pi, 2) for v in c._art.get_dof_positions().numpy()[0][6:12]]

    @staticmethod
    def _rot(q):
        w, x, y, z = q
        return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])

    def block_in_hand_mm(self, gt_name):
        """Report only: where the released block is, in the TCP frame (mm). Constant while the hand moves = still stuck to the hand."""
        gp = self.truth().get(gt_name); tp, tq = self.tcp()
        return None if gp is None else [round(1000 * float(v), 1) for v in self._rot(tq).T @ (np.asarray(gp) - np.asarray(tp))]

    async def release(self, gt_name=None, max_frames=45, open_deg=0.0):
        """Open fully (or to open_deg, 11/09 option) and WAIT until every gripper joint is open and still, recording a trace for the report."""
        self.grip["gripper_set_angle_deg"](open_deg); n = 0; trace = []; stable = 0; prev = None
        while n < max_frames:
            await self.frames(1); n += 1
            d = self.dofs_deg()
            trace.append(dict(n=n, dofs=d[:2], block_in_hand_mm=self.block_in_hand_mm(gt_name) if (gt_name and n % 3 == 0) else None))
            moving = prev is None or max(abs(a - b) for a, b in zip(d, prev)) > 0.15
            stable = 0 if moving else stable + 1; prev = d
            if n >= 6 and stable >= 3 and (max(abs(v) for v in d) < 2.0 if open_deg <= 0.0 else self.finger_deg() <= open_deg + 2.0):
                break
        return dict(frames=n, dofs_end=d, trace=[t for t in trace if t["block_in_hand_mm"] is not None])

    async def open_to(self, deg, thresh=3.0, max_frames=30):
        self.grip["gripper_set_angle_deg"](deg); n = 0
        while n < max_frames:
            await self.frames(1); n += 1
            if n >= 4 and self.finger_deg() <= deg + thresh:
                break
        return n

    # -------------------------------------------------------------- planning
    def basket_cells(self, bmin, bmax, base_xy):
        nx, ny = P["grid"]; m = P["wall_margin"]
        x0, x1, y0, y1 = bmin[0] + m, bmax[0] - m, bmin[1] + m, bmax[1] - m
        cells = [[x0 + (x1 - x0) * (i + .5) / nx, y0 + (y1 - y0) * (j + .5) / ny] for i in range(nx) for j in range(ny)]
        return sorted(cells, key=lambda c: math.dist(c, base_xy))

    def plan_all(self, doc, target):
        cfg = self.oc["CFG"]; table_z = cfg["table_z_m"]
        A = {d["id"]: d for d in doc["views"][0]["detections"]}
        objs = [dict(A[o["id"]], class_name=o["class_name"], conf=o["conf"], center_xyz_m=o["center_xyz_m"]) for o in doc["objects"]]
        base_xy = self.arm_base_xy()
        todo = self.planner.order_targets(objs, target, base_xy, None)
        plans = []
        for o in todo:
            pl = self.planner.plan_grasp(o, [x for x in objs if x is not o and x.get("center_xyz_m")], table_z)
            if pl["ok"]:
                width = next((t["width_mm"] for t in pl["tried"] if abs(t["yaw_deg"] - round(pl["yaw_deg"], 1)) < 0.2), None)
                plans.append(dict(obj=o, plan=pl, width=width))
        return objs, plans, table_z

    # -------------------------------------------------------------- run
    async def run_fast(self, target, max_picks=12):
        await _PK["ensure_sim"]()
        import omni.usd
        from pxr import Usd
        run = os.path.join(OUT, datetime.datetime.now().strftime("%Y%m%d_%H%M%S")); os.makedirs(run, exist_ok=True)
        st = omni.usd.get_context().get_stage()
        if st.GetPrimAtPath("/World/TestBlocks").IsValid():
            st.SetEditTarget(Usd.EditTarget(st.GetSessionLayer()))
        reg = builtins._arm310d_ik_follow_controller_registry
        g = reg["instance"]._on_update.__globals__
        old_cap = g["MAX_JOINT_DELTA_PER_STEP"]; g["MAX_JOINT_DELTA_PER_STEP"] = P["ik_cap"]
        result = dict(run_dir=run, params=P, target=target)
        try:
            self.t0 = time.time()
            truth0 = self.truth()
            self.watch = True
            t_w = time.time()
            doc = await self.perceive()                       # the only perception: detect() as before (arm goes home after)
            result["perceive_wall_s"] = round(time.time() - t_w, 1)
            await _PK["ensure_sim"](5)                        # a stalled Replicator step leaves timeline auto-update OFF -> arm frozen
            objs, plans, table_z = self.plan_all(doc, target)
            home_p, home_q = self.oc["_pose"]("/World/IKTarget")
            bmin, bmax = self.basket_box(); base_xy = self.arm_base_xy()
            cells = self.basket_cells(bmin, bmax, base_xy)
            # order: greedy nearest from home
            order, cur = [], np.array(home_p[:2])
            left = list(plans)
            while left and len(order) < max_picks:
                nxt = min(left, key=lambda q: math.dist(q["plan"]["tcp_grasp"][:2], cur)); left.remove(nxt); order.append(nxt); cur = np.array(nxt["plan"]["tcp_grasp"][:2])
            result["planned"] = [dict(id=q["obj"]["id"], cls=q["obj"]["class_name"], grasp=q["plan"]["tcp_grasp"], width=q["width"]) for q in order]
            top_all = max([o.get("top_z_m") or (table_z + (o.get("height_mm") or 60) / 1000) for o in objs] + [table_z])
            Zt = P["transit_z"]
            self.mark("detect_done")
            cycles = []; prev_xy_drop = None
            for k, q in enumerate(order):
                pl = q["plan"]; o = q["obj"]; g_ = list(pl["tcp_grasp"]); qd = list(self.pm.quat_down_with_yaw(pl["yaw_deg"]))
                cell = cells[k % len(cells)]
                c = dict(k=k + 1, id=o["id"], cls=o["class_name"], cell=cell)
                self.grip["gripper_set_angle_deg"](pl["open_deg"])           # pre-shape while travelling
                cur_p, _ = self.oc["_pose"]("/World/IKTarget")
                pts, rad = [], []
                if cur_p[2] < Zt - 0.02:                                   # leaving the basket: straight up first
                    pts.append([cur_p[0], cur_p[1], Zt]); rad.append(P["fillet"])
                pts.append([g_[0], g_[1], Zt]); rad.append(P["fillet_down"])
                pts.append(g_)
                self.mark(f"k{k+1}:leg_to_grasp_start")
                await self.fly(pts, qd, f"k{k+1}:to_grasp", rad)
                c['arrive_grasp_frames'] = await self.arrive(g_, 0.003, 150, qd)
                self.mark(f"k{k+1}:at_grasp"); t_arr = self.f
                c["grip"] = await self.close_once(q["width"] or 40.0)
                self.mark(f"k{k+1}:closed", **{kk: c["grip"][kk] for kk in ("holding", "frames")})
                hold_off = g_[2] - table_z
                z_place = P["floor_z"] + hold_off + P["place_clear"]
                pts = [[g_[0], g_[1], Zt], [cell[0], cell[1], Zt], [cell[0], cell[1], z_place]]
                await self.fly(pts, qd, f"k{k+1}:carry_to_place", [P["fillet"], P["fillet_down"]], carrying=True)
                c['arrive_place_frames'] = await self.arrive([cell[0], cell[1], z_place], 0.004, 150, qd)
                c["finger_deg_at_place"] = round(self.finger_deg(), 2)
                self.mark(f"k{k+1}:at_place")
                await self.open_to(0.0, 3.0, 30)
                self.mark(f"k{k+1}:released")
                c["from_arrive_to_release_s"] = round((self.f - t_arr) / 60.0, 2)
                cycles.append(c)
            # go home (above the cell, then home)
            cur_p, _ = self.oc["_pose"]("/World/IKTarget")
            await self.fly([[cur_p[0], cur_p[1], Zt], [home_p[0], home_p[1], Zt], home_p], home_q, "to_home", [P["fillet"]] * 2)
            await self.arrive(home_p, 0.004, 60)
            self.mark("home")
            result["cycles"] = cycles
            truth1 = self.truth(); result["truth_before"] = truth0; result["truth_after"] = truth1
            result["in_basket_after"] = {n: bool(bmin[0] < p[0] < bmax[0] and bmin[1] < p[1] < bmax[1] and p[2] < bmax[2] + 0.02) for n, p in truth1.items()}
            result["events"] = self.ev
            json.dump(self.watch_log, open(os.path.join(run, "truth_watch.json"), "w"))
            result["total_sim_s"] = round(self.f / 60.0, 2); result["wall_s"] = round(time.time() - self.t0, 1)
            arr = np.array(self.tcp_log); np.save(os.path.join(run, "tcp.npy"), arr)
            result["done"] = True
        except Exception:
            result["error"] = traceback.format_exc(); result["done"] = True; result["events"] = self.ev
        finally:
            g["MAX_JOINT_DELTA_PER_STEP"] = old_cap
            json.dump(result, open(os.path.join(run, "result.json"), "w"), indent=1, default=str)
            builtins._lab_fast_last = run
        return result


def start(target="sphere", max_picks=12):
    builtins._lab_fast_task = asyncio.ensure_future(FastPicker().run_fast(target, max_picks))
