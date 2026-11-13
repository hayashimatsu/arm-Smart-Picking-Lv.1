"""lab/live_pick.py -- phase 2: video-mode perception + continuous pick-and-place.

  * No detect() photo stops: the wrist D455 streams (lab/live_perception.py); the arm goes to view A once, the tracker fills in
    within ~1 s, then every pick uses the CURRENT tracks.
  * While the arm flies to the next object, the end point is corrected live from the track (retarget), so a stale or biased
    first estimate does not matter.
  * While an object is held (grasp -> release) the tracker is paused (a lifted object would look like a new object).
  * Records an annotated wrist video (class, confidence, world xyz, distance) -> outputs/lab/live/<run>/wrist.mp4.

    exec fast_pick.py, live_perception.py, live_pick.py ; start_live("sphere")
Result: outputs/lab/live/<time>/result.json
"""
import asyncio, builtins, collections, datetime, json, math, os, subprocess, time, traceback
import numpy as np

LROOT = ROOT
LOUT = os.path.join(LROOT, "outputs", "lab", "live")


def _lab_yaws(fp, _orig=None):
    """Wrist angles to try (lab only; the original planner keeps its own list): circle as before; rectangle: across the short side, then every 10 deg around it
    (nearest first, up to +-80), then across the long side. The width across the fingers must still be <= 70 mm, so a long block only allows small angles."""
    if fp[0] == "circle":
        return _orig(fp)
    n = fp[3] + 90.0
    return [n] + [n + s * a for a in range(10, 81, 10) for s in (1, -1)] + [n + 90.0]


def finger_span_neighbors(planner, xy, yaw_deg, gap_m, others, margin=None):
    """11/10. Neighbours (track ids) inside the area the two OPEN fingers span at view A planning: one rectangle over both finger boxes and the
    gap between them (+ margin, the planner's clearance), tested with the planner's own blocked_by on the view-A objects."""
    m = planner.CLEARANCE_M if margin is None else margin
    span = planner._rect_corners(np.asarray(xy, float), (gap_m + 2 * planner.FINGER_THICK_M + 2 * m, planner.FINGER_WIDTH_M + 2 * m), yaw_deg)
    return planner.blocked_by([span], others)


def approach_legs(cur_p, end_p, transit_z, top_z, q_end, fillet, fillet_down, cfg=None, rotate=False):
    """11/10 E (pure): the legs from the IK target cur_p down to end_p (a grasp point or a place point).
    cfg None and rotate False -> ONE leg, exactly the path of before: (up to transit_z if lower) -> above end_p at transit_z (corner rounded by fillet_down) -> end_p,
    the hand turning to q_end along the whole path (so it is still turning on the way down).
    Else -> leg 1 'fast': the same corner path but ending at pre_z = top_z + pre_above_m above end_p, with the full turn to q_end;
            leg 2 'slow': straight down to end_p, no turn, its own trapezoid (cfg v, a; None = the normal speed)."""
    pts, rad = [], []
    if cur_p[2] < transit_z - 0.02:
        pts.append([cur_p[0], cur_p[1], transit_z]); rad.append(fillet)
    pts.append([end_p[0], end_p[1], transit_z]); rad.append(fillet_down)
    if not cfg and not rotate:
        pts.append(list(end_p)); return [dict(pts=pts, radii=rad, q=q_end, vmax=None, amax=None, part="all")]
    cfg = cfg or {}
    pre_z = min(transit_z, max(float(top_z) + cfg.get("pre_above_m", 0.04), float(end_p[2])))
    pts.append([end_p[0], end_p[1], pre_z])
    return [dict(pts=pts, radii=rad, q=q_end, vmax=None, amax=None, part="fast", pre_z=pre_z),
            dict(pts=[list(end_p)], radii=[], q=q_end, vmax=cfg.get("v"), amax=cfg.get("a"), part="slow", pre_z=pre_z)]


def descent_metrics(rows, top_z, end_xy, low_z):
    """11/10 E (pure, record only): from tcp_log rows (t, TCP xyz, target xyz, finger, target turn to go, TCP turn to go) of one approach:
    the largest sideways TCP-target error and TCP offset from the grasp xy while the TCP is below low_z; the TCP speed (3-D and vertical) and the
    turn still to go when the TCP (open fingertip bottom) passes the part's top face (top_z)."""
    rows = [r for r in rows if len(r) >= 8]
    low = [r for r in rows if r[3] <= low_z]
    out = dict(n=len(rows), n_low=len(low),
               max_lat_err_mm=None if not low else round(1000 * max(math.hypot(r[1] - r[4], r[2] - r[5]) for r in low), 1),
               max_off_grasp_mm=None if not low else round(1000 * max(math.hypot(r[1] - end_xy[0], r[2] - end_xy[1]) for r in low), 1))
    for a, b in zip(rows, rows[1:]):
        if a[3] > top_z >= b[3] and b[0] > a[0]:
            dt = b[0] - a[0]
            out.update(top_cross_t=round(b[0], 3), top_cross_v_mm_s=round(1000 * math.dist(a[1:4], b[1:4]) / dt, 1), top_cross_vz_mm_s=round(1000 * (a[3] - b[3]) / dt, 1),
                       top_cross_turn_to_go_deg=b[9] if len(b) > 9 else None, top_cross_target_turn_to_go_deg=b[8] if len(b) > 8 else None)
            break
    return out


def chain_after_place(ok_pick, slipped, next_touch, have_next):
    """11/10 D (pure): after a place, fly straight to the next part only when the pick was good, the tug test did not slip, the next part is not
    within this grasp's finger range (+ margin) and there is a planned next part. Returns (go_straight, reason to go back to view A or None)."""
    if not ok_pick:
        return False, "pick failed"
    if slipped:
        return False, "slip in the tug test"
    if next_touch:
        return False, "next part within the finger range"
    if not have_next:
        return False, None
    return True, None


def queue_order(objs, base_xy, order_targets):
    """11/10 Q1 (pure, Lin 11/10 rule 1): every part whatever its class, nearest to the arm base first = the planner's own
    order_targets(objs, "any", base_xy) (world-XY distance; it stores o["dist_from_arm_base_m"]). Confidence is not used (rule 5)."""
    return order_targets(list(objs), "any", base_xy)


def queue_pick(ordered, deferred, plan, exclude=()):
    """11/10 Q1 (pure): ordered = parts nearest first (queue_order); deferred = track ids found blocked earlier in this round, in the order
    they were found. The queue = the not-deferred parts in distance order, then the deferred ones (a part blocked at first goes last and is
    planned again when its turn comes). plan(o) -> the planner's dict (ok, ...). Returns (track id, plan, ids newly deferred, queue ids);
    track id None = nothing in the queue is graspable now."""
    by_id = {o["_tid"]: o for o in ordered if o["_tid"] not in exclude}
    queue = [i for i in by_id if i not in deferred] + [i for i in deferred if i in by_id]
    newly = []
    for i in queue:
        pl = plan(by_id[i])
        if pl is not None and pl.get("ok"):
            return i, pl, newly, queue
        if i not in deferred:
            newly.append(i)
    return None, None, newly, queue


def place_legal(bmap, xy, r, yaw_deg, gap_m, planner, outline=None):
    """11/10 Q3-5 (the normal place check of BasketMap, no extra gap): the part (outline, else disc r) and the two open fingers at xy / yaw_deg
    touch no occupied cell and no wall band, and every outline / finger corner lies inside the bin minus the wall band (cells outside the map
    are not drawn, so a shape hanging over the edge is caught here). Returns (legal, dict(occupied cells hit, wall cells hit, corners outside))."""
    occ, wall, _ = bmap._grids(); m = bmap._mask(xy, r, yaw_deg, gap_m, planner, outline)
    R = bmap.RES; lo = (bmap.x0 + bmap.wall_m, bmap.y0 + bmap.wall_m); hi = (bmap.x0 + bmap.nx * R - bmap.wall_m, bmap.y0 + bmap.ny * R - bmap.wall_m)
    corners = list(bmap.outline_poly(xy, yaw_deg, outline)) if outline is not None else \
        [np.asarray(xy, float) + r * np.array([math.cos(a), math.sin(a)]) for a in np.radians(np.arange(0, 360, 15))]
    for b in planner.finger_boxes(np.asarray(xy, float), yaw_deg, 0.004, gap_m):
        corners += list(b)
    out = sum(1 for c in corners if not (lo[0] <= c[0] <= hi[0] and lo[1] <= c[1] <= hi[1]))
    n_occ = int((occ & m).sum()); n_wall = int((wall & m).sum())
    return (n_occ == 0 and n_wall == 0 and out == 0), dict(occ_cells=n_occ, wall_cells=n_wall, corners_out=out)


def fallback_outline_spot(bmap, r, outline, yaw_deg, gaps, planner, yaw_ok=None, step=0.01):
    """11/10 Q3-5 fallback_outline_check: the place point when every normal search failed, checked like a normal place (place_legal: part
    outline + open fingers inside the wall band, on no occupied cell). Candidates on a step grid whose centre is at least half the part's short
    side away from anything occupied / the wall band, the emptiest centre first (as emptiest_spot); at each, the closing-axis yaws 0..175 every 5
    (smallest turn from yaw_deg first; a yaw the wrist cannot take -> its +-180 twin, else skipped); gaps in the order given (the release opening
    first, then the smallest). Returns dict(xy, yaw_deg, gap_mm, centre_clearance_mm, tried) or None (no legal point)."""
    import cv2
    occ, wall, _ = bmap._grids(); R = bmap.RES
    d = cv2.distanceTransform((~(occ | wall)).astype(np.uint8), cv2.DIST_L2, 5) * R
    half = (min(outline["size"]) / 2.0) if outline is not None else r
    xs = np.arange(bmap.x0 + R / 2, bmap.x0 + bmap.nx * R, step); ys = np.arange(bmap.y0 + R / 2, bmap.y0 + bmap.ny * R, step)
    cands = []
    for x in xs:
        for y in ys:
            i, j = int((x - bmap.x0) / R), int((y - bmap.y0) / R)
            if 0 <= i < bmap.nx and 0 <= j < bmap.ny and d[i, j] >= half:
                cands.append((float(d[i, j]), float(x), float(y)))
    cands.sort(key=lambda t: -t[0])
    offs = sorted(range(0, 180, 5), key=lambda a: min(a, 180 - a))
    yaws = []
    for a in offs:
        y_ = yaw_deg + (a if a <= 90 else a - 180)
        if yaw_ok is not None and not yaw_ok(y_):
            y_ = next((v for v in (y_ + 180.0, y_ - 180.0) if yaw_ok(v)), None)
        if y_ is not None:
            yaws.append(y_)
    tried = 0
    for gap in gaps:
        for dc, x, y in cands:
            for yw in yaws:
                tried += 1
                if place_legal(bmap, (x, y), r, yw, gap, planner, outline)[0]:
                    return dict(xy=[round(x, 4), round(y, 4)], yaw_deg=round(float(yw), 2), gap_mm=round(1000 * gap, 1), centre_clearance_mm=round(1000 * dc, 1), tried=tried)
    return None


def own_bin_after(truth, box_of):
    """11/10 Q3-6 (pure, report only): every part against the bin of ITS class (box_of(name) -> (bin label, bmin, bmax) or None), the same
    inside test as in_basket_after. One run_live of the queue places into several bins; in_basket_after (kept) uses only the last bin."""
    out = {}
    for n, p in truth.items():
        b = box_of(n)
        if b is None:
            continue
        lab_, bmin, bmax = b
        out[n] = dict(bin=lab_, inside=bool(bmin[0] < p[0] < bmax[0] and bmin[1] < p[1] < bmax[1] and p[2] < bmax[2] + 0.02))
    return out


def axis_angle(yaw_deg, dir_deg):
    """11/11 Q4 (pure): angle 0..90 between a closing axis (yaw, symmetric mod 180) and a horizontal direction (deg)."""
    return abs(((float(yaw_deg) - float(dir_deg) + 90.0) % 180.0) - 90.0)


def round_along_yaw(grasp_xy, place_xy, y_now, j6_now, finger_hits, wrist_ok, failed_yaws=()):
    """11/11 Q4 A (pure, Lin "A + B"; flange_slip_review.md): the closing axis along the carry direction grasp -> place (both decided at view A,
    D7). theta and theta + 180 are the same grasp (two equal fingers): fold_yaw takes the one nearer the wrist. Checks, in order:
    a yaw that already failed on this part (a retry must change it), finger_hits(yaw) (the planner's finger boxes vs the view-A neighbours),
    wrist_ok(yaw) (the j6 prediction: grasp, place = grasp, home). Returns (yaw or None, record with the reason when None)."""
    theta = math.degrees(math.atan2(float(place_xy[1]) - float(grasp_xy[1]), float(place_xy[0]) - float(grasp_xy[0])))
    fy = fold_yaw(theta, y_now, j6_now)
    rec = dict(carry_dir_deg=round(theta, 2), a_yaw=fy["yaw"], a_j6_grasp_pred=fy["j6"], a_flipped=fy["flipped"])
    if any(axis_angle(fy["yaw"], f) < 5.0 for f in failed_yaws):
        return None, dict(rec, reason="retry: this yaw failed before on this part")
    hits = finger_hits(fy["yaw"])
    if hits:
        return None, dict(rec, reason="finger blocked", blocked_by=list(hits))
    if not wrist_ok(fy["yaw"]):
        return None, dict(rec, reason="wrist range")
    return fy["yaw"], rec


def round_place_step(bmap, spot, r, yaw_a, gap_open, planner, outline, grasp_xy, base_xy, spot_kw, wrist_ok=None, max_angle=15.0):
    """11/11 Q4 A (place = grasp yaw, no wrist turn in the carry): the spot found at the original yaw if it is still free at yaw_a (smallest gap
    tier, outline + fingers as every place check), else a new search at yaw_a only (pick_spot, offset 0). The carry direction to the spot used
    must stay within max_angle of the closing axis. Returns (spot or None, angle, reason)."""
    tiers = spot_kw.get("gap_tiers") or (0.003,)
    ol_a = None if outline is None else dict(outline, grasp_yaw=float(yaw_a))
    if not spot.get("fallback") and bmap.evaluate(spot["xy"], r, yaw_a, gap_open, planner, gap=min(tiers), outline=ol_a)[0]:
        sp = dict(spot, yaw_deg=round(float(yaw_a), 2), yaw_offset_deg=0.0)
    else:
        sp = bmap.pick_spot(r, yaw_a, gap_open, planner, base_xy, **dict(spot_kw, outline=ol_a, yaw_offsets=(0.0,), yaw_ok=wrist_ok))
        if sp is None:
            return None, None, "no place spot at this yaw"
        sp.update(fallback=False, no_spot=False, box_full=False, via="round_along")
    ang = axis_angle(yaw_a, math.degrees(math.atan2(sp["xy"][1] - grasp_xy[1], sp["xy"][0] - grasp_xy[0])))
    if ang > max_angle:
        return None, round(ang, 1), f"place spot moved: {ang:.0f} deg off the closing axis"
    return dict(sp, outline_used=ol_a), round(ang, 1), None


def round_place_offsets(grasp_yaw, offsets, yaw_ok=None):
    """11/11 Q4 B (pure): place-yaw offsets for a round part ordered by the wrist turn in the carry, |wrap(offset)| = |j6 place - j6 grasp|.
    Each offset and its +-180 twin (the same place for two equal fingers) are candidates; one the wrist cannot take is dropped (before: replaced
    by its twin in place, e.g. +15 -> -165, sort 000121 k3). Duplicates (same wrapped turn) once."""
    wrap = lambda a: (a + 180.0) % 360.0 - 180.0
    cand = {}
    for off in offsets:
        for d in (off, off + 180.0, off - 180.0):
            w = round(wrap(d), 6)
            if w in cand or (yaw_ok is not None and not yaw_ok(grasp_yaw + w)):
                continue
            cand[w] = w
    return sorted(cand, key=lambda w: (abs(w), -w))


def fold_yaw(theta, y_now, j6_now, lim=195.0):
    """11/10 F1 (pure): the two-finger grasp is the same after a 180 deg turn of the hand. Of theta and theta + 180 pick the one the wrist reaches
    with the smaller turn from the hand's present yaw y_now, keeping joint_6 (j6_now + turn) inside +-lim. Returns (yaw as y_now + turn, turn,
    joint_6 after, flipped = an odd multiple of 180 away from theta). Sort 163014 c4: theta 171.56, y_now 0, j6 41.1 -> -8.44 (joint 32.7, not 212.7)."""
    wrap = lambda a: (a + 180.0) % 360.0 - 180.0
    best = None
    for m in (0, 1):
        t = wrap(theta + 180.0 * m - y_now); j = j6_now + t
        if abs(j) <= lim and (best is None or abs(t) < abs(best[1])):
            best = (y_now + t, t, j, m == 1)
    if best is None:                                                   # neither fits: the smaller joint excursion
        t0, t1 = wrap(theta - y_now), wrap(theta + 180.0 - y_now)
        t, m = (t0, 0) if abs(j6_now + t0) <= abs(j6_now + t1) else (t1, 1)
        best = (y_now + t, t, j6_now + t, m == 1)
    return dict(yaw=round(best[0], 3), turn=round(best[1], 3), j6=round(best[2], 2), flipped=bool(best[3]))


def drop_detect(samples, thr_deg, window_s=0.2):
    """11/10 G2 U3 (pure): samples = [(sim t, finger deg)] during a carry (deg grows as the finger closes). A held part keeps the finger still;
    when it slips out the finger closes onto nothing. Returns dict(t, delta_deg) at the first sample whose finger is more than thr_deg above the
    smallest value of the last window_s, else None. Sort 162023 k1: 14.9 -> 19.2 deg in 0.15 s = drop; k3: +1.0 deg in 0.6 s = no drop."""
    for i, (t, f) in enumerate(samples):
        lo = min(f_ for t_, f_ in samples[:i + 1] if t - t_ <= window_s + 1e-9)
        if f - lo > thr_deg:
            return dict(t=round(float(t), 3), delta_deg=round(float(f - lo), 2))
    return None


def height_gate(tcp_z, plan_z, tol_m):
    """11/10 (pure): before the close, did the TCP reach the planned grasp height? tol_m None = no check (before). A TCP that stays
    higher than plan + tol_m (sort 152950 c8: 27.5 mm, the fingers sat on the wider lower part) must not close: record, back to view A,
    and the retry has to change the grasp (retry_change)."""
    dz = float(tcp_z) - float(plan_z)
    return dict(ok=tol_m is None or abs(dz) <= tol_m, dz_mm=round(1000 * dz, 1), tol_mm=None if tol_m is None else round(1000 * tol_m, 1))


def width_gate(mode, closed_mm, planned_mm, short_mm, tol, neighbors, dz_mm=None, height_tol_mm=None):
    """11/10. The width check after the close. mode "abort" = before: out of tolerance -> abort. mode "neighbor": out of tolerance aborts only
    when a neighbour was inside the open-finger span at view A (it may be caught between the fingers); an isolated part goes on (the holding
    and tug tests still decide), the excess is recorded. No contact (closed None) always aborts."""
    hi = planned_mm + tol[1]; lo = tol[0] * short_mm
    out = closed_mm is None or not (lo <= closed_mm <= hi)
    high = dz_mm is not None and height_tol_mm is not None and abs(dz_mm) > height_tol_mm          # 11/10: closed above the planned height
    abort = out if (mode != "neighbor" or closed_mm is None) else (out and (bool(neighbors) or high))
    return dict(mode=mode, out_of_tol=bool(out), abort=bool(abort), limit_mm=[round(lo, 1), round(hi, 1)], closed_mm=None if closed_mm is None else round(closed_mm, 1),
                over_mm=None if closed_mm is None or closed_mm <= hi else round(closed_mm - hi, 1), neighbors=list(neighbors or []), height_dz_mm=dz_mm,
                height_miss=bool(high))


class LivePicker(FastPicker):
    def __init__(self):
        super().__init__()
        orig = self.planner.candidate_yaws
        self.planner.candidate_yaws = lambda fp, _o=orig: _lab_yaws(fp, _o)
        self._open_margin0 = self.planner.OPEN_MARGIN_M
        self._fb_margin = None                                           # None = the planner's normal 6 mm finger clearance; 0.002 = last-resort tier (a crowded cluster)
        orig_fb = self.planner.finger_boxes
        self.planner.finger_boxes = lambda c, yaw, margin=None, gap_m=self.planner.FINGER_GAP_OPEN_M, _o=orig_fb: _o(
            c, yaw, margin if margin is not None else (self._fb_margin if self._fb_margin is not None else self.planner.CLEARANCE_M), gap_m)

    async def scan(self, live, target, min_s=0.6, max_s=3.0):
        t0 = self.f
        while True:
            await self.frames(2)
            dt = (self.f - t0) / 60.0
            tk = live.tracks(target, 3)
            if dt >= min_s and tk and all(t["n"] >= 4 for t in tk):
                break
            if dt >= max_s:
                break
        return live.tracks(target, 3)

    def choose(self, live, target, tcp_xy, table_z, exclude_ids, failed_keys=None):
        """Plan every tracked target; return the graspable one nearest to tcp_xy."""
        self.look_n = getattr(self, "look_n", 0) + 1                      # engineer round 2 (record): which view-A decision this is (replays)
        if P.get("queue") == "distance":                                 # 11/10 Q1: one queue over every class, nearest to the arm base first
            return self._choose_queue(live, table_z, exclude_ids, failed_keys)
        allt = live.tracks(None, 2); self.last_skips = []
        objs = []
        for t in allt:
            o = dict(t["obs"]); o["center_xyz_m"] = [float(v) for v in t["center"]]; o["class_name"] = t["class_name"]; o["_tid"] = t["id"]; o["id"] = t["id"]; objs.append(o)     # id = track id (unique)
        cands = []
        for o in objs:
            if o["class_name"] != target or o["_tid"] in exclude_ids:
                continue
            fk = (round(o["center_xyz_m"][0] / 0.02), round(o["center_xyz_m"][1] / 0.02))
            if failed_keys and failed_keys.get(fk, 0) >= 2:                # failed twice at this place: leave it (like the original flow)
                self.last_skips.append(dict(track=o["_tid"], cls=o["class_name"], reason="failed twice at this place")); continue
            tk = next(t for t in allt if t["id"] == o["_tid"])
            if tk["n"] < 3:
                continue
            pl = self.planner.plan_grasp(o, [x for x in objs if x is not o], table_z)
            pl = self._pick_select(o, pl, objs, table_z)                   # 11/11 R5 A1: off = pl unchanged
            pl = self._reach_gate(o, pl)                                   # 11/11 R5 step 2: off = pl unchanged
            if not pl["ok"]:
                self.last_skips.append(dict(track=o["_tid"], cls=o["class_name"], reason=pl["reason"]))
            if pl["ok"]:
                width = next((t["width_mm"] for t in pl["tried"] if abs(t["yaw_deg"] - round(pl["yaw_deg"], 1)) < 0.2), None)
                cands.append(dict(obj=o, plan=pl, width=width, tid=o["_tid"], d=math.dist(pl["tcp_grasp"][:2], tcp_xy)))
        if cands and P.get("choose_rank") == "clear":                   # 11/10 D: the part with the most room around it first (view-A queue)
            room = lambda c: min([math.dist(c["obj"]["center_xyz_m"][:2], o["center_xyz_m"][:2]) for o in objs if o is not c["obj"]] or [9.0])
            return max(cands, key=room), len(objs)
        return min(cands, key=lambda c: c["d"]) if cands else None, len(objs)

    def _pick_select(self, o, pl, objs, table_z):
        """11/11 R5 step 3 A1 (Lin 11/11 rules; task_R5_step3_from_supervisor.md). Both switches off -> pl as it is.
        P["pick_wall_check"]: the planner's own yaw candidates (candidate_yaws, its order), each also checked against the SOURCE basket's inner
          walls (L3: both open fingers along the descent from the pre-grasp, the place_legal idea on the source basket; walls = the source box
          from the stage shrunk by P["pick_wall_m"]); the first that passes; none -> not ok "fingers outside the basket wall".
        P["tilt_grasp"]: every closing yaw 0..175 (5 deg, the wrist turns freely), each: width <= the planner's max; the smallest tilt about the
          CLOSING axis that reaches it (lab/reach_table.py TILT_CLOSE, offline; L1: both pads at the same height); L2 (fix-A1, Lin "OK"): the
          lowest fingertip corner >= the floor (0 mm, recorded), the TCP raised from the planner's z only as far as that needs and never above
          (1) = floor + H/2 (H = the view-A height); fingers clear of the
          view-A neighbours (footprint widened by the lean) and inside the walls (L3, up to the transfer height); z_transfer / whether the
          lab carry height 0.80 is reachable after the turn are recorded (A2 carries at min(0.80, z_transfer), as v1 lowers it). Ranked: smallest tilt, then the narrowest width, then nearest to the planner's yaw.
          A1: a chosen tilt > 0 is NOT executed yet (the motion is A2): not ok "needs a tilted grasp ... (A2)"; tilt 0 -> the plan with that yaw.
        Every decision recorded (self.pick_log[track id], pl["pick_select"]). View-A data only (tracks, the stage's basket box, the offline table)."""
        if not (P.get("pick_wall_check") or P.get("tilt_grasp") or P.get("vertical_tube_rule") or P.get("mid_tube_m1")):
            return pl
        pm_ = self.planner; rt_ = reach_lib(); tilt_on = bool(P.get("tilt_grasp"))
        if not tilt_on and not pl.get("ok"):
            return pl                                                       # wall check only: nothing more to choose from
        fp = pm_.footprint(o); cz = [float(v) for v in o["center_xyz_m"]]; others = [x for x in objs if x is not o]
        top_z = o.get("top_z_m") or (cz[2] * 2 - table_z)
        h_obj = max(0.0, float(top_z) - table_z)                               # 11/11 fix-A1: the part's height seen at view A (L2's H)
        z = float(pl["tcp_grasp"][2]) if pl.get("ok") else max(min(cz[2], top_z - pm_.PAD_CONTACT_M), table_z + pm_.FINGER_DESCENT_M + pm_.FINGERTIP_TABLE_CLEARANCE_M)
        inner = getattr(self, "src_inner", None); rim = P.get("source_rim_z") or getattr(self, "src_rim", None) or 0.6856   # 11/11 A2: the stage's wall top
        base = getattr(self, "q_base_xy", None) or self.arm_base_xy()
        yaw_pl = float(pl["yaw_deg"]) if pl.get("ok") else float(pm_.natural_yaw(fp))
        yaws = [float(y) for y in range(0, 180, 5)] if tilt_on else [((y + 90) % 180) - 90 for y in pm_.candidate_yaws(fp)]
        wrap90 = lambda a: abs(((a + 90.0) % 180.0) - 90.0)
        var_ = (globals().get("VARIANT_BY_CLASS") or {}).get(o["class_name"]) or {}          # the sort's class variant turns the plan afterwards (bracket_mid +90,
        dy_ = float(var_.get("dyaw_deg", 0.0))                                              # width 60): the walls / reach / neighbours are checked at the yaw it ends at
        tried, ok_ = [], []
        pairs_ = [(y_pl, dy_, None) for y_pl in yaws]
        if P.get("mid_tube_m1") and o["class_name"] == "bracket_mid" and var_:      # 11/11 fix item 2 (2): M1 across the tube (no turn, width 60) first,
            pairs_ = [(y_pl, 0.0, "M1") for y_pl in yaws] + [(y_pl, dy_, "M2") for y_pl in yaws]   # M2 (the variant's +90) after it
        own_ = (o.get("_diag_arrays") or {}).get("pts") if P.get("vertical_tube_rule") else None    # 11/11 fix item 2 (1): this part's own view-A points
        raw_ = (o.get("_diag_arrays") or {}).get("pts")                        # whatever rule is on: keep the view-A points for replays (record)
        if raw_ is not None and len(raw_):
            if not hasattr(self, "viewa_pts"):
                self.viewa_pts = {}
            self.viewa_pts[o["_tid"]] = dict(pts=np.asarray(raw_, np.float32), obs={k_: o.get(k_) for k_ in ("class_name", "center_xyz_m", "top_face_size_mm",
                                             "top_face_yaw_deg", "top_z_m", "height_mm", "footprint_size_mm", "footprint_yaw_deg")})   # round 2: kept for replays (record)
            look_ = getattr(self, "look_n", 0)                                  # engineer round 2 (record): every look kept apart (look, track), with every
            if not hasattr(self, "viewa_looks"):                                # detection's fields (not only this one: the neighbours) and the look's
                self.viewa_looks = {}                                           # floor / walls / base, so a replay needs nothing else
            if look_ not in self.viewa_looks:
                self.viewa_looks[look_] = dict(table_z=float(table_z), inner=getattr(self, "src_inner", None), rim=P.get("source_rim_z") or getattr(self, "src_rim", None),
                                               base=getattr(self, "q_base_xy", None),
                                               switches={k_: P.get(k_) for k_ in ("pick_wall_check", "tilt_grasp", "vertical_tube_rule", "mid_tube_m1", "tilt_motion",
                                                                                   "transit_z", "l2_tol_m", "carry_wall_clear_m", "tube_round_min")},
                                               variant=globals().get("VARIANT_BY_CLASS"),
                                               objs={str(x["_tid"]): {k_: v_ for k_, v_ in x.items() if not str(k_).startswith("_")} for x in objs}, pts={})
            self.viewa_looks[look_]["pts"][str(o["_tid"])] = np.asarray(raw_, np.float32)
            if P.get("viewa_all_pts"):                                          # 11/11 C2 (record): every detection of this look keeps its points
                for x_ in objs:
                    px_ = (x_.get("_diag_arrays") or {}).get("pts")
                    if px_ is not None and len(px_) and str(x_["_tid"]) not in self.viewa_looks[look_]["pts"]:
                        self.viewa_looks[look_]["pts"][str(x_["_tid"])] = np.asarray(px_, np.float32)
        tube_top = None; rec_tube = None                                        # 11/11 round 2 (supervisor + plan_fix_A1 item 2): the vertical tube from the part's
        if P.get("vertical_tube_rule") and o["class_name"] in ("flange_round", "bracket_end") and own_ is not None and len(own_):   # own view-A top layer
            P_ = np.asarray(own_, float)
            tl_ = P_[P_[:, 2] >= float(np.percentile(P_[:, 2], 99.5)) - 0.010]     # engineer round 2: the top = the 99.5th percentile (a stray high point
            if len(tl_) >= 20:                                                  # cannot pick the layer)
                ws_ = [float(np.ptp(np.percentile(tl_[:, :2] @ np.array([math.cos(math.radians(a_)), math.sin(math.radians(a_))]), [2, 98])))
                       for a_ in range(0, 180, 5)]
                round_ = min(ws_) / max(max(ws_), 1e-9)                         # engineer round 2: an upright tube's top is round (11/11 GUI 120 parts:
                if round_ < P.get("tube_round_min", 0.85):                      # >= 0.924); a lying tube's top band is a strip (~0.63-0.67, estimate)
                    rec_tube = dict(fallback="top layer not round (lying?): the old width rule", round=round(round_, 3), n_top=int(len(tl_)))
                    tl_ = tl_[:0]
            if len(tl_) >= 20:
                wp_ = float(P.get("tube_width_pct", 2))                       # 11/12 W1 (supervisor): 0.5 -> 0.5-99.5 (the ring's sparse ends kept); none = 2-98
                lo_, hi_ = np.percentile(tl_[:, :2], wp_, axis=0), np.percentile(tl_[:, :2], 100 - wp_, axis=0); tc_ = (lo_ + hi_) / 2.0
                tube_top = tl_; rec_tube = dict(centre=[round(float(v), 4) for v in tc_], shift_mm=round(1000 * math.dist(tc_, cz[:2]), 1), n_top=int(len(tl_)),
                                                round=round(round_, 3))
                cz[0], cz[1] = float(tc_[0]), float(tc_[1])                   # grasp the tube's centre (not the whole mask's)
        for i_, (y_pl, dy_p, mode_p) in enumerate(pairs_):
            y = y_pl + dy_p
            if mode_p and fp[0] != "circle":                                    # M1 = closing axis >= 60 deg from the view-A long side, M2 = <= 30 deg
                d_m = wrap90(y - float(fp[3]))
                if (mode_p == "M1" and d_m < 60) or (mode_p == "M2" and d_m > 30):
                    continue
            w = pm_.width_across(fp, y_pl)
            if tube_top is not None:                                           # round 2: the tube's width along this closing axis, from its top layer
                u_ = np.array([math.cos(math.radians(y_pl)), math.sin(math.radians(y_pl))]); pr_ = tube_top[:, :2] @ u_   # y_pl, as the old rule
                w = float(np.percentile(pr_, 100 - wp_) - np.percentile(pr_, wp_)); r_wsrc = "own top layer"   # 11/12 W1: wp_ as the centre
            elif o["class_name"] == "flange_round" or (P.get("vertical_tube_rule") and o["class_name"] == "bracket_end"):   # the vertical tube: the same width at
                w = min(pm_.width_across(fp, yy) for yy in (0.0, 90.0, float(pm_.natural_yaw(fp))))         # every yaw (flange: Q4 interpretation 2; end: fix item 2 (1))
            r = dict(yaw=round(y, 1), width_mm=round(w * 1000, 1))
            if tube_top is not None:
                r["width_src"] = r_wsrc
            if mode_p:
                r["mid_mode_try"] = mode_p
            if dy_p:
                r["plan_yaw"] = round(y_pl, 1)
            tried.append(r)
            if w > pm_.MAX_OBJECT_WIDTH_M:
                r["why"] = "too wide"; continue
            gap = min(pm_.FINGER_GAP_OPEN_M, (var_["width_mm"] / 1000.0 if var_.get("width_mm") else w) + pm_.OPEN_MARGIN_M); tilt = 0; lk = None
            if tilt_on:
                lk = (rt_["tilt_lookup"](cz[:2], y, o["class_name"], rt_["table"], keep080=True) if P.get("keep080") and P.get("tilt_keep")   # 11/12 O2
                      else rt_["tilt_lookup"](cz[:2], y, o["class_name"], rt_["table"]))
                if not lk["ok"]:
                    r["why"] = "reach: " + lk["why"]; continue
                tilt = int(lk["tilt"]); r["tilt"] = tilt
                r["z_transfer"] = lk.get("z_transfer"); r["transit_080"] = bool(lk.get("transit_080"))
                if lk.get("keep080"):
                    r["keep080"] = True                                         # 11/12 O2 record: the kept tilt instead of the lowered carry   # record: A2 carries at min(0.80, z_transfer) (v1 lowers the transfer too)
                zt_l = float(P["transit_z"]) if lk.get("z_transfer") is None else min(float(P["transit_z"]), float(lk["z_transfer"]))     # 11/11 A2: the held part
                r["carry_clear_mm"] = round(1000 * (zt_l - (z - table_z) - rim), 1)          # (bottom = carry z - grasp height above the floor) over the crate wall top
                if r["carry_clear_mm"] < 1000 * (P.get("carry_wall_clear_m", 0.005) + P.get("carry_sag_m", 0.0)):   # 11/12 O1: + the measured sag (none = as before)
                    r["why"] = "carry: the held part would pass too close to / through the crate wall"; continue
            drop = pm_.FINGER_DESCENT_M + rt_["tilt_tip_drop"](tilt)            # 11/11 fix-A1 L2 (Lin "OK"): fingertips >= the floor (0 mm, the margin recorded),
            z_c = max(z, table_z + drop); half_ = table_z + h_obj / 2.0            # pad centre <= (1) = floor + H/2 (H from view A); the TCP may rise inside that
            r["z"] = round(z_c, 4); r["raise_mm"] = round(1000 * (z_c - z), 1); r["floor_mm"] = round(1000 * (z_c - drop - table_z), 1)
            r["pad_minus_half_h_mm"] = round(1000 * (z_c - half_), 1)
            if z_c > z + 1e-9 and z_c > half_ + P.get("l2_tol_m", 0.001):         # 11/11 R5 round 1 (supervisor): L2 limits a RAISE only (the planner's own height
                                                                                  # passes, 31 GUI closes); a raised pad must stay <= (1) + 1 mm (measurement)
                r["why"] = "L2: the fingertips would reach the floor with the pads at or below (1) = floor + H/2"; continue
            lean = rt_["tilt_frame"](y, tilt)[1]; sh = rt_["FINGER_L"] * abs(math.sin(math.radians(tilt)))
            hl = np.array([-lean[0], -lean[1]]); hl = hl / np.linalg.norm(hl) if np.linalg.norm(hl) > 1e-9 else hl
            u = np.array([math.cos(math.radians(y)), math.sin(math.radians(y))]); off = gap / 2 + pm_.FINGER_THICK_M / 2
            boxes = [pm_._rect_corners(np.asarray(cz[:2]) + s_ * off * u + hl * sh / 2, (pm_.FINGER_THICK_M + 2 * pm_.CLEARANCE_M, pm_.FINGER_WIDTH_M + 2 * pm_.CLEARANCE_M + sh), y)
                     for s_ in (1, -1)]
            hits = pm_.blocked_by(boxes, others)
            if hits:
                r["why"] = "fingers blocked by neighbours"; r["blocked_by"] = hits; continue
            if P.get("vertical_tube_rule"):                                     # 11/11 fix item 2 (1): fingers must not come down on the part itself
                if own_ is None or len(own_) == 0:
                    r["own_pts"] = None                                         # no view-A points kept for this track (recorded, not checked)
                else:
                    if tilt:                                                    # round 2: a tilted finger is a tilted box; a point counts only above its lower face
                        n_own = own_points_tilted(own_, cz[:2], y, tilt, gap, z_c - pm_.FINGER_DESCENT_M, pm_.CLEARANCE_M,
                                                  dims=(rt_["FINGER_T"], rt_["FINGER_W"], rt_["FINGER_L"]), descent=pm_.FINGER_DESCENT_M)
                    else:
                        n_own = own_points_in(boxes, own_, z_c - drop)
                    r["own_pts"] = n_own
                    if n_own > 3:
                        if tilt and P.get("plate_raise_end") and o["class_name"] == "bracket_end":     # 11/11 C2: kept for the plate raise below
                            r["_ctx"] = (i_, y_pl, w, gap, tilt, lk, z_c, half_, drop)
                        r["why"] = "fingers on the part itself (view-A depth)"; continue
            if inner is not None:
                ztop = max(z_c + 0.10, (lk or {}).get("z_transfer") or 0.0) if tilt else z_c + 0.10
                wc = rt_["finger_wall_clear"]([cz[0], cz[1], z_c], y, tilt, gap, inner, rim, ztop); r["wall_mm"] = round(1000 * wc, 1)
                if wc < 0:
                    r["why"] = "fingers outside the basket wall (L3)"; continue
            r["ok"] = True; ok_.append((i_, y_pl, w, gap, tilt, lk, r))
            r["_dy"] = dy_p; r["_mode"] = mode_p
        if not ok_ and P.get("plate_raise_end") and o["class_name"] == "bracket_end" and own_ is not None and len(own_):
            ok_ += plate_raise_pass(tried, own_, cz, rt_, pm_, inner, rim, P)  # 11/11 C2 (Lin "2-6 mm is OK"): only when every direction is blocked
        for r_ in tried:
            r_.pop("_ctx", None)
        rec = dict(switches=dict(pick_wall_check=bool(P.get("pick_wall_check")), tilt_grasp=tilt_on), planner_ok=bool(pl.get("ok")), planner_yaw=round(yaw_pl, 2),
                   grasp_z=round(z, 4), centroid_est_z=round(cz[2], 4), pad_minus_centroid_mm=round(1000 * (z - cz[2]), 1), inner=inner, n_tried=len(tried),
                   n_ok=len(ok_), why_counts={})
        if rec_tube is not None:                                               # round 2: where the tube was found (also when nothing is chosen; or the fallback)
            rec["tube"] = rec_tube
        for r in tried:
            if not r.get("ok"):
                k_ = r["why"].split(":")[0]; rec["why_counts"][k_] = rec["why_counts"].get(k_, 0) + 1
        if not hasattr(self, "pick_log"):
            self.pick_log = {}
        if not ok_:
            rec["chosen"] = None; self.pick_log[o["_tid"]] = dict(rec, track=o["_tid"], cls=o["class_name"], tried=tried)
            return dict(pl, ok=False, reason=f"no grasp direction meets the rules (L1-L3, reach): {rec['why_counts']}", pick_select=rec)
        if tilt_on:
            best = min(ok_, key=lambda t: (t[6].get("_mode") == "M2", abs(t[4]), not t[6].get("transit_080", True), round(t[2] * 200), wrap90(t[1] - yaw_pl)))
        else:
            best = min(ok_, key=lambda t: (t[6].get("_mode") == "M2", t[0])) if P.get("mid_tube_m1") else ok_[0]   # 11/11 fix item 2 (2): M1 first
        i_, y, w, gap, tilt, lk, r = best
        mode_c = r.pop("_mode", None); dy_c = r.pop("_dy", dy_)
        for t_ in ok_:
            t_[6].pop("_mode", None); t_[6].pop("_dy", None)
        rec["chosen"] = dict(r, z_transfer=(lk or {}).get("z_transfer"), lean_to_base=bool(tilt))
        rec["h_view_a_mm"] = round(1000 * h_obj, 1)
        if o["class_name"] == "bracket_mid" and fp[0] != "circle":            # 11/11 fix-A1 record: M1 = across the tube (closing across the view-A long side),
            d_ = wrap90(float(r["yaw"]) - float(fp[3]))                         # M2 = along it, pads on the open ends (closing along the long side)
            rec["chosen"]["mid_mode"] = "M2" if d_ < 30 else ("M1" if d_ > 60 else "between")
        self.pick_log[o["_tid"]] = dict(rec, track=o["_tid"], cls=o["class_name"], tried=tried)
        if tilt and not P.get("tilt_motion", True):                       # 11/11 A2: tilt_motion off = A1 (record only)
            return dict(pl, ok=False, reason=f"needs a tilted grasp ({tilt} deg about the closing axis, yaw {y:.0f}): the motion is not in lab yet (A2)",
                        pick_select=rec)
        if tilt_on and not r.get("transit_080", True) and not P.get("tilt_motion", True):     # straight, but the lab carry at 0.80 is not reachable here
            return dict(pl, ok=False, reason=f"needs a tilted grasp (lower carry {r.get('z_transfer')} m, yaw {y:.0f}): the motion is not in lab yet (A2)",
                        pick_select=rec)
        if mode_c:
            rec["mid_mode_chosen"] = mode_c
        extra_ = {"mid_mode": mode_c} if mode_c else {}
        if tilt_on and (tilt or not r.get("transit_080", True)):             # 11/11 A2: the run_live motion (tilted approach / extract / turn, lower carry)
            extra_["tilt"] = dict(tilt=abs(int(tilt)), z_transfer=r.get("z_transfer"), transit_080=bool(r.get("transit_080", True)), yaw_eff=float(r["yaw"]))
        if pl.get("ok") and abs(wrap90(y - float(pl["yaw_deg"]))) < 1e-6 and abs(r["z"] - float(pl["tcp_grasp"][2])) < 1e-6 \
                and math.dist(cz[:2], pl["tcp_grasp"][:2]) < 1e-6:
            return dict(pl, pick_select=rec, **extra_)
        top = max([x.get("top_z_m") or (table_z + (x.get("height_mm") or 60) / 1000) for x in others + [o]])
        z = r["z"]                                                             # 11/11 fix-A1: the L2 height (may be raised inside (1))
        return dict(ok=True, yaw_deg=round(y, 2), tcp_grasp=[cz[0], cz[1], z], tcp_pre=[cz[0], cz[1], max(z + 0.10, top + pm_.PRE_GRASP_ABOVE_M)],
                    open_deg=round(pm_.open_deg_for_gap(gap), 1), open_gap_mm=round(gap * 1000, 1), raised_for_table_mm=round((z - cz[2]) * 1000, 1),
                    tried=list(pl.get("tried") or []) + [dict(yaw_deg=round(y, 1), width_mm=round(w * 1000, 1), open_gap_mm=round(gap * 1000, 1), blocked_by=[])],
                    pick_select=rec, **extra_)

    async def _tilt_out(self, t_, k, carrying, on_frame=None):
        """11/11 A2: from the grasp, back out along the hand axis to t_["e"] (Zt - 5 mm), then a 5 mm slow straight leg up to Zt while the hand
        turns to straight down (REORIENT_S ~1 s: v = 5 mm/s), as v1 pick_moves (grasp_plan.py 505-517) but with a short leg instead of an
        in-place turn (fast_pick.fly would jump the orientation on a zero-length leg). Returns the straight-down quaternion (the new qd)."""
        ap = P.get("approach") or {}
        await self.fly([t_["e"]], t_["q"], f"k{k+1}:extract", [], carrying=carrying, vmax=ap.get("v"), amax=ap.get("a"), on_frame=on_frame)
        await self.fly([t_["e_up"]], t_["q_down"], f"k{k+1}:reorient", [], carrying=carrying, vmax=0.005 / t_["reorient_s"], amax=0.05, on_frame=on_frame)
        return list(t_["q_down"])

    def _reach_gate(self, o, pl):
        """11/11 R5 step 2 (c) (Lin 11/11 "look the table up"; P["reach_table"], default off = pl returned as it is). At view A (D7), for a
        plan the planner found graspable: lab/reach_table.py reach_decide on the planned grasp point and closing axis (scenes/area.json +
        the offline lowered heights; no IK). Not pickable with today's straight-down grasp at P["transit_z"] (tilt cells until step 3, the
        lowered cells that do not reach it, unknown) -> the plan becomes not ok with reason "unreachable (reach table): ...", so the part is
        skipped like any part the planner cannot grasp (no new stop); every such part is recorded in self.reach_log (track id -> record)."""
        if not P.get("reach_table") or not pl.get("ok"):
            return pl
        if P.get("tilt_grasp"):                                            # 11/11 R5 A1: _pick_select already used the (closing-axis) reach table
            return pl
        rt_ = reach_lib(); g_ = pl["tcp_grasp"]
        d_ = rt_["reach_decide"]((float(g_[0]), float(g_[1])), float(pl["yaw_deg"]), o["class_name"], rt_["table"], P["transit_z"])
        rec_ = dict(track=o["_tid"], cls=o["class_name"], xy=[round(float(g_[0]), 4), round(float(g_[1]), 4)], yaw=round(float(pl["yaw_deg"]), 2),
                    mode=d_["mode"], cell=d_.get("cell"), pick=d_["pick"], why=d_["why"])
        for k_ in ("z_max", "tilt_deg", "tilt_axis", "z_transfer", "table_says"):
            if d_.get(k_) is not None:
                rec_[k_] = d_[k_]
        if d_["pick"]:
            return dict(pl, reach=rec_)
        if not hasattr(self, "reach_log"):
            self.reach_log = {}
        self.reach_log[o["_tid"]] = rec_
        return dict(pl, ok=False, reason="unreachable (reach table): " + d_["why"], reach=rec_)

    def _choose_queue(self, live, table_z, exclude_ids, failed_keys=None):
        """11/10 Q1 (P["queue"] = "distance"). The view-A tracks (picked parts are not in them: the plan of every other part is made again
        without them, no new photo), every class, nearest to the arm base first (queue_order); the first graspable in the queue (queue_pick:
        blocked parts go last and are planned again when their turn comes). Skips as choose(): done / excluded tracks, failed twice here, n < 3.
        Reads only the tracks and the planner (no PhysX truth). q["queue"] = position in the queue, distance, base xy (recorded per part)."""
        allt = live.tracks(None, 2); self.last_skips = []
        objs = []
        for t in allt:
            o = dict(t["obs"]); o["center_xyz_m"] = [float(v) for v in t["center"]]; o["class_name"] = t["class_name"]; o["_tid"] = t["id"]; o["id"] = t["id"]; objs.append(o)
        base = getattr(self, "q_base_xy", None) or self.arm_base_xy()
        elig = []
        for o in objs:
            if o["_tid"] in exclude_ids:
                continue
            fk = (round(o["center_xyz_m"][0] / 0.02), round(o["center_xyz_m"][1] / 0.02))
            if failed_keys and failed_keys.get(fk, 0) >= 2:
                self.last_skips.append(dict(track=o["_tid"], cls=o["class_name"], reason="failed twice at this place")); continue
            if next(t for t in allt if t["id"] == o["_tid"])["n"] < 3:
                continue
            elig.append(o)
        ordered = queue_order(elig, base, self.planner.order_targets)
        held = getattr(self, "q_held", None)                              # planned during the carry: the part in the hand is no obstacle any more

        def plan(o):
            pl = self.planner.plan_grasp(o, [x for x in objs if x is not o and x["_tid"] != held], table_z)
            pl = self._pick_select(o, pl, [x for x in objs if x["_tid"] != held], table_z)   # 11/11 R5 A1: off = pl unchanged
            pl = self._reach_gate(o, pl)                                   # 11/11 R5 step 2: off = pl unchanged
            if not pl["ok"]:
                self.last_skips.append(dict(track=o["_tid"], cls=o["class_name"], reason=pl["reason"]))
            return pl
        if not hasattr(self, "q_deferred"):
            self.q_deferred = []
        tid, pl, newly, queue = queue_pick(ordered, self.q_deferred, plan)
        self.q_deferred += newly
        self.q_last = dict(queue=queue, deferred=list(self.q_deferred), base_xy=[round(float(v), 4) for v in base],
                           dist_m={o["_tid"]: o["dist_from_arm_base_m"] for o in ordered})
        if tid is None:
            return None, len(objs)
        o = next(x for x in ordered if x["_tid"] == tid)
        width = next((t["width_mm"] for t in pl["tried"] if abs(t["yaw_deg"] - round(pl["yaw_deg"], 1)) < 0.2), None)
        qinfo = dict(pos=queue.index(tid) + 1, of=len(queue), dist_m=o["dist_from_arm_base_m"], deferred=tid in self.q_deferred,
                     base_xy=self.q_last["base_xy"], queue=list(queue))
        return dict(obj=o, plan=pl, width=width, tid=tid, d=o["dist_from_arm_base_m"], queue=qinfo), len(objs)

    def _queue_bin(self, cls):
        """11/10 Q1: the bin of this part's class (self.use_bin, given by sort's SortPicker: it stores the map in hand back first).
        Returns (bmin, bmax, BasketMap of that bin) built exactly as at the start of run_live, or None when the bin did not change."""
        if not hasattr(self, "use_bin") or not self.use_bin(cls):
            return None
        bmin, bmax = self.basket_box()
        key_ = (round(bmin[0], 3), round(bmin[1], 3), round(bmax[0], 3), round(bmax[1], 3))
        mem = getattr(builtins, "_lab_basket", None)
        if not mem or mem["key"] != key_:
            mem = builtins._lab_basket = dict(key=key_, map=BasketMap(bmin, bmax, P["floor_z"], bmax[2], P["wall_margin"], known_empty=bool(getattr(builtins, "_lab_basket_empty", False)), res=P.get("place_map_res_m"), **({} if P.get("place_wall_m") is None else dict(wall_m=P["place_wall_m"]))))
            builtins._lab_basket_empty = False
        return bmin, bmax, mem["map"]

    def _fallback_outline(self, bmap, spot, r, outline, yaw_deg, gaps, wrist):
        """11/10 Q3-5: a fallback spot (no normal spot) is replaced by a legal one (fallback_outline_spot) when there is one; it stays a fallback
        (fallback=True, honest_counts counts it). None legal -> the spot as before. Recorded: old point legal?, search time (ms), result."""
        t_ = time.time(); g_ = [g for i, g in enumerate(gaps) if g is not None and g not in gaps[:i]]
        old_ok, old_why = place_legal(bmap, spot["xy"], r, float(spot.get("yaw_deg", yaw_deg)), g_[0], self.planner, outline)
        fc = fallback_outline_spot(bmap, r, outline, yaw_deg, g_, self.planner, None if wrist is None else wrist["ok"])
        rec = dict(old_xy=list(spot["xy"]), old_yaw_deg=spot.get("yaw_deg"), old_legal=bool(old_ok), old_check=old_why, found=fc is not None,
                   gaps_mm=[round(1000 * g, 1) for g in g_], ms=round(1000 * (time.time() - t_), 1))
        if fc is None:
            return dict(spot, outline_check=rec)
        rec.update(fc)
        return dict(spot, xy=list(fc["xy"]), yaw_deg=fc["yaw_deg"], yaw_offset_deg=round(fc["yaw_deg"] - yaw_deg, 2), outline_check=rec)

    def _round_along(self, q, pl, g_, spot, bmap, r_obj, gap_open, outline, home_q, live, spot_kw, base_xy, failed_keys):
        if pl.get("tilt") and P.get("tilt_grasp"):                             # 11/11 A2: a tilted / lower-carry plan keeps the yaw its rules chose
            return None, dict(used="original", reason="tilted grasp: the yaw is fixed by the L1-L3 selection")
        """11/11 Q4 A for one flange_round: round_along_yaw (fingers: planner finger_boxes / blocked_by at the PLANNED opening -- the boss is
        round, its width is the same at every yaw; wrist: _wrist_model, grasp joint unclamped inside the limits + place + home) and
        round_place_step. Returns (dict(yaw, spot, wrist, outline) or None, record). Only view-A tracks, the map and the arm's joint."""
        art_ = builtins._arm310d_ik_follow_controller_registry["instance"]._art; i6_ = list(art_.dof_names).index("joint_6_t")
        j6_ = math.degrees(float(art_.get_dof_positions().numpy()[0][i6_]))
        y_now_ = min(range(360), key=lambda y: self.pm.quat_angle_deg(self.tcp()[1], self.pm.quat_down_with_yaw(y)))
        others = [o for o in self._objs(live) if o["_tid"] != q["tid"]]
        gap = (pl.get("open_gap_mm") or (1000.0 * self.gap_mm_for_deg(pl["open_deg"]) / 1000.0)) / 1000.0
        hits = lambda yw: self.planner.blocked_by(self.planner.finger_boxes(np.asarray(g_[:2], float), yw, gap_m=gap), others)
        if (P.get("pick_wall_check") or P.get("tilt_grasp")) and getattr(self, "src_inner", None) is not None:      # 11/11 R5 A1: L3 for A's yaw too
            hits0_ = hits; rt_ = reach_lib()
            hits = lambda yw: hits0_(yw) + (["basket wall"] if rt_["finger_wall_clear"](list(g_), yw, 0, gap, self.src_inner, P.get("source_rim_z", 0.6856), g_[2] + 0.10) < 0 else [])
        wrap = lambda a: (a + 180.0) % 360.0 - 180.0

        def wrist_ok(yw):
            wm = self._wrist_model(yw, home_q); lim = wm["info"]["limits"]; jg = wm["info"]["j6_now"] + wrap(yw - wm["info"]["y_now"])
            return lim[0] <= jg <= lim[1] and wm["ok"](yw)
        fails = failed_keys.near(q["obj"]["center_xyz_m"][:2]) if hasattr(failed_keys, "near") else []
        yaw_a, rec = round_along_yaw(g_[:2], spot["xy"], float(y_now_), j6_, hits, wrist_ok, [f["yaw"] for f in fails if f.get("yaw") is not None])
        rec.update(orig_yaw=round(float(pl["yaw_deg"]), 2), orig_angle=round(axis_angle(pl["yaw_deg"], rec["carry_dir_deg"]), 1), j6_now=round(j6_, 1), y_now=y_now_)
        if yaw_a is None:
            return None, rec
        wm = self._wrist_model(yaw_a, home_q)
        sp, ang, why = round_place_step(bmap, spot, r_obj, yaw_a, gap_open, self.planner, outline, g_[:2], base_xy, spot_kw, wrist_ok=wm["ok"])
        rec.update(angle_after=ang)
        if sp is None:
            return None, dict(rec, reason=why)
        return dict(yaw=float(yaw_a), spot=sp, wrist=wm, outline=sp.pop("outline_used")), rec

    def _objs(self, live):
        objs = []
        for t in live.tracks(None, 2):
            o = dict(t["obs"]); o["center_xyz_m"] = [float(v) for v in t["center"]]; o["class_name"] = t["class_name"]; o["_tid"] = t["id"]; o["id"] = t["id"]; objs.append(o)
        return objs

    def plan_relocation(self, live, target, tcp_xy, table_z, exclude_ids, done_relocs):
        """Every target is blocked (fingers would hit a neighbour): find a neighbour that can be picked up and put down on a free spot of the table such that a blocked
        target becomes graspable, without blocking another one. Returns dict(blocker=q, spot=[x, y], unblocks=[track ids]) or None.
        Everything comes from the tracks (video), nothing from the stage."""
        pl_ = self.planner; objs = self._objs(live); cfg = self.oc["CFG"]
        blocked = []; self.reloc_diag = dict(blocked=[], blockers=[], spots_tried=0, why=None)
        for o in objs:
            if o["class_name"] != target or o["_tid"] in exclude_ids:
                continue
            pl = pl_.plan_grasp(o, [x for x in objs if x is not o], table_z)
            if pl["ok"]:
                return None                                              # a target is graspable after all
            ok_w = [t for t in pl["tried"] if t["width_mm"] <= pl_.MAX_OBJECT_WIDTH_M * 1000]
            if ok_w:
                blocked.append((o, min(ok_w, key=lambda t: len(t["blocked_by"]))["blocked_by"]))
                self.reloc_diag["blocked"].append(dict(tid=o["_tid"], cls=o["class_name"], blockers=blocked[-1][1], tried=[(t["yaw_deg"], t["width_mm"], t["blocked_by"]) for t in pl["tried"]]))
            else:
                self.reloc_diag["blocked"].append(dict(tid=o["_tid"], cls=o["class_name"], why="too wide for every yaw"))
        cands = []; seen_ids = set(); chain = []; frontier = [bid for _, ids in blocked for bid in ids]
        for _depth in range(3):                                      # a blocker may itself be blocked: follow the chain until a neighbour that CAN be picked up
            nxt = []
            for bid in frontier:
                b = next((x for x in objs if x["_tid"] == bid), None)
                if b is None or bid in seen_ids or done_relocs.get(bid, 0) >= 2:
                    continue
                seen_ids.add(bid)
                bp = pl_.plan_grasp(b, [x for x in objs if x is not b], table_z)
                self.reloc_diag["blockers"].append(dict(tid=bid, cls=b["class_name"], graspable=bool(bp["ok"]), reason=None if bp["ok"] else bp["reason"], done=done_relocs.get(bid, 0), depth=_depth))
                if bp["ok"]:
                    cands.append(dict(b=b, plan=bp))
                else:
                    chain.append(b)
                    ok_w = [t for t in bp["tried"] if t["width_mm"] <= pl_.MAX_OBJECT_WIDTH_M * 1000]
                    if ok_w:
                        nxt += min(ok_w, key=lambda t: len(t["blocked_by"]))["blocked_by"]
            frontier = nxt
            if cands:
                break
        cands.sort(key=lambda c: math.dist(c["plan"]["tcp_grasp"][:2], tcp_xy))
        wx, wy = cfg["workspace_x_m"], cfg["workspace_y_m"]; m = 0.05
        for c in cands:
            b = c["b"]; fpb = pl_.footprint(b); r_b = float(fpb[2]) if fpb[0] == "circle" else 0.5 * math.hypot(*fpb[2])
            rest = [x for x in objs if x is not b]
            xs = np.arange(wx[0] + m, wx[1] - m + 1e-9, 0.02); ys = np.arange(wy[0] + m, wy[1] - m + 1e-9, 0.02)
            pts = sorted(((x, y) for x in xs for y in ys), key=lambda q: math.dist(q, b["center_xyz_m"][:2]))
            for (x, y) in pts:
                if math.dist((x, y), b["center_xyz_m"][:2]) < 0.06:
                    continue
                boxes = pl_.finger_boxes([x, y], c["plan"]["yaw_deg"], 0.006, pl_.FINGER_GAP_OPEN_M) + [pl_._rect_corners(np.array([x, y]), (2 * r_b + 0.03, 2 * r_b + 0.03), 0.0)]
                self.reloc_diag["spots_tried"] += 1
                if pl_.blocked_by(boxes, rest):
                    continue
                moved = dict(b, center_xyz_m=[x, y, b["center_xyz_m"][2]])
                after = rest + [moved]
                freed = [o["_tid"] for o in [o_ for o_, _ in blocked] + chain if o is not b and pl_.plan_grasp(o, [z for z in after if z is not o], table_z)["ok"]]      # a target, or a blocked neighbour in the chain, becomes graspable
                others_ok = all(pl_.plan_grasp(o, [z for z in after if z is not o], table_z)["ok"] or not pl_.plan_grasp(o, [z for z in rest if z is not o], table_z)["ok"]
                                for o in objs if o["class_name"] == target and o["_tid"] not in exclude_ids and o is not b)
                if freed and others_ok:
                    return dict(blocker=dict(obj=b, plan=c["plan"], tid=b["_tid"], width=next((t["width_mm"] for t in c["plan"]["tried"] if abs(t["yaw_deg"] - round(c["plan"]["yaw_deg"], 1)) < 0.2), None)),
                                spot=[round(float(x), 4), round(float(y), 4)], unblocks=freed, r=r_b)
        return None

    def _wrist_model(self, grasp_yaw, home_q, margin=15.0):
        """11/09. joint_6_t (the wrist, axis = world +Z while the tool points down: +1 deg joint = +1 deg yaw) has a +-210 deg range, and every leg turns
        the hand the SHORT way (slerp). A place turn pushed it to the limit twice (sort 11/09 15:46 and 15:55: stuck at +210, every following leg 'too slow').
        Predicts the joint for view -> grasp -> place -> home; ok(place_yaw) = all three inside the range minus `margin`. pick_spot then only searches
        yaws the wrist can take (an unsafe one is replaced by its +-180 equivalent, the outline turned with it)."""
        art = builtins._arm310d_ik_follow_controller_registry["instance"]._art; nd = len(art.dof_names)
        i6 = list(art.dof_names).index("joint_6_t")
        j6 = math.degrees(float(art.get_dof_positions().numpy()[0][i6]))
        lo, hi = art.get_dof_limits(); lo = lo.numpy() if hasattr(lo, "numpy") else np.asarray(lo); hi = hi.numpy() if hasattr(hi, "numpy") else np.asarray(hi)
        lim = (math.degrees(float(lo.reshape(-1, nd)[0][i6])) + margin, math.degrees(float(hi.reshape(-1, nd)[0][i6])) - margin)
        yaw_of = lambda q: min(range(360), key=lambda y: self.pm.quat_angle_deg(q, self.pm.quat_down_with_yaw(y)))
        wrap = lambda a: (a + 180.0) % 360.0 - 180.0
        y_now = yaw_of(self.tcp()[1]); y_home = yaw_of(home_q)
        j_g = min(max(j6 + wrap(grasp_yaw - y_now), lim[0] - margin), lim[1] + margin)   # the joint cannot go past its real limit

        def pred(yk):
            j_p = j_g + wrap(yk - grasp_yaw); return dict(yaw=round(yk, 1), j6_grasp=round(j_g, 1), j6_place=round(j_p, 1), j6_home=round(j_p + wrap(y_home - yk), 1))
        # only place and home count: the grasp joint is decided before the place spot (sort 11/09 16:4x: a grasp at 212 deg made EVERY place "unsafe" ->
        # no spot -> fallback cell -> two bracket_mid stacked)
        # and no turn near 180 deg: the slerp direction is then undefined (it may go the long way into the limit)
        ok = lambda yk: abs(wrap(yk - grasp_yaw)) <= 170.0 and abs(wrap(y_home - yk)) <= 170.0 and \
            all(lim[0] <= v <= lim[1] for k_, v in pred(yk).items() if k_ in ("j6_place", "j6_home"))
        return dict(ok=ok, pred=pred, info=dict(j6_now=round(j6, 1), limits=[round(v, 1) for v in lim], y_now=y_now, y_home=y_home))

    async def confirm_view(self, live, k=None, tol=None, gate=0.03, rescan_target=None, exclude=None):
        """11/10 speed B, shared Live only: at view A, wait for k new still frames processed AFTER the arrival, compare every established track
        with the observations of those frames (track_status) and go at once when all are 'same'. A moved / missing track is dropped and the
        normal scan rebuilds it (rescan). Returns counts per class of the tracks seen since the arrival (n >= 2, not picked, not in exclude(track))."""
        k = P.get("live_confirm_frames", 2) if k is None else k; tol = P.get("live_confirm_tol_m", 0.005) if tol is None else tol
        t_arr = live.sim_f / 60.0; f0 = self.f; w0 = time.time(); i0 = len(live.obs_log)
        while True:
            await self.frames(1)
            good = [p_ for p_ in live.proc_log if p_[0] >= t_arr and p_[2]]
            if len(good) >= k or (self.f - f0) / 60.0 > 3.0:
                break
        new = [o for o in live.obs_log[i0:] if o[0] >= t_arr and not o[6] and o[7] <= live.max_speed]
        st = {}
        for tk in live.tracker.tr:
            if tk["picked"]:
                continue
            st[tk["id"]] = track_status(tk["center"][:2], [o[3][:2] for o in new if o[1] == tk["class_name"]], tol, gate) if tk["n"] >= 3 else \
                ("same" if any(math.dist(tk["center"][:2], o[3][:2]) <= gate for o in new) else "missing")
        dec = confirm_decision(st, len(good), k); removed = []
        if dec != "go":
            for tk in list(live.tracker.tr):
                if not tk["picked"] and st.get(tk["id"]) in ("moved", "missing"):
                    live.tracker.tr.remove(tk); removed.append(dict(id=tk["id"], cls=tk["class_name"], why=st[tk["id"]], xy=[round(float(v), 4) for v in tk["center"][:2]]))
            await self.scan(live, rescan_target, 0.6, 3.0)
        cnt = {}
        for tk in live.tracks(None, 2):
            if tk.get("last_t", -1) >= t_arr - 1e-6 and not (exclude and exclude(tk)):
                cnt[tk["class_name"]] = cnt.get(tk["class_name"], 0) + 1
        live.confirm_fresh = True
        return dict(decision=dec, valid=len(good) >= k, frames=len(good), k=k, sim_s=round((self.f - f0) / 60.0, 3), wall_s=round(time.time() - w0, 3), statuses=dict(collections.Counter(st.values())),
                    removed=removed, counts=cnt)

    async def run_live(self, target, max_picks=12, rec=True, debug_release=False, test=None, failed_keys=None, live=None):
        t_in_ = time.time(); rt_ = self.report_timing = {} if P.get("report_defer") else None    # 11/10 Q3 report_defer: timing only (s since entry), no behaviour change
        _rt = (lambda k_: rt_.__setitem__(k_, round(time.time() - t_in_, 4))) if rt_ is not None else (lambda k_: None)
        await _PK["ensure_sim"]()
        _rt("ensure_sim")
        import omni.usd
        from pxr import Usd
        run = os.path.join(LOUT, datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")); os.makedirs(run, exist_ok=True)
        _rt("run_dir")
        st = omni.usd.get_context().get_stage()
        if st.GetPrimAtPath("/World/TestBlocks").IsValid():
            st.SetEditTarget(Usd.EditTarget(st.GetSessionLayer()))
        reg = builtins._arm310d_ik_follow_controller_registry
        g = reg["instance"]._on_update.__globals__
        old_cap = g["MAX_JOINT_DELTA_PER_STEP"]; g["MAX_JOINT_DELTA_PER_STEP"] = P["ik_cap"]
        if P.get("ik_cap_scale"):                                           # 11/10 G3: the follower's joint speed (60 x cap rad/s) under the slowest limit
            try:
                g["MAX_JOINT_DELTA_PER_STEP"] = scaled_ik_cap(P["ik_cap"], _joint_vel_limits())
            except Exception:
                pass
        result = dict(run_dir=run, params=P, target=target, ik_cap_used=g["MAX_JOINT_DELTA_PER_STEP"]); shared = live is not None; cycles = []     # 11/10 speed B: live given = shared (sort opens it once)
        i_obs0 = len(live.obs_log) if shared else 0; i_proc0 = len(live.proc_log) if shared else 0
        paused0 = bool(getattr(live, "paused", False)) if shared else None   # 11/11 R1: the shared Live's tracker state on entry (restored after an error)
        try:
            self.t0 = time.time(); truth0 = self.truth(); self.watch = True
            self._truth0_rec = truth0                                           # 11/11 corner 1a (report only): the error path's truth_before
            if P.get("truth_quat") and hasattr(self, "truth_q") and not isinstance(self.watch_log, QuatWatch):   # 11/11 C2 fix (record): truth_watch rows
                self.watch_log = QuatWatch(self.watch_log, self)                # [t, {xyz}, {quaternion}]; truth() itself stays xyz
            _rt("picker_clock_t0_and_truth0")
            table_z = self.oc["CFG"]["table_z_m"]
            if not shared:
                live = Live(self.oc)
            live.fp_diag = bool(P.get("outline_diag"))                         # 11/10 stage 4a: record-only footprint diagnosis (default off)
            live.floor_filter = P.get("floor_filter_m")                       # 11/10 G4 U1b
            live.floor_filter_classes = tuple(P.get("floor_filter_classes") or ())
            if rec:
                live.rec_dir = os.path.join(run, "frames"); os.makedirs(live.rec_dir, exist_ok=True); live.rec_n = 0
            if not shared:
                await live.start()
            builtins._lab_live = live
            Zt = P["transit_z"]
            bmin, bmax = self.basket_box(); base_xy = self.arm_base_xy()
            if P.get("pick_wall_check") or P.get("tilt_grasp"):                   # 11/11 R5 A1: the source basket's inner walls (stage box - wall)
                self.src_inner = source_inner(P.get("source_prim", "/World/container_h20"), P.get("pick_wall_m", 0.025)); result["source_inner"] = self.src_inner
                self.src_rim = source_rim(P.get("source_prim", "/World/container_h20")); result["source_rim_z"] = self.src_rim     # 11/11 A2: the crate's wall top (0.7398 here)
            if P.get("queue") == "distance":                                   # 11/10 Q1: the base xy the queue really used (recorded)
                self.q_base_xy = list(base_xy); result["arm_base_xy"] = [round(float(v), 5) for v in base_xy]
            key_ = (round(bmin[0], 3), round(bmin[1], 3), round(bmax[0], 3), round(bmax[1], 3))
            mem = getattr(builtins, "_lab_basket", None)                                                     # the basket is REMEMBERED between runs (layout() empties it and resets this)
            if not mem or mem["key"] != key_:
                mem = builtins._lab_basket = dict(key=key_, map=BasketMap(bmin, bmax, P["floor_z"], bmax[2], P["wall_margin"], known_empty=bool(getattr(builtins, "_lab_basket_empty", False)), res=P.get("place_map_res_m"), **({} if P.get("place_wall_m") is None else dict(wall_m=P["place_wall_m"]))))
                builtins._lab_basket_empty = False
            bmap = live.basket_map = mem["map"]                                                              # + updated from the depth video whenever the hand is high near the basket
            live.basket_gate = lambda: (lambda t: t[2] >= Zt - 0.02 and t[0] >= bmin[0] - 0.35)(self.tcp()[0]); live.basket_on = True
            home_p, home_q = self.oc["_pose"]("/World/IKTarget")            # where the arm started = where it returns to
            if not shared or self.finger_deg() > P.get("shared_open_max_deg", 35.0):
                await self.open_to(0.0, 3.0, 40)                      # start with an empty, open gripper
            else:                                                     # 11/10 speed B: the fingers are already open from the release (Lin saw them open fully
                result["open_skipped_deg"] = round(self.finger_deg(), 2)    # and close again at view A); the leg to the grasp preshapes them as before
            # 1) to view A (one continuous leg), scan while video runs
            va = self.oc["_view_poses"]()[0]
            cur_p, _ = self.oc["_pose"]("/World/IKTarget")
            await self.fly([[cur_p[0], cur_p[1], max(cur_p[2], va["position_m"][2])], va["position_m"]], va["quaternion_wxyz"], "to_view", [P["fillet"]])
            await self.arrive(va["position_m"], 0.004, 90, va["quaternion_wxyz"])
            self.mark("at_view", basket_known=round(float(np.mean(~np.isnan(bmap.top[3:-3, 3:-3]))), 3), stamps=len(bmap.stamps)); live.hud = "scan"
            if shared:                                                # 11/10 speed B: tracks kept from the last look; confirm they did not move
                if getattr(live, "confirm_fresh", False):
                    result["confirm"] = dict(getattr(live, "last_confirm", None) or {}, reused=True)
                else:
                    result["confirm"] = live.last_confirm = await self.confirm_view(live, rescan_target=target, exclude=getattr(live, "count_exclude", None))
                live.confirm_fresh = False; tk = live.tracks(target, 3)
            else:
                tk = await self.scan(live, target)
            self.mark("scan_done", n_tracks=len(live.tracks()), n_target=len(tk))
            result["scan_s"] = round(self.f / 60.0, 2)
            cells = self.basket_cells(bmin, bmax, base_xy)                                                 # only the last resort when the map finds no free spot
            cycles = []; done_ids = set(); tries = {}; k = 0; done_relocs = {}; n_reloc = 0; pre_q = None; chained_in = False
            failed_keys = {} if failed_keys is None else failed_keys          # 11/10: sort hands in its PartFails (kept across cycles)
            while k < max_picks + P["max_relocations"]:
                tcp_p, _ = self.tcp(); reloc = None; self._fb_margin = None; self.planner.OPEN_MARGIN_M = self._open_margin0
                Zt = P["transit_z"]; tilt_ = None                               # 11/11 A2: every part starts from the normal transit height
                keep_mask_off(self)                                             # 11/11 C2: the last part's tilted place mask (if any) removed
                if k > 0 and not [t for t in live.tracks(target, 3) if t["id"] not in done_ids]:
                    self.mark("all_targets_done")                           # every target seen has been picked: go home now, no extra look
                    break
                if pre_q is not None and pre_q["tid"] not in done_ids:            # 11/10 D: the next part planned during the last carry
                    q, n_obj = pre_q, None; chained_in = True
                else:
                    q, n_obj = self.choose(live, target, tcp_p[:2], table_z, done_ids, failed_keys); chained_in = False
                pre_q = None
                if q is None:
                    tk = await self.scan(live, target, 0.4, 1.0); q, n_obj = self.choose(live, target, tcp_p[:2], table_z, done_ids, failed_keys)
                    if q is None:
                        reloc = self.plan_relocation(live, target, tcp_p[:2], table_z, done_ids, done_relocs) if n_reloc < P["max_relocations"] else None
                        if reloc is None:                                             # crowded cluster: last resort, finger clearance 2 mm instead of 6 mm (first for the target itself, then for a neighbour)
                            self._fb_margin = P["tight_margin"]; self.planner.OPEN_MARGIN_M = P["tight_open_margin"]            # also open the gripper only 10 mm (not 20 mm) wider than the block
                            q, n_obj = self.choose(live, target, tcp_p[:2], table_z, done_ids, failed_keys)
                            if q is not None:
                                c_tight = True
                            elif n_reloc < P["max_relocations"]:
                                reloc = self.plan_relocation(live, target, tcp_p[:2], table_z, done_ids, done_relocs)
                            if q is None and reloc is None:                          # still nothing (a ring of blocks that block each other): fingers may touch a neighbour by 3 mm (pads slide over its edge / nudge it)
                                self._fb_margin = P["touch_margin"]
                                q, n_obj = self.choose(live, target, tcp_p[:2], table_z, done_ids, failed_keys)
                            if q is None and reloc is None:
                                self.mark("nothing_graspable", skipped=self.last_skips, reloc=getattr(self, "reloc_diag", None))
                                break
                        if reloc is not None:
                            q = reloc["blocker"]; n_reloc += 1                      # every target is blocked by a neighbour: move that neighbour out of the way first
                q_bin = None
                if P.get("queue") == "distance" and not reloc:                    # 11/10 Q1: this part's own bin and bin map (the previous one stored back)
                    q_bin = self._queue_bin(q["obj"]["class_name"])
                    if q_bin is not None:
                        bmin, bmax, bmap = q_bin; live.basket_map = bmap; cells = self.basket_cells(bmin, bmax, base_xy)
                pl = q["plan"]
                if P.get("fold_yaw") and not reloc:                           # 11/10 F1: the wrist takes the equivalent grasp yaw nearest to where it is
                    art_ = builtins._arm310d_ik_follow_controller_registry["instance"]._art; i6_ = list(art_.dof_names).index("joint_6_t")
                    j6_ = math.degrees(float(art_.get_dof_positions().numpy()[0][i6_]))
                    y_now_ = min(range(360), key=lambda y: self.pm.quat_angle_deg(self.tcp()[1], self.pm.quat_down_with_yaw(y)))
                    fy_ = fold_yaw(float(pl["yaw_deg"]), float(y_now_), j6_)
                    fold_rec_ = dict(grasp_from=round(float(pl["yaw_deg"]), 2), y_now=y_now_, j6_now=round(j6_, 1), **fy_)
                    pl["yaw_deg"] = fy_["yaw"]
                else:
                    fold_rec_ = None
                g_ = list(pl["tcp_grasp"]); qd = list(self.pm.quat_down_with_yaw(pl["yaw_deg"]))
                fpo = self.planner.footprint(q["obj"]); r_obj = float(fpo[2]) if fpo[0] == "circle" else 0.5 * math.hypot(*fpo[2])
                gap_open = self.planner.FINGER_GAP_OPEN_M                    # the release opens the gripper fully: the open fingers must also find room
                if P.get("place_release_open") == "grasp" and not reloc:    # 11/09 option: release only as wide as for the grasp -> narrower fingers in the bin
                    gap_open = min(self.planner.FINGER_GAP_OPEN_M, (q["width"] or self.short_side_mm(q["obj"])) / 1000.0 + self.planner.OPEN_MARGIN_M)
                rw_margin = None if reloc else P.get("place_release_margin_m")      # 11/10: one function for the release opening (check = command)
                if rw_margin is not None:
                    plan_w_mm = float(q["width"] or self.short_side_mm(q["obj"]))
                    gap_open = gap_min = release_gap_m(plan_w_mm, None, rw_margin, self.planner.FINGER_GAP_OPEN_M)
                outline = None                                               # 11/09 option P["place_outline"]: place by the object's measured footprint, not the top-face disc
                if P.get("place_outline") and not reloc:
                    o_ = q["obj"]; fs_ = o_.get("footprint_size_mm"); fc_ = o_.get("footprint_center_xy_m")
                    if fs_ and fc_ is not None:
                        outline = dict(size=(fs_[0] / 1000.0, fs_[1] / 1000.0), yaw0=float(o_.get("footprint_yaw_deg") or 0.0), grasp_yaw=float(pl["yaw_deg"]),
                                       off=[float(fc_[0]) - g_[0], float(fc_[1]) - g_[1]])
                t_spot = time.time()
                keep_ = None                                                    # 11/11 C2 tilt_keep: this part's tilted hand (grasp -> place)
                if P.get("tilt_keep") and P.get("tilt_grasp") and (pl.get("tilt") or {}).get("tilt") and not reloc:
                    keep_ = keep_plan(pl, g_, self.arm_base_xy(), self.pm)       # (the record goes into c once c exists, below)
                round_rec_ = None; b_rec_ = None                                # 11/11 Q4 records (flange_round only, switches on)
                if reloc:
                    spot = dict(xy=list(reloc["spot"]), kind="table", unblocks=reloc["unblocks"])
                else:
                    offs = tuple(P.get("place_yaw_offsets", (0.0,)))
                    if P.get("place_align_deg") is not None and outline is not None:     # 11/09 option: turn so the long side points to place_align_deg (world), tried first
                        d_ = ((float(P["place_align_deg"]) - outline["yaw0"] + 90.0) % 180.0) - 90.0
                        offs = (d_,) + tuple(o_ for o_ in offs if abs(o_ - d_) > 1e-6)
                    wrist = self._wrist_model(float(pl["yaw_deg"]), home_q) if P.get("place_wrist_guard") else None
                    spot_kw = dict(outline=outline, rank=P.get("place_rank", "base"), yaw_offsets=offs, yaw_ok=None if wrist is None else wrist["ok"],
                                   **({} if not P.get("place_gap_tiers") else dict(gap_tiers=tuple(P["place_gap_tiers"]))))
                    if P.get("round_no_wrist_turn") and q["obj"]["class_name"] == "flange_round":     # 11/11 Q4 B: the smallest wrist turn first
                        spot_kw["yaw_offsets"] = tuple(round_place_offsets(float(pl["yaw_deg"]), offs, None if wrist is None else wrist["ok"]))
                        b_rec_ = dict(order=list(spot_kw["yaw_offsets"]))
                    if keep_ is not None:                                         # 11/11 C2 tilt_keep: the tilted hand at the place (mask + lean to the base)
                        keep_mask_on(self, bmap, keep_); yo_ = spot_kw.get("yaw_ok")
                        spot_kw["yaw_ok"] = (lambda y_, yo_=yo_: (yo_ is None or yo_(y_)) and keep_lean_ok(keep_, y_, bmin, bmax, base_xy))
                    if P.get("place_fallback") == "emptiest":                   # 11/10: no spot -> the emptiest point, no_spot / box_full recorded
                        spot = bmap.spot_or_fallback(r_obj, pl["yaw_deg"], gap_open, gap_min if rw_margin is not None else gap_open, self.planner, base_xy, **spot_kw)
                    else:
                        spot = bmap.pick_spot(r_obj, pl["yaw_deg"], gap_open, self.planner, base_xy, **spot_kw)
                    if spot is None:
                        spot = dict(xy=list(cells[k % len(cells)]), gap_mm=None, n_stamps=len(bmap.stamps), fallback=True)     # basket full: best effort
                    if P.get("fallback_outline_check") and spot.get("fallback"):  # 11/10 Q3-5: the fallback point checked like a normal place
                        spot = self._fallback_outline(bmap, spot, r_obj, outline, float(pl["yaw_deg"]), [gap_open, gap_min if rw_margin is not None else None], wrist)
                    if P.get("round_yaw_along_carry") and q["obj"]["class_name"] == "flange_round":   # 11/11 Q4 A: closing axis along the carry
                        a_, round_rec_ = self._round_along(q, pl, g_, spot, bmap, r_obj, gap_open, outline, home_q, live, spot_kw, base_xy, failed_keys)
                        round_rec_["used"] = "A" if a_ is not None else "original"
                        if a_ is not None:
                            pl["yaw_deg"] = a_["yaw"]; qd = list(self.pm.quat_down_with_yaw(a_["yaw"])); spot = a_["spot"]; wrist = a_["wrist"]; outline = a_["outline"]
                            spot_kw = dict(spot_kw, outline=outline, yaw_offsets=(0.0,), yaw_ok=wrist["ok"])   # a re-spot after the close keeps the A yaw
                cell = list(spot["xy"])
                c = dict(k=k + 1, tid=q["tid"], cls=q["obj"]["class_name"], conf=q["obj"]["conf"], plan_grasp=g_, cell=list(cell), spot=spot, place_changes=[], kind="relocate" if reloc else "pick", tight=self._fb_margin is not None,
                         grasp_yaw_deg=round(float(pl["yaw_deg"]), 2), place_search_ms=round(1000 * (time.time() - t_spot), 1), outline=outline)
                if q.get("queue") is not None:                                # 11/10 Q1 record: queue position, distance, bin of this part, bin switch
                    c["queue"] = dict(q["queue"], dest=getattr(self, "dest", None), bin_switched=q_bin is not None, chained_in=bool(chained_in))
                if P.get("outline_diag") and q["obj"].get("fp_diag") is not None:    # 11/10 stage 4a, record only: the chosen target as seen at view A
                    t_npz = time.time(); arr_ = q["obj"].get("_diag_arrays") or {}; f_npz = os.path.join(run, f"k{k+1}_target_view_a.npz")
                    try:
                        np.savez(f_npz, **arr_)
                    except Exception as e_:
                        f_npz = "error: " + repr(e_)
                    c["outline_diag"] = dict(q["obj"]["fp_diag"], npz=f_npz, used=dict(size_mm=q["obj"].get("footprint_size_mm"), yaw_deg=q["obj"].get("footprint_yaw_deg")),
                                             save_ms=round(1000 * (time.time() - t_npz), 1))
                place_yaw = float(spot.get("yaw_deg", pl["yaw_deg"])) if not reloc else float(pl["yaw_deg"])
                if P.get("save_map_snapshot") and not reloc:                  # 11/10 G0 (record only): the bin map this place decision used
                    try:
                        f_map = os.path.join(run, f"k{k+1}_bin_map.npz"); c["map_snapshot"] = dict(path=f_map, free_ratio=bmap.snapshot(f_map))
                    except Exception as e_:
                        c["map_snapshot"] = dict(error=repr(e_))
                if fold_rec_ is not None:                                     # 11/10 F1: a place turn over 90 deg -> the 180-deg twin (the part turns with it:
                    wrap_ = lambda a: (a + 180.0) % 360.0 - 180.0           # its outline flips) if that spot is still free and the wrist can take it
                    tn_ = wrap_(place_yaw - pl["yaw_deg"]); fold_rec_["place_from"] = round(place_yaw, 2); fold_rec_["place_turn_from"] = round(tn_, 2)
                    if abs(tn_) > 90.0 and not spot.get("fallback"):
                        alt_ = pl["yaw_deg"] + wrap_(tn_ + 180.0)
                        ok_alt_ = bmap.evaluate(spot["xy"], r_obj, alt_, gap_open, self.planner, gap=min(spot_kw.get("gap_tiers") or (0.003,)), outline=outline)[0] \
                            and (wrist is None or wrist["ok"](alt_))
                        fold_rec_["place_flip_tried"] = round(alt_, 2); fold_rec_["place_flip_free"] = bool(ok_alt_)
                        if ok_alt_:
                            place_yaw = alt_; spot = dict(spot, yaw_deg=round(alt_, 2), folded=True); c["spot"] = spot
                    fold_rec_["place_yaw"] = round(place_yaw, 2); fold_rec_["place_turn"] = round(wrap_(place_yaw - pl["yaw_deg"]), 2)
                    c["fold"] = fold_rec_
                qp = qd if abs(place_yaw - pl["yaw_deg"]) < 1e-6 else list(self.pm.quat_down_with_yaw(place_yaw))     # 11/09: the hand may turn for the place (yaw_offsets)
                if keep_ is not None:                                           # 11/11 C2: the place keeps the grasp's tilt (turned about world z only)
                    c["tilt_keep"] = dict(keep_["rec"])
                    qp = keep_q(self.pm, keep_, place_yaw); c["tilt_keep"]["place_yaw"] = round(float(place_yaw), 2)
                    c["tilt_keep"]["lean_to_base_at_place"] = bool(keep_lean_ok(keep_, place_yaw, bmin, bmax, base_xy))
                if P.get("place_wrist_guard") and not reloc and wrist is not None:     # 11/09: what the wrist joint is predicted to do (report)
                    c["wrist_guard"] = dict(wrist["info"], place=wrist["pred"](place_yaw))
                c["place_yaw_deg"] = round(place_yaw, 2); c["retry_change"] = q.get("retry_change")
                if round_rec_ is not None or b_rec_ is not None:                # 11/11 Q4 record: yaws, carry-axis angle, predicted wrist turn
                    wr_ = (place_yaw - pl["yaw_deg"] + 180.0) % 360.0 - 180.0
                    fin_ = dict(final_grasp_yaw=round(float(pl["yaw_deg"]), 2), place_yaw=round(place_yaw, 2), j6_turn_pred=round(wr_, 2),
                                angle_final=round(axis_angle(pl["yaw_deg"], math.degrees(math.atan2(cell[1] - g_[1], cell[0] - g_[0]))), 1))
                    if round_rec_ is not None:
                        c["round_yaw"] = dict(round_rec_, **fin_)
                    if b_rec_ is not None:
                        c["round_place"] = dict(b_rec_, **fin_)
                if P.get("width_check") == "neighbor" and not reloc:          # 11/10: neighbours in the open-finger span, from the view A tracks
                    nb_ = finger_span_neighbors(self.planner, g_[:2], pl["yaw_deg"], self.gap_mm_for_deg(pl["open_deg"]) / 1000.0,
                                                [o_ for o_ in self._objs(live) if o_["_tid"] != q["tid"]])
                else:
                    nb_ = None
                live.hud = f"{'move aside' if reloc else 'pick'} {k+1}: {c['cls']} track {q['tid']}"
                preshape = [pl["open_deg"]]                              # applied only once the hand is clear of the block it just released (feedback-2 #1)
                def _preshape_hook():
                    if preshape and self.tcp()[0][2] >= Zt - P["preshape_clear_m"]:
                        self.grip["gripper_set_angle_deg"](preshape.pop())
                cur_p, _ = self.oc["_pose"]("/World/IKTarget")
                top_z_ = float(q["obj"].get("top_z_m") or (g_[2] + 0.02))
                legs_ = approach_legs(cur_p, g_, Zt, top_z_, qd, P["fillet"], P["fillet_down"], P.get("approach"), P.get("rotate_before_descent"))   # 11/10 E
                if P.get("tilt_grasp") and pl.get("tilt") and not reloc:        # 11/11 A2: tilted / lower-carry grasp (the plan from _pick_select)
                    tilt_ = tilt_motion_plan(pl["tilt"], g_, float(pl["yaw_deg"]), self.arm_base_xy(), P["transit_z"], top_z_, (P.get("approach") or {}), self.pm)
                    Zt = tilt_["zt"]; qd = tilt_["q"]
                    legs_ = tilt_approach_legs(cur_p, tilt_, P["fillet"], P.get("approach") or {})
                    c["tilt"] = tilt_["rec"]
                tid = q["tid"]; c["chained"] = bool(chained_in)
                gt_name = min(truth0, key=lambda n: math.dist(truth0[n][:2], g_[:2])) if truth0 else None     # report only

                if test and test.get("shift") and k == 0:                 # test only: move the block after the scan (stage edit, invisible to the method)
                    from isaacsim.core.experimental.prims import RigidPrim
                    rp_ = self.oc["_cached_prim"](RigidPrim, f"/World/TestBlocks/{gt_name}")
                    pos_, quat_ = rp_.get_world_poses(); pos_ = pos_.numpy().copy(); pos_[0, 0] += test["shift"][0]; pos_[0, 1] += test["shift"][1]
                    rp_.set_world_poses(pos_, quat_.numpy().copy()); rp_.set_velocities(np.zeros((1, 3)), np.zeros((1, 3))); c["test_shift_m"] = test["shift"]
                g0 = [g_[0], g_[1]]; end_xy = [g_[0], g_[1]]; c["refine_log"] = []
                others_xy = [np.array(t["center"][:2]) for t in live.tracker.tr if t["id"] != tid]       # the neighbours we already know: never the block we are going for
                fp_ = self.planner.footprint(q["obj"])                       # expected blob extent (mm): a circle -> diameter; a rectangle -> short side .. diagonal (any yaw)
                ext_lo, ext_hi = (2000.0 * fp_[2],) * 2 if fp_[0] == "circle" else (1000.0 * min(fp_[2]), 1000.0 * math.hypot(*fp_[2]))

                def retarget(tp, tid=tid, g_=g_):
                    mode = P["visual_refine"]
                    if mode and g_[2] + P["refine_z_m"][0] <= tp[2] <= g_[2] + P["refine_z_m"][1]:
                        # Final correction from the video while the hand comes down (no stop, no extra leg). The class-free depth blob of the target:
                        # "change": used only when it is far from the scanned position (the block moved), because its centroid is 2-6 mm off (worse than the
                        # still-scan track, 0.1-0.6 mm); "estimate": always (kept to measure the difference).
                        b, age = live.blob_near(end_xy, 0.05, 0.25)
                        if b is not None and (not (ext_lo - 15 <= min(b["extent_mm"]) and max(b["extent_mm"]) <= ext_hi + 15)
                                              or any(float(np.hypot(b["xy"][0] - o_[0], b["xy"][1] - o_[1])) < 0.02 for o_ in others_xy)):
                            b = None                                       # not our block: wrong size (against ITS footprint: short side .. diagonal), or a known neighbour (or the hand)
                        if b is not None:
                            d = float(np.hypot(b["xy"][0] - g0[0], b["xy"][1] - g0[1]))
                            c["refine_log"].append(dict(tcp_z=round(float(tp[2]), 3), d_mm=round(1000 * d, 1)))
                            if mode == "estimate" or d > P["moved_m"]:
                                end_xy[0], end_xy[1] = b["xy"][0], b["xy"][1]
                                return b["xy"]
                        return None
                    if tp[2] < g_[2] + 0.10:                              # last 10 cm: the hand is in the way of the camera
                        return None
                    tk = next((t for t in live.tracker.tr if t["id"] == tid), None)
                    if tk is None or tk["n"] < 3 or live.sim_f / 60.0 - tk["last_t"] > 0.5:
                        return None
                    return tk["center"][:2]
                self.mark(f"k{k+1}:leg_start")
                self.retargets = 0; i_log0 = len(self.tcp_log)
                for li_, lg_ in enumerate(legs_):                              # 11/10 E: one leg as before, or fast to the pre height + slow straight down
                    if li_ == len(legs_) - 1 and len(legs_) > 1 and tilt_ is not None:     # 11/11 A2: down the hand axis, shifted as leg 1's end was retargeted
                        cur_t, _ = self.oc["_pose"]("/World/IKTarget")
                        end_xy[0], end_xy[1] = g_[0] + float(cur_t[0]) - tilt_["pre"][0], g_[1] + float(cur_t[1]) - tilt_["pre"][1]
                        lg_ = dict(lg_, pts=[[end_xy[0], end_xy[1], g_[2]]])
                    elif li_ == len(legs_) - 1 and len(legs_) > 1:
                        cur_t, _ = self.oc["_pose"]("/World/IKTarget"); end_xy[0], end_xy[1] = float(cur_t[0]), float(cur_t[1])   # straight down from where leg 1 ended
                        lg_ = dict(lg_, pts=[[end_xy[0], end_xy[1], g_[2]]])
                    await self.fly(lg_["pts"], lg_["q"], f"k{k+1}:to_grasp" if li_ == len(legs_) - 1 else f"k{k+1}:to_pre_grasp", lg_["radii"],
                                   retarget=retarget if li_ == 0 else None, on_frame=_preshape_hook, vmax=lg_["vmax"], amax=lg_["amax"])
                if preshape:
                    self.grip["gripper_set_angle_deg"](preshape.pop())
                c["retargets"] = self.retargets
                live.paused = True                                       # from here the hand hides the object: freeze the tracker
                af_ = 150 if P.get("grasp_arrive_s") is None else max(1, int(math.ceil(P["grasp_arrive_s"] * 60.0 / self.spf)))   # 11/10: arrive counts APP frames
                n_af = await self.arrive([end_xy[0], end_xy[1], g_[2]], 0.003, af_, qd)      # the corrected end point, not the planned one (else it waits the full 2.5 s)
                c["arrive_grasp"] = dict(frames=n_af, max_frames=af_, sim_s=round(n_af * self.spf / 60.0, 2), timed_out=n_af >= af_)
                fd_ = self.finger_deg(); c["finger_at_grasp"] = dict(deg=round(fd_, 2), gap_mm=round(self.gap_mm_for_deg(fd_), 1), planned_open_deg=pl.get("open_deg"),   # 11/10 record
                                                                     planned_gap_mm=None if pl.get("open_deg") is None else round(self.gap_mm_for_deg(pl["open_deg"]), 1))
                tkn = next((t for t in live.tracker.tr if t["id"] == tid), None)
                tcp_p, _ = self.tcp()
                c["tcp_at_close"] = [round(float(v), 4) for v in tcp_p]
                if P.get("pick_wall_check") or P.get("tilt_grasp"):                # 11/11 R5 A1 record (tilt_check C1 / C3): the hand at the close
                    c["hand_at_close"] = dict(tcp=[round(float(v), 5) for v in tcp_p], quat=[round(float(v), 6) for v in self.tcp()[1]])
                    c["part_at_close"] = part_at_close_rec(gt_name)                 # 11/11 fix-A1 record (scoring only, never used by the arm)
                c["descent"] = dict(descent_metrics(self.tcp_log[i_log0:], top_z_, end_xy, legs_[-1].get("pre_z", g_[2] + 0.10)),      # 11/10 E record
                                    legs=[lg_["part"] for lg_ in legs_], pre_z=legs_[-1].get("pre_z"), top_z=round(top_z_, 4))
                tru_ = self.truth().get(gt_name) if gt_name else None                       # report only: how far from the block's true centre the hand is when it closes
                if tru_:
                    c["grasp_err_mm"] = round(1000 * math.dist(tcp_p[:2], tru_[:2]), 1)
                self.mark(f"k{k+1}:at_grasp"); t_arr = self.f
                if getattr(self, "dbg", None) is not None and debug_release == "next":
                    c["release_debug_prev"] = dict(rows=self.dbg.rows, api_error=self.dbg.err); self.dbg.close(); self.dbg = None
                hg_ = height_gate(tcp_p[2], g_[2], None if reloc else P.get("grasp_height_tol_m"))
                if not hg_["ok"]:                                             # 11/10: the hand did not get down to the grasp height: do not close (-> abort, retry)
                    c["height_miss"] = hg_; c["grip"] = dict(holding=False, theta_contact_pred=None, contact_deg=None, frames=0, skipped="height not reached")
                    self.mark(f"k{k+1}:height_miss", dz_mm=hg_["dz_mm"])
                else:
                    c["grip"] = await self.close_once(q["width"] or 40.0)
                self.mark(f"k{k+1}:closed", holding=c["grip"]["holding"], frames=c["grip"]["frames"])
                pred_c = c["grip"]["theta_contact_pred"]; con_c = c["grip"]["contact_deg"]
                c["width_mismatch_deg"] = None if con_c is None else round(con_c - pred_c, 2)       # information only: the squeeze corrects small yaw errors
                gap_c = None if con_c is None else self.gap_mm_for_deg(con_c)
                short_mm = self.short_side_mm(q["obj"]); plan_w = q["width"] or short_mm
                c["grip_width_mm"] = dict(closed=None if gap_c is None else round(gap_c, 1), planned=round(plan_w, 1), short_side=round(short_mm, 1))
                c["width_mismatch"] = bool(gap_c is None or not (P["width_tol_mm"][0] * short_mm <= gap_c <= plan_w + P["width_tol_mm"][1]))
                if P.get("width_check", "abort") == "abort":
                    firm = c["grip"]["holding"] and not c["width_mismatch"]
                else:                                                         # 11/10: abort on the width only with a neighbour near the fingers
                    c["width_check"] = width_gate(P["width_check"], gap_c, plan_w, short_mm, P["width_tol_mm"], nb_, dz_mm=hg_["dz_mm"],
                                                  height_tol_mm=None if P.get("grasp_height_tol_m") is None else 1000 * P["grasp_height_tol_m"])
                    firm = c["grip"]["holding"] and not c["width_check"]["abort"]
                tugs = 0
                x_p, y_p = g_[0], g_[1]
                if firm:                                                      # tug test: lift 12 mm, the finger must not move and must keep pushing
                    while True:
                        a0, _ = self.grip_firm()
                        await self.fly([[x_p, y_p, g_[2] + P["tug_lift_m"]]], qd, f"k{k+1}:tug", [], carrying=True)
                        a1, lag1 = self.grip_firm(); tugs += 1
                        if a1 - a0 < P["tug_slip_deg"] and lag1 >= P["min_lag_deg"]:
                            break                                             # firm: carry on
                        if tugs >= 3:
                            firm = False; break
                        await self.fly([g_], qd, f"k{k+1}:back_down", [], carrying=True)       # slipping: set it down, squeeze harder, try again
                        await self.squeeze_more(P["squeeze_extra_deg"])
                    c["tug"] = dict(tests=tugs, finger_deg=round(a1, 2), lag_deg=round(lag1, 2))
                if firm and rw_margin is not None:                                # 11/10: the opening the release will really use, from the measured close
                    gap_open = release_gap_m(plan_w_mm, gap_c, rw_margin, self.planner.FINGER_GAP_OPEN_M); t_rc = time.time()
                    tier_ = min(spot_kw.get("gap_tiers") or (0.003,))
                    free_, _v = bmap.evaluate(cell, r_obj, place_yaw, gap_open, self.planner, gap=tier_, outline=outline)
                    c["spot_recheck"] = dict(free=bool(free_), gap_mm=round(1000 * gap_open, 1), tier_mm=round(1000 * tier_, 1), re_spot=False)
                    if not free_:
                        if P.get("place_fallback") == "emptiest":
                            new_ = bmap.spot_or_fallback(r_obj, pl["yaw_deg"], gap_open, gap_min, self.planner, base_xy, **spot_kw)
                            if P.get("fallback_outline_check") and new_.get("fallback"):    # 11/10 Q3-5 (sort 233340 k7 was this path)
                                new_ = self._fallback_outline(bmap, new_, r_obj, outline, float(pl["yaw_deg"]), [gap_open, gap_min], wrist)
                        else:
                            new_ = bmap.pick_spot(r_obj, pl["yaw_deg"], gap_open, self.planner, base_xy, **spot_kw) or spot
                        c["spot_recheck"].update(re_spot=True, from_xy=list(cell), spot=new_)
                        spot = c["spot"] = new_; cell[0], cell[1] = spot["xy"]
                        place_yaw = float(spot.get("yaw_deg", pl["yaw_deg"])); c["place_yaw_deg"] = round(place_yaw, 2)
                        qp = qd if abs(place_yaw - pl["yaw_deg"]) < 1e-6 else list(self.pm.quat_down_with_yaw(place_yaw))
                        if keep_ is not None:                                   # 11/11 C2: still the tilted hand after a new spot
                            qp = keep_q(self.pm, keep_, place_yaw)
                    c["spot_recheck"]["ms"] = round(1000 * (time.time() - t_rc), 1)
                    c["release_width"] = dict(planned_mm=round(plan_w_mm, 1), closed_mm=None if gap_c is None else round(gap_c, 1), margin_mm=round(1000 * rw_margin, 1),
                                              check_mm=round(1000 * gap_open, 2))
                self.abort_leg = None; drop_hist = []
                def _drop_hook():                                             # 11/10 G2 U3: the finger closing onto nothing = the part fell
                    if not P.get("drop_detect_deg") or self.abort_leg:
                        return
                    drop_hist.append((self.f / 60.0, self.finger_deg())); del drop_hist[:-40]
                    r_ = drop_detect(drop_hist[-12:], P["drop_detect_deg"], P.get("drop_window_s", 0.2))
                    if r_ is not None:
                        self.abort_leg = r_
                if firm and keep_ is not None:                                     # 11/11 C2 tilt_keep (Lin: lift at this angle, move, put down): no turn
                    cur_p, _ = self.oc["_pose"]("/World/IKTarget"); ap_ = P.get("approach") or {}; Zt = P["transit_z"]
                    await self.fly([[cur_p[0], cur_p[1], cur_p[2] + float(ap_.get("lift_m", 0.03))]], qd, f"k{k+1}:lift_slow", [], carrying=True,
                                   vmax=ap_.get("v"), amax=ap_.get("a"), on_frame=_drop_hook)
                elif firm and tilt_ is not None:                                   # 11/11 A2: back out along the hand axis, turn to straight down, carry at Zt
                    qd = await self._tilt_out(tilt_, k, carrying=True, on_frame=_drop_hook)
                if firm and P.get("approach") and P["approach"].get("lift_m") and tilt_ is None:      # 11/10 E: slow first lift before the fast carry
                    cur_p, _ = self.oc["_pose"]("/World/IKTarget")
                    await self.fly([[cur_p[0], cur_p[1], cur_p[2] + P["approach"]["lift_m"]]], qd, f"k{k+1}:lift_slow", [], carrying=True,
                                   vmax=P["approach"].get("v"), amax=P["approach"].get("a"), on_frame=_drop_hook)
                if firm and P.get("chain") and not reloc:                         # 11/10 D: plan the next part now (during the carry), from the view-A tracks
                    self.q_held = tid if P.get("queue") == "distance" else None    # 11/10 Q1: the held part is not an obstacle of the next plan
                    nq_, _ = self.choose(live, target, g_[:2], table_z, done_ids | {tid}, failed_keys)
                    self.q_held = None
                    touch_ = bool(nq_ is not None and finger_span_neighbors(self.planner, g_[:2], pl["yaw_deg"], self.gap_mm_for_deg(pl["open_deg"]) / 1000.0,
                                                                         [nq_["obj"]], margin=self.planner.CLEARANCE_M + P.get("chain_touch_m", 0.006)))   # finger range = fingers + the planner's clearance; + 6 mm
                    c["chain_next"] = dict(next_tid=None if nq_ is None else nq_["tid"], touch=touch_, slip=tugs > 1)
                else:
                    nq_ = None; touch_ = False
                if firm:
                    cur_p, _ = self.oc["_pose"]("/World/IKTarget")
                    hold_off = g_[2] - table_z
                    z_place = (table_z if reloc else P["floor_z"]) + hold_off + P["place_clear"]
                    if keep_ is not None:                                         # 11/11 C2: the lowest tilted fingertip >= the bin floor (raise the place if not)
                        up_ = max(0.0, (P["floor_z"] + self.planner.FINGER_DESCENT_M + keep_["tip_drop"]) - z_place)
                        z_place += up_; c["tilt_keep"]["place_raise_mm"] = round(1000 * up_, 1)
                    pts = [[cur_p[0], cur_p[1], Zt], [cell[0], cell[1], Zt], [cell[0], cell[1], z_place]]
                    legs_p = approach_legs(cur_p, [cell[0], cell[1], z_place], Zt, z_place, qp, P["fillet"], P["fillet_down"], P.get("approach"), P.get("rotate_before_descent"))

                    def retarget_place(tp):
                        # In flight, while the hand is still high and far from the spot: the depth video now sees the basket. If the chosen spot is not free any more
                        # (something rolled there / was never seen), take the nearest free one -- the end point of the leg is blended over to it, no stop.
                        if tp[2] < Zt - 0.03 or math.dist(tp[:2], cell) < 0.10 or bmap.fresh < 2:
                            return None
                        free, verified = bmap.evaluate(cell, r_obj, place_yaw, gap_open, self.planner, outline=outline)
                        c["place_verified"] = verified
                        if free:
                            return None
                        new = bmap.pick_spot(r_obj, place_yaw, gap_open, self.planner, base_xy, near=cell, outline=outline)
                        if new is None or math.dist(new["xy"], cell) < 0.01:
                            return None
                        c["place_changes"].append(dict(from_xy=[round(v, 4) for v in cell], to_xy=new["xy"], tcp_xy=[round(float(v), 3) for v in tp[:2]]))
                        cell[0], cell[1] = new["xy"]
                        return list(cell)
                    bmap.fresh = 0
                    if len(legs_p) == 1:
                        await self.fly(pts, qp, f"k{k+1}:carry_to_place", [P["fillet"], P["fillet_down"]], carrying=True, retarget=None if reloc else retarget_place, on_frame=_drop_hook)
                    else:                                                     # 11/10 E: turn finished above the spot, then straight down slowly
                        lp0 = dict(legs_p[0], pts=[[cur_p[0], cur_p[1], Zt], [cell[0], cell[1], Zt], [cell[0], cell[1], legs_p[0]["pre_z"]]])
                        await self.fly(lp0["pts"], qp, f"k{k+1}:carry_to_pre_place", [P["fillet"], P["fillet_down"]], carrying=True, retarget=None if reloc else retarget_place, on_frame=_drop_hook)
                        if not self.abort_leg:
                            await self.fly([[cell[0], cell[1], z_place]], qp, f"k{k+1}:carry_to_place", [], carrying=True, vmax=legs_p[1]["vmax"], amax=legs_p[1]["amax"], on_frame=_drop_hook)
                    if self.abort_leg:                                            # 11/10 G2 U3: the part fell in the carry: no place, open, back to view A (failure path)
                        c["drop_in_carry"] = dict(self.abort_leg, leg=self.ev[-1]["name"] if self.ev else None); self.abort_leg = None
                        self.mark(f"k{k+1}:drop_in_carry", **c["drop_in_carry"])
                        await self.open_to(0.0, 3.0, 30)
                        c["cell"] = [round(v, 4) for v in cell]; c["finger_deg_at_place"] = None; c["holding_at_place"] = False
                        c["from_arrive_to_release_s"] = round((self.f - t_arr) / 60.0, 2); c["leave_trace_block_in_hand_mm"] = []
                    else:
                        c["cell"] = [round(v, 4) for v in cell]; c["basket_frames_seen"] = bmap.fresh
                        await self.arrive([cell[0], cell[1], z_place], 0.004, 150, qp)
                        c["finger_deg_at_place"] = round(self.finger_deg(), 2); self.mark(f"k{k+1}:at_place")
                        if debug_release:                                         # diagnosis: per-frame block/hand record, original fast leave (no slow first lift)
                            self.dbg = ReleaseDebug(self, gt_name); self.dbg_label = "release"
                        rel_deg = self.planner.open_deg_for_gap(gap_open) if gap_open < self.planner.FINGER_GAP_OPEN_M else 0.0
                        c["release"] = await self.release(gt_name, open_deg=rel_deg)
                        if c.get("release_width") is not None:                        # 11/10: the command used the same value as the check
                            rw_ = c["release_width"]; rw_.update(cmd_mm=round(1000 * gap_open, 2), cmd_open_deg=round(rel_deg, 2),
                                                                 cmd_back_mm=round(self.gap_mm_for_deg(rel_deg), 1)); rw_["equal"] = rw_["cmd_mm"] == rw_["check_mm"]
                        c["release_gap_mm"] = round(1000 * gap_open, 1)
                        if not reloc and outline is not None:
                            bmap.add_outline(cell, place_yaw, outline)                 # 11/09: remember the outline, not the top-face disc
                        elif not reloc:
                            bmap.stamps.append((cell[0], cell[1], r_obj + 0.005))
                        self.mark(f"k{k+1}:released", frames=c["release"]["frames"])
                        cur_p, _ = self.oc["_pose"]("/World/IKTarget")
                        lift_tr = []
                        if debug_release == "next":                               # keep recording through the NEXT pick's approach (normal flow, nothing changed)
                            self.dbg_label = "next"
                        elif debug_release:
                            self.dbg_label = "leave"
                            if debug_release == "rounded":                       # as in t5: the next leg starts up, but the corner is rounded (r = P["fillet"]) so sideways motion starts early
                                await self.fly([[cur_p[0], cur_p[1], Zt], [cur_p[0] - 0.30, cur_p[1], Zt]], qd, f"k{k+1}:leave_rounded", [P["fillet"]])
                            else:                                                # straight up to the travel height
                                await self.fly([[cur_p[0], cur_p[1], Zt]], qd, f"k{k+1}:leave_fast", [])
                            await self.frames(20)
                            c["release_debug"] = dict(rows=self.dbg.rows, api_error=self.dbg.err); self.dbg.close(); self.dbg = None
                        elif P.get("slow_leave", True):                              # leave slowly first; trace where the block is relative to the hand
                            def _tr():
                                if self.f % 3 == 0:
                                    lift_tr.append(self.block_in_hand_mm(gt_name))
                            await self.fly([[cur_p[0], cur_p[1], cur_p[2] + 0.03]], qp, f"k{k+1}:leave_slow", [], on_frame=_tr, vmax=0.12, amax=P.get("leave_slow_a", 1.0))
                        c["leave_trace_block_in_hand_mm"] = lift_tr
                        c["from_arrive_to_release_s"] = round((self.f - t_arr) / 60.0, 2)
                        c["holding_at_place"] = bool(self.hold_cmd - c["finger_deg_at_place"] >= P["min_lag_deg"])
                else:                                                         # do not carry a bad grasp: open, go up, look again
                    await self.open_to(0.0, 3.0, 30)
                    if tilt_ is not None:                                         # 11/11 A2: out along the hand axis first, turned straight down
                        qd = await self._tilt_out(tilt_, k, carrying=False)
                    cur_p, _ = self.oc["_pose"]("/World/IKTarget")
                    await self.fly([[cur_p[0], cur_p[1], Zt]], qd, f"k{k+1}:abort_up", [],      # 11/11 fix A (supervisor): a 2-point empty-hand leg through
                                   **degenerate_leg_kw(cur_p, [[cur_p[0], cur_p[1], Zt]]))   # corner_profile has speed 0 (T ~ 3.5e5 s); with vmax it is a trapezoid
                    self.mark(f"k{k+1}:abort", pred=pred_c, contact=con_c, tugs=tugs)
                    c["finger_deg_at_place"] = None; c["from_arrive_to_release_s"] = round((self.f - t_arr) / 60.0, 2); c["holding_at_place"] = False
                ok_pick = c["holding_at_place"]
                c["ok"] = ok_pick
                tries[tid] = tries.get(tid, 0) + 1
                if reloc:
                    done_relocs[tid] = done_relocs.get(tid, 0) + 1
                elif ok_pick or tries[tid] >= 2:
                    done_ids.add(tid)
                for t_ in live.tracker.tr:
                    if t_["id"] == tid and ok_pick and not reloc:
                        t_["picked"] = True
                if not ok_pick or reloc:                                           # failed grasp: forget the track, look again
                    fk = (round(g_[0] / 0.02), round(g_[1] / 0.02))
                    if not reloc and hasattr(failed_keys, "record"):              # 11/10: sort's PartFails, keyed by the object centre (as choose reads it)
                        failed_keys.record(q["obj"]["center_xyz_m"][:2], cls=c["cls"], yaw=float(pl["yaw_deg"]), grasp_xy=[float(g_[0]), float(g_[1])],
                                           why="height not reached" if c.get("height_miss") else ("width check" if c.get("width_mismatch") and not firm and c["grip"]["holding"] else "not holding"),
                                           closed_mm=(c.get("grip_width_mm") or {}).get("closed"), retry_change=c.get("retry_change"))
                    elif not reloc:
                        failed_keys[fk] = failed_keys.get(fk, 0) + 1
                    t_ = next((t for t in live.tracker.tr if t["id"] == tid), None)
                    if t_ is not None:
                        live.tracker.tr.remove(t_)
                await self.frames(6)
                live.paused = False
                if P.get("chain") and not reloc:                               # 11/10 D
                    if P.get("queue") == "distance":                           # 11/10 Q1b (Lin 23:2x "piles; not so conservative"): in the queue the finger
                        c["next_touch"] = bool(touch_); touch_ = False        # range is recorded only, never a reason to go back to view A
                    go_, why_ = chain_after_place(ok_pick, tugs > 1, touch_, nq_ is not None)
                    c["chain"] = dict(go_straight=go_, back_to_view_a=why_)
                    if go_:
                        pre_q = nq_
                    elif ok_pick and why_:                                    # good pick, but the next part may have been touched / the grasp slipped: look again
                        cur_p, _ = self.oc["_pose"]("/World/IKTarget")
                        await self.fly([[cur_p[0], cur_p[1], Zt], va["position_m"]], va["quaternion_wxyz"], f"k{k+1}:view_check", [P["fillet"]],
                                       **degenerate_leg_kw(cur_p, [[cur_p[0], cur_p[1], Zt], va["position_m"]]))   # 11/11 fix A+: from Zt it is 2 points
                        await self.arrive(va["position_m"], 0.004, 90, va["quaternion_wxyz"])
                        if shared:
                            c["chain"]["look"] = await self.confirm_view(live, rescan_target=target)
                        else:
                            await self.scan(live, target, 0.6, 2.5)
                if not ok_pick or reloc:                                  # back to view A to find the object again (a moved neighbour: see where it is now)
                    cur_p, _ = self.oc["_pose"]("/World/IKTarget")
                    await self.fly([[cur_p[0], cur_p[1], Zt], va["position_m"]], va["quaternion_wxyz"], f"k{k+1}:rescan_view", [P["fillet"]],
                                   **degenerate_leg_kw(cur_p, [[cur_p[0], cur_p[1], Zt], va["position_m"]]))   # 11/11 fix A+: after abort_up it starts at Zt -> 2 points
                    await self.scan(live, target, 1.2 if reloc else 0.8, 2.5)
                cycles.append(c); k += 1
            live.hud = "done"
            keep_mask_off(self)                                                 # 11/11 C2
            Zt = P["transit_z"]                                                 # 11/11 A2: home at the normal transit height
            cur_p, _ = self.oc["_pose"]("/World/IKTarget")
            if shared and math.dist(cur_p, home_p) < 0.01:             # 11/10 speed A/B: already at view A (nothing picked) -> no dip to transit height and back
                result["to_home_skipped"] = True
            else:
                await self.fly([[cur_p[0], cur_p[1], Zt], [home_p[0], home_p[1], Zt], home_p], home_q, "to_home", [P["fillet"]] * 2)
            await self.arrive(home_p, 0.004, 60); self.mark("home")
            _rt("home")
            truth1 = self.truth()
            _rt("truth1")
            json.dump(self.watch_log, open(os.path.join(run, "truth_watch.json"), "w"))
            _rt("truth_watch_json")
            if P.get("save_tcp_log"):                                          # 11/10 E: (sim t, TCP xyz, target xyz, finger deg, target / TCP turn to go deg)
                json.dump([[round(float(v), 5) for v in r] for r in self.tcp_log], open(os.path.join(run, "tcp_log.json"), "w"))
            _rt("tcp_log_json")
            result.update(cycles=cycles, truth_before=truth0, truth_after=truth1, events=self.ev,
                          in_basket_after={n: bool(bmin[0] < p[0] < bmax[0] and bmin[1] < p[1] < bmax[1] and p[2] < bmax[2] + 0.02) for n, p in truth1.items()},
                          total_sim_s=round(self.f / 60.0, 2), wall_s=round(time.time() - self.t0, 1),
                          live=dict(app_frames=live.sim_f, cam_frames=live.n_frames, processed=live.n_proc,
                                    latency_ms_mean=round(float(np.mean(live.lat_ms[i_proc0:])), 0) if live.lat_ms[i_proc0:] else None, errors=live.errors[:5]))
            if hasattr(self, "own_bin_box"):                                   # 11/10 Q3-6 (report only): each part against its own class's bin
                try:
                    result["in_own_bin_after"] = own_bin_after(truth1, self.own_bin_box)
                except Exception as e_:
                    result["in_own_bin_after"] = dict(error=repr(e_))
            _rt("result_update")
            # truth-vs-track report (stage truth only for the report)
            rows = []
            for t in live.tracker.tr:
                nm = min(truth0, key=lambda n: math.dist(truth0[n][:2], t["center"][:2]))
                rows.append(dict(track=t["id"], cls=t["class_name"], conf=round(t["conf"], 2), n=t["n"], truth=nm,
                                 err_xy_mm=round(1000 * math.dist(truth0[nm][:2], t["center"][:2]), 1), picked=t["picked"]))
            result["tracks_vs_truth"] = rows
            _rt("tracks_vs_truth")
            json.dump(live.obs_log[i_obs0:], open(os.path.join(run, "obs_log.json"), "w"))
            _rt("obs_log_json")
            if shared:
                result["proc_log"] = live.proc_log[i_proc0:]                  # 11/10: per processed frame (sim t, worker ms, still view frame, n obs)
            result["done"] = True
        except Exception:
            result["error"] = traceback.format_exc(); result["done"] = True; result["events"] = self.ev; result["cycles"] = cycles      # 11/09: keep what was done before the error
        finally:
            g["MAX_JOINT_DELTA_PER_STEP"] = old_cap
            try:
                if live is not None and not shared:
                    await live.stop()
            except Exception:
                pass
            _rt("live_stop")
            if result.get("error") or not result.get("done"):                 # 11/11 R1+R3: error (or cancelled) path only; the normal path never enters here
                exc_after_error(self, live, shared, paused0, result, run, i_obs0)
            if P.get("pick_wall_check") or P.get("tilt_grasp"):                    # 11/11 R5 A1: every selection (record)
                result["pick_select"] = list(getattr(self, "pick_log", {}).values())
            if getattr(self, "viewa_pts", None):                                   # 11/11 round 2 (record, for replays): every evaluated track's own view-A points
                try:
                    vp_ = self.viewa_pts
                    np.savez_compressed(os.path.join(run, "viewa_tracks.npz"), **{f"t{t_}": v_["pts"] for t_, v_ in vp_.items()})
                    json.dump({str(t_): v_["obs"] for t_, v_ in vp_.items()}, open(os.path.join(run, "viewa_tracks.json"), "w"), default=str)
                    result["viewa_tracks"] = sorted(vp_)
                except Exception as e_:
                    result["viewa_tracks"] = repr(e_)[:120]
            if getattr(self, "viewa_looks", None):                                 # engineer round 2 (record, for replay_gate.py): every look, every track
                try:
                    vl_ = self.viewa_looks
                    np.savez_compressed(os.path.join(run, "viewa_looks.npz"), **{f"l{l_}_t{t_}": p_ for l_, v_ in vl_.items() for t_, p_ in v_["pts"].items()})
                    json.dump({str(l_): {k_: v_ for k_, v_ in d_.items() if k_ != "pts"} for l_, d_ in vl_.items()}, open(os.path.join(run, "viewa_looks.json"), "w"),
                              default=lambda v_: v_.tolist() if hasattr(v_, "tolist") else str(v_))      # numpy -> lists / numbers (a replay reads them back)
                    result["viewa_looks"] = {str(l_): sorted(d_["pts"]) for l_, d_ in vl_.items()}
                except Exception as e_:
                    result["viewa_looks"] = repr(e_)[:120]
            if P.get("reach_table"):                                           # 11/11 R5 step 2: the parts the reach table kept out (record)
                result["reach"] = dict(unreachable=list(getattr(self, "reach_log", {}).values()))
            json.dump(result, open(os.path.join(run, "result.json"), "w"), indent=1, default=str)
            _rt("result_json")
            if rec and live is not None and live.rec_n:
                try:
                    subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "12", "-i", os.path.join(run, "frames", "f%05d.jpg"),
                                      "-c:v", "libx264", "-pix_fmt", "yuv420p", os.path.join(run, "wrist.mp4")])
                except Exception:
                    pass
            builtins._lab_live_last = run
            _rt("ffmpeg_popen")
            try:
                _write_html(run, result)
            except Exception:
                pass
            _rt("html")
        return result


_REACH = {}


REORIENT_S = 1.0                                                     # v1 grasp_plan.py 460: turning a tilted hand back to straight down


def tilt_quat(pm, yaw, tilt_signed):
    """11/11 A2 (pure): quat_down_with_yaw(yaw) turned about the WORLD closing axis c = (cos yaw, sin yaw, 0) by tilt_signed deg; the hand axis
    becomes down*cos t + (c x down)*sin t = reach_table.tilt_frame (env_area._tilt_plans about_close); the closing axis stays horizontal (L1)."""
    h = math.radians(tilt_signed) / 2.0; c = (math.cos(math.radians(yaw)), math.sin(math.radians(yaw)))
    return list(pm.quat_multiply((math.cos(h), c[0] * math.sin(h), c[1] * math.sin(h), 0.0), pm.quat_down_with_yaw(yaw)))


def tilt_motion_plan(ti, g, yaw, base_xy, transit_z, top_z, ap, pm):
    """11/11 A2 (pure): the tilted pick of one part. ti = pl["tilt"] (|tilt|, z_transfer, transit_080); g = grasp TCP; yaw = the FINAL closing
    yaw (after the variant / fold). The sign leans the hand's top toward the base (as the offline table). Zt = min(transit_z, z_transfer).
    pre = on the hand axis at the straight path's pre height; e = on the axis at Zt - 5 mm; e_up = 5 mm above e (the turn leg)."""
    tilt = int(ti.get("tilt") or 0); zt_tab = ti.get("z_transfer")
    zt = float(transit_z) if zt_tab is None else min(float(transit_z), float(zt_tab))
    sgn = 1
    if tilt:
        c = (math.cos(math.radians(yaw)), math.sin(math.radians(yaw))); side = (-c[1], c[0])
        sgn = 1 if (-side[0]) * (base_xy[0] - g[0]) + (-side[1]) * (base_xy[1] - g[1]) >= 0 else -1
    ts = sgn * tilt; t = math.radians(ts); c3 = (math.cos(math.radians(yaw)), math.sin(math.radians(yaw)), 0.0)
    a = (-c3[1] * math.sin(t), c3[0] * math.sin(t), -math.cos(t))
    along = lambda z_: [g[0] - a[0] * (z_ - g[2]) / -a[2], g[1] - a[1] * (z_ - g[2]) / -a[2], z_]
    pre_z = min(zt, max(float(top_z) + float(ap.get("pre_above_m", 0.04)), float(g[2])))
    e = along(zt - 0.005)
    q = tilt_quat(pm, yaw, ts); q_down = list(pm.quat_down_with_yaw(yaw))
    rec = dict(tilt_deg=ts, yaw=round(yaw, 2), approach=[round(v, 5) for v in a], zt=round(zt, 4), z_transfer_table=zt_tab, transit_080=ti.get("transit_080"),
               pre=[round(v, 4) for v in along(pre_z)], e=[round(v, 4) for v in e], q=[round(v, 6) for v in q], reorient_s=REORIENT_S)
    return dict(zt=zt, q=q, q_down=q_down, a=a, pre=along(pre_z), e_app=along(zt), e=e, e_up=[e[0], e[1], zt], reorient_s=REORIENT_S, rec=rec)


def tilt_approach_legs(cur_p, t_, fillet, ap):
    """11/11 A2 (pure): leg 1 (fast) up to Zt if lower -> the point of the hand axis at Zt -> down the axis to the pre-grasp, turning to the tilted
    hand on the way; leg 2 (slow, approach v / a) down the axis to the grasp (run_live gives it the grasp point)."""
    pts, rad = [], []
    if cur_p[2] < t_["zt"] - 0.02:
        pts.append([cur_p[0], cur_p[1], t_["zt"]]); rad.append(fillet)
    pts += [list(t_["e_app"]), list(t_["pre"])]; rad += [fillet, 0.0]
    return [dict(pts=pts, radii=rad, q=t_["q"], vmax=None, amax=None, part="fast", pre_z=t_["pre"][2]),
            dict(pts=[list(t_["pre"])], radii=[], q=t_["q"], vmax=ap.get("v"), amax=ap.get("a"), part="slow", pre_z=t_["pre"][2])]


def degenerate_leg_kw(cur_p, pts, tol=1e-6):
    """11/11 fix A+ (supervisor: only where the leg really degenerates; pure but reads P): {"vmax", "amax"} = P's v_free / a_free when
    P["abort_up_trapezoid"] is on AND the leg [cur_p] + pts has only 2 distinct points (neighbours closer than tol dropped, as rounded_path skips a
    corner with a leg < 1e-6 m and corner_profile drops 0-length segments); else {} (fast_pick.fly's own choice, as before). A 2-point empty-hand
    leg through corner_profile has speed 0 at both ends and no node between (T ~ 3.5e5 s)."""
    if not P.get("abort_up_trapezoid"):
        return {}
    q_ = [np.asarray(cur_p, float)]
    for p_ in pts:
        p_ = np.asarray(p_, float)
        if float(np.linalg.norm(p_ - q_[-1])) >= tol:
            q_.append(p_)
    return {"vmax": P["v_free"], "amax": P["a_free"]} if len(q_) <= 2 else {}


class QuatWatch(list):
    """11/11 C2 fix (record only): fast_pick appends (t, truth()) every 3 frames; this list adds the parts' quaternions (picker.truth_q()) as a
    third column, so truth() keeps returning xyz to every consumer (block_in_hand_mm, ReleaseDebug, grasp_err, in_basket_after, ...)."""
    def __init__(self, rows, pk):
        super().__init__(rows); self._pk = pk

    def append(self, row):
        try:
            row = tuple(row) + (self._pk.truth_q(),)
        except Exception as e_:
            row = tuple(row) + ({"error": repr(e_)[:80]},)
        super().append(row)


def keep_plan(pl, g, base_xy, pm):
    """11/11 C2 tilt_keep (pure): the part's grasp tilt (|tilt| from the plan, the sign leaning the hand's top to the base, as tilt_motion_plan) at
    the FINAL grasp yaw; the hand's top lean direction (horizontal unit); the tilted quaternion; the lowest fingertip's drop below the pads."""
    tilt = int(pl["tilt"]["tilt"]); yaw = float(pl["yaw_deg"])
    c = (math.cos(math.radians(yaw)), math.sin(math.radians(yaw))); side = (-c[1], c[0])
    sgn = 1 if (-side[0]) * (base_xy[0] - g[0]) + (-side[1]) * (base_xy[1] - g[1]) >= 0 else -1
    ts = sgn * tilt; lean = (-side[0] * sgn, -side[1] * sgn)                 # the hand's top (and the finger bodies) lean this way
    q = tilt_quat(pm, yaw, ts); drop = 0.027 / 2 * math.sin(math.radians(tilt))
    return dict(tilt=ts, yaw=yaw, lean=lean, q=q, tip_drop=drop, sh=0.060 * math.sin(math.radians(tilt)),
                rec=dict(tilt_deg=ts, grasp_yaw=round(yaw, 2), lean=[round(v, 4) for v in lean], tip_drop_mm=round(1000 * drop, 1)))


def keep_lean_at(keep, yaw):
    """the lean direction when the closing axis is at yaw (the hand turned about world z by yaw - grasp yaw)"""
    d = math.radians(float(yaw) - keep["yaw"]); l0 = keep["lean"]
    return (l0[0] * math.cos(d) - l0[1] * math.sin(d), l0[0] * math.sin(d) + l0[1] * math.cos(d))


def keep_lean_ok(keep, yaw, bmin, bmax, base_xy):
    """the hand's top leans toward the arm base at the place (as the offline bin reach, planC: top toward the base)"""
    l_ = keep_lean_at(keep, yaw); cx, cy = (bmin[0] + bmax[0]) / 2, (bmin[1] + bmax[1]) / 2
    return l_[0] * (base_xy[0] - cx) + l_[1] * (base_xy[1] - cy) > 0


def keep_q(pm, keep, yaw):
    """the tilted grasp quaternion turned about world z to the place yaw (the part stays flat)"""
    h = math.radians(float(yaw) - keep["yaw"]) / 2.0
    return list(pm.quat_multiply((math.cos(h), 0.0, 0.0, math.sin(h)), keep["q"]))


class _KeepFingers:
    """the planner seen through a tilted hand: finger_boxes widened toward the lean by FINGER_L sin t (the finger bodies above the tips)"""
    def __init__(self, planner, keep):
        self._p, self._k = planner, keep

    def __getattr__(self, n):
        return getattr(self._p, n)

    def finger_boxes(self, center_xy, yaw_deg, margin=None, gap_m=None):
        p = self._p; m = p.CLEARANCE_M if margin is None else margin; g = p.FINGER_GAP_OPEN_M if gap_m is None else gap_m
        a = math.radians(yaw_deg); u = np.array([math.cos(a), math.sin(a)]); off = g / 2 + p.FINGER_THICK_M / 2
        l_ = np.array(keep_lean_at(self._k, yaw_deg)); sh = self._k["sh"]
        return [p._rect_corners(np.asarray(center_xy, float) + s_ * off * u + l_ * sh / 2, (p.FINGER_THICK_M + 2 * m, p.FINGER_WIDTH_M + 2 * m + sh), yaw_deg)
                for s_ in (1, -1)]


def keep_mask_on(pk, bmap, keep):
    """this bin map's _mask sees the tilted fingers until keep_mask_off (instance override; the class method is untouched)"""
    keep_mask_off(pk)
    orig = type(bmap)._mask
    bmap._mask = lambda xy, r, yaw_deg, gap_m, planner, outline=None, orig=orig, bm=bmap: orig(bm, xy, r, yaw_deg, gap_m, _KeepFingers(planner, keep), outline)
    pk._keep_bmap = bmap


def keep_mask_off(pk):
    bm = getattr(pk, "_keep_bmap", None)
    if bm is not None and "_mask" in vars(bm):
        del bm._mask
    pk._keep_bmap = None


def plate_raise_pass(tried, own, cz, rt_, pm_, inner, rim, P):
    """11/11 C2 plate_raise_end (Lin: the pads may sit 2-6 mm over H/2 to clear the plate): every direction is blocked; the ones blocked by the
    part's own points (tilted) are tried again with the TCP raised 1, 2 .. mm; the smallest clear raise + 2 mm, at most 6 mm over (1); the walls
    checked at that height. Returns the ok_ tuples it opened (the rows get ok, z, plate_raise_mm)."""
    out = []
    for r in tried:
        ctx = r.get("_ctx")
        if not ctx:
            continue
        i_, y_pl, w, gap, tilt, lk, z_c, half_, drop = ctx; y = float(r["yaw"])
        for k in range(1, 7):
            dz = k / 1000.0
            n = own_points_tilted(own, cz[:2], y, tilt, gap, z_c + dz - pm_.FINGER_DESCENT_M, pm_.CLEARANCE_M,
                                  dims=(rt_["FINGER_T"], rt_["FINGER_W"], rt_["FINGER_L"]), descent=pm_.FINGER_DESCENT_M)
            if n <= 3:
                up = dz + 0.002
                if up > 0.006 + 1e-9 or z_c + up > half_ + 0.006 + 1e-9:
                    r["plate_raise_why"] = f"clears at +{1000 * dz:.0f} mm but +2 mm margin is over the 6 mm limit"; break
                zr = z_c + up
                if inner is not None:
                    ztop = max(zr + 0.10, (lk or {}).get("z_transfer") or 0.0)
                    wc = rt_["finger_wall_clear"]([cz[0], cz[1], zr], y, tilt, gap, inner, rim, ztop)
                    if wc < 0:
                        r["plate_raise_why"] = "raised: fingers outside the basket wall"; break
                    r["wall_mm"] = round(1000 * wc, 1)
                r.update(ok=True, z=round(zr, 4), plate_raise_mm=round(1000 * up, 1), raise_mm=round(1000 * (zr - z_c), 1) + r.get("raise_mm", 0.0),
                         pad_minus_half_h_mm=round(1000 * (zr - half_), 1), own_pts_raised=n, why_before=r.pop("why", None))
                out.append((i_, y_pl, w, gap, tilt, lk, r)); r["_dy"] = r.get("_dy", 0.0); r["_mode"] = r.get("_mode")
                break
        else:
            r.setdefault("plate_raise_why", "own points still under the fingers at +6 mm")
    return out


def own_points_tilted(pts, c_xy, yaw, tilt, gap, z_tip0, margin, dims=None, descent=0.0):
    """11/11 round 2 (pure): own view-A points under a TILTED open finger. Each finger = a box FINGER_T (along the closing axis c) x FINGER_W (across,
    m) x FINGER_L (up the hand axis), turned by tilt about c; its tip-edge centre at z_tip0, at c_xy +- c (gap/2 + FINGER_T/2). In the plane (h, z),
    h = the horizontal lean direction, the finger's section is a rectangle turned by t: its LOWER face at a horizontal offset d from the tip-edge
    centre is z_tip0 + w0' - (d + w0) tan t for -w0 <= d <= w0 (the tip face) and z_tip0 - w0' + (d - w0) / tan t beyond (the long side rising),
    with w0 = (W/2) cos t, w0' = (W/2) sin t (W = FINGER_W; the margin widens the c and d extents only). A point counts when inside the finger's
    c-extent (+margin) and the box's d-extent and higher than that lower face. dims = (FINGER_T, FINGER_W, FINGER_L) (engineer round 2: the
    reach_table constants; None = the round-2 literals); descent (engineer round 2): the tips move this far along the hand axis while closing, i.e.
    descent sin t towards -h, so d_min reaches that much further (0 = as round 2). Straight-down grasps keep own_points_in."""
    P_ = np.asarray(pts, float)
    t = math.radians(abs(tilt)); c = np.array([math.cos(math.radians(yaw)), math.sin(math.radians(yaw))])
    side = np.array([-c[1], c[0]]) * (1 if tilt >= 0 else -1)                 # tilt_frame: a_h = side * sin t -> the hand's top leans to -side
    h = -side
    T0, W, L = dims or (0.031, 0.027, 0.060)
    T_ = T0 + 2 * margin                                                      # the margin widens the footprint (c and d), not the finger's lower face
    w0, w0z = (W / 2) * math.cos(t), (W / 2) * math.sin(t)
    d_min, d_max = -w0 - margin - descent * math.sin(t), L * math.sin(t) + w0 + margin
    n = 0
    for s in (1, -1):
        tip = np.asarray(c_xy, float) + c * s * (gap / 2 + T0 / 2)
        rel = P_[:, :2] - tip
        along_c = rel @ c; d = rel @ h
        inside = (np.abs(along_c) <= T_ / 2) & (d >= d_min) & (d <= d_max)
        low = np.where(d <= w0, z_tip0 + w0z - (d + w0) * math.tan(t),            # the tip face: from (-w0, +w0z) down to (w0, -w0z)
                       z_tip0 - w0z + (d - w0) / max(math.tan(t), 1e-9))        # the long side rising away from the tip
        n += int((inside & (P_[:, 2] > low)).sum())
    return n


def own_points_in(boxes, pts, z_tip):
    """11/11 fix item 2 (1) (pure): how many of the part's own view-A points lie inside a finger footprint (convex quads) and higher than the
    lowest fingertip (they would be under a finger coming down)."""
    P_ = np.asarray(pts, float)
    P_ = P_[P_[:, 2] > z_tip]
    if len(P_) == 0:
        return 0
    inside = np.zeros(len(P_), bool)
    for b in boxes:
        b = np.asarray(b, float); ok = np.ones(len(P_), bool); sgn = None
        for i in range(len(b)):
            a_, c_ = b[i], b[(i + 1) % len(b)]
            cr = (c_[0] - a_[0]) * (P_[:, 1] - a_[1]) - (c_[1] - a_[1]) * (P_[:, 0] - a_[0])
            sgn = np.sign(cr) if sgn is None else sgn
            ok &= (cr * sgn) >= 0
        inside |= ok
    return int(inside.sum())


def source_rim(prim):
    """11/11 A2: the source crate's top z (its stage box max z; 0.7398 m in lab_cell_g1, not v1's 0.6856); None if not readable."""
    try:
        return round(float(globals()["_box"](prim)[1][2]), 4)
    except Exception:
        return None


def part_at_close_rec(name):
    """11/11 fix-A1 (report only, tilt_check C2): the target part's PhysX pose at the close (the sort's part_pose), never read by any decision."""
    try:
        return dict(name=name, pose=[round(float(v), 5) for v in globals()["part_pose"](name)])
    except Exception as e_:
        return dict(name=name, error=repr(e_)[:120])


def source_inner(prim, wall_m):
    """11/11 R5 A1: ((x0, y0), (x1, y1)) inner walls of the source basket = its stage box (the sort's _box) minus wall_m; None when the box
    cannot be read (then the wall check is skipped and that is recorded)."""
    try:
        lo, hi = globals()["_box"](prim)
        return [[round(lo[0] + wall_m, 4), round(lo[1] + wall_m, 4)], [round(hi[0] - wall_m, 4), round(hi[1] - wall_m, 4)]]
    except Exception:
        return None


def reach_lib():
    """11/11 R5 step 2: lab/reach_table.py (pure, area.json) loaded once, next to this file; its table loaded once."""
    if not _REACH:
        p_ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reach_table.py"); ns_ = {"__name__": "reach_table", "__file__": p_}
        exec(compile(open(p_, encoding="utf-8").read(), p_, "exec"), ns_)
        _REACH.update(reach_decide=ns_["reach_decide"], table=ns_["load_table"]())
        _REACH.update({k_: ns_[k_] for k_ in ("tilt_lookup", "tilt_frame", "finger_wall_clear", "tilt_tip_drop", "FINGER_L", "FINGER_T", "FINGER_W") if k_ in ns_})   # 11/11 R5 A1
    return _REACH


def exc_after_error(pk, live, shared, paused0, result, run, i_obs0):
    """11/11 R1+R3 (P6c diag 3): after run_live failed.
    R1 recover_unpause (default on; P["recover_unpause"]=False = as before): run_live sets live.paused=True before the grasp and only the normal
    path sets it back, so after an error the shared Live stayed paused and every frame at view A was gated out (0 frames). Put back the entry value.
    R3 exception_logs (default on, record only): write truth_watch.json / tcp_log.json (if save_tcp_log, as the normal path) / obs_log.json too."""
    rec = {}
    if shared and live is not None and P.get("recover_unpause", True):
        try:
            rec["paused_at_error"] = bool(getattr(live, "paused", False)); live.paused = paused0; rec["paused_restored_to"] = paused0
        except Exception as e_:
            rec["unpause_error"] = repr(e_)
    if P.get("exception_logs", True):
        files = []
        for fn, get_, on_ in (("truth_watch.json", lambda: pk.watch_log, True),
                              ("tcp_log.json", lambda: [[round(float(v), 5) for v in r] for r in pk.tcp_log], bool(P.get("save_tcp_log"))),
                              ("obs_log.json", lambda: live.obs_log[i_obs0:], live is not None)):
            if not on_:
                continue
            try:
                json.dump(get_(), open(os.path.join(run, fn), "w")); files.append(fn)
            except Exception as e_:
                rec.setdefault("log_errors", {})[fn] = repr(e_)
        rec["logs_written"] = files
    rec.update(error_scoring_rec(pk))                                          # 11/11 corner 1a (report only): scoring on the error path too
    if rec:
        result["after_error"] = rec


def _write_html(run, r):
    """Small report: video + per-pick times + what ended up in the basket (stage truth is for this report only)."""
    rows = "".join(f"<tr><td>{c['k']}</td><td>{c['cls']}{' (moved aside)' if c.get('kind') == 'relocate' else ''}</td><td>{c['conf']:.2f}</td><td>{'yes' if c.get('ok') else 'no'}</td>"
                   f"<td>{c['from_arrive_to_release_s']}</td></tr>" for c in r.get("cycles", []))
    ev = {e["name"]: e["sim_s"] for e in r.get("events", [])}
    rel = [e["sim_s"] for e in r.get("events", []) if e["name"].endswith(":released")]
    gaps = ", ".join(f"{b - a:.1f}" for a, b in zip(rel, rel[1:])) or "-"
    tb, ta, inb = r.get("truth_before", {}), r.get("truth_after", {}), r.get("in_basket_after", {})
    moved = "".join(f"<tr><td>{n}</td><td>{'in basket' if inb.get(n) else 'on table'}</td></tr>" for n in ta if ta[n] != tb.get(n))
    html = f"""<!doctype html><meta charset=utf-8><title>detect_pick (video mode)</title>
<style>body{{font:14px system-ui;max-width:1000px;margin:24px auto;padding:0 16px}}td,th{{border-bottom:1px solid #ddd;padding:6px;text-align:left}}video{{max-width:100%}}</style>
<h2>detect_pick({r.get('target')!r}) -- video mode</h2>
<p>Scan {ev.get('scan_done', 0):.1f} s, total {r.get('total_sim_s')} sim-s. Seconds from arriving at an object to releasing it: see table.
Seconds from one release to the next: {gaps}.</p>
<video src="wrist.mp4" controls autoplay loop muted></video>
<table><tr><th>pick</th><th>class</th><th>conf</th><th>holding at the basket</th><th>arrive to release (s)</th></tr>{rows}</table>
<h3>Blocks that moved (stage truth, report only)</h3><table>{moved}</table>
<p>{r.get('error') or ''}</p>"""
    open(os.path.join(run, "index.html"), "w", encoding="utf-8").write(html)


def error_scoring_rec(pk):
    """11/11 corner 1a (report only, never a decision): on the error path the scoring (check_acceptance) needs every part vs its own class's bin
    and the start / end positions, as the normal path's in_own_bin_after / truth_before / truth_after."""
    out = {}
    if hasattr(pk, "own_bin_box"):
        try:
            ta_ = pk.truth(); out["truth_after"] = ta_; out["in_own_bin_after"] = own_bin_after(ta_, pk.own_bin_box)
            if getattr(pk, "_truth0_rec", None) is not None:
                out["truth_before"] = pk._truth0_rec
        except Exception as e_:
            out["in_own_bin_after_error"] = repr(e_)[:120]
    return out


def start_live(target="sphere", max_picks=12, rec=True, debug_release=False, test=None):
    builtins._lab_live_task = asyncio.ensure_future(LivePicker().run_live(target, max_picks, rec, debug_release, test))
