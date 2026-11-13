"""lab/live_perception.py -- video mode: the wrist D455 stays on, every rendered frame goes through YOLO + depth, and
object class / confidence / world position are kept up to date in a tracker while the arm moves.

Pieces
  LiveCam    render product always on; annotators rgb + depth + camera_params (the camera pose of THE RENDERED frame,
             so a frame that arrives 1-2 app frames late is still converted with the right pose)
  worker     (thread) PNG -> /dev/shm, YOLO (lab/yolo_live.py, port 18778), depth -> world, per-object estimate
  Tracker    world-frame tracks (class vote, running median of the centre), poll with live.tracks()
Use:  live = Live(oc); await live.start(); ... live.tracks("sphere") ...; await live.stop()
"""
import asyncio, builtins, concurrent.futures, json, math, os, socket, subprocess, sys, time
import numpy as np

def _project_root():
    """Project root: parent of lab/ when loaded with __file__ set, else parent of the folder holding the open scene."""
    f = globals().get("__file__")
    if f:
        return os.path.dirname(os.path.dirname(os.path.abspath(f)))
    import omni.usd
    return os.path.dirname(os.path.dirname(os.path.abspath(omni.usd.get_context().get_stage().GetRootLayer().realPath)))


ROOT = _project_root()
LIVE_PORT = 18778
SHM = "/dev/shm/nhc_live"


def _req(obj, timeout=10):
    s = socket.create_connection(("127.0.0.1", LIVE_PORT), timeout=3); s.settimeout(timeout)
    try:
        d = json.dumps(obj).encode(); s.sendall(len(d).to_bytes(4, "big") + d)
        head = b""
        while len(head) < 4:
            head += s.recv(4 - len(head))
        n = int.from_bytes(head, "big"); buf = bytearray()
        while len(buf) < n:
            buf.extend(s.recv(n - len(buf)))
        return json.loads(buf)
    finally:
        s.close()


def _steps_per_frame():
    """Same as lab/fast_pick.physics_steps_per_frame (sim frames of 1/60 s per app frame)."""
    import carb, omni.timeline
    tc = omni.timeline.get_timeline_interface().get_time_codes_per_seconds()
    mfr = carb.settings.get_settings().get("/persistent/simulation/minFrameRate") or 60
    return max(1, int(round(60.0 * min(1.0 / tc, 1.0 / mfr))))


def box_class(d):
    """cube vs cuboid from the depth dimensions (independent of YOLO). Catalog: a cube has height = both top edges (40/50/60 mm); every cuboid is 60 mm long and lies on its
    30 / 40 / 20 mm side, so its height is never its long top edge. Uses the long top edge and the height only: a neighbour touching the block can clip the SHORT
    top edge (a 60 mm cube measured 60 x 46.6), but not the height. Measured with depth: top edges and heights within 1 mm of the stage truth (class_eval_c1.json)."""
    top, h = d.get("top_face_size_mm"), d.get("height_mm")
    if not top or not h:
        return None
    long_, short_ = float(top[0]), float(top[1])
    if long_ <= 0 or h <= 0:
        return None
    return "cube" if abs(h - long_) <= 0.12 * long_ and short_ >= 0.55 * long_ else "cuboid"


def _min_rect(xy):
    import cv2
    m = xy.mean(0); (cx, cy), (w, h), ang = cv2.minAreaRect(((xy - m) * 1000.0).astype(np.float32))
    L, S, yl = (w, h, ang) if w >= h else (h, w, ang + 90.0)
    return m + np.array([cx, cy]) / 1000.0, L / 1000.0, S / 1000.0, ((yl + 90.0) % 180.0) - 90.0


def footprint_diag(pts, floor_z, above_m=0.010, pct=(2.0, 98.0), edge_m=0.003):
    """11/10 stage 4a, RECORD ONLY (never read by a decision): the footprint of geometry.estimate (minAreaRect of every valid mask point) next to
    two variants -- only points higher than floor_z + above_m; only points inside the pct percentiles along the all-points rectangle's two axes --
    and, for each side of the all-points rectangle, the points within edge_m of it: how many, their z (min / median / max above the floor) and
    how many of them are lower than floor_z + above_m."""
    pts = np.asarray(pts, float)
    if len(pts) < 30:
        return dict(n=int(len(pts)), note="too few points")
    out = dict(n=int(len(pts)), floor_z=round(float(floor_z), 4))
    c, L, S, yl = _min_rect(pts[:, :2])
    rnd = lambda c_, L_, S_, y_, n_: dict(size_mm=[round(1000 * L_, 1), round(1000 * S_, 1)], yaw_deg=round(float(y_), 2), centre_xy=[round(float(c_[0]), 5), round(float(c_[1]), 5)], n=int(n_))
    out["all"] = rnd(c, L, S, yl, len(pts))
    hi = pts[pts[:, 2] > floor_z + above_m]
    out["above_floor"] = rnd(*_min_rect(hi[:, :2]), len(hi)) if len(hi) >= 30 else dict(n=int(len(hi)), note="too few points")
    a = math.radians(yl); u = np.array([math.cos(a), math.sin(a)]); v = np.array([-u[1], u[0]])
    pu, pv = (pts[:, :2] - c) @ u, (pts[:, :2] - c) @ v
    keep = (pu >= np.percentile(pu, pct[0])) & (pu <= np.percentile(pu, pct[1])) & (pv >= np.percentile(pv, pct[0])) & (pv <= np.percentile(pv, pct[1]))
    out["pct"] = rnd(*_min_rect(pts[keep, :2]), int(keep.sum()))
    edges = {}
    for name, d_, half in (("long+", pv, S / 2), ("long-", -pv, S / 2), ("short+", pu, L / 2), ("short-", -pu, L / 2)):
        sel = d_ >= half - edge_m; z = pts[sel, 2] - floor_z
        edges[name] = dict(n=int(sel.sum()), z_mm=[round(1000 * float(z.min()), 1), round(1000 * float(np.median(z)), 1), round(1000 * float(z.max()), 1)] if sel.any() else None,
                           n_low=int((z < above_m).sum()))
    out["edges"] = edges
    return out


def detection_gate(cam_z, table_z, speed, paused, view_h=(0.40, 0.62), max_speed=0.05):
    """11/10 G1 U1a (pure): THE rule for using a YOLO frame (the tracker, and through it choose / retarget / the chain neighbour check):
    the camera 0.40-0.62 m above the table (views like the training data), (nearly) still, and the tracker not paused. Frames that fail are
    still logged (obs_log) and drawn grey on the HUD, never used."""
    h = float(cam_z) - float(table_z)
    return bool((not paused) and view_h[0] <= h <= view_h[1] and float(speed) <= max_speed)


def track_status(center_xy, obs_xy, tol_m=0.005, gate_m=0.03):
    """11/10 speed B (pure): an existing track vs the observations of the same class taken since the arm arrived at view A.
    'same' = the nearest new observation is within tol_m of the track centre; 'moved' = nearest within gate_m but farther than tol_m
    (the part was bumped: the tracker would ignore it as a jump, so the track must be rebuilt); 'missing' = nothing within gate_m."""
    if not len(obs_xy):
        return "missing"
    d = min(math.dist(center_xy, o) for o in obs_xy)
    return "same" if d <= tol_m else ("moved" if d <= gate_m else "missing")


def confirm_decision(statuses, n_good_frames, k):
    """11/10 speed B (pure): go without a new scan only when k still frames were processed since the arrival and every track is 'same';
    'wait' = not enough frames yet; 'rescan' = some track moved / missing (drop those, the normal scan rebuilds them)."""
    if n_good_frames < k:
        return "wait"
    return "go" if all(v == "same" for v in statuses.values()) else "rescan"


def footprint_floor_filter(pts, floor_z, below_m=0.003, min_pts=30):
    """11/10 G4 U1b (pure): the view-A footprint (minAreaRect as perception/geometry.estimate) from the mask points NOT lower than floor_z - below_m.
    The extra footprint width came from a few mask pixels whose depth is the table seen through the basket lattice, ~9.5 mm under the floor
    (stage 4b npz). Returns the footprint fields to replace (None if too few points remain)."""
    pts = np.asarray(pts, float); keep = pts[:, 2] >= floor_z - below_m
    if int(keep.sum()) < min_pts:
        return None
    c, L, S, yl = _min_rect(pts[keep, :2])
    return dict(footprint_size_mm=[round(1000 * L, 1), round(1000 * S, 1)], footprint_yaw_deg=round(float(yl), 2),
                footprint_center_xy_m=[round(float(c[0]), 5), round(float(c[1]), 5)], footprint_dropped_pts=int((~keep).sum()))


def release_gap_m(planned_mm, closed_mm, margin_m, full_m):
    """11/10. THE release opening (pad gap, m): max(planned grasp width, width measured after the close) + margin, at most fully open.
    The place-spot check (finger boxes) and the release command both take this one value (closed_mm None = before the close)."""
    w = max(float(planned_mm), float(closed_mm) if closed_mm is not None else 0.0) / 1000.0 + float(margin_m)
    return min(float(full_m), w)


class BasketMap:
    """Top-view height map of the basket floor, filled from the wrist depth while the hand carries a block over the basket (Live._process calls update()).
    The basket floor is a lattice and the table shows through it, so the height of a bin is the highest point seen in it above the lattice (floor_z).
    A spot is free when the block's footprint AND the two open fingers (they go down beside the block) touch no occupied bin, no basket wall and no earlier
    placement of this run (stamps). Bins never seen count as free (the spot is verified again in flight, when the camera sees it)."""
    RES = 0.01

    def __init__(self, bmin, bmax, floor_z, rim_z, centre_margin=0.06, wall_m=0.03, occ_h=0.012, known_empty=False, res=None):
        import threading
        if res:
            self.RES = float(res)                                # 11/09 option: a finer grid (the default 10 mm made the 3 mm gap tier a 10 mm one)
        self.x0, self.y0 = float(bmin[0]), float(bmin[1]); R = self.RES
        self.nx = int(math.ceil((bmax[0] - bmin[0]) / R)); self.ny = int(math.ceil((bmax[1] - bmin[1]) / R))
        self.top = np.full((self.nx, self.ny), 0.0 if known_empty else np.nan, np.float32); self.seen_n = 0       # known_empty: layout() just emptied the basket, nothing to look at
        self.floor_z, self.rim_z, self.cm, self.wall_m, self.occ_h = floor_z, rim_z, centre_margin, wall_m, occ_h
        self.stamps = []; self.lock = threading.Lock(); self.fresh = 0

    def update(self, P, ok):
        R = self.RES
        sel = ok & (P[..., 0] >= self.x0) & (P[..., 0] < self.x0 + self.nx * R) & (P[..., 1] >= self.y0) & (P[..., 1] < self.y0 + self.ny * R) & (P[..., 2] < self.rim_z - 0.004)
        if int(sel.sum()) < 3000:
            return                                                          # the camera does not look into the basket in this frame
        pts = P[sel]; ix = ((pts[:, 0] - self.x0) / R).astype(int); iy = ((pts[:, 1] - self.y0) / R).astype(int)
        cur = np.full((self.nx, self.ny), -9.0, np.float32); np.maximum.at(cur, (ix, iy), pts[:, 2].astype(np.float32))
        seen = cur > -8
        with self.lock:
            self.top[seen] = cur[seen] - self.floor_z; self.fresh += 1
            if self.stamps:                                        # a remembered placement the depth now sees EMPTY (block taken out / basket emptied) is forgotten
                import cv2
                keep = []
                for x, y, r in self.stamps:
                    m_ = np.zeros((self.nx, self.ny), np.uint8); cv2.circle(m_, (int((y - self.y0) / R), int((x - self.x0) / R)), max(1, int(round(r / R))), 1, -1); m_ = m_ > 0
                    if m_.any() and seen[m_].mean() >= 0.8 and float(np.nan_to_num(self.top[m_], nan=0.0).max()) < self.occ_h:
                        continue
                    keep.append((x, y, r))
                self.stamps = keep

    def _grids(self):
        import cv2
        R = self.RES
        with self.lock:
            top = self.top.copy()
        occ = np.nan_to_num(top, nan=0.0) > self.occ_h
        for x, y, r in self.stamps:
            m_ = occ.astype(np.uint8); cv2.circle(m_, (int((y - self.y0) / R), int((x - self.x0) / R)), int(round(r / R)), 1, -1); occ = m_ > 0
        if getattr(self, "outlines", None):                       # 11/09: placements remembered as their outline (rectangle), see add_outline()
            m_ = occ.astype(np.uint8)
            for poly in self.outlines:
                cv2.fillConvexPoly(m_, np.array([self._px(c) for c in poly], np.int32), 1)
            occ = m_ > 0
        wall = np.zeros_like(occ); w = int(round(self.wall_m / R))
        wall[:w, :] = wall[-w:, :] = True; wall[:, :w] = wall[:, -w:] = True
        return occ, wall, ~np.isnan(top)

    def _px(self, p):
        R = self.RES
        return (int(round((p[1] - self.y0) / R)), int(round((p[0] - self.x0) / R)))      # cv2 wants (col, row) = (y bin, x bin)

    @staticmethod
    def outline_poly(xy, yaw_deg, ol, grow=0.0):
        """11/09. Corners of the held object's top-view footprint when the TCP is at xy with closing axis yaw_deg.
        ol = dict(size=(long, short) m, yaw0=long-side direction at the grasp (deg), grasp_yaw=closing axis at the grasp (deg),
                  off=footprint centre - TCP at the grasp (world xy, m)). The object turns with the hand by (yaw_deg - grasp_yaw)."""
        d = math.radians(yaw_deg - ol["grasp_yaw"]); cd, sd = math.cos(d), math.sin(d)
        c = np.asarray(xy, float) + np.array([cd * ol["off"][0] - sd * ol["off"][1], sd * ol["off"][0] + cd * ol["off"][1]])
        a = math.radians(ol["yaw0"]) + d; u = np.array([math.cos(a), math.sin(a)]); v = np.array([-u[1], u[0]])
        L, S = ol["size"][0] / 2 + grow, ol["size"][1] / 2 + grow
        return [c + sx * L * u + sy * S * v for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1))]

    def add_outline(self, xy, yaw_deg, ol, grow=0.005):
        """Remember a placement as its outline (+5 mm, like the disc stamps' r + 5 mm)."""
        if not hasattr(self, "outlines"):
            self.outlines = []
        self.outlines.append(self.outline_poly(xy, yaw_deg, ol, grow))

    def _mask(self, xy, r, yaw_deg, gap_m, planner, outline=None):
        """Bins covered by the block (disc r, or its outline when given) and the two open fingers at xy with closing axis yaw_deg."""
        import cv2
        R = self.RES; m = np.zeros((self.nx, self.ny), np.uint8)
        px = self._px
        if outline is None:
            cv2.circle(m, px(xy), max(1, int(round(r / R))), 1, -1)
        else:
            cv2.fillConvexPoly(m, np.array([px(c) for c in self.outline_poly(xy, yaw_deg, outline)], np.int32), 1)
        for b in planner.finger_boxes(np.asarray(xy, float), yaw_deg, 0.004, gap_m):
            cv2.fillConvexPoly(m, np.array([px(c) for c in b], np.int32), 1)
        return m > 0

    def evaluate(self, xy, r, yaw_deg, gap_m, planner, gap=0.008, outline=None):
        """(free, verified): free = nothing occupied within `gap` of the block+fingers; verified = most of that area was actually seen."""
        import cv2
        occ, wall, seen = self._grids(); k = max(1, int(round(gap / self.RES)))
        bad = cv2.dilate((occ | wall).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))) > 0
        m = self._mask(xy, r, yaw_deg, gap_m, planner, outline)
        return (not bool((bad & m).any())), bool(seen[m].mean() >= 0.7)

    def pick_spot(self, r, yaw_deg, gap_m, planner, base_xy, near=None, outline=None, rank="base", yaw_offsets=(0.0,), yaw_ok=None, gap_tiers=(0.015, 0.008, 0.003)):
        """Free spot for a block of radius r held with closing axis yaw_deg: nearest to `near` (in flight: smallest change) else to the arm base. Gaps 15 / 8 / 3 mm.
        11/09 options (defaults = the behaviour before): outline = the object's footprint instead of the disc r; rank="clearance" = of the free spots the one
        farthest from anything occupied (spreads the parts over the bin; ties, e.g. an empty bin -> the usual order); yaw_offsets = closing-axis turns to try,
        in order, when no spot is free at yaw_deg (the object turns with the hand); rank="rows" = the first free spot in row order from the corner nearest
        the arm base (with P["place_align_deg"] every part is turned to the same direction first). The result also says which yaw was used."""
        import cv2
        occ, wall, seen = self._grids(); R = self.RES
        step = 0.02 if outline is None else 0.01            # 11/09: with an outline every 10 mm (at 20 mm a 2nd bracket_mid + open fingers missed the room by 16-40 mm offline)
        xs = np.arange(self.x0 + self.cm, self.x0 + self.nx * R - self.cm, step); ys = np.arange(self.y0 + self.cm, self.y0 + self.ny * R - self.cm, step)
        ref = near if near is not None else base_xy
        # clearance = distance to the nearest OCCUPIED cell (walls are only a limit, not something to stay away from: an empty bin then fills from the
        # usual corner, and every further part goes as far as possible from the ones already there -- a centred first part blocked a 2nd bracket_mid offline)
        clear = (cv2.distanceTransform((~occ).astype(np.uint8), cv2.DIST_L2, 5) * R if occ.any() else np.full(occ.shape, 9.0, np.float32)) if rank == "clearance" else None
        for off in yaw_offsets:
            yaw = yaw_deg + off
            if yaw_ok is not None and not yaw_ok(yaw):            # 11/09: a yaw the wrist cannot take -> its equivalent +-180 (symmetric gripper), else skip
                yaw = next((y_ for y_ in (yaw + 180.0, yaw - 180.0) if yaw_ok(y_)), None)
                if yaw is None:
                    continue
            if np.isnan(self.top).any():                              # parts never seen: prefer spots the video has seen (an unseen spot may hold something)
                cost = lambda c, yaw=yaw: math.dist(c, ref) + (0.0 if seen[self._mask(c, r, yaw, gap_m, planner, outline)].mean() >= 0.7 else 1.0)
            else:
                cost = lambda c: math.dist(c, ref)
            if rank == "rows":                                    # 11/09 Lin: fill in rows from the corner nearest the arm base: along y first, then the next row in x
                xs0 = xs[0] if abs(xs[0] - base_xy[0]) <= abs(xs[-1] - base_xy[0]) else xs[-1]
                ys0 = ys[0] if abs(ys[0] - base_xy[1]) <= abs(ys[-1] - base_xy[1]) else ys[-1]
                cost = lambda c: (round(abs(c[0] - xs0) / step), abs(c[1] - ys0))
            cands = sorted(((x, y) for x in xs for y in ys), key=cost)
            masks = {}                                            # each candidate's mask is drawn once and reused by the three gap tiers
            for gap in gap_tiers:
                k = max(1, int(round(gap / R)))
                bad = cv2.dilate((occ | wall).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))) > 0
                best = None
                for c in cands:
                    ci, cj = int((c[0] - self.x0) / R), int((c[1] - self.y0) / R)
                    if 0 <= ci < self.nx and 0 <= cj < self.ny and bad[ci, cj]:
                        continue                                  # the centre itself is taken: no need to draw the mask
                    m = masks.get(c)
                    if m is None:
                        m = masks[c] = self._mask(c, r, yaw, gap_m, planner, outline)
                    if not (bad & m).any():
                        if clear is None:
                            best = (c, None); break
                        cl = float(clear[m].min())
                        if best is None or cl > best[1] + 1e-9:
                            best = (c, cl)
                if best is not None:
                    c = best[0]
                    return dict(xy=[round(float(c[0]), 4), round(float(c[1]), 4)], gap_mm=round(1000 * gap, 1), n_stamps=len(self.stamps) + len(getattr(self, "outlines", [])),
                                yaw_deg=round(float(yaw), 2), yaw_offset_deg=float(off), clearance_mm=None if best[1] is None else round(1000 * best[1], 1))
        return None

    def snapshot(self, path):
        """11/10 G0 (record only): the map as the place decision saw it -> npz (top heights, depth-occupied, outline and stamp cells, wall band,
        seen cells, outline corners, stamps, box geometry). Returns the free ratio at that moment."""
        occ, wall, seen = self._grids()
        with self.lock:
            top = self.top.copy()
        np.savez_compressed(path, top=top.astype(np.float32), occ=occ, wall=wall, seen=seen, depth_occ=np.nan_to_num(top, nan=0.0) > self.occ_h,
                            outlines=np.asarray(getattr(self, "outlines", []) or np.zeros((0, 4, 2)), np.float32),
                            stamps=np.asarray(self.stamps or np.zeros((0, 3)), np.float32),
                            geom=np.array([self.x0, self.y0, self.RES, self.nx, self.ny, self.floor_z, self.rim_z, self.wall_m, self.occ_h], np.float64))
        return self.free_ratio()

    def free_ratio(self):
        """11/10. Share of the bin floor inside the wall band that is not occupied (depth, stamps, outlines)."""
        occ, wall, _ = self._grids(); inner = ~wall
        return round(float((~occ & inner).sum()) / max(1, int(inner.sum())), 3)

    def emptiest_spot(self):
        """11/10. The point farthest from anything occupied AND from the wall band (distance transform, as rank="clearance")."""
        import cv2
        occ, wall, _ = self._grids(); R = self.RES
        d = cv2.distanceTransform((~(occ | wall)).astype(np.uint8), cv2.DIST_L2, 5)
        i, j = np.unravel_index(int(np.argmax(d)), d.shape)
        return [round(self.x0 + (i + 0.5) * R, 4), round(self.y0 + (j + 0.5) * R, 4)], round(1000 * float(d.max()) * R, 1)

    def spot_or_fallback(self, r, yaw_deg, gap_m, min_gap_m, planner, base_xy, outline=None, rank="base", yaw_offsets=(0.0,), yaw_ok=None,
                         gap_tiers=(0.015, 0.008, 0.003)):
        """11/10. pick_spot(); when it finds nothing:
          1. full scan at the ACTUAL opening gap_m: every yaw 0-175 (every 5; pick_spot tries the +-180 equivalent when the wrist rejects one), smallest
             gap tier, wrist check kept -> a spot = used (via="full_scan", not a fallback; supervisor review 11/10)
          2. else the emptiest point of the map (no more fixed fallback cells), recorded as no_spot; box_full only if even the smallest release opening
             (min_gap_m), the smallest gap tier and every yaw with the wrist ignored find nothing. found_but_wrist = that last scan found a spot that the
             actual opening or the wrist does not allow. The fallback yaw = the first yaw_offsets yaw the wrist can take (else the grasp yaw)."""
        s = self.pick_spot(r, yaw_deg, gap_m, planner, base_xy, outline=outline, rank=rank, yaw_offsets=yaw_offsets, yaw_ok=yaw_ok, gap_tiers=gap_tiers)
        if s is not None:
            s.update(fallback=False, no_spot=False, box_full=False); return s
        t0 = time.time(); every = tuple(a - yaw_deg for a in range(0, 180, 5)); tier = (min(gap_tiers),)
        act = self.pick_spot(r, yaw_deg, gap_m, planner, base_xy, outline=outline, rank="base", yaw_offsets=every, yaw_ok=yaw_ok, gap_tiers=tier)
        if act is not None:
            act.update(fallback=False, no_spot=False, box_full=False, via="full_scan", free_ratio=self.free_ratio(),
                       full_scan=dict(gap_mm=round(1000 * gap_m, 1), ms=round(1000 * (time.time() - t0), 1)))
            return act
        full = self.pick_spot(r, yaw_deg, min_gap_m, planner, base_xy, outline=outline, rank="base", yaw_offsets=every, yaw_ok=None, gap_tiers=tier)
        xy, cl = self.emptiest_spot()
        yaw = yaw_deg
        if yaw_ok is not None:
            for off in yaw_offsets:
                y_ = next((v for v in (yaw_deg + off, yaw_deg + off + 180.0, yaw_deg + off - 180.0) if yaw_ok(v)), None)
                if y_ is not None:
                    yaw = y_; break
        return dict(xy=xy, gap_mm=None, n_stamps=len(self.stamps) + len(getattr(self, "outlines", [])), yaw_deg=round(float(yaw), 2),
                    yaw_offset_deg=round(float(yaw - yaw_deg), 2), fallback=True, no_spot=True, box_full=full is None, found_but_wrist=full is not None,
                    free_ratio=self.free_ratio(), clearance_mm=cl,
                    box_full_check=dict(gap_mm=round(1000 * gap_m, 1), min_gap_mm=round(1000 * min_gap_m, 1),
                                        found=None if full is None else dict(xy=full["xy"], yaw_deg=full["yaw_deg"]), ms=round(1000 * (time.time() - t0), 1)))


class Tracker:
    def __init__(self, gate_m=0.03, keep=12):
        self.tr = []; self.gate = gate_m; self.keep = keep; self.next_id = 1

    def update(self, obs, t, cam_pos):
        for o in obs:
            c = np.asarray(o["center_xyz_m"], float)
            best, bd = None, self.gate
            for tk in self.tr:
                d = float(np.linalg.norm(tk["center"][:2] - c[:2]))
                if d < bd:
                    best, bd = tk, d
            if best is not None and best["n"] >= 3 and bd > 0.012:       # established track: ignore a jump (partial view / blur)
                continue
            if best is None:
                best = dict(id=self.next_id, hist=[], votes={}, first_t=t, picked=False); self.next_id += 1; self.tr.append(best)
            best["hist"].append(c); best["hist"] = best["hist"][-self.keep:]
            best["center"] = np.median(np.array(best["hist"]), 0)
            best["votes"][o["class_name"]] = best["votes"].get(o["class_name"], 0.0) + o["conf"]
            best["class_name"] = max(best["votes"], key=best["votes"].get)
            best["conf"] = o["conf"]; best["n"] = len(best["hist"]); best["last_t"] = t; best["obs"] = o
            best["cam_dist_m"] = float(np.linalg.norm(c - np.asarray(cam_pos)))

    def snapshot(self, cls=None, min_n=1):
        return [tk for tk in self.tr if not tk["picked"] and tk["n"] >= min_n and (cls is None or tk["class_name"] == cls)]


class Live:
    def __init__(self, oc):
        self.oc = oc; self.geo = None; self.tracker = Tracker(); self.running = False
        self.n_frames = 0; self.n_proc = 0; self.lat_ms = []; self.fps_log = []
        self.pool = concurrent.futures.ThreadPoolExecutor(1); self.job = None
        self.cfg = oc["CFG"]; self.log = []; self.sim_f = 0; self.errors = []
        self.spf = _steps_per_frame()       # sim frames (1/60 s) per app frame; 1 with the default timeline
        self.blobs_latest = None; self.cam_hist = []; self.view_h = (0.40, 0.62); self.max_speed = 0.05; self.min_conf = 0.5; self.rescue_fg = 0.6; self.basket_map = None; self.basket_on = False; self.basket_gate = None; self.obs_log = []; self.proc_log = []; self.paused = False; self.min_cam_h = 0.40; self.rec_dir = None; self.rec_n = 0; self.hud = ""

    async def start(self):
        import omni.kit.app, omni.replicator.core as rep, omni.usd
        os.makedirs(SHM, exist_ok=True)
        try:
            r = _req({"cmd": "ping"}, 3)
            if r.get("src_mtime") != os.path.getmtime(os.path.join(ROOT, "lab", "yolo_live.py")):
                _req({"cmd": "quit"}, 3); raise RuntimeError("restart")
        except Exception:
            py = os.path.join(ROOT, self.cfg["yolo_python"]) if not os.path.isabs(self.cfg["yolo_python"]) else self.cfg["yolo_python"]
            subprocess.Popen([py, os.path.join(ROOT, "lab", "yolo_live.py")], cwd=ROOT, stdout=open(os.path.join(ROOT, "outputs", "lab", "yolo_live.log"), "a"),
                             stderr=subprocess.STDOUT, start_new_session=True)
            for _ in range(60):
                await asyncio.sleep(1)
                try:
                    _req({"cmd": "ping"}, 3); break
                except Exception:
                    pass
        gp = os.path.join(ROOT, "perception", "geometry.py"); g = {"__file__": gp}
        exec(compile(open(gp, encoding="utf-8").read(), gp, "exec"), g); self.geo = type("G", (), g)
        self.ns = self.oc["_load_arm_stack"](); await self.ns["_setup"]()
        reg = self.ns["_registry"](); self.rp = reg["render_product"]; self.rgb_ann = reg["rgb"]; self.depth_ann = reg["depth"]
        self.cp = rep.AnnotatorRegistry.get_annotator("camera_params"); self.cp.attach(self.rp)
        self.K = self.ns["camera_intrinsics"](omni.usd.get_context().get_stage(), self.ns["CAMERA_PATHS"]["left"])
        self.ns["_enable"](self.rp, True)
        for _ in range(8): await omni.kit.app.get_app().next_update_async()
        self.running = True; self.prev_pos = None; self.t0 = time.time()
        self.task = asyncio.ensure_future(self._loop())

    async def stop(self):
        self.running = False
        try:
            await self.task
        except Exception:
            pass
        self.ns["_enable"](self.rp, False)

    # ------------------------------------------------------------ main-thread loop: grab frames, hand to worker
    async def _loop(self):
        import omni.kit.app
        app = omni.kit.app.get_app()
        while self.running:
            await app.next_update_async(); self.sim_f += self.spf
            try:
                self._poll_job()
                d = self.cp.get_data(); v = np.asarray(d["cameraViewTransform"], float).reshape(4, 4)
                inv = np.linalg.inv(v); pos = inv[3, :3]
                if self.prev_pos is None or not np.allclose(pos, self.prev_pos, atol=1e-7):
                    self.n_frames += 1                          # camera moved = a new rendered frame (a still camera gives the same pose)
                self.prev_pos = pos
                if self.job is not None:
                    continue                                    # worker busy: drop this frame (always process the newest)
                r = np.asarray(self.rgb_ann.get_data()); dp = np.asarray(self.depth_ann.get_data(), dtype=np.float32)
                if r.ndim != 3 or dp.ndim != 2 or r[:, :, :3].mean() < 1.0:
                    continue
                cam = dict(position_m=pos.tolist(), right=inv[0, :3].tolist(), up=inv[1, :3].tolist(), forward=(-inv[2, :3]).tolist())
                if self.basket_on and self.basket_map is not None and (self.sim_f // self.spf) % 2 == 0 and (self.basket_gate is None or self.basket_gate()):
                    try:                                         # basket height map from every 2nd depth frame while the hand is high near the basket (no YOLO needed, the arm keeps moving)
                        self.basket_map.update(self.geo.world_points(dp, cam, self.K, self.cfg.get("pixel_center_offset", 0.5)), np.isfinite(dp) & (dp > 0) & (dp < 5))
                    except Exception as e_:
                        self.errors.append("basket_map: " + repr(e_))
                self.cam_hist.append((self.sim_f, pos.copy())); self.cam_hist = self.cam_hist[-40:]
                older = [(f, p) for f, p in self.cam_hist if self.sim_f - f >= 6]
                cam["speed"] = float(np.linalg.norm(pos - older[-1][1]) / ((self.sim_f - older[-1][0]) / 60.0)) if older else 9.9
                self.job = (self.pool.submit(self._process, r[:, :, :3].astype(np.uint8).copy(), dp.copy(), cam), self.sim_f / 60.0, time.time(), cam)
            except Exception as e:
                self.errors.append(repr(e))

    def _poll_job(self):
        if self.job is None or not self.job[0].done():
            return
        fut, t_sim, t_wall, cam = self.job; self.job = None
        try:
            obs, info = fut.result()
        except Exception as e:
            self.errors.append("worker: " + repr(e)); return
        self.n_proc += 1; self.lat_ms.append((time.time() - t_wall) * 1000)
        h_ = cam["position_m"][2] - self.cfg["table_z_m"]; sp_ = cam.get("speed", 9.9)
        self.proc_log.append((round(t_sim, 3), round(self.lat_ms[-1], 1), detection_gate(cam["position_m"][2], self.cfg["table_z_m"], sp_, self.paused, self.view_h, self.max_speed), len(obs)))   # 11/10: per processed frame (time, worker ms, used frame, n obs)
        for o in obs:
            self.obs_log.append((round(t_sim, 3), o["class_name"], round(o["conf"], 3), o["center_xyz_m"], round(cam["position_m"][2], 3), o.get("mask_pixels"), bool(self.paused), round(cam.get("speed", -1), 3),
                                 o.get("yolo_class"), o.get("top_face_size_mm"), o.get("height_mm"), o.get("fg_frac")))
        h_cam = cam["position_m"][2] - self.cfg["table_z_m"]
        good = detection_gate(cam["position_m"][2], self.cfg["table_z_m"], cam.get("speed", 9.9), self.paused, self.view_h, self.max_speed)   # 11/10 G1: one shared gate
                                                                                 # (views like the training data, a still camera: a moving one blurs the path-traced image)
        if good:
            self.tracker.update([o for o in obs if o["conf"] >= self.min_conf or o.get("fg_frac", 0.0) >= self.rescue_fg], t_sim, cam["position_m"])
        self.last_info = info; self.last_cam = cam
        self.blobs_latest = (t_sim, info.get("blobs", []), cam)

    # ------------------------------------------------------------ worker thread
    def _process(self, rgb, depth, cam):
        import cv2
        t = time.time(); geo = self.geo; cfg = self.cfg
        png = os.path.join(SHM, "f.png"); cv2.imwrite(png, rgb[:, :, ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 1])
        mpath = os.path.join(SHM, "m.npy")
        rep = _req({"cmd": "detect", "rgb_path": png, "masks_path": mpath, "conf": cfg["candidate_conf"]})
        if not rep.get("ok"):
            raise RuntimeError(rep.get("error"))
        t_yolo = time.time() - t
        masks_all = np.load(mpath).astype(bool)
        P = geo.world_points(depth, cam, self.K, cfg.get("pixel_center_offset", 0.5))
        mg = cfg.get("detection_region_margin_m", 0.0)
        wx = [cfg["workspace_x_m"][0] - mg, cfg["workspace_x_m"][1] + mg]; wy = [cfg["workspace_y_m"][0] - mg, cfg["workspace_y_m"][1] + mg]
        fg = geo.foreground(depth, P, cfg["table_z_m"], wx, wy)
        raw = [dict(cls=d["class_name"], conf=d["conf"], px=d["mask_pixels"], bbox=d["bbox_xyxy"]) for d in rep["detections"]]
        cand = rep["detections"]; finger = self._finger_mask(depth.shape)
        if finger is not None:                                  # the fingers are rigid with the camera: never an object; a mask mostly on them is dropped, the rest is cut off
            fg = fg & ~finger; keep_i = []
            for i, d in enumerate(cand):
                if i < len(masks_all) and masks_all[i].sum() and (masks_all[i] & finger).sum() > cfg.get("finger_mask_drop_frac", 0.2) * masks_all[i].sum():
                    raw[i]["reason"] = "dropped: on the finger region"
                else:
                    keep_i.append(i)
            cand = [cand[i] for i in keep_i]
            masks_all = masks_all[keep_i] & ~finger if len(masks_all) else masks_all
        dets, masks = geo.select_detections(cand, masks_all, fg, cfg["conf"])
        kept_boxes = {tuple(d["bbox_xyxy"]) for d in dets}
        for r_ in raw:
            if "reason" not in r_:
                r_["reason"] = "kept" if tuple(r_["bbox"]) in kept_boxes else "dropped: conf<%.2f and not rescued by depth" % cfg["conf"]
        union = masks.any(0) if len(masks) else None
        out = []; H, W = depth.shape
        for i, d in enumerate(dets):
            m = masks[i]
            ys, xs = np.nonzero(m)
            if len(xs) == 0 or xs.min() <= 1 or ys.min() <= 1 or xs.max() >= W - 2 or ys.max() >= H - 2:
                self._mark_raw(raw, d, "dropped: mask touches the image border")
                continue                                        # clipped by the image border -> centre would be biased
            others = (union & ~m) if union is not None else None
            d["yolo_class"] = d["class_name"]
            d.update(geo.estimate(m, depth, P, cam, d["class_name"], cfg["table_z_m"], exclude=others))
            if d.get("center_xyz_m") is None:
                self._mark_raw(raw, d, "dropped: too few valid depth points")
                continue
            d["fg_frac"] = round(float((m & fg).sum()) / max(1, int(m.sum())), 3)        # share of the mask that stands on the table in depth (a real object, whatever YOLO's confidence)
            if getattr(self, "floor_filter", None) is not None and d.get("footprint_size_mm") and \
                    (not getattr(self, "floor_filter_classes", None) or d.get("class_name") in self.floor_filter_classes):   # 11/10 G4 U1b (brackets only in sort)
                ff_ = footprint_floor_filter(P[m & np.isfinite(depth) & (depth > 0) & (depth < 5)], cfg["table_z_m"], self.floor_filter)
                if ff_ is not None:
                    d["footprint_raw"] = dict(size_mm=d["footprint_size_mm"], yaw_deg=d.get("footprint_yaw_deg"), centre_xy=d.get("footprint_center_xy_m"))
                    d.update(ff_)
            if getattr(self, "fp_diag", False) and cam.get("speed", 9.9) <= self.max_speed:      # 11/10 stage 4a, record only, still camera (view A) frames only:
                                                                   # footprint variants + the raw mask / points for the npz (~10 ms per detection, worker thread)
                t_d = time.time(); ok_ = np.isfinite(depth) & (depth > 0) & (depth < 5); ys_, xs_ = np.nonzero(m)
                y0_, x0_ = int(ys_.min()), int(xs_.min())
                d["fp_diag"] = footprint_diag(P[m & ok_], cfg["table_z_m"])
                d["_diag_arrays"] = dict(mask_crop=m[y0_:int(ys_.max()) + 1, x0_:int(xs_.max()) + 1].copy(), mask_origin_yx=np.array([y0_, x0_]),
                                         pts=P[m & ok_].astype(np.float32), floor_z=np.float32(cfg["table_z_m"]), cam_pos=np.asarray(cam["position_m"], np.float32))
                d["fp_diag"]["ms"] = round(1000 * (time.time() - t_d), 1)
            if d["yolo_class"] in ("cube", "cuboid"):
                sc = box_class(d)
                if sc:
                    d["class_name"] = sc
            c = d["center_xyz_m"]
            if wx[0] <= c[0] <= wx[1] and wy[0] <= c[1] <= wy[1]:
                out.append(d)
            else:
                self._mark_raw(raw, d, "dropped: centre outside the table workspace")
        self.last_raw = raw
        blobs = []                                            # class-free depth blobs standing on the table (usable close to the table where YOLO is not)
        try:
            m8 = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            nb, lab, stats, _ = cv2.connectedComponentsWithStats(m8, connectivity=8)
            for k_ in range(1, nb):
                if stats[k_, cv2.CC_STAT_AREA] < 80:
                    continue
                pts = P[lab == k_]
                blobs.append(dict(xy=[round(float(pts[:, 0].mean()), 4), round(float(pts[:, 1].mean()), 4)], top_z=round(float(np.percentile(pts[:, 2], 98)), 4),
                                  px=int(stats[k_, cv2.CC_STAT_AREA]), extent_mm=[round(1000 * float(np.ptp(pts[:, 0])), 1), round(1000 * float(np.ptp(pts[:, 1])), 1)]))
        except Exception as e_:
            self.errors.append("blobs: " + repr(e_))
        if self.rec_dir:
            try:
                self._record(rgb, masks, dets, out, cam)
            except Exception as e:
                self.errors.append("rec: " + repr(e))
        return out, dict(yolo_s=round(t_yolo, 3), total_s=round(time.time() - t, 3), n_det=len(out), predict_ms=rep.get("predict_ms"), blobs=blobs)

    def _finger_mask(self, shape):
        """Fixed image region of the gripper fingers (config finger_mask_png, white = finger), or None when not configured."""
        p = self.cfg.get("finger_mask_png")
        if not p:
            return None
        if getattr(self, "_finger", None) is None or self._finger.shape != tuple(shape):
            import cv2
            m = cv2.imread(p if os.path.isabs(p) else os.path.join(ROOT, p), cv2.IMREAD_GRAYSCALE)
            if m is None or m.shape != tuple(shape):
                self.errors.append(f"finger mask {p} missing or not {tuple(shape)}: not applied")
                self._finger = np.zeros(shape, bool)
            else:
                self._finger = m > 127
        return self._finger

    @staticmethod
    def _mark_raw(raw, d, why):
        for r_ in raw:
            if tuple(r_["bbox"]) == tuple(d["bbox_xyxy"]):
                r_["reason"] = why

    def _record(self, rgb, masks, dets, out, cam):
        """Annotated wrist frame: outline + #id on each object, and ONE list in the top-left corner
        (id, class, confidence, world x y z, distance from the camera) that changes as the arm moves."""
        import cv2
        img = np.ascontiguousarray(rgb[:, :, ::-1]); keep = {id(d) for d in out}
        tracks = list(self.tracker.tr); rows = []
        used = detection_gate(cam["position_m"][2], self.cfg["table_z_m"], cam.get("speed", 9.9), self.paused, self.view_h, self.max_speed)   # 11/10 G1: grey = not used
        for i, d in enumerate(dets):
            if i >= len(masks) or id(d) not in keep:
                continue
            col = [(66, 135, 245), (60, 60, 235), (80, 220, 90), (40, 200, 245), (200, 70, 230)][d.get("class_id", 0) % 5] if used else (150, 150, 150)
            cs, _ = cv2.findContours(masks[i].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cs, -1, col, 2 if used else 1)
            c = d["center_xyz_m"]
            near = [t for t in tracks if np.linalg.norm(t["center"][:2] - np.asarray(c[:2])) < 0.03]
            tid = near[0]["id"] if near else 0
            x0, y0 = int(d["bbox_xyxy"][0]), int(max(12, d["bbox_xyxy"][1]))
            cv2.putText(img, f"#{tid}", (x0, y0 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img, f"#{tid}", (x0, y0 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
            rows.append((tid, d["class_name"], d["conf"], c, d["distance_from_camera_m"], col))
        rows.sort(key=lambda r: r[0])
        lines = [((self.hud or "video") + ("" if used else "  [unused: camera not at view height / moving]"), (0, 255, 255) if used else (150, 150, 150))] + \
                [(f"#{t:<2d} {n:<9s} {cf:4.2f}  x{c[0]:7.3f} y{c[1]:7.3f} z{c[2]:6.3f}  d{dd:5.3f}m", col) for t, n, cf, c, dd, col in rows]
        h = 6 + 16 * len(lines)
        ov = img.copy(); cv2.rectangle(ov, (0, 0), (420, h), (0, 0, 0), -1); img = cv2.addWeighted(ov, 0.55, img, 0.45, 0)
        for k, (txt, col) in enumerate(lines):
            cv2.putText(img, txt, (5, 16 + 16 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
        self.rec_n += 1
        cv2.imwrite(os.path.join(self.rec_dir, f"f{self.rec_n:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])

    def blob_near(self, xy, radius=0.04, max_age_s=0.25):
        """Newest class-free blob within `radius` of xy: (blob, age_s), or (None, age_s) / (None, None)."""
        if not self.blobs_latest:
            return None, None
        t, bl, cam = self.blobs_latest; age = self.sim_f / 60.0 - t
        if age > max_age_s:
            return None, age
        best = None
        for b in bl:
            d = float(np.hypot(b["xy"][0] - xy[0], b["xy"][1] - xy[1]))
            if d <= radius and (best is None or d < best[0]):
                best = (d, b)
        return (best[1] if best else None), age

    def tracks(self, cls=None, min_n=1):
        return self.tracker.snapshot(cls, min_n)


async def selftest(tag="t1", seconds=6.0):
    """Arm moves along a sweep while video mode runs; report rates, latency, and track stability vs stage truth."""
    import omni.kit.app
    app = omni.kit.app.get_app(); out = {}
    try:
        oc = builtins._lab_oc; pose = oc["_pose"]
        live = Live(oc); await live.start()
        # go to view A (existing config) and let the arm settle while the stream runs
        va = oc["_view_poses"]()[0]
        p0, q0 = pose("/World/IKTarget")
        n = 90
        for i in range(1, n + 1):
            u = i / n; s = u * u * (3 - 2 * u)
            pose("/World/IKTarget", (np.asarray(p0) + s * (np.asarray(va["position_m"]) - np.asarray(p0))).tolist(), oc["_slerp"](q0, va["quaternion_wxyz"], s))
            await app.next_update_async()
        t0 = time.time(); f0 = live.sim_f; nf0, np0 = live.n_frames, live.n_proc
        for i in range(int(seconds * 60)):
            await app.next_update_async()
        dt = time.time() - t0
        tk = live.tracks()
        out = dict(wall_s=round(dt, 2), app_fps=round((live.sim_f - f0) / dt, 1), cam_frames_per_s=round((live.n_frames - nf0) / dt, 1),
                   processed_per_s=round((live.n_proc - np0) / dt, 1), latency_ms_mean=round(float(np.mean(live.lat_ms)), 0) if live.lat_ms else None,
                   latency_ms_p95=round(float(np.percentile(live.lat_ms, 95)), 0) if live.lat_ms else None, last_info=getattr(live, "last_info", None),
                   errors=live.errors[:5],
                   tracks=[dict(id=t["id"], cls=t["class_name"], conf=round(t["conf"], 2), n=t["n"], center=[round(float(v), 4) for v in t["center"]],
                                spread_mm=round(1000 * float(np.max(np.std(np.array(t["hist"]), 0))), 2)) for t in tk])
        await live.stop()
    except Exception:
        import traceback; out["error"] = traceback.format_exc()
    os.makedirs(os.path.join(ROOT, "outputs", "lab", "live"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "outputs", "lab", "live", f"selftest_{tag}.json"), "w"), indent=1, default=str)


def start_selftest(tag="t1", seconds=6.0):
    builtins._lab_self_task = asyncio.ensure_future(selftest(tag, seconds))
