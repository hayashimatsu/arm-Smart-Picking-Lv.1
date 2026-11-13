"""Grasp planning (pure python + numpy, no Kit): which object to pick next and how.

Inputs are the perception objects (world frame). Conventions:
  - yaw_deg = direction of the finger closing axis in the world XY plane (0 = world +X).
    The tool points straight down; quat = pose_math.quat_down_with_yaw(yaw_deg).
  - The TCP prim sits at the fingertip bottom (measured 2026-09-29: fingertips 0.6 mm above TCP).
User decisions 2026-09-29:
  - order by XY distance (world frame) from the arm base
  - try the natural yaw, then yaw + 90 deg; if both fail -> skip and report
  - grasp height = object center; if the fingertips would touch the table, raise the TCP
"""
import math
import numpy as np

# Robotiq 2F-85, measured from the stage at rest (open): inner-finger faces 87 mm apart,
# each finger 31 mm along the closing axis and 27 mm across it.
FINGER_GAP_OPEN_M = 0.085
FINGER_THICK_M = 0.031
FINGER_WIDTH_M = 0.027
MAX_OBJECT_WIDTH_M = 0.070        # object width across the fingers must be below this
CLEARANCE_M = 0.006               # extra margin around fingers when checking neighbours
FINGERTIP_TABLE_CLEARANCE_M = 0.004
FINGER_DESCENT_M = 0.013          # the Robotiq fingertips move DOWN while closing: 8 mm at 16 deg, 13 mm at >= 33 deg
                                  # (measured 2026-09-30). The TCP (open fingertip bottom) must stay this much higher,
                                  # else the pads land on the table while closing (cuboids 20-30 mm tall were dropped).
PRE_GRASP_ABOVE_M = 0.12          # pre-grasp TCP height above the tallest object top
PAD_CONTACT_M = 0.020             # fingertips at least this far below the object top (short objects slipped
                                  # when grasped at their center: 2026-09-29 cylinder 60x30 mm dropped twice)
OPEN_MARGIN_M = 0.020             # open only to object width + this (smaller finger footprint -> fewer blocked grasps)


# Pad gap (inner faces, mm) vs finger angle (deg), measured with the gripper closing in free air (2026-09-30).
GAP_CURVE = [(0.0, 87.1), (10.0, 70.8), (16.0, 60.2), (21.5, 50.0), (26.0, 41.4), (30.0, 33.7), (33.0, 27.7),
             (35.0, 23.8), (37.0, 19.8), (40.0, 13.8), (47.0, 0.0)]


def open_deg_for_gap(gap_m):
    """Finger angle giving an inner pad gap of gap_m (interpolated on GAP_CURVE)."""
    g = gap_m * 1000.0
    if g >= GAP_CURVE[0][1]:
        return 0.0
    for (a0, g0), (a1, g1) in zip(GAP_CURVE, GAP_CURVE[1:]):
        if g1 <= g <= g0:
            return round(a0 + (g0 - g) / (g0 - g1) * (a1 - a0), 2)
    return GAP_CURVE[-1][0]


def footprint(o):
    """2D footprint of a perceived object: ('rect', center, (long, short), yaw_deg) or ('circle', center, r)."""
    c = np.array(o["center_xyz_m"][:2])
    if o.get("class_name") == "sphere" or o.get("radius_mm"):
        r = (o.get("radius_mm") or o.get("height_mm", 40) / 2) / 1000.0
        return ("circle", c, r)
    size = o.get("top_face_size_mm")
    if size is None:
        return ("circle", c, 0.03)
    L, S = size[0] / 1000.0, size[1] / 1000.0
    if o.get("class_name") == "cylinder" and abs(L - S) < 0.006:      # upright cylinder
        return ("circle", c, L / 2)
    if o.get("class_name") == "cylinder" and L > 1.6 * S and o.get("footprint_size_mm"):
        # lying cylinder: the top band is only a thin strip -> use the full top-view footprint
        f = o["footprint_size_mm"]
        return ("rect", c, (f[0] / 1000.0, f[1] / 1000.0), o.get("footprint_yaw_deg", o.get("top_face_yaw_deg", 0.0)))
    return ("rect", c, (L, S), o.get("top_face_yaw_deg", 0.0))


def _rect_corners(c, size, yaw_deg):
    a = math.radians(yaw_deg); u = np.array([math.cos(a), math.sin(a)]); v = np.array([-u[1], u[0]])
    L, S = size
    return np.array([c + sx * L / 2 * u + sy * S / 2 * v for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1))])


def _sat_rect_rect(p, q):
    for poly in (p, q):
        for i in range(4):
            e = poly[(i + 1) % 4] - poly[i]; n = np.array([-e[1], e[0]])
            a, b = p @ n, q @ n
            if a.max() < b.min() or b.max() < a.min():
                return False
    return True


def _cross2(a, b):
    # 2-D cross product (np.cross rejects 2-D vectors in NumPy >= 2.5)
    return a[0] * b[1] - a[1] * b[0]


def _rect_circle(p, c, r):
    # distance from circle center to polygon (inside -> 0)
    inside = all(_cross2(p[(i + 1) % 4] - p[i], c - p[i]) >= 0 for i in range(4)) or \
        all(_cross2(p[(i + 1) % 4] - p[i], c - p[i]) <= 0 for i in range(4))
    if inside:
        return True
    d = min(np.linalg.norm(c - (p[i] + np.clip(np.dot(c - p[i], p[(i + 1) % 4] - p[i]) / np.dot(p[(i + 1) % 4] - p[i], p[(i + 1) % 4] - p[i]), 0, 1) * (p[(i + 1) % 4] - p[i])))
            for i in range(4))
    return d < r


def finger_boxes(center_xy, yaw_deg, margin=CLEARANCE_M, gap_m=FINGER_GAP_OPEN_M):
    """The two open-finger footprints (rectangles) for a grasp at center_xy with closing axis yaw_deg."""
    a = math.radians(yaw_deg); u = np.array([math.cos(a), math.sin(a)])
    off = gap_m / 2 + FINGER_THICK_M / 2
    size = (FINGER_THICK_M + 2 * margin, FINGER_WIDTH_M + 2 * margin)
    return [_rect_corners(np.asarray(center_xy) + s * off * u, size, yaw_deg) for s in (1, -1)]


def width_across(fp, yaw_deg):
    """Object width along the finger closing axis."""
    if fp[0] == "circle":
        return 2 * fp[2]
    _, c, (L, S), oy = fp
    d = math.radians(yaw_deg - oy)
    return abs(L * math.cos(d)) + abs(S * math.sin(d))


def blocked_by(boxes, others):
    hits = []
    for o in others:
        fp = footprint(o)
        for b in boxes:
            hit = _rect_circle(b, fp[1], fp[2]) if fp[0] == "circle" else _sat_rect_rect(b, _rect_corners(fp[1], fp[2], fp[3]))
            if hit:
                hits.append(o["id"]); break
    return hits


def natural_yaw(fp):
    """Close across the short side (rect) -> closing axis perpendicular to the long side."""
    if fp[0] == "circle":
        return 0.0
    return fp[3] + 90.0


def candidate_yaws(fp):
    """Yaws to try, in order of preference.
    circle (sphere / upright cylinder): any direction grips the same -> 0, 15, -15, 30 ... (15 deg steps).
    rect: across the short side, then +-15/+-30 around it, then across the long side (+90)."""
    if fp[0] == "circle":
        return [0.0] + [s * a for a in range(15, 91, 15) for s in (1, -1)][:-1]
    n = natural_yaw(fp)
    return [n, n + 15, n - 15, n + 30, n - 30, n + 90.0]


def plan_grasp(target, others, table_z):
    """Returns dict(ok, yaw_deg, tcp_grasp, tcp_pre, tried=[...], reason)."""
    fp = footprint(target)
    c = target["center_xyz_m"]
    tried = []
    for yaw in candidate_yaws(fp):
        yaw = ((yaw + 90) % 180) - 90                      # closing axis is symmetric: keep in (-90, 90]
        w = width_across(fp, yaw)
        gap = min(FINGER_GAP_OPEN_M, w + OPEN_MARGIN_M)
        hits = blocked_by(finger_boxes(c[:2], yaw, gap_m=gap), others)
        tried.append(dict(yaw_deg=round(yaw, 1), width_mm=round(w * 1000, 1), open_gap_mm=round(gap * 1000, 1), blocked_by=hits))
        if w <= MAX_OBJECT_WIDTH_M and not hits:
            top_z = target.get("top_z_m") or (c[2] * 2 - table_z)
            z = max(min(c[2], top_z - PAD_CONTACT_M), table_z + FINGER_DESCENT_M + FINGERTIP_TABLE_CLEARANCE_M)
            top = max([o.get("top_z_m") or (table_z + (o.get("height_mm") or 60) / 1000) for o in others + [target]])
            pre = max(z + 0.10, top + PRE_GRASP_ABOVE_M)
            return dict(ok=True, yaw_deg=round(yaw, 2), tcp_grasp=[c[0], c[1], z], tcp_pre=[c[0], c[1], pre],
                        open_deg=round(open_deg_for_gap(gap), 1), open_gap_mm=round(gap * 1000, 1),
                        raised_for_table_mm=round((z - c[2]) * 1000, 1), tried=tried)
    reason = "too wide" if all(t["width_mm"] > MAX_OBJECT_WIDTH_M * 1000 for t in tried) else "fingers blocked by neighbours"
    return dict(ok=False, tried=tried, reason=reason)


def order_targets(objects, target_class, arm_base_xy, camera_xyz=None):
    """Objects of the requested class, nearest first.
    camera_xyz None -> world-XY distance from the arm base; else 3D world distance from that camera position.
    The key used is stored in o["dist_from_arm_base_m"] / o["dist_from_camera_m"]; o["order_key_m"] is the one sorted on."""
    sel = [o for o in objects if o.get("center_xyz_m") and (target_class in (None, "any") or o["class_name"] == target_class)]
    for o in sel:
        o["dist_from_arm_base_m"] = round(math.dist(o["center_xyz_m"][:2], arm_base_xy), 4)
        if camera_xyz is not None:
            o["dist_from_camera_m"] = round(math.dist(o["center_xyz_m"], camera_xyz), 4)
        o["order_key_m"] = o["dist_from_camera_m"] if camera_xyz is not None else o["dist_from_arm_base_m"]
    return sorted(sel, key=lambda o: o["order_key_m"])


def choose_next(objects, target_class, arm_base_xy, table_z):
    """First graspable target in distance order. Returns (target, plan, skipped[])."""
    skipped = []
    for t in order_targets(objects, target_class, arm_base_xy):
        others = [o for o in objects if o is not t and o.get("center_xyz_m")]
        p = plan_grasp(t, others, table_z)
        if p["ok"]:
            return t, p, skipped
        skipped.append(dict(id=t["id"], class_name=t["class_name"], reason=p["reason"], tried=p["tried"]))
    return None, None, skipped


# ------------------------------------------------------------------ basket drop spot (depth)
def choose_drop_cell(P, depth, basket_min, basket_max, arm_base_xy, wall_margin=0.06, grid=(4, 3), occupied_h=0.012):
    """P: world points (H,W,3) of a depth image looking into the basket. Split the basket floor into cells;
    a cell is occupied if any point in it stands > occupied_h above the floor. Pick the free cell nearest
    the arm base. Returns dict(cell_center_xy, floor_z, cells=[...])."""
    ok = np.isfinite(depth) & (depth > 0) & (depth < 5)
    x0, y0 = basket_min[0] + wall_margin, basket_min[1] + wall_margin
    x1, y1 = basket_max[0] - wall_margin, basket_max[1] - wall_margin
    inside = ok & (P[..., 0] > x0) & (P[..., 0] < x1) & (P[..., 1] > y0) & (P[..., 1] < y1) & (P[..., 2] < basket_max[2])
    pts = P[inside]
    if len(pts) < 200:
        return None
    floor = float(np.percentile(pts[:, 2], 10))
    cells = []
    nx, ny = grid
    for i in range(nx):
        for j in range(ny):
            cx0, cx1 = x0 + (x1 - x0) * i / nx, x0 + (x1 - x0) * (i + 1) / nx
            cy0, cy1 = y0 + (y1 - y0) * j / ny, y0 + (y1 - y0) * (j + 1) / ny
            m = (pts[:, 0] >= cx0) & (pts[:, 0] < cx1) & (pts[:, 1] >= cy0) & (pts[:, 1] < cy1)
            n = int(m.sum())
            h = float(np.percentile(pts[m, 2], 98) - floor) if n else None
            center = [(cx0 + cx1) / 2, (cy0 + cy1) / 2]
            cells.append(dict(center_xy=[round(v, 4) for v in center], points=n,
                              max_height_mm=None if h is None else round(h * 1000, 1),
                              free=bool(n > 50 and h < occupied_h),
                              dist_from_arm_base_m=round(math.dist(center, arm_base_xy), 3)))
    free = [c for c in cells if c["free"]]
    pick = min(free, key=lambda c: c["dist_from_arm_base_m"]) if free else None
    return dict(floor_z=round(floor, 4), cells=cells, chosen=pick)
