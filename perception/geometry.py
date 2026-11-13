"""Geometry for one view: mask + depth -> world-frame object center.

Pure numpy + cv2 (both available inside Isaac Sim's python and the YOLO venv), no Kit imports,
so saved runs can be re-evaluated offline.

World-frame center estimate (object assumed resting on the table, table_z known):
  top_z    = robust top-face height (median of points within TOP_BAND of the 98th percentile)
  center_z = (top_z + table_z) / 2
  center_xy:
    sphere          -> sphere fit tangent to the table (cz = table_z + r), eroded mask; overrides center_z
    lying cylinder  -> (top strip clearly elongated) circle fit in the cross-section, tangent to the table
    everything else -> center of the minimum-area rectangle around the top-face points
                       (top face = pixels near the mask whose world z is within TOP_BAND of top_z)
"""
import numpy as np

TOP_BAND_M = 0.003
CUBE_RATIO_MAX = 1.15   # box with max/min edge ratio below this = cube (catalog: cubes 1.0, cuboids >= 1.5)
MASK_DILATE_PX = 4
MIN_POINTS = 20


def world_points(depth, cam, K, px_offset=0.5):
    """Per-pixel world xyz (H,W,3) from axial depth. cam: position_m/right/up/forward."""
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    z = depth.astype(np.float64)
    x = (u + px_offset - K["cx"]) * z / K["fx"]
    y = -(v + px_offset - K["cy"]) * z / K["fy"]
    R = np.stack([cam["right"], cam["up"], cam["forward"]], axis=1)
    return np.asarray(cam["position_m"], float) + np.stack([x, y, z], -1) @ R.T


def _dilate(mask, px):
    import cv2
    k = np.ones((2 * px + 1, 2 * px + 1), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), k) > 0


def _top_face_component(near, mask):
    """Keep only the real top face: a 3x3 opening removes thin strips (e.g. the band of a taller
    neighbour's side face that happens to sit at this object's top height), then keep the
    connected component that overlaps this object's mask the most."""
    import cv2
    m = cv2.morphologyEx(near.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab = cv2.connectedComponents(m, connectivity=8)
    if n <= 1:
        return near
    counts = np.bincount(lab[mask & (lab > 0)], minlength=n)
    best = int(np.argmax(counts))
    return (lab == best) if counts[best] > 0 else near


def _rect_center(xy):
    """Minimum-area rectangle center. cv2.minAreaRect misbehaves on tiny float coordinates
    (metres ~0.7 with sub-mm spread gave a wrong 80x87 mm rectangle for an 80x53 mm face),
    so work in millimetres around the centroid."""
    import cv2
    m = xy.mean(0)
    (cx, cy), (w, h), ang = cv2.minAreaRect(((xy - m) * 1000.0).astype(np.float32))
    return m + np.array([cx, cy]) / 1000.0, (w / 1000.0, h / 1000.0, ang)


def _sphere_xy(p, r, cz):
    # (x-cx)^2 + (y-cy)^2 = r^2 - (z-cz)^2  ->  linear in (cx, cy, k=cx^2+cy^2)
    rhs = r * r - (p[:, 2] - cz) ** 2
    A = np.stack([2 * p[:, 0], 2 * p[:, 1], -np.ones(len(p))], 1)
    b = p[:, 0] ** 2 + p[:, 1] ** 2 - rhs
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    return sol[:2]


def _sphere_tangent_fit(mask, ok, P, table_z, erode_px=2):
    """Fit a sphere resting on the table (cz = table_z + r) to the eroded-mask surface points.
    Erosion drops grazing-angle edge pixels whose depth mixes sphere and background."""
    import cv2
    from scipy.optimize import least_squares
    m = cv2.erode(mask.astype(np.uint8), np.ones((2 * erode_px + 1,) * 2, np.uint8)) > 0
    pts = P[m & ok]
    if len(pts) < MIN_POINTS:
        return None
    c0 = pts.mean(0)
    f = lambda x: np.linalg.norm(pts - np.r_[x[0], x[1], table_z + x[2]], axis=1) - x[2]
    s = least_squares(f, np.r_[c0[0], c0[1], max(0.01, (pts[:, 2].max() - table_z) / 2)],
                      loss="soft_l1", f_scale=0.002).x
    return np.r_[s[0], s[1], table_z + s[2], s[2]]


def _lying_cylinder_fit(mask, ok, P, table_z, rect_xy, rect, erode_px=2):
    """Cylinder lying on the table: axis = long side of the top strip. In the plane perpendicular
    to the axis the cross-section is a circle tangent to the table -> fit its offset and radius."""
    import cv2
    from scipy.optimize import least_squares
    w, l, ang = rect
    a = np.deg2rad(ang if w >= l else ang + 90.0)
    axis = np.array([np.cos(a), np.sin(a)])
    perp = np.array([-axis[1], axis[0]])
    m = cv2.erode(mask.astype(np.uint8), np.ones((2 * erode_px + 1,) * 2, np.uint8)) > 0
    pts = P[m & ok]
    if len(pts) < MIN_POINTS:
        return None
    s = (pts[:, :2] - rect_xy) @ perp
    z = pts[:, 2]
    f = lambda x: np.hypot(s - x[0], z - (table_z + x[1])) - x[1]
    sol = least_squares(f, np.r_[0.0, max(0.01, (z.max() - table_z) / 2)], loss="soft_l1", f_scale=0.002).x
    return rect_xy + sol[0] * perp, table_z + sol[1], sol[1]


def box_shape_check(d):
    """cube vs cuboid from measured dimensions (depth), independent of YOLO.
    Batch 2026-09-28: 167/167 box-family detections correct (YOLO alone 160/167)."""
    if d.get("top_face_size_mm") is None or d.get("height_mm") is None:
        return None
    dims = [d["top_face_size_mm"][0], d["top_face_size_mm"][1], d["height_mm"]]
    if min(dims) <= 0:
        return None
    ratio = max(dims) / min(dims)
    return dict(box_dims_mm=[round(x, 1) for x in dims], edge_ratio=round(ratio, 2),
                shape_class="cube" if ratio < CUBE_RATIO_MAX else "cuboid")


def estimate(mask, depth, P, cam, class_name, table_z, exclude=None):
    """mask (H,W) bool, depth (H,W), P = world_points(...). exclude = union of the OTHER detections'
    masks, so a neighbour's top face at the same height is never counted as this object's."""
    ok = np.isfinite(depth) & (depth > 0) & (depth < 5)
    sel = mask & ok
    n = int(sel.sum())
    out = dict(depth_valid_fraction=round(n / max(1, int(mask.sum())), 3))
    if n < MIN_POINTS:
        out.update(center_xyz_m=None, note="too few valid depth points")
        return out
    pts = P[sel]
    z = pts[:, 2]
    p98 = np.percentile(z, 98)
    top_z = float(np.median(z[z > p98 - TOP_BAND_M]))
    height = top_z - table_z
    cz = table_z + height / 2
    surf = np.median(pts, 0)
    if class_name == "sphere":
        c = _sphere_tangent_fit(mask, ok, P, table_z)
        if c is not None:
            xy, cz, r = c[:2], c[2], c[3]
            height = 2 * r
            method, extra = "sphere_fit_tangent_to_table", dict(radius_mm=round(1000 * r, 1))
        else:
            xy = _sphere_xy(pts, height / 2, cz)
            method, extra = "sphere_fit_known_radius", {}
    else:
        near = _dilate(mask, MASK_DILATE_PX) & ok & (np.abs(P[..., 2] - top_z) < TOP_BAND_M)
        if exclude is not None:
            near &= ~(exclude & ~mask)
        near = _top_face_component(near, mask)
        top_pts = P[near]
        if len(top_pts) < MIN_POINTS:
            top_pts = pts[np.abs(z - top_z) < TOP_BAND_M]
        xy, (w, l, ang) = _rect_center(top_pts[:, :2])
        method = "top_face_rect"
        if class_name == "cylinder" and max(w, l) > 1.6 * max(1e-6, min(w, l)):
            fit = _lying_cylinder_fit(mask, ok, P, table_z, xy, (w, l, ang))
            if fit is not None:
                xy, cz, r = fit
                height = 2 * r
                method = "lying_cylinder_fit_tangent_to_table"
        long_yaw = ang if w >= l else ang + 90.0            # direction of the long side, world XY, degrees
        fxy, (fw, fl, fa) = _rect_center(pts[:, :2])       # full top-view footprint (all visible points)
        f_long = fa if fw >= fl else fa + 90.0
        extra = dict(top_face_size_mm=[round(1000 * max(w, l), 1), round(1000 * min(w, l), 1)],
                     top_face_yaw_deg=round(((long_yaw + 90) % 180) - 90, 2),
                     footprint_size_mm=[round(1000 * max(fw, fl), 1), round(1000 * min(fw, fl), 1)],
                     footprint_yaw_deg=round(((f_long + 90) % 180) - 90, 2),
                     footprint_center_xy_m=[round(float(fxy[0]), 5), round(float(fxy[1]), 5)],      # 11/09: centre of that rectangle (placement outline)
                                              top_face_points=int(len(top_pts)))
    c = np.array([xy[0], xy[1], cz])
    out.update(center_xyz_m=[round(float(v), 5) for v in c], center_method=method,
               top_z_m=round(top_z, 5), height_mm=round(height * 1000, 1),
               visible_surface_xyz_m=[round(float(v), 4) for v in surf],
               distance_from_camera_m=round(float(np.linalg.norm(c - np.asarray(cam["position_m"]))), 4),
               **extra)
    return out


# ---------------------------------------------------------------- detection selection (depth-assisted)
FG_HEIGHT_M = (0.008, 0.15)      # "something standing on the table" band above the table surface
RESCUE_FG_FRACTION = 0.6         # low-score candidate kept if >= 60 % of its mask is table foreground ...
RESCUE_MAX_OVERLAP = 0.2         # ... and it is not mostly covering an already accepted object
UNEXPLAINED_MIN_PX = 200


def foreground(depth, P, table_z, wx, wy):
    """Pixels whose world point stands 8-150 mm above the table inside the workspace."""
    ok = np.isfinite(depth) & (depth > 0) & (depth < 5)
    h = P[..., 2] - table_z
    return ok & (h > FG_HEIGHT_M[0]) & (h < FG_HEIGHT_M[1]) & \
        (P[..., 0] >= wx[0]) & (P[..., 0] <= wx[1]) & (P[..., 1] >= wy[0]) & (P[..., 1] <= wy[1])


def select_detections(dets, masks, fg, conf_min):
    """Keep YOLO detections >= conf_min; rescue lower-score candidates whose mask is real
    foreground (depth) not already explained by a kept detection. Returns (dets, masks)."""
    keep = [i for i, d in enumerate(dets) if d["conf"] >= conf_min]
    union = masks[keep].any(0) if keep else np.zeros(fg.shape, bool)
    for i, d in sorted(enumerate(dets), key=lambda t: -t[1]["conf"]):
        if i in keep or i >= len(masks):
            continue
        m = masks[i]; n = int(m.sum())
        if n < UNEXPLAINED_MIN_PX:
            continue
        if (m & fg).sum() / n >= RESCUE_FG_FRACTION and (m & union).sum() / n < RESCUE_MAX_OVERLAP:
            d["rescued"] = True
            keep.append(i); union |= m
    keep.sort(key=lambda i: -dets[i]["conf"])
    out = [dets[i] for i in keep]
    for k, d in enumerate(out):
        d["id"] = k + 1
    return out, (masks[keep] if keep else np.zeros((0,) + fg.shape, bool))


def unexplained_regions(fg, masks, P):
    """Foreground blobs that no kept mask covers (possible missed objects)."""
    import cv2
    covered = _dilate(masks.any(0), 3) if len(masks) else np.zeros(fg.shape, bool)
    rest = cv2.morphologyEx((fg & ~covered).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(rest, connectivity=8)
    out = []
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] >= UNEXPLAINED_MIN_PX:
            pts = P[lab == k]
            out.append(dict(mask=(lab == k), pixels=int(stats[k, cv2.CC_STAT_AREA]),
                            center_xyz_m=[round(float(v), 4) for v in np.median(pts, 0)],
                            height_mm=round(float((np.percentile(pts[:, 2], 98) - 0.5952) * 1000), 1)))
    return out
