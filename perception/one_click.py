"""One-click perception: open this file in the Isaac Sim Script Editor, Ctrl+Enter to run once.

Flow: start IK -> start YOLO server (if not running) -> move arm to view pose A -> D455 capture
-> YOLO detection -> use depth to compute the world-space coordinates of each object's visible
surface -> write HTML/JSON -> move arm back home.

Prerequisite: scenes/arm310d_d455_ik_demo.usd (the one in this folder) is already open.
Result: outputs/oneclick/<timestamp>/index.html, opened automatically in a browser when done.
From the Script Editor you can call one_click() to run again, or one_click_status() for progress.
Progress is also written to outputs/oneclick/latest_status.json (agents use this file to confirm results).
"""
import asyncio, builtins, datetime, json, os, socket, subprocess, sys, time, traceback
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(sys._getframe().f_code.co_filename)))
if not os.path.isfile(os.path.join(ROOT, "perception", "config.json")):
    # The Isaac Sim Script Editor execs a copy of this file from a temp path (e.g. /tmp/...),
    # so co_filename does not point at the real project directory. Fall back to the folder of the open scene.
    import omni.usd
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(omni.usd.get_context().get_stage().GetRootLayer().realPath)))
CFG = json.load(open(os.path.join(ROOT, "perception", "config.json"), encoding="utf-8"))
OUT = os.path.join(ROOT, "outputs", "oneclick")
REG = "_oneclick_registry"
ARM_MODULES = ["pose_math"]   # same-named modules an older project may have left in sys.modules


def _reg():
    if not hasattr(builtins, REG):
        setattr(builtins, REG, {"task": None, "status": {"state": "idle"}})
    return getattr(builtins, REG)


def _set_status(**kw):
    st = _reg()["status"]; st.update(kw); st["updated"] = datetime.datetime.now().isoformat(timespec="seconds")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "latest_status.json"), "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=2, default=str)


def _load_arm_stack():
    """Load this folder's scripts/ IK and capture code into a separate namespace (always reads the latest version from disk)."""
    import omni.usd
    stage_path = omni.usd.get_context().get_stage().GetRootLayer().realPath
    if not stage_path or os.path.dirname(os.path.dirname(os.path.abspath(stage_path))) != ROOT:
        raise RuntimeError(f"The currently open stage is not the one in this folder's scenes/: {stage_path}")
    scripts = os.path.join(ROOT, "scripts")
    for m in ARM_MODULES:            # avoid reusing same-named modules already loaded by an older project
        sys.modules.pop(m, None)
    if scripts in sys.path:
        sys.path.remove(scripts)
    sys.path.insert(0, scripts)
    import omni.timeline
    tl = omni.timeline.get_timeline_interface()
    tl.set_auto_update(True)        # a stalled Replicator step can leave it off -> physics frozen (see scripts/d455_capture.py)
    if not tl.is_playing():
        tl.play()
    ns = {}
    for name in ("ik_controller.py", "d455_capture.py"):
        path = os.path.join(scripts, name)
        ns["__file__"] = path
        exec(compile(open(path, encoding="utf-8").read(), path, "exec"), ns)
    return ns


def _cached_prim(cls, path):
    """Shared cache, see scripts/prim_cache.py (avoids leaking physics views)."""
    import builtins as _b, os as _os
    fn = getattr(_b, "_nhc_cached_prim_fn", None)
    if fn is None:
        p = _os.path.join(os.path.join(ROOT, "scripts"), "prim_cache.py"); g = {"__file__": p}
        exec(compile(open(p, encoding="utf-8").read(), p, "exec"), g)
        fn = g["cached_prim"]; setattr(_b, "_nhc_cached_prim_fn", fn)
    return fn(cls, path)


def _pose(path, xyz=None, quat=None):
    from isaacsim.core.experimental.prims import XformPrim
    import isaacsim.core.experimental.utils.backend as backend_utils
    prim = _cached_prim(XformPrim, path)
    with backend_utils.use_backend("fabric", raise_on_fallback=True):
        if xyz is None:
            p, q = prim.get_world_poses()
            return p.numpy()[0].tolist(), q.numpy()[0].tolist()
        prim.set_world_poses(np.array([xyz]), np.array([quat]))


def _slerp(q0, q1, t):
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + t * (q1 - q0)
    else:
        th = np.arccos(d)
        q = (np.sin((1 - t) * th) * q0 + np.sin(t * th) * q1) / np.sin(th)
    return (q / np.linalg.norm(q)).tolist()


async def _move_to(ns, xyz, quat):
    """Move the IK target to the goal in steps, then wait for the arm to converge. Returns convergence info."""
    import omni.kit.app
    app = omni.kit.app.get_app()
    p0, q0 = _pose("/World/IKTarget")
    n = CFG["move_steps"]
    for i in range(1, n + 1):
        t = i / n
        _pose("/World/IKTarget", (np.asarray(p0) + t * (np.asarray(xyz) - np.asarray(p0))).tolist(), _slerp(q0, quat, t))
        for _ in range(3):
            await app.next_update_async()
    ctrl = getattr(builtins, "_arm310d_ik_follow_controller_registry", {}).get("instance")
    err = None
    for frame in range(CFG["converge_max_frames"]):
        await app.next_update_async()
        err = ctrl.status().get("last_error") if ctrl else None
        if isinstance(err, dict) and err.get("pos_error_norm_m", 9) <= CFG["converge_pos_m"] \
                and err.get("rot_error_norm_rad", 9) <= CFG["converge_rot_rad"]:
            return dict(converged=True, frames=frame, error=err)
    return dict(converged=False, frames=CFG["converge_max_frames"], error=err)


async def _request(obj, timeout=60):
    reader, writer = await asyncio.wait_for(asyncio.open_connection("127.0.0.1", CFG["port"]), timeout=3)
    try:
        data = json.dumps(obj).encode()
        writer.write(len(data).to_bytes(4, "big") + data); await writer.drain()
        n = int.from_bytes(await asyncio.wait_for(reader.readexactly(4), timeout), "big")
        return json.loads(await asyncio.wait_for(reader.readexactly(n), timeout))
    finally:
        writer.close()


async def _ensure_server():
    import omni.kit.app
    src = os.path.join(ROOT, "perception", "yolo_server.py")
    try:
        r = await _request({"cmd": "ping"}, 5)
        if r.get("src_mtime") == os.path.getmtime(src):
            return "already_running"
        await _request({"cmd": "quit"}, 5)      # server code has changed -> restart to avoid running a stale version
        for _ in range(30):
            await omni.kit.app.get_app().next_update_async()
    except Exception:
        pass
    py = CFG["yolo_python"] if os.path.isabs(CFG["yolo_python"]) else os.path.join(ROOT, CFG["yolo_python"])
    log = open(os.path.join(OUT, "yolo_server.log"), "a")
    subprocess.Popen([py, os.path.join(ROOT, "perception", "yolo_server.py")], cwd=ROOT,
                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    t = time.time()
    while time.time() - t < 120:
        for _ in range(10):
            await omni.kit.app.get_app().next_update_async()
        try:
            await _request({"cmd": "ping"}, 5); return "started"
        except Exception:
            pass
    raise RuntimeError("YOLO server did not start within 120 seconds, check outputs/oneclick/yolo_server.log")


def _geometry():
    """Load perception/geometry.py fresh from disk every run (avoids Kit's stale-module cache)."""
    import types
    path = os.path.join(ROOT, "perception", "geometry.py")
    mod = types.ModuleType("oneclick_geometry"); mod.__file__ = path
    exec(compile(open(path, encoding="utf-8").read(), path, "exec"), mod.__dict__)
    return mod


def _view_poses():
    """View list from config: each has name + IK target pose (offset_z_m is relative to view A)."""
    a = CFG["view_pose_A"]
    poses = []
    for v in CFG.get("views", [{"name": "A"}]):
        p = list(v.get("position_m", a["position_m"]))
        p[2] += v.get("offset_z_m", 0.0)
        poses.append(dict(name=v["name"], position_m=p, quaternion_wxyz=v.get("quaternion_wxyz", a["quaternion_wxyz"])))
    return poses


def _dump_gt(stage, path):
    """Evaluation/report only (never used by the method): stage truth of test blocks and /World/object.
    center = world bbox center (true geometric center); translate = prim world translation."""
    from pxr import Usd, UsdGeom
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    prims = []
    tb = stage.GetPrimAtPath("/World/TestBlocks")
    if tb.IsValid():
        prims += [c for c in tb.GetChildren() if c.IsA(UsdGeom.Gprim)]
    if stage.GetPrimAtPath("/World/object").IsValid():
        prims.append(stage.GetPrimAtPath("/World/object"))
    gt = {}
    for p in prims:
        r = cache.ComputeWorldBound(p).ComputeAlignedRange()
        mn, mx = r.GetMin(), r.GetMax()
        tr = UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
        gt[p.GetName()] = dict(center_xyz_m=[(mn[i] + mx[i]) / 2 for i in range(3)], top_z_m=mx[2],
                               translate_xyz_m=[float(tr[i]) for i in range(3)],
                               size_mm=[round(1000 * (mx[i] - mn[i]), 1) for i in range(3)])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(gt, f, indent=1)


async def _observe(ns, geo, run, view):
    """Move -> capture -> YOLO -> per-object world center (+ cube/cuboid size check), for one view."""
    import omni.usd
    vdir = os.path.join(run, "view_" + view["name"]); os.makedirs(vdir, exist_ok=True)
    t = time.time()
    arrival = await _move_to(ns, view["position_m"], view["quaternion_wxyz"])
    t_move = time.time() - t; t = time.time()
    await ns["capture"](os.path.join(vdir, "capture"))
    stage = omni.usd.get_context().get_stage()
    left = ns["CAMERA_PATHS"]["left"]
    fr = ns["camera_frame"](stage, left); K = ns["camera_intrinsics"](stage, left)
    cam = {k: [float(x) for x in fr[k]] for k in ("position_m", "right", "up", "forward")}
    with open(os.path.join(vdir, "camera.json"), "w") as f:
        json.dump(dict(camera=cam, intrinsics={k: K[k] for k in ("fx", "fy", "cx", "cy")}), f, indent=1)
    t_cap = time.time() - t; t = time.time()
    rgb = os.path.join(vdir, "capture", "rgb_left.png")
    reply = await _request({"cmd": "detect", "rgb_path": rgb, "out_dir": vdir, "conf": CFG["candidate_conf"]})
    if not reply.get("ok"):
        raise RuntimeError("YOLO: " + str(reply.get("error")))
    t_yolo = time.time() - t
    depth = np.load(os.path.join(vdir, "capture", "depth_axial_left.npy"))
    masks_all = np.load(os.path.join(vdir, "masks.npz"))["masks"]
    P = geo.world_points(depth, cam, K, CFG.get("pixel_center_offset", 0.5))
    mg = CFG.get("detection_region_margin_m", 0.0)
    fg = geo.foreground(depth, P, CFG["table_z_m"], [CFG["workspace_x_m"][0] - mg, CFG["workspace_x_m"][1] + mg],
                        [CFG["workspace_y_m"][0] - mg, CFG["workspace_y_m"][1] + mg])
    dets, masks = geo.select_detections(reply["detections"], masks_all, fg, CFG["conf"])
    unexplained = geo.unexplained_regions(fg, masks, P)
    # a depth blob no detection explains becomes an "unknown" object (position/size from depth only)
    for u in unexplained:
        dets.append(dict(id=len(dets) + 1, class_id=-1, class_name="unknown", conf=0.0, depth_only=True,
                         bbox_xyxy=None, mask_pixels=u["pixels"]))
        masks = np.concatenate([masks, u.pop("mask")[None]], 0) if len(masks) else u.pop("mask")[None]
    np.savez_compressed(os.path.join(vdir, "masks_kept.npz"), masks=masks)
    union = masks.any(0) if len(masks) else None
    for i, d in enumerate(dets):
        others = (union & ~masks[i]) if union is not None else None
        d["yolo_class"] = d["class_name"]
        d.update(geo.estimate(masks[i], depth, P, cam, d["class_name"], CFG["table_z_m"], exclude=others))
        if d["yolo_class"] in ("cube", "cuboid"):
            sc = geo.box_shape_check(d)
            if sc:
                d["shape_check"] = sc
                d["class_name"] = sc["shape_class"]      # depth-measured edge ratio decides cube vs cuboid
    mg = CFG.get("detection_region_margin_m", 0.0)   # table workspace + margin; drops e.g. the background boxes
    wx = [CFG["workspace_x_m"][0] - mg, CFG["workspace_x_m"][1] + mg]; wy = [CFG["workspace_y_m"][0] - mg, CFG["workspace_y_m"][1] + mg]
    inside = [c is None or (wx[0] <= c[0] <= wx[1] and wy[0] <= c[1] <= wy[1]) for c in (d.get("center_xyz_m") for d in dets)]
    outside = [dict(class_name=d["class_name"], conf=d["conf"], center_xyz_m=d.get("center_xyz_m")) for d, ok in zip(dets, inside) if not ok]
    if outside:
        keep = [i for i, ok in enumerate(inside) if ok]
        dets = [dets[i] for i in keep]; masks = masks[keep]
        for k, d in enumerate(dets):
            d["id"] = k + 1
        np.savez_compressed(os.path.join(vdir, "masks_kept.npz"), masks=masks)
    return dict(name=view["name"], dir=vdir, rgb=rgb, arrival=arrival, detections=dets, unexplained=unexplained,
                outside_workspace=outside,
                camera=cam, timing_s=dict(move=t_move, capture=t_cap, yolo=t_yolo))


def _needs_second_view(a):
    """Reasons to visit view B after view A (empty list = skip B)."""
    thr = CFG.get("class_from_second_view_below_conf", 0.7)
    why = [f"#{i + 1} {d['class_name']} conf {d['conf']:.2f}" for i, d in enumerate(a["detections"]) if d["conf"] < thr]
    why += [f"unexplained depth region at ({u['center_xyz_m'][0]:.3f}, {u['center_xyz_m'][1]:.3f})" for u in a["unexplained"]]
    return why


def _merge(views):
    """One id per physical object across views (matched by world XY), so ids/colors agree between
    the two pictures. Final class: view A; view B's class only when A's YOLO confidence is below
    class_from_second_view_below_conf. Box family: cube/cuboid decided by the measured edge ratio.
    Final center: view A (fallback: B)."""
    base, others = views[0], views[1:]
    thr = CFG.get("class_from_second_view_below_conf", 0.7)
    for i, d in enumerate(base["detections"]):
        d["id"] = i + 1
    next_id = len(base["detections"]) + 1
    for v in others:
        taken = set()
        for d in v["detections"]:
            d["id"] = None
        pairs = []
        for j, o in enumerate(v["detections"]):
            if o.get("center_xyz_m") is None:
                continue
            for a in base["detections"]:
                if a.get("center_xyz_m") is None:
                    continue
                dist = float(np.hypot(*(np.array(o["center_xyz_m"][:2]) - a["center_xyz_m"][:2])))
                if dist <= CFG.get("match_xy_m", 0.03):
                    pairs.append((dist, j, a["id"]))
        for dist, j, aid in sorted(pairs):
            if v["detections"][j]["id"] is None and aid not in taken:
                v["detections"][j]["id"] = aid; taken.add(aid)
        for d in v["detections"]:
            if d["id"] is None:
                d["id"] = next_id; next_id += 1
    objects = []
    for a in base["detections"]:
        per = {base["name"]: a}
        for v in others:
            m = next((o for o in v["detections"] if o["id"] == a["id"]), None)
            if m is not None:
                per[v["name"]] = m
        src = base["name"]; cls = a["class_name"]; conf = a["conf"]
        if a["conf"] < thr and len(per) > 1:
            k2 = next(k for k in per if k != base["name"])
            if per[k2]["conf"] > a["conf"]:
                src = k2; cls, conf = per[k2]["class_name"], per[k2]["conf"]
        if cls in ("cube", "cuboid") and a.get("shape_check"):
            # view A's depth-measured edge ratio beats any YOLO confidence for cube vs cuboid
            # (batch 2026-09-28: 158/158 correct; view B's oblique top face once gave a wrong size)
            cls = a["shape_check"]["shape_class"]; src = base["name"]; conf = a["conf"]
        csrc = next((k for k in [base["name"]] + [v["name"] for v in others] if per.get(k, {}).get("center_xyz_m")), base["name"])
        objects.append(dict(id=a["id"], class_name=cls, conf=conf, class_from_view=src,
                            center_xyz_m=per[csrc].get("center_xyz_m"), center_from_view=csrc,
                            height_mm=per[csrc].get("height_mm"),
                            per_view={k: dict(class_name=x["class_name"], yolo_class=x.get("yolo_class"), conf=x["conf"],
                                              center_xyz_m=x.get("center_xyz_m"), height_mm=x.get("height_mm"),
                                              shape_check=x.get("shape_check")) for k, x in per.items()}))
    extra = [dict(o, view=v["name"]) for v in others for o in v["detections"]
             if not any(o["id"] == a["id"] for a in base["detections"])]
    return objects, extra


def _hex(i):
    b, g, r = CFG["palette_bgr"][(i - 1) % len(CFG["palette_bgr"])]
    return f"#{r:02x}{g:02x}{b:02x}"


def _gt_lookup(run):
    """Report only: stage truth for each measured center (nearest object within 5 cm)."""
    p = os.path.join(run, "gt_eval_only.json")
    gt = json.load(open(p)) if os.path.exists(p) else {}
    def find(c):
        if c is None or not gt:
            return None
        name = min(gt, key=lambda k: np.hypot(gt[k]["center_xyz_m"][0] - c[0], gt[k]["center_xyz_m"][1] - c[1]))
        g = gt[name]
        return (name, g) if np.hypot(g["center_xyz_m"][0] - c[0], g["center_xyz_m"][1] - c[1]) <= 0.05 else None
    return find


def _rows(items, find, class_note):
    """items: list of (id, class html, conf html, center, height_mm). Returns table rows + per-class axis errors."""
    html, errs = [], []
    for oid, cls, conf, c, h in items:
        g = find(c)
        if c is None:
            cc, hh = "<i>no depth</i>", ""
        elif g is None:
            cc = f"<div class=m>{' , '.join(f'{v:.4f}' for v in c)}</div><div class=s>no stage object here</div>"
            hh = f"{h:.1f}" if h else ""
        else:
            name, gg = g
            t = gg["center_xyz_m"]; e = [1000 * (t[k] - c[k]) for k in range(3)]
            th = gg["size_mm"][2]; eh = 100 * (th - h) / th if h else None
            errs.append(dict(cls=name.split("_")[0] if name != "object" else "cube", e=e, eh=eh))
            cc = (f"<div class=s>{' , '.join(f'{v:.4f}' for v in t)}</div>"
                  f"<div class=m>{' , '.join(f'{v:.4f}' for v in c)}</div>"
                  f"<div class=e>{' , '.join(f'{v:+.2f}' for v in e)}</div>")
            hh = (f"<div class=s>{th:.1f}</div><div class=m>{h:.1f}</div>"
                  f"<div class=e>{'' if eh is None else f'{eh:+.2f}%'}</div>")
        html.append(f"<tr><td><span class=sw style='background:{_hex(oid)}'>{oid}</span></td><td>{cls}</td><td>{conf}</td>"
                    f"<td class=c>{cc}</td><td class=c>{hh}</td></tr>")
    return "".join(html), errs


def _axis_table(errs_by_view):
    classes = ["cube", "cuboid", "cylinder", "sphere"]
    out = ["<table><tr><th>view</th><th>class</th><th>n</th><th>mean |ΔX|</th><th>mean |ΔY|</th><th>mean |ΔZ|</th>"
           "<th>max |ΔX|</th><th>max |ΔY|</th><th>max |ΔZ|</th><th>max |height err|</th></tr>"]
    for vname, errs in errs_by_view.items():
        for c in classes:
            e = np.array([x["e"] for x in errs if x["cls"] == c])
            if not len(e):
                continue
            eh = [abs(x["eh"]) for x in errs if x["cls"] == c and x["eh"] is not None]
            a = np.abs(e); big = int(np.argmax(a.mean(0)))
            cells = [f"<td{' class=b' if k == big else ''}>{a[:, k].mean():.2f}</td>" for k in range(3)]
            out.append(f"<tr><td>{vname}</td><td>{c}</td><td>{len(e)}</td>{''.join(cells)}"
                       f"{''.join(f'<td>{a[:, k].max():.2f}</td>' for k in range(3))}<td>{max(eh):.2f}%</td></tr>")
    return "".join(out) + "</table>"


def _second_view_line(doc):
    sv = doc.get("second_view") or {}
    if not sv:
        return ""
    if not sv.get("visited"):
        return "Second view: <b>not needed</b> (every view-A object has confidence ≥ 0.7 and no unexplained depth region)."
    return "Second view: <b>visited</b> because " + "; ".join(sv.get("reasons") or ["policy = always"]) + "."


def _unexplained_html(doc):
    rows = [f"<li>view {v['name']}: depth region with no YOLO detection → listed as <b>unknown</b> ({u['center_xyz_m'][0]:.3f}, {u['center_xyz_m'][1]:.3f}, {u['center_xyz_m'][2]:.3f}) m, "
            f"{u['pixels']} px, top {u['height_mm']:.0f} mm above table</li>" for v in doc["views"] for u in v.get("unexplained", [])]
    rows += [f"<li>view {v['name']}: ignored {o['class_name']} {o['conf']:.2f} outside the table workspace</li>"
             for v in doc["views"] for o in v.get("outside_workspace", [])]
    return ("<h3>Notes</h3><ul>" + "".join(rows) + "</ul>") if rows else ""


def _html(run, doc):
    find = _gt_lookup(run)
    vA = doc["views"][0]["name"]
    itemsA = []
    for o in doc["objects"]:
        pa = o["per_view"][vA]
        note = ""
        if o["class_from_view"] != vA:
            note = f"<div class=s>from {o['class_from_view']} ({vA} YOLO {pa['yolo_class']} {pa['conf']:.2f})</div>"
        a_det = next(x for x in doc["views"][0]["detections"] if x["id"] == o["id"])
        if a_det.get("rescued"):
            note += "<div class=s>low-score YOLO candidate confirmed by depth</div>"
        if a_det.get("depth_only"):
            note += "<div class=s>no YOLO detection — object found from depth only</div>"
        sc = pa.get("shape_check")
        if sc:
            note += f"<div class=s>size {sc['box_dims_mm'][0]:.0f}×{sc['box_dims_mm'][1]:.0f}×{sc['box_dims_mm'][2]:.0f} mm"
            note += (f", YOLO said {pa['yolo_class']}" if pa["yolo_class"] != sc["shape_class"] else "") + "</div>"
        itemsA.append((o["id"], f"<b>{o['class_name']}</b>{note}", f"{o['conf']:.2f}", o["center_xyz_m"], o.get("height_mm")))
    rowsA, errsA = _rows(itemsA, find, True)
    errs = {vA: errsA}
    tables = ""
    for v in doc["views"][1:]:
        items = []
        for d in sorted(v["detections"], key=lambda d: d["id"]):
            sc = d.get("shape_check")
            note = ""
            if sc and d.get("yolo_class") != sc["shape_class"]:
                note = f"<div class=s>YOLO said {d['yolo_class']}</div>"
            items.append((d["id"], f"{d['class_name']}{note}", f"{d['conf']:.2f}", d.get("center_xyz_m"), d.get("height_mm")))
        r, e = _rows(items, find, False); errs[v["name"]] = e
        tables += (f"<h2>View {v['name']} (reference)</h2><table><tr><th>#</th><th>class</th><th>YOLO conf</th>"
                   f"<th>center X , Y , Z (m)<div class=s>stage</div><div class=m>measured</div><div class=e>error = stage − measured (mm)</div></th>"
                   f"<th>height (mm)<div class=s>stage</div><div class=m>measured</div><div class=e>error %</div></th></tr>{r}</table>")
    tm = doc["timing_s"]
    imgs = "".join(f"<figure><img src='{'overlay.png' if i == 0 else 'view_' + v['name'] + '/overlay_ids.png'}'>"
                   f"<figcaption>View {v['name']}</figcaption></figure>" for i, v in enumerate(doc["views"]))
    html = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>One-Click Perception {doc['run_id']}</title>
<style>body{{font:14px system-ui;max-width:1200px;margin:24px auto;padding:0 16px;color:#222}}
img{{max-width:100%;border:1px solid #ccc}}figure{{margin:0}}figcaption{{color:#555}}
table{{border-collapse:collapse;width:100%;margin-bottom:8px}}td,th{{border-bottom:1px solid #ddd;padding:5px 8px;text-align:left;vertical-align:top}}
.g{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}.s{{color:#777;font-size:12px}}.m{{font-weight:600}}.e{{color:#b03030;font-size:12px}}
td.c{{font-family:ui-monospace,monospace;font-size:12px}}.sw{{display:inline-block;min-width:22px;text-align:center;color:#fff;border-radius:4px;font-weight:700;text-shadow:0 0 2px #000}}
td.b{{background:#fde2e2;font-weight:700}}</style>
<h1>One-Click Perception Result</h1>
<p>{doc['run_id']} | <b>{len(doc['objects'])}</b> object(s) in view {vA} | views: {', '.join(v['name'] for v in doc['views'])} | total {tm['total']:.1f} s.
Same number and color = same object in every picture and table.</p>
<p>{_second_view_line(doc)}</p>
<div class=g>{imgs}</div>
{_unexplained_html(doc)}
<h2>View {vA} — result</h2>
<table><tr><th>#</th><th>class</th><th>YOLO conf</th>
<th>center X , Y , Z (m)<div class=s>stage</div><div class=m>measured</div><div class=e>error = stage − measured (mm)</div></th>
<th>height (mm)<div class=s>stage</div><div class=m>measured</div><div class=e>error %</div></th></tr>{rowsA}</table>
<p class=s>Class: view {vA}; view B's class is used only when {vA}'s YOLO confidence &lt; {CFG.get('class_from_second_view_below_conf', 0.7)}.
cube vs cuboid is decided by the depth-measured edge ratio (&lt; {1.15} → cube). Stage values are read from the stage for comparison only; the method never uses them.</p>
{tables}
<h2>Error by class and axis (mm; the largest mean axis is highlighted)</h2>{_axis_table(errs)}
<h2>Timing (seconds)</h2><p>{' | '.join(f'{k}: {v:.1f}' for k, v in tm.items())}</p>
<p><a href="result.json">result.json</a></p></html>"""
    with open(os.path.join(run, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)


async def _run():
    import omni.kit.app, omni.usd
    t0 = time.time(); tm = {}
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run = os.path.join(OUT, run_id); os.makedirs(run, exist_ok=True)
    _set_status(state="running", step="load", run_id=run_id, run_dir=run, error=None)
    ns = _load_arm_stack(); geo = _geometry()
    ns["ik_follow_start"]()
    for _ in range(600):
        await omni.kit.app.get_app().next_update_async()
        if ns["ik_follow_status"]().get("running"):
            break
    else:
        raise RuntimeError(f"IK did not start: {ns['ik_follow_status']()}")
    _set_status(step="yolo_server"); server = await _ensure_server(); tm["setup"] = time.time() - t0
    _dump_gt(omni.usd.get_context().get_stage(), os.path.join(run, "gt_eval_only.json"))

    home = _pose("/World/IKTarget")
    views = []; second_view = dict(policy=CFG.get("second_view_policy", "on_low_conf"), reasons=[], visited=False)
    for i, vp in enumerate(_view_poses()):
        if i > 0:
            second_view["reasons"] = _needs_second_view(views[0])
            if second_view["policy"] != "always" and not second_view["reasons"]:
                continue
            second_view["visited"] = True
        _set_status(step="view_" + vp["name"])
        v = await _observe(ns, geo, run, vp); views.append(v)
        for k, x in v["timing_s"].items():
            tm[f"{v['name']}_{k}"] = x
    objects, extra = _merge(views)

    # redraw every view with the shared ids/colors (legend shows the final class for view A)
    labelsA = [dict(d, label=f"{o['id']} {o['class_name']} {o['conf']:.2f}") for d, o in zip(views[0]["detections"], objects)]
    jobs = [(views[0], labelsA, os.path.join(run, "overlay.png"))]
    for v in views[1:]:
        jobs.append((v, [dict(d, label=f"{d['id']} {d['class_name']} {d['conf']:.2f}") for d in v["detections"]],
                     os.path.join(v["dir"], "overlay_ids.png")))
    for v, dets, outp in jobs:
        reply = await _request({"cmd": "draw", "rgb_path": v["rgb"], "masks_path": os.path.join(v["dir"], "masks_kept.npz"),
                                "detections": dets, "out_path": outp})
        if not reply.get("ok"):
            raise RuntimeError("draw: " + str(reply.get("error")))

    _set_status(step="return"); t = time.time()
    back = await _move_to(ns, home[0], home[1]); tm["return"] = time.time() - t
    tm["total"] = time.time() - t0
    doc = dict(run_id=run_id, objects=objects, only_in_other_views=extra,
               views=[dict(name=v["name"], arrival=v["arrival"], camera=v["camera"], detections=v["detections"],
                           unexplained=v["unexplained"], outside_workspace=v["outside_workspace"]) for v in views], second_view=second_view,
               returned=back["converged"], server=server, model=CFG["model"], timing_s=tm)
    with open(os.path.join(run, "result.json"), "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2, default=float)
    _html(run, doc)
    _set_status(state="done", step="done", html=os.path.join(run, "index.html"), n_objects=len(objects),
                arrival_converged=all(v["arrival"]["converged"] for v in views), total_s=round(tm["total"], 1))
    if CFG.get("open_browser", True):
        try:
            import webbrowser; webbrowser.open("file://" + os.path.join(run, "index.html"))
        except Exception:
            pass
    return doc


async def _wrapped():
    try:
        return await _run()
    except Exception as exc:
        _set_status(state="error", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        print(traceback.format_exc())


def one_click():
    """Schedule one run of one-click perception and return immediately; use one_click_status() for progress."""
    reg = _reg()
    if reg["task"] is not None and not reg["task"].done():
        return {"state": "already_running"}
    reg["task"] = asyncio.ensure_future(_wrapped())
    return {"state": "scheduled", "status_file": os.path.join(OUT, "latest_status.json")}


def one_click_status():
    return dict(_reg()["status"])


if not globals().get("NO_AUTORUN"):
    print(one_click())
