"""Wrist D455 capture (left eye only): RGB png + axial depth npy, camera pose and intrinsics.

Replaces the old capture_d455.py chain (7 modules), which also segmented / annotated the old red-cube demo.
Loaded by perception/one_click.py._load_arm_stack() into the same namespace as ik_controller.py.

    await capture(out_dir)            -> writes out_dir/rgb_left.png, out_dir/depth_axial_left.npy; returns info dict
    await snapshot(camera, png, size) -> one RGB picture from any camera prim (e.g. /World/FrontCamera)
Cameras render ONLY while a picture is taken (render products are created once, then switched on/off): an extra
camera left rendering every frame slowed the whole simulation down (2026-09-30: 3 leftover cameras -> 8.9 FPS).
    camera_frame(stage, path)         -> world position + right/up/forward axes
    camera_intrinsics(stage, path)    -> fx, fy, cx, cy (pinhole, from focal length / aperture)
"""
import asyncio, builtins, math, os
import numpy as np

MOUNT_PATH = "/World/ArmWithHandOnly/Robotiq_2F_85/Robotiq_2F_85/base_link/d455_camera"
CAMERA_PATHS = {"left": MOUNT_PATH + "/RSD455/Camera_OmniVision_OV9782_Left"}
IK_TARGET_PATH = "/World/IKTarget"
WIDTH, HEIGHT = 640, 480                 # the YOLO model was trained on this size
REGISTRY = "_nhc_d455_capture_left"
_SCRIPTS_DIR = __import__("os").path.dirname(__import__("os").path.abspath(__file__))   # loaded with __file__ set
STEP_WATCHDOG_S = 5.0                    # step_async normally returns in < 1 s
STEP_GIVE_UP_S = 30.0


def _cached_prim(cls, path):
    """Shared cache, see scripts/prim_cache.py (avoids leaking physics views)."""
    import builtins as _b, os as _os
    fn = getattr(_b, "_nhc_cached_prim_fn", None)
    if fn is None:
        p = _os.path.join(_SCRIPTS_DIR, "prim_cache.py"); g = {"__file__": p}
        exec(compile(open(p, encoding="utf-8").read(), p, "exec"), g)
        fn = g["cached_prim"]; setattr(_b, "_nhc_cached_prim_fn", fn)
    return fn(cls, path)


def _registry():
    if not hasattr(builtins, REGISTRY):
        setattr(builtins, REGISTRY, dict(stage_id=None, rgb=None, depth=None))
    return getattr(builtins, REGISTRY)


async def _setup():
    """Render product + annotators for the left camera, created once per open stage."""
    import omni.usd, omni.replicator.core as rep
    ctx = omni.usd.get_context(); reg = _registry()
    if reg["stage_id"] == ctx.get_stage_id() and reg["rgb"] is not None:
        return
    path = CAMERA_PATHS["left"]
    if not ctx.get_stage().GetPrimAtPath(path).IsValid():
        raise RuntimeError(f"Camera prim not found: {path}")
    rp = rep.create.render_product(path, (WIDTH, HEIGHT))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb"); rgb.attach(rp)
    depth = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane"); depth.attach(rp)
    reg.update(stage_id=ctx.get_stage_id(), rgb=rgb, depth=depth, render_product=rp, extra={})
    await render_step(8)
    _enable(rp, False)


def _enable(rp, on):
    try:
        rp.hydra_texture.set_updates_enabled(on)
    except Exception:
        pass


async def _rendering(rp, frames=3):
    """Switch a render product on and let it produce fresh frames."""
    import omni.kit.app
    _enable(rp, True)
    for _ in range(frames):
        await omni.kit.app.get_app().next_update_async()


async def step_async_with_watchdog(rep, rt_subframes):
    """rep.orchestrator.step_async() with a recovery for a hang seen in Isaac Sim 6.0 (Replicator 1.13).

    step_async() switches the timeline's auto-update OFF and then waits for the render dispatcher; sometimes it waits
    forever -> physics and the arm freeze (2026-09-30: pick("sphere") stuck at view_A; any target; about half of the
    captures stall). Recovery (verified live): turn auto-update back on -> step_async() finishes by itself.
    Returns None when it finished normally, else what the watchdog did."""
    import omni.timeline
    task = asyncio.ensure_future(rep.orchestrator.step_async(rt_subframes=rt_subframes, pause_timeline=False))
    done, _ = await asyncio.wait({task}, timeout=STEP_WATCHDOG_S)
    if done:
        task.result(); return None
    tl = omni.timeline.get_timeline_interface()
    tl.set_auto_update(True)
    if not tl.is_playing():
        tl.play()
    done, _ = await asyncio.wait({task}, timeout=STEP_GIVE_UP_S)
    if done:
        task.result(); return "step_async stalled; auto-update restored -> finished"
    task.cancel(); tl.set_auto_update(True)
    raise RuntimeError("step_async stalled even after restoring auto-update; cancelled")


async def render_step(rt_subframes=4):
    """One synchronized render. Falls back to plain app updates if Replicator's step fails."""
    import omni.kit.app, omni.replicator.core as rep
    try:
        return await step_async_with_watchdog(rep, rt_subframes)
    except Exception as exc:
        for _ in range(max(1, rt_subframes)):
            await omni.kit.app.get_app().next_update_async()
        return f"fallback to app updates ({type(exc).__name__}: {exc})"


async def _wait_for_ik_settle(max_frames=180, stable_frames=5):
    import omni.kit.app
    app = omni.kit.app.get_app()
    ctrl = getattr(builtins, "_arm310d_ik_follow_controller_registry", {}).get("instance")
    if ctrl is None:
        for _ in range(10):
            await app.next_update_async()
        return dict(settled=True, reason="IK controller is not running")
    stable = 0
    for frame in range(1, max_frames + 1):
        await app.next_update_async()
        err = ctrl.status().get("last_error")
        ok = isinstance(err, dict) and err.get("pos_error_norm_m", 9) <= 0.003 and err.get("rot_error_norm_rad", 9) <= 0.05
        stable = stable + 1 if ok else 0
        if stable >= stable_frames:
            return dict(settled=True, frames=frame)
    return dict(settled=False, frames=max_frames)


async def capture(out_dir):
    """Hide the IK target marker, wait for the arm to settle, render, save rgb_left.png + depth_axial_left.npy."""
    import omni.kit.app, omni.usd
    from pxr import Usd, UsdGeom
    from PIL import Image
    await _setup()
    reg = _registry(); stage = omni.usd.get_context().get_stage(); app = omni.kit.app.get_app()
    os.makedirs(out_dir, exist_ok=True)
    target = stage.GetPrimAtPath(IK_TARGET_PATH); img = None; prev = None
    if target.IsValid() and target.IsA(UsdGeom.Imageable):
        img = UsdGeom.Imageable(target); prev = img.GetVisibilityAttr().Get()
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            img.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    try:
        settle = await _wait_for_ik_settle()
        await _rendering(reg["render_product"], 5)
        watchdog = await render_step(4)
        rgb = depth = None
        for _ in range(20):                        # right after switching a camera on, frames can be empty/black
            r = np.asarray(reg["rgb"].get_data()); d = np.asarray(reg["depth"].get_data(), dtype=np.float32)
            if r.ndim == 3 and d.ndim == 2 and r[:, :, :3].mean() >= 1.0:
                rgb, depth = r[:, :, :3].astype(np.uint8), d
                break
            await render_step(2)
        if rgb is None:
            raise RuntimeError("D455 left camera returned no image after 20 render steps")
    finally:
        _enable(reg["render_product"], False)
        if img is not None:
            with Usd.EditContext(stage, stage.GetSessionLayer()):
                img.GetVisibilityAttr().Set(prev or UsdGeom.Tokens.inherited)
    Image.fromarray(rgb).save(os.path.join(out_dir, "rgb_left.png"))
    np.save(os.path.join(out_dir, "depth_axial_left.npy"), depth)
    return dict(out_dir=out_dir, settle=settle, watchdog=watchdog, rgb_mean=round(float(rgb.mean()), 1))


async def snapshot(camera_path, png_path, size=(960, 540)):
    """One RGB picture from any camera prim. Its render product is kept (not re-created per picture) and only
    renders during the picture."""
    import omni.replicator.core as rep
    from PIL import Image
    await _setup()
    extra = _registry().setdefault("extra", {})
    if camera_path not in extra:
        rp = rep.create.render_product(camera_path, size)
        ann = rep.AnnotatorRegistry.get_annotator("rgb"); ann.attach(rp)
        extra[camera_path] = (rp, ann)
    rp, ann = extra[camera_path]
    try:
        await _rendering(rp, 10)
        img = None
        for _ in range(20):
            await render_step(4)
            a = np.asarray(ann.get_data())
            if a.ndim == 3 and a[..., :3].mean() >= 1.0:
                img = a[..., :3].astype(np.uint8); break
        if img is None:
            raise RuntimeError(f"{camera_path} returned no image after 20 render steps")
        Image.fromarray(img).save(png_path)
    finally:
        _enable(rp, False)


def camera_frame(stage, camera_path):
    """World pose of a camera prim: position_m and right/up/forward unit axes (cameras look down local -Z)."""
    import omni.timeline
    prim = stage.GetPrimAtPath(camera_path)
    if not prim.IsValid():
        raise RuntimeError(f"Camera prim not found: {camera_path}")
    if omni.timeline.get_timeline_interface().is_playing():
        from isaacsim.core.experimental.prims import XformPrim
        pos, quat = _cached_prim(XformPrim, camera_path).get_world_poses()
        position = pos.numpy()[0].astype(np.float64)
        w, x, y, z = quat.numpy()[0].astype(np.float64)
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        right, up, back = R[:, 0], R[:, 1], R[:, 2]
    else:
        from pxr import Usd, UsdGeom
        m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        position = np.asarray(m.ExtractTranslation(), dtype=np.float64)
        r = m.ExtractRotationMatrix()
        axes = np.array([[r[i][j] for j in range(3)] for i in range(3)], dtype=np.float64)
        right, up, back = axes[0], axes[1], axes[2]
    n = lambda v: (v / np.linalg.norm(v)).tolist()
    return dict(position_m=position.tolist(), right=n(right), up=n(up), forward=n(-back))


def camera_intrinsics(stage, camera_path):
    """Pinhole intrinsics from the USD camera's focal length and horizontal aperture (square pixels)."""
    from pxr import UsdGeom
    cam = UsdGeom.Camera(stage.GetPrimAtPath(camera_path))
    fov = 2 * math.atan(cam.GetHorizontalApertureAttr().Get() / (2 * cam.GetFocalLengthAttr().Get()))
    fx = (WIDTH / 2.0) / math.tan(fov / 2.0)
    return dict(fx=fx, fy=fx, cx=WIDTH / 2.0, cy=HEIGHT / 2.0)
