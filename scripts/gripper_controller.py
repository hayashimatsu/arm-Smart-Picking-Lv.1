"""Gripper angle command and observation for the Robotiq 2F-85.

Drives articulation index 6 (`finger_joint`); the other five gripper DOFs
are PhysX mimic joints and follow on their own.
"""

from __future__ import annotations

import math

ARTICULATION_PATH = "/World/ArmWithHandOnly"
GRIPPER_DOF_INDEX = 6
FINGER_A_PATH = "/World/ArmWithHandOnly/Robotiq_2F_85/Robotiq_2F_85/left_inner_finger"
FINGER_B_PATH = "/World/ArmWithHandOnly/Robotiq_2F_85/Robotiq_2F_85/right_inner_finger"
DEG_MIN = 0.0
DEG_MAX = 47.0

_last_target_deg = None
_SCRIPTS_DIR = __import__("os").path.dirname(__import__("os").path.abspath(__file__))   # loaded with __file__ set


def _cached_prim(cls, path):
    """Shared cache, see scripts/prim_cache.py (avoids leaking physics views)."""
    import builtins as _b, os as _os
    fn = getattr(_b, "_nhc_cached_prim_fn", None)
    if fn is None:
        p = _os.path.join(_SCRIPTS_DIR, "prim_cache.py"); g = {"__file__": p}
        exec(compile(open(p, encoding="utf-8").read(), p, "exec"), g)
        fn = g["cached_prim"]; setattr(_b, "_nhc_cached_prim_fn", fn)
    return fn(cls, path)



def finger_origin_separation_mm(a, b):
    """Euclidean distance between two world-frame points, in mm.

    Args:
        a: World xyz point (metres).
        b: World xyz point (metres).

    Returns:
        Distance in mm.

    Note:
        This is NOT the pad gap. It is the distance between the
        left_inner_finger/right_inner_finger link origins, confirmed by
        direct human observation to be INVERSELY correlated with the real
        pad opening (origin distance grows as the pads visually close).
        Useful only as a monotonic indicator of finger_joint angle, never
        as a measure of clearance or opening width.
    """
    ax, ay, az = a
    bx, by, bz = b
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2) * 1000.0


def gripper_set_angle_deg(deg):
    """Command the gripper's finger_joint to a target angle.

    Args:
        deg: Target angle in degrees, within [DEG_MIN, DEG_MAX].

    Returns:
        dict with target_deg and target_rad.
    """
    if deg < DEG_MIN or deg > DEG_MAX:
        raise ValueError(
            f"target {deg} deg outside gripper travel [{DEG_MIN}, {DEG_MAX}]"
        )
    import numpy as np
    from isaacsim.core.experimental.prims import Articulation

    global _last_target_deg
    art = _cached_prim(Articulation, ARTICULATION_PATH)
    target_rad = deg * (math.pi / 180.0)
    art.set_dof_position_targets(np.array([target_rad]), dof_indices=[GRIPPER_DOF_INDEX])
    _last_target_deg = deg
    return {"target_deg": deg, "target_rad": target_rad}


def gripper_status():
    """Current gripper joint angle, target tracking error, and fingertip poses.

    Returns:
        dict with target_deg, achieved_deg, achieved_rad,
        finger_origin_separation_mm, tracking_error_rad, fingertip_a_world,
        fingertip_b_world.
    """
    from isaacsim.core.experimental.prims import Articulation, XformPrim
    import isaacsim.core.experimental.utils.backend as backend_utils

    art = _cached_prim(Articulation, ARTICULATION_PATH)
    positions = art.get_dof_positions().numpy()[0]
    achieved_rad = float(positions[GRIPPER_DOF_INDEX])
    achieved_deg = achieved_rad * (180.0 / math.pi)

    fa = _cached_prim(XformPrim, FINGER_A_PATH)
    fb = _cached_prim(XformPrim, FINGER_B_PATH)
    with backend_utils.use_backend("fabric", raise_on_fallback=True):
        pa, _ = fa.get_world_poses()
        pb, _ = fb.get_world_poses()
    a_pos = pa.numpy()[0].tolist()
    b_pos = pb.numpy()[0].tolist()

    target_deg = _last_target_deg
    target_rad = target_deg * (math.pi / 180.0) if target_deg is not None else None
    tracking_error_rad = (
        (target_rad - achieved_rad) if target_rad is not None else None
    )

    return {
        "target_deg": target_deg,
        "achieved_deg": achieved_deg,
        "achieved_rad": achieved_rad,
        "finger_origin_separation_mm": finger_origin_separation_mm(a_pos, b_pos),
        "tracking_error_rad": tracking_error_rad,
        "fingertip_a_world": a_pos,
        "fingertip_b_world": b_pos,
    }
