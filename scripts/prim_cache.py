"""One Isaac Sim prim wrapper per (type, path), reused across calls.

Creating a new Articulation / XformPrim / RigidPrim on every call leaks physics views: measured 2026-09-30,
300 gripper_status() calls took the simulation from 46 to 31 FPS, and a long pick session fell to 10 FPS.
A cached wrapper is re-created only when the open stage changed or its physics view became invalid (Stop/Play).
"""
import builtins

_CACHE = "_nhc_prim_cache"


def cached_prim(cls, path):
    import omni.usd
    sid = omni.usd.get_context().get_stage_id()
    cache = getattr(builtins, _CACHE, None)
    if cache is None or cache.get("_stage_id") != sid:
        cache = {"_stage_id": sid}; setattr(builtins, _CACHE, cache)
    key = (cls.__name__, path)
    obj = cache.get(key)
    if obj is not None and hasattr(obj, "is_physics_tensor_entity_valid"):
        try:
            if not obj.is_physics_tensor_entity_valid():
                obj = None
        except Exception:
            obj = None
    if obj is None:
        obj = cls(path); cache[key] = obj
    return obj
