"""Isaac Sim 6's NumPy rejects 2-D np.cross (grasp/planner.py:85 -> ValueError). Exec once per Isaac Sim session
before detect_pick() until planner.py is fixed. Does not touch project files."""
import numpy as np
if not getattr(np.cross, "_lab_shim", False):
    _orig = np.cross
    def _cross(a, b, *args, **kw):
        a = np.asarray(a); b = np.asarray(b)
        if a.shape[-1] == 2 and b.shape[-1] == 2:
            return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
        return _orig(a, b, *args, **kw)
    _cross._lab_shim = True
    np.cross = _cross
