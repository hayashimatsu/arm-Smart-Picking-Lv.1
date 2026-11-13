"""demo_start.py -- Isaac Sim Script Editor: open scenes/lab_cell_g1.usda, then open this file and press Ctrl+Enter ONCE.
It only gets the arm ready (arm IK, wrist D455, YOLO server; about 10 s) and prints "demo_start: READY". Nothing moves until you call:

    layout(seed, 8)     8 parts (flange_round / bracket_mid / bracket_end, mixed) dropped into the basket; same seed = same layout.
    sort()              the arm sorts the basket by itself: flange_round -> red, bracket_mid -> blue, bracket_end -> yellow bin,
                        until view A counts nothing left. Report: outputs/sort/<time>_s<seed>_n<count>/result.json.

Both return at once and run in the background; only one action runs at a time. Usage: README.md.
"""
import os
import omni.usd

_root = os.path.dirname(os.path.dirname(os.path.abspath(omni.usd.get_context().get_stage().GetRootLayer().realPath)))
_api = os.path.join(_root, "lab", "api2.py")
if not os.path.isfile(_api):
    raise RuntimeError(f"Open this project's scene (scenes/lab_cell_g1.usda) first: {_api} not found")
exec(compile(open(os.path.join(_root, "lab", "shim_numpy_cross.py"), encoding="utf-8").read(), "shim_numpy_cross.py", "exec"), globals())
__file__ = _api
exec(compile(open(_api, encoding="utf-8").read(), _api, "exec"), globals())
