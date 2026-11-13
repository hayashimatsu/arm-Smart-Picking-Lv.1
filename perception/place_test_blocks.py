"""Randomly place a few test blocks on the table workspace (session layer only, never saved into the stage file).

Running this file in the Script Editor = clear old test blocks and place N new ones (default 8).
Can also call place_test_blocks(n=8, seed=None) / clear_test_blocks().
Sizes: CATALOG_GRASP (all edges <= 60 mm, with physics) by default; CATALOG_VISION = the 40-80 mm perception catalog, colors are random, matching training-time generation.
"""
import random
PARENT = "/World/TestBlocks"
TABLE_Z = 0.5952
WS_X, WS_Y = (-0.72, -0.36), (-0.15, 0.12)   # shrink the workspace slightly so blocks don't fall out of frame
CATALOG_VISION = [  # (name, class, size x,y,z mm) -- 40-80 mm catalog used for the perception accuracy batches
    ("cube_40", "cube", (40, 40, 40)), ("cube_60", "cube", (60, 60, 60)), ("cube_80", "cube", (80, 80, 80)),
    ("cuboid_2_1_1", "cuboid", (80, 40, 40)), ("cuboid_4_2_1", "cuboid", (80, 40, 20)), ("cuboid_3_2_1", "cuboid", (80, 53.3, 26.7)),
    ("cylinder_d40_h60", "cylinder", (40, 40, 60)), ("cylinder_d60_h40", "cylinder", (60, 60, 40)),
    ("sphere_50", "sphere", (50, 50, 50)), ("sphere_70", "sphere", (70, 70, 70)),
    ("cylinder_d40_l80_lying", "cylinder", (80, 40, 40)), ("cylinder_d50_l70_lying", "cylinder", (70, 50, 50)),
]
# Grasp catalog (user decision 2026-09-29): every edge / diameter <= 60 mm so the 85 mm gripper always fits.
CATALOG_GRASP = [
    ("cube_40", "cube", (40, 40, 40)), ("cube_50", "cube", (50, 50, 50)), ("cube_60", "cube", (60, 60, 60)),
    ("cuboid_2_1_1", "cuboid", (60, 30, 30)), ("cuboid_3_2_1", "cuboid", (60, 40, 20)), ("cuboid_2_2_1", "cuboid", (60, 60, 30)),
    ("cylinder_d40_h60", "cylinder", (40, 40, 60)), ("cylinder_d50_h40", "cylinder", (50, 50, 40)), ("cylinder_d60_h30", "cylinder", (60, 60, 30)),
    ("cylinder_d40_l60_lying", "cylinder", (60, 40, 40)),
    ("sphere_40", "sphere", (40, 40, 40)), ("sphere_50", "sphere", (50, 50, 50)), ("sphere_60", "sphere", (60, 60, 60)),
]
CATALOG = CATALOG_GRASP


MASS_KG = 0.05
FRICTION = 1.0


def _physics_material(stage):
    from pxr import UsdShade, UsdPhysics
    path = PARENT + "/PhysicsMaterial"
    if not stage.GetPrimAtPath(path).IsValid():
        m = UsdShade.Material.Define(stage, path)
        api = UsdPhysics.MaterialAPI.Apply(m.GetPrim())
        api.CreateStaticFrictionAttr(FRICTION); api.CreateDynamicFrictionAttr(FRICTION); api.CreateRestitutionAttr(0.0)
    return UsdShade.Material(stage.GetPrimAtPath(path))


def _add_physics(stage, prim):
    """Rigid body + collision + mass + high-friction material, so the gripper can actually pick it up."""
    from pxr import UsdPhysics, UsdShade
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(MASS_KG)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(_physics_material(stage), UsdShade.Tokens.weakerThanDescendants, "physics")


def _session_edit_target(stage):
    """PhysX writes simulated poses to the stage's current edit target. The blocks live in the session layer,
    so the edit target must be the session layer too -- otherwise the poses land in the root layer, the
    session-layer opinion wins, and the rendered picture keeps showing the block where it started
    (found 2026-09-29: the cube was physically in the basket while the camera still saw it on the table)."""
    from pxr import Usd
    stage.SetEditTarget(Usd.EditTarget(stage.GetSessionLayer()))


def _clean_root_layer(stage):
    """Remove any over-specs physics wrote into the root layer for the test blocks (keeps the scene file clean)."""
    from pxr import Sdf
    root = stage.GetRootLayer()
    if root.GetPrimAtPath(PARENT):
        with Sdf.ChangeBlock():
            edit = Sdf.BatchNamespaceEdit(); edit.Add(PARENT, Sdf.Path.emptyPath)
            root.Apply(edit)


def clear_test_blocks():
    import omni.usd
    from pxr import Usd
    stage = omni.usd.get_context().get_stage()
    _clean_root_layer(stage)
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        if stage.GetPrimAtPath(PARENT).IsValid():
            stage.RemovePrim(PARENT)


def place_test_blocks(n=8, seed=None, catalog=None, physics=True, include_class=None):
    import omni.usd
    from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf
    rng = random.Random(seed)
    clear_test_blocks()
    stage = omni.usd.get_context().get_stage()
    cat = catalog or CATALOG
    if include_class:          # every catalog item of this class + random others up to n (pick tests)
        must = [c for c in cat if c[1] == include_class]
        rest = [c for c in cat if c[1] != include_class]
        picks = must + rng.sample(rest, max(0, min(n - len(must), len(rest))))
        rng.shuffle(picks)
    else:
        picks = rng.sample(cat, min(n, len(cat)))
    placed, out = [], []
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        UsdGeom.Xform.Define(stage, PARENT)
        for i, (name, cls, (sx, sy, sz)) in enumerate(picks):
            r = max(sx, sy) / 2000 * 1.1
            for _ in range(200):
                x, y = rng.uniform(*WS_X), rng.uniform(*WS_Y)
                if all((x - px) ** 2 + (y - py) ** 2 > (r + pr + 0.01) ** 2 for px, py, pr in placed):
                    break
            else:
                continue
            placed.append((x, y, r))
            path = f"{PARENT}/{name}"
            if cls in ("cube", "cuboid"):
                g = UsdGeom.Cube.Define(stage, path); g.CreateSizeAttr(1.0)
            elif cls == "sphere":
                g = UsdGeom.Sphere.Define(stage, path); g.CreateRadiusAttr(sx / 2000)
            else:
                g = UsdGeom.Cylinder.Define(stage, path)
                if name.endswith("_lying"):
                    g.CreateRadiusAttr(sz / 2000); g.CreateHeightAttr(sx / 1000); g.CreateAxisAttr("X")
                else:
                    g.CreateRadiusAttr(sx / 2000); g.CreateHeightAttr(sz / 1000); g.CreateAxisAttr("Z")
            xf = UsdGeom.Xformable(g.GetPrim())
            xf.AddTranslateOp().Set(Gf.Vec3d(x, y, TABLE_Z + sz / 2000))
            xf.AddRotateZOp().Set(rng.uniform(0, 360))
            if cls in ("cube", "cuboid"):
                xf.AddScaleOp().Set(Gf.Vec3f(sx / 1000, sy / 1000, sz / 1000))
            mat = UsdShade.Material.Define(stage, path + "/Mat")
            sh = UsdShade.Shader.Define(stage, path + "/Mat/Shader"); sh.CreateIdAttr("UsdPreviewSurface")
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*(rng.uniform(.05, .95) for _ in range(3))))
            sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rng.uniform(.2, .9))
            mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
            UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(mat)
            if physics:
                _add_physics(stage, g.GetPrim())
            out.append(dict(name=name, cls=cls, xy=[round(x, 3), round(y, 3)]))
    if physics:
        _session_edit_target(stage)
    print(out)
    return out


if not globals().get("NO_AUTORUN"):
    place_test_blocks()
