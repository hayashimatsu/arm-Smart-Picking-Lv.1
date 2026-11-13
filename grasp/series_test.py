"""Test series (Script Editor): for each class -> fresh layout (stop, place blocks incl. every catalog item of
that class, play) -> pick all of that class. Summary: outputs/pick_series/<time>/summary.json"""
import asyncio, builtins, datetime, json, os, traceback
import omni.usd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(omni.usd.get_context().get_stage().GetRootLayer().realPath)))
CLASSES = ["cuboid", "cylinder", "sphere"]
N_BLOCKS = 10


def _load(rel):
    p = os.path.join(ROOT, rel); g = dict(NO_AUTORUN=True, __file__=p)
    exec(compile(open(p, encoding="utf-8").read(), p, "exec"), g); return g


async def relayout(cls, seed, n=N_BLOCKS):
    import omni.timeline, omni.kit.app
    tl = omni.timeline.get_timeline_interface(); app = omni.kit.app.get_app()
    tl.stop()                                   # physics reset: blocks/basket content go back to authored state
    for _ in range(10):
        await app.next_update_async()
    placed = _load("perception/place_test_blocks.py")["place_test_blocks"](n=n, seed=seed, include_class=cls)
    tl.play()
    for _ in range(120):                        # let blocks settle
        await app.next_update_async()
    return placed


async def series(classes, seed0=100):
    out = os.path.join(ROOT, "outputs", "pick_series", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out, exist_ok=True); rows = []
    for i, cls in enumerate(classes):
        try:
            placed = await relayout(cls, seed0 + i)
            doc = await _load("grasp/pick.py")["Picker"]().run(cls, 12)
            d = doc["diagnosis"]
            rows.append(dict(cls=cls, seed=seed0 + i, placed=[p["name"] for p in placed], expected=d["expected_truth"],
                             in_basket=d["in_basket"], issues=d["issues"], attempts=len([c for c in doc["cycles"] if not c.get("done")]),
                             total_s=doc["total_s"], report=os.path.join(doc["run_dir"], "index.html")))
        except Exception:
            rows.append(dict(cls=cls, error=traceback.format_exc()))
        json.dump(dict(state="running", rows=rows), open(os.path.join(out, "summary.json"), "w"), indent=1, default=str)
    json.dump(dict(state="done", rows=rows), open(os.path.join(out, "summary.json"), "w"), indent=1, default=str)


if not globals().get("NO_AUTORUN"):
    builtins._series_task = asyncio.ensure_future(series(CLASSES))
    print("series scheduled", CLASSES)
