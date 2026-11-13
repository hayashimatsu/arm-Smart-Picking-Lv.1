# arm-detect-and-pick: how the blocks case worked (notes)

> 11/12: the entry is now `demo_start.py` (was `demo_start2.py`) with the lab cell `scenes/lab_cell_g1.usda`: `layout(seed, 8)` then `sort()` -- see the root `README.md`. This page keeps the blocks-case usage (`layout_blocks()` / `detect_pick()`, scene `arm310d_d455_ik_demo.usd`) as a record.


## Goal

The NHC12 arm with the wrist D455 picks blocks of one type into the basket, with these properties:
- **Video instead of two photos.** The wrist D455 streams; YOLO + depth run on the video, so class, confidence, world position and distance are tracked continuously. The annotated video has one list in the top-left corner that changes as the arm moves.
- **Continuous motion.** One trajectory per leg, no stop except at the grasp and the place.
- **Grasp without stop-and-go, place without dropping.** One continuous close plus a firm squeeze; the block is lowered onto the basket floor and released there.

## Usage

**Every time after starting Isaac Sim**

1. Open `scenes/arm310d_d455_ik_demo.usd` from the project folder (no need to press Play).
2. Open `demo_start.py` (project root) in the Script Editor and press Ctrl+Enter **once**. It only gets the arm ready (about 10 s) and prints `demo_start: READY`. Nothing moves.
3. Call:

```python
layout("sphere")          # (testing) 10 new random blocks, every size of that type included; basket emptied; the original red cube is removed
detect_pick("sphere")     # pick every sphere into the basket, video mode
```

| Call | What it does | Result |
|---|---|---|
| `layout("sphere")` | New random blocks (physics on). `layout()` = any 10 random blocks. The red `object` cube of the original scene is removed for the session (never saved). | prints how many of each type |
| `detect_pick("sphere")` | `"cube"` / `"cuboid"` / `"cylinder"` / `"sphere"` (default `"cube"`). Scans from the photo pose for about 1 s, then picks one by one. | `outputs/lab/live/run_<time>/index.html` |

- Both calls return at once and run in the background; the console prints `started` and `done` (with a short result). Only one action runs at a time.
- The Script Editor does not reload a file that changed on disk. `lab/api2.py` and the modules it uses are read from disk on every `detect_pick()`, so the newest code runs; only `demo_start.py` itself is the loader.
- If Isaac Sim gets slow, reopen `scenes/arm310d_d455_ik_demo.usd` (no save) and run `demo_start.py` again.

## What you get

`outputs/lab/live/run_<time>/`:
- `index.html`: the video, one row per pick (class, confidence, held at the basket, seconds from arriving at the block to releasing it), and which blocks moved.
- `wrist.mp4`: the annotated wrist video. Top-left list: `#id class confidence x y z (world, m) d (distance from the camera, m)`, updated with every video frame.
- `result.json`: every number (events with sim-seconds, grip data, tug test, tracks vs stage truth). `frames/` has the single frames.

## How it works (short)

1. **Scan.** The arm goes to the photo pose A once and waits until it is still. The video tracker collects about 1 s of frames. Tracks only update from frames where the camera is nearly still and at a training-like height; a moving camera blurs the path-traced image.
2. **Choose.** The nearest graspable block of the target type among the tracks (grasp direction and opening as in the original planner).
3. **Fly.** One continuous path to above the block, then down. The path end is corrected live from the track while the hand is still high. The gripper is pre-shaped for the block only when the hand is at travel height (otherwise it closes on the block it has just released).
4. **Close.** Jump to just before contact, ramp until the finger lags the command (contact), then squeeze 5° more and wait 12 frames so the block turns flat between the pads.
5. **Check.** Closed width must be between the block's short side and the planned width (else wrong object: open, go up, scan again). Tug test: lift 12 mm; the finger must not move and must keep pushing, else squeeze harder (up to 3 tests).
6. **Place.** Travel at the travel height over the basket, lower onto the floor of a grid cell, open, wait until all gripper joints are open, leave slowly for the first 30 mm.

## Known limits (preliminary)

- The positions used for control come from the still scan (and re-scans after a failed grasp); while the arm moves the video keeps showing live numbers, but they are not used to steer except the live end-point correction in step 3.
- Blocks too close together ("fingers blocked by neighbours") are skipped, so a run can stop with some blocks left.
- Tested through the MCP with `demo_start2.py` (now `demo_start.py`) loaded; **not** yet run by pressing Ctrl+Enter in the Script Editor by a person.
- Success rates are being re-measured on the clean flow (see `lab/user-review/feedback-2.md`, steps 2 and 4); the numbers in `STATUS.md` are the latest.

## Files

| Path | Content |
|---|---|
| `demo_start.py` (project root) | Entry point: Ctrl+Enter once. A short loader for `lab/api2.py` |
| `lab/api2.py` | `layout()` and `detect_pick()` |
| `lab/live_pick.py` | The pick loop (scan, choose, fly, close, check, place), the HTML report |
| `lab/live_perception.py` | Video mode: stream, YOLO + depth per frame, tracker, annotated frames |
| `lab/fast_pick.py` | Continuous trajectory, gripper close / release, parameters (`P`) |
| `lab/yolo_live.py` | Low-latency YOLO server for the video (port 18778, started automatically; log `outputs/lab/yolo_live.log`) |
| `lab/user-review/` | The user's reviews and the task list |
| `lab/debug_release.py`, `lab/release_debug_report.py`, `lab/batch2.py`, `lab/diag_*.py`, `lab/measure_speed.py` | Test and diagnosis tools (not needed for normal use) |
| `outputs/lab/` | Everything the lab code writes (symbolic link to the result disk) |
