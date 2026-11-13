"""YOLO inference server (runs inside the YOLO venv, not inside Isaac Sim).

Usage: <venv>/bin/python perception/yolo_server.py [--port 18777]
Started automatically by one_click.py; can also be run manually in a terminal for debugging.

Protocol (listens on 127.0.0.1 only): 4-byte big-endian length + UTF-8 JSON.
  {"cmd": "ping"}                                  -> {"ok": true}
  {"cmd": "detect", "rgb_path": ..., "out_dir": ...} -> {"ok": true, "detections": [...]}
detect writes masks.npz (N x H x W bool) and overlay.png (contours + ids) into out_dir.
The image is passed as a file path and read by ultralytics itself (BGR), matching training-time loading.
"""
import argparse, json, socket, sys, time, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_MTIME = Path(__file__).stat().st_mtime
CFG = json.loads((ROOT / "perception/config.json").read_text(encoding="utf-8"))
COLORS = [tuple(c) for c in CFG["palette_bgr"]]   # shared with the HTML report (one_click.py)


def recv_msg(conn):
    head = b""
    while len(head) < 4:
        chunk = conn.recv(4 - len(head))
        if not chunk:
            return None
        head += chunk
    n = int.from_bytes(head, "big")
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return json.loads(buf)


def send_msg(conn, obj):
    data = json.dumps(obj).encode()
    conn.sendall(len(data).to_bytes(4, "big") + data)


def draw_overlay(img, dets, masks):
    """Draw mask fill + contour + a small id marker per object, plus a top-left legend
    listing id/class/confidence, instead of per-object text labels (which overlap when
    objects sit close together)."""
    import cv2
    import numpy as np
    over = img.copy()
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
    for i, d in enumerate(dets):
        col = COLORS[(d["id"] - 1) % len(COLORS)]   # color follows the object id, so views match
        m = masks[i].astype(np.uint8) if i < len(masks) else None
        if m is not None:
            over[m > 0] = (0.5 * over[m > 0] + 0.5 * np.array(col)).astype(np.uint8)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(over, cnts, -1, col, 2)
            ys, xs = np.nonzero(m)
            cx, cy = int(xs.mean()), int(ys.mean())
            obj_r = min(xs.max() - xs.min(), ys.max() - ys.min()) / 2
        else:
            x1, y1, x2, y2 = d["bbox_xyxy"]
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            obj_r = min(x2 - x1, y2 - y1) / 2
        mark = str(d["id"])
        (mw, mh), mbase = cv2.getTextSize(mark, font, scale, thick)
        # keep the marker small relative to the object so it never hides a small block
        r = int(np.clip(obj_r * 0.5, 7, max(mw, mh) // 2 + 3))
        cv2.circle(over, (cx, cy), r, (0, 0, 0), -1)
        cv2.circle(over, (cx, cy), r, col, 1, cv2.LINE_AA)
        cv2.putText(over, mark, (cx - mw // 2, cy + mh // 2), font, scale, (255, 255, 255), thick, cv2.LINE_AA)

    order = sorted(range(len(dets)), key=lambda k: dets[k]["id"])   # legend sorted by id
    rows = [dets[k].get("label") or f'{dets[k]["id"]} {dets[k]["class_name"]} {dets[k]["conf"]:.2f}' for k in order]
    if rows:
        pad, sw, lh = 6, 14, 18
        tw = max(cv2.getTextSize(r, font, scale, thick)[0][0] for r in rows)
        box_w, box_h = pad * 2 + sw + 6 + tw, pad * 2 + lh * len(rows)
        roi = over[pad:pad + box_h, pad:pad + box_w]
        roi[:] = (0.35 * roi + 0.65 * np.zeros_like(roi)).astype(np.uint8)
        cv2.rectangle(over, (pad, pad), (pad + box_w, pad + box_h), (255, 255, 255), 1, cv2.LINE_AA)
        for i, row in enumerate(rows):
            y0 = pad + i * lh
            col = COLORS[(dets[order[i]]["id"] - 1) % len(COLORS)]
            cv2.rectangle(over, (pad + 4, y0 + 4), (pad + 4 + sw, y0 + 4 + sw), col, -1)
            cv2.putText(over, row, (pad + sw + 10, y0 + lh - 5), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return over


def detect(model, req):
    import cv2
    import numpy as np
    t0 = time.perf_counter()
    res = model.predict(req["rgb_path"], conf=req.get("conf", CFG["conf"]), imgsz=CFG["imgsz"],
                        retina_masks=True, verbose=False)[0]
    ms = (time.perf_counter() - t0) * 1000
    h, w = res.orig_shape
    dets, masks = [], np.zeros((0, h, w), bool)
    if res.boxes is not None and len(res.boxes):
        if res.masks is not None:
            masks = res.masks.data.cpu().numpy().astype(bool)
        # YOLO26 is an NMS-free end-to-end model; the same object occasionally gets two boxes -> dedupe by mask IoU>0.7 (keep the higher-confidence one)
        order = sorted(range(len(res.boxes)), key=lambda k: -float(res.boxes.conf[k]))
        keep = []
        for k in order:
            if k < len(masks) and any((masks[k] & masks[j]).sum() > 0.7 * (masks[k] | masks[j]).sum() for j in keep):
                continue
            keep.append(k)
        masks = masks[keep] if len(masks) else masks
        for i, k in enumerate(keep):
            c = int(res.boxes.cls[k])
            dets.append(dict(id=i + 1, class_id=c, class_name=res.names[c],
                             conf=round(float(res.boxes.conf[k]), 4),
                             bbox_xyxy=[round(float(v), 1) for v in res.boxes.xyxy[k].tolist()],
                             mask_pixels=int(masks[i].sum()) if i < len(masks) else 0))
    out = Path(req["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "masks.npz", masks=masks)
    img = cv2.imread(req["rgb_path"])
    over = draw_overlay(img, dets, masks)
    cv2.imwrite(str(out / "overlay.png"), over)
    return dict(ok=True, detections=dets, predict_ms=round(ms, 1), image_hw=[h, w])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=CFG["port"])
    args = ap.parse_args()
    from ultralytics import YOLO
    model = YOLO(str(ROOT / CFG["model"]))
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", args.port)); srv.listen(4)
    print("READY", flush=True)
    while True:
        conn, _ = srv.accept()
        with conn:
            try:
                req = recv_msg(conn)
                if req is None:
                    continue
                if req.get("cmd") == "ping":
                    send_msg(conn, dict(ok=True, model=CFG["model"], src_mtime=SRC_MTIME))
                elif req.get("cmd") == "detect":
                    send_msg(conn, detect(model, req))
                elif req.get("cmd") == "draw":
                    import cv2, numpy as np
                    masks = np.load(req["masks_path"])["masks"]
                    cv2.imwrite(req["out_path"], draw_overlay(cv2.imread(req["rgb_path"]), req["detections"], masks))
                    send_msg(conn, dict(ok=True))
                elif req.get("cmd") == "quit":
                    send_msg(conn, dict(ok=True)); return
                else:
                    send_msg(conn, dict(ok=False, error="unknown cmd"))
            except Exception as exc:
                traceback.print_exc()
                try:
                    send_msg(conn, dict(ok=False, error=f"{type(exc).__name__}: {exc}"))
                except Exception:
                    pass


if __name__ == "__main__":
    sys.exit(main())
