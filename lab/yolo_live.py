"""lab/yolo_live.py -- minimal low-latency YOLO server for the video mode (own port, no overlay/compressed files).
Run with perception/.venv/bin/python lab/yolo_live.py [--port 18778]
Request  {"cmd":"detect","rgb_path":<png in /dev/shm>,"masks_path":<npy out>,"conf":0.05}
Reply    {"ok":true,"detections":[{id,class_name,conf,bbox_xyxy,mask_pixels}],"predict_ms":..}   masks -> masks_path (uint8 N,H,W)
Same model, imgsz, retina_masks and mask-IoU de-duplication as perception/yolo_server.py."""
import argparse, json, socket, sys, time, traceback
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_MTIME = Path(__file__).stat().st_mtime
CFG = json.loads((ROOT / "perception/config.json").read_text(encoding="utf-8"))


def recv_msg(conn):
    head = b""
    while len(head) < 4:
        c = conn.recv(4 - len(head))
        if not c:
            return None
        head += c
    n = int.from_bytes(head, "big"); buf = bytearray()
    while len(buf) < n:
        c = conn.recv(n - len(buf))
        if not c:
            return None
        buf.extend(c)
    return json.loads(buf)


def send_msg(conn, obj):
    d = json.dumps(obj).encode(); conn.sendall(len(d).to_bytes(4, "big") + d)


def detect(model, req):
    t0 = time.perf_counter()
    res = model.predict(req["rgb_path"], conf=req.get("conf", CFG["conf"]), imgsz=CFG["imgsz"], retina_masks=True, verbose=False)[0]
    ms = (time.perf_counter() - t0) * 1000
    h, w = res.orig_shape; dets = []; masks = np.zeros((0, h, w), bool)
    if res.boxes is not None and len(res.boxes):
        if res.masks is not None:
            masks = res.masks.data.cpu().numpy().astype(bool)
        order = sorted(range(len(res.boxes)), key=lambda k: -float(res.boxes.conf[k])); keep = []
        for k in order:
            if k < len(masks) and any((masks[k] & masks[j]).sum() > 0.7 * (masks[k] | masks[j]).sum() for j in keep):
                continue
            keep.append(k)
        masks = masks[keep] if len(masks) else masks
        for i, k in enumerate(keep):
            c = int(res.boxes.cls[k])
            dets.append(dict(id=i + 1, class_id=c, class_name=res.names[c], conf=round(float(res.boxes.conf[k]), 4),
                             bbox_xyxy=[round(float(v), 1) for v in res.boxes.xyxy[k].tolist()],
                             mask_pixels=int(masks[i].sum()) if i < len(masks) else 0))
    np.save(req["masks_path"], masks.astype(np.uint8))
    return dict(ok=True, detections=dets, predict_ms=round(ms, 1), image_hw=[h, w])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=18778); args = ap.parse_args()
    from ultralytics import YOLO
    model = YOLO(str(ROOT / CFG["model"]))
    model.predict(np.zeros((480, 640, 3), np.uint8), imgsz=CFG["imgsz"], verbose=False)       # warm-up
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", args.port)); srv.listen(4); print("READY", flush=True)
    while True:
        conn, _ = srv.accept()
        with conn:
            try:
                req = recv_msg(conn)
                if req is None:
                    continue
                if req.get("cmd") == "ping":
                    send_msg(conn, dict(ok=True, src_mtime=SRC_MTIME))
                elif req.get("cmd") == "detect":
                    send_msg(conn, detect(model, req))
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
