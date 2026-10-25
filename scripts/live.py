"""Run the detector on a webcam (default), a video file, or a folder/glob of images.

    python scripts/live.py                      # webcam 0
    python scripts/live.py --source results/live-on-test-images.mp4 --save out.mp4 --no-show
    python scripts/live.py --source datasets/hardhat/images/test --save out.mp4 --no-show
    python scripts/live.py --source "my_photos/*.jpg" --save out.mp4 --no-show
"""
import argparse, glob, os, time
import cv2
from ultralytics import YOLO

COLORS = {0: (15, 72, 217), 1: (61, 163, 232)}  # BGR: vermilion = hardhat, amber = no-hardhat

p = argparse.ArgumentParser()
p.add_argument("--weights", default="runs/hardhat/weights/best.mlpackage")
p.add_argument("--source", default="0")
p.add_argument("--conf", type=float, default=0.4)
p.add_argument("--save", default=None)
p.add_argument("--no-show", action="store_true")
p.add_argument("--fps", type=float, default=10, help="playback rate when the source is a set of images")
a = p.parse_args()

model = YOLO(a.weights, task="detect")
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def image_list(src):
    if os.path.isdir(src):
        return sorted(f for f in glob.glob(os.path.join(src, "*")) if f.lower().endswith(IMG_EXT))
    if any(ch in src for ch in "*?["):
        return sorted(f for f in glob.glob(src) if f.lower().endswith(IMG_EXT))
    if src.lower().endswith(IMG_EXT):
        return [src]
    return None


images = image_list(a.source)
if images is not None:
    if not images:
        raise SystemExit(f"no images found in {a.source}")
    cap, src_fps = None, a.fps
    frames = (cv2.imread(f) for f in images)
else:
    cap = cv2.VideoCapture(int(a.source) if a.source.isdigit() else a.source)
    if not cap.isOpened():
        raise SystemExit(f"cannot open source {a.source}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25

    def _read():
        while True:
            ok, f = cap.read()
            if not ok:
                return
            yield f
    frames = _read()
writer, fps_avg, n = None, 0.0, 0
for frame in frames:
    if frame is None:
        continue
    t = time.perf_counter()
    r = model.predict(frame, conf=a.conf, verbose=False)[0]
    fps = 1 / (time.perf_counter() - t)
    n += 1
    fps_avg += (fps - fps_avg) / n
    for b in r.boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        c = int(b.cls)
        col = COLORS.get(c, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        label = f"{model.names[c]} {float(b.conf):.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), col, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(frame, f"{fps_avg:.0f} FPS", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (32, 20, 42), 2)
    if a.save:
        if writer is None:
            h, w = frame.shape[:2]
            if images is not None:
                frame_size = (w, h)
            writer = cv2.VideoWriter(a.save, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, h))
        if images is not None and frame.shape[1::-1] != frame_size:
            frame = cv2.resize(frame, frame_size)
        writer.write(frame)
    if not a.no_show:
        cv2.imshow("yolo-custom-detector", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
if cap is not None:
    cap.release()
if writer:
    writer.release()
print(f"frames={n} mean_fps={fps_avg:.1f}")
