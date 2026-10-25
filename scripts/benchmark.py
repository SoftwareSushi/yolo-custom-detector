"""Measure inference speed of each exported format on the same test images.

Reports frames per second for model inference (preprocess + inference + postprocess),
averaged over the test split after a warm-up.
"""
import argparse, glob, json, platform, time
from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument("--images", default="datasets/hardhat/images/test")
p.add_argument("--warmup", type=int, default=10)
p.add_argument("--out", default="results/benchmark.json")
a = p.parse_args()
imgs = sorted(glob.glob(f"{a.images}/*.jpg"))
targets = [
    ("PyTorch (MPS)", "runs/hardhat/weights/best.pt", "mps"),
    ("PyTorch (CPU)", "runs/hardhat/weights/best.pt", "cpu"),
    ("CoreML", "runs/hardhat/weights/best.mlpackage", None),
    ("ONNX Runtime (CPU)", "runs/hardhat/weights/best.onnx", "cpu"),
]
res = {"machine": platform.machine(), "processor": platform.processor(), "images": len(imgs), "results": []}
for label, w, dev in targets:
    m = YOLO(w, task="detect")
    kw = {"device": dev} if dev else {}
    for im in imgs[: a.warmup]:
        m.predict(im, verbose=False, **kw)
    t = time.perf_counter()
    for im in imgs:
        m.predict(im, verbose=False, **kw)
    dt = time.perf_counter() - t
    r = {"format": label, "fps": round(len(imgs) / dt, 1), "ms_per_image": round(1000 * dt / len(imgs), 1)}
    print(r)
    res["results"].append(r)
import os; os.makedirs("results", exist_ok=True)
json.dump(res, open(a.out, "w"), indent=2)
