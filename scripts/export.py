"""Export trained weights to CoreML and ONNX."""
import argparse
from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument("--weights", default="runs/hardhat/weights/best.pt")
p.add_argument("--imgsz", type=int, default=640)
a = p.parse_args()
m = YOLO(a.weights)
print(m.export(format="coreml", imgsz=a.imgsz, nms=True))
print(m.export(format="onnx", imgsz=a.imgsz, simplify=True))
