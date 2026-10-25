"""Fine-tune a pretrained YOLO model on Apple Silicon (MPS)."""
import argparse
from pathlib import Path
from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument("--data", default="datasets/hardhat/data.yaml")
p.add_argument("--model", default="yolo11n.pt")
p.add_argument("--epochs", type=int, default=30)
p.add_argument("--imgsz", type=int, default=640)
p.add_argument("--batch", type=int, default=16)
p.add_argument("--device", default="mps")
p.add_argument("--name", default="hardhat")
a = p.parse_args()
# An absolute project path keeps Ultralytics from nesting runs under runs/detect/.

model = YOLO(a.model)
model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
            project=str(Path("runs").resolve()), name=a.name, exist_ok=True, seed=0, workers=4, plots=True)
m = model.val(data=a.data, split="test", device=a.device, project=str(Path("runs").resolve()), name=a.name + "-test", exist_ok=True)
print(f"test mAP50={m.box.map50:.4f} mAP50-95={m.box.map:.4f} P={m.box.mp:.4f} R={m.box.mr:.4f}")
