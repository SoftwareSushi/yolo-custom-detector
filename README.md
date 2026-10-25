# yolo-custom-detector

Fine-tune a pretrained YOLO model to find hard hats, export it to CoreML and ONNX, and run it frame by frame on video. Everything here ran on one Apple M4 laptop, with training on its GPU through PyTorch MPS.

This is the companion repository for the Software Sushi article [Train a Custom YOLO Object Detector: Dataset to Inference](https://softwaresushi.com/blog-details/object-detection-using-yolo).

## What you get

| Step | Script | Output |
|---|---|---|
| 1. Get and convert the dataset | `scripts/prepare_dataset.py` | `datasets/hardhat/` in YOLO format and `data.yaml` |
| 2. Fine-tune | `scripts/train.py` | `runs/hardhat/weights/best.pt`, curves, confusion matrix |
| 3. Export | `scripts/export.py` | `best.mlpackage` (CoreML) and `best.onnx` |
| 4. Measure speed | `scripts/benchmark.py` | `results/benchmark.json` |
| 5. Run live | `scripts/live.py` | boxes drawn on a webcam stream or a video file |

## Quick start

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/prepare_dataset.py           # downloads about 1.1 GB once, keeps 2,000 / 400 / 200 images
python scripts/train.py                     # 30 epochs, device=mps
python scripts/export.py                    # CoreML + ONNX
python scripts/benchmark.py                 # speed of each format on the test images
python scripts/live.py                      # webcam 0; press q to quit
python scripts/live.py --source my.mp4 --save out.mp4 --no-show
```

To use the webcam, run `live.py` from a terminal app that has camera permission in **System Settings > Privacy & Security > Camera**. An SSH session has no camera access on macOS.

## Dataset

[Hard Hat Detection](https://huggingface.co/datasets/keremberke/hard-hat-detection), a Roboflow Universe export ("Hard Hats", resized to 640 × 640) with two classes, `hardhat` and `no-hardhat`, licensed **CC BY 4.0**. `prepare_dataset.py` converts its COCO annotations to YOLO text labels (`class x_center y_center width height`, normalized) and keeps a fixed random subset (seed 0):

| Split | Images | Boxes |
|---|---|---|
| train | 2,000 | 5,610 |
| val | 400 | 1,140 |
| test | 200 | 534 |

The images are not redistributed here. Credit for the images and labels goes to the dataset authors.

## Our results

These are from one run on 2026-10-25 on a MacBook Air with an Apple M4 and 16 GB of memory: YOLO11n from pretrained weights, 30 epochs, image size 640, batch 16, `device=mps`. Training took 0.685 hours (41 minutes).

Held-out test split (200 images, 534 boxes), `best.pt`:

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| all | 0.864 | 0.847 | 0.894 | 0.538 |
| hardhat | 0.879 | 0.814 | 0.853 | 0.452 |
| no-hardhat | 0.849 | 0.880 | 0.934 | 0.625 |

Validation split at epoch 30 (400 images, 1,140 boxes): precision 0.847, recall 0.834, mAP50 0.877, mAP50-95 0.510.

Speed on the 200 test images after 10 warm-up runs (`results/benchmark.json`). It includes reading each JPEG and pre- and postprocessing:

| Format | Images per second | ms per image |
|---|---|---|
| PyTorch (MPS) | 43.6 | 23.0 |
| PyTorch (CPU) | 37.8 | 26.4 |
| CoreML | 142.1 | 7.0 |
| ONNX Runtime (CPU) | 54.2 | 18.4 |

`results/` holds the training curves (`results.csv`, `results.png`), the confusion matrices and PR curves for the validation and test splits, and a short recording of `live.py` on held-out test images. That recording is a video built from still test images, not a webcam, because the run happened over SSH.

One run on one small subset is not a benchmark. Your numbers will change with the dataset, the seed and the machine.

## License

The code is AGPL-3.0, the same license as [Ultralytics](https://github.com/ultralytics/ultralytics), which it imports. The dataset is CC BY 4.0 from its authors.
