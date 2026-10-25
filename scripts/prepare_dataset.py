"""Download the Hard Hat dataset (CC BY 4.0) and convert a fixed subset to YOLO format.

Source: https://huggingface.co/datasets/keremberke/hard-hat-detection
(Roboflow Universe "Hard Hats", resized to 640x640, CC BY 4.0).
"""
import argparse, json, random, shutil, urllib.request, zipfile
from pathlib import Path

BASE = "https://huggingface.co/datasets/keremberke/hard-hat-detection/resolve/main/data"
CLASSES = ["hardhat", "no-hardhat"]


def fetch(split, raw):
    raw.mkdir(parents=True, exist_ok=True)
    z = raw / f"{split}.zip"
    if not z.exists():
        print(f"downloading {split}.zip ...")
        urllib.request.urlretrieve(f"{BASE}/{split}.zip", z)
    out = raw / split
    if not out.exists():
        with zipfile.ZipFile(z) as f:
            f.extractall(out)
    return out


def convert(src, dst, split, limit, seed):
    coco = json.loads((src / "_annotations.coco.json").read_text())
    names = {c["id"]: c["name"] for c in coco["categories"]}
    cls = {cid: CLASSES.index(n) for cid, n in names.items() if n in CLASSES}
    anns = {}
    for a in coco["annotations"]:
        if a["category_id"] in cls:
            anns.setdefault(a["image_id"], []).append(a)
    imgs = [i for i in coco["images"] if i["id"] in anns]
    random.Random(seed).shuffle(imgs)
    imgs = imgs[:limit]
    (dst / "images" / split).mkdir(parents=True, exist_ok=True)
    (dst / "labels" / split).mkdir(parents=True, exist_ok=True)
    n_boxes = 0
    for im in imgs:
        w, h = im["width"], im["height"]
        shutil.copy(src / im["file_name"], dst / "images" / split / im["file_name"])
        lines = []
        for a in anns[im["id"]]:
            x, y, bw, bh = a["bbox"]
            if bw <= 0 or bh <= 0:
                continue
            lines.append(f"{cls[a['category_id']]} {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}")
        n_boxes += len(lines)
        (dst / "labels" / split / (Path(im["file_name"]).stem + ".txt")).write_text("\n".join(lines) + "\n")
    print(f"{split}: {len(imgs)} images, {n_boxes} boxes")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="datasets/hardhat")
    p.add_argument("--raw", default="raw")
    p.add_argument("--train", type=int, default=2000)
    p.add_argument("--val", type=int, default=400)
    p.add_argument("--test", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    out = Path(a.out)
    for split, src_split, n in [("train", "train", a.train), ("val", "valid", a.val), ("test", "test", a.test)]:
        convert(fetch(src_split, Path(a.raw)), out, split, n, a.seed)
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES))
    )
    print("wrote", out / "data.yaml")


if __name__ == "__main__":
    main()
