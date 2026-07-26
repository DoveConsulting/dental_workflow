#!/usr/bin/env python3
"""
split_and_train_obb.py

Randomly splits a YOLO-OBB dataset into train/val sets, writes a data.yaml,
and (optionally) trains an Ultralytics YOLO OBB model on it.

Expected input layout (typical CVAT / Datumaro "YOLO OBB" export):

    dataset/
        images/          # img1.jpg, img2.jpg, ...
        labels/          # img1.txt, img2.txt, ...  (same basename as image)
        classes.txt      # one class name per line, in class-index order
        notes.json       # metadata, not required for training (ignored)

Each label line is expected in YOLO-OBB format:
    class_index x1 y1 x2 y2 x3 y3 x4 y4   (all coords normalized 0-1)

Output layout (created fresh, originals are left untouched):

    <output>/
        images/train/...
        images/val/...
        labels/train/...
        labels/val/...
        data.yaml

Usage
-----
    pip install ultralytics

    # split only
    python split_and_train_obb.py --dataset ./dataset --output ./dataset_split --no-train

    # split + train
    python split_and_train_obb.py --dataset ./dataset --output ./dataset_split \
        --model yolo26n-obb.pt --epochs 100 --imgsz 1024 --batch 8 --device 0

    # already split, just train
    python split_and_train_obb.py --dataset ./dataset --output ./dataset_split \
        --skip-split --model yolo26n-obb.pt
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=str, required=True, default="./dataset", help="Path to source dataset folder (contains images/, labels/, classes.txt)")
    p.add_argument("--output", type=str, default="dataset_split", help="Where to write the split dataset + data.yaml")
    p.add_argument("--val-split", type=float, default=0.2, help="Fraction of data used for validation (default 0.2)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducible splits")
    p.add_argument("--mode", choices=["copy", "symlink", "move"], default="copy", help="How to place files into the split folders")
    p.add_argument("--allow-empty-labels", action="store_true", help="Allow images with a missing/empty label file (treated as background images)")

    p.add_argument("--skip-split", action="store_true", help="Skip splitting, assume --output already has the train/val layout + data.yaml")
    p.add_argument("--no-train", action="store_true", help="Only perform the split, do not launch training")

    p.add_argument("--model", type=str, default="yolo26n-obb.pt", help="Base model/checkpoint to train from (e.g. yolo11n-obb.pt, yolov8n-obb.pt, or a path to a .pt/.yaml)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=1024, help="Train image size (source images are 1920x1080; 1024 or 1280 are good OBB defaults)")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", type=str, default=None, help="e.g. '0' for GPU 0, '0,1' for multi-GPU, 'cpu'")
    p.add_argument("--project", type=str, default="runs_obb", help="Ultralytics project (output) dir")
    p.add_argument("--name", type=str, default="train", help="Ultralytics run name")
    p.add_argument("--patience", type=int, default=50, help="Early-stopping patience")
    p.add_argument("--resume", action="store_true", help="Resume the last training run")
    return p.parse_args()


def read_classes(dataset_dir: Path):
    classes_file = dataset_dir / "classes.txt"
    if not classes_file.exists():
        sys.exit(f"classes.txt not found at {classes_file}")
    names = [line.strip() for line in classes_file.read_text().splitlines() if line.strip()]
    if not names:
        sys.exit("classes.txt is empty")
    return names


def gather_pairs(dataset_dir: Path, allow_empty_labels: bool):
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    if not images_dir.is_dir():
        sys.exit(f"Images folder not found: {images_dir}")
    if not labels_dir.is_dir():
        sys.exit(f"Labels folder not found: {labels_dir}")

    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not images:
        sys.exit(f"No images found in {images_dir}")

    pairs = []
    skipped = 0
    for img in images:
        label = labels_dir / f"{img.stem}.txt"
        if not label.exists():
            if allow_empty_labels:
                pairs.append((img, None))
            else:
                skipped += 1
            continue
        pairs.append((img, label))

    if skipped:
        print(f"Warning: skipped {skipped} image(s) with no matching label file "
              f"(pass --allow-empty-labels to include them as background images).")
    if not pairs:
        sys.exit("No valid image/label pairs found.")
    return pairs


def place_file(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "move":
        shutil.move(str(src), dst)
    elif mode == "symlink":
        if dst.exists():
            dst.unlink()
        dst.symlink_to(src.resolve())


def split_dataset(args) -> Path:
    dataset_dir = Path(args.dataset).resolve()
    output_dir = Path(args.output).resolve()

    class_names = read_classes(dataset_dir)
    pairs = gather_pairs(dataset_dir, args.allow_empty_labels)

    rng = random.Random(args.seed)
    rng.shuffle(pairs)

    n_val = max(1, round(len(pairs) * args.val_split))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    if not train_pairs:
        sys.exit("Train split is empty — lower --val-split or add more data.")

    for split_name, split_pairs in (("train", train_pairs), ("val", val_pairs)):
        for img, label in split_pairs:
            place_file(img, output_dir / "images" / split_name / img.name, args.mode)
            if label is not None:
                place_file(label, output_dir / "labels" / split_name / label.name, args.mode)
            else:
                # background image: create an empty label file so Ultralytics is happy
                empty_label = output_dir / "labels" / split_name / f"{img.stem}.txt"
                empty_label.parent.mkdir(parents=True, exist_ok=True)
                empty_label.touch()

    data_yaml = output_dir / "data.yaml"
    names_block = "\n".join(f"  {i}: {name}" for i, name in enumerate(class_names))
    data_yaml.write_text(
        f"path: {output_dir}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n{names_block}\n"
    )

    print(f"Split complete: {len(train_pairs)} train / {len(val_pairs)} val images.")
    print(f"data.yaml written to: {data_yaml}")
    return data_yaml


def train(args, data_yaml: Path):
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics is not installed. Run: pip install ultralytics")

    model = YOLO(args.model)  # an "-obb" checkpoint/config selects the OBB task automatically
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        resume=args.resume,
    )


def main():
    args = parse_args()

    if args.skip_split:
        data_yaml = Path(args.output).resolve() / "data.yaml"
        if not data_yaml.exists():
            sys.exit(f"--skip-split was given but {data_yaml} does not exist.")
    else:
        data_yaml = split_dataset(args)

    if not args.no_train:
        train(args, data_yaml)


if __name__ == "__main__":
    main()
