"""Step 1: videos -> dense frames -> person boxes (detect + interpolate) -> crops -> MobileNetV2 features.

Expected layout:  <data>/<class_name>/*.avi      (python prepare_data.py --data kth_videos)
Resumable: every finished video is cached in cache/videos/, so you can stop (Ctrl+C) and continue later.
"""
import argparse
import re
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from common import DETECT_EVERY, PersonDetector, encode_crops, get_device, preprocess_video
from model import CNNEncoder

# Standard KTH protocol: split by PERSON so the same subject never appears in train and test.
KTH_TRAIN = {11, 12, 13, 14, 15, 16, 17, 18}
KTH_VAL = {19, 20, 21, 23, 24, 25, 1, 4}
KTH_TEST = {22, 2, 3, 5, 6, 7, 8, 9, 10}


def make_split(names, y, seed=42):
    ids = [re.search(r"person(\d+)", n, re.I) for n in names]
    if all(ids):
        pid = [int(m.group(1)) for m in ids]
        print("Using subject-wise KTH split")
        return np.array(["train" if p in KTH_TRAIN else "val" if p in KTH_VAL else "test" for p in pid])
    print("Filenames have no personXX id -> stratified random 70/15/15 split")
    idx = np.arange(len(y))
    tr, rest = train_test_split(idx, test_size=0.3, stratify=y, random_state=seed)
    va, te = train_test_split(rest, test_size=0.5, stratify=y[rest], random_state=seed)
    split = np.empty(len(y), dtype=object)
    split[tr], split[va], split[te] = "train", "val", "test"
    return split.astype(str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="cache/features.npz")
    ap.add_argument("--detect_every", type=int, default=DETECT_EVERY)
    args = ap.parse_args()

    root = Path(args.data)
    classes = sorted(p.name for p in root.iterdir() if p.is_dir())
    exts = {".avi", ".mp4", ".mov", ".mkv"}
    videos = [(v, ci) for ci, c in enumerate(classes)
              for v in sorted((root / c).rglob("*")) if v.suffix.lower() in exts]
    print(f"Classes: {classes}\nVideos found: {len(videos)}")

    out = Path(args.out)
    vid_dir = out.parent / "videos"
    sample_dir = out.parent / "samples"
    vid_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(exist_ok=True)

    dev = get_device()
    detector, enc = PersonDetector(dev), CNNEncoder().to(dev).eval()

    cnn, boxf, y, names, lengths, rates = [], [], [], [], [], []
    saved = set()
    for path, ci in tqdm(videos):
        cf = vid_dir / f"{classes[ci]}__{path.stem}.npz"
        if cf.exists():
            with np.load(cf) as z:
                c, b, rate = z["cnn"], z["boxf"], float(z["rate"])
        else:
            try:
                pre = preprocess_video(path, detector, args.detect_every)
            except Exception as e:                       # corrupted video -> skip, report
                print(f"SKIPPED {path}: {e}")
                continue
            c = encode_crops(enc, pre["crops"], dev).astype(np.float16)
            b, rate = pre["boxf"], pre["n_det"] / pre["n_key"]
            np.savez(cf, cnn=c, boxf=b, rate=rate)
            if ci not in saved:                          # one example per class (for the report)
                saved.add(ci)
                mid = len(pre["frames"]) // 2
                x1, y1, x2, y2 = pre["boxes"][mid]
                a = pre["frames"][mid].copy()
                cv2.rectangle(a, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.imwrite(str(sample_dir / f"{classes[ci]}_bbox.jpg"), cv2.cvtColor(a, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(sample_dir / f"{classes[ci]}_crop.jpg"), cv2.cvtColor(pre["crops"][mid], cv2.COLOR_RGB2BGR))
        cnn.append(c); boxf.append(b); y.append(ci); names.append(path.name)
        lengths.append(len(c)); rates.append(rate)

    y = np.array(y)
    split = make_split(names, y)
    np.savez(out, cnn=np.concatenate(cnn), boxf=np.concatenate(boxf), lengths=np.array(lengths),
             y=y, split=split, names=np.array(names), classes=np.array(classes))

    print(f"\nSaved {out}: {len(y)} videos, {sum(lengths)} frames in total (mean {np.mean(lengths):.0f} per video)")
    print(f"Person detected in {np.mean(rates):.1%} of the checked key frames")
    for s in ("train", "val", "test"):
        print(f"  {s:5s}: {np.bincount(y[split == s], minlength=len(classes)).tolist()}  (per class {classes})")


if __name__ == "__main__":
    main()
