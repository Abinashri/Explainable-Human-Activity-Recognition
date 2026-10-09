"""Step 3: evaluate on the held-out TEST persons (never used for training or model selection)."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from common import get_device
from featstore import load_features, predict_video_probs
from model import load_model

ap = argparse.ArgumentParser()
ap.add_argument("--features", default="cache/features.npz")
ap.add_argument("--ckpt", default="outputs/har_model.pt")
ap.add_argument("--out", default="outputs")
args = ap.parse_args()

dev = get_device()
model, classes = load_model(args.ckpt, dev)
mean, std = model.box_mean.cpu().numpy(), model.box_std.cpu().numpy()
fs = load_features(args.features)
te = np.where(fs["split"] == "test")[0]

y_true, y_pred, conf, win_ok, win_n = [], [], [], 0, 0
for v in te:
    _, p = predict_video_probs(model.head, fs["cnn"][v], (fs["boxf"][v] - mean) / std, dev)
    mp = p.mean(0)                                   # video prediction = mean over all clips
    y_true.append(fs["y"][v]); y_pred.append(int(mp.argmax())); conf.append(float(mp.max()))
    win_ok += int((p.argmax(1) == fs["y"][v]).sum()); win_n += len(p)

y_true, y_pred = np.array(y_true), np.array(y_pred)
acc = accuracy_score(y_true, y_pred)
print(f"Test videos: {len(te)}   VIDEO accuracy: {acc:.4f}   (single-clip accuracy: {win_ok / win_n:.4f})\n")
print(classification_report(y_true, y_pred, target_names=classes, digits=3))
cm = confusion_matrix(y_true, y_pred)
print("Confusion matrix (rows = actual, cols = predicted):\n", cm)

outd = Path(args.out)
fig, ax = plt.subplots(figsize=(6, 5))
ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right")
ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
for i in range(len(classes)):
    for j in range(len(classes)):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion matrix (test videos)")
plt.tight_layout(); plt.savefig(outd / "confusion_matrix.png", dpi=150)

json.dump({"video_accuracy": acc, "clip_accuracy": win_ok / win_n,
           "report": classification_report(y_true, y_pred, target_names=classes, output_dict=True),
           "confusion_matrix": cm.tolist()}, open(outd / "metrics.json", "w"), indent=2)
with open(outd / "test_predictions.csv", "w", newline="") as f:      # find the wrong videos easily
    w = csv.writer(f); w.writerow(["video", "actual", "predicted", "confidence", "correct"])
    for v, t, p, c in zip(te, y_true, y_pred, conf):
        w.writerow([fs["names"][v], classes[t], classes[p], f"{c:.3f}", t == p])
print(f"\nSaved confusion_matrix.png, metrics.json, test_predictions.csv in {outd}")
