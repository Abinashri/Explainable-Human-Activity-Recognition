"""Step 2: train the LSTM on random 16-frame clips cut from the cached per-frame features."""
import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from common import NUM_FRAMES, get_device
from featstore import load_features, predict_video_probs
from model import HARModel

ap = argparse.ArgumentParser()
ap.add_argument("--features", default="cache/features.npz")
ap.add_argument("--out", default="outputs")
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--weight_decay", type=float, default=1e-4)
ap.add_argument("--hidden", type=int, default=256)
ap.add_argument("--layers", type=int, default=2)
ap.add_argument("--dropout", type=float, default=0.4)
ap.add_argument("--in_dropout", type=float, default=0.3)
ap.add_argument("--windows_per_video", type=int, default=8, help="random clips per training video per epoch")
ap.add_argument("--label_smoothing", type=float, default=0.1)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
out = Path(args.out); out.mkdir(exist_ok=True)
dev = get_device()
N = NUM_FRAMES

fs = load_features(args.features)
y, split, classes = fs["y"], fs["split"], fs["classes"]
tr, va = np.where(split == "train")[0], np.where(split == "val")[0]

hp = dict(hidden=args.hidden, layers=args.layers, dropout=args.dropout, in_dropout=args.in_dropout)
model = HARModel(len(classes), pretrained=True, **hp).to(dev)

# standardise the box features with TRAIN statistics (stored inside the model)
allbox = np.concatenate([fs["boxf"][i] for i in tr])
mean, std = allbox.mean(0), allbox.std(0) + 1e-6
model.box_mean.copy_(torch.tensor(mean)); model.box_std.copy_(torch.tensor(std))
boxn = [((b - mean) / std).astype(np.float32) for b in fs["boxf"]]

opt = torch.optim.Adam(model.head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)


def sample_windows(idx, k):
    xs, ys = [], []
    for v in idx:
        L = len(fs["cnn"][v])
        for s in np.random.randint(0, max(1, L - N + 1), size=k):    # random temporal position = free augmentation
            xs.append(np.concatenate([fs["cnn"][v][s:s + N].astype(np.float32), boxn[v][s:s + N]], 1))
            ys.append(y[v])
    return torch.from_numpy(np.stack(xs)), torch.tensor(ys).long()


def eval_videos(idx):
    """Video-level: average the probabilities of all sliding clips of a video."""
    correct, nll = 0, 0.0
    for v in idx:
        _, p = predict_video_probs(model.head, fs["cnn"][v], boxn[v], dev)
        mp = p.mean(0)
        correct += int(mp.argmax() == y[v])
        nll += -np.log(mp[y[v]] + 1e-8)
    return nll / len(idx), correct / len(idx)


hist = dict(train_loss=[], train_acc=[], val_loss=[], val_acc=[])
best = (-1, 1e9, None)
for ep in range(1, args.epochs + 1):
    model.head.train()
    X, Y = sample_windows(tr, args.windows_per_video)
    perm = torch.randperm(len(Y))
    loss_sum = correct = 0
    for i in range(0, len(Y), args.bs):
        b = perm[i:i + args.bs]
        xb, yb = X[b].to(dev), Y[b].to(dev)
        logits = model.head(xb)
        loss = crit(logits, yb)
        opt.zero_grad(); loss.backward(); opt.step()
        loss_sum += loss.item() * len(b); correct += (logits.argmax(1) == yb).sum().item()
    sched.step()
    tl, ta = loss_sum / len(Y), correct / len(Y)
    vl, vacc = eval_videos(va)
    for k, v in zip(hist, (tl, ta, vl, vacc)):
        hist[k].append(float(v))
    if (vacc, -vl) > (best[0], -best[1]):
        best = (vacc, vl, {k: t.cpu().clone() for k, t in model.head.state_dict().items()})
    print(f"ep {ep:3d} | train loss {tl:.3f} clip-acc {ta:.3f} | val NLL {vl:.3f} video-acc {vacc:.3f}")

model.head.load_state_dict(best[2])
torch.save({"state_dict": model.state_dict(), "classes": classes, "hparams": hp}, out / "har_model.pt")
json.dump(hist, open(out / "history.json", "w"), indent=2)
json.dump({**vars(args), "optimizer": "Adam", "loss": "CrossEntropy (label smoothing)", "scheduler": "CosineAnnealing",
           "clip_length": N, "best_val_video_acc": best[0], "classes": classes},
          open(out / "train_config.json", "w"), indent=2)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
ax[0].plot(hist["train_loss"], label="train (clips)"); ax[0].plot(hist["val_loss"], label="val (videos, NLL)"); ax[0].set_title("Loss")
ax[1].plot(hist["train_acc"], label="train (clips)"); ax[1].plot(hist["val_acc"], label="val (videos)"); ax[1].set_title("Accuracy")
for a in ax:
    a.set_xlabel("epoch"); a.legend()
plt.tight_layout(); plt.savefig(out / "curves.png", dpi=150)
print(f"\nBest val video accuracy {best[0]:.3f}. Saved {out/'har_model.pt'}")
