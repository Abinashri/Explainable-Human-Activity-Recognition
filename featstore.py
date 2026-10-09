"""Helpers for the cached per-frame features (used by train.py and evaluate.py)."""
import numpy as np
import torch

from common import NUM_FRAMES, window_starts


def load_features(path):
    d = np.load(path)
    lengths = d["lengths"]
    off = np.concatenate([[0], np.cumsum(lengths)])
    cnn_all, box_all = d["cnn"], d["boxf"]
    return dict(
        cnn=[cnn_all[off[i]:off[i + 1]] for i in range(len(lengths))],     # list of (L,1280) float16
        boxf=[box_all[off[i]:off[i + 1]] for i in range(len(lengths))],    # list of (L,6)
        y=d["y"], split=d["split"],
        names=[str(n) for n in d["names"]], classes=[str(c) for c in d["classes"]],
    )


@torch.no_grad()
def predict_video_probs(head, cnn, boxn, dev, n=NUM_FRAMES, hop=8):
    """Slide a 16-frame window over the whole video. Returns (starts, probs[W,C]).
    cnn: (L,1280); boxn: (L,6) already standardised."""
    head.eval()
    starts = window_starts(len(cnn), n, hop)
    X = np.stack([np.concatenate([cnn[s:s + n].astype(np.float32), boxn[s:s + n]], 1) for s in starts])
    probs = head(torch.from_numpy(X).to(dev)).softmax(1).cpu().numpy()
    return starts, probs
