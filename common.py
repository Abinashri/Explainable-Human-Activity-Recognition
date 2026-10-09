"""Shared config + preprocessing. Used by feature extraction, training AND the app."""
import cv2
import numpy as np
import torch
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_fpn,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
)

NUM_FRAMES = 16        # frames per clip (LSTM sequence length)
IMG_SIZE = 224
TARGET_FPS = 12.5      # frames are sampled at ~12.5 fps (every 2nd frame for 25-fps KTH videos)
MAX_SAMPLED = 400      # at most ~32 s of video is used
DETECT_EVERY = 8       # run the person detector on every 8th sampled frame, interpolate between
BOX_DIM = 6            # cx, cy, w, h, d_cx, d_cy  (box position / size / speed)
MAX_SIDE = 640         # larger frames are shrunk (KTH is 160x120, unaffected)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- reading frames
def read_sampled_frames(path, target_fps=TARGET_FPS, max_frames=MAX_SAMPLED):
    """Read the video at ~target_fps (dense, consecutive sampling - NOT 16 frames spread over the whole video)."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps < 1 or fps > 240:
        fps = 25.0
    stride = max(1, int(round(fps / target_fps)))
    frames, i = [], 0
    while len(frames) < max_frames:
        if i % stride == 0:
            ok, f = cap.read()
            if not ok:
                break
            h, w = f.shape[:2]
            s = MAX_SIDE / max(h, w)
            if s < 1:
                f = cv2.resize(f, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        elif not cap.grab():
            break
        i += 1
    cap.release()
    if not frames:
        raise ValueError(f"Cannot read video: {path}")
    return frames


def window_starts(L, n=NUM_FRAMES, hop=8):
    """Start indices of the sliding 16-frame clips over a video of L sampled frames."""
    if L <= n:
        return [0]
    starts = list(range(0, L - n + 1, hop))
    if starts[-1] != L - n:
        starts.append(L - n)
    return starts


# ---------------------------------------------------------------- person detection
class PersonDetector:
    """Faster R-CNN (MobileNetV3 backbone, COCO). COCO label 1 = person."""

    def __init__(self, device=None, score_thr=0.5, min_size=480, max_size=640):
        self.device = device or get_device()
        weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        self.model = fasterrcnn_mobilenet_v3_large_fpn(
            weights=weights, min_size=min_size, max_size=max_size).to(self.device).eval()
        self.score_thr = score_thr

    @torch.no_grad()
    def detect(self, rgb):
        """Return (x1,y1,x2,y2) of the largest confident person, or None."""
        x = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).to(self.device)
        out = self.model([x])[0]
        best, best_area = None, 0.0
        for box, label, score in zip(out["boxes"], out["labels"], out["scores"]):
            if label.item() == 1 and score.item() >= self.score_thr:
                x1, y1, x2, y2 = box.tolist()
                area = (x2 - x1) * (y2 - y1)
                if area > best_area:
                    best, best_area = (x1, y1, x2, y2), area
        return best


def interpolate_boxes(det_idx, dets, L, w, h):
    """Linear interpolation of boxes between detected key frames -> one box per frame (L,4).
    Frames before/after the first/last detection use the nearest box; no detection at all -> full frame."""
    ok = [(i, b) for i, b in zip(det_idx, dets) if b is not None]
    if not ok:
        return np.tile(np.array([0, 0, w, h], np.float32), (L, 1))
    xs = np.array([i for i, _ in ok], dtype=np.float64)
    B = np.array([b for _, b in ok], dtype=np.float64)
    t = np.arange(L)
    return np.stack([np.interp(t, xs, B[:, k]) for k in range(4)], 1).astype(np.float32)


def box_features(boxes, w, h):
    """(L,4) boxes -> (L,6): centre x/y, width, height (all relative to frame) + centre velocity.
    The velocity restores the 'how fast does the person move' cue that cropping removes."""
    cx = (boxes[:, 0] + boxes[:, 2]) / 2 / w
    cy = (boxes[:, 1] + boxes[:, 3]) / 2 / h
    bw = (boxes[:, 2] - boxes[:, 0]) / w
    bh = (boxes[:, 3] - boxes[:, 1]) / h
    dcx = np.diff(cx, prepend=cx[0])
    dcy = np.diff(cy, prepend=cy[0])
    return np.stack([cx, cy, bw, bh, dcx, dcy], 1).astype(np.float32)


def expand_box(box, w, h, margin=0.1):
    """Make the box square-ish with a small margin, clipped to the image."""
    x1, y1, x2, y2 = [float(v) for v in box]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * (1 + margin)
    nx1, nx2 = int(max(0, cx - side / 2)), int(min(w, cx + side / 2))
    ny1, ny2 = int(max(0, cy - side / 2)), int(min(h, cy + side / 2))
    if nx2 - nx1 < 2 or ny2 - ny1 < 2:
        return 0, 0, w, h
    return nx1, ny1, nx2, ny2


def preprocess_video(path, detector, detect_every=DETECT_EVERY):
    """video -> dense frames -> detect (key frames) + interpolate -> crop -> resize 224."""
    frames = read_sampled_frames(path)
    while len(frames) < NUM_FRAMES:                 # very short video: repeat last frame
        frames.append(frames[-1])
    L = len(frames)
    h, w = frames[0].shape[:2]

    det_idx = list(range(0, L, detect_every))
    if det_idx[-1] != L - 1:
        det_idx.append(L - 1)
    dets = [detector.detect(frames[i]) for i in det_idx]
    n_det = sum(b is not None for b in dets)

    boxes_f = interpolate_boxes(det_idx, dets, L, w, h)
    boxf = box_features(boxes_f, w, h)
    boxes = [expand_box(b, w, h) for b in boxes_f]
    crops = np.stack([cv2.resize(f[y1:y2, x1:x2], (IMG_SIZE, IMG_SIZE))
                      for f, (x1, y1, x2, y2) in zip(frames, boxes)])
    return dict(crops=crops, frames=frames, boxes=boxes, boxf=boxf, n_det=n_det, n_key=len(det_idx))


def to_tensor(crops):
    """(T,H,W,3) uint8 -> normalised float tensor (T,3,H,W)."""
    x = (crops.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()


@torch.no_grad()
def encode_crops(encoder, crops, dev, bs=64):
    """MobileNetV2 -> global-average-pooled feature per frame, (L,1280) float32 numpy."""
    encoder.eval()
    out = []
    for i in range(0, len(crops), bs):
        out.append(encoder(to_tensor(crops[i:i + bs]).to(dev)).mean((2, 3)).cpu())
    return torch.cat(out).numpy()
