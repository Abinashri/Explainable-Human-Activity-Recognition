"""Grad-CAM through the whole CNN+box-features+LSTM model for one 16-frame clip.

Gradient of the class score w.r.t. MobileNetV2's last conv feature maps of every frame in the clip
(the gradient flows back through the FC layer and the LSTM):
    alpha_k = mean_ij(dy/dA_ij^k),    L = ReLU(sum_k alpha_k * A^k)
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F


def explain(model, x, boxf, class_idx=None):
    """x: normalised tensor (T,3,H,W); boxf: raw box features (T,6) numpy.
    Returns probs (C,), explained class, cams (T,H,W) in [0,1]."""
    model.eval()
    dev = next(model.parameters()).device
    x = x.unsqueeze(0).to(dev)
    b = torch.from_numpy(np.asarray(boxf, dtype=np.float32)).unsqueeze(0).to(dev)
    with torch.backends.cudnn.flags(enabled=False):   # cuDNN RNN can't backprop in eval mode
        model.zero_grad()
        logits, fmap = model(x, b, return_fmap=True)  # fmap: (T,1280,7,7)
        fmap.retain_grad()
        probs = logits.softmax(1)[0].detach().cpu().numpy()
        c = int(logits.argmax(1)) if class_idx is None else int(class_idx)
        logits[0, c].backward()

    w = fmap.grad.mean((2, 3), keepdim=True)          # (T,1280,1,1)
    cam = F.relu((w * fmap).sum(1)).detach()          # (T,7,7)
    cam = F.interpolate(cam[:, None], size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    cam = (cam / (cam.max() + 1e-8)).cpu().numpy()    # one shared scale -> frames are comparable
    return probs, c, cam


def overlay(rgb, cam, alpha=0.45):
    """Blend a [0,1] heatmap onto an RGB uint8 image of the same size."""
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return np.uint8((1 - alpha) * rgb + alpha * heat)
