"""CNN (MobileNetV2) + box-motion features -> LSTM -> FC."""
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

from common import BOX_DIM


class CNNEncoder(nn.Module):
    """MobileNetV2 without its classifier. Output: feature maps (N,1280,7,7)."""

    def __init__(self, pretrained=True):
        super().__init__()
        m = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT if pretrained else None)
        self.features = m.features

    def forward(self, x):
        return self.features(x)


class TemporalClassifier(nn.Module):
    """2-layer LSTM over a 16-frame clip; the outputs of ALL time steps are averaged -> class scores."""

    def __init__(self, num_classes, in_dim=1280 + BOX_DIM, hidden=256, layers=2, dropout=0.4, in_dropout=0.3):
        super().__init__()
        self.in_drop = nn.Dropout(in_dropout)
        self.lstm = nn.LSTM(in_dim, hidden, layers, batch_first=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):                      # (B,T,1286)
        out, _ = self.lstm(self.in_drop(x))
        return self.fc(self.drop(out.mean(1)))


class HARModel(nn.Module):
    def __init__(self, num_classes, hidden=256, layers=2, dropout=0.4, in_dropout=0.3, pretrained=True):
        super().__init__()
        self.encoder = CNNEncoder(pretrained)
        self.head = TemporalClassifier(num_classes, 1280 + BOX_DIM, hidden, layers, dropout, in_dropout)
        self.register_buffer("box_mean", torch.zeros(BOX_DIM))   # set from the training data
        self.register_buffer("box_std", torch.ones(BOX_DIM))

    def combine(self, cnn_feats, boxf):        # (B,T,1280), (B,T,6 raw) -> (B,T,1286)
        return torch.cat([cnn_feats, (boxf - self.box_mean) / self.box_std], dim=-1)

    def forward(self, frames, boxf, return_fmap=False):   # frames (B,T,3,H,W)
        B, T = frames.shape[:2]
        fmap = self.encoder(frames.flatten(0, 1))          # (B*T,1280,7,7)
        cnn = fmap.mean((2, 3)).view(B, T, -1)
        logits = self.head(self.combine(cnn, boxf))
        return (logits, fmap) if return_fmap else logits


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = HARModel(len(ck["classes"]), pretrained=False, **ck["hparams"])
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck["classes"]
