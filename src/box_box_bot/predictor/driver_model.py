"""The driver-ranking models, ported from f1-drivers-predictor.

Pairwise-ranking network trained with a points-gap-weighted
nn.MarginRankingLoss, so the raw output is a per-driver score to be
sorted, not a probability or a points prediction - same kind of output as
the constructor models, but a separate architecture.

The source project ships 3 checkpoints (v2), all the same widened
(256/128/64) shape but each trained with a DIFFERENT activation function
(prost=LeakyReLU, schumacher=ReLU, senna=GELU), and each with its own
dedicated classifier file - activations have no learnable params, so
loading a checkpoint's weights into the wrong activation's class fails
silently (no error, just wrong scores) rather than loudly. That's why
there are three near-identical classes below instead of one shared class,
mirroring the source's own per-checkpoint file split.
"""

from pathlib import Path

import torch
from torch import nn

from box_box_bot.predictor.driver_features import FEATURE_COLUMNS

WEIGHTS_DIR = Path(__file__).parent / "weights"
INPUT_DIM = len(FEATURE_COLUMNS)


class F1DriversRankClassifierProst(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.25),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.35),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.layer(x).squeeze(-1)


class F1DriversRankClassifierSchumacher(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.layer(x).squeeze(-1)


class F1DriversRankClassifierSenna(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.layer(x).squeeze(-1)


DRIVER_MODEL_FILES = {
    "Prost": ("prost_model.pt", F1DriversRankClassifierProst),
    "Schumacher": ("schumacher_model.pt", F1DriversRankClassifierSchumacher),
    "Senna": ("senna_model.pt", F1DriversRankClassifierSenna),
}


def load_model(name: str) -> nn.Module:
    filename, model_cls = DRIVER_MODEL_FILES[name]
    model = model_cls(INPUT_DIM, 1)
    model.load_state_dict(torch.load(WEIGHTS_DIR / filename, map_location="cpu"))
    # Has BatchNorm1d layers, which error on a batch of size 1 in train
    # mode - eval mode uses running stats instead, and inference always
    # needs eval mode regardless of batch size.
    model.eval()
    return model
