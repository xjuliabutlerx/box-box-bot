"""The constructor-ranking model, ported from f1-constructors-predictor.

Architecture and weights (monaco_model_v3.pt) are unchanged from the
source project - a pairwise-ranking network trained with
nn.MarginRankingLoss, so the raw output is a per-team score to be sorted,
not a probability or a points prediction.
"""

from pathlib import Path

import torch
from torch import nn

WEIGHTS_PATH = Path(__file__).parent / "weights" / "monaco_model_v3.pt"
INPUT_DIM = 21


class F1ConstructorsClassifier(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, output_dim),
        )

    def forward(self, x):
        return self.layer(x).squeeze(-1)


def load_model() -> F1ConstructorsClassifier:
    model = F1ConstructorsClassifier(INPUT_DIM, 1)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
    # Has BatchNorm1d layers, which error on a batch of size 1 in train
    # mode - eval mode uses running stats instead, and inference always
    # needs eval mode regardless of batch size.
    model.eval()
    return model
