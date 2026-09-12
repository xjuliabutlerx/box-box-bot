"""The constructor-ranking model, ported from f1-constructors-predictor.

Architecture is unchanged from the source project - a pairwise-ranking
network trained with nn.MarginRankingLoss, so the raw output is a
per-team score to be sorted, not a probability or a points prediction.

The source project ships 5 checkpoints (v3), each from a separate
training run - the filenames are post-hoc quality labels (best overall /
most consistent / best mid-field / best at the extremes / most robust in
volatile seasons), not different architectures or feature snapshots. All
5 share this exact class and the 21-column feature table in features.py.
"""

from pathlib import Path

import torch
from torch import nn

WEIGHTS_DIR = Path(__file__).parent / "weights"
INPUT_DIM = 21

CONSTRUCTOR_MODEL_FILES = {
    "Monaco": "monaco_model_v3.pt",
    "Silverstone": "silverstone_model_v3.pt",
    "Suzuka": "suzuka_model_v3.pt",
    "Spa-Francorchamps": "spa-francorchamps_model_v3.pt",
    "Baku": "baku_model_v3.pt",
}


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


def load_model(weights_filename: str = "monaco_model_v3.pt") -> F1ConstructorsClassifier:
    model = F1ConstructorsClassifier(INPUT_DIM, 1)
    model.load_state_dict(torch.load(WEIGHTS_DIR / weights_filename, map_location="cpu"))
    # Has BatchNorm1d layers, which error on a batch of size 1 in train
    # mode - eval mode uses running stats instead, and inference always
    # needs eval mode regardless of batch size.
    model.eval()
    return model
