import pytest
import torch

from box_box_bot.predictor.model import INPUT_DIM, F1ConstructorsClassifier, load_model


@pytest.fixture(scope="module")
def model():
    # Real bundled checkpoint - tiny (63KB), local, free to load, same
    # tier as fastembed in test_embeddings.py. Loaded once per test file.
    return load_model()


def test_load_model_returns_eval_mode_model(model):
    assert isinstance(model, F1ConstructorsClassifier)
    assert model.training is False


def test_model_produces_one_score_per_team(model):
    x = torch.rand(6, INPUT_DIM)
    with torch.no_grad():
        scores = model(x)
    assert scores.shape == (6,)


def test_model_handles_single_team_batch(model):
    # Regression guard: BatchNorm1d errors on a batch of size 1 in train
    # mode. eval mode (asserted above) uses running stats instead, so a
    # single-row batch - which happens whenever a season has fewer than
    # two active constructors in the feature table - must not crash.
    x = torch.rand(1, INPUT_DIM)
    with torch.no_grad():
        scores = model(x)
    assert scores.shape == (1,)


def test_model_scores_are_distinct_enough_to_rank(model):
    x = torch.rand(10, INPUT_DIM)
    with torch.no_grad():
        scores = model(x)
    assert len(set(scores.tolist())) > 1
