import pytest
import torch

from box_box_bot.predictor.driver_model import DRIVER_MODEL_FILES, INPUT_DIM, load_model

MODEL_NAMES = list(DRIVER_MODEL_FILES)


@pytest.fixture(scope="module", params=MODEL_NAMES)
def model_name(request):
    return request.param


@pytest.fixture(scope="module")
def model(model_name):
    # Real bundled checkpoints - tiny, local, free to load, same tier as
    # fastembed in test_embeddings.py. Loaded once per checkpoint name.
    return load_model(model_name)


def test_load_model_returns_eval_mode_model(model, model_name):
    _, model_cls = DRIVER_MODEL_FILES[model_name]
    assert isinstance(model, model_cls)
    assert model.training is False


def test_model_produces_one_score_per_driver(model):
    x = torch.rand(6, INPUT_DIM)
    with torch.no_grad():
        scores = model(x)
    assert scores.shape == (6,)


def test_model_handles_single_driver_batch(model):
    # Regression guard: BatchNorm1d errors on a batch of size 1 in train
    # mode. eval mode (asserted above) uses running stats instead, so a
    # single-row batch must not crash.
    x = torch.rand(1, INPUT_DIM)
    with torch.no_grad():
        scores = model(x)
    assert scores.shape == (1,)


def test_model_scores_are_distinct_enough_to_rank(model):
    x = torch.rand(10, INPUT_DIM)
    with torch.no_grad():
        scores = model(x)
    assert len(set(scores.tolist())) > 1


def test_input_dim_matches_feature_columns():
    from box_box_bot.predictor.driver_features import FEATURE_COLUMNS

    assert INPUT_DIM == len(FEATURE_COLUMNS)


def test_each_checkpoint_loads_into_its_own_class():
    # Regression guard: activations aren't learnable params, so loading a
    # checkpoint's weights into the wrong activation's class would fail
    # silently (no shape mismatch, just wrong scores) rather than loudly.
    # This only proves the *shapes* line up for each dedicated class -
    # not that a checkpoint was loaded into the *correct* one - but a
    # shape mismatch here would still catch a copy-paste mistake.
    for name, (_, model_cls) in DRIVER_MODEL_FILES.items():
        loaded = load_model(name)
        assert isinstance(loaded, model_cls)
