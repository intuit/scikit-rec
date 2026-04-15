from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

from skrec.estimator.classification.deep_fm_classifier import (  # noqa: E402
    CrossNetwork,
    DeepFactorizationMachineNetwork,
)
from skrec.util.config_loader import load_config
from skrec.util.torch_device import select_torch_device  # noqa: E402


@pytest.fixture
def setup_fixture():
    test = {}
    files_path = Path.cwd() / "skrec/examples/estimators/"

    test["multi_class_df"] = pd.read_csv(files_path / "multi_class_data.csv")
    test["reward_model_df"] = pd.read_csv(files_path / "reward_model_data_classification.csv")
    test["personalized_df"] = pd.read_csv(files_path / "generated_personalized_data.csv")
    test["estimator_config"] = load_config(files_path / "estimator_hyperparameters.yaml")

    test["reward_model_y"] = test["reward_model_df"]["y"].to_numpy()
    test["reward_model_x"] = test["reward_model_df"].drop(columns=["y"])

    test["multioutput_x"] = pd.DataFrame({"Age": [28, 49, 35, 30], "Gender": [1, 0, 1, 0]})
    test["multioutput_y"] = np.array(
        [
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 1, 0],
        ]
    )
    return test


def test_select_torch_device():
    # explicit device string
    assert select_torch_device("cpu") == torch.device("cpu")
    assert select_torch_device("mps") == torch.device("mps")

    # invalid device string raises ValueError
    with pytest.raises(ValueError, match="Invalid device 'gpu'"):
        select_torch_device("gpu")

    # auto-detect: cuda available → cuda
    with patch("torch.cuda.is_available", return_value=True):
        assert select_torch_device(None) == torch.device("cuda")

    # auto-detect: cuda not available → cpu
    with patch("torch.cuda.is_available", return_value=False):
        assert select_torch_device(None) == torch.device("cpu")


def test_CrossNetwork():
    # simple test to make sure model's forward pass runs
    n_samples, input_dim = 10, 8
    X = torch.randn(n_samples, input_dim)

    num_layers = 4

    network = CrossNetwork(
        input_dim=input_dim,
        num_layers=num_layers,
    )

    # check params have been set
    assert len(network.weights) == num_layers
    for weight in network.weights:
        assert (weight != 0).all()

    assert len(network.biases) == num_layers
    for bias in network.biases:
        assert (bias == 0).all()

    out = network(X)
    assert out.shape == (n_samples, input_dim)
    assert not torch.equal(out, X)


@pytest.mark.parametrize("use_cross_layer", [True, False])
@pytest.mark.parametrize("use_batch_norm", [True, False])
def test_DeepFactorizationMachineNetwork_output_shape(setup_fixture, use_cross_layer, use_batch_norm):
    # simple test to make sure model's forward pass runs and returns
    # expected shape
    X, _ = setup_fixture["multioutput_x"], setup_fixture["multioutput_y"]
    X = torch.Tensor(X.values)

    n_samples, input_dim = X.shape
    embedding_dim = 32
    hidden_dims = [32, 32]
    dropout = 0.1

    network = DeepFactorizationMachineNetwork(
        input_dim=input_dim,
        embedding_dim=embedding_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        use_cross_layer=use_cross_layer,
        use_batch_norm=use_batch_norm,
    )
    out = network(X)

    exp_output_dim = 1
    assert out.shape == (n_samples, exp_output_dim)
