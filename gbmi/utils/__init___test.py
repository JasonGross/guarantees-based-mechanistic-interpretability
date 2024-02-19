from dataclasses import dataclass
import torch
from torch import tensor
import numpy as np
from gbmi.utils import (
    set_params,
    deep_getattr_or_item,
    deep_setattr_or_item,
    log_softmax,
)
from gbmi.utils.testing import TestCase


# mock classes
@dataclass
class ModelConfig:
    d_vocab: int


@dataclass
class Experiment:
    model_config: ModelConfig


@dataclass
class Config:
    experiment: Experiment


class TestInit(TestCase):
    def test_deep_getattr_setattr(self):
        # Directly define a simple configuration
        config = Config(Experiment(ModelConfig(d_vocab=64)))
        self.assertTrue(
            deep_getattr_or_item(config, ["experiment", "model_config", "d_vocab"])
            == 64
        )
        deep_setattr_or_item(config, ["experiment", "model_config", "d_vocab"], 128)
        self.assertTrue(config.experiment.model_config.d_vocab == 128)

    def test_set_params_immutable(self):
        # Directly define a simple configuration
        config = Config(Experiment(ModelConfig(d_vocab=64)))
        # Simulate the set_params functionality for updating config
        # Assuming set_params returns a new instance of Config with the updated value
        config_2 = set_params(config, {("experiment", "model_config", "d_vocab"): 1})

        self.assertEqual(config.experiment.model_config.d_vocab, 64)
        self.assertEqual(config_2.experiment.model_config.d_vocab, 1)

    def log_softmax_tester_helper(self, x, expected, dim=None, **kwargs):
        res = log_softmax(x, dim=dim)
        diff = torch.abs(res - expected)
        self.assertAllClose(
            res,
            expected,
            msg=f"log_softmax({x}{f', dim={dim}' if dim is not None else ''}) == {res} != {expected} within {kwargs} ({diff} difference)",
            **kwargs,
        )

    def test_log_softmax(self):
        for x, expected in [
            (tensor([1000, 1]), tensor([0.0, -999.0])),
            # we shouldn't return zero on the smallest subnormal input
            (
                tensor(
                    [-np.log(np.finfo(np.float32).smallest_subnormal), 0],
                    dtype=torch.float32,
                ),
                tensor(
                    [float.fromhex("-0x1.00000p-149"), float.fromhex("-0x1.9d1dap+6")],
                    dtype=torch.float32,
                ),
            ),
            (
                tensor(
                    [-np.log(np.finfo(np.float64).smallest_subnormal), 0],
                    dtype=torch.float64,
                ),
                tensor(
                    [
                        float.fromhex("-0x0.0000000000001p-1022"),
                        float.fromhex("-0x1.74385446d71c3p+9"),
                    ],
                    dtype=torch.float64,
                ),
            ),
            # Expected value computed using mpmath (with mpmath.mp.dps = 200) and then
            # converted to float.
            (
                torch.arange(4),
                tensor(
                    [
                        -3.4401896985611953,
                        -2.4401896985611953,
                        -1.4401896985611953,
                        -0.44018969856119533,
                    ]
                ),
            ),
            # scalar
            (tensor(1.0), 0.0),
        ]:
            self.log_softmax_tester_helper(x, expected, rtol=1e-13)

    def test_log_softmax_2d_dim1(self):
        x = torch.arange(8).reshape(2, 4)
        expected = tensor(
            [
                [
                    -3.4401896985611953,
                    -2.4401896985611953,
                    -1.4401896985611953,
                    -0.44018969856119533,
                ],
                [
                    -3.4401896985611953,
                    -2.4401896985611953,
                    -1.4401896985611953,
                    -0.44018969856119533,
                ],
            ]
        )
        self.log_softmax_tester_helper(x, expected, rtol=1e-13, dim=1)
