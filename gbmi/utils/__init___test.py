from transformer_lens import HookedTransformerConfig
from dataclasses import dataclass
from gbmi.exp_max_of_n.train import MaxOfN, MAX_OF_10_CONFIG
from gbmi.model import Config
from gbmi.utils import set_params, deep_getattr_or_item, deep_setattr_or_item
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
