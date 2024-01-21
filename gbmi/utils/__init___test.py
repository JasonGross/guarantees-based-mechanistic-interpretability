from transformer_lens import HookedTransformerConfig

from gbmi.exp_max_of_n.train import MaxOfN, MAX_OF_10_CONFIG
from gbmi.model import Config
from gbmi.utils import set_params, deep_getattr_or_item, deep_setattr_or_item
from gbmi.utils.testing import TestCase


class TestInit(TestCase):
    def test_deep_getattr_setattr(self):
        config = MAX_OF_10_CONFIG
        self.assertTrue(
            deep_getattr_or_item(config, ["experiment", "model_config", "d_vocab"])
            == config.experiment.model_config.d_vocab
        )
        deep_setattr_or_item(config, ["experiment", "model_config", "d_vocab"], 128)
        self.assertTrue(config.experiment.model_config.d_vocab == 128)

    def test_set_params_immutable(self):
        config = MAX_OF_10_CONFIG
        self.assertExpectedPretty(
            config,
            """\
Config(
    experiment=MaxOfN(
        model_config=HookedTransformerConfig(
            n_layers=1,
            d_model=32,
            n_ctx=10,
            d_head=32,
            n_heads=1,
            d_vocab=128,
            normalization_type=None,
            device='cpu',
            attn_only=True,
            seed=613947648,
            initializer_range=float64(0.1414213562373095),
            d_vocab_out=64,
            n_params=4096
        ),
        seq_len=10
    ),
    train_for=(50000, 'steps'),
    validate_every=None
)""",
        )

        old_vocab = config.experiment.model_config.d_vocab
        new_vocab = 1
        config_2 = set_params(
            config, {("experiment", "model_config", "d_vocab"): new_vocab}
        )
        self.assertEqual(config.experiment.model_config.d_vocab, old_vocab)
        self.assertEqual(config_2.experiment.model_config.d_vocab, new_vocab)

        self.assertExpectedPretty(
            config,
            """\
Config(
    experiment=MaxOfN(
        model_config=HookedTransformerConfig(
            n_layers=1,
            d_model=32,
            n_ctx=10,
            d_head=32,
            n_heads=1,
            d_vocab=128,
            normalization_type=None,
            device='cpu',
            attn_only=True,
            seed=613947648,
            initializer_range=float64(0.1414213562373095),
            d_vocab_out=64,
            n_params=4096
        ),
        seq_len=10
    ),
    train_for=(50000, 'steps'),
    validate_every=None
)""",
        )
        self.assertExpectedPretty(
            config_2,
            """\
Config(
    experiment=MaxOfN(
        model_config=HookedTransformerConfig(
            n_layers=1,
            d_model=32,
            n_ctx=10,
            d_head=32,
            n_heads=1,
            d_vocab=1,
            normalization_type=None,
            device='cpu',
            attn_only=True,
            seed=613947648,
            initializer_range=float64(0.1414213562373095),
            d_vocab_out=64,
            n_params=4096
        ),
        seq_len=10
    ),
    train_for=(50000, 'steps'),
    validate_every=None
)""",
        )

        self.assertExpectedPretty(
            config,
            """\
Config(
    experiment=MaxOfN(
        model_config=HookedTransformerConfig(
            n_layers=1,
            d_model=32,
            n_ctx=10,
            d_head=32,
            n_heads=1,
            d_vocab=128,
            normalization_type=None,
            device='cpu',
            attn_only=True,
            seed=613947648,
            initializer_range=float64(0.1414213562373095),
            d_vocab_out=64,
            n_params=4096
        ),
        seq_len=10
    ),
    train_for=(50000, 'steps'),
    validate_every=None
)""",
        )
