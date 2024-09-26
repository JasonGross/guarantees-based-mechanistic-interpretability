from gbmi.exp_modular_arithmetic.train import PIZZA_CONFIG, train_or_load_model
from gbmi.utils import set_params

runtime, model = train_or_load_model(
    set_params(
        PIZZA_CONFIG,
        {
            ("experiment", "seq_len"): 3,
            ("experiment", "use_end_of_sequence"): True,
            ("experiment", "d_mlp"): 1024,
        },
        post_init=True,
    ),
    # force="train",
    # force="load",
)
