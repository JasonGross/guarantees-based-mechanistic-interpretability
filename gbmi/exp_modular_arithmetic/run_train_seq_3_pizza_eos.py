from gbmi.exp_modular_arithmetic.train import train_or_load_model, PIZZA_CONFIG
from gbmi.utils import set_params

runtime, model = train_or_load_model(
    set_params(
        PIZZA_CONFIG,
        {("experiment", "seq_len"): 3, ("experiment", "use_end_of_sequence"): True},
        post_init=True,
    )
)  # , force="train")
