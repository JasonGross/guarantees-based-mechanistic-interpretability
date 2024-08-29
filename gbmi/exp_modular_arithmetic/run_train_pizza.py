from gbmi.exp_modular_arithmetic.train import PIZZA_CONFIG, train_or_load_model

runtime, model = train_or_load_model(PIZZA_CONFIG, force="train")
