# Guarantees-Based Mechanistic Interpretability


This is the codebase for the [_Guarantees-Based Mechanistic Interpretability_](https://www.cambridgeaisafety.org/mars/jason-gross) MARS stream.
Successor to https://github.com/JasonGross/neural-net-coq-interp.

## Writeups

### Compact Proofs of Model Performance via Mechanistic Interpretability

```bibtex
@misc{gross2024compact,
  author      = {Jason Gross and Rajashree Agrawal and Thomas Kwa and Euan Ong and Chun Hei Yip and Alex Gibson and Soufiane Noubir and Lawrence Chan},
  title       = {Compact Proofs of Model Performance via Mechanistic Interpretability},
  year        = {2024},
  month       = {June},
  doi         = {10.48550/arxiv.2406.11779},
  eprint      = {2406.11779},
  url         = {https://arxiv.org/abs/2406.11779},
  eprinttype  = {arXiv},
}
```
Abstract:
> In this work, we propose using mechanistic interpretability – techniques for reverse engineering model weights into human-interpretable algorithms – to derive and compactly prove formal guarantees on model performance. We prototype this approach by formally proving lower bounds on the accuracy of 151 small transformers trained on a Max-of-K task. We create 102 different computer-assisted proof strategies and assess their length and tightness of bound on each of our models. Using quantitative metrics, we find that shorter proofs seem to require and provide more mechanistic understanding. Moreover, we find that more faithful mechanistic understanding leads to tighter performance bounds. We confirm these connections by qualitatively examining a subset of our proofs. Finally, we identify compounding structureless noise as a key challenge for using mechanistic interpretability to generate compact proofs on model performance.

- [Blog post](https://www.alignmentforum.org/posts/bRsKimQcPTX3tNNJZ/compact-proofs-of-model-performance-via-mechanistic)
- [arXiv](https://arxiv.org/abs/2406.11779)
- [ICML 2024 Mechanistic Interpretability Workshop Spotlight](https://openreview.net/forum?id=4B5Ovl9MLE)

## Setup

The code can be run under any environment with Python 3.9 and above.

We use [poetry](https://python-poetry.org) for dependency management, which can be installed following the instructions [here](https://python-poetry.org/docs/#installation).

To build a virtual environment with the required packages, simply run

```bash
poetry config virtualenvs.in-project true
poetry install
```

Notes
- On some systems you may need to set the environment variable `PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring` to avoid keyring-based errors.
- The first line tells poetry to create the virtual environment in the project directory, which allows VS Code to find the virtual environment.
- If you are using caches from other machines, if you see errors like "dbm.error: db type is dbm.gnu, but the module is not available", you can probably solve the issue by following instructions from [StackOverflow](https://stackoverflow.com/a/49597001/377022):
    - `sudo apt-get install libgdbm-dev python3-gdbm`
    - If you are using `conda` or some other Python version management, you can inspect the output of `dpkg -L python3-gdbm` and copy the `lib-dynload/_gdbm.cpython-*-x86_64-linux-gnu.so` file to the corresponding `lib/` directory associated to the python you are using.

## Running notebooks

To open a Jupyter notebook, run

```bash
poetry run jupyter lab
```

If this doesn't work (e.g. you have multiple Jupyter kernels already installed on your system), you may need to make a new kernel for this project:

```bash
poetry run python -m ipykernel install --user --name=gbmi
```

## Training models

Models for existing experiments can be trained by running e.g.

```bash
poetry run python -m gbmi.exp_max_of_n.train
```

or by running e.g.

```python
from gbmi.exp_max_of_n.train import MAX_OF_10_CONFIG
from gbmi.model import train_or_load_model

rundata, model = train_or_load_model(MAX_OF_10_CONFIG)
```

from a Jupyter notebook.

This function will attempt to pull a trained model with the specified config from Weights and Biases; if such a model does not exist, it will train the relevant model and save the weights to Weights and Biases.

## Adding new experiments

The convention for this codebase is to store experiment-specific code in an `exp_[NAME]/` folder, with
- `exp_[NAME]/analysis.py` storing functions for visualisation / interpretability
- `exp_[NAME]/verification.py` storing functions for verification
- `exp_[NAME]/train.py` storing training / dataset code

See the `exp_template` directory for more details.

## Adding dependencies

To add new dependencies, run `poetry add my-package`.

## Code Style

We use black to format our code.
To set up the pre-commit hooks that enforce code formatting, run

```bash
make pre-commit-install
```


## Tests

This codebase advocates for [expect tests](https://blog.janestreet.com/the-joy-of-expect-tests) in machine learning, and as such uses @ezyang's [expecttest](https://github.com/ezyang/expecttest) library for unit and regression tests.

[TODO: add tests?]
