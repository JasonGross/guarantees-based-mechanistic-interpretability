from __future__ import annotations

import copy
import logging
import os
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass, is_dataclass
from functools import partial
from itertools import islice
from pathlib import Path
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    Generic,
    Hashable,
    Iterable,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
import torch
from jaxtyping import Float, Integer
from lightning import Callback
from numpy.random import Generator
from torch import Tensor
from torch.utils.data import Dataset, IterableDataset
from transformer_lens import HookedTransformer
from transformer_lens.components import (
    MLP,
    Attention,
    LayerNorm,
    LayerNormPre,
    RMSNorm,
    RMSNormPre,
)

from gbmi.utils import ein
from gbmi.utils.dataclass import dataclass_map
from gbmi.utils.hashing import get_hash

PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_WANDB_ENTITY = "gbmi"

A = TypeVar("A")
T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")

if sys.version_info >= (3, 12):
    from itertools import batched
else:

    class batched(Iterator[Tuple[T, ...]], Generic[T]):
        def __init__(self, iterable: Iterable[T], n: int):
            if n < 1:
                raise ValueError("n must be at least one")
            self.iterator = iter(iterable)
            self.n = n

        def __iter__(self) -> "batched[T]":
            return self

        def __next__(self) -> Tuple[T, ...]:
            batch = tuple(islice(self.iterator, self.n))
            if not batch:
                raise StopIteration
            return batch


def get_trained_model_dir(create: bool = True) -> Path:
    """
    Returns the base path for saving models. If `save_in_google_drive` is True, returns the path to the Google Drive
    folder where models are saved. Otherwise, returns the path to the local folder where models are saved.
    """
    pth_base_path = PROJECT_ROOT / "trained-models"

    if create and not os.path.exists(pth_base_path):
        os.makedirs(pth_base_path)

    return pth_base_path


def default_device(deterministic: bool = False) -> str:
    return "cuda" if torch.cuda.is_available() and not deterministic else "cpu"


def shuffle_data(data, rng: Generator):
    indices = np.array(range(len(data)))
    rng.shuffle(indices)
    data = data[indices]
    return data


def add_eos(
    x: Float[Tensor, "b n"], eos: int  # noqa: F722
) -> Float[Tensor, "b (n + 1)"]:  # noqa: F722, F821
    return ein.array(lambda i: torch.cat([x[i], torch.tensor([eos])]))


def add_bos(
    bos: int,
    x: Float[Tensor, "b n"],  # noqa: F722
) -> Float[Tensor, "b (n + 1)"]:  # noqa: F722, F821
    return ein.array(lambda i: torch.cat([torch.tensor([bos]), x[i]]))


class MetricsCallback(Callback):
    """PyTorch Lightning callback to save metrics in a Python object."""

    def __init__(self) -> None:
        super().__init__()
        self.metrics: List[Dict[str, float]] = []
        self.steps = 0

    def log_metrics(self, trainer):
        metrics = copy.deepcopy(trainer.callback_metrics)
        metrics["epoch"] = trainer.current_epoch
        metrics["step"] = self.steps
        self.metrics.append(metrics)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.steps += 1

    def on_train_epoch_end(self, trainer, pl_module):
        self.log_metrics(trainer)

    def on_validation_epoch_end(self, trainer, module):
        self.log_metrics(trainer)


class SingleTensorDataset(IterableDataset[Tensor]):
    r"""Dataset wrapping a single tensor.

    Each sample will be retrieved by indexing tensor along the first dimension.

    Args:
        tensor (Tensor): a tensor to be contained within dataset
    """

    tensor: Tensor

    def __init__(self, tensor: Tensor) -> None:
        self.tensor = tensor

    def __getitem__(self, index):
        return self.tensor[index]

    def __len__(self):
        return self.tensor.size(0)

    def __iter__(self):
        return iter(self.tensor)


class TupleCollectionDataset(Dataset[Tuple[Collection, ...]]):
    r"""Dataset wrapping a tuple of arrays.

    Each sample will be retrieved by indexing the arguments along the first dimension.
    """

    data: Tuple[Collection, ...]

    def __init__(self, *data: Collection) -> None:
        self.data = data
        assert all(
            len(data[0]) == len(datum) for datum in data
        ), f"Size mismatch: {data} ({set(map(len, data))})"

    def __getitem__(self, index):
        return tuple(datum[index] for datum in self.data)

    def __len__(self):
        return len(self.data[0])


# Function to check if current directory is a Git repository
def is_git_repo():
    try:
        subprocess.check_output(["git", "rev-parse", "--is-inside-work-tree"])
        return True
    except subprocess.CalledProcessError:
        return False


# Function to handle the warning, prompt, and error logic
def handle_size_warnings_and_prompts(
    error_on_diff_larger_than=None, prompt_if_diff_larger_than=1024
):
    if not is_git_repo():
        return
    size_kb = len(subprocess.check_output(["git", "diff"])) / 1024

    print(
        f"Warning: Code-saving on wandb may upload a diff/file of size {size_kb} KiB",
        file=sys.stderr,
    )

    if error_on_diff_larger_than is not None and size_kb > error_on_diff_larger_than:
        raise Exception(
            f"The size ({size_kb} KiB) is larger than the specified limit ({error_on_diff_larger_than} KB)."
        )

    if prompt_if_diff_larger_than is not None and size_kb > prompt_if_diff_larger_than:
        input("Press enter to continue or Ctrl+C to break")


def log_softmax(x: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
    """
    log_softmax is only precise to around 2e-7, cf https://github.com/pytorch/pytorch/issues/113708
    we can get better precision by using log1p
    """
    x_argmax = torch.argmax(x, dim=dim, keepdim=True)
    x_max = torch.take_along_dim(x, indices=x_argmax, dim=dim)
    finite_max_mask = x_max.isfinite()
    x_max = torch.where(~finite_max_mask, 0, x_max)
    tmp = x - x_max
    exp_tmp = torch.exp(tmp)
    # we know that exp_tmp at the location of the max is either 1 or infinite,
    # depending on finite_max_mask, so we can set it to zero and use log1p
    exp_tmp_max = torch.take_along_dim(exp_tmp, indices=x_argmax, dim=dim)
    exp_tmp_max = torch.where(finite_max_mask, 0, exp_tmp_max)
    if dim is not None:
        exp_tmp = torch.scatter(
            exp_tmp,
            dim=dim,
            index=x_argmax,
            src=exp_tmp_max,
        )
    else:
        exp_tmp_flattened = torch.scatter(
            torch.flatten(exp_tmp),
            dim=0,
            index=torch.flatten(x_argmax),
            src=torch.flatten(exp_tmp_max),
        )
        exp_tmp = torch.reshape(exp_tmp_flattened, exp_tmp.shape)
    return tmp - torch.log1p(torch.sum(exp_tmp, dim=dim, keepdim=True))


def cross_entropy(
    input: Union[
        Float[Tensor, "C"],  # noqa: F821
        Float[Tensor, "N C"],  # noqa: F722
        Float[Tensor, "N C *ds"],  # noqa: F722
    ],
    target: Union[
        Float[Tensor, ""],  # noqa: F722
        Float[Tensor, "N"],  # noqa: F821
        Float[Tensor, "N *ds"],  # noqa: F821
    ],
    weight: Optional[Tensor] = None,
    ignore_index: int = -100,
    reduction: Literal["none", "mean", "sum"] = "mean",
    label_smoothing: float = 0.0,
) -> Union[
    Float[Tensor, ""], Float[Tensor, "N"], Float[Tensor, "N *ds"]  # noqa: F821, F722
]:
    # like torch.nn.functional.cross_entropy, but using log1p for better precision
    # mostly written by ChatGPT 4

    # Step 3: Compute log softmax of inputs
    log_probs = torch.nn.functional.log_softmax(input, dim=1)

    if target.dim() == input.dim():
        # Assuming target contains class probabilities
        target_prob = target
    else:
        # Assuming target contains class indices
        target_prob = torch.nn.functional.one_hot(target, num_classes=input.size(1))
        if ignore_index is not None and ignore_index >= 0:
            ignore_mask = target == ignore_index
            target_prob[ignore_mask] = 0

    # Step 2: Apply label smoothing if required
    if label_smoothing > 0:
        n_classes = input.size(1)
        if (
            target.dim() < input.dim()
            and ignore_index is not None
            and ignore_index >= 0
        ):
            n_classes -= 1
        smooth_value = label_smoothing / n_classes
        target_prob = (1 - label_smoothing) * target_prob + smooth_value

    # Step 4: Apply class weights if provided
    if weight is not None:
        raise NotImplementedError(
            "Weighted cross entropy not implemented, I'm not sure how to get it right"
        )
        log_probs = log_probs * weight

    # Step 5: Compute the loss
    loss = -(target_prob * log_probs).sum(dim=1)

    # Step 7: Apply reduction
    if reduction == "mean":
        if (
            target.dim() < input.dim()
            and ignore_index is not None
            and ignore_index >= 0
        ):
            loss = loss.sum() / (ignore_mask.logical_not().sum())
        else:
            loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


cross_entropy.__doc__ = torch.nn.functional.cross_entropy.__doc__


def deep_getattr_or_item(obj: object, key: Union[str, Sequence[str]]) -> Any:
    if isinstance(key, str):
        return getattr_or_item(obj, key)
    elif len(key) == 1:
        return getattr_or_item(obj, key[0])
    else:
        return deep_getattr_or_item(getattr_or_item(obj, key[0]), key[1:])


def deep_setattr_or_item(
    obj: object, key: Union[str, Sequence[str]], value: Any
) -> None:
    if isinstance(key, str):
        setattr_or_item(obj, key, value)
    elif len(key) == 1:
        setattr_or_item(obj, key[0], value)
    else:
        deep_setattr_or_item(getattr_or_item(obj, key[0]), key[1:], value)


def setattr_or_item(obj: object, key: str, value: Any) -> None:
    if hasattr(obj, "__setitem__"):  # dict-like
        obj[key] = value  # type: ignore
    else:
        setattr(obj, key, value)


def getattr_or_item(obj: object, key: str) -> Any:
    if hasattr(obj, "__getitem__"):  # dict-like
        return obj[key]  # type: ignore
    else:
        return getattr(obj, key)


def set_params(
    cfg: T,
    params: Dict[Union[str, Sequence[str]], Any],
    *,
    warn_if_not_default: bool = False,
    post_init: bool = False,
) -> T:
    # TODO: warn if not default
    cfg = copy.deepcopy(cfg)
    assert not warn_if_not_default, "Not implemented"
    for k, v in params.items():
        deep_setattr_or_item(cfg, k, v)
    if hasattr(cfg, "__post_init__") and post_init:
        cfg.__post_init__()
    return cfg

    #         if (
    #             config.experiment.model_config.seed != config.seed
    #             and config.experiment.model_config.seed is not None
    #         ):
    #             logging.warning(
    #                 f"Overwriting transformer seed (set to {config.experiment.model_config.seed}) to {config.seed}"
    #             )
    #         config.experiment.model_config.seed = config.seed


def reseed(x: Hashable, label: str) -> int:
    # 4 bytes make an int32!
    return int.from_bytes(get_hash((x, label))[:4], byteorder="little")


def dropnan(x: Tensor) -> Tensor:
    return x[~torch.isnan(x)]


def map_values(f: Callable[[V], T], d: dict[K, V]) -> dict[K, T]:
    return {k: f(v) for k, v in d.items()}


_pre_subsuperscripts = " !#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
_superscripts0 = " !#$%&'⁽⁾*⁺,⁻./⁰¹²³⁴⁵⁶⁷⁸⁹:;<⁼>?@ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾQᴿˢᵀᵁⱽᵂˣʸᶻ[\\]^_`ᵃᵇᶜᵈᵉᶠᵍʰᶦʲᵏˡᵐⁿᵒᵖᑫʳˢᵗᵘᵛʷˣʸᶻ{|}~"
_subscripts0 = " !#$%&'₍₎*₊,₋./₀₁₂₃₄₅₆₇₈₉:;<₌>?@ₐ₈CDₑբGₕᵢⱼₖₗₘₙₒₚQᵣₛₜᵤᵥᵥᵥₓᵧZ[\\]^_`ₐ₆꜀ₔₑբ₉ₕᵢⱼₖₗₘₙₒₚqᵣₛₜᵤᵥᵥᵥₓᵧ₂{|}~"
_superscript = {k: v for k, v in zip(_pre_subsuperscripts, _superscripts0) if k != v}
_subscript = {k: v for k, v in zip(_pre_subsuperscripts, _subscripts0) if k != v}
_unsuperscript = {v: k for k, v in _superscript.items()}
_unsubscript = {v: k for k, v in _subscript.items()}


def subscript(s: str) -> str:
    """Converts a string of digits to subscript."""
    return "".join(map(_subscript.__getitem__, s))


def superscript(s: str) -> str:
    """Converts a string of digits to superscript."""
    return "".join(map(_superscript.__getitem__, s))


def unsuperscript(s: str) -> str:
    """Converts a string of superscript digits to regular digits."""
    return "".join(map(_unsuperscript.__getitem__, s))


def unsubscript(s: str) -> str:
    """Converts a string of subscript digits to regular digits."""
    return "".join(map(_unsubscript.__getitem__, s))


@torch.no_grad()
def shuffle_tensor(t: Tensor) -> Tensor:
    return t.flatten()[torch.randperm(t.numel())].reshape(t.shape)


@torch.no_grad()
def shuffle_tensors(*ts: Tensor) -> Iterator[Tensor]:
    for t in ts:
        yield shuffle_tensor(t)


def zero_biases_of_module(module: torch.nn.Module):
    for name, param in module.named_parameters():
        if "b_" in name:
            param.requires_grad = False


def zero_biases_of_HookedTransformer(
    model: HookedTransformer,
    biases_to_zero: Collection[
        Literal["Embed", "Unembed", "PosEmbed", "LayerNorm", "Attention", "MLP"]
    ],
):
    classes = {
        "LayerNorm": (
            torch.nn.LayerNorm,
            RMSNorm,
            RMSNormPre,
            LayerNorm,
            LayerNormPre,
        ),
        "Attention": (torch.nn.MultiheadAttention, Attention),
        "MLP": (MLP,),
    }
    for name in biases_to_zero:
        match name:
            case "Embed":
                zero_biases_of_module(model.embed)
            case "Unembed":
                zero_biases_of_module(model.unembed)
            case "PosEmbed":
                zero_biases_of_module(model.pos_embed)
            case _:
                for module in model.modules():
                    if any(isinstance(module, cls) for cls in classes[name]):
                        zero_biases_of_module(module)


def bits_of_type(dtype) -> int:
    try:
        return dtype.itemsize * 8
    except AttributeError:
        pass
    try:
        return np.dtype(dtype).itemsize * 8
    except TypeError:
        pass
    raise TypeError(
        f"Unsupported dtype {dtype}. Please provide a NumPy or PyTorch floating point type."
    )


def to_device(
    obj: A,
    device: Union[str, torch.device],
    *,
    ignore_types: Tuple[type, ...] = (type(None), int, str, float),
    print_details: bool = True,
) -> A:
    """
    Recursively moves any subobject with a `.to` method to the specified device.
    This function skips objects of types specified in `ignore_types`.

    Args:
        obj (A): The object to move.
        device (Union[str, torch.device]): The device to move the object to.
        ignore_types (Sequence[type]): A sequence of types to ignore.

    Returns:
        A: The object with elements moved to the specified device.
    """
    if isinstance(obj, ignore_types):
        return obj

    if hasattr(obj, "to"):
        # if hasattr(obj, "device"):
        #     print(f"Moving object of type {type(obj)} from device {obj.device} to {device}")
        if isinstance(device, HookedTransformer):
            return obj.to(device, print_details=print_details)
        return obj.to(device)

    if isinstance(obj, (list, tuple)):
        return type(obj)(
            to_device(
                item, device, ignore_types=ignore_types, print_details=print_details
            )
            for item in obj
        )

    if isinstance(obj, dict):
        return type(obj)(
            {
                key: to_device(
                    value,
                    device,
                    ignore_types=ignore_types,
                    print_details=print_details,
                )
                for key, value in obj.items()
            }
        )

    if is_dataclass(obj):
        return dataclass_map(
            partial(
                to_device,
                device=device,
                ignore_types=ignore_types,
                print_details=print_details,
            ),
            obj,
        )

    logging.warning("Skipping object of type %s: %s", type(obj), obj)

    return obj


@contextmanager
def patch_map(
    obj: object, mapping: dict[str | Sequence[str], Callable] = {}, **kwargs: Callable
):
    """
    Temporarily patches an object with new methods or attributes.

    Args:
        obj (object): The object to patch.
        mapping (dict[str | Sequence[str], Callable]): A mapping of attribute names to functions.
    """
    mapping = mapping | kwargs
    original = {}
    for key, upd in mapping.items():
        original[key] = value = deep_getattr_or_item(obj, key)
        deep_setattr_or_item(obj, key, upd(value))
    try:
        yield
    finally:
        for key, value in original.items():
            deep_setattr_or_item(obj, key, value)


def patch(obj: object, mapping: dict[str | Sequence[str], Any] = {}, **kwargs: Any):
    """
    Temporarily patches an object with new methods or attributes.

    Args:
        obj (object): The object to patch.
        mapping (dict[str | Sequence[str], Any]): A mapping of attribute names to values.
    """
    mapping = mapping | kwargs
    return patch_map(obj, {k: lambda x: v for k, v in mapping.items()})


def is_valid_torch_dtype_for(*values, dtype: torch.dtype) -> bool:
    vmin, vmax = min(values), max(values)
    if isinstance(vmin, torch.Tensor):
        vmin = vmin.item()
    if isinstance(vmax, torch.Tensor):
        vmax = vmax.item()
    try:
        torch.tensor(vmin, dtype=dtype)
        torch.tensor(vmax, dtype=dtype)
        return True
    except (TypeError, RuntimeError):
        return False


def is_valid_torch_dtype(dtype: Any) -> bool:
    try:
        torch.tensor(0, dtype=dtype)
        return True
    except (TypeError, RuntimeError):
        return False


def get_int_dtypes(
    mod,
    only_signed: bool = False,
    allow_signed: bool = False,
    check: Optional[Callable[[int, Any], bool]] = lambda k, dtype: np.log2(k) % 1 == 0,
) -> tuple:
    prefixes = (["uint"] if not only_signed else []) + (
        ["int"] if only_signed or allow_signed else []
    )
    dtype_names = {
        (int(n[len(prefix) :]), prefix): n
        for prefix in prefixes
        for n in dir(mod)
        if n.startswith(prefix) and n[len(prefix) :].isdigit()
    }
    if check:
        dtype_names = {
            (bits, p): v
            for (bits, p), v in dtype_names.items()
            if check(bits, getattr(mod, v))
        }
    return tuple([getattr(mod, dtype_names[k]) for k in sorted(dtype_names.keys())])


def smallest_dtype_holding(
    *values: int, allow_negative: bool = True, only_signed: bool = False
) -> torch.dtype:
    """
    Returns the smallest torch dtype that can hold the given integer value.

    Parameters:
    - *values (int): The integer values to check.
    - allow_negative (bool): Whether the value is allowed to be negative.

    Returns:
    - torch.dtype: The smallest dtype that can hold the value.
    """

    # Define possible data types to check in order of increasing size
    dtypes = get_int_dtypes(
        torch, allow_signed=allow_negative, only_signed=only_signed, check=None
    )

    vmin, vmax = min(values), max(values)

    for dtype in dtypes:
        if is_valid_torch_dtype_for(vmin, vmax, dtype=dtype):
            return dtype

    raise ValueError(
        f"Values {values} ({vmin} to {vmax}) are out of range for any standard PyTorch integer dtype ({dtypes})."
    )


def compress_int_tensor(
    v: Integer[Tensor, "..."], allow_negative: bool = True, only_signed: bool = False
) -> Integer[Tensor, "..."]:
    dtype: torch.dtype = smallest_dtype_holding(
        v.min().item(),
        v.max().item(),
        allow_negative=allow_negative,
        only_signed=only_signed,
    )
    return v.to(dtype=dtype)


def backup(filename: str | Path, ext: str = ".bak") -> Optional[Path]:
    filename = Path(filename)
    assert ext != ""
    backup_name = filename.with_suffix(filename.suffix + ext)
    if filename.exists():
        if backup_name.exists():
            backup(backup_name, ext=ext)
            assert not backup_name.exists()
            filename.rename(backup_name)
            return backup_name
    return None


import sys


def deep_getsizeof(obj: object, seen: Optional[set] = None) -> int:
    """Recursively find the size of objects, including contained objects."""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0  # Avoid double-counting objects already visited.
    seen.add(obj_id)

    if isinstance(obj, dict):
        size += sum(
            deep_getsizeof(k, seen) + deep_getsizeof(v, seen) for k, v in obj.items()
        )
    elif isinstance(obj, (list, tuple, set, frozenset)):
        size += sum(deep_getsizeof(i, seen) for i in obj)

    return size
