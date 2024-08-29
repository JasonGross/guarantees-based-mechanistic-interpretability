# %%
# Written in part by ChatGPT 4
# from __future__ import  annotations
import dataclasses
import itertools
from dataclasses import fields, is_dataclass, replace
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    List,
    Literal,
    Mapping,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

from beartype.door import LiteralTypeHint, TypeHint, UnionTypeHint

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

    A = TypeVar("A", bound=DataclassInstance)
else:
    A = TypeVar("A")

V = TypeVar("V")


class DataclassMapping(Generic[V], Mapping[str, V]):
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __iter__(self):
        return iter(dataclasses.asdict(self))  # type: ignore

    def __len__(self):
        return len(dataclasses.asdict(self))  # type: ignore


def get_values_of_type(ty: type) -> List[Any]:
    """Get all possible values for a field type."""
    # Handle basic types directly
    field_type_hint = TypeHint(ty)
    if ty is bool:
        return [True, False]
    elif ty is type(None):  # Comparing directly to NoneType
        return [None]

    # Handle Union types, including recursive calls for nested Unions or Literals
    origin = get_origin(ty)
    if isinstance(field_type_hint, LiteralTypeHint) or origin is Literal:
        return list(field_type_hint.args or get_args(ty))
    elif isinstance(field_type_hint, UnionTypeHint) or origin is Union:
        values = []
        for arg in field_type_hint.args or get_args(ty):
            values.extend(get_values_of_type(arg))
        # Python is stupid and thinks 1 == True
        return [v for v, _ in set((v, type(v)) for v in values)]

    raise ValueError(
        f"Unsupported type: {ty} ({field_type_hint}, origin: {origin}, args: {get_args(ty)}"
    )


def enumerate_dataclass_values(dc: type):
    # Generate all possible values for each field
    field_values = []
    for field in fields(dc):
        try:
            values = get_values_of_type(field.type)
            field_values.append(values)
        except ValueError as e:
            raise ValueError(f"Error processing {field.name}: {e}")

    # Create combinations of all field values
    for combination in itertools.product(*field_values):
        yield dc(*combination)


def dataclass_map(func: Callable, obj: A) -> A:
    """
    Applies a given function to all fields of a dataclass and returns a new instance with updated fields.

    Args:
        obj (A): The dataclass instance.
        func (Callable): The function to apply to each field.

    Returns:
        A: A new instance of the dataclass with updated fields.
    """
    if not is_dataclass(obj):
        raise ValueError("The provided object is not a dataclass instance.")

    updated_fields = {
        field.name: func(getattr(obj, field.name)) for field in fields(obj)
    }
    return replace(obj, **updated_fields)
