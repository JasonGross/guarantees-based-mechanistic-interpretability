"""
Generate stable hashes for Python data objects.
Contains no business logic.

The hashes should be stable across interpreter implementations and versions.

Supports dataclass instances, datetimes, and JSON-serializable objects.

Empty dataclass fields are ignored, to allow adding new fields without
the hash changing. Empty means one of: None, '', (), [], or {}.

The dataclass type is ignored: two instances of different types
will have the same hash if they have the same attribute/value pairs.

"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
from collections.abc import Collection
from typing import Any, Callable, Mapping, Optional, Union
from typing import Dict
from functools import partial
import base64

import torch
from transformer_lens import HookedTransformer
import numpy

# Implemented for https://github.com/lemon24/reader/issues/179


# The first byte of the hash contains its version,
# to allow upgrading the implementation without changing existing hashes.
# (In practice, it's likely we'll just let the hash change and update
# the affected objects again; nevertheless, it's good to have the option.)
#
# A previous version recommended using a check_hash(thing, hash) -> bool
# function instead of direct equality checking; it was removed because
# it did not allow objects to cache the hash.

_VERSION = 0
_EXCLUDE = "_hash_exclude_"

ExcludeFilter = Union[
    None,
    bool,
    Collection[str],
    Mapping[str, bool],
    Callable[[object], Union[None, bool, Collection[str], Mapping[str, bool]]],
]


def get_hash(
    thing: object,
    exclude_filter: ExcludeFilter = None,
    dictify_by_default: bool = False,
) -> bytes:
    """
    Returns a stable hash for the given object.

    exclude_filter is a callable that takes an object and returns either:
    - False indicates exclusion
    - True indicates inclusion
    - None indicates default behavior (inclusion or recursive application of a parent filter)
    - Collection[str] indicates exclusion of the listed attributes
    - Mapping[str, ExcludeFilter] indicates an exclusion filter to apply to specified attributes
    - Callable[[object], ExcludeFilter] indicates a filter to apply recursively to any object which is mapped to None by
    """
    prefix = _VERSION.to_bytes(1, "big")
    digest = hashlib.md5(
        _json_dumps(
            thing, exclude_filter=exclude_filter, dictify_by_default=dictify_by_default
        ).encode("utf-8")
    ).digest()
    return prefix + digest[:-1]


def get_hash_ascii(
    thing: object,
    exclude_filter: ExcludeFilter = None,
    dictify_by_default: bool = False,
) -> str:
    return base64.b64encode(
        get_hash(
            thing, exclude_filter=exclude_filter, dictify_by_default=dictify_by_default
        )
    ).decode("ascii")


def _json_dumps(
    thing: object,
    exclude_filter: ExcludeFilter = None,
    dictify_by_default: bool = False,
) -> str:
    return json.dumps(
        thing,
        default=partial(
            _json_default,
            exclude_filter=exclude_filter,
            dictify_by_default=dictify_by_default,
        ),
        # force formatting-related options to known values
        ensure_ascii=False,
        sort_keys=True,
        indent=None,
        separators=(",", ":"),
    )


def _json_default(
    thing: object,
    exclude_filter: ExcludeFilter = None,
    dictify_by_default: bool = False,
) -> Any:
    if exclude_filter is True:
        return None
    try:
        return _dataclass_dict(thing, exclude_filter=exclude_filter)
    except TypeError:
        pass
    if isinstance(thing, datetime.datetime):
        return thing.isoformat(timespec="microseconds")
    elif isinstance(thing, torch.device) or isinstance(thing, torch.dtype):
        return str(thing)
    elif isinstance(thing, set) or isinstance(thing, frozenset):
        return _json_dumps(
            sorted(thing),
            exclude_filter=exclude_filter,
            dictify_by_default=dictify_by_default,
        )
    elif isinstance(thing, torch.Tensor) or isinstance(thing, numpy.ndarray):
        return _json_dumps(
            thing.tolist(),
            exclude_filter=exclude_filter,
            dictify_by_default=dictify_by_default,
        )
    elif isinstance(thing, HookedTransformer):
        return _json_dumps(
            thing.to("cpu", print_details=False).__dict__,
            exclude_filter=exclude_filter,
            dictify_by_default=dictify_by_default,
        )
    elif isinstance(thing, type):
        return f"{thing.__module__}.{thing.__name__}"
    elif (
        hasattr(thing, "__dict__")
        and dictify_by_default
        and (not isinstance(thing, Callable) or thing.__dict__)
    ):
        return _json_dumps(
            thing.__dict__,
            exclude_filter=exclude_filter,
            dictify_by_default=dictify_by_default,
        )
    elif (
        isinstance(thing, Callable)
        and not thing.__dict__
        and thing.__closure__ is None
        and thing.__module__ is not None
    ):
        return f"{thing.__module__}.{thing.__name__}"
    raise TypeError(f"Object {thing} of type {type(thing)} is not JSON serializable")


def getattr_or_exclude(
    field_name: str, thing: object, exclude_filter: ExcludeFilter = None
) -> Optional[Any]:
    # first we check for any fields listed in the _EXCLUDE attribute
    if field_name in getattr(thing, _EXCLUDE, ()):
        return None
    # now we exclude None and empty collections
    value = getattr(thing, field_name)
    if value is None or (isinstance(value, Collection) and len(value) == 0):
        return None

    # if the exclude filter contains no exclusions, we're done
    if exclude_filter is None or exclude_filter is False:
        return value
    # if the exclude filter is a collection of field names or maps field names to booleans, we can just check the field name
    elif isinstance(exclude_filter, Collection):
        return value if field_name not in exclude_filter else None
    elif isinstance(exclude_filter, Mapping):
        return value if not exclude_filter[field_name] else None
    # if the exclude filter is callable, we will pre-emptively exclude the child if the filter returns True on the child
    elif callable(exclude_filter):
        value = getattr_or_exclude(field_name, thing, exclude_filter(thing))
        if value is None or exclude_filter(value) is True:
            return None
        return value
    else:
        assert False


def _dataclass_dict(
    thing: object, exclude_filter: ExcludeFilter = None
) -> Dict[str, Any]:
    # we could have used dataclasses.asdict()
    # with a dict_factory that drops empty values,
    # but asdict() is recursive and we need to intercept and check
    # the _hash_exclude_ of nested dataclasses;
    # this way, json.dumps() does the recursion instead of asdict()

    # raises TypeError for non-dataclasses
    fields = dataclasses.fields(thing)
    # ... but doesn't for dataclass *types*
    if isinstance(thing, type):
        raise TypeError("got type, expected instance")

    rv = {}
    for field in fields:
        value = getattr_or_exclude(field.name, thing, exclude_filter=exclude_filter)
        if value is not None:
            rv[field.name] = value

    return {"__name__": thing.__class__.__name__, **rv}
