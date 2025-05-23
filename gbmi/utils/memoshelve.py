import os
import shelve
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

from dill import Pickler, Unpickler

from gbmi.utils import backup as backup_file
from gbmi.utils.hashing import get_hash_ascii

# monkeypatch shelve as per https://stackoverflow.com/q/52927236/377022
shelve.Pickler = Pickler
shelve.Unpickler = Unpickler

memoshelve_cache: Dict[str, Dict[str, Any]] = {}


def compact(filename: Union[Path, str], backup: bool = True):
    entries = {}
    with shelve.open(filename) as db:
        for k in db.keys():
            entries[k] = db[k]
    if backup:
        backup_name = backup_file(filename)
    else:
        backup_name = None
        os.remove(filename)
    with shelve.open(filename) as db:
        for k in entries.keys():
            db[k] = entries[k]
    if backup_name:
        assert backup_name != filename, backup_name
        os.remove(backup_name)


def memoshelve(
    value: Callable,
    filename: Union[Path, str],
    cache: Dict[str, Dict[str, Any]] = memoshelve_cache,
    get_hash: Callable = get_hash_ascii,
    get_hash_mem: Optional[Callable] = None,
    print_cache_miss: bool = False,
):
    """Lightweight memoziation using shelve + in-memory cache"""
    filename = str(Path(filename).absolute())
    mem_db = cache.setdefault(filename, {})
    if get_hash_mem is None:
        get_hash_mem = get_hash

    @contextmanager
    def open_db():
        with shelve.open(filename) as db:

            def delegate(*args, **kwargs):
                mkey = get_hash_mem((args, kwargs))
                try:
                    return mem_db[mkey]
                except KeyError:
                    if print_cache_miss:
                        print(f"Cache miss (mem): {mkey}")
                    key = get_hash((args, kwargs))
                    try:
                        mem_db[mkey] = db[key]
                    except Exception as e:
                        if isinstance(e, KeyError):
                            if print_cache_miss:
                                print(f"Cache miss (disk): {key}")
                        elif isinstance(e, (KeyboardInterrupt, SystemExit)):
                            raise e
                        else:
                            print(f"Error {e} in {filename} with key {key}")
                        if not isinstance(e, (KeyError, AttributeError)):
                            raise e
                        mem_db[mkey] = db[key] = value(*args, **kwargs)
                    return mem_db[mkey]

            yield delegate

    return open_db


def uncache(
    *args,
    filename: Union[Path, str],
    cache: Dict[str, Dict[str, Any]] = memoshelve_cache,
    get_hash: Callable = get_hash_ascii,
    get_hash_mem: Optional[Callable] = None,
    **kwargs,
):
    """Lightweight memoziation using shelve + in-memory cache"""
    filename = str(Path(filename).absolute())
    mem_db = cache.setdefault(filename, {})
    if get_hash_mem is None:
        get_hash_mem = get_hash

    with shelve.open(filename) as db:
        mkey = get_hash_mem((args, kwargs))
        if mkey in mem_db:
            del mem_db[mkey]
        key = get_hash((args, kwargs))
        if key in db:
            del db[key]


# for decorators
def cache(
    filename: Path | str | None = None,
    cache: Dict[str, Dict[str, Any]] = memoshelve_cache,
    get_hash: Callable = get_hash_ascii,
    get_hash_mem: Optional[Callable] = None,
    print_cache_miss: bool = False,
    disable: bool = False,
):
    def wrap(value: Callable):
        path = Path(filename or f".cache/{value.__name__}.shelve")
        path.parent.mkdir(parents=True, exist_ok=True)

        @wraps(value)
        def wrapper(*args, **kwargs):
            if disable:
                return value(*args, **kwargs)
            else:
                with memoshelve(
                    value,
                    filename=path,
                    cache=cache,
                    get_hash=get_hash,
                    get_hash_mem=get_hash_mem,
                    print_cache_miss=print_cache_miss,
                )() as f:
                    return f(*args, **kwargs)

        return wrapper

    return wrap
