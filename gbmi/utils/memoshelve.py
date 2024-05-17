import shelve
from frozendict import frozendict
from pathlib import Path
from typing import Union, TypeVar, Any, Callable, Dict, Optional
from contextlib import contextmanager
from gbmi.utils.hashing import get_hash_ascii

memoshelve_cache: Dict[str, Dict[str, Any]] = {}


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
