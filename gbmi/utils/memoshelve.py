import shelve
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
                    except KeyError:
                        if print_cache_miss:
                            print(f"Cache miss (disk): {key}")
                        mem_db[mkey] = db[key] = value(*args, **kwargs)
                    except Exception as e:
                        print(f"Error in {filename} with key {key}")
                        if isinstance(e, (KeyboardInterrupt, SystemExit)):
                            raise e
                        elif isinstance(e, (AttributeError,)):
                            del mem_db[mkey]
                            raise e
                        else:
                            raise e
                    return mem_db[mkey]

            yield delegate

    return open_db
