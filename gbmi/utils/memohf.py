from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional, cast

from datasets import Dataset, DatasetDict, load_dataset
from datasets.data_files import EmptyDatasetError

from gbmi.utils.hashing import get_hash_ascii

memohf_cache: Dict[str, Dict[str, Any]] = {}


def memohf(
    value: Callable,
    repo_id: str,
    dataset_key: str,
    cache: Dict[str, Dict[str, Any]] = memohf_cache,
    get_hash: Callable = get_hash_ascii,
    get_hash_mem: Optional[Callable] = None,
    print_cache_miss: bool = False,
    save: bool = True,
    **kwargs,
):
    """Lightweight memoziation using shelve + in-memory cache"""
    mem_db = cache.setdefault(repo_id, {}).setdefault(dataset_key, {})
    if get_hash_mem is None:
        get_hash_mem = get_hash

    @contextmanager
    def open_db():
        data_modified = False  # Flag to track if data was modified
        try:
            dataset = cast(DatasetDict, load_dataset(repo_id, **kwargs))
            db = cast(dict, dataset[dataset_key].to_dict())
        except EmptyDatasetError:
            db = {}

        def delegate(*args, **kwargs):
            nonlocal data_modified  # Ensure we track modifications
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
                            print(f"Cache miss (huggingface): {key}")
                    elif isinstance(e, (KeyboardInterrupt, SystemExit)):
                        raise e
                    else:
                        print(f"Error {e} in {dataset_key} in {repo_id} with key {key}")
                    if not isinstance(e, (KeyError, AttributeError)):
                        raise e
                    mem_db[mkey] = db[key] = value(*args, **kwargs)
                    data_modified = True
                return mem_db[mkey]

        try:
            yield delegate
        finally:
            if save and data_modified:
                try:
                    dataset = cast(DatasetDict, load_dataset(repo_id, **kwargs))
                except EmptyDatasetError:
                    dataset = DatasetDict()
                dataset[dataset_key] = Dataset.from_dict(db)
                dataset.push_to_hub(repo_id)

    return open_db


def uncache(
    *args,
    repo_id: str,
    dataset_key: str,
    cache: Dict[str, Dict[str, Any]] = memohf_cache,
    get_hash: Callable = get_hash_ascii,
    get_hash_mem: Optional[Callable] = None,
    save: bool = True,
    load_dataset_kwargs: Dict[str, Any] = {},
    **kwargs,
):
    """Lightweight memoziation using shelve + in-memory cache"""
    mem_db = cache.setdefault(repo_id, {}).setdefault(dataset_key, {})
    if get_hash_mem is None:
        get_hash_mem = get_hash

    mkey = get_hash_mem((args, kwargs))
    if mkey in mem_db:
        del mem_db[mkey]

    key = get_hash((args, kwargs))
    try:
        dataset = cast(DatasetDict, load_dataset(repo_id, **load_dataset_kwargs))
        db = cast(dict, dataset[dataset_key].to_dict())
        if key in db:
            del db[key]
            if save:
                dataset[dataset_key] = Dataset.from_dict(db)
                dataset.push_to_hub(repo_id)
    except EmptyDatasetError:
        pass
