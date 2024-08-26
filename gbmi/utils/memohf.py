from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional, Tuple, cast
import pickle
from functools import cache

from datasets import Dataset, DatasetDict, load_dataset
from datasets.data_files import EmptyDatasetError
from datasets.exceptions import DataFilesNotFoundError

from gbmi.utils.hashing import get_hash_ascii

memohf_cache: Dict[str, Dict[str, Any]] = {}


@cache
def _load_dataset_dict_key(*args, **kwargs) -> DatasetDict:
    return cast(
        DatasetDict,
        load_dataset(*args, keep_in_memory=True, **kwargs),
    )


def load_dataset_dict_key(*args, **kwargs) -> DatasetDict:
    try:
        return _load_dataset_dict_key(*args, **kwargs)
    except (EmptyDatasetError, KeyError, DataFilesNotFoundError, ValueError):
        return DatasetDict()


def load_db_from_dataset(
    lazy_dataset: Callable[..., DatasetDict],
    *args,
    default_dataset: Callable[[], DatasetDict] = DatasetDict,
    **kwargs,
) -> Tuple[DatasetDict, dict]:
    try:
        dataset = lazy_dataset(*args, **kwargs)
        return dataset, cast(dict, pickle.loads(dataset["all"]["data"][0]))
    except (EmptyDatasetError, KeyError, DataFilesNotFoundError, ValueError):
        return default_dataset(), {}


def save_db_to_dataset_and_hub(
    repo_id: str,
    dataset_key: str,
    dataset: DatasetDict,
    db: dict,
):
    dataset["all"] = Dataset.from_dict({"data": [pickle.dumps(db)]})
    dataset.push_to_hub(repo_id, config_name=dataset_key)


@cache
def load_db(*args, **kwargs) -> dict:
    _dataset, db = load_db_from_dataset(_load_dataset_dict_key, *args, **kwargs)
    return db


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
        db = load_db(repo_id, name=dataset_key, **kwargs)

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
                dataset, db = load_db_from_dataset(
                    _load_dataset_dict_key, repo_id, name=dataset_key, **kwargs
                )
                save_db_to_dataset_and_hub(repo_id, dataset_key, dataset, db)

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
        dataset, db = load_db_from_dataset(
            _load_dataset_dict_key, repo_id, name=dataset_key, **load_dataset_kwargs
        )
        if key in db:
            del db[key]
            if save:
                save_db_to_dataset_and_hub(repo_id, dataset_key, dataset, db)
    except (EmptyDatasetError, KeyError, DataFilesNotFoundError, ValueError):
        pass
