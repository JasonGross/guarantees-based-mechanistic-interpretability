import pickle
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional, cast

from datasets import Dataset, DatasetDict, load_dataset
from datasets.data_files import EmptyDatasetError
from datasets.exceptions import DataFilesNotFoundError

from gbmi.utils.hashing import get_hash_ascii

PUSH_INTERVAL: float = (
    10  # Push only if more than 10 seconds have passed since the last push
)
USE_ALL_DATA: bool = True

last_push_time: Dict[str, float] = {}
memohf_cache: Dict[str, Dict[str, Any]] = {}


def should_push(repo_id: str) -> bool:
    """Determines if we should push based on the last push time."""
    last_time = last_push_time.get(repo_id, 0)
    return time.time() - last_time > PUSH_INTERVAL


def update_last_push_time(repo_id: str):
    """Updates the last push time for the given repo."""
    last_push_time[repo_id] = time.time()


class HFOpenDictLike(dict):
    """A dict-like object that supports push_to_hub."""

    def __init__(self, dataset: DatasetDict, repo_id: str):
        super().__init__()
        self.repo_id = repo_id
        self.dataset = dataset
        self.update(self._load_db())  # Load the data and update the internal dict
        self.modified = False

    def _load_db(self) -> Dict[str, Any]:
        """Loads the dataset data from the Hugging Face hub based on the mode."""
        return {
            key: pickle.loads(self.dataset[key]["data"][0])
            for key in self.dataset.keys()
        }

    def push_to_hub(self):
        """Pushes the current state of the database back to the Hugging Face hub."""
        for key, data in self.items():
            serialized_data = pickle.dumps(data)
            self.dataset[key] = Dataset.from_dict({"data": [serialized_data]})
        self.dataset.push_to_hub(self.repo_id)
        update_last_push_time(self.repo_id)
        self.modified = False

    def __setitem__(self, key: Any, value: Any):
        """Set a key in the dictionary and mark as modified."""
        super().__setitem__(key, value)
        self.modified = True

    def __delitem__(self, key: Any):
        """Delete a key in the dictionary and mark as modified."""
        super().__delitem__(key)
        self.modified = True

    def clear(self):
        """Clear all items in the dictionary and mark as modified."""
        super().clear()
        self.modified = True

    def pop(self, key: Any, default: Any = None):
        """Pop a key from the dictionary and mark as modified."""
        value = super().pop(key, default)
        self.modified = True
        return value

    def popitem(self):
        """Pop the last item from the dictionary and mark as modified."""
        item = super().popitem()
        self.modified = True
        return item

    def update(self, *args, **kwargs):
        """Update the dictionary and mark as modified."""
        super().update(*args, **kwargs)
        self.modified = True

    def setdefault(self, key: Any, default: Any = None):
        """Set a default value if the key is not in the dictionary and mark as modified."""
        result = super().setdefault(key, default)
        if result is default:
            self.modified = True
        return result


@contextmanager
def hf_open(
    repo_id: str,
    name: Optional[str] = None,
    save: bool = True,
    **kwargs,
):
    """Context manager for opening a Hugging Face dataset in dict-like format."""
    try:
        # Load the dataset and keep it in memory
        dataset = cast(
            DatasetDict, load_dataset(repo_id, name=name, keep_in_memory=True, **kwargs)
        )
    except (EmptyDatasetError, DataFilesNotFoundError):
        dataset = DatasetDict()

    db = HFOpenDictLike(dataset, repo_id)

    try:
        yield db
    finally:
        if save:
            db.push_to_hub()


@contextmanager
def hf_open_staged(repo_id, use_all_data: bool = True, save: bool = True, **kwargs):
    """Context manager for opening a Hugging Face dataset in dict-like format."""
    if use_all_data:

        @contextmanager
        def inner(name: str):
            with hf_open(repo_id, name=name, save=save, **kwargs) as db:
                yield db.setdefault("all", {})

        yield inner
    else:
        with hf_open(repo_id, save=save, **kwargs) as db:

            @contextmanager
            def inner(name: str):
                try:
                    yield db.setdefault(name, {})
                finally:
                    if save and db.modified and should_push(repo_id):
                        db.push_to_hub()

            yield inner


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
        with hf_open_staged(
            repo_id, use_all_data=USE_ALL_DATA, save=save, **kwargs
        ) as open_db:
            with open_db(dataset_key) as db:

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
                                    print(f"Cache miss (huggingface): {key}")
                            elif isinstance(e, (KeyboardInterrupt, SystemExit)):
                                raise e
                            else:
                                print(
                                    f"Error {e} in {dataset_key} in {repo_id} with key {key}"
                                )
                            if not isinstance(e, (KeyError, AttributeError)):
                                raise e
                            mem_db[mkey] = db[key] = value(*args, **kwargs)
                        return mem_db[mkey]

                yield delegate

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

    with hf_open_staged(
        repo_id, use_all_data=USE_ALL_DATA, save=save, **load_dataset_kwargs
    ) as open_db:
        with open_db(dataset_key) as db:
            if key in db:
                del db[key]
