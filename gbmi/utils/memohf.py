import logging
import pickle
import time
from contextlib import contextmanager
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Literal,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
)

from datasets import Dataset, DatasetDict, load_dataset
from datasets.data_files import EmptyDatasetError
from datasets.exceptions import DataFilesNotFoundError

from gbmi.utils.contextlib_extra import chain_contextmanagers_data
from gbmi.utils.hashing import get_hash_ascii

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

PUSH_INTERVAL: float = (
    10  # Push only if more than 10 seconds have passed since the last push
)

last_push_time: Dict[str, float] = {}
memohf_cache: Dict[str, Dict[str, Any]] = {}

T = TypeVar("T")
K1 = TypeVar("K1")
K2 = TypeVar("K2")
V = TypeVar("V")
K = TypeVar("K")


def should_push(repo_id: str) -> bool:
    """Determines if we should push based on the last push time."""
    last_time = last_push_time.get(repo_id, 0)
    return time.time() - last_time > PUSH_INTERVAL


def update_last_push_time(repo_id: str):
    """Updates the last push time for the given repo."""
    last_push_time[repo_id] = time.time()


class HFOpenDictLike(dict):
    """A dict-like object that supports push_to_hub."""

    def __init__(
        self,
        dataset: DatasetDict,
        repo_id: str,
        config_name: str = "default",
        hash_function: Callable = hash,
    ):
        super().__init__()
        self.repo_id = repo_id
        self.dataset = dataset
        self.config_name = config_name
        self.update(self._load_db())  # Load the data and update the internal dict
        self.hash_function = hash_function
        self._reset_hash()

    def _gethash(self):
        return self.hash_function(tuple(self.items()))

    def _reset_hash(self):
        """Resets the hash to the current state of the database."""
        self.init_hash = self._gethash()

    def _load_db(self) -> Dict[str, Any]:
        """Loads the dataset data from the Hugging Face hub based on the mode."""
        return {
            key: pickle.loads(self.dataset[key]["data"][0])
            for key in self.dataset.keys()
        }

    def push_to_hub(self):
        """Pushes the current state of the database back to the Hugging Face hub."""
        if len(self.items()) == 0:
            return
        for key, data in self.items():
            serialized_data = pickle.dumps(data)
            self.dataset[key] = Dataset.from_dict({"data": [serialized_data]})
        self.dataset.push_to_hub(self.repo_id, config_name=self.config_name)
        update_last_push_time(self.repo_id)
        self._reset_hash()

    @property
    def modified(self) -> bool:
        """Determines if the database has been modified."""
        return self.init_hash != self._gethash()


@contextmanager
def hf_open(
    repo_id: str,
    name: Optional[str] = None,
    save: bool = True,
    hash_function: Callable = hash,
    **kwargs,
):
    """Context manager for opening a Hugging Face dataset in dict-like format."""
    try:
        # Load the dataset and keep it in memory
        logger.debug(
            f"load_dataset({repo_id!r}, name={name!r}, keep_in_memory=True, **{kwargs!r})"
        )
        dataset = cast(
            DatasetDict, load_dataset(repo_id, name=name, keep_in_memory=True, **kwargs)
        )
        logger.debug("Dataset loaded")
    except (EmptyDatasetError, DataFilesNotFoundError) as e:
        logger.debug(e)
        dataset = DatasetDict()
    except ValueError as e:
        logger.warning(e)
        dataset = DatasetDict()

    db = HFOpenDictLike(
        dataset, repo_id, hash_function=hash_function, config_name=name or "default"
    )

    try:
        yield db
    finally:
        if save:
            db.push_to_hub()


def merge_subdicts(
    *dicts_keys: Tuple[dict[K1, dict[K2, V]], K1],
    default_factory: Callable[[], dict[K2, V]] = dict,
) -> dict[K2, V]:
    """Merges multiple sub dictionaries into a single dictionary."""
    (dict0, k0), *rest_dicts_keys = dicts_keys
    merged = dict0.setdefault(k0, default_factory())
    for d, k in rest_dicts_keys:
        old = d.setdefault(k, merged)
        if old is not merged:
            d[k] = merged
            merged.update(old)

    return merged


StorageMethod = Union[
    Literal["single_data_file", "named_data_files", "data_splits"],
    Tuple[Literal["single_named_data_file", "single_split_data_file"], str],
]

# @contextmanager
# def open_named_hf(
#     repo_id: str,
#     *names: str,
#     save: bool = True,
#     hash_function: Callable = hash,
#     **kwargs,
# ):
#     """Context manager for opening a Hugging Face dataset in dict-like format."""
#     name, *rest_names = names
#     with hf_open(repo_id, name=name, save=save, hash_function=hash_function, **kwargs) as db:

#         if rest_names:


@contextmanager
def hf_open_staged(
    repo_id,
    storage_methods: Union[StorageMethod, Iterable[StorageMethod]] = "single_data_file",
    save: bool = True,
    hash_function: Callable = hash,
    **kwargs,
):
    """Context manager for opening a Hugging Face dataset in dict-like format."""
    if isinstance(storage_methods, str) or (
        isinstance(storage_methods, tuple)
        and len(storage_methods) == 2
        and storage_methods[0] in ("single_named_data_file", "single_split_data_file")
        and isinstance(storage_methods[1], str)
    ):
        storage_methods = [storage_methods]
    else:
        storage_methods = list(storage_methods)

    if (
        "single_data_file" in storage_methods
        or "data_splits" in storage_methods
        or any(
            isinstance(x, tuple)
            and x[0] in ("single_named_data_file", "single_split_data_file")
            for x in storage_methods
        )
    ):
        extra_db_opens = []
        db_keys = []
        db = None
        if "data_splits" in storage_methods or "single_data_file" in storage_methods:
            extra_db_opens.append(
                (
                    hf_open,
                    (repo_id,),
                    dict(save=save, hash_function=hash_function, **kwargs),
                    ((lambda db: None, (), {}),),
                )
            )
        for storage_method in storage_methods:
            if isinstance(storage_method, tuple):
                match storage_method[0]:
                    case "single_named_data_file":
                        extra_db_opens.append(
                            (
                                hf_open,
                                (repo_id,),
                                dict(
                                    name=storage_method[1],
                                    save=save,
                                    hash_function=hash_function,
                                    **kwargs,
                                ),
                                (
                                    (
                                        lambda db, name: db_keys.append(
                                            (db.setdefault("alldata", {}), name)
                                        )
                                    ),
                                    (),
                                    {},
                                ),
                            )
                        )
                    case "single_split_data_file":
                        extra_db_opens.append(
                            (
                                hf_open,
                                (repo_id,),
                                dict(
                                    save=save,
                                    hash_function=hash_function,
                                    **kwargs,
                                ),
                                (
                                    (
                                        lambda db, name, storage_name: db_keys.append(
                                            (
                                                db.setdefault(storage_name, {}),
                                                name,
                                            )
                                        )
                                    ),
                                    (storage_method[1],),
                                    {},
                                ),
                            )
                        )
        with chain_contextmanagers_data(*extra_db_opens) as extra_dbs:
            if (
                "data_splits" in storage_methods
                or "single_data_file" in storage_methods
            ):
                (db, (_func, _args, _kwargs)), *extra_dbs = extra_dbs

            @contextmanager
            def inner(name: str):
                if "data_splits" in storage_methods:
                    db_keys.append((db, name))
                if "single_data_file" in storage_methods:
                    db_keys.append((db.setdefault("alldata", {}), name))
                extra_specific_db_opens = []
                if "named_data_files" in storage_methods:
                    extra_specific_db_opens.append(
                        (
                            hf_open,
                            (repo_id,),
                            dict(
                                name=name,
                                save=save,
                                hash_function=hash_function,
                                **kwargs,
                            ),
                            ((lambda db: db_keys.append((db, "alldata")), (), {})),
                        )
                    )

                if extra_dbs:
                    for extra_db, (
                        append_db,
                        append_db_args,
                        append_db_kwargs,
                    ) in extra_dbs:
                        append_db(extra_db, name, *append_db_args, **append_db_kwargs)

                try:
                    if extra_specific_db_opens:
                        with chain_contextmanagers_data(
                            *extra_specific_db_opens
                        ) as extra_specific_dbs:
                            for extra_db, (
                                append_db,
                                append_db_args,
                                append_db_kwargs,
                            ) in extra_specific_dbs:
                                append_db(extra_db, *append_db_args, **append_db_kwargs)
                            try:
                                yield merge_subdicts(*db_keys)
                            finally:
                                if save and should_push(repo_id):
                                    for extra_db, _ in extra_specific_dbs:
                                        if extra_db.modified:
                                            extra_db.push_to_hub()
                    else:
                        yield merge_subdicts(*db_keys)
                finally:
                    if save and should_push(repo_id):
                        if db and db.modified:
                            db.push_to_hub()
                        if extra_dbs:
                            for extra_db, _ in extra_dbs:
                                if extra_db.modified:
                                    extra_db.push_to_hub()

            yield inner
    elif storage_methods:
        assert "named_data_files" in storage_methods

        @contextmanager
        def inner(name: str):
            with hf_open(
                repo_id, name=name, save=save, hash_function=hash_function, **kwargs
            ) as db:
                yield db.setdefault("alldata", {})

        yield inner
    else:
        logger.warning("No storage methods provided for %s", repo_id)

        @contextmanager
        def inner(name: str):
            logger.warning("No storage methods provided for %s, %s", repo_id, name)
            yield {}

        yield inner


@contextmanager
def memohf_staged(
    repo_id: str,
    *,
    save: bool = True,
    storage_methods: Union[StorageMethod, Iterable[StorageMethod]] = "single_data_file",
    hash_function: Callable = pickle.dumps,
    **kwargs,
):
    with hf_open_staged(
        repo_id,
        storage_methods=storage_methods,
        save=save,
        hash_function=hash_function,
        **kwargs,
    ) as open_db:

        @contextmanager
        def inner(
            value: Callable,
            dataset_key: str,
            cache: Dict[str, Dict[str, Any]] = memohf_cache,
            get_hash: Callable = get_hash_ascii,
            get_hash_mem: Optional[Callable] = None,
            print_cache_miss: bool = False,
        ):
            mem_db = cache.setdefault(repo_id, {}).setdefault(dataset_key, {})
            if get_hash_mem is None:
                get_hash_mem = get_hash

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

        yield inner


def memohf(
    value: Callable,
    repo_id: str,
    dataset_key: str,
    *,
    cache: Dict[str, Dict[str, Any]] = memohf_cache,
    get_hash: Callable = get_hash_ascii,
    get_hash_mem: Optional[Callable] = None,
    print_cache_miss: bool = False,
    save: bool = True,
    storage_methods: Union[StorageMethod, Iterable[StorageMethod]] = "single_data_file",
    **kwargs,
):
    """Memoziation using huggingface + in-memory cache"""

    @contextmanager
    def open_db():
        with memohf_staged(
            repo_id, save=save, storage_methods=storage_methods, **kwargs
        ) as staged_db:
            with staged_db(
                value,
                dataset_key,
                cache=cache,
                get_hash=get_hash,
                get_hash_mem=get_hash_mem,
                print_cache_miss=print_cache_miss,
            ) as func:
                yield func

    return open_db


def uncache(
    *args,
    repo_id: str,
    dataset_key: str,
    storage_methods: Union[StorageMethod, Iterable[StorageMethod]] = "single_data_file",
    cache: Dict[str, Dict[str, Any]] = memohf_cache,
    get_hash: Callable = get_hash_ascii,
    get_hash_mem: Optional[Callable] = None,
    hash_function: Callable = pickle.dumps,
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
        repo_id,
        storage_methods=storage_methods,
        save=save,
        hash_function=hash_function,
        **load_dataset_kwargs,
    ) as open_db:
        with open_db(dataset_key) as db:
            if key in db:
                del db[key]
