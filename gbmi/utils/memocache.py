import inspect
import pickle
from typing import IO, Any, Callable, Optional, Tuple, TypeVar, Union
from contextlib import (
    AbstractContextManager,
    contextmanager,
    asynccontextmanager,
)
import concurrent.futures
from frozendict import frozendict
import threading
from filelock import FileLock
from pathlib import Path
import base64
import shelve
import asyncio
import tempfile, os

pd = None
try:
    if not os.getenv("MEMOCACHE_NO_PANDAS"):
        import pandas as pd
except ImportError:
    pass
torch = None
np = None
try:
    import torch
except ImportError:
    pass
try:
    import numpy as np
except ImportError:
    pass

__all__ = ["Memoize", "USE_PANDAS"]

USE_PANDAS = pd is not None

T = TypeVar("T", bound=AbstractContextManager)
KEY = Tuple[tuple, frozendict]


def to_immutable(arg: Any) -> Any:
    """Converts a list or dict to an immutable version of itself."""
    if isinstance(arg, list) or isinstance(arg, tuple):
        return tuple(to_immutable(e) for e in arg)
    elif isinstance(arg, dict):
        return frozendict({k: to_immutable(v) for k, v in arg.items()})
    elif np is not None and isinstance(arg, np.ndarray):
        return to_immutable(arg.tolist())
    elif torch is not None and isinstance(arg, torch.Tensor):
        return to_immutable(arg.tolist())
    else:
        return arg


class DummyContextWrapper(AbstractContextManager):
    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass


def wrap_context(ctx: T, skip: bool = False) -> Union[T, DummyContextWrapper]:
    """If skip is True, returns a dummy context manager that does nothing."""
    return ctx if not skip else DummyContextWrapper()


def write_via_temp(file_path: Union[str, Path], do_write: Callable[[IO[bytes]], Any]):
    """Writes to a file by writing to a temporary file and then renaming it.
    This ensures that the file is never in an inconsistent state."""
    temp_dir = Path(file_path).parent
    with tempfile.NamedTemporaryFile(
        dir=temp_dir, delete=False, mode="wb"
    ) as temp_file:
        temp_file_path = temp_file.name
        try:
            do_write(temp_file)
        except Exception:
            # If there's an error, clean up the temporary file and re-raise the exception
            temp_file.close()
            os.remove(temp_file_path)
            raise
    # Rename the existing cache file to a backup file
    backup_file_path = f"{file_path}.bak"
    try:
        if os.path.exists(file_path):
            os.rename(file_path, backup_file_path)

        # Rename the temporary file to the cache file
        os.rename(temp_file_path, file_path)
    finally:
        # Delete the backup file
        if os.path.exists(backup_file_path):
            if not os.path.exists(file_path):
                os.rename(backup_file_path, file_path)
            else:
                os.remove(backup_file_path)


# https://stackoverflow.com/a/63425191/377022
_pool = concurrent.futures.ThreadPoolExecutor()


@asynccontextmanager
async def async_lock(lock):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(_pool, lock.acquire)
    try:
        yield  # the lock is held
    finally:
        lock.release()


class Memoize:
    """A memoization decorator that caches the results of a function call.
    The cache is stored in a file, so it persists between runs.
    The cache is also thread-safe, so it can be used in a multithreaded environment.
    The cache is also exception-safe, so it won't be corrupted if there's an error.
    If use_shelf is true, Python's shelve is used to store the cache, avoiding in-memory overhead.
    """

    instances = {}

    cache_base_dir = "cache"

    def __init__(
        self,
        func: Callable,
        name: Optional[str] = None,
        cache_file: Optional[Union[str, Path]] = None,
        disk_write_only: bool = False,
        use_pandas: Optional[bool] = None,
        force_async: bool = False,
        use_shelf: Optional[bool] = None,
    ):
        if use_pandas is None:
            use_pandas = USE_PANDAS
        if isinstance(func, Memoize):
            self.func = func.func
            self.name: str = name or func.name
            self.cache_file = Path(cache_file or func.cache_file).absolute()
            self.cache: Union[dict, shelve.Shelf[Any]] = func.cache
            self.df_cache: set = func.df_cache
            self.df = func.df
            self.df_thread_lock: threading.Lock = func.df_thread_lock
            self.thread_lock: threading.Lock = func.thread_lock
            self.file_lock = func.file_lock
            self.force_async = func.force_async
            self.use_shelf = func.use_shelf
            assert (
                use_shelf is None or use_shelf == func.use_shelf
            ), f"Not equal! func.use_shelf={func.use_shelf}, use_shelf={use_shelf}"
            if name is not None:
                Memoize.instances[name] = self
        else:
            if use_shelf is None:
                use_shelf = False
            self.func = func
            self.name = name or func.__name__
            self.cache_file = Path(
                cache_file or (Path(Memoize.cache_base_dir) / f"{self.name}_cache.pkl")
            ).absolute()
            self.use_shelf = use_shelf
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            self.cache: Union[dict, shelve.Shelf[Any]] = (
                shelve.open(str(self.cache_file.absolute())) if use_shelf else {}
            )
            self.df_cache: set = set()
            self.df = (
                pd.DataFrame(columns=["input", "output"])
                if use_pandas and pd is not None
                else None
            )
            self.df_thread_lock: threading.Lock = threading.Lock()
            self.thread_lock: threading.Lock = threading.Lock()
            self.file_lock: FileLock = FileLock(f"{self.cache_file}.lock")
            self.force_async = force_async
            Memoize.instances[self.name] = self
            self._load_cache_from_disk()
        self.disk_write_only: int = disk_write_only
        self.disk_write_only_lock: threading.Lock = threading.Lock()
        for attr in ("__doc__", "__name__", "__module__"):
            if hasattr(func, attr):
                setattr(self, attr, getattr(func, attr))

    def _use_shelf_and_sync(self, use_lock: bool = True):
        if not self.use_shelf:
            return False
        assert isinstance(self.cache, shelve.Shelf)
        with wrap_context(self.thread_lock, skip=not use_lock):
            with wrap_context(self.file_lock, skip=not use_lock):
                self.cache.sync()

    def _load_cache_from_disk(self, use_lock: bool = True):
        """Loads the cache from disk.  If use_lock is True, then the cache is locked while it's being loaded."""
        if self._use_shelf_and_sync(use_lock=use_lock):
            return
        with wrap_context(self.file_lock, skip=not use_lock):
            try:
                with open(self.cache_file, "rb") as f:
                    disk_cache = pickle.load(f)
            except FileNotFoundError:
                return
        with wrap_context(self.thread_lock, skip=not use_lock):
            self.cache.update(disk_cache)

    def _write_cache_to_disk(self, skip_load: bool = False, use_lock: bool = True):
        """Writes the cache to disk.  The cache is locked while it's being written."""
        if self._use_shelf_and_sync(use_lock=use_lock):
            return
        with wrap_context(self.thread_lock, skip=not use_lock):
            with wrap_context(self.file_lock, skip=not use_lock):
                if not skip_load:
                    self._load_cache_from_disk(use_lock=False)
                # use a tempfile so that we don't corrupt the cache if there's an error
                write_via_temp(self.cache_file, (lambda f: pickle.dump(self.cache, f)))

    def kwargs_of_key(self, key: KEY) -> frozendict:
        """Returns the kwargs of a key."""
        return key[1]

    def args_of_key(self, key: KEY) -> tuple:
        """Returns the args of a key."""
        return key[0]

    def _str_key(self, key: KEY) -> str:
        """Returns a string representation of a key."""
        return base64.b64encode(pickle.dumps(key)).decode("ascii")

    def _str_key_fast(self, key: KEY) -> str:
        """Returns a string representation of a key only if self.cache is a Shelf."""
        return self._str_key(key) if isinstance(self.cache, shelve.Shelf) else ""

    def _uncache(self, key: KEY):
        """Removes a key from the cache."""
        str_key = self._str_key_fast(key)
        with self.thread_lock:
            with self.file_lock:
                self._load_cache_from_disk(use_lock=False)
                if isinstance(self.cache, shelve.Shelf):
                    del self.cache[str_key]
                else:
                    del self.cache[key]
                self._write_cache_to_disk(skip_load=True, use_lock=False)

    def uncache(self, *args, **kwargs):
        return self._uncache((to_immutable(args), to_immutable(kwargs)))

    @classmethod
    def sync_all(cls):
        """Writes all caches to disk."""
        for instance in cls.instances.values():
            instance._write_cache_to_disk()

    def key_of_args(self, *args, **kwargs) -> KEY:
        return (to_immutable(args), to_immutable(kwargs))

    def _update_df(self, key: KEY, val: Any):
        if self.df is not None and pd is not None:
            with self.df_thread_lock:
                if key not in self.df_cache:
                    self.df_cache.add(key)
                    new_row = pd.DataFrame({"input": [key], "output": [val]})
                    self.df = pd.concat([self.df, new_row], ignore_index=True)

    def _sync_call(self, *args, **kwargs):
        """Calls the function, caching the result if it hasn't been called with the same arguments before."""
        key = self.key_of_args(*args, **kwargs)

        if not self.disk_write_only:
            self._load_cache_from_disk()

        str_key = self._str_key_fast(key)
        try:
            if isinstance(self.cache, shelve.Shelf):
                with self.thread_lock:
                    val = self.cache[str_key]
            else:
                val = self.cache[key]
        except KeyError:
            val = self.func(*args, **kwargs)
            if isinstance(self.cache, shelve.Shelf):
                with self.thread_lock:
                    self.cache[str_key] = val
            else:
                with self.thread_lock:
                    self.cache[key] = val

            self._write_cache_to_disk()

        self._update_df(key, val)
        return val

    async def _async_call(self, *args, **kwargs):
        """Calls the function, caching the result if it hasn't been called with the same arguments before."""
        key = self.key_of_args(*args, **kwargs)

        if not self.disk_write_only:
            await asyncio.to_thread(self._load_cache_from_disk)

        str_key = self._str_key_fast(key)
        try:
            if isinstance(self.cache, shelve.Shelf):
                async with async_lock(self.thread_lock):
                    val = self.cache[str_key]
            else:
                val = self.cache[key]
        except KeyError:
            val = await self.func(*args, **kwargs)
            if isinstance(self.cache, shelve.Shelf):
                async with async_lock(self.thread_lock):
                    self.cache[str_key] = val
            else:
                async with async_lock(self.thread_lock):
                    self.cache[key] = val
            await asyncio.to_thread(self._write_cache_to_disk)

        self._update_df(key, val)
        return val

    def __call__(self, *args, **kwargs):
        """Calls the function, caching the result if it hasn't been called with the same arguments before."""
        if inspect.iscoroutinefunction(self.func) or self.force_async:
            return self._async_call(*args, **kwargs)
        else:
            return self._sync_call(*args, **kwargs)

    @contextmanager
    def sync_cache(self, inplace: bool = False):
        """Syncs the cache to disk on entering the context.

        If inplace is False, returns a copy of the function that can be called without incurring the overhead of reading from disk.
        If inplace is True, mutates the current function to avoid reading from disk.
        """
        self._load_cache_from_disk()
        if inplace:
            with self.disk_write_only_lock:
                self.disk_write_only += 1
            try:
                yield self
            finally:
                with self.disk_write_only_lock:
                    self.disk_write_only -= 1
        else:
            yield Memoize(self, disk_write_only=True)

    def __repr__(self):
        return f"Memoize(func={self.func!r}, name={self.name!r}, cache_file={self.cache_file!r}, use_shelf={self.use_shelf!r})"

    def __str__(self):
        return f"Memoize(func={self.func}, name={self.name}, cache_file={self.cache_file}, use_shelf={self.use_shelf!r})"
