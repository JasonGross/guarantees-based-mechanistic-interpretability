from contextlib import contextmanager
from typing import (
    Any,
    Callable,
    ContextManager,
    Generator,
    Tuple,
    TypeVar,
    TypeVarTuple,
)

T = TypeVar("T")
V = TypeVar("V")
As = TypeVarTuple("As")


@contextmanager
def chain_contextmanagers_data(
    *funcs_data: tuple[
        # flake8 doesn't yet support TypeVarTuple in type hints apparently
        Callable[[*As], ContextManager[V]],  # noqs: E999
        Tuple[*As],  # noqs: E999
        dict[str, Any],
        T,
    ]
) -> Generator[tuple[tuple[V, T], ...], Any, None]:
    """
    Chains multiple context managers and yields a tuple of results from each one.

    This function allows you to pass multiple context manager functions, and it will
    execute them sequentially, nesting the `with` blocks. It yields a tuple of the
    results from all the context managers, maintaining the order in which they were passed.

    Parameters:
    *funcs: Variadic argument representing context manager functions.
            Each function should be callable without arguments and return a context manager.

    Yields:
    tuple: A tuple containing the results of all the context managers executed in sequence.

    Example:
        >>> @contextmanager
        ... def context_a():
        ...     yield "A"
        ...
        >>> @contextmanager
        ... def context_b():
        ...     yield "B"
        ...
        >>> with chain_contextmanagers_data((context_a, (), {}, "a"), (context_b, (), {}, "b")) as results:
        ...     print(results)
        ...
        (('A', 'a'), ('B', 'b'))
    """
    if funcs_data:
        (func, args, kwargs, data), *rest_funcs_data = funcs_data
        with func(*args, **kwargs) as res:
            if rest_funcs_data:
                with chain_contextmanagers_data(*rest_funcs_data) as rest_res:
                    yield ((res, data),) + rest_res
            else:
                yield ((res, data),)
    else:
        yield None


@contextmanager
def chain_contextmanagers(*funcs):
    """
    Chains multiple context managers and yields a tuple of results from each one.

    This function allows you to pass multiple context manager functions, and it will
    execute them sequentially, nesting the `with` blocks. It yields a tuple of the
    results from all the context managers, maintaining the order in which they were passed.

    Parameters:
    *funcs: Variadic argument representing context manager functions.
            Each function should be callable without arguments and return a context manager.

    Yields:
    tuple: A tuple containing the results of all the context managers executed in sequence.

    Example:
        >>> @contextmanager
        ... def context_a():
        ...     yield "A"
        ...
        >>> @contextmanager
        ... def context_b():
        ...     yield "B"
        ...
        >>> with chain_contextmanagers(context_a, context_b) as results:
        ...     print(results)
        ...
        ('A', 'B')
    """
    with chain_contextmanagers_data(
        *[(func, (), {}, None) for func in funcs]
    ) as results:
        yield tuple(res for res, _ in results)
