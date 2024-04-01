__all__ = ["lazy"]


from typing import Callable, Generic, Optional, TypeVar

T = TypeVar("T")


class lazy(Generic[T]):
    def __init__(
        self,
        generate: Callable[[], T] = (lambda: None),
        always_regenerate: bool = False,
        *args,
        **kwargs,
    ):
        self._generate: Callable[[], T] = generate
        self._always_regenerate: bool = always_regenerate
        self._args = args
        self._kwargs = kwargs

    def force(self, regenerate: bool = False) -> T:
        if regenerate or self._always_regenerate or not hasattr(self, "_value"):
            self._value: T = self._generate(*self._args, **self._kwargs)
        return self._value

    def __str__(self):
        return str(self.force())

    def __repr__(self):
        return repr(self.force())
