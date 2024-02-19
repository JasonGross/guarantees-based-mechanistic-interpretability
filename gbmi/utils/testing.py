import sys
import expecttest
from unittest.util import safe_repr
import torch
from prettyprinter import install_extras, pformat


class AlwaysTrue:
    def __contains__(self, item):
        return True


install_extras(include=["dataclasses", "attrs"])
sys.modules["prettyprinter.prettyprinter"].IMPLICIT_MODULES = AlwaysTrue()  # type: ignore


class TestCase(expecttest.TestCase):
    def assertExpectedIgnoreWhitespace(self, actual, expected):
        return self.assertExpectedInline(
            "".join(actual.split()), "".join(expected.split()), skip=1
        )

    def assertExpectedPretty(
        self, actual, expected, width=120, postprocess=None, **kwargs
    ):
        actual = pformat(actual, width=width, **kwargs)
        if postprocess:
            actual = postprocess(actual)
        return self.assertExpectedInline(actual, expected, skip=1)

    def assertAllClose(self, first, second, msg=None, equal_nan=True, **kwargs):
        """
        Fail if the two objects are unequal as determined by torch.allclose
        """
        kwargs = dict(equal_nan=equal_nan, **kwargs)
        if not isinstance(first, torch.Tensor):
            first = torch.tensor(first)
        if not isinstance(second, torch.Tensor):
            second = torch.tensor(second)
        diff = (first - second).abs()
        if msg is None:
            msg = f"{safe_repr(first)} != {safe_repr(second)} within {safe_repr(kwargs)} ({safe_repr(diff)} difference)"
        return self.assertTrue(torch.allclose(first, second, **kwargs), msg=msg)
