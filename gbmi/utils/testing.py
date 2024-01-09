import sys
import expecttest
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
