import re
import sys
from itertools import chain
from prettyprinter.utils import take
from prettyprinter.doc import concat
from prettyprinter.prettyprinter import (
    _is_cnamedtuple,
    pretty_cnamedtuple,
    _is_namedtuple,
    pretty_namedtuple,
    LBRACKET,
    LPAREN,
    RBRACKET,
    RPAREN,
    RBRACE,
    LBRACE,
    pretty_call_alt,
    ELLIPSIS,
    build_fncall,
    general_identifier,
    pretty_python_value,
    MULTILINE_STRATEGY_PLAIN,
    commentdoc,
    sequence_of_docs,
    MULTILINE_STRATEGY_HANG,
)
from prettyprinter import register_pretty, install_extras
from typing import List, TypeVar, Tuple
from prettyprinter.doctypes import Annotated

from gbmi.utils.testing import AlwaysTrue

T = TypeVar("T")


def split_seq(seq: List[T], cmp=lambda x, y: x == y) -> List[Tuple[T, int]]:
    results = []
    prev_elt = None
    cnt = 0
    for elt in seq:
        if prev_elt is None:
            prev_elt = elt
            cnt = 1
        elif cmp(prev_elt, elt):
            cnt += 1
        else:
            results.append((prev_elt, cnt))
            prev_elt = elt
            cnt = 1
    if prev_elt is not None:
        results.append((prev_elt, cnt))
    return results


# Code lifted from https://github.com/tommikaikkonen/prettyprinter/blob/master/prettyprinter/__init__.py
# Hackish / for experimentation only
def register_deduplicated_lists(n_to_dedup=3):
    @register_pretty(tuple)
    @register_pretty(list)
    @register_pretty(set)
    def pretty_bracketable_iterable(value, ctx, trailing_comment=None):
        constructor = type(value)

        if isinstance(value, tuple):
            if _is_cnamedtuple(value):
                try:
                    return pretty_cnamedtuple(
                        value, ctx, trailing_comment=trailing_comment
                    )
                except Exception:
                    pass  # render as a normal tuple
            elif _is_namedtuple(value):
                return pretty_namedtuple(value, ctx, trailing_comment=trailing_comment)

        is_native_type = constructor in (tuple, list, set)
        if len(value) > ctx.max_seq_len:
            truncation_comment = "...and {} more elements".format(
                len(value) - ctx.max_seq_len
            )

            trailing_comment = (
                truncation_comment + ". " + trailing_comment
                if trailing_comment
                else truncation_comment
            )

        dangle = False

        if isinstance(value, list):
            left, right = LBRACKET, RBRACKET
        elif isinstance(value, tuple):
            left, right = LPAREN, RPAREN
            if len(value) == 1:
                dangle = True
        elif isinstance(value, set):
            left, right = LBRACE, RBRACE

        if not value:
            if isinstance(value, (list, tuple)):
                if is_native_type:
                    return concat([left, right])
                return pretty_call_alt(ctx, constructor)
            else:
                # E.g. set() or SubclassOfSet()
                return pretty_call_alt(ctx, constructor)

        if ctx.depth_left == 0:
            if isinstance(value, (list, tuple)):
                literal = concat([left, ELLIPSIS, right])
                if is_native_type:
                    return literal
                return build_fncall(
                    ctx,
                    general_identifier(constructor),
                    argdocs=(literal,),
                    hug_sole_arg=True,
                )
            else:
                return pretty_call_alt(ctx, constructor, args=(...,))

        if len(value) == 1:
            sole_value = list(value)[0]
            els = [
                pretty_python_value(
                    sole_value,
                    ctx=(
                        ctx.nested_call().use_multiline_strategy(
                            MULTILINE_STRATEGY_PLAIN
                        )
                    ),
                )
            ]
        else:
            els = [
                pretty_python_value(
                    el,
                    ctx=(
                        ctx.nested_call().use_multiline_strategy(
                            MULTILINE_STRATEGY_HANG
                        )
                    ),
                )
                for el in take(ctx.max_seq_len, value)
            ]

            def cmp(x, y):
                # warning: very hacky.
                x = re.sub("at 0x[0-9a-f]+", "", str(x))
                y = re.sub("at 0x[0-9a-f]+", "", str(y))
                return x == y

            group_els = split_seq(els, cmp)
            seq = []
            for el, cnt in group_els:
                if cnt >= n_to_dedup:
                    seq.append(el)
                    seq.append(commentdoc(f"... {cnt - 1} more"))
                else:
                    seq += [el for _ in range(cnt)]
            els = seq

        if trailing_comment:
            els = chain(els, [commentdoc(trailing_comment)])
            dangle = False

        literal = sequence_of_docs(
            ctx, left, els, right, dangle=dangle, force_break=bool(trailing_comment)
        )

        if is_native_type:
            return literal

        return build_fncall(
            ctx, general_identifier(constructor), argdocs=(literal,), hug_sole_arg=True
        )

    install_extras(include=["dataclasses", "attrs"])
    sys.modules["prettyprinter.prettyprinter"].IMPLICIT_MODULES = AlwaysTrue()
