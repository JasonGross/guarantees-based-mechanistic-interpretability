# %%
from collections import Counter
from functools import cache

import nltk
import nltk.corpus
import numpy as np
from nltk import ngrams
from nltk.corpus.reader.api import CorpusReader

DEFAULT_CORPUS = "webtext"


@cache
def get_corpus(corpus: str = DEFAULT_CORPUS) -> CorpusReader:
    nltk.download(corpus)
    c = getattr(nltk.corpus, corpus)
    c.ensure_loaded()
    return c


@cache
def get_ngrams(
    n: int = 1,
    *,
    corpus: str = DEFAULT_CORPUS,
    lower: bool = True,
    strip_nonalpha: bool = True,
    strip_nonalnum: bool = True,
    strip_space: bool = True,
    strip_nonascii: bool = True,
) -> Counter:
    c = get_corpus(corpus)
    words = c.words()  # type: ignore
    text = " ".join(words)
    text = text.lower() if lower else text
    text = text.replace(" ", "") if strip_space else text
    text = "".join(filter(str.isalpha, text)) if strip_nonalpha else text
    text = "".join(filter(str.isalnum, text)) if strip_nonalnum else text
    text = "".join(filter(str.isascii, text)) if strip_nonascii else text
    return Counter(ngrams(text, n))


@cache
def ngram_count_table(*args, **kwargs) -> np.ndarray:
    ngrams = get_ngrams(*args, **kwargs)
    ndim = max(len(k) for k in ngrams.keys())
    keys = [sorted(set(k[i] for k in ngrams.keys() if len(k) > i)) for i in range(ndim)]
    assert (
        len(set(len(k) for k in keys)) == 1
    ), f"All keys must have the same length, got {keys}"
    table = np.zeros(tuple(len(k) for k in keys))
    for k, v in ngrams.items():
        table[tuple(ki.index(kp) for ki, kp in zip(keys, k))] = v
    return table


@cache
def ngram_table(*args, **kwargs) -> np.ndarray:
    table = ngram_count_table(*args, **kwargs)
    return table / table.sum()


def ngram(key: str, *, corpus: str = DEFAULT_CORPUS) -> float:
    table = ngram_table(n=len(key), corpus=corpus)
    return table[tuple(ord(c.lower()) - ord("a") for c in key)]
