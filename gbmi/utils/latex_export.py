import re
from typing import Any, Union, Tuple
import numpy as np


def key_to_command(key: str, prefix: str = "", postfix: str = "") -> Tuple[str, bool]:
    r"""
    Transforms a given key into a LaTeX command or environment name,
    and indicates whether it needs expansion.

    Parameters:
    - key: str, the input key to transform.

    Returns:
    - Tuple[str, bool]: A tuple where the first element is the transformed
      LaTeX command or environment name, and the second element is a boolean
      indicating whether the command or environment name needs to be
      expanded using \expandafter.

    The function handles four cases:
    1. If the key consists only of letters or a single non-letter and non-space character,
       it's prefixed with a backslash.
    2. If the key already starts with a backslash followed by letters or a single non-letter character,
       it's returned as is.
    3. If the key does not contain a backslash, it's wrapped within \csname and \endcsname.
    4. If the key contains a backslash or is otherwise not a simple command name,
       it's wrapped within \csname and \endcsname, and \detokenize is applied.

    Examples:
    >>> key_to_command("textbf")
    ('\\textbf', False)

    >>> key_to_command("\\textbf")
    ('\\textbf', False)

    >>> key_to_command("%")
    ('\\%', False)

    >>> key_to_command("123")
    ('\\csname 123\\endcsname', True)

    >>> key_to_command("table\\row")
    ('\\csname \\detokenize{table\\row}\\endcsname', True)
    """
    if re.match(r"^[a-zA-Z]+$|^[^a-zA-Z\s]$", key):
        return rf"\{prefix}{key}", False
    if re.match(r"^\\[a-zA-Z]+$|^\\[^a-zA-Z\s]$", key):
        return f"{key[0]}{prefix}{key[1:]}", False
    if "\\" not in key:
        return rf"\csname {prefix}{key}\endcsname", True
    return rf"\csname \detokenize{{{prefix}{key}}}\endcsname", True


SupportedLaTeXType = Union[int, float, bool, str]


def to_latex_defs(values: dict[str, SupportedLaTeXType], sort: bool = True) -> str:
    lines = []
    items = values.items()
    if sort:
        items = sorted(items)
    for key, value in items:
        if isinstance(value, bool):
            if key[0] == "\\":
                key = key[1:]
            lines.append(rf"\newboolean{{{key}}}")
            lines.append(rf"\setboolean{{{key}}}{{{str(value).lower()}}}")
        else:
            key, expand = key_to_command(key)
            expand = r"\expandafter" if expand else ""
            if isinstance(value, (str, int, np.integer)):
                lines.append(rf"{expand}\newcommand{key}{{{value}}}")
            elif isinstance(value, (float, np.floating)):
                lines.append(rf"{expand}\newcommand{key}{{{value}}}")
            else:
                raise ValueError(f"Unsupported type {type(value)} for {key} ({value})")
    return "\n".join(lines)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
