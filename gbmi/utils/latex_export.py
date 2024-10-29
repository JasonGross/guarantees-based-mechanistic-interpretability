# %%
import math
import re
from typing import Tuple, TypeVar, Union

import numpy as np

from gbmi.utils.instructions import InstructionCount, PerfCounter

T = TypeVar("T")


def get_float_type(v: Union[float, np.floating]) -> type[np.float32] | type[np.float64]:
    for ty in (np.float32,):
        if np.isfinite(v) and (v > np.finfo(ty).max or v < np.finfo(ty).min):
            continue
        if np.array(v, dtype=ty).item() == v:
            return ty
    return np.float64


def get_mantissa_bits(f: float, explicit_only: bool = False) -> int:
    return {np.float32: 23, np.float64: 52}[get_float_type(f)] + (
        1 if not explicit_only else 0
    )


def get_precision(f: float, base: int = 10) -> int:
    float_type = get_float_type(f)
    mantissa_bits = get_mantissa_bits(f, explicit_only=False)

    # Calculate the required precision in decimal digits
    precision_bits = math.ceil(mantissa_bits * math.log(base, 2))

    # Adjust precision based on the logarithm of the number
    if f != 0:
        exponent_adjustment = max(0, -math.floor(math.log(abs(f), base)))
    else:
        exponent_adjustment = 0

    return precision_bits + exponent_adjustment


def format_float_full_precision(f: float) -> str:
    if np.isnan(f):
        return "nan"
    if np.isinf(f):
        return "inf" if f > 0 else "-inf"
    precision = get_precision(f)
    return f"{f:.{precision}f}".rstrip("0").rstrip(".")


def format_float_full_precision_if_float(f: Union[T, float]) -> Union[str, T]:
    if isinstance(f, (float, np.floating)):
        return format_float_full_precision(f)
    return f


# # Test function to ensure enough precision
# def test_enough_precision(f: float):
#     formatted = format_float_full_precision(f)
#     parsed_back = float(formatted)
#     return np.isclose(f, parsed_back, atol=0, rtol=1e-15)

# test values to be moved into a test file
# # Example usage and test
# if __name__ == "__main__":
#     f = 1.234567890123456789
#     precision = get_precision(f)
#     formatted = format_float_full_precision(f)
#     print(f"Precision: {precision} decimal places")
#     print(f"Formatted float: {formatted}")

#     # Run the test
#     result = test_enough_precision(f)
#     print(f"Test passed: {result}")

#     # Additional tests
#     test_values = [
#         1.234567890123456789,
#         -1.234567890123456789,
#         0.000000000123456789,
#         -0.000000000123456789,
#         123456789.123456789,
#         -123456789.123456789
#     ]

#     for value in test_values:
#         formatted_value = format_float_full_precision(value)
#         print(f"Original: {value}, Formatted: {formatted_value}")
#         assert test_enough_precision(value), f"Test failed for value: {value}"


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
                lines.append(
                    rf"{expand}\newcommand{key}{{{format_float_full_precision(value)}}}"
                )
            else:
                raise ValueError(f"Unsupported type {type(value)} for {key} ({value})")
    return "\n".join(lines)


def latex_values_of_counter(prefix: str, counters: PerfCounter) -> dict[str, int]:
    return {
        f"{prefix}Perf{latex_attr}": getattr(counters, attr)
        for attr, latex_attr in (
            ("time_enabled_ns", "TimeEnabledNS"),
            ("instruction_count", "InstructionCount"),
            ("branch_misses", "BranchMisses"),
            ("page_faults", "PageFaults"),
        )
        if hasattr(counters, attr)
    }


def latex_values_of_instruction_count(
    prefix: str, count: InstructionCount
) -> dict[str, int]:
    return {
        f"{prefix}InstructionCount": count.flop,
        f"{prefix}InstructionCountInt": count.int_op,
        f"{prefix}InstructionCountBranch": count.branch,
    }


if __name__ == "__main__":
    import doctest

    doctest.testmod()
