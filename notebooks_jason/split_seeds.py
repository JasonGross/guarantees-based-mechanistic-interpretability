#!/usr/bin/env python
import argparse
from typing import Optional, Sequence


def parse_arguments(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="Process and print seeds in groups.")
    parser.add_argument(
        "--seeds",
        default="50,104,123,519,742,913,1185,1283,1412,1490,1681,1696,1895,1951,2236,2306,2345,2549,2743,2773,3175,3254,3284,4157,4305,4430,4647,4729,4800,4810,5358,5615,5781,5928,6082,6155,6159,6204,6532,6549,6589,6910,7098,7238,7310,7467,7790,7884,8048,8299,8721,8745,8840,8893,9132,9134,9504,9816,10248,11124,11130,11498,11598,11611,12141,12287,12457,12493,12552,12561,13036,13293,13468,13654,13716,14095,14929,15043,15399,15622,15662,16069,16149,16197,16284,17080,17096,17194,17197,18146,18289,18668,19004,19093,19451,19488,19538,19917,20013,20294,20338,20415,20539,20751,20754,20976,21317,21598,22261,22286,22401,22545,23241,23367,23447,23633,23696,24144,24173,24202,24262,24438,24566,25516,26278,26374,26829,26932,27300,27484,27584,27671,27714,28090,28716,28778,29022,29052,29110,29195,29565,29725,29726,30371,30463,30684,30899,31308,32103,32374,32382",
    )
    parser.add_argument(
        "--exclude", nargs="*", type=int, default=[], help="List of seeds to exclude"
    )
    parser.add_argument(
        "--exclude-after",
        type=int,
        help="Exclude seeds greater than or equal to this value",
    )
    parser.add_argument(
        "--exclude-before",
        type=int,
        help="Exclude seeds less than or equal to this value",
    )
    parser.add_argument(
        "--num-groups",
        type=int,
        default=20,
        help="Number of groups to split the seeds into",
    )
    parser.add_argument(
        "--prefix",
        default='          - "',
        help="Prefix to add to each group of seeds",
    )
    parser.add_argument(
        "--suffix",
        default='"',
        help="Suffix to add to each group of seeds",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_arguments(argv)

    # Convert seeds from string to list of integers
    seeds = list(map(int, args.seeds.split(",")))

    # Exclude specific seeds if provided
    if args.exclude:
        seeds = [s for s in seeds if s not in args.exclude]

    # Exclude seeds after a certain value if provided
    if args.exclude_after is not None:
        seeds = [s for s in seeds if s < args.exclude_after]

    # Exclude seeds before a certain value if provided
    if args.exclude_before is not None:
        seeds = [s for s in seeds if s > args.exclude_before]

    # Determine the number of seeds per group

    group_size = len(seeds) // args.num_groups
    remainder = len(seeds) % args.num_groups

    # Create groups
    groups = []
    start_index = 0
    for i in range(args.num_groups):
        end_index = start_index + group_size + (1 if i < remainder else 0)
        groups.append(seeds[start_index:end_index])
        start_index = end_index

    # Print each group with specified prefix
    for group in groups:
        print(f"{args.prefix}{','.join(map(str, group))}{args.suffix}")


if __name__ == "__main__":
    main()
