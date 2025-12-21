#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
read_parquet.py —— 在终端中友好地查看 Parquet 文件
"""

import argparse
import pathlib
import sys

import pandas as pd
import contextlib


def read_parquet(path, columns=None):
    """
    读取 parquet 文件；如指定 columns 则只读取相应列
    """
    return pd.read_parquet(path, columns=columns)


def show_dataframe(df, n=10, full=False):
    """
    在终端打印 DataFrame
    """
    if full:
        ctx = pd.option_context(
            "display.max_columns", None,
            "display.max_colwidth", None,
            "display.width", None
        )
    else:
        # 不做特殊处理
        ctx = contextlib.nullcontext()

    with ctx:
        print(df.head(n))


def main():
    parser = argparse.ArgumentParser(
        description="Read a .parquet file and print its contents."
    )
    parser.add_argument("file", help="Path to parquet file")
    parser.add_argument(
        "-n", "--num", type=int, default=10,
        help="Number of rows to display (default=10)"
    )
    parser.add_argument(
        "--cols", nargs="+",
        help="Only load / display specified columns"
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Print without pandas truncation (show all columns/full width)"
    )
    parser.add_argument(
        "--info", action="store_true",
        help="Print DataFrame.info() before data"
    )

    args = parser.parse_args()

    parquet_path = pathlib.Path(args.file)
    if not parquet_path.exists() or not parquet_path.is_file():
        print(f"Error: '{parquet_path}' does not exist or is not a file.", file=sys.stderr)
        sys.exit(1)

    try:
        df = read_parquet(parquet_path, columns=args.cols)
    except Exception as e:
        print(f"Failed to read parquet: {e}", file=sys.stderr)
        sys.exit(1)

    if args.info:
        print("=== DataFrame.info() ===")
        print(df.info(show_counts=True))
        print()

    print(f"=== First {args.num} rows (DataFrame shape={df.shape}) ===")
    show_dataframe(df, n=args.num, full=args.full)


if __name__ == "__main__":
    main()
