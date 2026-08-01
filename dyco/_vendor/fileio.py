"""
FILEIO: FILE DISCOVERY AND PARQUET READING
===========================================

Copied from diive `core/io/filereader.py` (`search_files`). The parquet reader
replaces diive's `load_parquet`, which dyco only ever called with both of its
optional behaviours switched off - see `read_parquet` below.

Part of the dyco package: https://github.com/holukas/dyco
"""

import fnmatch
import os
from pathlib import Path

import pandas as pd
from pandas import DataFrame


def search_files(searchdirs: str or list, pattern: str) -> list:
    """Search directories recursively for files matching a glob *pattern*.

    Args:
        searchdirs: One directory as a string, or a list of directories.
        pattern: Filename glob, e.g. `'*.csv'` or `'CH-DAS_*.csv.gz'`.

    Returns:
        Sorted list of `Path` objects.
    """
    foundfiles = []
    if isinstance(searchdirs, str):
        searchdirs = [searchdirs]
    for searchdir in searchdirs:
        for root, dirs, files in os.walk(searchdir):
            for filename in files:
                if fnmatch.fnmatch(filename, pattern):
                    foundfiles.append(Path(root) / filename)
    foundfiles.sort()
    return foundfiles


def read_parquet(filepath: str or Path) -> DataFrame:
    """Read a Parquet file into a DataFrame, preserving the stored index.

    Replaces diive's `load_parquet`. dyco only ever called that with
    `output_middle_timestamp=False, sanitize_timestamp=False`, which switched
    off both of the things it added over a plain read - timestamp sanitization
    and end/start-to-middle conversion. With both off it reduced to exactly this.

    Raw high-frequency files are not on a sanitizable regular grid anyway, so
    the sanitizing path was never appropriate here.
    """
    return pd.read_parquet(filepath)
