"""
    DYCO Dynamic Lag Compensation
    Copyright (C) 2020-2025 Lukas Hörtnagl

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""

from pathlib import Path

import pandas as pd

from dyco._vendor.fileio import read_parquet

# Compression suffixes pandas infers on its own. A file named `x.csv.gz` has
# Path.suffix == '.gz', so dispatching on that alone rejects every compressed
# file - which is what happened until v3, even though `FileSplitterMulti` writes
# `.csv.gz` when compress_splits=True and its output could not be read back.
_COMPRESSION_SUFFIXES = {'.gz', '.gzip', '.bz2', '.zip', '.xz', '.zst', '.zstd', '.tar'}


def data_suffix(filepath) -> str:
    """Return the format-bearing suffix of *filepath*, ignoring compression.

    ``'raw.csv'`` and ``'raw.csv.gz'`` both give ``'.csv'``. Returns ``''`` for a
    name with no usable suffix.
    """
    suffixes = [s.lower() for s in Path(filepath).suffixes]
    while suffixes and suffixes[-1] in _COMPRESSION_SUFFIXES:
        suffixes.pop()
    return suffixes[-1] if suffixes else ''


def read_raw_data(filepath, data_timestamp_format):
    """
    Read raw data files

    Compressed files are handled: `.csv.gz` and friends read the same as a plain
    `.csv`, because pandas infers the compression from the name. Only the format
    suffix decides how the file is parsed.

    Parameters
    ----------
    filepath: Path
        Full path to raw data file.
    data_timestamp_format: str
        Datetime format of the timestamp in each data row in the raw data file.

    Returns
    -------
    pandas DataFrame that contains raw data from the file in filepath
    """

    file_ext = data_suffix(filepath)

    if file_ext == '.csv':
        data_df = read_raw_data_csv(filepath, data_timestamp_format)

    elif file_ext == '.parquet':
        data_df = read_parquet(filepath)

    else:
        raise ValueError(
            f"Cannot read {Path(filepath).name}: the format suffix must be '.csv' or "
            f"'.parquet', optionally followed by a compression suffix such as '.gz'. "
            f"Detected format suffix: {file_ext!r}.")

    return data_df


def read_raw_data_csv(filepath, data_timestamp_format):
    header_rows_list = [0]
    skip_rows_list = []
    header_section_rows = [0]

    num_data_cols = \
        length_data_cols(filepath=filepath,
                         header_rows_list=header_rows_list,
                         skip_rows_list=skip_rows_list)

    num_header_cols, header_cols_df = \
        length_header_cols(filepath=filepath,
                           header_rows_list=header_rows_list,
                           skip_rows_list=skip_rows_list)

    more_data_cols_than_header_cols, num_missing_header_cols = \
        data_vs_header(num_data_cols=num_data_cols,
                       num_header_cols=num_header_cols)

    header_cols_list = \
        generate_missing_cols(header_cols_df=header_cols_df,
                              more_data_cols_than_header_cols=more_data_cols_than_header_cols,
                              num_missing_header_cols=num_missing_header_cols)

    if data_timestamp_format:
        # parse = lambda x: dt.datetime.strptime(x, data_timestamp_format)  # now deprecated
        # date_parser = parse  # now deprecated
        parse_dates = True
        index_col = 0
    else:
        # date_parser = None  # now deprecated
        parse_dates = False
        index_col = None

    data_df = pd.read_csv(filepath,
                          skiprows=header_section_rows,
                          header=None,
                          names=header_cols_list,
                          na_values=-9999,
                          encoding='utf-8',
                          delimiter=',',
                          # mangle_dupe_cols=True,  # now deprecated
                          # keep_date_col=False,  # now deprecated
                          parse_dates=parse_dates,
                          # date_parser=date_parser,  # now deprecated
                          date_format=data_timestamp_format,
                          index_col=index_col,
                          dtype=None,
                          engine='c',
                          nrows=None)

    return data_df


def generate_missing_cols(header_cols_df, more_data_cols_than_header_cols, num_missing_header_cols):
    """
    Insert additional column names in data header

    Additional columns are created if the number of data columns
    does not match the number of header columns.

    Parameters
    ----------
    header_cols_df: pandas DataFrame
        A small DataFrame that only contains the header columns.
    more_data_cols_than_header_cols: bool
        True if more data columns than header columns were found in
        the data file. This can happen due to irregularities during
        raw data collection.
    num_missing_header_cols: int
        Number of missing header columns in comparison to data columns.

    Returns
    -------
    list of header columns that contains labels for additionally created columns
    """
    # Generate missing header columns if necessary
    header_cols_list = header_cols_df.columns.to_list()
    generated_missing_header_cols_list = []
    if more_data_cols_than_header_cols:
        for m in list(range(1, num_missing_header_cols + 1)):
            missing_col = (f'unknown_{m}')
            generated_missing_header_cols_list.append(missing_col)
            header_cols_list.append(missing_col)
    return header_cols_list


def length_data_cols(filepath, header_rows_list, skip_rows_list):
    """
    Check number of columns of the first data row after the header part

    Parameters
    ----------
    filepath: Path
        Path to raw data file
    header_rows_list: list
        List of integers that give the row positions of the header lines
    skip_rows_list: list
        List of skipped rows

    Returns
    -------
    Number of data columns
    """
    skip_num_lines = len(header_rows_list) + len(skip_rows_list)
    first_data_row_df = pd.read_csv(filepath,
                                    skiprows=skip_num_lines,
                                    header=None,
                                    nrows=1)

    return first_data_row_df.columns.size


def length_header_cols(filepath, header_rows_list, skip_rows_list):
    """
    Check number of columns of the header part

    Parameters
    ----------
    filepath: Path
        Path to raw data file
    header_rows_list: list
        List of integers that give the row positions of the header lines
    skip_rows_list: list
        List of skipped rows

    Returns
    -------
    Number of header columns and a pandas DataFrame that contains only the header columns
    """
    header_cols_df = pd.read_csv(filepath,
                                 skiprows=skip_rows_list,
                                 header=header_rows_list,
                                 nrows=0)
    return header_cols_df.columns.size, header_cols_df


def data_vs_header(num_data_cols, num_header_cols):
    """
    Check if there are more data columns than header columns

    Parameters
    ----------
    num_data_cols: int
        Number of data columns
    num_header_cols: int
        Number of header columns

    Returns
    -------
    more_data_cols_than_header_cols: bool
        True if number of data columns > number of header columns
    num_missing_header_cols: int
        Number of missing header columns compared to number of data columns
    """
    if num_data_cols > num_header_cols:
        more_data_cols_than_header_cols = True
        num_missing_header_cols = num_data_cols - num_header_cols
    else:
        more_data_cols_than_header_cols = False
        num_missing_header_cols = 0
    return more_data_cols_than_header_cols, num_missing_header_cols
