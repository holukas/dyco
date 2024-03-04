"""
    DYCO Dynamic Lag Compensation
    Copyright (C) 2020-2024 Lukas Hörtnagl

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

import datetime as dt

import numpy as np
import pandas as pd


def read_segment_lagtimes_file(filepath):
    """
    Read file.

    Reading segment covariances and lag search results for each segment.
    Can be used for all text files for which the .read_csv args are valid.

    Parameters
    ----------
    filepath: str

    Returns
    -------
    pandas DataFrame

    """
    # parse = lambda x: dt.datetime.strptime(x, '%Y%m%d%H%M%S')
    found_lags_df = pd.read_csv(filepath,
                                skiprows=None,
                                header=0,
                                # names=header_cols_list,
                                # na_values=-9999,
                                encoding='utf-8',
                                delimiter=',',
                                # mangle_dupe_cols=True,
                                # keep_date_col=False,
                                parse_dates=False,
                                # date_parser=parse,
                                index_col=0,
                                dtype=None,
                                engine='c')
    return found_lags_df


def read_raw_data(filepath, data_timestamp_format):
    """
    Read raw data files

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
        parse = lambda x: dt.datetime.strptime(x, data_timestamp_format)
        date_parser = parse
        parse_dates = True
        index_col = 0
    else:
        date_parser = None
        parse_dates = False
        index_col = None

    data_df = pd.read_csv(filepath,
                          skiprows=header_section_rows,
                          header=None,
                          names=header_cols_list,
                          na_values=-9999,
                          encoding='utf-8',
                          delimiter=',',
                          # mangle_dupe_cols=True,
                          keep_date_col=False,
                          parse_dates=parse_dates,
                          date_parser=date_parser,
                          index_col=index_col,
                          dtype=None,
                          engine='c',
                          nrows=None)

    return data_df


def calc_true_resolution(num_records, data_nominal_res, expected_records, expected_duration):
    """
    Calculate the true resolution of the raw data

    Parameters
    ----------
    num_records: int
        Number of raw data records.
    data_nominal_res: float
        Nominal time resolution of the raw data, e.g. 0.05 for 20 Hz data
        (one measurement every 0.05 seconds)
    expected_records: int or float
        Expected number of records in the raw data file, based on the nominal
        time resolution of the raw data.
    expected_duration: int
        Expected duration of the raw data file in seconds.

    Returns
    -------
    float that gives the true resolution in seconds, e.g. 0.05s for 20 Hz (1/20 = 0.05)
    """
    ratio = num_records / expected_records
    if (ratio > 0.999) and (ratio < 1.001):
        # file_complete = True
        true_resolution = np.float64(expected_duration / num_records)
    else:
        # file_complete = False
        true_resolution = data_nominal_res
    return true_resolution


def insert_timestamp(df, file_info_row, num_records, data_nominal_res, expected_records, expected_duration):
    """
    Calculate the timestamp for each row record if not available

    Parameters
    ----------
    df: pd.DataFrame
        Raw data without timestamp. In case data already have a timestamp, it
        will be overwritten.
    file_info_row: pandas Series
        Contains info about the current raw data file.
    num_records: int
        Number or records in raw data file.
    data_nominal_res: float
        Nominal time resolution of the raw data, e.g. 0.05 (one measurement every
        0.05 seconds, 20 Hz).
    expected_records: int or float
        Number of expected records in raw data file, e.g. a 30-minute file of
        20 Hz data should contain 36000 records (20 * 60 * 30).
    expected_duration, int or float
        Expected duration of the raw data file in seconds, e.g. 1800 for a
        30-minute file.

    Returns
    -------
    df: pandas DataFrame with timestamp index
    true_resolution: time resolution of raw data measurements
    """
    true_resolution = calc_true_resolution(num_records=num_records, data_nominal_res=data_nominal_res,
                                           expected_records=expected_records, expected_duration=expected_duration)
    df['sec'] = df.index * true_resolution
    df['file_start_dt'] = file_info_row['start']
    df['TIMESTAMP'] = pd.to_datetime(df['file_start_dt']) \
                      + pd.to_timedelta(df['sec'], unit='s')
    df.drop(['sec', 'file_start_dt'], axis=1, inplace=True)
    df.set_index('TIMESTAMP', inplace=True)
    return df, true_resolution


def add_data_stats(df, true_resolution, filename, files_overview_df, found_records, fnm_date_format):
    """
    Collect additional info about raw data file

    Parameters
    ----------
    df: pandas DataFrame
        Raw data from the file.
    true_resolution: float
        Time resolution of the raw data records in seconds, e.g. 0.05 for 20 Hz data.
    filename: str
        Filename of the raw data file, without extension.
    files_overview_df: pandas DataFrame
        Overview of all raw data files, with stats.
    found_records: int
        Number of records in the raw data file.
    fnm_date_format: str
        Datetime format of the datetime info in the raw data filename.

    Returns
    -------
    pandas DataFrame with additional info for current raw data file
    """
    # Detect overall frequency
    data_duration = found_records * true_resolution
    data_freq = np.float64(found_records / data_duration)

    idx = dt.datetime.strptime(filename, fnm_date_format)  # Use filename datetime info as index

    files_overview_df.loc[idx, 'first_record'] = df.index[0]
    files_overview_df.loc[idx, 'last_record'] = df.index[-1]
    files_overview_df.loc[idx, 'file_duration'] = (df.index[-1] - df.index[0]).total_seconds()
    files_overview_df.loc[idx, 'found_records'] = found_records
    files_overview_df.loc[idx, 'data_freq'] = data_freq

    return files_overview_df


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
