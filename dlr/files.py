import datetime as dt
import fnmatch
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd





def read_segments_file(filepath):
    """
    Read file.

    Is used for reading segment covariances and lag search
    results for each segment. Can be used for all text files
    for which the .read_csv args are valid.

    Parameters
    ----------
    filepath: str

    Returns
    -------
    pandas DataFrame

    """
    # parse = lambda x: dt.datetime.strptime(x, '%Y%m%d%H%M%S')
    import time
    start_time = time.time()
    found_lags_df = pd.read_csv(filepath,
                                skiprows=None,
                                header=0,
                                # names=header_cols_list,
                                # na_values=-9999,
                                encoding='utf-8',
                                delimiter=',',
                                mangle_dupe_cols=True,
                                # keep_date_col=False,
                                parse_dates=False,
                                # date_parser=parse,
                                index_col=0,
                                dtype=None,
                                engine='c')
    # print(f"Read file {filepath} in {time.time() - start_time}s")
    return found_lags_df


def read_raw_data(filepath, data_timestamp_format):
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

    start_time = time.time()
    data_df = pd.read_csv(filepath,
                          skiprows=header_section_rows,
                          header=None,
                          names=header_cols_list,
                          na_values=-9999,
                          encoding='utf-8',
                          delimiter=',',
                          mangle_dupe_cols=True,
                          keep_date_col=False,
                          parse_dates=parse_dates,
                          date_parser=date_parser,
                          index_col=index_col,
                          dtype=None,
                          engine='c',
                          nrows=None)
    print(f"Reading file took {time.time() - start_time}s")

    return data_df


def calc_true_resolution(num_records, data_nominal_res, expected_records, expected_duration):
    ratio = num_records / expected_records
    if (ratio > 0.999) and (ratio < 1.001):
        file_complete = True
        true_resolution = np.float64(expected_duration / num_records)
    else:
        file_complete = False
        true_resolution = data_nominal_res
    return true_resolution


def insert_timestamp(df, file_info_row, num_records, data_nominal_res, expected_records, expected_duration):
    true_resolution = calc_true_resolution(num_records=num_records, data_nominal_res=data_nominal_res,
                                           expected_records=expected_records, expected_duration=expected_duration)
    df['sec'] = df.index * true_resolution
    df['file_start_dt'] = file_info_row['start']
    df['TIMESTAMP'] = pd.to_datetime(df['file_start_dt']) \
                      + pd.to_timedelta(df['sec'], unit='s')
    df.drop(['sec', 'file_start_dt'], axis=1, inplace=True)
    df.set_index('TIMESTAMP', inplace=True)

    return df


def add_data_stats(df, true_resolution, filename, files_overview_df, found_records):
    # Detect overall frequency
    data_duration = found_records * true_resolution
    data_freq = np.float64(found_records / data_duration)

    files_overview_df.loc[filename, 'first_record'] = df.index[0]
    files_overview_df.loc[filename, 'last_record'] = df.index[-1]
    files_overview_df.loc[filename, 'file_duration'] = (df.index[-1] - df.index[0]).total_seconds()
    files_overview_df.loc[filename, 'found_records'] = found_records
    files_overview_df.loc[filename, 'data_freq'] = data_freq

    return files_overview_df


def generate_missing_cols(header_cols_df, more_data_cols_than_header_cols, num_missing_header_cols):
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
    # Check number of columns of the first data row after the header part
    skip_num_lines = len(header_rows_list) + len(skip_rows_list)
    first_data_row_df = pd.read_csv(filepath,
                                    skiprows=skip_num_lines,
                                    header=None,
                                    nrows=1)
    return first_data_row_df.columns.size


def length_header_cols(filepath, header_rows_list, skip_rows_list):
    # Check number of columns of the header part
    header_cols_df = pd.read_csv(filepath,
                                 skiprows=skip_rows_list,
                                 header=header_rows_list,
                                 nrows=0)
    return header_cols_df.columns.size, header_cols_df


def data_vs_header(num_data_cols, num_header_cols):
    # Check if there are more data columns than header columns
    if num_data_cols > num_header_cols:
        more_data_cols_than_header_cols = True
        num_missing_header_cols = num_data_cols - num_header_cols
    else:
        more_data_cols_than_header_cols = False
        num_missing_header_cols = 0
    return more_data_cols_than_header_cols, num_missing_header_cols



