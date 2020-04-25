import fnmatch
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt


def read_found_lags_file(filepath):
    parse = lambda x: dt.datetime.strptime(x, '%Y%m%d%H%M%S')
    found_lags_df = pd.read_csv(filepath,
                                skiprows=None,
                                header=0,
                                # names=header_cols_list,
                                # na_values=-9999,
                                encoding='utf-8',
                                delimiter=',',
                                mangle_dupe_cols=True,
                                keep_date_col=False,
                                parse_dates=True,
                                date_parser=parse,
                                index_col=0,
                                dtype=None,
                                engine='python')

    found_lags_df['shift_median'] = np.nan
    found_lags_df['shift_P25'] = np.nan
    found_lags_df['shift_P75'] = np.nan

    args = dict(window=100, min_periods=50, center=True)
    found_lags_df['shift_median'] = found_lags_df['cov_max_shift'].rolling(**args).median()
    found_lags_df['search_win_upper'] = found_lags_df['shift_median'] + 100
    found_lags_df['search_win_lower'] = found_lags_df['shift_median'] - 100
    found_lags_df['shift_P25'] = found_lags_df['cov_max_shift'].rolling(**args).quantile(0.25)
    found_lags_df['shift_P75'] = found_lags_df['cov_max_shift'].rolling(**args).quantile(0.75)

    cols = ['shift_median', 'cov_max_shift', 'shift_P25','shift_P75','search_win_upper','search_win_lower']
    found_lags_df[cols].plot()
    plt.show()
    return None


def read_raw_data(filepath, nrows, time_res_hz, df_start_dt, file_duration):
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

    data_df = pd.read_csv(filepath,
                          skiprows=header_section_rows,
                          header=None,
                          names=header_cols_list,
                          na_values=-9999,
                          encoding='utf-8',
                          delimiter=',',
                          mangle_dupe_cols=True,
                          keep_date_col=False,
                          parse_dates=False,
                          date_parser=None,
                          index_col=None,
                          dtype=None,
                          engine='python',
                          nrows=nrows)

    data_df = insert_datetime_index(df=data_df, df_start_dt=df_start_dt, file_duration=file_duration)

    stats(df=data_df, time_res=time_res_hz)

    return data_df


def insert_datetime_index(df, df_start_dt, file_duration):
    """Insert true timestamp based on number of records in the file and the
    file duration.

    Files measured at a given time resolution may still produce
    more or less than the expected number of errors.

    For example, a six-hour file with data recorded at 20Hz is expected to have
    432 000 records, but may in reality produce slightly more or less than that
    due to small inaccuracies in the measurements instrument's internal clock.
    This in turn would mean that the defined time resolution of 20Hz is not
    completely accurate with the true frequency being slightly higher or lower.

    This causes a (minor) issue when merging mutliple data files due to overlapping
    record timestamps, i.e. the last timestamp in file #1 is the same as the first
    timestamp in file #2, resulting in duplicate entries in the timestamp index column
    during merging of files #1 and #2.

    In addition, sometimes more than one timestamp can overlap, resulting in more
    overlapping timestamps and therefore more data loss. Although this data loss is
    minor (1-2 records per 432 000 records), missing records are not desirable when
    calculating covariances between times series. The time series must be as complete
    and without missing records as possible to avoid errors.
    """
    num_records = len(df)
    timestamp_step = file_duration / num_records
    df['sec'] = df.index * timestamp_step
    df['file_start_dt'] = df_start_dt
    df['TIMESTAMP'] = pd.to_datetime(df['file_start_dt']) \
                      + pd.to_timedelta(df['sec'], unit='s')
    df.drop(['sec', 'file_start_dt'], axis=1, inplace=True)
    df.set_index('TIMESTAMP', inplace=True)
    return df


def stats(df, time_res):
    # Detect overall frequency
    data_records = len(df)
    data_duration = data_records / time_res
    data_freq = float(data_records / data_duration)

    print(f"First record: {df.index[0]}")
    print(f"Last record: {df.index[-1]}")
    print(f"Data duration: {df.index[-1] - df.index[0]}")
    print(f"Data records: {data_records}")
    print(f"Data frequency: {data_freq}Hz")


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


def search(dir, pattern):
    found_files = []
    for root, dirs, files in os.walk(dir):
        root = Path(root)
        for idx, filename in enumerate(files):
            if fnmatch.fnmatch(filename, pattern):
                filepath = Path(root) / Path(filename)
                found_files.append(filepath)
    found_files.sort()  # Sorts inplace
    return found_files


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
