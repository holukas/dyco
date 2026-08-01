"""
TIMES: TIMESTAMP RECONSTRUCTION FOR RAW EC FILES
=================================================

Build a true timestamp index for a raw data file from its record count and
expected duration.

Copied verbatim (docstrings condensed) from diive
`core/times/times.py` (`calc_true_resolution`, `create_timestamp`).

Both still exist in diive - they were deliberately kept there rather than
moved, so these are true copies, not relocations. If a bug is found in one,
check the other.

Part of the dyco package: https://github.com/holukas/dyco
"""

import numpy as np
import pandas as pd


def calc_true_resolution(num_records: int,
                         data_nominal_res: float,
                         expected_records: int,
                         expected_duration: int) -> float:
    """Calculate the true time resolution of the raw data, in seconds.

    Files measured at a given nominal resolution may still produce more or
    fewer records than expected, because of small inaccuracies in the logger's
    internal clock. When the record count is within 0.1% of expectation the
    resolution is recomputed from the actual count; otherwise the nominal
    resolution is kept.

    Args:
        num_records: Number of raw data records found in the file.
        data_nominal_res: Nominal resolution in seconds, e.g. 0.05 for 20 Hz.
        expected_records: Records expected given the nominal resolution.
        expected_duration: Expected file duration in seconds.

    Returns:
        True resolution in seconds, e.g. 0.05 for 20 Hz.
    """
    ratio = num_records / expected_records
    if (ratio > 0.999) and (ratio < 1.001):
        true_resolution = np.float64(expected_duration / num_records)
    else:
        true_resolution = data_nominal_res
    return true_resolution


def create_timestamp(df, file_start, data_nominal_res, expected_duration):
    """Insert a true timestamp index based on record count and file duration.

    Deriving the timestamp from the true resolution rather than the nominal one
    avoids overlapping records when consecutive files are merged: without it the
    last timestamp of file N can equal the first timestamp of file N+1, and the
    duplicate is dropped. The loss is small (order 3 records in 432 000) but
    missing records are undesirable when calculating covariances.

    Args:
        df: Raw data without timestamp. An existing timestamp is overwritten.
        file_start: Start time of the file.
        data_nominal_res: Nominal resolution in seconds, e.g. 0.05 for 20 Hz.
        expected_duration: Expected file duration in seconds, e.g. 1800.

    Returns:
        (df, true_resolution) - the frame with a TIMESTAMP index, and the
        resolution in seconds that was used to build it.
    """
    n_records = len(df)
    expected_records = int(expected_duration / data_nominal_res)
    true_resolution = calc_true_resolution(num_records=n_records,
                                           data_nominal_res=data_nominal_res,
                                           expected_records=expected_records,
                                           expected_duration=expected_duration)
    df['sec'] = df.index * true_resolution
    df['file_start_dt'] = file_start
    df['TIMESTAMP'] = pd.to_datetime(df['file_start_dt']) + pd.to_timedelta(df['sec'], unit='s')
    df = df.drop(['sec', 'file_start_dt'], axis=1, inplace=False)
    df = df.set_index('TIMESTAMP', inplace=False)
    df.index = df.index.round(freq='50ms')  # Round to 50 ms accuracy
    return df, true_resolution
