"""
FRAMES: DATAFRAME HELPERS
==========================

Copied from diive `core/dfun/frames.py` (`trim_frame`).

`trim_frame` still exists in diive - it was deliberately kept there rather than
moved, so this is a true copy, not a relocation. If a bug is found in one, check
the other.

Part of the dyco package: https://github.com/holukas/dyco
"""

import pandas as pd
from pandas import DataFrame


def trim_frame(df: DataFrame, var: str) -> DataFrame:
    """Trim the start and end of *df* to the first and last valid record of *var*.

    Only the start and end are trimmed; missing values of *var* in between are
    ignored. If *df* holds data between 05:00 and 06:00 but *var* is only
    available between 05:20 and 05:50, everything before 05:20 and after 05:50
    is removed.

    Args:
        df: Dataframe that contains *var*.
        var: Name of the variable used to trim *df*.

    Returns:
        Trimmed dataframe, or an empty one if *var* has no valid records.
    """
    records = df[var].copy()
    records = records.dropna()
    if records.empty:
        df = pd.DataFrame()
    else:
        first_record = records.index[0]
        last_record = records.index[-1]
        keep = (df.index >= first_record) & (df.index <= last_record)
        df = df[keep].copy()
    return df
