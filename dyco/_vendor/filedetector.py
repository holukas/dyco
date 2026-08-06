"""
FILEDETECTOR: OVERVIEW OF AVAILABLE AND MISSING RAW FILES
==========================================================

Builds a dataframe of expected file start times and marks which of them are
actually present, so downstream code knows about gaps in the record.

Copied from diive `core/io/filedetector.py` (`FileDetector` only). The other
helpers in that module are not used by dyco. diive's `error()`-then-`sys.exit()`
on an empty file list is replaced by a `ValueError` - a library should not exit
the interpreter.

Part of the dyco package: https://github.com/holukas/dyco
"""

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame


def add_data_stats(df, true_resolution, filename, found_records) -> DataFrame:
    """Return a one-row stats DataFrame for a single raw file.

    Columns: first_record, last_record, file_duration, found_records, data_freq.

    A second, six-argument variant lived in `files.py` and served the v2
    covariance-maximization path; it went with that path in v3.0.0. This is the
    diive version, and now the only one. Used by the file splitter.
    """
    cols = ['first_record', 'last_record', 'file_duration', 'found_records', 'data_freq']
    filestats_df = DataFrame(columns=cols)

    data_duration = found_records * true_resolution
    data_freq = np.float64(found_records / data_duration)

    filestats_df.loc[filename, 'first_record'] = df.index[0]
    filestats_df.loc[filename, 'last_record'] = df.index[-1]
    filestats_df.loc[filename, 'file_duration'] = (df.index[-1] - df.index[0]).total_seconds()
    filestats_df.loc[filename, 'found_records'] = found_records
    filestats_df.loc[filename, 'data_freq'] = data_freq

    return filestats_df


class FileDetector:
    """Build an overview of available and missing (expected) data files."""

    def __init__(self,
                 filelist: list,
                 file_date_format: str,
                 file_generation_res: str,
                 data_res: float,
                 files_how_many: int = None):
        """Create overview dataframe of available and missing (expected) files.

        Args:
            filelist: List of found files (`Path` objects).
            file_date_format: Datetime format of the info contained in the file
                name, e.g. `'CH-DAS_%Y%m%d%H%M.csv.gz'`.
            file_generation_res: Regular interval at which files were created,
                e.g. `'6h'` for every 6 hours.
            data_res: Interval in seconds at which data are logged, e.g. 0.05.
            files_how_many: Optional cap on the number of available files kept.
        """
        self.filelist = filelist
        self.file_date_format = file_date_format
        self.file_generation_res = file_generation_res
        self.data_res = data_res
        self.files_how_many = files_how_many

        if not self.filelist:
            raise ValueError("*filelist* must not be empty.")

        self._files_overview_df = DataFrame()

    @property
    def files_overview_df(self) -> DataFrame:
        """Return the files-overview DataFrame."""
        if not isinstance(self._files_overview_df, DataFrame):
            raise Exception('No results available.')
        return self._files_overview_df

    def get_results(self) -> DataFrame:
        """Return the files-overview DataFrame."""
        return self.files_overview_df

    def run(self):
        """Execute full processing stack."""
        self._files_overview_df = self.add_expected()
        self._files_overview_df = self.add_unexpected()
        self._files_overview_df = self.calc_expected_values()
        self._files_overview_df.loc[:, 'file_available'] = \
            self.files_overview_df.loc[:, 'file_available'].fillna(0, inplace=False)
        self._files_overview_df = self.restrict_numfiles()

    def restrict_numfiles(self) -> DataFrame:
        """Trim the overview to the first *files_how_many* available files (if set)."""
        _files_overview_df = self.files_overview_df.copy()
        if self.files_how_many:
            for idx in _files_overview_df.index:
                _restricted_df = _files_overview_df.loc[_files_overview_df.index[0]:idx]
                num_available_files = _restricted_df['file_available'].sum()
                if num_available_files >= self.files_how_many:
                    _files_overview_df = _restricted_df.copy()
                    break
        return _files_overview_df

    def add_expected(self) -> DataFrame:
        """Index the expected (regular) file start times and mark which exist."""
        first_file_dt = dt.datetime.strptime(self.filelist[0].name, self.file_date_format)
        last_file_dt = dt.datetime.strptime(self.filelist[-1].name, self.file_date_format)
        expected_end_dt = last_file_dt + pd.Timedelta(self.file_generation_res)
        expected_index_dt = pd.date_range(first_file_dt, expected_end_dt, freq=self.file_generation_res)
        files_df = pd.DataFrame(index=expected_index_dt)

        for filepath in self.filelist:
            filename = filepath.name
            file_start_dt = dt.datetime.strptime(filename, self.file_date_format)
            if file_start_dt in files_df.index:
                files_df.loc[file_start_dt, 'file_available'] = 1
                files_df.loc[file_start_dt, 'filename'] = filename
                files_df.loc[file_start_dt, 'start'] = file_start_dt
                files_df.loc[file_start_dt, 'filepath'] = filepath
                files_df.loc[file_start_dt, 'filesize'] = Path(filepath).stat().st_size

        files_df.insert(0, 'expected_file', files_df.index)
        return files_df

    def add_unexpected(self) -> DataFrame:
        """Add files whose start time falls outside the regular grid."""
        files_df = self.files_overview_df.copy()
        for filepath in self.filelist:
            filename = filepath.name
            file_start_dt = dt.datetime.strptime(filename, self.file_date_format)
            if file_start_dt not in files_df.index:
                files_df.loc[file_start_dt, 'file_available'] = 1
                files_df.loc[file_start_dt, 'filename'] = filename
                files_df.loc[file_start_dt, 'start'] = file_start_dt
                files_df.loc[file_start_dt, 'filepath'] = filepath
                files_df.loc[file_start_dt, 'filesize'] = Path(filepath).stat().st_size
        return files_df.sort_index(inplace=False)

    def calc_expected_values(self) -> DataFrame:
        """Calculate expected end time, duration and record count per file."""
        files_df = self.files_overview_df.copy()
        files_df['expected_end'] = files_df.index
        files_df['expected_end'] = files_df['expected_end'].shift(-1)
        files_df['expected_duration'] = (files_df['expected_end'] - files_df['start']).dt.total_seconds()
        files_df['expected_records'] = files_df['expected_duration'] / self.data_res
        return files_df
