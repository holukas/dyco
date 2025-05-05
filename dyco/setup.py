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

import datetime as dt
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
from diive.core.io.filereader import search_files


def set_dirs(indir: None or Path,
             outdir: None or Path):
    """
    In case no input or output directory is given, use folders 'input'
    and 'output' in current working directory.

    Returns
    -------
    Input and output directories as Path

    """
    cwd = Path(os.path.dirname(os.path.abspath(__file__)))  # Current working directory
    if not indir:
        indir = cwd / 'input'
    if not outdir:
        outdir = cwd / 'output'
    return indir, outdir


def create_logger(name: str, logfile_path: Path = None):
    """
    Create name logger and log outputs to file

    A new logger is only created if name does not exist.

    Parameters
    ----------
    logfile_path: Path
        Path to the log file to which the log output is saved.
    name:
        Corresponds to the __name__ of the calling file.

    Returns
    -------
    logger class

    References
    ----------
    https://www.youtube.com/watch?v=jxmzY9soFXg
    https://stackoverflow.com/questions/53129716/how-to-check-if-a-logger-exists
    """

    logger = logging.getLogger(name)

    # Only create new logger if it does not exist already for the respective module,
    # otherwise the log output would be printed x times because a new logger is
    # created everytime the respective module is called.
    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s:%(name)s:  %(message)s')

        file_handler = logging.FileHandler(logfile_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler(stream=sys.stdout)
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger


class CreateOutputDirs:
    """

    Setup output folder structure

    """

    def __init__(self, dyco_instance):
        self.root_dir = dyco_instance.outdir
        self.del_previous_results = dyco_instance.del_previous_results

    def required_dirs(self):
        """Define output folder names."""
        outdirs = [
            '0_log',
            '1_overview',
            '2_covariances',
            '3_covariances_plots',
            '4_time_lags_overview',
            '5_time_lags_overview_histograms',
            '6_time_lags_overview_timeseries',
            '7_time_lags_lookup_table',
            '8_time_lags_corrected_files',
        ]
        return outdirs

    def setup_output_dirs(self):
        """Make output directories."""
        required_dirs = self.required_dirs()
        outdirs = {}

        # Create Path to required directories, store keys and full paths in dict
        for nd in required_dirs:
            outdirs[nd] = Path(self.root_dir) / nd

        # Delete dirs if needed
        for key, path in outdirs.items():
            if Path.is_dir(path) and self.del_previous_results:
                print(f"Deleting folder {path} ...")
                shutil.rmtree(path)

        # Make dirs
        for key, path in outdirs.items():
            if not Path.is_dir(path):
                print(f"Creating folder {path} ...")
                os.makedirs(path)

        return outdirs


class FilesDetector:
    """
    Create overview dataframe of available and missing (expected) files
    """
    found_files = []
    files_overview_df = pd.DataFrame()

    def __init__(self,
                 dyco_instance,
                 logfile_path: Path = None,
                 outdir: bool or Path = False):
        """
        Parameters
        ----------
        outdir: Path or False
            Export folder to save the files overview to.
        """

        self.indir = dyco_instance.indir
        self.fnm_pattern = dyco_instance.filename_pattern
        self.fnm_date_format = dyco_instance.filename_date_format
        self.file_generation_res = dyco_instance.file_generation_res
        self.dat_recs_nominal_timeres = dyco_instance.data_nominal_timeres
        self.files_how_many = dyco_instance.files_how_many
        self.outdir = outdir

        self.logger = create_logger(logfile_path=logfile_path, name=__name__)

    def run(self):
        """Execute processing stack"""
        self.logger.info("Start file search")
        self.found_files = search_files(searchdirs=self.indir,
                                        pattern=self.fnm_pattern)
        if not self.found_files:
            self.logger.error(f"\n(!)ERROR No files found with pattern {self.fnm_pattern}. Stopping script.")
            sys.exit()

        self.files_overview_df = self.add_expected()
        self.files_overview_df = self.add_unexpected()
        self.files_overview_df = self.calc_expected_values()
        _temp = self.files_overview_df.loc[:, 'file_available'].copy()
        _temp = _temp.fillna(0)
        self.files_overview_df.loc[:, 'file_available'] = _temp
        # self.files_overview_df.loc[:, 'file_available'].fillna(0, inplace=True)
        self.files_overview_df = self.limit_num_files()
        self.logger.info(f"Found {int(self.files_overview_df['file_available'].sum())} available files")

        if self.outdir:
            self.export()

    def get(self):
        """Get files overview as DataFrame."""
        return self.files_overview_df

    def export(self):
        """Export dataframe to csv."""
        outfile = '0_files_overview.csv'
        outpath = self.outdir / outfile
        self.files_overview_df.to_csv(outpath)
        self.logger.info(f"Exported file {outfile}")

    def add_expected(self):
        """
        Create index of expected files (regular start time) and check
        which of these regular files are available

        Returns
        -------
        pandas Dataframe
        """

        first_file_dt = dt.datetime.strptime(self.found_files[0].stem, self.fnm_date_format)
        last_file_dt = dt.datetime.strptime(self.found_files[-1].stem, self.fnm_date_format)
        expected_end_dt = last_file_dt + pd.Timedelta(self.file_generation_res)
        expected_index_dt = pd.date_range(first_file_dt, expected_end_dt, freq=self.file_generation_res)
        files_df = pd.DataFrame(index=expected_index_dt)

        for file_idx, filepath in enumerate(self.found_files):
            filename = filepath.stem
            file_start_dt = dt.datetime.strptime(filename, self.fnm_date_format)

            if file_start_dt in files_df.index:
                files_df.loc[file_start_dt, 'file_available'] = 1
                files_df.loc[file_start_dt, 'filename'] = filename
                files_df.loc[file_start_dt, 'start'] = file_start_dt
                files_df.loc[file_start_dt, 'filepath'] = filepath
                files_df.loc[file_start_dt, 'filesize'] = Path(filepath).stat().st_size

        files_df.insert(0, 'expected_file', files_df.index)  # inplace
        return files_df

    def add_unexpected(self):
        """
        Add info about unexpected files (irregular start time)

        Returns
        -------
        pandas DataFrame with added info about irregular files
        """
        files_df = self.files_overview_df.copy()
        for file_idx, filepath in enumerate(self.found_files):
            filename = filepath.stem
            file_start_dt = dt.datetime.strptime(filename, self.fnm_date_format)

            if file_start_dt not in files_df.index:
                files_df.loc[file_start_dt, 'file_available'] = 1
                files_df.loc[file_start_dt, 'filename'] = filename
                files_df.loc[file_start_dt, 'start'] = file_start_dt
                files_df.loc[file_start_dt, 'filepath'] = filepath
                files_df.loc[file_start_dt, 'filesize'] = Path(filepath).stat().st_size

        files_df.sort_index(inplace=True)
        return files_df

    def calc_expected_values(self):
        """
        Calculate expected end time, duration and number of records for each file

        Returns
        -------
        pandas DataFrame with added info about expected values
        """
        files_df = self.files_overview_df.copy()
        files_df['expected_end'] = files_df.index
        files_df['expected_end'] = files_df['expected_end'].shift(-1)
        files_df['expected_duration'] = (files_df['expected_end'] - files_df['start']).dt.total_seconds()
        files_df['expected_records'] = files_df['expected_duration'] / self.dat_recs_nominal_timeres
        return files_df

    def limit_num_files(self):
        """
        Limit the number of files used

        Returns
        -------
        pandas DataFrame
        """
        if self.files_how_many:
            self.logger.info(f"Limit number of files to {int(self.files_how_many)}")
            for idx, file in self.files_overview_df.iterrows():
                _df = self.files_overview_df.loc[self.files_overview_df.index[0]:idx]
                num_available_files = _df['file_available'].sum()
                if num_available_files >= self.files_how_many:
                    # Stop if enough files
                    self.files_overview_df = _df.copy()
                    break

        files_available = self.files_overview_df['file_available'].sum()
        if files_available == 0:
            msg = f"No files available, stopping script (files_available = {files_available})"
            self.logger.critical(msg)
            sys.exit()

        return self.files_overview_df


def generate_run_id():
    """Generate unique id for this run"""
    script_start_time = time.strftime("%Y-%m-%d %H:%M:%S")
    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_id = f"DYCO-{run_id}"
    return run_id, script_start_time


def set_logfile_path(run_id: str, outdir: Path):
    """Set full path to log file"""
    name = f'{run_id}.log'
    path = outdir / name
    return path
