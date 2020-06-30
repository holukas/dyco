import datetime as dt
import fnmatch
import os
import sys
from pathlib import Path

import pandas as pd


class Setup:
    def __init__(self, outdir, del_previous_results, dir_input, fnm_pattern, fnm_date_format,
                 file_generation_res, dat_recs_nominal_timeres, files_how_many):
        self.outdir = outdir
        self.del_previous_results = del_previous_results
        self.dir_input = dir_input
        self.fnm_pattern = fnm_pattern
        self.fnm_date_format = fnm_date_format
        self.file_generation_res = file_generation_res
        self.dat_recs_nominal_timeres = dat_recs_nominal_timeres
        self.files_how_many = files_how_many

    def run(self):
        self.outdirs = self.setup_output_dirs()
        self.files_overview_df = self.search_files()

    def get(self):
        return self.outdirs, self.files_overview_df

    def search_files(self):
        """
        Search available files.

        Returns
        -------
        pandas DataFrame

        """
        fd = FilesDetector(dir_input=self.dir_input,
                           outdir=self.outdirs['0-0_Found_Files'],
                           file_pattern=self.fnm_pattern,
                           file_date_format=self.fnm_date_format,
                           file_generation_res=self.file_generation_res,
                           data_res=self.dat_recs_nominal_timeres)
        fd.run()
        files_overview_df = fd.get()

        # Consider file limit
        if self.files_how_many:
            for idx, file in files_overview_df.iterrows():
                _df = files_overview_df.loc[files_overview_df.index[0]:idx]
                num_available_files = _df['file_available'].sum()
                if num_available_files >= self.files_how_many:
                    files_overview_df = _df.copy()
                    break
        return files_overview_df

    def new_dirs(self):
        new_dirs = [
            '0-0_Found_Files',
            '1-0_Covariances',
            '1-1_____Plots',
            '2-0_Segment_Lag_Times',
            '2-1_____Histograms',
            '2-2_____Timeseries',
            '3-0_XXX']
        return new_dirs

    def setup_output_dirs(self):
        """Make output directories."""
        new_dirs = self.new_dirs()
        outdirs = {}

        # Store keys and full paths in dict
        for nd in new_dirs:
            outdirs[nd] = self.outdir / nd

        # Make dirs
        for key, path in outdirs.items():
            if not Path.is_dir(path):
                print(f"Creating folder {path} ...")
                os.makedirs(path)
            else:
                if self.del_previous_results:
                    for filename in os.listdir(path):
                        filepath = os.path.join(path, filename)
                        try:
                            if os.path.isfile(filepath) or os.path.islink(filepath):
                                print(f"Deleting file {filepath} ...")
                                os.unlink(filepath)
                            # elif os.path.isdir(filepath):
                            #     shutil.rmtree(filepath)
                        except Exception as e:
                            print('Failed to delete %s. Reason: %s' % (filepath, e))

        return outdirs


class FilesDetector:
    """Create overview dataframe of available and missing (expected) files."""
    found_files = []
    files_overview_df = pd.DataFrame()

    def __init__(self, dir_input, outdir, file_pattern, file_date_format, file_generation_res, data_res):
        """Initialize with basic file information.

        :param dir_input: Input directory to search for files
        :type dir_input: Path
        :param file_pattern: Pattern to identify files
        :type file_pattern: str
        :param file_date_format: Datetime information contained in file name
        :type file_date_format: str
        :param file_generation_res: Regular interval at which files were created, e.g. '6H' for every 6 hours
        :type file_generation_res: str
        :param data_res: Interval in seconds at which data are logged, e.g. 0.05
        :type data_res: float
        """

        self.dir_input = dir_input
        self.outdir = outdir
        self.pattern = file_pattern
        self.file_date_format = file_date_format
        self.file_generation_res = file_generation_res
        self.data_res = data_res

    def run(self):
        """Execute full processing stack."""
        self.found_files = self.search_available(dir=self.dir_input, pattern=self.pattern)
        if not self.found_files:
            print(f"\n(!)ERROR No files found with pattern {self.pattern}. Stopping script.")
            sys.exit()

        self.files_overview_df = self.add_expected()
        self.files_overview_df = self.add_unexpected()
        self.files_overview_df = self.calc_expected_values()
        self.files_overview_df.loc[:, 'file_available'].fillna(0, inplace=True)
        self.export()

    def get(self):
        return self.files_overview_df

    def export(self):
        """Export dataframe to csv."""
        # self.dir_output = self.dir_output / '1_found_files'
        # if not os.path.isdir(self.dir_output):
        #     os.makedirs(self.dir_output)
        outpath = self.outdir / '0_files_overview.csv'
        self.files_overview_df.to_csv(outpath)

    @staticmethod
    def search_available(dir, pattern):
        """Search files in dir.

        :param dir: Directory that is searched
        :param pattern: Pattern to identify files
        :return: List of found files
        :rtype: list
        """

        found_files = []
        for root, dirs, files in os.walk(dir):
            root = Path(root)
            for idx, filename in enumerate(files):
                if fnmatch.fnmatch(filename, pattern):
                    filepath = Path(root) / Path(filename)
                    found_files.append(filepath)
        found_files.sort()  # Sorts inplace
        return found_files

    def add_expected(self):
        """Create index of expected files (regular start time) and check
        which of these regular files are available.

        :return: DataFrame with info about regular (expected) files
        :rtype: pandas DataFrame
        """
        first_file_dt = dt.datetime.strptime(self.found_files[0].stem, self.file_date_format)
        last_file_dt = dt.datetime.strptime(self.found_files[-1].stem, self.file_date_format)
        expected_end_dt = last_file_dt + pd.Timedelta(self.file_generation_res)
        expected_index_dt = pd.date_range(first_file_dt, expected_end_dt, freq=self.file_generation_res)
        files_df = pd.DataFrame(index=expected_index_dt)

        for file_idx, filepath in enumerate(self.found_files):
            filename = filepath.stem
            file_start_dt = dt.datetime.strptime(filename, self.file_date_format)

            if file_start_dt in files_df.index:
                files_df.loc[file_start_dt, 'file_available'] = 1
                files_df.loc[file_start_dt, 'filename'] = filename
                files_df.loc[file_start_dt, 'start'] = file_start_dt
                files_df.loc[file_start_dt, 'filepath'] = filepath
                files_df.loc[file_start_dt, 'filesize'] = Path(filepath).stat().st_size
                # files_df.loc[file_start_dt, 'expected_file'] = file_start_dt

        files_df.insert(0, 'expected_file', files_df.index)  # inplace
        return files_df

    def add_unexpected(self):
        """Add info about unexpected files (irregular start time).

        :return: DataFrame with added info about irregular files
        :rtype: pandas DataFrame
        """
        files_df = self.files_overview_df.copy()
        for file_idx, filepath in enumerate(self.found_files):
            filename = filepath.stem
            file_start_dt = dt.datetime.strptime(filename, self.file_date_format)

            if file_start_dt not in files_df.index:
                files_df.loc[file_start_dt, 'file_available'] = 1
                files_df.loc[file_start_dt, 'filename'] = filename
                files_df.loc[file_start_dt, 'start'] = file_start_dt
                files_df.loc[file_start_dt, 'filepath'] = filepath
                files_df.loc[file_start_dt, 'filesize'] = Path(filepath).stat().st_size

        files_df.sort_index(inplace=True)
        return files_df

    def calc_expected_values(self):
        """Calculate expected end time, duration and records of files

        :return: DataFrame with added info about expected values
        """
        files_df = self.files_overview_df.copy()
        files_df['expected_end'] = files_df.index
        files_df['expected_end'] = files_df['expected_end'].shift(-1)
        files_df['expected_duration'] = (files_df['expected_end'] - files_df['start']).dt.total_seconds()
        files_df['expected_records'] = files_df['expected_duration'] / self.data_res
        # files_df['expected_end'] = files_df['start'] + pd.Timedelta(file_generation_res)
        # files_df.loc[files_df['file_available'] == 1, 'next_file'] = files_df['expected_file']
        # files_df['next_file'] = files_df['next_file'].shift(-1)
        return files_df
