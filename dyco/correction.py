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

import time
from pathlib import Path

import numpy as np
import pandas as pd

from dyco import files, setup


class RemoveLags:
    """
    Remove time lags: use look-up table to normalize time lags across files
    """

    def __init__(self, files_overview_df, data_timestamp_format, outdirs, var_target, lag_n_iter, logger, lut):
        self.files_overview_df = files_overview_df
        self.dat_recs_timestamp_format = data_timestamp_format
        self.outdirs = outdirs
        self.var_target = var_target
        self.lgs_num_iter = lag_n_iter
        self.logger=logger
        self.lut_df = lut

        self.lut_col = 'correction'

        self.run()

    def run(self):
        """Run lag removal"""

        # Loop files
        num_files = self.files_overview_df['file_available'].sum()
        times_needed = []
        files_counter = 0

        for file_idx, file_info_row in self.files_overview_df.iterrows():
            start = time.time()
            txt_info = ""

            # Check file availability
            if file_info_row['file_available'] == 0:
                continue

            this_date = file_info_row['start'].date()
            this_date = pd.to_datetime(this_date)
            shift_correction = self.lut_df.loc[this_date][self.lut_col]

            # Read and prepare data file
            data_df = files.read_raw_data(filepath=file_info_row['filepath'],
                                          data_timestamp_format=self.dat_recs_timestamp_format)  # nrows for testing

            shift_correction = int(shift_correction)
            data_df = self.shift_var_target(df=data_df, shift=shift_correction)
            data_df = data_df.fillna(-9999)

            # No timestamp for final output files
            self.save_dyco_files(outdir=self.outdirs[f'8_time_lags_corrected_files'],
                                 original_filename=file_info_row['filename'],
                                 df=data_df,
                                 export_timestamp=False)

            time_needed = time.time() - start
            times_needed.append(time_needed)
            files_counter += 1
            times_needed_mean = np.mean(times_needed)
            remaining_files = num_files - files_counter
            remaining_sec = times_needed_mean * remaining_files
            progress = (files_counter / num_files) * 100
            txt_info += f"File #{files_counter}: {file_info_row['filename']}" \
                        f"    shift correction: {shift_correction}    remaining time: {remaining_sec:.0f}s" \
                        f"    remaining files: {int(remaining_files)}    progress: {progress:.2f}%"
            self.logger.info(txt_info)

    def save_dyco_files(self, df, outdir, original_filename, export_timestamp):
        """
        Save lag-removed raw data files as new files with the suffix _DYCO

        Parameters
        ----------
        df: pandas DataFrame
            Data of the raw data file where the lag was removed.
        outdir: Path
            Folder where lag-removed raw data files will be stored.
        original_filename: str
            The original filename of the raw data file.
        export_timestamp: bool
            Defines if the timestamp index is saved to the files.

        Returns
        -------
        None
        """
        df.fillna(-9999, inplace=True)
        outpath = outdir / f"{Path(original_filename).stem}_DYCO.csv"
        df.to_csv(outpath, index=export_timestamp)

    def shift_var_target(self, df, shift):
        """
        Shift data of target columns by found lag

        Parameters
        ----------
        df: pandas DataFrame
        shift: int
            Amount by which target columns are shifted, given as number of records.

        Returns
        -------
        pandas DataFrame of shifted raw data
        """
        for col in self.var_target:
            outcol = f"{col}_DYCO"
            df[outcol] = df[col].shift(shift)  # Shift col by found lag
            df.drop([col], axis=1, inplace=True)  # Remove col that was not shifted
        return df

    def read_lut_time_lags(self):
        """Read csv file that contains the look-up table for lag correction"""
        filepath = self.outdirs[
                       f'6_normalization_lookup_table'] / f'LUT_default_agg_time_lags.csv'
        # parse = lambda x: dt.datetime.strptime(x, '%Y-%m-%d')  # now deprecated
        df = pd.read_csv(filepath,
                         skiprows=None,
                         header=0,
                         # names=header_cols_list,
                         # na_values=-9999,
                         encoding='utf-8',
                         delimiter=',',
                         # mangle_dupe_cols=True,
                         # keep_date_col=False,
                         parse_dates=True,
                         # date_parser=parse,  # now deprecated
                         date_format='%Y-%m-%d',
                         index_col=0,
                         dtype=None,
                         engine='c')
        return df
