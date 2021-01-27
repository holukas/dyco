"""
    DYCO Dynamic Lag Compensation
    Copyright (C) 2020  holukas

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""

import datetime as dt
import time
from pathlib import Path

import numpy as np
import pandas as pd

import files, setup_dyco


class RemoveLags:
    """
    Remove time lags: use look-up table to normalize time lags across files
    """

    def __init__(self, dyco_instance):
        self.files_overview_df = dyco_instance.files_overview_df
        self.dat_recs_timestamp_format = dyco_instance.dat_recs_timestamp_format
        self.outdirs = dyco_instance.outdirs
        self.var_target = dyco_instance.var_target
        self.phase = dyco_instance.phase
        self.phase_files = dyco_instance.phase_files
        self.new_iteration_data = dyco_instance.new_iteration_data  # Indicates if new iteration was executed
        self.lgs_num_iter = dyco_instance.lgs_num_iter

        self.logger = setup_dyco.create_logger(logfile_path=dyco_instance.logfile_path, name=__name__)

        if self.phase == 3:

            dir = Path(self.outdirs[f'{self.phase}-6_{self.phase_files}_final_time_lags_lookup_table'])
            file = Path('LUT_final_time_lags.csv')
            self.lut_df = files.read_segment_lagtimes_file(filepath=dir / file)
            self.lut_col = 'INSTANTANEOUS_LAG'
        else:
            # Read default lag times for reference gas
            self.lut_df = self.read_lut_time_lags()
            self.lut_col = 'correction'

        self.run()

    def run(self):
        # If no new iteration data was created, skip normalization
        if not self.new_iteration_data:
            self.logger.warning("")
            self.logger.warning(f"{'*' * 60}")
            self.logger.warning(f"(!) No new iteration data was created in PHASE {self.phase} for {self.phase_files}.")
            self.logger.warning(f"(!) Skipping PHASE {self.phase}, Step 4: no normalized files will be generated.")
            self.logger.warning(f"{'*' * 60}")
            self.logger.warning("")
            return None

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

            if not self.phase == 3:
                this_date = file_info_row['start'].date()
                shift_correction = self.lut_df.loc[this_date][self.lut_col]
            else:
                this_date = file_info_row['start']
                # self.lut_df['file_date'] = pd.to_datetime(self.lut_df['file_date'])

                # If time lags are calculated for segments shorter than the file they
                # are part of, then there are more than one found lag times per file.
                # Therefore, make list of all found time lags and calculate the median.
                # In case segment length is the same as file length and there is only
                # one lag time for this file, then the median has no effect.
                self.lut_df.index = pd.to_datetime(self.lut_df.index)
                shift_correction = self.lut_df.loc[self.lut_df.index == this_date, self.lut_col].to_list()
                shift_correction = np.median(shift_correction)

            if pd.isnull(shift_correction):
                shift_correction = np.nan
                txt_info += f"(!)No lag found in LUT for "
            else:
                # Read and prepare data file
                data_df = files.read_raw_data(filepath=file_info_row['filepath'],
                                              data_timestamp_format=self.dat_recs_timestamp_format)  # nrows for testing

                shift_correction = int(shift_correction)
                data_df = self.shift_var_target(df=data_df,
                                                shift=shift_correction)

                self.save_dyco_files(outdir=self.outdirs[f'{self.phase}-7_{self.phase_files}_normalized'],
                                     original_filename=file_info_row['filename'],
                                     df=data_df,
                                     export_timestamp=True)

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
        df.fillna(-9999, inplace=True)
        outpath = outdir / f"{original_filename}_DYCO.csv"
        df.to_csv(outpath, index=export_timestamp)

    def shift_var_target(self, df, shift):
        for col in self.var_target:
            outcol = f"{col}_DYCO"
            df[outcol] = df[col].shift(shift)  # Shift col by found lag
            df.drop([col], axis=1, inplace=True)  # Remove col that was not shifted
        return df

    def read_lut_time_lags(self):
        filepath = self.outdirs[
                       f'{self.phase}-6_{self.phase_files}_normalization_lookup_table'] / f'LUT_default_agg_time_lags.csv'
        parse = lambda x: dt.datetime.strptime(x, '%Y-%m-%d')
        df = pd.read_csv(filepath,
                         skiprows=None,
                         header=0,
                         # names=header_cols_list,
                         # na_values=-9999,
                         encoding='utf-8',
                         delimiter=',',
                         mangle_dupe_cols=True,
                         # keep_date_col=False,
                         parse_dates=True,
                         date_parser=parse,
                         index_col=0,
                         dtype=None,
                         engine='c')
        return df
