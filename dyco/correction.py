import datetime as dt
import time

import numpy as np
import pandas as pd

import files
from _setup import create_logger


class NormalizeLags:
    def __init__(self, dyco_instance):
        self.files_overview_df = dyco_instance.files_overview_df
        self.dat_recs_timestamp_format = dyco_instance.dat_recs_timestamp_format
        self.outdirs = dyco_instance.outdirs
        self.normalize_lag_for_cols = dyco_instance.normalize_lag_for_cols

        self.logger = create_logger(logfile_path=dyco_instance.logfile_path, name=__name__)

        self.lut_default_lag_times_df = self.read_lut()

        self.run()

    def run(self):
        # Loop files
        num_files = self.files_overview_df['file_available'].sum()
        times_needed = []
        files_counter = 0
        for file_idx, file_info_row in self.files_overview_df.iterrows():

            # Check file availability
            if file_info_row['file_available'] == 0:
                continue

            start = time.time()

            # Read and prepare data file
            data_df = files.read_raw_data(filepath=file_info_row['filepath'],
                                          data_timestamp_format=self.dat_recs_timestamp_format)  # nrows for testing

            this_date = file_info_row['start'].date()


            shift_correction = self.lut_default_lag_times_df.loc[this_date]['correction']
            if pd.isnull(shift_correction):
                shift_correction = np.nan
            else:
                shift_correction = int(shift_correction)
                data_df = self.normalize_default_lag(df=data_df,
                                                     shift=shift_correction)

                self.save_dyco_files(outdir=self.outdirs['1-7_input_files_normalized'],
                                     original_filename=file_info_row['filename'],
                                     df=data_df,
                                     export_timestamp=True)

            time_needed = time.time() - start
            times_needed.append(time_needed)
            files_counter += 1
            times_needed_mean = np.mean(times_needed)
            remaining_files = num_files - files_counter
            remaining_sec = times_needed_mean * remaining_files
            # remaining_hours = int(remaining_sec / 60 / 60)
            # remaining_mins = int(remaining_sec % 60 % 60)

            progress = (files_counter / num_files) * 100

            txt_info = f"File #{files_counter}: {file_info_row['filename']}" \
                       f"    shift correction: {shift_correction}    remaining time: {remaining_sec:.0f}s" \
                       f"    remaining files: {int(remaining_files)}    progress: {progress:.2f}%"
            self.logger.info(txt_info)




    def save_dyco_files(self, df, outdir, original_filename, export_timestamp):
        df.fillna(-9999, inplace=True)
        outpath = outdir / f"{original_filename}_DYCO.csv"
        df.to_csv(outpath, index=export_timestamp)

    def normalize_default_lag(self, df, shift):
        for col in self.normalize_lag_for_cols:
            outcol = f"{col}_DYCO"
            df[outcol] = df[col].shift(shift)
        return df

    def read_lut(self):
        filepath = self.outdirs['1-6_input_files_normalization_lookup_table'] / f'LUT_default_lag_times.csv'
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
