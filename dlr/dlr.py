"""

DYNAMIC LAG REMOVER - DLR
-------------------------

DYLACO: Dynamic Lag Compensation
DLR: Dynamic Lag Remover - A Python package to detect and compensate for shifting lag times in ecosystem time series

File:       Data file
Segment:    Segment in file, e.g. one 6-hour file may consist of 12 half-hour segments.

"""

import pandas as pd

pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 1000)

import files
import plot
from lag import LagSearch
from wind import WindRotation
from files import FilesDetector


class DynamicLagRemover:
    # Dataframes for results collection
    files_overview_df = pd.DataFrame()
    file_stats_df = pd.DataFrame()
    segment_lagtimes_df = pd.DataFrame(columns=['start', 'end', 'cov_max_shift', 'cov_max', 'cov_max_timestamp'])

    def __init__(self, u_col, v_col, w_col, scalar_col, dir_input, dir_output, dir_root,
                 file_date_format, file_generation_res, data_res, data_segment_duration,
                 data_segment_overhang, data_timewin_lag, testing):
        self.u_col = u_col
        self.v_col = v_col
        self.w_col = w_col
        self.scalar_col = scalar_col
        self.scalar_col = scalar_col
        self.dir_root = dir_root
        self.dir_input = self.dir_root / dir_input
        self.dir_output = self.dir_root / dir_output
        self.file_date_format = file_date_format
        self.file_generation_res = file_generation_res
        self.data_res = data_res
        self.data_segment_duration = data_segment_duration
        self.data_segment_overhang = data_segment_overhang
        self.data_timewin_lag = data_timewin_lag
        self.testing = testing

        if self.testing:
            self.nrows = 10000
        else:
            self.nrows = None

        self.run()

    def run(self):

        # FILES DETECTOR
        # --------------
        fd = FilesDetector(dir_input=self.dir_input,
                           dir_output=self.dir_output,
                           file_pattern='*.csv',
                           file_date_format=self.file_date_format,
                           file_generation_res=self.file_generation_res,
                           data_res=self.data_res)
        fd.run()
        self.files_overview_df = fd.get()

        # LOOP THROUGH FILES
        # ------------------
        self.loop_files()

        # self.analyse_found_lags()

    def analyse_found_lags(self):
        found_lags_df = files.read_found_lags_file(filepath=self.dir_output / '2_segments_found_lag_times.csv')
        plot.results(df=found_lags_df)
        plot.plot_results_hist(df=found_lags_df, dir_output=self.dir_output)

    def check_if_file_exists(self, file_idx, filepath):
        print(f"\n-------------------\n{file_idx}")
        file_exists = True

        # The last listed file does not exist, only used for timestamp endpoint.
        if file_idx == self.files_overview_df.index[-1]:
            print(f"(!) Last listed file does not exist ({filepath})")
            file_exists = False
            return file_exists

        # Found files have a filepath, expected but missing files not.
        if filepath:
            print(f"Data found in {filepath}")
            file_exists = True
        else:
            print(f"(!) No data found ({filepath})")
            file_exists = False

        return file_exists

    def loop_files(self):
        """Loop through all found files."""

        data_collection_df = pd.DataFrame()

        for file_idx, file_info_row in self.files_overview_df.iterrows():
            # if file_idx > 0:  # for testing
            #     break

            # Check file availability
            file_exists = self.check_if_file_exists(file_idx=file_idx,
                                                    filepath=file_info_row['filepath'])
            if not file_exists:
                continue

            # Read data file
            data_df, true_resolution = files.read_raw_data(filepath=file_info_row['filepath'],
                                                           nrows=self.nrows,
                                                           df_start_dt=file_info_row['start'],
                                                           file_info_row=file_info_row)  # nrows for testing

            # Add raw data info to overview
            self.files_overview_df = files.add_data_stats(df=data_df,
                                                          true_resolution=true_resolution,
                                                          filename=file_info_row['filename'],
                                                          files_overview_df=self.files_overview_df,
                                                          found_records=len(data_df))

            # Collect all file data
            data_collection_df = self.collect_file_data(data_df=data_df,
                                                        file_idx=file_idx,
                                                        data_collection_df=data_collection_df)

            # Loop through data segments in file
            self.loop_segments(data_df=data_df,
                               file_idx=file_idx,
                               filename=file_info_row['filename'])

        # # Save file stats
        # self.file_stats_df.insert(0, 'filename', self.file_stats_df.index)  # inplace
        # self.file_stats_df.to_csv(self.dir_output / '1_file_stats.csv', index=False)


        # Save and plot found lag times
        self.segment_lagtimes_df['cov_max_shift'] = self.segment_lagtimes_df['cov_max_shift'].astype(int)
        self.segment_lagtimes_df['cov_max'] = self.segment_lagtimes_df['cov_max'].astype(float)
        self.segment_lagtimes_df['cov_max_timestamp'] = self.segment_lagtimes_df['cov_max_timestamp'].astype(str)
        self.segment_lagtimes_df.to_csv(self.dir_output / '2_segments_and_lag_times.csv')
        plot.segment_lagtimes(df=self.segment_lagtimes_df, dir_output=self.dir_output)

    def collect_file_data(self, data_df, file_idx, data_collection_df):
        if file_idx == self.files_overview_df.index[0]:
            data_collection_df = data_df.copy()
        else:
            data_collection_df = data_collection_df.append(data_df)
        return data_collection_df

    def loop_segments(self, data_df, file_idx, filename):
        """Loop through all data segments in each file.

        For example, one six hour raw data file consists of twelve half-hour segments.

        :param data_df: All data in the data file
        :type data_df: pandas DataFrame
        :return:
        """
        counter_segment = -1
        data_df['index'] = pd.to_datetime(data_df.index)
        segment_grouped = data_df.groupby(pd.Grouper(key='index', freq=self.data_segment_duration))
        for segment_key, segment_df in segment_grouped:
            counter_segment += 1
            # if counter_segment > 0:  # for testing
            #     break
            segment_name = segment_df.index[0].strftime('%Y%m%d%H%M%S')
            segment_start = segment_df.index[0]
            segment_end = segment_df.index[-1]

            print(f"\n-----------------Working on segment {segment_name} ...")
            print(f"Start: {segment_start}    End: {segment_end}")

            # todo expand segment df
            # Extend segment
            this_segment_end = segment_df.index[-1]
            segment_extension_end = this_segment_end + pd.to_timedelta(self.data_segment_overhang)
            extension_range_df = data_df.loc[this_segment_end:segment_extension_end]
            segment_df = segment_df.append(extension_range_df[1:])  # Remove 1st row, overlaps w/ segment

            wind_rot_df, w_rot_turb_col, scalar_turb_col = \
                WindRotation(wind_df=segment_df[[self.u_col, self.v_col, self.w_col, self.scalar_col]],
                             u_col=self.u_col, v_col=self.v_col,
                             w_col=self.w_col, scalar_col=self.scalar_col).get()

            cov_max_shift, cov_max, cov_max_timestamp = \
                LagSearch(wind_rot_df=wind_rot_df,
                          segment_name=segment_name,
                          segment_start=segment_start,
                          segment_end=segment_end,
                          ref_sig=w_rot_turb_col,
                          lagged_sig=scalar_turb_col,
                          dir_out=self.dir_output,
                          data_timewin_lag=self.data_timewin_lag,
                          file_idx=file_idx,
                          filename=filename).get()

            # Collect results
            self.segment_lagtimes_df.loc[segment_name, 'start'] = segment_start
            self.segment_lagtimes_df.loc[segment_name, 'end'] = segment_end
            self.segment_lagtimes_df.loc[segment_name, 'cov_max_shift'] = int(cov_max_shift)  # todo give in sec
            self.segment_lagtimes_df.loc[segment_name, 'cov_max'] = float(cov_max)
            self.segment_lagtimes_df.loc[segment_name, 'cov_max_timestamp'] = str(cov_max_timestamp)

# if __name__ == "__main__":
#     DynamicLagRemover()
