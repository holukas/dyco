"""

DYNAMIC LAG REMOVER - DLR
-------------------------
A Python package to detect and compensate for shifting lag times in ecosystem time series



File:       Data file
Segment:    Segment in file, e.g. one 6-hour file may consist of 12 half-hour segments.

Step (1)
FilesDetector (files.FilesDetector)

"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 1000)

import files
import lag
from wind import WindRotation
from files import FilesDetector
import plot


class DynamicLagRemover:
    # Dataframes for results collection
    files_overview_df = pd.DataFrame()

    def __init__(self, u_col, v_col, w_col, scalar_col, dir_input, dir_output, dir_root, file_date_format,
                 file_generation_res, data_nominal_res, data_segment_duration, data_segment_overhang, win_lagsearch,
                 num_iterations, hist_remove_fringe_bins, del_previous_results, hist_perc_threshold, dir_input_ext,
                 file_pattern, dir_output_ext, files_how_many):
        self.u_col = u_col
        self.v_col = v_col
        self.w_col = w_col
        self.scalar_col = scalar_col
        self.scalar_col = scalar_col
        self.dir_root = dir_root
        self.dir_input = self.dir_root / dir_input
        self.dir_input_ext = dir_input_ext
        self.dir_output = self.dir_root / dir_output
        self.dir_output_ext = dir_output_ext
        self.file_date_format = file_date_format
        self.file_generation_res = file_generation_res
        self.data_nominal_res = data_nominal_res
        self.data_segment_duration = data_segment_duration
        self.data_segment_overhang = data_segment_overhang
        self.win_lagsearch = win_lagsearch
        self.num_iterations = num_iterations
        self.hist_remove_fringe_bins = hist_remove_fringe_bins
        self.del_previous_results = del_previous_results
        self.hist_perc_threshold = hist_perc_threshold
        self.file_pattern = file_pattern
        self.files_how_many = files_how_many

        self.set_ext_dirs()
        self.run()

    def run(self):

        # Setup
        self.outdirs = files.setup_output_dirs(outdir=self.dir_output,
                                               del_previous_results=self.del_previous_results)

        # Search files
        self.files_overview_df = self.search_files()

        # Loop through files and their segments
        for iteration in range(1, 1 + self.num_iterations):
            self.loop_files(iteration=iteration)

        # FINALIZE LAG SEARCH RESULTS
        # ---------------------------
        # Plot covariances from lag search for each segment #todo activate
        plot.cov_collection(indir=self.outdirs['1-0___[DATA]__Segment_Covariances'],
                            outdir=self.outdirs['1-1___[PLOTS]_Segment_Covariances'])

        # Read found lag time results from very last iteration
        segment_lagtimes_df = files.read_segments_file(
            filepath=self.outdirs['2-0___[DATA]__Segment_Lag_Times']
                     / f'{self.num_iterations}_segments_found_lag_times_after_iteration-{self.num_iterations}.csv')

        last_win_lagsearch = [int(segment_lagtimes_df.iloc[-1]['lagsearch_start']),
                              int(segment_lagtimes_df.iloc[-1]['lagsearch_end'])]

        # Set search window for lag, depending on iteration #todo activate
        hist_series = self.filter_series(filter_col='iteration', filter_equal_to=self.num_iterations,
                                         df=segment_lagtimes_df, series_col='shift_peak_cov_abs_max')
        _, hist_num_bins = lag.search_settings(win_lagsearch=last_win_lagsearch)
        _ = lag.AdjustLagsearchWindow(series=hist_series,
                                      outdir=self.outdirs['2-1___[PLOTS]_Segment_Lag_Times_Histograms'],
                                      iteration=self.num_iterations,
                                      plot=True,
                                      hist_num_bins=hist_num_bins,
                                      remove_fringe_bins=False,
                                      perc_threshold=self.hist_perc_threshold).get()

        plot.timeseries_segment_lagtimes(df=segment_lagtimes_df,
                                         outdir=self.outdirs['2-2___[PLOTS]_Segment_Lag_Times_Timeseries'],
                                         iteration=self.num_iterations,
                                         show_all=True)

        # ANALYSE LAGS
        # ------------
        # self.analyse_found_lags()

    def search_files(self):
        # SEARCH FILES
        # ------------
        fd = FilesDetector(dir_input=self.dir_input,
                           outdir=self.outdirs['0-0___[DATA]__Found_Files'],
                           file_pattern=self.file_pattern,
                           file_date_format=self.file_date_format,
                           file_generation_res=self.file_generation_res,
                           data_res=self.data_nominal_res)
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

    def analyse_found_lags(self):
        last_iteration = self.num_iterations
        filepath_last_iteration = self.outdirs['2-0___[DATA]__Segment_Lag_Times'] \
                                  / f'{last_iteration}_segments_found_lag_times_after_iteration-{last_iteration}.csv'
        segment_lagtimes_df = files.read_segments_file(filepath=filepath_last_iteration)
        _df = self.filter_dataframe(filter_col='iteration', filter_equal_to=last_iteration,
                                    df=segment_lagtimes_df)

        _df.set_index('start', inplace=True)
        _df.index = pd.to_datetime(_df.index)
        # _df = _df.resample('1D').median()
        # analyse_df = pd.DataFrame(index=_df.index,
        #                           columns=['median', 'q25', 'q75'])
        # analyse_df['median'] = _df.resample('1D').median()
        # analyse_df['shift_q25'] = np.nan
        # analyse_df['shift_q75'] = np.nan
        # analyse_df=analyse_df.resample('1D').median()
        # analyse_df['shift_P25'] = _df['cov_max_shift'].quantile(0.25)
        # analyse_df.index = _df.index.date

        # Median
        df = pd.DataFrame()
        grouped = _df.groupby(_df.index.date)
        for idx, grouped_df in grouped:
            df.loc[idx, 'median'] = grouped_df['cov_max_shift'].median()
            df.loc[idx, 'q20'] = grouped_df['cov_max_shift'].quantile(0.20)
            df.loc[idx, 'q80'] = grouped_df['cov_max_shift'].quantile(0.80)

        df['qrange'] = df['q20'] - df['q80']
        df['qrange'] = df['qrange'].abs()

        df['rm_median'] = df['median'].rolling(window=7, min_periods=4, center=True).median()
        df['rm_q25'] = df['q25'].rolling(window=7, min_periods=4, center=True).median()
        df['rm_q75'] = df['q75'].rolling(window=7, min_periods=4, center=True).median()

        df[['rm_median', 'rm_q25', 'rm_q75']].plot()
        plt.show()
        print("x")

        # filter_daily = _df.index.date == idx

        # # Median
        # grouped = _df.groupby(_df.index.date)
        # for idx, grouped_df in grouped:
        #     filter_daily = _df.index.date == idx
        #     analyse_df.loc[filter_daily, 'shift_median'] = grouped_df['cov_max_shift'].median()
        #     analyse_df.loc[filter_daily, 'shift_P25'] = grouped_df['cov_max_shift'].quantile(0.25)
        #     analyse_df.loc[filter_daily, 'shift_P75'] = grouped_df['cov_max_shift'].quantile(0.75)

        _df['shift_median'].rolling(window=200, min_periods=1, center=True).mean().plot()
        plt.show()

        quantiles_df = lag.calc_quantiles(df=_df).copy()
        plot.results(df=quantiles_df)

        # self.find_default(df=found_lags_df)
        # plot.results(df=found_lags_df)
        # plot.plot_results_hist(df=found_lags_df, dir_output=self.dir_output)

    def find_default(self, df):
        plot_df = df[['cov_max_shift']].copy()

        for b in range(1, 4, 1):
            bins = 2
            plot_df['group'] = pd.cut(plot_df['cov_max_shift'],
                                      bins=bins, retbins=False,
                                      duplicates='drop', labels=False)
            plot_df_agg = plot_df.groupby('group').agg(['count', 'min', 'max'])
            idxmax = plot_df_agg['cov_max_shift']['count'].idxmax()  # Most counts
            group_max = plot_df_agg.iloc[idxmax].name

            plot_df_agg['count_maxperc'] = \
                plot_df_agg['cov_max_shift']['count'] / plot_df_agg['cov_max_shift']['count'].sum()
            # plot_df_agg['cov_max_shift']['count'] / plot_df_agg.iloc[idxmax]['cov_max_shift']['count']

            plot_df = plot_df.loc[plot_df['group'] == group_max]

        median = plot_df['cov_max_shift'].median()
        _min = plot_df['cov_max_shift'].min()
        _max = plot_df['cov_max_shift'].max()

        print(plot_df)
        print(f"{median}  {_min}  {_max}")

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

    def check_for_prev_results(self, iteration):
        """
        Check if previous results exist.



        Returns
        -------

        """
        filepath_this_iteration = self.outdirs['2-0___[DATA]__Segment_Lag_Times'] \
                                  / f'{iteration}_segments_found_lag_times_after_iteration-{iteration}.csv'
        filepath_prev_iteration = self.outdirs['2-0___[DATA]__Segment_Lag_Times'] \
                                  / f'{iteration - 1}_segments_found_lag_times_after_iteration-{iteration - 1}.csv'
        if os.path.exists(filepath_this_iteration):
            print(f"(!) Results for iteration {iteration} already exist in file {filepath_this_iteration},"
                  f"    skipping to next iteration.")
            return None
        else:
            if os.path.exists(filepath_prev_iteration):
                segment_lagtimes_df = files.read_segments_file(filepath=filepath_prev_iteration)

                # Adjusted lagsearch window, from a previous iteration
                win_lagsearch = \
                    [segment_lagtimes_df['lagsearch_next_start'].unique()[0],
                     segment_lagtimes_df['lagsearch_next_end'].unique()[0]]
            else:
                segment_lagtimes_df = pd.DataFrame()
                # segment_lagtimes_df = pd.DataFrame(columns=['start', 'end', 'cov_max_shift', 'cov_max',
                #                                             'cov_max_timestamp', 'shift_median', 'shift_P25',
                #                                             'shift_P75'])
                # Initial lagsearch window
                win_lagsearch = self.win_lagsearch

            return segment_lagtimes_df, win_lagsearch

    def loop_files(self, iteration):
        """Loop through all found files."""

        prev_results = self.check_for_prev_results(iteration=iteration)
        if not prev_results:
            return None
        else:
            segment_lagtimes_df = prev_results[0]
            self.win_lagsearch = prev_results[1]

        shift_stepsize, hist_num_bins = lag.search_settings(win_lagsearch=self.win_lagsearch)
        num_files = self.files_overview_df['file_available'].sum()
        # data_collection_df = pd.DataFrame() # todo activate

        # Loop files
        for file_idx, file_info_row in self.files_overview_df.iterrows():

            # Check file availability
            if file_info_row['file_available'] == 0:
                continue

            # Read and prepare data file
            data_df = files.read_raw_data(filepath=file_info_row['filepath'],
                                          nrows=None,
                                          df_start_dt=file_info_row['start'],
                                          file_info_row=file_info_row)  # nrows for testing
            data_df, true_resolution = files.insert_datetime_index(df=data_df,
                                                                   file_info_row=file_info_row,
                                                                   data_nominal_res=self.data_nominal_res)

            # Add raw data info to overview when files are first read (during iteration 1)
            if iteration == 1:
                self.files_overview_df = files.add_data_stats(df=data_df,
                                                              true_resolution=true_resolution,
                                                              filename=file_info_row['filename'],
                                                              files_overview_df=self.files_overview_df,
                                                              found_records=len(data_df))

            # Loop through data segments in file
            segment_lagtimes_df = self.loop_segments(data_df=data_df,
                                                     file_idx=file_idx,
                                                     filename=file_info_row['filename'],
                                                     segment_lagtimes_df=segment_lagtimes_df,
                                                     iteration=iteration,
                                                     win_lagsearch=self.win_lagsearch,
                                                     num_files=num_files,
                                                     shift_stepsize=shift_stepsize)

            # # Collect all file data # todo activate?
            # data_collection_df = self.collect_file_data(data_df=data_df,
            #                                             file_idx=file_idx,
            #                                             data_collection_df=data_collection_df)

            # segment_lagtimes_df = lag.calc_quantiles(df=segment_lagtimes_df)

        # Set new search window
        hist_series = self.filter_series(filter_col='iteration', filter_equal_to=iteration,
                                         df=segment_lagtimes_df, series_col='shift_peak_cov_abs_max')
        win_lagsearch_adj = lag.AdjustLagsearchWindow(series=hist_series,
                                                      outdir=self.outdirs['2-1___[PLOTS]_Segment_Lag_Times_Histograms'],
                                                      iteration=iteration,
                                                      plot=True,
                                                      remove_fringe_bins=self.hist_remove_fringe_bins,
                                                      hist_num_bins=hist_num_bins,
                                                      perc_threshold=self.hist_perc_threshold).get()
        self.win_lagsearch = win_lagsearch_adj  # Update lag search window to new range

        # Add time window start and end for next iteration lagsearch
        self.add_lagsearch_adj_info(segment_lagtimes_df=segment_lagtimes_df,
                                    iteration=iteration,
                                    next_win_lagsearch=win_lagsearch_adj)

        # plot.lagsearch_collection(lagsearch_df_collection=lagsearch_df_collection,
        #                           outdir=self.outdirs['1-1_PLOTS_segment_covariance'])

        # Plot all found segment lag times
        plot.timeseries_segment_lagtimes(df=segment_lagtimes_df,
                                         outdir=self.outdirs['2-2___[PLOTS]_Segment_Lag_Times_Timeseries'],
                                         iteration=iteration)

    def filter_series(self, filter_col, filter_equal_to, df, series_col):
        """
        Filter series based on values in another column.

        Parameters
        ----------
        filter_col: str
            Name of column that is used for filtering.
        filter_equal_to: int
            Value that must be contained in filter_col to keep value.
        df: pandas DataFrame
        series_col: str
            Name of column that is filtered.

        Returns
        -------
        pandas Series

        """
        filter_this_iteration = df[filter_col] == filter_equal_to
        series_filtered = df.loc[filter_this_iteration, series_col]
        return series_filtered

    def filter_dataframe(self, filter_col, filter_equal_to, df):
        filter_this_iteration = df[filter_col] == filter_equal_to
        df_filtered = df.loc[filter_this_iteration, :]
        return df_filtered

    def add_lagsearch_adj_info(self, segment_lagtimes_df, iteration, next_win_lagsearch):
        # Add adjusted lagsearch window for next iteration to CSV
        segment_lagtimes_df.loc[:, 'lagsearch_next_start'] = next_win_lagsearch[0]
        segment_lagtimes_df.loc[:, 'lagsearch_next_end'] = next_win_lagsearch[1]

        # Save found segment lag times with next lagsearch info after all files
        filename_segments = f'{iteration}_segments_found_lag_times_after_iteration-{iteration}.csv'
        print(f"Saving file: {filename_segments} ...")
        segment_lagtimes_df.to_csv(
            self.outdirs['2-0___[DATA]__Segment_Lag_Times'] / filename_segments)
        return

    def collect_file_data(self, data_df, file_idx, data_collection_df):
        if file_idx == self.files_overview_df.index[0]:
            data_collection_df = data_df.copy()
        else:
            data_collection_df = data_collection_df.append(data_df)
        return data_collection_df

    def loop_segments(self, data_df, file_idx, filename, segment_lagtimes_df, iteration, win_lagsearch,
                      num_files, shift_stepsize):
        """
        Loop through all data segments in each file.

        For example, one six hour raw data file consists of twelve half-hour segments.

        Parameters
        ----------
        data_df
        file_idx
        filename
        segment_lagtimes_df
        iteration
        win_lagsearch
        num_files

        Returns
        -------

        """

        counter_segment = -1
        data_df['index'] = pd.to_datetime(data_df.index)

        # Loop segments
        segment_grouped = data_df.groupby(pd.Grouper(key='index', freq=self.data_segment_duration))
        for segment_key, segment_df in segment_grouped:
            counter_segment += 1
            # if counter_segment > 0:  # for testing
            #     break
            segment_name = f"{segment_df.index[0].strftime('%Y%m%d%H%M%S')}_iter{iteration}"
            segment_start = segment_df.index[0]
            segment_end = segment_df.index[-1]

            print(f"\n-----Working on file {file_idx} of {num_files} ...")
            print(f"-----------------Working on segment {segment_name}, iteration {iteration} ...")
            print(f"Start: {segment_start}    End: {segment_end}")

            # todo expand segment df
            # Extend segment
            this_segment_end = segment_df.index[-1]
            segment_extension_end = this_segment_end + pd.to_timedelta(self.data_segment_overhang)
            extension_range_df = data_df.loc[this_segment_end:segment_extension_end]
            segment_df = segment_df.append(extension_range_df[1:])  # Remove 1st row, overlaps w/ segment
            print(f"Adding segment overhang {self.data_segment_overhang}, new end time {segment_df.index[-1]} ... ")

            # Rotate wind
            wind_rot_df, w_rot_turb_col, scalar_turb_col = \
                WindRotation(wind_df=segment_df[[self.u_col, self.v_col, self.w_col, self.scalar_col]],
                             u_col=self.u_col, v_col=self.v_col,
                             w_col=self.w_col, scalar_col=self.scalar_col).get()

            # Search lag

            lagsearch_df, props_peak_auto = \
                lag.LagSearch(wind_rot_df=wind_rot_df,
                              segment_name=segment_name,
                              segment_start=segment_start,
                              segment_end=segment_end,
                              ref_sig=w_rot_turb_col,
                              lagged_sig=scalar_turb_col,
                              outdir_plots=self.outdirs['1-1___[PLOTS]_Segment_Covariances'],
                              outdir_data=self.outdirs['1-0___[DATA]__Segment_Covariances'],
                              win_lagsearch=win_lagsearch,
                              file_idx=file_idx,
                              filename=filename,
                              iteration=iteration,
                              shift_stepsize=shift_stepsize).get()

            # Collect results
            ref_sig_numvals = wind_rot_df[w_rot_turb_col].dropna().size
            lagged_sig_numvals = wind_rot_df[scalar_turb_col].dropna().size

            segment_lagtimes_df.loc[segment_name, 'start'] = segment_start
            segment_lagtimes_df.loc[segment_name, 'end'] = segment_end
            segment_lagtimes_df.loc[segment_name, f'numvals_{w_rot_turb_col}'] = ref_sig_numvals
            segment_lagtimes_df.loc[segment_name, f'numvals_{scalar_turb_col}'] = lagged_sig_numvals
            segment_lagtimes_df.loc[segment_name, f'lagsearch_start'] = win_lagsearch[0]
            segment_lagtimes_df.loc[segment_name, f'lagsearch_end'] = win_lagsearch[1]
            segment_lagtimes_df.loc[segment_name, f'iteration'] = iteration

            segment_lagtimes_df.loc[segment_name, 'shift_peak_cov_abs_max'] = np.nan
            segment_lagtimes_df.loc[segment_name, 'cov_peak_cov_abs_max'] = np.nan
            segment_lagtimes_df.loc[segment_name, 'timestamp_peak_cov_abs_max'] = np.nan
            segment_lagtimes_df.loc[segment_name, 'shift_peak_auto'] = np.nan
            segment_lagtimes_df.loc[segment_name, 'cov_peak_auto'] = np.nan
            segment_lagtimes_df.loc[segment_name, 'timestamp_peak_auto'] = np.nan

            idx_peak_cov_abs_max, idx_peak_auto = \
                lag.LagSearch.get_peak_idx(df=lagsearch_df)

            # if (ref_sig_vals == 0) or (lagged_sig_vals == 0):
            #     segment_lagtimes_df.loc[segment_name, 'cov_max_shift'] = np.nan
            #     segment_lagtimes_df.loc[segment_name, 'cov_max'] = np.nan
            #     segment_lagtimes_df.loc[segment_name, 'cov_max_timestamp'] = np.nan
            # else:
            if idx_peak_cov_abs_max:
                segment_lagtimes_df.loc[segment_name, 'shift_peak_cov_abs_max'] = \
                    lagsearch_df.iloc[idx_peak_cov_abs_max]['shift']  # todo give in sec
                segment_lagtimes_df.loc[segment_name, 'cov_peak_cov_abs_max'] = \
                    lagsearch_df.iloc[idx_peak_cov_abs_max]['cov']
                segment_lagtimes_df.loc[segment_name, 'timestamp_peak_cov_abs_max'] = \
                    lagsearch_df.iloc[idx_peak_cov_abs_max]['index']

            if idx_peak_auto:
                segment_lagtimes_df.loc[segment_name, 'shift_peak_auto'] = \
                    lagsearch_df.iloc[idx_peak_auto]['shift']
                segment_lagtimes_df.loc[segment_name, 'cov_peak_auto'] = \
                    lagsearch_df.iloc[idx_peak_auto]['cov']
                segment_lagtimes_df.loc[segment_name, 'timestamp_peak_auto'] = \
                    lagsearch_df.iloc[idx_peak_auto]['index']

            # Save found segment lag times after each segment
            filename_segments = f'{iteration}_segments_found_lag_times_after_iteration-{iteration}.csv'
            print(f"Saving file: {filename_segments} ...")
            segment_lagtimes_df.to_csv(
                self.outdirs['2-0___[DATA]__Segment_Lag_Times'] / filename_segments)

        return segment_lagtimes_df

    def set_ext_dirs(self):
        # In case external dir was selected for input, set it as source dir.
        if self.dir_input_ext:
            self.dir_input = self.dir_input_ext  # Override
        if self.dir_output_ext:
            self.dir_output = self.dir_output_ext  # Override

# if __name__ == "__main__":
#     DynamicLagRemover()
