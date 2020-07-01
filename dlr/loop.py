import os

import numpy as np
import pandas as pd

import files
import lag
import plot


class Loop:
    def __init__(self, dat_recs_timestamp_format, dat_recs_nominal_timeres, iteration,
                 lgs_hist_remove_fringe_bins, lgs_hist_perc_thres, outdirs, lgs_segment_dur,
                 lgs_refsig, lgs_lagsig, lgs_num_iter, lgs_winsize, files_overview_df):
        self.dat_recs_timestamp_format = dat_recs_timestamp_format
        self.dat_recs_nominal_timeres = dat_recs_nominal_timeres
        self.iteration = iteration
        self.lgs_hist_remove_fringe_bins = lgs_hist_remove_fringe_bins
        self.lgs_hist_perc_thres = lgs_hist_perc_thres
        self.outdirs = outdirs
        self.lgs_segment_dur = lgs_segment_dur
        self.lgs_refsig = lgs_refsig
        self.lgs_lagsig = lgs_lagsig
        self.lgs_num_iter = lgs_num_iter
        self.lgs_winsize = lgs_winsize
        self.files_overview_df = files_overview_df

        self.run()

    def run(self):
        self.loop_files()

    # def get(self): todo
    #     return XXX

    def loop_files(self):
        """Loop through all found files."""

        prev_results = self.check_for_prev_results(iteration=self.iteration)
        if not prev_results:
            return None
        else:
            segment_lagtimes_df = prev_results[0]
            self.lgs_winsize = prev_results[1]

        shift_stepsize, hist_num_bins = self.search_settings(win_lagsearch=self.lgs_winsize)
        num_files = self.files_overview_df['file_available'].sum()
        # data_collection_df = pd.DataFrame() # todo activate

        # Loop files
        for file_idx, file_info_row in self.files_overview_df.iterrows():

            # Check file availability
            if file_info_row['file_available'] == 0:
                continue

            # Read and prepare data file
            data_df = files.read_raw_data(filepath=file_info_row['filepath'],
                                          data_timestamp_format=self.dat_recs_timestamp_format)  # nrows for testing

            # Insert timestamp if missing
            if self.dat_recs_timestamp_format:
                true_resolution = files.calc_true_resolution(num_records=len(data_df),
                                                             data_nominal_res=self.dat_recs_nominal_timeres,
                                                             expected_records=file_info_row['expected_records'],
                                                             expected_duration=file_info_row['expected_duration'])
            else:
                # todo test
                data_df, true_resolution = files.insert_timestamp(df=data_df,
                                                                  file_info_row=file_info_row,
                                                                  num_records=len(data_df),
                                                                  data_nominal_res=self.dat_recs_nominal_timeres,
                                                                  expected_records=file_info_row['expected_records'],
                                                                  expected_duration=file_info_row['expected_duration'])

            # Add raw data info to overview when files are first read (during iteration 1)
            if self.iteration == 1:
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
                                                     iteration=self.iteration,
                                                     win_lagsearch=self.lgs_winsize,
                                                     num_files=num_files,
                                                     shift_stepsize=shift_stepsize)

            # # Collect all file data # todo activate?
            # data_collection_df = self.collect_file_data(data_df=data_df,
            #                                             file_idx=file_idx,
            #                                             data_collection_df=data_collection_df)

            # segment_lagtimes_df = lag.calc_quantiles(df=segment_lagtimes_df)

        # Set new search window
        hist_series = self.filter_series(filter_col='iteration', filter_equal_to=self.iteration,
                                         df=segment_lagtimes_df, series_col='shift_peak_cov_abs_max')
        win_lagsearch_adj = lag.AdjustLagsearchWindow(series=hist_series,
                                                      outdir=self.outdirs['2-1_____Histograms'],
                                                      iteration=self.iteration,
                                                      plot=True,
                                                      remove_fringe_bins=self.lgs_hist_remove_fringe_bins,
                                                      hist_num_bins=hist_num_bins,
                                                      perc_threshold=self.lgs_hist_perc_thres).get()
        self.lgs_winsize = win_lagsearch_adj  # Update lag search window to new range

        # Add time window start and end for next iteration lagsearch
        self.add_lagsearch_adj_info(segment_lagtimes_df=segment_lagtimes_df,
                                    iteration=self.iteration,
                                    next_win_lagsearch=win_lagsearch_adj)

        # plot.lagsearch_collection(lagsearch_df_collection=lagsearch_df_collection,
        #                           outdir=self.outdirs['1-1_PLOTS_segment_covariance'])

        # Plot all found segment lag times
        plot.timeseries_segment_lagtimes(df=segment_lagtimes_df,
                                         outdir=self.outdirs['2-2_____Timeseries'],
                                         iteration=self.iteration)

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
        segment_grouped = data_df.groupby(pd.Grouper(key='index', freq=self.lgs_segment_dur))
        for segment_key, segment_df in segment_grouped:
            counter_segment += 1
            segment_name = f"{segment_df.index[0].strftime('%Y%m%d%H%M%S')}_iter{iteration}"
            segment_start = segment_df.index[0]
            segment_end = segment_df.index[-1]

            # print(f"\n-----Working on file {file_idx} of {num_files} ...")
            # print(f"-----------------Working on segment {split_name}, iteration {iteration} ...")
            print(f"[SEGMENT]    START: {segment_start}    END: {segment_end}    ITER: {iteration}    "
                  f"SOURCE FILE: {file_idx}")

            # # todo expand segment df
            # # Extend segment
            # this_segment_end = segment_df.index[-1]
            # segment_extension_end = this_segment_end + pd.to_timedelta(self.data_segment_overhang)
            # extension_range_df = data_df.loc[this_segment_end:segment_extension_end]
            # segment_df = segment_df.append(extension_range_df[1:])  # Remove 1st row, overlaps w/ segment
            # print(f"Adding segment overhang {self.data_segment_overhang}, new end time {segment_df.index[-1]} ... ")

            # Search lag
            lagsearch_df, props_peak_auto = \
                lag.LagSearch(segment_df=segment_df,
                              segment_name=segment_name,
                              segment_start=segment_start,
                              segment_end=segment_end,
                              ref_sig=self.lgs_refsig,
                              lagged_sig=self.lgs_lagsig,
                              outdir_plots=self.outdirs['1-1_____Plots'],
                              outdir_data=self.outdirs['1-0_Covariances'],
                              win_lagsearch=win_lagsearch,
                              file_idx=file_idx,
                              filename=filename,
                              iteration=iteration,
                              shift_stepsize=shift_stepsize).get()

            # Collect results
            ref_sig_numvals = segment_df[self.lgs_refsig].dropna().size
            lagged_sig_numvals = segment_df[self.lgs_lagsig].dropna().size

            segment_lagtimes_df.loc[segment_name, 'start'] = segment_start
            segment_lagtimes_df.loc[segment_name, 'end'] = segment_end
            segment_lagtimes_df.loc[segment_name, f'numvals_{self.lgs_refsig}'] = ref_sig_numvals
            segment_lagtimes_df.loc[segment_name, f'numvals_{self.lgs_lagsig}'] = lagged_sig_numvals
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
                self.outdirs['2-0_Segment_Lag_Times'] / filename_segments)

        return segment_lagtimes_df

    @staticmethod
    def filter_series(filter_col, filter_equal_to, df, series_col):
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

    def check_for_prev_results(self, iteration):
        """
        Check if previous results exist.



        Returns
        -------

        """
        filepath_this_iteration = self.outdirs['2-0_Segment_Lag_Times'] \
                                  / f'{iteration}_segments_found_lag_times_after_iteration-{iteration}.csv'
        filepath_prev_iteration = self.outdirs['2-0_Segment_Lag_Times'] \
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
                win_lagsearch = self.lgs_winsize

            return segment_lagtimes_df, win_lagsearch

    def add_lagsearch_adj_info(self, segment_lagtimes_df, iteration, next_win_lagsearch):
        # Add adjusted lagsearch window for next iteration to CSV
        segment_lagtimes_df.loc[:, 'lagsearch_next_start'] = next_win_lagsearch[0]
        segment_lagtimes_df.loc[:, 'lagsearch_next_end'] = next_win_lagsearch[1]

        # Save found segment lag times with next lagsearch info after all files
        filename_segments = f'{iteration}_segments_found_lag_times_after_iteration-{iteration}.csv'
        print(f"Saving file: {filename_segments} ...")
        segment_lagtimes_df.to_csv(
            self.outdirs['2-0_Segment_Lag_Times'] / filename_segments)
        return

    @staticmethod
    def search_settings(win_lagsearch):
        """
        Set step size of shift during lag search and calculate number of
        histogram bins.

        Parameters
        ----------
        win_lagsearch: list
            Contains start record for lag search in [0] and end record
            in [1], e.g. [-1000, 1000] searches between records -1000 and
            +1000.


        Returns
        -------
        shift_stepsize: int
            Step-size (number of records) for lag search.
        hist_num_bins: range
            Bin range for the histogram of found lag times. The histogram
            is used to narrow down win_lagsearch.

        Examples
        --------
        >> shift_stepsize, hist_num_bins = lagsearch_settings(win_lagsearch=-1000, 1000])
        >> print(shift_stepsize)
        10
        >> print(hist_num_bins)
        range(-1000, 1000, 50)

        """
        shift_stepsize = int(np.sum(np.abs(win_lagsearch)) / 100 / 2)
        hist_num_bins = range(int(win_lagsearch[0]), int(win_lagsearch[1]), int(shift_stepsize * 5))
        return shift_stepsize, hist_num_bins


class PlotLoopResults:
    def __init__(self, outdirs, lgs_num_iter, lgs_hist_perc_thres,
                 plot_cov_collection=True, plot_hist=True, plot_timeseries_segment_lagtimes=True):
        self.outdirs = outdirs
        self.lgs_num_iter = lgs_num_iter
        self.lgs_hist_perc_thres = lgs_hist_perc_thres
        self.plot_cov_collection = plot_cov_collection
        self.plot_hist = plot_hist
        self.plot_timeseries_segment_lagtimes = plot_timeseries_segment_lagtimes

    def run(self):
        # Covariance collection
        if self.plot_cov_collection:
            # Plot covariances from lag search for each segment in one plot
            plot.cov_collection(indir=self.outdirs['1-0_Covariances'],
                                outdir=self.outdirs['1-1_____Plots'])

        # Histogram
        if self.plot_hist:
            # Read found lag time results from very last iteration
            segment_lagtimes_df = files.read_segments_file(
                filepath=self.outdirs['2-0_Segment_Lag_Times']
                         / f'{self.lgs_num_iter}_segments_found_lag_times_after_iteration-{self.lgs_num_iter}.csv')

            # Set search window for lag, depending on iteration
            hist_series = Loop.filter_series(filter_col='iteration', filter_equal_to=self.lgs_num_iter,
                                             df=segment_lagtimes_df, series_col='shift_peak_cov_abs_max')
            last_win_lagsearch = [int(segment_lagtimes_df.iloc[-1]['lagsearch_start']),
                                  int(segment_lagtimes_df.iloc[-1]['lagsearch_end'])]
            _, hist_num_bins = Loop.search_settings(win_lagsearch=last_win_lagsearch)
            _ = lag.AdjustLagsearchWindow(series=hist_series,
                                          outdir=self.outdirs['2-1_____Histograms'],
                                          iteration=self.lgs_num_iter,
                                          plot=True,
                                          hist_num_bins=hist_num_bins,
                                          remove_fringe_bins=False,
                                          perc_threshold=self.lgs_hist_perc_thres).get()

        # Timeseries of lag times
        if self.plot_timeseries_segment_lagtimes:
            # Read found lag time results from very last iteration
            segment_lagtimes_df = files.read_segments_file(
                filepath=self.outdirs['2-0_Segment_Lag_Times']
                         / f'{self.lgs_num_iter}_segments_found_lag_times_after_iteration-{self.lgs_num_iter}.csv')
            plot.timeseries_segment_lagtimes(df=segment_lagtimes_df,
                                             outdir=self.outdirs['2-2_____Timeseries'],
                                             iteration=self.lgs_num_iter,
                                             show_all=True)
