import os
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import files
import lag
import plot
from _setup import create_logger


class Loop:
    """
    Loop through all files and their segments
    """

    def __init__(self,
                 dyla_instance,
                 iteration: int = 1):
        """

        Parameters
        ----------
        dyla_instance:

        iteration: int
            Number of current iteration.
        """
        self.dat_recs_timestamp_format = dyla_instance.dat_recs_timestamp_format
        self.dat_recs_nominal_timeres = dyla_instance.dat_recs_nominal_timeres
        self.lgs_hist_remove_fringe_bins = dyla_instance.lgs_hist_remove_fringe_bins
        self.lgs_hist_perc_thres = dyla_instance.lgs_hist_perc_thres
        self.outdirs = dyla_instance.outdirs
        self.lgs_segment_dur = dyla_instance.lgs_segment_dur
        self.lgs_refsig = dyla_instance.lgs_refsig
        self.lgs_lagsig = dyla_instance.lgs_lagsig
        self.lgs_num_iter = dyla_instance.lgs_num_iter
        self.files_overview_df = dyla_instance.files_overview_df
        self.logfile_path = dyla_instance.logfile_path

        self.lgs_winsize = dyla_instance.lgs_winsize
        # force_min_stepsize = True if self.iteration == self.lgs_num_iter else False
        self.shift_stepsize, \
        self.hist_bin_range = self.lagsearch_settings(win_lagsearch=self.lgs_winsize,
                                                      force_min_stepsize=False)

        self.iteration = iteration

        self.logger = create_logger(logfile_path=self.logfile_path, name=__name__)

    def run(self):
        """Loop through all found files"""

        self.logger.info(f"Start FILE LOOP - ITERATION {self.iteration} {'-' * 40}")

        # Use results from previous iteration, if available
        prev_results = self.check_for_prev_results(iteration=self.iteration)
        if not prev_results:
            return None
        else:
            segment_lagtimes_df = prev_results[0]
            self.lgs_winsize = prev_results[1]

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
                                                     segment_lagtimes_df=segment_lagtimes_df)

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
                                                      hist_num_bins=self.hist_bin_range,
                                                      perc_threshold=self.lgs_hist_perc_thres).get()
        self.lgs_winsize = win_lagsearch_adj  # Update lag search window to new range

        # Add time window start and end for next iteration lagsearch
        self.add_lagsearch_adj_info(segment_lagtimes_df=segment_lagtimes_df,
                                    iteration=self.iteration,
                                    next_win_lagsearch=win_lagsearch_adj)

        # plot.lagsearch_collection(lagsearch_df_collection=lagsearch_df_collection,
        #                           outdir=self.outdirs['1-1_PLOTS_segment_covariance'])

        # Plot all found segment lag times
        plot.timeseries_segment_lagtimes(segment_lagtimes_df=segment_lagtimes_df,
                                         outdir=self.outdirs['2-2_____Timeseries'],
                                         iteration=self.iteration)

    def get(self):
        return self.lgs_winsize

    def loop_segments(self,
                      data_df: pd.DataFrame,
                      file_idx: pd.Timestamp,
                      filename: str,
                      segment_lagtimes_df: pd.DataFrame):
        """
        Loop through all data segments in each file

        For example, one six hour raw data file consists of twelve half-hour segments.

        Parameters
        ----------
        data_df: pandas DataFrame
            Contains file data.
        file_idx: pandas Timestamp
            Date and time of the current file.
        filename: str
            The file name of the file that contains the current segment.
        segment_lagtimes_df: pandas DataFrame
            Contains lag time info for each segment. Needed as attribute to
            add results of each iteration to results from all previous iterations.
            DataFrame is empty during iteration 1 and filled with the first results
            after iteration 1 has finished.

        Returns
        -------
        pandas DataFrame with lag time for each segment.

        """
        counter_segment = -1
        data_df['index'] = pd.to_datetime(data_df.index)

        # Loop segments
        segment_grouped = data_df.groupby(pd.Grouper(key='index', freq=self.lgs_segment_dur))
        for segment_key, segment_df in segment_grouped:

            counter_segment += 1
            segment_name = f"{segment_df.index[0].strftime('%Y%m%d%H%M%S')}_iter{self.iteration}"
            segment_start = segment_df.index[0]
            segment_end = segment_df.index[-1]
            self.logger.info(f"    Working on FILE: {filename}    SEGMENT: {segment_name}    "
                             f"Lag search window: {self.lgs_winsize}    Step-size: {self.shift_stepsize}    "
                             f"segment start: {segment_start}    segment end: {segment_end}")

            # # todo expand segment df
            # # Extend segment
            # this_segment_end = segment_df.index[-1]
            # segment_extension_end = this_segment_end + pd.to_timedelta(self.data_segment_overhang)
            # extension_range_df = data_df.loc[this_segment_end:segment_extension_end]
            # segment_df = segment_df.append(extension_range_df[1:])  # Remove 1st row, overlaps w/ segment
            # print(f"Adding segment overhang {self.data_segment_overhang}, new end time {segment_df.index[-1]} ... ")

            # Search lag
            lagsearch_df, props_peak_auto = \
                lag.LagSearch(loop_instance=self,
                              segment_df=segment_df,
                              segment_name=segment_name,
                              segment_start=segment_start,
                              segment_end=segment_end,
                              outdir_plots=self.outdirs['1-1_____Plots'],
                              outdir_data=self.outdirs['1-0_Covariances'],
                              file_idx=file_idx,
                              filename=filename).get()

            # Collect results
            ref_sig_numvals = segment_df[self.lgs_refsig].dropna().size
            lagged_sig_numvals = segment_df[self.lgs_lagsig].dropna().size

            segment_lagtimes_df.loc[segment_name, 'start'] = segment_start
            segment_lagtimes_df.loc[segment_name, 'end'] = segment_end
            segment_lagtimes_df.loc[segment_name, f'numvals_{self.lgs_refsig}'] = ref_sig_numvals
            segment_lagtimes_df.loc[segment_name, f'numvals_{self.lgs_lagsig}'] = lagged_sig_numvals
            segment_lagtimes_df.loc[segment_name, f'lagsearch_start'] = self.lgs_winsize[0]
            segment_lagtimes_df.loc[segment_name, f'lagsearch_end'] = self.lgs_winsize[1]
            segment_lagtimes_df.loc[segment_name, f'iteration'] = self.iteration

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
            outfile = f'{self.iteration}_segments_found_lag_times_after_iteration-{self.iteration}.csv'
            segment_lagtimes_df.to_csv(self.outdirs['2-0_Segment_Lag_Times'] / outfile)
            # self.logger.info(f"Saved segment lag times in {outfile}")

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

    def check_for_prev_results(self, iteration: int):
        """
        Check if previous results exist

        Returns
        -------
        segment_lagtimes_df

        """
        filepath_this_iteration = self.outdirs['2-0_Segment_Lag_Times'] \
                                  / f'{iteration}_segments_found_lag_times_after_iteration-{iteration}.csv'
        filepath_prev_iteration = self.outdirs['2-0_Segment_Lag_Times'] \
                                  / f'{iteration - 1}_segments_found_lag_times_after_iteration-{iteration - 1}.csv'
        if os.path.exists(filepath_this_iteration):
            # Check if current iteration was already done previously
            self.logger.info(f"(!) Results for iteration {iteration} already exist in file {filepath_this_iteration}, "
                             f"skipping to next iteration.")
            return None
        else:
            if os.path.exists(filepath_prev_iteration):
                # Get results from previous iteration
                segment_lagtimes_df = files.read_segment_lagtimes_file(filepath=filepath_prev_iteration)

                # Adjusted lagsearch window, from a previous iteration
                win_lagsearch = \
                    [segment_lagtimes_df['lagsearch_next_start'].unique()[0],
                     segment_lagtimes_df['lagsearch_next_end'].unique()[0]]
                self.logger.info(f"    Found results from previous iteration in {filepath_prev_iteration}, "
                                 f"time window for lag search set to {win_lagsearch}")
            else:
                # No results from previous iteration found
                segment_lagtimes_df = pd.DataFrame()
                win_lagsearch = self.lgs_winsize  # Initial lagsearch window
                self.logger.info(f"Initial time window for lag search set to {win_lagsearch}")

            return segment_lagtimes_df, win_lagsearch

    def add_lagsearch_adj_info(self, segment_lagtimes_df, iteration, next_win_lagsearch):
        """Add adjusted lagsearch window for next iteration to CSV"""
        segment_lagtimes_df.loc[:, 'lagsearch_next_start'] = next_win_lagsearch[0]
        segment_lagtimes_df.loc[:, 'lagsearch_next_end'] = next_win_lagsearch[1]

        # Save found segment lag times with next lagsearch info after all files
        filename_segments = f'{iteration}_segments_found_lag_times_after_iteration-{iteration}.csv'
        segment_lagtimes_df.to_csv(
            self.outdirs['2-0_Segment_Lag_Times'] / filename_segments)
        return

    @staticmethod
    def lagsearch_settings(win_lagsearch: list, force_min_stepsize: bool = False):
        """
        Set step size of shift during lag search and calculate number of
        histogram bins.

        Parameters
        ----------
        win_lagsearch: list
            Contains start record for lag search in [0] and end record
            in [1], e.g. [-1000, 1000] searches between records -1000 and
            +1000.

        force_min_stepsize: bool
            If True, sets the stepsize for lag search to 1, otherwise
            stepsize is calculated from win_lagsearch.


        Returns
        -------
        shift_stepsize: int
            Step-size (number of records) for lag search.
        hist_bin_range: range
            Bin range for the histogram of found lag times. The histogram
            is used to narrow down win_lagsearch.

        Examples
        --------
        >> shift_stepsize, hist_bin_range = lagsearch_settings(win_lagsearch=-1000, 1000])
        >> print(shift_stepsize)
        10
        >> print(hist_bin_range)
        range(-1000, 1000, 50)

        >> shift_stepsize, hist_bin_range = lagsearch_settings(win_lagsearch=-1000, 1000], force_min_stepsize=True)
        >> print(shift_stepsize)
        1
        >> print(hist_bin_range)
        range(-1000, 1000, 50)

        """
        # range_win_lagsearch = np.sum(np.abs(win_lagsearch))
        range_win_lagsearch = np.abs(win_lagsearch[0] - win_lagsearch[1])

        if not force_min_stepsize:
            shift_stepsize = int(range_win_lagsearch / 200)
            # shift_stepsize = int(range_win_lagsearch / 100 / 2)
            shift_stepsize = 1 if shift_stepsize < 1 else shift_stepsize  # Step-size cannot be less than 1
        else:
            shift_stepsize = 1

        # if shift_stepsize != 1:
        #     bins_stepsize = int(shift_stepsize * 5)
        # else:
        #     bins_stepsize = int(shift_stepsize)
        max_num_bins = range_win_lagsearch / shift_stepsize
        max_allowed_bins = int(max_num_bins / 3)
        bins_stepsize = int(range_win_lagsearch / max_allowed_bins)

        hist_bin_range = range(int(win_lagsearch[0]), int(win_lagsearch[1]), bins_stepsize)

        # if shift_stepsize != 1:
        #     bins_stepsize = int(range_win_lagsearch / (shift_stepsize * 5))
        #     hist_bin_range = range(int(win_lagsearch[0]), int(win_lagsearch[1]), bins_stepsize)
        #     # hist_bin_range = range(int(win_lagsearch[0]), int(win_lagsearch[1]), int(shift_stepsize * 5))
        # else:
        #     bins_stepsize = shift_stepsize
        #     hist_bin_range = range(int(win_lagsearch[0]), int(win_lagsearch[1]), bins_stepsize)

        return shift_stepsize, hist_bin_range


class PlotLoopResults:
    def __init__(self, dyla_instance, plot_cov_collection=True,
                 plot_hist=True, plot_timeseries_segment_lagtimes=True):
        self.dyla_instance = dyla_instance
        self.outdirs = dyla_instance.outdirs
        self.lgs_num_iter = dyla_instance.lgs_num_iter
        self.lgs_hist_perc_thres = dyla_instance.lgs_hist_perc_thres
        self.plot_cov_collection = plot_cov_collection
        self.plot_hist = plot_hist
        self.plot_timeseries_segment_lagtimes = plot_timeseries_segment_lagtimes

        self.logger = create_logger(logfile_path=dyla_instance.logfile_path, name=__name__)

    def run(self):
        # Covariance collection
        if self.plot_cov_collection:
            # Plot covariances from lag search for each segment in one plot
            self.cov_collection(indir=self.outdirs['1-0_Covariances'],
                                outdir=self.outdirs['1-1_____Plots'],
                                logfile_path=self.dyla_instance.logfile_path)
            # self.logger.info(f"Saved covariance collection plot in {outfile}")

        # Histogram
        if self.plot_hist:
            # Read found lag time results from very last iteration
            segment_lagtimes_df = files.read_segment_lagtimes_file(
                filepath=self.outdirs['2-0_Segment_Lag_Times'] /
                         f'{self.lgs_num_iter}_segments_found_lag_times_after_iteration-{self.lgs_num_iter}.csv')

            # Set search window for lag, depending on iteration
            hist_series = Loop.filter_series(filter_col='iteration', filter_equal_to=self.lgs_num_iter,
                                             df=segment_lagtimes_df, series_col='shift_peak_cov_abs_max')
            last_win_lagsearch = [int(segment_lagtimes_df.iloc[-1]['lagsearch_start']),
                                  int(segment_lagtimes_df.iloc[-1]['lagsearch_end'])]
            _, hist_num_bins = Loop.lagsearch_settings(win_lagsearch=last_win_lagsearch)
            _ = lag.AdjustLagsearchWindow(series=hist_series,
                                          outdir=self.outdirs['2-1_____Histograms'],
                                          iteration=self.lgs_num_iter,
                                          plot=True,
                                          hist_num_bins=hist_num_bins,
                                          remove_fringe_bins=False,
                                          perc_threshold=self.lgs_hist_perc_thres).get()
            self.logger.info(f"Created histogram plot for lag search range {last_win_lagsearch}")

        # Timeseries of lag times
        if self.plot_timeseries_segment_lagtimes:
            # Read found lag time results from very last iteration
            segment_lagtimes_df = files.read_segment_lagtimes_file(
                filepath=self.outdirs['2-0_Segment_Lag_Times']
                         / f'{self.lgs_num_iter}_segments_found_lag_times_after_iteration-{self.lgs_num_iter}.csv')
            outfile = plot.timeseries_segment_lagtimes(segment_lagtimes_df=segment_lagtimes_df,
                                                       outdir=self.outdirs['2-2_____Timeseries'],
                                                       iteration=self.lgs_num_iter,
                                                       show_all=True)
            self.logger.info(f"Created time series plot of {len(segment_lagtimes_df)} segments "
                             f"across {self.lgs_num_iter} iterations")

    @staticmethod
    def cov_collection(indir, outdir, logfile_path):
        """
        Read and plot segment covariance files.

        Parameters
        ----------
        indir: Path
        outdir: Path

        Returns
        -------
        None

        """
        from _setup import create_logger
        logger = create_logger(logfile_path=logfile_path, name=__name__)

        gs = gridspec.GridSpec(3, 1)  # rows, cols
        gs.update(wspace=0.3, hspace=0.2, left=0.03, right=0.97, top=0.95, bottom=0.03)
        fig = plt.Figure(facecolor='white', figsize=(16, 12))
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
        ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)

        # Read results from last iteration
        cov_collection_df = pd.DataFrame()
        filelist = os.listdir(str(indir))
        num_files = len(filelist)
        logger.info(f"Plotting covariance collection from {num_files} files")
        for idx, filename in enumerate(filelist):
            # if idx > 1000:
            #     break
            # print(f"Reading segment covariance file {idx + 1} of {num_files}: {filename}")
            filepath = os.path.join(str(indir), filename)
            segment_cov_df = files.read_segment_lagtimes_file(filepath=filepath)
            cov_collection_df = cov_collection_df.append(segment_cov_df)

            args = dict(alpha=0.05, c='black', lw=0.5, marker='None', zorder=98)
            ax1.plot(segment_cov_df['shift'], segment_cov_df['cov'], **args)
            ax2.plot(segment_cov_df['shift'], segment_cov_df['cov_abs'], **args)
            ax3.plot(segment_cov_df['shift'], segment_cov_df['cov_abs'].divide(segment_cov_df['cov_abs'].max()), **args)

        # Median lines
        _df = cov_collection_df[cov_collection_df['segment_name'].str.contains('iter')]
        _df = _df.groupby('shift').agg('median')
        args = dict(alpha=1, c='red', lw=1, marker='None', zorder=98)
        ax1.plot(_df.index, _df['cov'], label='median', **args)
        ax2.plot(_df.index, _df['cov_abs'], **args)
        ax3.plot(_df.index, _df['cov_abs'].divide(_df['cov_abs'].max()), **args)

        ax1.set_ylim(cov_collection_df['cov'].quantile(0.05), cov_collection_df['cov'].quantile(0.95))
        ax2.set_ylim(cov_collection_df['cov_abs'].quantile(0.05), cov_collection_df['cov_abs'].quantile(0.95))

        fig.suptitle("Results for all segments and from all iterations")
        text_args = dict(horizontalalignment='left', verticalalignment='top',
                         size=12, color='black', backgroundcolor='none', zorder=100)
        ax1.text(0.02, 0.97, "Covariances",
                 transform=ax1.transAxes, **text_args)
        ax2.text(0.02, 0.97, "Absolute covariances",
                 transform=ax2.transAxes, **text_args)
        ax3.text(0.02, 0.97, "Normalized absolute covariances\n(normalized to max of median line)",
                 transform=ax3.transAxes, **text_args)

        ax1.legend(frameon=False, loc='upper right')
        plot.default_format(ax=ax1, txt_xlabel='', txt_ylabel='covariance', txt_ylabel_units='')
        plot.default_format(ax=ax2, txt_xlabel='', txt_ylabel='absolute covariance', txt_ylabel_units='')
        plot.default_format(ax=ax3, txt_xlabel='shift [records]', txt_ylabel='normalized absolute covariance',
                            txt_ylabel_units='')

        outfile = '1_covariance_collection_all_segments.png'
        outpath = outdir / outfile
        fig.savefig(f"{outpath}", format='png', bbox_inches='tight', facecolor='w',
                    transparent=True, dpi=150)

        return None
