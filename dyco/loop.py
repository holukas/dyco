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

import os
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import files, lag, plot, setup_dyco


class Loop:
    """
    Loop through all files and their segments
    """

    def __init__(self,
                 dyco_instance,
                 iteration: int = 1):
        """

        Parameters
        ----------
        dyco_instance:

        iteration: int
            Number of current iteration.
        """
        self.dat_recs_timestamp_format = dyco_instance.dat_recs_timestamp_format
        self.dat_recs_nominal_timeres = dyco_instance.dat_recs_nominal_timeres
        self.lgs_hist_remove_fringe_bins = dyco_instance.lgs_hist_remove_fringe_bins
        self.lgs_hist_perc_thres = dyco_instance.lgs_hist_perc_thres
        self.outdirs = dyco_instance.outdirs
        self.lgs_segment_dur = dyco_instance.lgs_segment_dur
        self.var_reference = dyco_instance.var_reference
        self.var_lagged = dyco_instance.var_lagged
        self.lgs_num_iter = dyco_instance.lgs_num_iter
        self.files_overview_df = dyco_instance.files_overview_df
        self.logfile_path = dyco_instance.logfile_path
        self.lgs_winsize = dyco_instance.lgs_winsize
        self.fnm_date_format = dyco_instance.fnm_date_format
        self.phase = dyco_instance.phase
        self.phase_files = dyco_instance.phase_files
        self.new_iteration_data = dyco_instance.new_iteration_data  # Indicates if new iteration was executed
        # # force_min_stepsize = True if self.iteration == self.lgs_num_iter else False
        # self.shift_stepsize, \
        # self.hist_bin_range = self.lagsearch_settings(win_lagsearch=self.lgs_winsize,
        #                                               force_min_stepsize=False)

        self.iteration = iteration

        self.logger = setup_dyco.create_logger(logfile_path=self.logfile_path, name=__name__)

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
            self.new_iteration_data = True  # New iteration output will be generated

        self.shift_stepsize, \
        self.hist_bin_range = self.lagsearch_settings(lgs_winsize=self.lgs_winsize,
                                                      force_min_stepsize=False)

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
                                                              found_records=len(data_df),
                                                              fnm_date_format=self.fnm_date_format)

            # Loop through data segments in file
            segment_lagtimes_df = self.loop_segments(data_df=data_df,
                                                     file_idx=file_idx,
                                                     filename=file_info_row['filename'],
                                                     segment_lagtimes_df=segment_lagtimes_df)

        # Set new search window
        hist_series = self.filter_series(filter_col='iteration', filter_equal_to=self.iteration,
                                         df=segment_lagtimes_df, series_col='PEAK-COVABSMAX_SHIFT')
        lgs_winsize_adj = lag.AdjustLagsearchWindow(series=hist_series,
                                                    outdir=self.outdirs[
                                                        f'{self.phase}-4_{self.phase_files}_time_lags_overview_histograms'],
                                                    iteration=self.iteration,
                                                    plot=True,
                                                    remove_fringe_bins=self.lgs_hist_remove_fringe_bins,
                                                    hist_num_bins=self.hist_bin_range,
                                                    perc_threshold=self.lgs_hist_perc_thres).get()
        self.lgs_winsize = lgs_winsize_adj  # Update lag search window to new range

        # Add time window start and end for next iteration lagsearch
        self.add_lagsearch_adj_info(segment_lagtimes_df=segment_lagtimes_df,
                                    iteration=self.iteration,
                                    next_lgs_winsize=lgs_winsize_adj)

        # Plot all found segment lag times
        self.plot_segment_lagtimes_ts(segment_lagtimes_df=segment_lagtimes_df,
                                      outdir=self.outdirs[
                                          f'{self.phase}-5_{self.phase_files}_time_lags_overview_timeseries'],
                                      iteration=self.iteration,
                                      phase=self.phase)

    @staticmethod
    def plot_segment_lagtimes_ts(segment_lagtimes_df: pd.DataFrame,
                                 outdir,
                                 phase: int = 1,
                                 iteration: int = 1,
                                 show_all: bool = False,
                                 overlay_default: bool = False,
                                 overlay_default_df: pd.DataFrame = None,
                                 overlay_target_val: False or int = False):
        """
        Plot time series of found segment lag times

        Parameters
        ----------
        segment_lagtimes_df: pandas DataFrame
            Contains found lag times for all segments.
        outdir
        iteration: int
            Current lag search iteration.
        show_all: bool
            If False, plots only data from current iteration. In this case,
            segment_lagtimes_df is filtered to contain data from the current
            iteration only, other iterations are ignored.
            If True, plots data from all iterations in one plot.
        overlay_default: bool
            Option to plot default lag times for each day.
            If True, plots the 'median' column of overlay_default_df in addition
            to the iteration results.
        overlay_default_df: pandas DataFrame
            Contains the median default lag calculated from the last iteration
            in column 'median'.
        overlay_target_val: False or int
            Draws a horizontal line of the target lag. All files are normalized
            in a way so that the found default lag for each day corresponds to
            the target lag.

        Returns
        -------

        """
        _df = segment_lagtimes_df.copy()
        if not show_all:
            _df = _df.loc[_df['iteration'] == iteration, :]
            txt_info = f"PHASE {phase}, ITERATION {iteration}: found time lags"
            outfile = f"PHASE-{phase}_ITERATION-{iteration}_TIMESERIES-PLOT_segment_lag_times_iteration"
        else:
            txt_info = f"Found lag times from ALL iterations"
            outfile = f"TIMESERIES-PLOT_segment_lag_times_FINAL"

        gs, fig, ax = plot.setup_fig_ax()
        cmap = plt.cm.get_cmap('rainbow', iteration)
        colors = cmap(np.linspace(0, 1, iteration))

        alpha = .5
        iteration_grouped = _df.groupby('iteration')
        for idx, group_df in iteration_grouped:
            ax.plot_date(pd.to_datetime(group_df['start']), group_df['PEAK-COVABSMAX_SHIFT'],
                         alpha=alpha, marker='o', ms=6, color=colors[int(idx - 1)], lw=0, ls='-',
                         label=f'found lag times in iteration {int(idx)}', markeredgecolor='None')

        plot.default_format(ax=ax, label_color='black', fontsize=12,
                            txt_xlabel='segment date', txt_ylabel='lag', txt_ylabel_units='[records]')

        if overlay_default:
            Loop._add_overlay(overlay_default_df=overlay_default_df, ax=ax)

        # if overlay_target_val:
        #     ax.axhline(overlay_target_val, color='black', ls='--', label='target: normalized default lag')

        txt_info = f"PHASE {phase}, ITERATION {iteration}: FOUND TIME LAGS"
        font = {'family': 'sans-serif', 'color': 'black', 'weight': 'bold', 'size': 20, 'alpha': 1, }
        ax.set_title(txt_info, fontdict=font)

        ax.legend(frameon=True, loc='upper right').set_zorder(100)

        # Automatic tick locations and formats
        locator = mdates.AutoDateLocator(minticks=5, maxticks=20)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        # Save
        outpath = outdir / outfile
        # print(f"Saving time series of found segment lag times in {outpath} ...")
        fig.savefig(f"{outpath}.png", format='png', bbox_inches='tight', facecolor='w',
                    transparent=True, dpi=150)

        return outfile

    @staticmethod
    def _add_overlay(overlay_default_df, ax):
        if not overlay_default_df.empty:
            # Plot horizontal broken bars
            ax_yrange = np.abs(ax.get_ylim()).sum()
            yrange_offset = (ax_yrange / 50) / 2  # Height of bars
            make_label = True
            for current_date, row_data in overlay_default_df.iterrows():
                current_date = pd.to_datetime(current_date)
                width = pd.Timedelta(days=1)
                xrange = (current_date, width)
                yrange = (row_data['median'] - yrange_offset, yrange_offset * 2)
                yrange = sorted(yrange)
                yrange_target = (row_data['target_lag'] - yrange_offset, yrange_offset * 2)
                if make_label:
                    # Found lag times
                    ax.broken_barh([xrange], yrange, facecolors='#FDD835', alpha=1, edgecolor='None',
                                   label='3-day median lag time (centered)\nfrom high-quality covariance peaks')

                    # Lag times after normalization
                    ax.broken_barh([xrange], yrange_target, facecolors='#8BC34A', alpha=1, edgecolor='None',
                                   label='daily default lag after normalization')

                    # Correction arrow
                    # https://matplotlib.org/api/_as_gen/matplotlib.patches.FancyArrowPatch.html
                    try:
                        arrow = mpatches.FancyArrowPatch(
                            (current_date + pd.Timedelta(hours=12), row_data['median']),
                            (current_date + pd.Timedelta(hours=12), row_data['target_lag']),
                            mutation_scale=20, edgecolor='None', label='normalization correction')
                        ax.add_patch(arrow)
                        make_label = False
                    except:
                        pass

                else:
                    try:
                        ax.broken_barh([xrange], yrange, facecolors='#FDD835', alpha=1, edgecolor='None')
                        ax.broken_barh([xrange], yrange_target, facecolors='#8BC34A', alpha=1, edgecolor='None')
                        try:
                            arrow = mpatches.FancyArrowPatch(
                                (current_date + pd.Timedelta(hours=12), row_data['median']),
                                (current_date + pd.Timedelta(hours=12), row_data['target_lag']),
                                mutation_scale=20, edgecolor='None')
                            ax.add_patch(arrow)
                        except:
                            pass
                    except:
                        pass


        else:
            ax.text(0.5, 0.5, "No high-quality lags found, lag normalization failed",
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes,
                    size=20, color='white', backgroundcolor='red', zorder=100)

    def get(self):
        """
        After looping through all files, get time lag search window and
        flag that indicates if new iteration was executed

        Returns
        -------
        lgs_winsize: list
                    Contains min and max of time lag search window, given as number of records
        new_iteration_data: bool
                    Indicates if new iteration was executed
        """
        return self.lgs_winsize, self.new_iteration_data

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

            # Search lag
            cov_df, props_peak_auto = \
                lag.LagSearch(loop_instance=self,
                              segment_df=segment_df,
                              segment_name=segment_name,
                              segment_start=segment_start,
                              segment_end=segment_end,
                              outdir_plots=self.outdirs[f'{self.phase}-2_{self.phase_files}_covariances_plots'],
                              outdir_data=self.outdirs[f'{self.phase}-1_{self.phase_files}_covariances'],
                              file_idx=file_idx,
                              filename=filename).get()

            # Collect results
            numvals_var_reference = segment_df[self.var_reference].dropna().size
            numvals_var_lagged = segment_df[self.var_lagged].dropna().size

            segment_lagtimes_df.loc[segment_name, 'file_date'] = file_idx
            segment_lagtimes_df.loc[segment_name, 'start'] = segment_start
            segment_lagtimes_df.loc[segment_name, 'end'] = segment_end
            segment_lagtimes_df.loc[segment_name, f'numvals_{self.var_reference}'] = numvals_var_reference
            segment_lagtimes_df.loc[segment_name, f'numvals_{self.var_lagged}'] = numvals_var_lagged
            segment_lagtimes_df.loc[segment_name, f'lagsearch_start'] = self.lgs_winsize[0]
            segment_lagtimes_df.loc[segment_name, f'lagsearch_end'] = self.lgs_winsize[1]
            segment_lagtimes_df.loc[segment_name, f'iteration'] = self.iteration

            # Store lag results in overview
            stor_dict = {'stor_idx': segment_name,
                         'flag_df': cov_df}
            segment_lagtimes_df = self.store_search_results(stor_df=segment_lagtimes_df,
                                                            stor_cols={'shift': 'PEAK-COVABSMAX_SHIFT',
                                                                       'cov': 'PEAK-COVABSMAX_COV',
                                                                       'timestamp': 'PEAK-COVABSMAX_TIMESTAMP'},
                                                            flag_col='flag_peak_max_cov_abs', **stor_dict)
            segment_lagtimes_df = self.store_search_results(stor_df=segment_lagtimes_df,
                                                            stor_cols={'shift': 'PEAK-AUTO_SHIFT',
                                                                       'cov': 'PEAK-AUTO_COV',
                                                                       'timestamp': 'PEAK-AUTO_TIMESTAMP'},
                                                            flag_col='flag_peak_auto', **stor_dict)
            segment_lagtimes_df = self.store_search_results(stor_df=segment_lagtimes_df,
                                                            stor_cols={'shift': 'DEFAULT-LAG_SHIFT',
                                                                       'cov': 'DEFAULT-LAG_COV',
                                                                       'timestamp': 'DEFAULT-LAG_TIMESTAMP'},
                                                            flag_col='flag_instantaneous_default_lag', **stor_dict)

            # Save found segment lag times after each segment
            outfile = f'{self.iteration}_segments_found_lag_times_after_iteration-{self.iteration}.csv'
            segment_lagtimes_df.to_csv(self.outdirs[f'{self.phase}-3_{self.phase_files}_time_lags_overview'] / outfile)

        return segment_lagtimes_df

    def store_search_results(self, stor_df: pd.DataFrame, stor_cols: dict, stor_idx: int,
                             flag_df: pd.DataFrame, flag_col: str):
        """
        Get flag info from a DataFrame and store info in another DataFrame.

        Parameters
        ----------
        stor_df: pandas DataFrame
            Stores overview info.
        stor_cols: dict
            Contains column names used to store info in stor_df.
        stor_idx: int
        flag_df: pandas DataFrame
            Contains flag info.
        flag_col: str
            Flag column name in flag_df.

        Returns
        -------
        pandas DataFrame

        """
        # Init columns
        for key, col in stor_cols.items():
            stor_df.loc[stor_idx, col] = np.nan

        # Get indices of peaks and instantaneous default lag
        flag_idx = lag.LagSearch.get_peak_idx(df=flag_df, flag_col=flag_col)

        # Get info from flag_df and store it in stor_df
        if flag_idx:
            stor_df.loc[stor_idx, stor_cols['shift']] = flag_df.iloc[flag_idx]['shift']  # todo (maybe) give in sec
            stor_df.loc[stor_idx, stor_cols['cov']] = flag_df.iloc[flag_idx]['cov']
            stor_df.loc[stor_idx, stor_cols['timestamp']] = flag_df.iloc[flag_idx]['index']
        return stor_df

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
        filepath_this_iteration = self.outdirs[f'{self.phase}-3_{self.phase_files}_time_lags_overview'] \
                                  / f'{iteration}_segments_found_lag_times_after_iteration-{iteration}.csv'
        filepath_prev_iteration = self.outdirs[f'{self.phase}-3_{self.phase_files}_time_lags_overview'] \
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
                lgs_winsize = \
                    [segment_lagtimes_df['lagsearch_next_start'].unique()[0],
                     segment_lagtimes_df['lagsearch_next_end'].unique()[0]]
                self.logger.info(f"    Found results from previous iteration in {filepath_prev_iteration}, "
                                 f"time window for lag search set to {lgs_winsize}")
            else:
                # No results from previous iteration found
                segment_lagtimes_df = pd.DataFrame()
                lgs_winsize = self.lgs_winsize  # Initial lagsearch window
                self.logger.info(f"Initial time window for lag search set to {lgs_winsize}")

            return segment_lagtimes_df, lgs_winsize

    def add_lagsearch_adj_info(self, segment_lagtimes_df, iteration, next_lgs_winsize):
        """
        Add info about adjusted lagsearch window for next iteration to CSV

        Parameters
        ----------
        segment_lagtimes_df: pandas DataFrame
            Lag search results for each segment
        iteration: int
            Current iteration number
        next_lgs_winsize: list
            Time window for lag search in next iteration

        Returns
        -------
        None
        """
        segment_lagtimes_df.loc[:, 'lagsearch_next_start'] = next_lgs_winsize[0]
        segment_lagtimes_df.loc[:, 'lagsearch_next_end'] = next_lgs_winsize[1]

        # Save found segment lag times with next lagsearch info after all files
        filename_segments = f'{iteration}_segments_found_lag_times_after_iteration-{iteration}.csv'
        segment_lagtimes_df.to_csv(
            self.outdirs[f'{self.phase}-3_{self.phase_files}_time_lags_overview'] / filename_segments)
        return

    @staticmethod
    def lagsearch_settings(lgs_winsize: list, force_min_stepsize: bool = False):
        """
        Set step size of shift during lag search and calculate number of
        histogram bins.

        Parameters
        ----------
        lgs_winsize: list
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
        range_lgs_winsize = np.abs(lgs_winsize[0] - lgs_winsize[1])

        if not force_min_stepsize:
            shift_stepsize = int(range_lgs_winsize / 200)
            # shift_stepsize = int(range_win_lagsearch / 100 / 2)
            shift_stepsize = 1 if shift_stepsize < 1 else shift_stepsize  # Step-size cannot be less than 1
        else:
            shift_stepsize = 1

        max_num_bins = range_lgs_winsize / shift_stepsize
        max_allowed_bins = int(max_num_bins / 3)
        bins_stepsize = int(range_lgs_winsize / max_allowed_bins)

        hist_bin_range = range(int(lgs_winsize[0]), int(lgs_winsize[1]), bins_stepsize)

        return shift_stepsize, hist_bin_range


class PlotLoopResults:
    """

    Plot results after looping through all files

    """
    def __init__(self, dyco_instance, plot_cov_collection=True,
                 plot_hist=True, plot_timeseries_segment_lagtimes=True):
        self.dyco_instance = dyco_instance
        self.outdirs = dyco_instance.outdirs
        self.lgs_num_iter = dyco_instance.lgs_num_iter
        self.lgs_hist_perc_thres = dyco_instance.lgs_hist_perc_thres
        self.plot_cov_collection = plot_cov_collection
        self.plot_hist = plot_hist
        self.plot_timeseries_segment_lagtimes = plot_timeseries_segment_lagtimes
        self.phase = dyco_instance.phase
        self.phase_files = dyco_instance.phase_files
        self.new_iteration_data = dyco_instance.new_iteration_data

        self.logger = setup_dyco.create_logger(logfile_path=dyco_instance.logfile_path, name=__name__)

    def run(self):
        """Generate plots"""

        # Covariance collection
        if self.plot_cov_collection:
            # Plot covariances from lag search for each segment in one plot
            self.cov_collection(indir=self.outdirs[f'{self.phase}-1_{self.phase_files}_covariances'],
                                outdir=self.outdirs[f'{self.phase}-2_{self.phase_files}_covariances_plots'],
                                logfile_path=self.dyco_instance.logfile_path)

        # Histogram
        if self.plot_hist:
            # Read found lag time results from very last iteration
            segment_lagtimes_df = files.read_segment_lagtimes_file(
                filepath=self.outdirs[f'{self.phase}-3_{self.phase_files}_time_lags_overview'] /
                         f'{self.lgs_num_iter}_segments_found_lag_times_after_iteration-{self.lgs_num_iter}.csv')

            # Set search window for lag, depending on iteration
            hist_series = Loop.filter_series(filter_col='iteration', filter_equal_to=self.lgs_num_iter,
                                             df=segment_lagtimes_df, series_col='PEAK-COVABSMAX_SHIFT')
            last_lgs_winsize = [int(segment_lagtimes_df.iloc[-1]['lagsearch_start']),
                                int(segment_lagtimes_df.iloc[-1]['lagsearch_end'])]
            _, hist_num_bins = Loop.lagsearch_settings(lgs_winsize=last_lgs_winsize)
            _ = lag.AdjustLagsearchWindow(series=hist_series,
                                          outdir=self.outdirs[
                                              f'{self.phase}-4_{self.phase_files}_time_lags_overview_histograms'],
                                          iteration=self.lgs_num_iter,
                                          plot=True,
                                          hist_num_bins=hist_num_bins,
                                          remove_fringe_bins=False,
                                          perc_threshold=self.lgs_hist_perc_thres).get()
            self.logger.info(f"Created histogram plot for lag search range {last_lgs_winsize}")

        # Timeseries of lag times
        if self.plot_timeseries_segment_lagtimes:
            # Read found lag time results from very last iteration
            segment_lagtimes_df = files.read_segment_lagtimes_file(
                filepath=self.outdirs[f'{self.phase}-3_{self.phase_files}_time_lags_overview']
                         / f'{self.lgs_num_iter}_segments_found_lag_times_after_iteration-{self.lgs_num_iter}.csv')
            outfile = Loop.plot_segment_lagtimes_ts(segment_lagtimes_df=segment_lagtimes_df,
                                                    outdir=self.outdirs[
                                                        f'{self.phase}-5_{self.phase_files}_time_lags_overview_timeseries'],
                                                    iteration=self.lgs_num_iter,
                                                    show_all=True,
                                                    phase=self.phase)
            self.logger.info(f"Created time series plot of {len(segment_lagtimes_df)} segments "
                             f"across {self.lgs_num_iter} iterations")

    @staticmethod
    def cov_collection(indir, outdir, logfile_path):
        """
        Read and plot segment covariance files

        Parameters
        ----------
        indir: Path
        outdir: Path

        Returns
        -------
        None

        """

        logger = setup_dyco.create_logger(logfile_path=logfile_path, name=__name__)

        # Figure setup
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
            filepath = os.path.join(str(indir), filename)
            segment_cov_df = files.read_segment_lagtimes_file(filepath=filepath)
            cov_collection_df = cov_collection_df.append(segment_cov_df)  # Collect for median and quantiles calc

            # Plot each segment covariance file
            args = dict(alpha=0.05, c='black', lw=0.5, marker='None', zorder=98)
            ax1.plot(segment_cov_df['shift'], segment_cov_df['cov'], **args)
            ax2.plot(segment_cov_df['shift'], segment_cov_df['cov_abs'], **args)
            ax3.plot(segment_cov_df['shift'], segment_cov_df['cov_abs'].divide(segment_cov_df['cov_abs'].max()), **args)

        # Median and quantile lines
        def q25(x):
            return x.quantile(0.25)

        def q75(x):
            return x.quantile(0.75)

        f = {'median', q25, q75}
        labels = ['median', 'q25', 'q75']
        linestyle = ['-', '--', '--']
        colors = ['#f44336', '#2196F3', '#3F51B5']
        iter_df = cov_collection_df[cov_collection_df['segment_name'].str.contains('iter')]
        _df = iter_df.groupby('shift').agg(f)
        args = dict(alpha=1, lw=1, marker='None', zorder=98)
        for label_idx, label_name in enumerate(labels):
            ax1.plot(_df.index, _df['cov'][label_name],
                     label=label_name, ls=linestyle[label_idx], c=colors[label_idx], **args)
            ax2.plot(_df.index, _df['cov_abs'][label_name],
                     label=label_name, ls=linestyle[label_idx], c=colors[label_idx], **args)
            if label_name == 'median':
                series = _df['cov_abs'][label_name].divide(_df['cov_abs'][label_name].max())
                ax3.plot(_df.index, series,
                         label=label_name, ls=linestyle[label_idx], c=colors[label_idx], **args)

        # Format
        ax1.set_ylim(cov_collection_df['cov'].quantile(.1), cov_collection_df['cov'].quantile(.9))
        ax2.set_ylim(cov_collection_df['cov_abs'].quantile(.1), cov_collection_df['cov_abs'].quantile(.9))

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

        # Save
        outfile = '1_covariance_collection_all_segments.png'
        outpath = outdir / outfile
        fig.savefig(f"{outpath}", format='png', bbox_inches='tight', facecolor='w',
                    transparent=True, dpi=150)
