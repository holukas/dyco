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
from files import FilesDetector
import plot


class DynamicLagRemover:
    # Dataframes for results collection

    files_overview_df = pd.DataFrame()

    def __init__(self, lgs_refsig, lgs_lagsig, dir_root,
                 fnm_date_format='%Y%m%d%H%M%S',
                 del_previous_results=False,
                 fnm_pattern='*.csv',
                 dat_recs_timestamp_format=False,
                 files_how_many=False,
                 file_generation_res='30T',
                 file_duration='30T',
                 lgs_segment_dur=False,
                 lgs_hist_perc_thres=0.9,
                 lgs_hist_remove_fringe_bins=True,
                 dat_recs_nominal_timeres=0.05,
                 lgs_winsize=1000,
                 lgs_num_iter=3,
                 dir_input_ext=False,
                 dir_output_ext=False,
                 dir_input='input',
                 dir_output='output'):

        # Input and output directories
        self.dir_root = dir_root
        self.dir_input, self.dir_output = self.set_dirs(dir_input=dir_input, dir_input_ext=dir_input_ext,
                                                        dir_output=dir_output, dir_output_ext=dir_output_ext)

        # Measurements
        self.lgs_refsig = lgs_refsig
        self.lgs_lagsig = lgs_lagsig

        # File settings
        self.fnm_date_format = fnm_date_format
        self.file_generation_res = file_generation_res
        self.fnm_pattern = fnm_pattern
        self.files_how_many = files_how_many
        self.del_previous_results = del_previous_results

        self.dat_recs_nominal_timeres = dat_recs_nominal_timeres
        self.dat_recs_timestamp_format = dat_recs_timestamp_format
        self.file_duration = file_duration
        # self.data_segment_overhang = data_segment_overhang

        self.lgs_segment_dur = lgs_segment_dur
        self.lgs_winsize = [lgs_winsize * -1, lgs_winsize]
        self.lgs_num_iter = lgs_num_iter
        self.lgs_hist_remove_fringe_bins = lgs_hist_remove_fringe_bins
        self.lgs_hist_perc_thres = lgs_hist_perc_thres

        self.run()

    def run(self):

        # SETUP
        # =====
        self.outdirs = files.setup_output_dirs(outdir=self.dir_output,
                                               del_previous_results=self.del_previous_results)
        self.files_overview_df = self.search_files()

        # # LAG SEARCH
        # # ==========
        # # Loop through files and their segments todo activate
        # for iteration in range(1, 1 + self.lgs_num_iter):
        #     self.loop_files(iteration=iteration)
        # self.plot_loop_results()  # Plot results after loops finished

        # ANALYZE RESULTS
        # ===============
        self.generate_lut_default_lag_times()

    def generate_lut_default_lag_times(self):
        """Analyse found lag times from last iteration."""

        # Load results from last iteration
        last_iteration = self.lgs_num_iter
        filepath_last_iteration = self.outdirs['2-0___[DATA]__Segment_Lag_Times'] \
                                  / f'{last_iteration}_segments_found_lag_times_after_iteration-{last_iteration}.csv'
        results_last_iteration_df = self.filter_dataframe(filter_col='iteration',
                                                          filter_equal_to=last_iteration,
                                                          df=files.read_segments_file(filepath=filepath_last_iteration))

        # High-quality covariance peaks
        peaks_hq_S = self.get_hq_peaks(df=results_last_iteration_df)

        lut_df = self.make_lut(series=peaks_hq_S)

        lut_df['median'].plot()
        plt.show()

        # counts, divisions = lag.AdjustLagsearchWindow.calc_hist(series=peaks_hq_S,
        #                                                         bins=20,
        #                                                         remove_fringe_bins=False)

        # {start}  {end}:
        # print(f"Max bin b/w {divisions[np.argmax(counts)]} and {divisions[np.argmax(counts) + 1]}    "

        # quantiles_df = lag.calc_quantiles(df=_df).copy()
        # plot.results(df=quantiles_df)

    def plot_loop_results(self):
        # Plot covariances from lag search for each segment in one plot
        plot.cov_collection(indir=self.outdirs['1-0___[DATA]__Segment_Covariances'],
                            outdir=self.outdirs['1-1___[PLOTS]_Segment_Covariances'])

        # Read found lag time results from very last iteration
        segment_lagtimes_df = files.read_segments_file(
            filepath=self.outdirs['2-0___[DATA]__Segment_Lag_Times']
                     / f'{self.lgs_num_iter}_segments_found_lag_times_after_iteration-{self.lgs_num_iter}.csv')

        # Set search window for lag, depending on iteration #todo activate
        hist_series = self.filter_series(filter_col='iteration', filter_equal_to=self.lgs_num_iter,
                                         df=segment_lagtimes_df, series_col='shift_peak_cov_abs_max')
        last_win_lagsearch = [int(segment_lagtimes_df.iloc[-1]['lagsearch_start']),
                              int(segment_lagtimes_df.iloc[-1]['lagsearch_end'])]
        _, hist_num_bins = lag.search_settings(win_lagsearch=last_win_lagsearch)
        _ = lag.AdjustLagsearchWindow(series=hist_series,
                                      outdir=self.outdirs['2-1___[PLOTS]_Segment_Lag_Times_Histograms'],
                                      iteration=self.lgs_num_iter,
                                      plot=True,
                                      hist_num_bins=hist_num_bins,
                                      remove_fringe_bins=False,
                                      perc_threshold=self.lgs_hist_perc_thres).get()

        plot.timeseries_segment_lagtimes(df=segment_lagtimes_df,
                                         outdir=self.outdirs['2-2___[PLOTS]_Segment_Lag_Times_Timeseries'],
                                         iteration=self.lgs_num_iter,
                                         show_all=True)

    def search_files(self):
        """
        Search available files.

        Returns
        -------
        pandas DataFrame

        """
        fd = FilesDetector(dir_input=self.dir_input,
                           outdir=self.outdirs['0-0___[DATA]__Found_Files'],
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

    def get_hq_peaks(self, df):
        """
        Detect high-quality covariance peaks in results from last lag search iteration

        High-quality means that during the covariance calculations the max covariance
        peak and the automatically detected peak yielded the same results, i.e. the
        same record.

        Parameters
        ----------
        df: pandas DataFrame containing results from the last lag search iteration

        Returns
        -------
        pandas Series of high-quality lag times, given as number of records

        """
        df.set_index('start', inplace=True)
        df.index = pd.to_datetime(df.index)
        peaks_hq_S = df.loc[df['shift_peak_cov_abs_max'] == df['shift_peak_auto'],
                            'shift_peak_cov_abs_max']
        peaks_hq_S.index = peaks_hq_S.index.to_pydatetime()  # Convert to DatetimeIndex
        return peaks_hq_S

    def make_lut(self, series):
        """
        Generate look-up table that contains the default lag time for each day

        Default lag times are determined by
            (1) pooling data of the current day with data of the day before and
                the day after,
            (2) calculating the median of the pooled data.

        Parameters
        ----------
        series: pandas Series containing high-quality lag times

        Returns
        -------
        pandas DataFrame with default lag times for each day

        """
        lut_df = pd.DataFrame()
        unique_dates = np.unique(series.index.date)
        for this_date in unique_dates:
            from_date = this_date - pd.Timedelta('1D')
            to_date = this_date + pd.Timedelta('1D')
            filter_around_this_day = (series.index.date > from_date) & \
                                     (series.index.date <= to_date)
            subset = series[filter_around_this_day]
            num_vals = len(subset)
            print(f"{this_date}    {num_vals}    {subset.median()}")
            lut_df.loc[this_date, 'median'] = subset.median()
            lut_df.loc[this_date, 'counts'] = subset.count()
            lut_df.loc[this_date, 'from'] = from_date
            lut_df.loc[this_date, 'to'] = to_date

        lut_df['correction'] = -100 - lut_df['median']
        return lut_df

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
                win_lagsearch = self.lgs_winsize

            return segment_lagtimes_df, win_lagsearch

    def loop_files(self, iteration):
        """Loop through all found files."""

        prev_results = self.check_for_prev_results(iteration=iteration)
        if not prev_results:
            return None
        else:
            segment_lagtimes_df = prev_results[0]
            self.lgs_winsize = prev_results[1]

        shift_stepsize, hist_num_bins = lag.search_settings(win_lagsearch=self.lgs_winsize)
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
                                                     win_lagsearch=self.lgs_winsize,
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
                                                      remove_fringe_bins=self.lgs_hist_remove_fringe_bins,
                                                      hist_num_bins=hist_num_bins,
                                                      perc_threshold=self.lgs_hist_perc_thres).get()
        self.lgs_winsize = win_lagsearch_adj  # Update lag search window to new range

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
                              outdir_plots=self.outdirs['1-1___[PLOTS]_Segment_Covariances'],
                              outdir_data=self.outdirs['1-0___[DATA]__Segment_Covariances'],
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
                self.outdirs['2-0___[DATA]__Segment_Lag_Times'] / filename_segments)

        return segment_lagtimes_df

    def set_dirs(self, dir_input, dir_input_ext, dir_output, dir_output_ext):
        """
        In case external dir was selected for input, set it as source dir.

        Parameters
        ----------
        dir_input
        dir_input_ext
        dir_output
        dir_output_ext

        Returns
        -------

        """
        if dir_input_ext:
            dir_input = dir_input_ext  # Override
        else:
            dir_input = self.dir_root / dir_input
        if dir_output_ext:
            dir_output = dir_output_ext  # Override
        else:
            dir_output = self.dir_root / dir_output
        return dir_input, dir_output

# if __name__ == "__main__":
#     DynamicLagRemover()
