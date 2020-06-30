"""

DYNAMIC LAG REMOVER - DLR
-------------------------
A Python package to detect and compensate for shifting lag times in ecosystem time series



File:       Data file
Segment:    Segment in file, e.g. one 6-hour file may consist of 12 half-hour segments.

Step (1)
FilesDetector (files.FilesDetector)

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 1000)

import files
import loop


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
        from setup import Setup
        setup = Setup(outdir=self.dir_output,
                      del_previous_results=self.del_previous_results,
                      dir_input=self.dir_input,
                      fnm_pattern=self.fnm_pattern,
                      fnm_date_format=self.fnm_date_format,
                      file_generation_res=self.file_generation_res,
                      dat_recs_nominal_timeres=self.dat_recs_nominal_timeres,
                      files_how_many=self.files_how_many)
        setup.run()
        self.outdirs, self.files_overview_df = setup.get()

        # FILE LOOPS, LAG SEARCH
        # ======================
        # Loop through files and their segments
        for iteration in range(1, 1 + self.lgs_num_iter):
            loop_iter = loop.Loop(dat_recs_timestamp_format=self.dat_recs_timestamp_format,
                                  dat_recs_nominal_timeres=self.dat_recs_nominal_timeres,
                                  iteration=iteration,
                                  lgs_hist_remove_fringe_bins=self.lgs_hist_remove_fringe_bins,
                                  lgs_hist_perc_thres=self.lgs_hist_perc_thres,
                                  outdirs=self.outdirs,
                                  lgs_segment_dur=self.lgs_segment_dur,
                                  lgs_refsig=self.lgs_refsig,
                                  lgs_lagsig=self.lgs_lagsig,
                                  lgs_num_iter=self.lgs_num_iter,
                                  lgs_winsize=self.lgs_winsize,
                                  files_overview_df=self.files_overview_df)
            loop_iter.run()

        # PLOT LOOP RESULTS
        # =================
        loop_plots = loop.PlotLoopResults(outdirs=self.outdirs,
                                          lgs_num_iter=self.lgs_num_iter,
                                          lgs_hist_perc_thres=self.lgs_hist_perc_thres)
        loop_plots.run()
        # XXX = loop.get() todo

        # ANALYZE RESULTS
        # ===============
        self.generate_lut_default_lag_times()

    def generate_lut_default_lag_times(self):
        """Analyse found lag times from last iteration."""

        # Load results from last iteration
        last_iteration = self.lgs_num_iter
        filepath_last_iteration = self.outdirs['2-0_Segment_Lag_Times'] \
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

    def filter_dataframe(self, filter_col, filter_equal_to, df):
        filter_this_iteration = df[filter_col] == filter_equal_to
        df_filtered = df.loc[filter_this_iteration, :]
        return df_filtered

    def collect_file_data(self, data_df, file_idx, data_collection_df):
        if file_idx == self.files_overview_df.index[0]:
            data_collection_df = data_df.copy()
        else:
            data_collection_df = data_collection_df.append(data_df)
        return data_collection_df

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
