import sys

import numpy as np
import pandas as pd

import files
import plot
from _setup import create_logger


class AnalyzeLoopResults:
    def __init__(self, lgs_num_iter, outdirs=None, lag_target=-100, logfile_path=None,
                 direct_path_to_segment_lagtimes_file=None):
        self.lgs_num_iter = lgs_num_iter
        self.outdirs = outdirs
        self.lag_target = lag_target
        self.direct_path_to_segment_lagtimes_file = direct_path_to_segment_lagtimes_file

        self.logger = create_logger(logfile_path=logfile_path, name=__name__)

    def run(self):
        self.lut_default_lag_times_df, self.lut_success = self.generate_lut_default_lag_times()

        if self.outdirs:
            self.save_lut(lut=self.lut_default_lag_times_df,
                          outdir=self.outdirs['3-0_Lookup_Table_Normalization'])
            self.plot_segment_lagtimes_with_default()

    def get(self):
        return self.lut_default_lag_times_df, self.lut_success

    def plot_segment_lagtimes_with_default(self):
        # Read found lag time results from very last iteration
        segment_lagtimes_df = files.read_segment_lagtimes_file(
            filepath=self.outdirs['2-0_Segment_Lag_Times']
                     / f'{self.lgs_num_iter}_segments_found_lag_times_after_iteration-{self.lgs_num_iter}.csv')
        plot.timeseries_segment_lagtimes(df=segment_lagtimes_df,
                                         outdir=self.outdirs['3-0_Lookup_Table_Normalization'],
                                         iteration=self.lgs_num_iter,
                                         show_all=True,
                                         overlay_default=True,
                                         overlay_default_df=self.lut_default_lag_times_df,
                                         overlay_target_val=-100)

    def save_lut(self, lut, outdir):
        outpath = outdir / f'LUT_default_lag_times'
        lut.to_csv(f"{outpath}.csv")

    def generate_lut_default_lag_times(self):
        """Analyse found lag times from last iteration."""

        # Load results from last iteration
        last_iteration = self.lgs_num_iter
        if self.outdirs:
            filepath_last_iteration = self.outdirs['2-0_Segment_Lag_Times'] \
                                      / f'{last_iteration}_segments_found_lag_times_after_iteration-{last_iteration}.csv'

        else:
            filepath_last_iteration = self.direct_path_to_segment_lagtimes_file  # Implemented for unittest

        segment_lagtimes_last_iteration_df = self.filter_dataframe(filter_col='iteration',
                                                                   filter_equal_to=last_iteration,
                                                                   df=files.read_segment_lagtimes_file(
                                                                       filepath=filepath_last_iteration))
        # High-quality covariance peaks
        peaks_hq_S = self.get_hq_peaks(df=segment_lagtimes_last_iteration_df)

        if not peaks_hq_S.empty:
            lut_df = self.make_lut(series=peaks_hq_S,
                                   lag_target=self.lag_target)
            lut_success = True
            self.logger.info(f"Finished creating look-up table for default lag times and normalization correction")
        else:
            lut_df = pd.DataFrame()
            lut_success = False
            self.logger.critical(f"(!) Look-up Table for default lag times and normalization correction is empty, "
                                 f"stopping script.")
            sys.exit()

        return lut_df, lut_success

        # counts, divisions = lag.AdjustLagsearchWindow.calc_hist(series=peaks_hq_S,
        #                                                         bins=20,
        #                                                         remove_fringe_bins=False)

        # {start}  {end}:
        # print(f"Max bin b/w {divisions[np.argmax(counts)]} and {divisions[np.argmax(counts) + 1]}    "

        # quantiles_df = lag.calc_quantiles(df=_df).copy()
        # plot.results(df=quantiles_df)

    def filter_dataframe(self, filter_col, filter_equal_to, df):
        filter_this_iteration = df[filter_col] == filter_equal_to
        df_filtered = df.loc[filter_this_iteration, :]
        return df_filtered

    def make_lut(self, series, lag_target):
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

            if num_vals >= 5:
                # print(f"{this_date}    {num_vals}    {subset.median()}")
                lut_df.loc[this_date, 'median'] = subset.median()
                lut_df.loc[this_date, 'counts'] = subset.count()
                lut_df.loc[this_date, 'from'] = from_date
                lut_df.loc[this_date, 'to'] = to_date
            else:
                lut_df.loc[this_date, 'median'] = np.nan
                lut_df.loc[this_date, 'counts'] = subset.count()
                lut_df.loc[this_date, 'from'] = from_date
                lut_df.loc[this_date, 'to'] = to_date

        lut_df['correction'] = -1 * (lag_target - lut_df['median'])

        self.logger.info(f"Created look-up table for {len(lut_df.index)} dates")
        self.logger.info(f"    First date: {lut_df.index[0]}    Last date: {lut_df.index[-1]}")

        # Fill gaps in 'correction'
        missing_df = self.check_missing(df=lut_df,
                                        col='correction')
        self.logger.warning(f"No correction could be generated from data for dates: {missing_df.index.to_list()}")
        self.logger.warning(f"Filling missing corrections for dates: {missing_df.index.to_list()}")
        lut_df['correction'].fillna(method='ffill', inplace=True, limit=1)
        return lut_df

    def check_missing(self, df, col):
        filter_missing = df[col].isnull()
        missing_df = df[filter_missing]
        return missing_df

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

# def find_default(self, df):
#     plot_df = df[['cov_max_shift']].copy()
#
#     for b in range(1, 4, 1):
#         bins = 2
#         plot_df['group'] = pd.cut(plot_df['cov_max_shift'],
#                                   bins=bins, retbins=False,
#                                   duplicates='drop', labels=False)
#         plot_df_agg = plot_df.groupby('group').agg(['count', 'min', 'max'])
#         idxmax = plot_df_agg['cov_max_shift']['count'].idxmax()  # Most counts
#         group_max = plot_df_agg.iloc[idxmax].name
#
#         plot_df_agg['count_maxperc'] = \
#             plot_df_agg['cov_max_shift']['count'] / plot_df_agg['cov_max_shift']['count'].sum()
#         # plot_df_agg['cov_max_shift']['count'] / plot_df_agg.iloc[idxmax]['cov_max_shift']['count']
#
#         plot_df = plot_df.loc[plot_df['group'] == group_max]
#
#     median = plot_df['cov_max_shift'].median()
#     _min = plot_df['cov_max_shift'].min()
#     _max = plot_df['cov_max_shift'].max()
#
#     print(plot_df)
#     print(f"{median}  {_min}  {_max}")
