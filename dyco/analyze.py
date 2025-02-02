"""
    DYCO Dynamic Lag Compensation
    Copyright (C) 2020-2024 Lukas Hörtnagl

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""

import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from diive.pkgs.outlierdetection.zscore import zScoreRolling

from dyco import loop, plot


class AnalyzeLags:
    """
    Analyze lag search results and create look-up table for lag-time normalization

    * Creates LUT for daily default time lags (aggregated) needed in Phase 1 and Phase 2.
    * Creates LUT for instantaneous time lags needed in Phase 3.

    """

    def __init__(self,
                 lgs_num_iter,
                 outdirs,
                 target_lag,
                 logger,
                 outlier_winsize,
                 outlier_thres_zscore,
                 lags: pd.DataFrame):

        self.lgs_num_iter = lgs_num_iter
        self.outdirs = outdirs
        self.target_lag = target_lag
        self.logger = logger
        self.lags = lags.copy()

        self.outlier_winsize = outlier_winsize
        self.outlier_thres_zscore = outlier_thres_zscore

        self.lut_lag_times_df = pd.DataFrame()
        self.lut_available = False

        self.run()

    def get_lut(self):
        return self.lut_lag_times_df

    def run(self):
        """Run the lag analysis"""

        # segment_lagtimes_last_iteration_df['PEAK-COVABSMAX_SHIFT']

        # Make lookup table from aggregated daily lags
        self.lut_lag_times_df, self.lut_available = self.make_lut_agg()

        if self.lut_available:
            self.logger.info(f"Finished creating look-up table for default lag times and normalization correction")
        else:
            self.logger.critical(f"(!) Look-up Table for default lag times and normalization correction is empty, "
                                 f"stopping script.")
            sys.exit()

        if self.outdirs:
            self.save_lut(lut=self.lut_lag_times_df,
                          outdir=self.outdirs['7_time_lags_lookup_table'],
                          outfile='LUT_default_agg_time_lags')
            self.plot_segment_lagtimes_with_agg_default()

    @staticmethod
    def plot_final_instantaneous_lagtimes(outdir, phase, df):
        """
        Read and plot final lag search result: the instantaneous time lags

        Parameters
        ----------
        outdir: Path
            Output folder where results are stored
        phase: int
            Phase of the processing chain.
        df: pandas DataFrame

        Returns
        -------
        None
        """

        # Get data
        lagsearch_start = int(df['LAGSEARCH_START'].unique()[0])
        lagsearch_end = int(df['LAGSEARCH_END'].unique()[0])
        abs_limit = int(df['ABS_LIMIT'].unique()[0])

        # Plot
        gs, fig, ax = plot.setup_fig_ax()

        # Accepted reference lags
        ax.plot_date(pd.to_datetime(df.index), df['INSTANTANEOUS_LAG'],
                     alpha=1, fmt='o', ms=6, color='black', lw=0, ls='-',
                     label=f'final reference time lag (absolute limit {abs_limit})', markeredgecolor='None', zorder=100)

        # Found lags in Phase 3
        ax.plot_date(pd.to_datetime(df.index), df['PEAK-COVABSMAX_SHIFT'],
                     alpha=1, fmt='o', ms=12, color='#FFC107', lw=0, ls='-', markeredgecolor='None', zorder=99,
                     label=f'found Phase 3 time lag (search between {lagsearch_start} and {lagsearch_end})')

        # Marks lags that were outside limit and therefore set to the default lag
        set_to_default_lags = df.loc[df['SET_TO_DEFAULT'] == True, ['INSTANTANEOUS_LAG']]
        ax.plot_date(pd.to_datetime(set_to_default_lags.index), set_to_default_lags['INSTANTANEOUS_LAG'],
                     alpha=1, fmt='o', ms=12, color='#8BC34A', lw=0, ls='-', markeredgecolor='None', zorder=99,
                     label=f'found Phase 3 time lag was set to default')

        plot.default_format(ax=ax, label_color='black', fontsize=12,
                            txt_xlabel='segment date', txt_ylabel='lag', txt_ylabel_units='[records]')

        txt_info = f"PHASE {phase}: FINAL REFERENCE TIME LAGS"
        font = {'family': 'sans-serif', 'color': 'black', 'weight': 'bold', 'size': 20, 'alpha': 1, }
        ax.set_title(txt_info, fontdict=font)

        ax.axhline(0, color='black', ls='-', lw=1, label='default lag', zorder=98)
        limit = df['ABS_LIMIT'].unique()[0]
        ax.axhline(limit, color='#d32f2f', ls='--', lw=1, label='upper lag acceptance limit', zorder=98)
        ax.axhline(limit * -1, color='#7B1FA2', ls='--', lw=1, label='lower lag acceptance limit', zorder=98)
        font = {'family': 'sans-serif', 'size': 10}
        ax.legend(frameon=True, loc='upper right', prop=font).set_zorder(100)

        # Automatic tick locations and formats
        locator = mdates.AutoDateLocator(minticks=5, maxticks=20)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        # Save
        outpath = outdir / f'PHASE-{phase}_FINAL_TIME_LAGS_FOR_REFERENCE_VAR'
        # print(f"Saving time series of found segment lag times in {outpath} ...")
        fig.savefig(f"{outpath}.png", format='png', bbox_inches='tight', facecolor='w',
                    transparent=True, dpi=150)
        plt.close(fig)

    def plot_segment_lagtimes_with_agg_default(self):
        """
        Read and plot found time lags from all or only very last iteration(s),
        used in Phase 1 and Phase 2.
        """
        loop.Loop.plot_segment_lagtimes_ts(segment_lagtimes_df=self.lags,
                                           outdir=self.outdirs[f'7_time_lags_lookup_table'],
                                           show_all=False,
                                           overlay_default=True,
                                           overlay_default_df=self.lut_lag_times_df,
                                           overlay_target_val=self.target_lag)

    def save_lut(self, lut, outdir, outfile):
        """
        Save the look-up table for lag correction to csv file

        Parameters
        ----------
        lut: pandas DataFrame
            Contains look-up table information.
        outdir: Path
            Output folder for outfile.
        outfile: str
            Filename without extension.

        Returns
        -------
        None
        """
        outpath = outdir / outfile
        lut.to_csv(f"{outpath}.csv")


    def make_lut_instantaneous(self, segment_lagtimes_df: pd.DataFrame, default_lag: int):
        """
        Generate instantaneous lag look-up table that contains the found lag time
        for each averaging interval

        Used in Phase 3.

        Parameters
        ----------
        segment_lagtimes_df: pandas DataFrame
            Contains found lag times for each segment.
        default_lag: int
            The median of high-quality peaks is moved to target_lag.

        Returns
        -------
        pandas DataFrame with default lag times for each day

        """
        # Prepare df
        _segment_lagtimes_df = segment_lagtimes_df.set_index('file_date')
        lut_df = _segment_lagtimes_df[['PEAK-COVABSMAX_SHIFT']].copy()  # file_date needed for look-up
        # lut_df = segment_lagtimes_df[['file_date', 'PEAK-COVABSMAX_SHIFT']].copy()  # file_date needed for look-up
        # lut_df.set_index('file_date', inplace=True)
        lut_df['LAGSEARCH_START'] = np.nan
        lut_df['LAGSEARCH_END'] = np.nan
        lut_df['ABS_LIMIT'] = np.nan
        lut_df['DEFAULT_LAG'] = np.nan
        lut_df['SET_TO_DEFAULT'] = np.nan
        lut_df['INSTANTANEOUS_LAG'] = np.nan
        lut_df['DEFAULT_LAG'] = default_lag

        lut_df['ABS_LIMIT'] = 50
        lut_df['LAGSEARCH_START'] = _segment_lagtimes_df['lagsearch_start']
        lut_df['LAGSEARCH_END'] = _segment_lagtimes_df['lagsearch_end']

        # Replace found lags above absolute threshold with default lag, keep others
        filter_set_to_default = (lut_df['PEAK-COVABSMAX_SHIFT'].abs() > lut_df['ABS_LIMIT']) | \
                                (lut_df['PEAK-COVABSMAX_SHIFT'].isnull())
        lut_df['SET_TO_DEFAULT'] = filter_set_to_default
        lut_df.loc[filter_set_to_default, 'INSTANTANEOUS_LAG'] = lut_df.loc[filter_set_to_default, 'DEFAULT_LAG']
        lut_df.loc[~filter_set_to_default, 'INSTANTANEOUS_LAG'] = lut_df.loc[
            ~filter_set_to_default, 'PEAK-COVABSMAX_SHIFT']

        self.logger.info(f"Created look-up table for {len(lut_df.index)} dates")
        self.logger.info(f"    First date: {lut_df.index[0]}    Last date: {lut_df.index[-1]}")

        # Check for gaps
        missing_df = self.check_missing(df=lut_df,
                                        col='INSTANTANEOUS_LAG')
        if missing_df.empty:
            # All lags available
            pass
        else:
            # Some or all lags missing, fill with default lag
            self.logger.warning(f"No lag was available for dates: {missing_df.index.to_list()}")
            self.logger.warning(f"Filling missing lags with default lag, affected dates: {missing_df.index.to_list()}")
            lut_df['INSTANTANEOUS_LAG'].fillna(lut_df['DEFAULT_LAG'])

        lut_available = True
        return lut_df, lut_available

    def _remove_outliers(self, peaks_hq_S):
        self.outlier_winsize = int(len(peaks_hq_S) / 70) if not self.outlier_winsize else self.outlier_winsize
        zsr = zScoreRolling(
            series=peaks_hq_S,
            thres_zscore=self.outlier_thres_zscore,
            winsize=self.outlier_winsize,
            showplot=True,
            plottitle="z-score in a rolling window",
            verbose=True)
        zsr.calc(repeat=True)

        fig = zsr.fig

        # Save
        outdir = self.outdirs[f'7_time_lags_lookup_table']
        outfile = f"TIMESERIES-PLOT_segment_lag_times_FINAL_outlierRemoved"
        outpath = outdir / outfile
        # print(f"Saving time series of found segment lag times in {outpath} ...")
        fig.savefig(f"{outpath}.png", format='png', bbox_inches='tight', facecolor='w',
                    transparent=True, dpi=150)
        plt.close(fig)

        flag = zsr.get_flag()
        peaks_hq_S_cleaned = peaks_hq_S.loc[flag == 0].copy()
        return peaks_hq_S_cleaned

    def make_lut_agg(self):
        """
        Generate aggregated look-up table that contains the default lag time for each day

        Default lag times are determined by
            (1) pooling data of the current day with data of the day before and
                the day after,
            (2) calculating the median of the pooled data.

        Used in Phase 1 and Phase 2.

        Parameters
        ----------
        segment_lagtimes_df: pandas DataFrame
            Contains found lag times for each segment.
        target_lag: int
            The median of high-quality peaks is moved to target_lag.

        Returns
        -------
        pandas DataFrame with default lag times for each day

        """

        # Initiate empty LUT
        lut_df = pd.DataFrame()

        # Find high-quality covariance peaks
        peaks_hq_S = self.get_hq_peaks()
        peaks_hq_S = peaks_hq_S.sort_index(inplace=False)
        peaks_hq_S_cleaned = self._remove_outliers(peaks_hq_S)

        if peaks_hq_S_cleaned.empty:
            lut_available = False
            return lut_df, lut_available

        unique_dates = np.unique(peaks_hq_S_cleaned.index.date)
        for this_date in unique_dates:
            from_date = this_date - pd.Timedelta('2D')
            to_date = this_date + pd.Timedelta('2D')
            filter_around_this_day = (peaks_hq_S_cleaned.index.date > from_date) & \
                                     (peaks_hq_S_cleaned.index.date <= to_date)
            subset = peaks_hq_S_cleaned[filter_around_this_day]
            num_vals = len(subset)

            if num_vals >= 10:
                # print(f"{this_date}    {num_vals}    {subset.median()}")
                lut_df.loc[this_date, 'median'] = subset.median()
            else:
                lut_df.loc[this_date, 'median'] = np.nan

            lut_df.loc[this_date, 'counts'] = subset.count()
            lut_df.loc[this_date, 'from'] = from_date
            lut_df.loc[this_date, 'to'] = to_date

        # Make sure LUT has all dates between start and end dates
        fullrange = pd.date_range(lut_df.index.min(), lut_df.index.max(), freq='d')
        lut_df = lut_df.reindex(fullrange)

        # Filling missing median values with rolling median in a 5-day window, centered
        n_missing_medians = lut_df['median'].isnull().sum()
        if n_missing_medians > 0:
            lut_df['median'] = lut_df['median'].fillna(
                lut_df['median'].rolling(window=5, min_periods=1, center=True).median())
            self.logger.info(f"(!)WARNING: Missing look-up values for {n_missing_medians} days "
                             f"were gap-filled with the median value of the 2 preceding "
                             f"and the 2 following days.")

        lut_df['target_lag'] = self.target_lag
        lut_df['correction'] = -1 * (lut_df['target_lag'] - lut_df['median'])

        self.logger.info(f"Created look-up table for {len(lut_df.index)} dates")
        self.logger.info(f"    First date: {lut_df.index[0]}    Last date: {lut_df.index[-1]}")

        # Fill gaps in 'correction'
        missing_df = self.check_missing(df=lut_df,
                                        col='correction')
        self.logger.warning(f"No correction could be generated from data for dates: {missing_df.index.to_list()}")
        self.logger.warning(f"Filling missing corrections for dates: {missing_df.index.to_list()}")
        lut_df['correction'] = lut_df['correction'].ffill(limit=1)
        lut_df['correction'] = lut_df['correction'].bfill(limit=1)
        # lut_df['correction'] = lut_df['correction'].fillna(method='ffill', inplace=False, limit=1)  # deprecated

        lut_available = True
        return lut_df, lut_available

    def check_missing(self, df, col):
        """
        Check for missing values in data rows

        Parameters
        ----------
        df: pandas DataFrame
        col: str
            Column name of the variable that is checked for missing data.

        Returns
        -------
        pandas DataFrame that only contains data rows where a value for col is missing
        """
        filter_missing = df[col].isnull()
        missing_df = df[filter_missing]
        return missing_df

    def get_hq_peaks(self):
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
        df = self.lags.copy()
        df.set_index('start', inplace=True)
        df.index = pd.to_datetime(df.index, format="mixed")
        peaks_hq_S = df.loc[df['PEAK-COVABSMAX_SHIFT'] == df['PEAK-AUTO_SHIFT'], 'PEAK-COVABSMAX_SHIFT']
        peaks_hq_S.index = peaks_hq_S.index.to_pydatetime()  # Convert to DatetimeIndex
        return peaks_hq_S
