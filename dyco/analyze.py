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

import matplotlib.dates as mdates
import numpy as np
import pandas as pd

import files, loop, plot, setup_dyco


class AnalyzeLags:
    """
    Analyze lag search results and create look-up table for lag-time normalization

    * Creates LUT for daily default time lags (aggregated) needed in Phase 1 and Phase 2.
    * Creates LUT for instantaneous time lags needed in Phase 3.

    """

    def __init__(self,
                 dyco_instance,
                 direct_path_to_segment_lagtimes_file=None):

        self.lgs_num_iter = dyco_instance.lgs_num_iter
        self.outdirs = dyco_instance.outdirs
        self.target_lag = dyco_instance.target_lag
        self.direct_path_to_segment_lagtimes_file = direct_path_to_segment_lagtimes_file
        self.phase = dyco_instance.phase
        self.phase_files = dyco_instance.phase_files

        self.logger = setup_dyco.create_logger(logfile_path=dyco_instance.logfile_path, name=__name__)

        self.run()

    def run(self):
        """Run the lag analysis"""
        self.lut_lag_times_df, self.lut_available = self.generate_lut_time_lags()

        if self.outdirs:
            if self.phase != 3:
                self.save_lut(lut=self.lut_lag_times_df,
                              outdir=self.outdirs[f'{self.phase}-6_{self.phase_files}_normalization_lookup_table'],
                              outfile='LUT_default_agg_time_lags')
                self.plot_segment_lagtimes_with_agg_default()
            if self.phase == 3:
                self.save_lut(lut=self.lut_lag_times_df,
                              outdir=self.outdirs[f'{self.phase}-6_{self.phase_files}_final_time_lags_lookup_table'],
                              outfile='LUT_final_time_lags')
                outdir = self.outdirs[f'{self.phase}-6_{self.phase_files}_final_time_lags_lookup_table']
                AnalyzeLags.plot_final_instantaneous_lagtimes(
                    outdir=outdir,
                    phase=self.phase,
                    df=files.read_segment_lagtimes_file(filepath=outdir / 'LUT_final_time_lags.csv'))

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

    def plot_segment_lagtimes_with_agg_default(self):
        """
        Read and plot found time lags from all or only very last iteration(s),
        used in Phase 1 and Phase 2.
        """
        segment_lagtimes_df = files.read_segment_lagtimes_file(
            filepath=self.outdirs[f'{self.phase}-3_{self.phase_files}_time_lags_overview']
                     / f'{self.lgs_num_iter}_segments_found_lag_times_after_iteration-{self.lgs_num_iter}.csv')
        loop.Loop.plot_segment_lagtimes_ts(segment_lagtimes_df=segment_lagtimes_df,
                                           outdir=self.outdirs[
                                               f'{self.phase}-6_{self.phase_files}_normalization_lookup_table'],
                                           iteration=self.lgs_num_iter,
                                           show_all=False,
                                           overlay_default=True,
                                           overlay_default_df=self.lut_lag_times_df,
                                           overlay_target_val=self.target_lag,
                                           phase=self.phase)

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

    def generate_lut_time_lags(self):
        """
        Generate LUT based on found time lags from last iteration
        """

        # Load results from last iteration
        last_iteration = self.lgs_num_iter
        if self.outdirs:
            filepath_last_iteration = self.outdirs[f'{self.phase}-3_{self.phase_files}_time_lags_overview'] \
                                      / f'{last_iteration}_segments_found_lag_times_after_iteration-{last_iteration}.csv'

        else:
            filepath_last_iteration = self.direct_path_to_segment_lagtimes_file  # Implemented for unittest

        segment_lagtimes_last_iteration_df = self.filter_dataframe(filter_col='iteration',
                                                                   filter_equal_to=last_iteration,
                                                                   df=files.read_segment_lagtimes_file(
                                                                       filepath=filepath_last_iteration))

        if self.phase != 3:
            # LUT for Phase 1 or Phase 2, aggregated daily lags
            lut_df, lut_available = self.make_lut_agg(target_lag=self.target_lag,
                                                      segment_lagtimes_df=segment_lagtimes_last_iteration_df)
        else:
            # LUT for Phase 3, instantaneous lags
            lut_df, lut_available = self.make_lut_instantaneous(segment_lagtimes_df=segment_lagtimes_last_iteration_df,
                                                                default_lag=self.target_lag)

        if lut_available:
            self.logger.info(f"Finished creating look-up table for default lag times and normalization correction")
        else:
            self.logger.critical(f"(!) Look-up Table for default lag times and normalization correction is empty, "
                                 f"stopping script.")
            sys.exit()

        return lut_df, lut_available

    @staticmethod
    def filter_dataframe(filter_col, filter_equal_to, df):
        """
        Keep only rows of df where the value in filter_col matches filter_equal_to

        Parameters
        ----------
        filter_col: str
            Column name in df.
        filter_equal_to: value
            Filtering condition, only rows where filter_col == filter_equal_to is True
            are kept, other rows are removed.
        df: pandas Dataframe

        Returns
        -------
        Filtered pandas DataFrame where filter_col == filter_equal_to
        """
        filter_this_iteration = df[filter_col] == filter_equal_to
        df_filtered = df.loc[filter_this_iteration, :]
        return df_filtered

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

    def make_lut_agg(self, target_lag: int, segment_lagtimes_df: pd.DataFrame):
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
        lut_df = pd.DataFrame()
        peaks_hq_S = self.get_hq_peaks(df=segment_lagtimes_df)  # High-quality covariance peaks

        if peaks_hq_S.empty:
            lut_available = False
            return lut_df, lut_available

        unique_dates = np.unique(peaks_hq_S.index.date)
        for this_date in unique_dates:
            from_date = this_date - pd.Timedelta('1D')
            to_date = this_date + pd.Timedelta('1D')
            filter_around_this_day = (peaks_hq_S.index.date > from_date) & \
                                     (peaks_hq_S.index.date <= to_date)
            subset = peaks_hq_S[filter_around_this_day]
            num_vals = len(subset)

            # lut_df.loc[this_date, 'median'] = 9

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

        lut_df['target_lag'] = target_lag
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

        try:
            lut_df['recommended_default_winsize'] = \
                int((np.abs(lut_df['correction'].min() - lut_df['correction'].max())) / 2) * 1.2
        except ValueError:
            lut_df['recommended_default_winsize'] = np.nan

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
        df.index = pd.to_datetime(df.index, format="mixed")
        peaks_hq_S = df.loc[df['PEAK-COVABSMAX_SHIFT'] == df['PEAK-AUTO_SHIFT'],
                            'PEAK-COVABSMAX_SHIFT']
        peaks_hq_S.index = peaks_hq_S.index.to_pydatetime()  # Convert to DatetimeIndex
        return peaks_hq_S
