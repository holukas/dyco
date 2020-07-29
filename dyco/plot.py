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

import datetime as dt
import os
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd

from . import analyze, files, loop, setup_dyco


class SummaryPlots():
    def __init__(self, instance_phase_1, instance_phase_2, instance_phase_3):
        self.instance_phase_1 = instance_phase_1
        self.instance_phase_2 = instance_phase_2
        self.instance_phase_3 = instance_phase_3

        self.outdir_summary = self._make_outdir()

        self.logger = setup_dyco.create_logger(logfile_path=instance_phase_3.logfile_path, name=__name__)

        self.run()

    def run(self):
        self.logger.info("CREATING SUMMARY ...")
        collection_df = pd.DataFrame()

        # Plot and collect time lags and covariances from Phase 1 and Phase 2
        self.logger.info("Plotting found time lags, Phases 1-3")
        phase_instances = [self.instance_phase_1, self.instance_phase_2]
        for phase_instance in phase_instances:
            self.summary_plot_segment_lagtimes(phase=phase_instance.phase,
                                               phase_files=phase_instance.phase_files,
                                               outdirs=phase_instance.outdirs,
                                               last_iteration=phase_instance.lgs_num_iter,
                                               target_lag=phase_instance.target_lag)
            collection_df = self._collect_peak_covariances(phase=phase_instance.phase,
                                                           phase_files=phase_instance.phase_files,
                                                           outdirs=phase_instance.outdirs,
                                                           last_iteration=phase_instance.lgs_num_iter,
                                                           collection_df=collection_df)

        # Phase 3
        phase_instances = [self.instance_phase_3]
        for phase_instance in phase_instances:
            self.summary_plot_instantaneous_lagtimes(phase=phase_instance.phase,
                                                     phase_files=phase_instance.phase_files,
                                                     outdirs=phase_instance.outdirs)
            collection_df, shift_colname_out = self._collect_instantaneous_lags(phase=phase_instance.phase,
                                                                                phase_files=phase_instance.phase_files,
                                                                                outdirs=phase_instance.outdirs,
                                                                                collection_df=collection_df)
            collection_df = self._collect_instantaneous_covariances(phase=phase_instance.phase,
                                                                    phase_files=phase_instance.phase_files,
                                                                    outdirs=phase_instance.outdirs,
                                                                    collection_df=collection_df,
                                                                    shift_colname_out=shift_colname_out)

        self.summary_plot_covariances(collection_df=collection_df,
                                      outdir=self.outdir_summary,
                                      last_iter_phase_1=self.instance_phase_1.lgs_num_iter,
                                      last_iter_phase_2=self.instance_phase_2.lgs_num_iter)

        self.logger.info("CREATING SUMMARY ... Done.")
        self.logger.info("Finished DYCO processing.")

    def _add_lines_cov(self, ax, df, last_iter_phase_1, last_iter_phase_2):
        props = {'y': df[f'PHASE-1_ITER-{last_iter_phase_1}_PEAK-COVABSMAX_COV'],
                 'label': f'Phase 1',
                 # 'label': f'PHASE-1_ITER-{last_iter_phase_1}_PEAK-COVABSMAX_COV',
                 'zorder': 100, 'alpha': 1, 'marker': 's', 'ms': 6, 'lw': 1, 'ls': 'dotted',
                 'markeredgecolor': '#80DEEA', 'color': '#80DEEA', 'markerfacecolor': 'white'}
        ax.plot_date(df.index, **props)

        props = {'y': df[f'PHASE-2_ITER-{last_iter_phase_1}_PEAK-COVABSMAX_COV'],
                 'label': f'Phase 2',
                 # 'label': f'PHASE-2_ITER-{last_iter_phase_1}_PEAK-COVABSMAX_COV',
                 'zorder': 100, 'alpha': 1, 'marker': 's', 'ms': 6, 'lw': 1, 'ls': '--',
                 'markeredgecolor': '#F48FB1', 'color': '#F48FB1', 'markerfacecolor': 'white'}
        ax.plot_date(df.index, **props)

        props = {'y': df['PHASE-3_INSTANTANEOUS_LAG_COV'],
                 'label': 'Phase 3',
                 # 'label': 'PHASE-3_INSTANTANEOUS_LAG_COV',
                 'zorder': 100, 'alpha': 1, 'marker': 'o', 'ms': 6, 'lw': 2, 'ls': '-',
                 'markeredgecolor': '#8BC34A', 'color': '#8BC34A', 'markerfacecolor': '#8BC34A'}
        ax.plot_date(df.index, **props)

    def _format_plot(self, ax, title, show_legend=True):
        # Automatic tick locations and formats
        locator = mdates.AutoDateLocator(minticks=5, maxticks=20)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        default_format(ax=ax, label_color='black', fontsize=12,
                       txt_xlabel='file date', txt_ylabel='covariance', txt_ylabel_units='')

        font = {'family': 'sans-serif', 'color': 'black', 'weight': 'bold', 'size': 12, 'alpha': 1}
        ax.set_title(title, fontdict=font, loc='left', pad=12)

        ax.axhline(0, color='black', ls='-', lw=1, zorder=1)
        font = {'family': 'sans-serif', 'size': 10}
        if show_legend:
            ax.legend(frameon=True, loc='upper right', prop=font).set_zorder(100)

    def summary_plot_covariances(self, collection_df, outdir, last_iter_phase_1, last_iter_phase_2):
        self.logger.info("Plotting covariance evolution, Phases 1-3 ...")
        gs = gridspec.GridSpec(2, 1)  # rows, cols
        gs.update(wspace=0.3, hspace=0.3, left=0.03, right=0.97, top=0.97, bottom=0.03)
        fig = plt.Figure(facecolor='white', figsize=(16, 9))
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
        props_ax = {'last_iter_phase_1': last_iter_phase_1, 'last_iter_phase_2': last_iter_phase_2}

        # ax1: Covariance evolution
        # =========================
        self._add_lines_cov(ax=ax1, df=collection_df, **props_ax)
        self._format_plot(ax=ax1, title="Half-hourly covariance evolution")

        # ax2: Cumulative
        # ===============
        collection_df_cum = collection_df.cumsum()
        self._add_lines_cov(ax=ax2, df=collection_df_cum, **props_ax)
        self._format_plot(ax=ax2, title="Cumulative covariance evolution", show_legend=False)

        # Save
        outfile = f'summary_covariance_evolution'
        outpath = outdir / outfile
        # print(f"Saving time series of found segment lag times in {outpath} ...")
        fig.savefig(f"{outpath}.png", format='png', bbox_inches='tight', facecolor='w', transparent=True, dpi=150)
        a = 1

    def summary_plot_segment_lagtimes(self, phase: int, phase_files: str, outdirs,
                                      last_iteration: int, target_lag: int):
        """Plot time lags found in last iteration of each Phase incl. aggregated defaults"""
        # Get data
        segment_lagtimes_df = self._get_segment_lagtimes_last_iteration(outdirs=outdirs, phase=phase,
                                                                        phase_files=phase_files,
                                                                        last_iteration=last_iteration)
        lut_df = self._get_lut_last_iteration(outdirs=outdirs, phase=phase, phase_files=phase_files)

        # Make plot
        loop.Loop.plot_segment_lagtimes_ts(segment_lagtimes_df=segment_lagtimes_df,
                                           outdir=self.outdir_summary,
                                           iteration=last_iteration,
                                           show_all=False,
                                           overlay_default=True,
                                           overlay_default_df=lut_df,
                                           overlay_target_val=target_lag,
                                           phase=phase)

    def summary_plot_instantaneous_lagtimes(self, phase, phase_files, outdirs):
        df = self._get_instantaneous_time_lags(phase=phase, phase_files=phase_files, outdirs=outdirs)
        analyze.AnalyzeLags.plot_final_instantaneous_lagtimes(outdir=self.outdir_summary, phase=phase, df=df)

    def _collect_instantaneous_covariances(self, outdirs, phase, phase_files, collection_df, shift_colname_out):
        """Get instantaneous covariances at instantaneous time lags from Phase 3"""
        collection_df.index = pd.to_datetime(collection_df.index)
        indir = outdirs[f'{phase}-1_{phase_files}_covariances']
        filelist = os.listdir(str(indir))
        num_files = len(filelist)
        for idx, filename in enumerate(filelist):
            file_date = dt.datetime.strptime(filename, '%Y%m%d%H%M%S_iter1_segment_covariance_iteration-1.csv')

            # Check if datetime of covariance file is contained in collection
            if file_date in collection_df.index:
                # Read covariance file containing all covariances per shift/lag
                filepath = os.path.join(str(indir), filename)
                cov_df = files.read_segment_lagtimes_file(filepath=filepath)

                # From the collection, get the instantaneous shift/lag that was found for this file
                shift_instantaneous = \
                    collection_df.loc[collection_df.index == file_date, shift_colname_out].values[0]

                # From the covariance file, get the covariance for the instantaneous shift/lag
                filter_instantaneous = cov_df['shift'] == shift_instantaneous
                cov_instantaneous = cov_df.loc[filter_instantaneous, 'cov'].values[0]

                # Store instantaneous covariance in collection
                cov_colname_out = f'PHASE-{phase}_INSTANTANEOUS_LAG_COV'
                collection_df.loc[collection_df.index == file_date, cov_colname_out] = cov_instantaneous
        return collection_df

    def _collect_instantaneous_lags(self, phase, phase_files, outdirs, collection_df):
        df = self._get_instantaneous_time_lags(phase=phase, phase_files=phase_files, outdirs=outdirs)
        df.index = pd.to_datetime(df.index)
        shift_colname_out = f'PHASE-{phase}_INSTANTANEOUS_LAG_SHIFT'
        collection_df[shift_colname_out] = df['INSTANTANEOUS_LAG']
        return collection_df, shift_colname_out

    def _get_instantaneous_time_lags(self, phase, outdirs, phase_files):
        """Get instantaneous time lags from Phase 3 LUT file"""
        indir = outdirs[f'{phase}-6_{phase_files}_final_time_lags_lookup_table']
        df = files.read_segment_lagtimes_file(filepath=indir / 'LUT_final_time_lags.csv')
        return df

    def _collect_peak_covariances(self, outdirs, phase, phase_files, last_iteration,
                                  collection_df):

        segment_lagtimes_df = self._get_segment_lagtimes_last_iteration(outdirs=outdirs, phase=phase,
                                                                        phase_files=phase_files,
                                                                        last_iteration=last_iteration)
        _add_df = segment_lagtimes_df[['file_date', 'PEAK-COVABSMAX_COV', 'PEAK-COVABSMAX_SHIFT']]
        _add_df.set_index('file_date', inplace=True)
        cov_colname_out = f'PHASE-{phase}_ITER-{last_iteration}_PEAK-COVABSMAX_COV'
        shift_colname_out = f'PHASE-{phase}_ITER-{last_iteration}_PEAK-COVABSMAX_SHIFT'

        if phase == 1:
            collection_df = _add_df.copy()
            collection_df.rename(columns={'PEAK-COVABSMAX_COV': cov_colname_out}, inplace=True)
            collection_df.rename(columns={'PEAK-COVABSMAX_SHIFT': shift_colname_out}, inplace=True)
        else:
            collection_df[cov_colname_out] = _add_df['PEAK-COVABSMAX_COV']
            collection_df[shift_colname_out] = _add_df['PEAK-COVABSMAX_SHIFT']

        return collection_df

    def _get_segment_lagtimes_last_iteration(self, outdirs, phase, phase_files, last_iteration):
        filepath = outdirs[f'{phase}-3_{phase_files}_time_lags_overview'] \
                   / f'{last_iteration}_segments_found_lag_times_after_iteration-{last_iteration}.csv'
        df = \
            analyze.AnalyzeLags.filter_dataframe(filter_col='iteration',
                                                 filter_equal_to=last_iteration,
                                                 df=files.read_segment_lagtimes_file(
                                                     filepath=filepath))
        return df

    def _get_lut_last_iteration(self, outdirs, phase, phase_files):
        filepath_lut = outdirs[f'{phase}-6_{phase_files}_normalization_lookup_table'] / 'LUT_default_agg_time_lags.csv'
        lut_df = files.read_segment_lagtimes_file(filepath=filepath_lut)
        return lut_df

    def _make_outdir(self):
        outdir = self.instance_phase_1.outdir / 'SUMMARY'
        if not Path.is_dir(outdir):
            print(f"Creating folder {outdir} ...")
            os.makedirs(outdir)
        return outdir


def limit_data_range_percentiles(df, col, perc_limits):
    p_lower = df[col].quantile(perc_limits[0])
    p_upper = df[col].quantile(perc_limits[1])
    p_filter = (df[col] >= p_lower) & (df[col] <= p_upper)
    df = df[p_filter]
    return df


def default_format(ax, fontsize=12, label_color='black',
                   txt_xlabel='', txt_ylabel='', txt_ylabel_units='',
                   width=1, length=5, direction='in', colors='black', facecolor='white'):
    """Apply default format to plot."""
    ax.set_facecolor(facecolor)
    ax.tick_params(axis='x', width=width, length=length, direction=direction, colors=colors, labelsize=fontsize,
                   top=True)
    ax.tick_params(axis='y', width=width, length=length, direction=direction, colors=colors, labelsize=fontsize,
                   right=True)
    format_spines(ax=ax, color=colors, lw=1)
    if txt_xlabel:
        ax.set_xlabel(txt_xlabel, color=label_color, fontsize=fontsize, fontweight='bold')
    if txt_ylabel and txt_ylabel_units:
        ax.set_ylabel(f'{txt_ylabel}  {txt_ylabel_units}', color=label_color, fontsize=fontsize, fontweight='bold')
    if txt_ylabel and not txt_ylabel_units:
        ax.set_ylabel(f'{txt_ylabel}', color=label_color, fontsize=fontsize, fontweight='bold')


def format_spines(ax, color, lw):
    spines = ['top', 'bottom', 'left', 'right']
    for spine in spines:
        ax.spines[spine].set_color(color)
        ax.spines[spine].set_linewidth(lw)


def setup_fig_ax():
    """Setup grid with one figure and one axis."""
    gs = gridspec.GridSpec(1, 1)  # rows, cols
    gs.update(wspace=0.3, hspace=0.3, left=0.03, right=0.97, top=0.97, bottom=0.03)
    fig = plt.Figure(facecolor='white', figsize=(16, 9))
    ax = fig.add_subplot(gs[0, 0])
    return gs, fig, ax
