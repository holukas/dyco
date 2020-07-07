import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

import files


def timeseries_segment_lagtimes(df, outdir, iteration, show_all=False, overlay_default=False,
                                overlay_default_df=None, overlay_target_val=False,
                                overlay_default_target=False):
    _df = df.copy()
    if not show_all:
        _df = _df.loc[_df['iteration'] == iteration, :]
        txt_info = f"Found lag times in iteration {iteration}"
        outfile = f"{iteration}_TIMESERIES-PLOT_found_segment_lag_times_iteration-{iteration}"
    else:
        txt_info = f"Found lag times from ALL iterations"
        outfile = f"TIMESERIES-PLOT_found_segment_lag_times_iteration-ALL"

    gs, fig, ax = setup_fig_ax()

    cmap = plt.cm.get_cmap('Spectral', iteration)
    colors = cmap(np.linspace(0, 1, iteration))

    alpha = 0.4
    iteration_grouped = _df.groupby('iteration')
    for idx, group_df in iteration_grouped:
        ax.plot_date(pd.to_datetime(group_df['start']), group_df['shift_peak_cov_abs_max'],
                     alpha=alpha, marker='o', ms=6, color=colors[int(idx - 1)], lw=1, ls='-',
                     label=f'iteration {int(idx)}')

    default_format(ax=ax, label_color='black', fontsize=12,
                   txt_xlabel='segment date', txt_ylabel='lag', txt_ylabel_units='[records]')

    ax.text(0.02, 0.98, txt_info,
            horizontalalignment='left', verticalalignment='top',
            transform=ax.transAxes, size=14, color='black',
            backgroundcolor='none', zorder=100)

    if overlay_default:
        if not overlay_default_df.empty:
            ax.plot_date(overlay_default_df.index, overlay_default_df['median'], alpha=1,
                         marker='s', ms=12, color='black', lw=3, ls='-',
                         label='3-day median lag time (centered)\nfrom high-quality covariance peaks')

        else:
            ax.text(0.5, 0.5, "No high-quality lags found, lag normalization failed",
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes,
                    size=20, color='white', backgroundcolor='red', zorder=100)

    if overlay_target_val:
        ax.axhline(overlay_target_val, color='black', ls='--', label='target: normalized default lag')

    ax.legend(frameon=True, loc='upper right').set_zorder(100)

    outpath = outdir / outfile
    # print(f"Saving time series of found segment lag times in {outpath} ...")
    fig.savefig(f"{outpath}.png", format='png', bbox_inches='tight', facecolor='w',
                transparent=True, dpi=150)

    # years = mdates.YearLocator()  # every year
    # months = mdates.MonthLocator()  # every month
    # years_fmt = mdates.DateFormatter('%Y')
    # ax.xaxis.set_major_locator(years)
    # ax.xaxis.set_major_formatter(years_fmt)
    # ax.xaxis.set_minor_locator(months)
    # myFmt = mdates.DateFormatter('%Y-%m-%d %H:%M')
    # ax.xaxis.set_major_formatter(myFmt)

    return outfile


def results(df):
    # todo
    cols = ['shift_median', 'cov_max_shift', 'shift_P25', 'shift_P75', 'search_win_upper', 'search_win_lower']
    _df = df[cols].copy()

    _df[cols].plot()
    plt.show()
    return None


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
    ax.tick_params(axis='x', width=width, length=length, direction=direction, colors=colors, labelsize=fontsize)
    ax.tick_params(axis='y', width=width, length=length, direction=direction, colors=colors, labelsize=fontsize)
    format_spines(ax=ax, color=colors, lw=1)
    if txt_xlabel:
        ax.set_xlabel(txt_xlabel, color=label_color, fontsize=fontsize, fontweight='bold')
    if txt_ylabel and txt_ylabel_units:
        ax.set_ylabel(f'{txt_ylabel}  {txt_ylabel_units}', color=label_color, fontsize=fontsize, fontweight='bold')
    if txt_ylabel and not txt_ylabel_units:
        ax.set_ylabel(f'{txt_ylabel}', color=label_color, fontsize=fontsize, fontweight='bold')
    return None


def format_spines(ax, color, lw):
    spines = ['top', 'bottom', 'left', 'right']
    for spine in spines:
        ax.spines[spine].set_color(color)
        ax.spines[spine].set_linewidth(lw)
    return None


def setup_fig_ax():
    """Setup grid with one figure and one axis."""
    gs = gridspec.GridSpec(1, 1)  # rows, cols
    gs.update(wspace=0.3, hspace=0.3, left=0.03, right=0.97, top=0.97, bottom=0.03)
    fig = plt.Figure(facecolor='white', figsize=(16, 9))
    ax = fig.add_subplot(gs[0, 0])
    return gs, fig, ax





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
    filelist = os.listdir(indir)
    num_files = len(filelist)
    logger.info(f"Plotting covariance collection from {num_files} files")
    for idx, filename in enumerate(filelist):
        # if idx > 1000:
        #     break
        # print(f"Reading segment covariance file {idx + 1} of {num_files}: {filename}")
        filepath = os.path.join(indir, filename)
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
    default_format(ax=ax1, txt_xlabel='', txt_ylabel='covariance', txt_ylabel_units='')
    default_format(ax=ax2, txt_xlabel='', txt_ylabel='absolute covariance', txt_ylabel_units='')
    default_format(ax=ax3, txt_xlabel='shift [records]', txt_ylabel='normalized absolute covariance',
                   txt_ylabel_units='')

    outfile = '1_covariance_collection_all_segments.png'
    outpath = outdir / outfile
    fig.savefig(f"{outpath}", format='png', bbox_inches='tight', facecolor='w',
                transparent=True, dpi=150)

    return None
